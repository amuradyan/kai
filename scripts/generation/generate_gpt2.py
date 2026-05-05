#!/usr/bin/env python3
"""Generate RISC-V binaries from prompts using a GPT-2 + PEFT checkpoint.

Parallel to scripts/generation/generate_binary.py. No Unsloth.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(checkpoint_path: str):
    t0 = time.time()
    print(f"[{time.time()-t0:5.1f}s] Loading {checkpoint_path}...")
    with open(f"{checkpoint_path}/adapter_config.json") as f:
        adapter_config = json.load(f)
    base = adapter_config.get("base_model_name_or_path", "openai-community/gpt2")
    print(f"  Base model: {base}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print(f"✅ [{time.time()-t0:5.1f}s] Model loaded")
    return model, tokenizer


def format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_binary(model, tokenizer, prompt, max_new_tokens=900):
    t0 = time.time()
    formatted = format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    print(f"\nPrompt: {prompt}")
    print(f"[{time.time()-t0:5.1f}s] Generating up to {max_new_tokens} tokens...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    print(f"[{time.time()-t0:5.1f}s] Decoding...")
    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "### Response:" in full:
        response = full.split("### Response:")[1].strip()
    else:
        response = full
    if "<END_BINARY>" in response:
        response = response.split("<END_BINARY>")[0]
        print("  Found END_BINARY")
    else:
        print("  Warning: END_BINARY not found")
    return response


def save_binary(hex_string, output_path):
    try:
        clean = ''.join(hex_string.split())
        data = bytes.fromhex(clean)
        out = Path(output_path)
        out.write_bytes(data)
        out.chmod(0o755)
        print(f"✅ Saved {output_path} ({len(data)} bytes)")
        return True
    except ValueError as e:
        print(f"❌ Invalid hex: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./models/checkpoints/gpt2-small-lora")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="./generated_binary")
    p.add_argument("--max-tokens", type=int, default=900)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, tok = load_model(args.checkpoint)
    hex_out = generate_binary(model, tok, args.prompt, args.max_tokens)
    Path(args.output + ".raw").write_text(hex_out)
    save_binary(hex_out, args.output)


if __name__ == "__main__":
    main()
