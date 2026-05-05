#!/usr/bin/env python3
"""Generate RISC-V binaries via the Qwen3-0.6B + compact pipeline.

Parallel to scripts/generation/generate_gpt2_compact.py: same wrapper
template, same CFI nibble detection, same splice. Only difference is
the model loader — Unsloth FastLanguageModel (matches the QLoRA
training in train_qwen_compact.py) instead of HF AutoModel + PEFT.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

VAR_LEN = 22


def load_model(checkpoint_path: str, max_seq_length: int = 256, quantize_4bit: bool = False):
    t0 = time.time()
    mode = "4-bit" if quantize_4bit else "bf16"
    print(f"[{time.time()-t0:5.1f}s] Loading {checkpoint_path} ({mode})...")
    with open(f"{checkpoint_path}/adapter_config.json") as f:
        adapter_config = json.load(f)
    base = adapter_config.get("base_model_name_or_path", "./models/base/qwen3-0.6b")
    print(f"  Base model: {base}")

    model, _ = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_seq_length,
        dtype=None if quantize_4bit else torch.bfloat16,
        load_in_4bit=quantize_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    FastLanguageModel.for_inference(model)
    print(f"✅ [{time.time()-t0:5.1f}s] Model loaded")
    return model, tokenizer


def format_prompt(instruction: str, tokenizer) -> str:
    """Qwen3 chat template, thinking disabled, with assistant role primed."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_variable_region(model, tokenizer, prompt, max_new_tokens=64):
    t0 = time.time()
    formatted = format_prompt(prompt, tokenizer)
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

    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Qwen chat template wraps the assistant turn with:
    #   <|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>
    # The empty <think>...</think> block is injected even with enable_thinking=False.
    if "<|im_start|>assistant" in full:
        response = full.split("<|im_start|>assistant", 1)[1]
    else:
        response = full
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    # Stop at either marker, whichever comes first.
    for marker in ("<END_BINARY>", "<|im_end|>"):
        if marker in response:
            response = response.split(marker)[0]
            print(f"  Found {marker}")
            break
    else:
        print("  Warning: no end marker found")
    response = ''.join(response.split())  # strip whitespace/newlines
    return response


def detect_cfi_nibble(code_hex: str) -> str:
    """Return the .eh_frame CFI length nibble for the given variable region.

    Locates the ecall opcode byte (0x73) at a 2-byte-aligned offset
    within the region; the offset determines code length and therefore
    the CFI nibble.
    """
    for byte_pos, cfi in ((6, "a"), (8, "c"), (10, "e")):
        h = byte_pos * 2
        if len(code_hex) >= h + 2 and code_hex[h:h + 2] == "73":
            return cfi
    return "c"  # fallback: most common pattern


def assemble_full_binary(var_region: str, template: dict) -> str:
    """Splice variable region + CFI nibbles into the constant wrapper."""
    wrapper = template["wrapper_with_placeholders"]
    var_start, var_end = template["variable_region"]
    expected = var_end - var_start

    if len(var_region) < expected:
        print(f"  ⚠️  generated {len(var_region)} chars, padding with zeros to {expected}")
        var_region = var_region + "0" * (expected - len(var_region))
    elif len(var_region) > expected:
        print(f"  ⚠️  generated {len(var_region)} chars, truncating to {expected}")
        var_region = var_region[:expected]

    cfi = detect_cfi_nibble(var_region)
    print(f"  Detected pattern → CFI nibble: '{cfi}'")

    chars = list(wrapper)
    for i, c in enumerate(var_region):
        chars[var_start + i] = c
    for pos in template["cfi_nibble_positions"]:
        chars[pos] = cfi

    if "_" in chars or "?" in chars:
        raise RuntimeError("Wrapper still has placeholder chars after splicing")
    return "".join(chars)


def save_binary(hex_string: str, output_path: str):
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
    p.add_argument("--checkpoint", default="./models/checkpoints/qwen3-compact")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="./generated_binary")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--template",
        default="dataset/processed/phase1_wrapper_template.json",
        help="Wrapper template produced by generate_phase1_compact.py",
    )
    p.add_argument("--quantize-4bit", action="store_true",
                   help="Load in 4-bit. Use only if the checkpoint was trained "
                        "with --quantize-4bit; default is bf16.")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.template).exists():
        print(f"❌ Wrapper template not found: {args.template}")
        sys.exit(1)

    with open(args.template) as f:
        template = json.load(f)
    print(f"Loaded wrapper template ({len(template['wrapper_with_placeholders'])} chars, "
          f"variable region {template['variable_region']})")

    model, tok = load_model(args.checkpoint, quantize_4bit=args.quantize_4bit)
    var_region = generate_variable_region(model, tok, args.prompt, args.max_tokens)
    print(f"  Model output (variable region): {var_region}")

    full_hex = assemble_full_binary(var_region, template)
    Path(args.output + ".raw").write_text(full_hex)
    print(f"  Wrote raw hex to {args.output}.raw ({len(full_hex)} chars)")
    save_binary(full_hex, args.output)


if __name__ == "__main__":
    main()
