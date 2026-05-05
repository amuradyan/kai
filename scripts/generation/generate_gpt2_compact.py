#!/usr/bin/env python3
"""Generate RISC-V binaries via the compact GPT-2 + PEFT pipeline.

The model emits only the 22-hex-char variable region. This script
reconstructs the full 848-byte ELF binary by splicing the generated
bytes into the constant wrapper template (saved at training-data
generation time), then patches the two .eh_frame CFI length nibbles
based on the detected code-length pattern.

Pattern detection: scan the variable region for the ecall opcode byte
(0x73). Its 2-byte-aligned position within the region determines the
total code length, which determines the CFI nibble:
  - ecall at byte 6  → 10-byte code → CFI nibble = 'a'
  - ecall at byte 8  → 12-byte code → CFI nibble = 'c'
  - ecall at byte 10 → 14-byte code → CFI nibble = 'e'
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

VAR_LEN = 22


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


def generate_variable_region(model, tokenizer, prompt, max_new_tokens=64):
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

    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = full.split("### Response:")[1].strip() if "### Response:" in full else full
    if "<END_BINARY>" in response:
        response = response.split("<END_BINARY>")[0]
        print("  Found END_BINARY")
    else:
        print("  Warning: END_BINARY not found")
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

    # Pad / truncate to the expected length.
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
    p.add_argument("--checkpoint", default="./models/checkpoints/gpt2-small-compact")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="./generated_binary")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument(
        "--template",
        default="dataset/processed/phase1_wrapper_template.json",
        help="Wrapper template produced by generate_phase1_compact.py",
    )
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

    model, tok = load_model(args.checkpoint)
    var_region = generate_variable_region(model, tok, args.prompt, args.max_tokens)
    print(f"  Model output (variable region): {var_region}")

    full_hex = assemble_full_binary(var_region, template)
    Path(args.output + ".raw").write_text(full_hex)
    print(f"  Wrote raw hex to {args.output}.raw ({len(full_hex)} chars)")
    save_binary(full_hex, args.output)


if __name__ == "__main__":
    main()
