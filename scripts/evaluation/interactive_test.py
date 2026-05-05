#!/usr/bin/env python3
"""Interactive testing mode - load model once and test multiple prompts."""

import sys
import tempfile
import subprocess
from pathlib import Path
from unsloth import FastLanguageModel

def load_model(checkpoint_path):
    """Load model once at startup."""
    print(f"Loading model from {checkpoint_path}...")

    # For PEFT/LoRA models, we need to handle the vocab size before loading the adapter
    # Load tokenizer first to check vocab size
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    expected_vocab_size = len(tokenizer)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        resize_embedding_vocab=expected_vocab_size,  # This handles the resize before loading adapter
    )

    # Check if END_BINARY token is already in the model
    if '<END_BINARY>' in tokenizer.get_vocab():
        print("✓ END_BINARY token found in checkpoint")
    else:
        print("Warning: END_BINARY token not found. This checkpoint may not support END_BINARY.")

    FastLanguageModel.for_inference(model)
    print("✅ Model loaded and ready!")
    return model, tokenizer

def generate_and_test(model, tokenizer, value):
    """Generate binary for a value and test it."""
    prompt = f"Write a program that returns {value}"
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    print(f"Generating for value {value}...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=False,
    )

    # Decode and extract hex
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "### Response:" in full_output:
        response = full_output.split("### Response:")[1].strip()
    else:
        response = full_output

    # Check for END_BINARY
    has_end_binary = "<END_BINARY>" in response
    if has_end_binary:
        hex_string = response.split("<END_BINARY>")[0]
        print(f"✓ Found END_BINARY token")
    else:
        hex_string = response
        print(f"✗ No END_BINARY token found")

    # Clean and measure
    hex_clean = ''.join(hex_string.split())
    print(f"Generated {len(hex_clean)} hex chars")

    if len(hex_clean) < 100:
        print(f"❌ Generation too short! Only {len(hex_clean)} chars")
        return

    # Test with QEMU
    try:
        binary_data = bytes.fromhex(hex_clean)

        with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as f:
            f.write(binary_data)
            f.flush()
            temp_path = f.name

        Path(temp_path).chmod(0o755)

        result = subprocess.run(
            ["qemu-riscv64", temp_path],
            capture_output=True,
            timeout=1
        )

        expected = value % 256
        actual = result.returncode

        if actual == expected:
            print(f"✅ PASS: Value {value} → Exit code {actual} (expected {expected})")
        else:
            print(f"❌ FAIL: Value {value} → Exit code {actual} (expected {expected})")

        Path(temp_path).unlink()

    except Exception as e:
        print(f"❌ Error testing: {e}")

    print("-" * 60)

def interactive_mode(model, tokenizer):
    """Interactive testing loop."""
    print("\n" + "="*60)
    print("Interactive Testing Mode")
    print("="*60)
    print("Commands:")
    print("  <number>  - Test with that return value")
    print("  'q'       - Quit")
    print("  'batch'   - Test multiple values")
    print("-"*60)

    while True:
        try:
            user_input = input("\nEnter value to test (or 'q' to quit): ").strip()

            if user_input.lower() == 'q':
                break

            if user_input.lower() == 'batch':
                values_str = input("Enter values separated by spaces: ").strip()
                values = [int(v) for v in values_str.split()]
                for v in values:
                    generate_and_test(model, tokenizer, v)
                continue

            value = int(user_input)
            generate_and_test(model, tokenizer, value)

        except ValueError:
            print("Invalid input! Enter a number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive model testing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/all-non-zeros-x5",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--values",
        type=int,
        nargs='+',
        help="Test specific values and exit (non-interactive)"
    )

    args = parser.parse_args()

    # Load model once
    model, tokenizer = load_model(args.checkpoint)

    if args.values:
        # Non-interactive mode - test specified values and exit
        for value in args.values:
            generate_and_test(model, tokenizer, value)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()