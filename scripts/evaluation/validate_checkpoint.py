#!/usr/bin/env python3
"""Validate checkpoint on specific values - gate before Phase 2."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from unsloth import FastLanguageModel


CUT_MARKER = "6d6d656e74002e72697363762e61747472696275746573"


def get_footer_from_dataset():
    """Get footer from any dataset example."""
    dataset_path = Path("dataset/processed/returns_dataset.jsonl")

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        first_line = f.readline()
        data = json.loads(first_line)

    binary_hex = data["binary_hex"]
    cut_pos = binary_hex.find(CUT_MARKER)

    if cut_pos == -1:
        print(f"Error: Cut marker not found in dataset")
        sys.exit(1)

    footer_start = cut_pos + len(CUT_MARKER)
    footer = binary_hex[footer_start:]

    return footer


def generate_binary(model, tokenizer, value):
    """Generate binary for a specific value."""
    prompt = f"Write a program that returns {value}"

    # Format with instruction template
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Tokenize
    inputs = tokenizer(
        [formatted],
        return_tensors="pt"
    ).to("cuda")

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.1,
        do_sample=False,
        use_cache=True
    )

    # Decode (skip input tokens)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


def test_with_fixed_footer(raw_hex, expected_value):
    """Fix raw model output with dataset footer and run with QEMU."""
    # Strip trailing zeros
    raw_hex = raw_hex.rstrip("0")

    # Find cut marker and extract model's part
    cut_pos = raw_hex.find(CUT_MARKER)

    if cut_pos == -1:
        print(f"  Error: Cut marker not found in raw output")
        return None

    model_part = raw_hex[:cut_pos + len(CUT_MARKER)]

    # Get footer from dataset
    footer = get_footer_from_dataset()

    # Combine model part + dataset footer
    combined_hex = model_part + footer

    # Convert to binary
    try:
        binary_data = bytes.fromhex(combined_hex)
    except ValueError as e:
        print(f"  Error: Invalid hex data: {e}")
        return None

    # Write to temporary file and run with QEMU
    with tempfile.NamedTemporaryFile(delete=False, suffix="_test") as tmp:
        tmp.write(binary_data)
        tmp_path = tmp.name

    try:
        # Make executable
        subprocess.run(["chmod", "+x", tmp_path], check=True)

        # Run with QEMU
        result = subprocess.run(
            ["qemu-riscv64", tmp_path],
            capture_output=True
        )

        exit_code = result.returncode
        return exit_code

    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def validate_checkpoint(checkpoint_path, test_values):
    """Validate checkpoint on specific values.

    Args:
        checkpoint_path: Path to model checkpoint
        test_values: List of values to test

    Returns:
        True if all tests pass, False if any fail
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    results = []
    all_pass = True

    for value in test_values:
        print(f"\nTesting value {value}...")

        # Save raw output for debugging
        raw_file = Path(f"test_value_{value}.raw")

        try:
            # Generate binary
            generated_hex = generate_binary(model, tokenizer, value)

            # Clean hex (remove whitespace)
            generated_hex = ''.join(generated_hex.split())

            # Save raw output
            with open(raw_file, 'w') as f:
                f.write(generated_hex)
            print(f"  Raw output saved to: {raw_file}")

            # Test with fixed footer
            exit_code = test_with_fixed_footer(generated_hex, value)

            if exit_code is None:
                print(f"  ❌ FAIL - Could not generate valid binary")
                all_pass = False
                results.append({
                    'value': value,
                    'passed': False,
                    'error': 'Invalid binary'
                })
            else:
                expected_exit_code = value % 256
                passed = (exit_code == expected_exit_code)

                if passed:
                    print(f"  Generated exit code: {exit_code}")
                    print(f"  Expected exit code: {expected_exit_code}")
                    print(f"  ✅ PASS")
                else:
                    print(f"  Generated exit code: {exit_code}")
                    print(f"  Expected exit code: {expected_exit_code}")
                    print(f"  ❌ FAIL")
                    all_pass = False

                results.append({
                    'value': value,
                    'passed': passed,
                    'generated_exit_code': exit_code,
                    'expected_exit_code': expected_exit_code
                })

        except Exception as e:
            print(f"  ❌ FAIL - Error: {e}")
            all_pass = False
            results.append({
                'value': value,
                'passed': False,
                'error': str(e)
            })

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    for result in results:
        value = result['value']
        if result['passed']:
            print(f"Value {value}: ✅ PASS")
        else:
            error = result.get('error', 'Wrong exit code')
            print(f"Value {value}: ❌ FAIL ({error})")

    if all_pass:
        print(f"\n✅ All tests passed! Ready for Phase 2.")
        return True
    else:
        print(f"\n❌ Some tests failed. Continue training Phase 1.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate checkpoint on specific values"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-values",
        type=int,
        nargs='+',
        default=[15, 750],
        help="Values to test (default: 15 750)"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Validate
    all_pass = validate_checkpoint(str(checkpoint_path), args.test_values)

    # Exit code 0 if all pass, 1 if any fail
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()