#!/usr/bin/env python3
"""Test raw model output by fixing footer and running with QEMU."""

import sys
import json
import subprocess
import tempfile
from pathlib import Path


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


def test_raw_output(raw_file):
    """Fix raw model output with dataset footer and run with QEMU."""
    # Read raw hex from file
    with open(raw_file, "r") as f:
        raw_hex = f.read().strip()

    # Strip trailing zeros
    raw_hex = raw_hex.rstrip("0")

    # Find cut marker and extract model's part
    cut_pos = raw_hex.find(CUT_MARKER)

    if cut_pos == -1:
        print(f"Error: Cut marker not found in raw output")
        print(f"Raw output length: {len(raw_hex)} hex chars")
        sys.exit(1)

    model_part = raw_hex[:cut_pos + len(CUT_MARKER)]

    # Get footer from dataset
    footer = get_footer_from_dataset()

    # Combine model part + dataset footer
    combined_hex = model_part + footer

    # Convert to binary
    try:
        binary_data = bytes.fromhex(combined_hex)
    except ValueError as e:
        print(f"Error: Invalid hex data: {e}")
        sys.exit(1)

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

        print("=" * 80)
        print("QEMU EXECUTION RESULT")
        print("=" * 80)
        print(f"\nModel output: {len(model_part)} hex chars")
        print(f"Footer added: {len(footer)} hex chars")
        print(f"Total binary: {len(binary_data)} bytes")
        print(f"\nExit code: {exit_code}")

        return exit_code

    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_with_fixed_footer.py <raw_output_file>")
        print("  raw_output_file: File containing raw hex output from model")
        sys.exit(1)

    raw_file = Path(sys.argv[1])

    if not raw_file.exists():
        print(f"Error: Raw output file not found: {raw_file}")
        sys.exit(1)

    test_raw_output(raw_file)


if __name__ == "__main__":
    main()
