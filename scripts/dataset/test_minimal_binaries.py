#!/usr/bin/env python3
"""Test minimal static binaries with QEMU to verify they execute correctly."""

import argparse
import json
import random
import subprocess
import tempfile
import time
from pathlib import Path

from tqdm import tqdm


def test_binary(binary_hex: str, expected_return: int) -> tuple[bool, str]:
    """Test a binary by executing it with QEMU.

    Args:
        binary_hex: Hex-encoded binary
        expected_return: Expected return code

    Returns:
        Tuple of (success, message)
    """
    try:
        # Convert hex to binary
        binary_bytes = bytes.fromhex(binary_hex)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.elf') as f:
            binary_path = Path(f.name)
            binary_path.write_bytes(binary_bytes)
            binary_path.chmod(0o755)

        # Execute with QEMU
        result = subprocess.run(
            ["qemu-riscv64", str(binary_path)],
            capture_output=True,
            timeout=5
        )

        binary_path.unlink()

        # Check return code (modulo 256 for valid exit codes)
        actual_return = result.returncode
        expected_exit = expected_return % 256

        if actual_return == expected_exit:
            return True, f"✅ Return code: {actual_return}"
        else:
            return False, f"❌ Expected {expected_exit}, got {actual_return}"

    except subprocess.TimeoutExpired:
        return False, "❌ Timeout"
    except Exception as e:
        return False, f"❌ Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Test minimal static binaries with QEMU"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=50,
        help="Number of random samples to test (default: 50)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all binaries in the dataset"
    )
    args = parser.parse_args()

    dataset_path = Path("dataset/processed/returns_dataset.jsonl")

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Run: python scripts/dataset/generate_returns_dataset.py")
        return

    print("Loading dataset...")
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")
    print("=" * 80)

    # Determine sample
    if args.all:
        sample = examples
        sample_size = len(examples)
        print(f"Testing all {sample_size} binaries...")
    else:
        sample_size = min(args.num_samples, len(examples))
        sample = random.sample(examples, sample_size)
        print(f"Testing {sample_size} random binaries...")

    passed = 0
    failed = 0
    failures = []

    start_time = time.time()

    # Test with progress bar
    for example in tqdm(sample, desc="Testing", unit=" binary"):
        prompt = example['prompt']
        # Extract expected return value from prompt
        expected = int(prompt.split("returns ")[1])
        binary_hex = example['binary_hex']

        success, message = test_binary(binary_hex, expected)

        if success:
            passed += 1
        else:
            failed += 1
            failures.append((prompt, message))

    elapsed = time.time() - start_time

    # Show failures if any
    if failures and len(failures) <= 10:
        print("\n❌ Failures:")
        for prompt, message in failures:
            print(f"  {prompt}: {message}")
    elif failures:
        print(f"\n❌ {len(failures)} failures (showing first 10):")
        for prompt, message in failures[:10]:
            print(f"  {prompt}: {message}")

    print(f"\n{'=' * 80}")
    print(f"Results: {passed}/{sample_size} passed")
    print(f"Time: {elapsed:.1f}s ({elapsed/sample_size:.3f}s per binary)")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {failed} tests failed ({failed/sample_size*100:.1f}%)")


if __name__ == "__main__":
    main()
