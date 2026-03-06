#!/usr/bin/env python3
"""
DEPRECATED: This script expects the old dataset format with category/source_code fields.
Use scripts/dataset/verify_returns_dataset.py instead for the simplified dataset format.

Original purpose: Verify RISC-V binaries from the dataset using QEMU emulation.
Extracts binaries from the dataset and runs them with qemu-riscv64
to verify they execute correctly and produce expected return codes.
"""

import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, List, Optional


def find_qemu() -> Optional[str]:
    """Find qemu-riscv64 binary in system or Nix store."""
    # Try common locations
    candidates = [
        "qemu-riscv64",  # In PATH
        "/nix/store/*/bin/qemu-riscv64",  # Nix store
    ]

    # First try which
    try:
        result = subprocess.run(
            ["which", "qemu-riscv64"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    # Search Nix store with glob
    try:
        import glob
        matches = glob.glob("/nix/store/*/bin/qemu-riscv64")
        if matches:
            return matches[0]
    except:
        pass

    return None


def run_binary(qemu_path: str, binary_hex: str) -> Dict:
    """
    Run a RISC-V binary using QEMU.

    Args:
        qemu_path: Path to qemu-riscv64
        binary_hex: Hex-encoded binary data

    Returns:
        Dict with success, return_code, stdout, stderr
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        binary_file = tmpdir / "test.elf"

        # Write binary
        try:
            binary_data = bytes.fromhex(binary_hex)
            binary_file.write_bytes(binary_data)
            binary_file.chmod(0o755)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write binary: {e}"
            }

        # Run with QEMU
        try:
            result = subprocess.run(
                [qemu_path, str(binary_file)],
                capture_output=True,
                text=True,
                timeout=5
            )

            return {
                "success": True,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timeout (>5s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {e}"
            }


def compute_expected_return(prompt: str) -> Optional[int]:
    """
    Compute expected return code from prompt.

    Note: Shell return codes are limited to 0-255, so we take modulo 256.
    """
    try:
        # Extract value from prompt: "Write a program that returns N"
        if "returns" in prompt:
            value = int(prompt.split("returns")[-1].strip())
            return value % 256
    except Exception:
        # If parsing fails, we can't verify
        return None

    return None


def verify_dataset_samples(dataset_path: str, samples_per_category: int = 5):
    """
    Verify samples from each category in the dataset.

    Args:
        dataset_path: Path to JSONL dataset
        samples_per_category: Number of samples to test per category
    """
    qemu_path = find_qemu()
    if not qemu_path:
        print("ERROR: qemu-riscv64 not found!")
        print("Please ensure QEMU is installed and in PATH or Nix store.")
        return False

    print(f"Using QEMU: {qemu_path}")
    print("=" * 70)

    # Load dataset
    dataset_path = Path(dataset_path)
    examples_by_category = {}

    with open(dataset_path) as f:
        for line in f:
            ex = json.loads(line)
            cat = ex["category"]
            if cat not in examples_by_category:
                examples_by_category[cat] = []
            examples_by_category[cat].append(ex)

    print(f"Loaded {sum(len(v) for v in examples_by_category.values())} examples")
    print(f"Categories: {list(examples_by_category.keys())}")
    print("=" * 70)

    # Test samples from each category
    total_tested = 0
    total_passed = 0
    total_failed = 0

    for category, examples in sorted(examples_by_category.items()):
        print(f"\n{category.upper()} ({len(examples)} total)")
        print("-" * 70)

        # Take first N samples
        samples = examples[:samples_per_category]

        category_passed = 0
        category_failed = 0

        for i, ex in enumerate(samples, 1):
            prompt = ex["prompt"]
            source = ex["source_code"]
            binary_hex = ex["binary_hex"]

            # Run binary
            result = run_binary(qemu_path, binary_hex)

            if not result["success"]:
                print(f"  [{i}] ✗ EXECUTION FAILED")
                print(f"      Prompt: {prompt[:60]}")
                print(f"      Error: {result.get('error', 'Unknown')}")
                category_failed += 1
                total_failed += 1
                total_tested += 1
                continue

            # Compute expected return
            expected = compute_expected_return(category, source)
            actual = result["return_code"]

            if expected is None:
                print(f"  [{i}] ? CANNOT VERIFY (parsing failed)")
                print(f"      Prompt: {prompt[:60]}")
                print(f"      Actual return: {actual}")
                continue

            # Verify
            if actual == expected:
                print(f"  [{i}] ✓ PASS")
                print(f"      Prompt: {prompt[:60]}")
                print(f"      Expected: {expected}, Got: {actual}")
                category_passed += 1
                total_passed += 1
            else:
                print(f"  [{i}] ✗ FAIL")
                print(f"      Prompt: {prompt[:60]}")
                print(f"      Source: {source[:70]}")
                print(f"      Expected: {expected}, Got: {actual}")
                category_failed += 1
                total_failed += 1

            total_tested += 1

        print(f"\n  Category summary: {category_passed} passed, {category_failed} failed")

    # Final summary
    print("\n" + "=" * 70)
    print(f"OVERALL RESULTS:")
    print(f"  Total tested: {total_tested}")
    print(f"  Passed: {total_passed} ({100*total_passed//total_tested if total_tested > 0 else 0}%)")
    print(f"  Failed: {total_failed} ({100*total_failed//total_tested if total_tested > 0 else 0}%)")

    if total_failed == 0 and total_tested > 0:
        print("\n✓ ALL TESTS PASSED - Binaries execute correctly!")
        return True
    elif total_tested == 0:
        print("\n? NO TESTS RUN")
        return False
    else:
        print(f"\n✗ SOME TESTS FAILED - {total_failed}/{total_tested} binaries failed")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify RISC-V binaries from dataset using QEMU"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="./dataset/processed/returns_dataset.jsonl",
        help="Path to dataset JSONL file"
    )
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=5,
        help="Number of samples to test per category (default: 5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all examples in the dataset (overrides -n)"
    )

    args = parser.parse_args()

    # If --all is specified, use a very large number to get all examples
    samples = None if args.all else args.samples
    if samples is None:
        samples = 999999  # Effectively unlimited

    success = verify_dataset_samples(args.dataset, samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
