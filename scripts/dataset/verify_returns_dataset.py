#!/usr/bin/env python3
"""
Verify RISC-V binaries from the simplified returns dataset using QEMU.

Tests that binaries execute correctly and produce expected return codes.
"""

import json
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Dict, Optional


def find_qemu() -> Optional[str]:
    """Find qemu-riscv64 binary in system or Nix store."""
    # Try which first
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

    # Search Nix store
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
                timeout=1
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
                "error": "Timeout (>1s)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {e}"
            }


def extract_expected_value(prompt: str) -> Optional[int]:
    """Extract expected return value from prompt."""
    try:
        # "Write a program that returns N"
        if "returns" in prompt:
            value = int(prompt.split("returns")[-1].strip())
            return value % 256  # Shell return codes are 0-255
    except:
        return None
    return None


def verify_dataset(dataset_path: str, num_samples: int = 100):
    """
    Verify samples from the returns dataset.

    Args:
        dataset_path: Path to JSONL dataset
        num_samples: Number of samples to test (0 for all)
    """
    qemu_path = find_qemu()
    if not qemu_path:
        print("ERROR: qemu-riscv64 not found!")
        print("Please ensure QEMU is installed.")
        return False

    print(f"Using QEMU: {qemu_path}")
    print("=" * 70)

    # Load dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return False

    examples = []
    with open(dataset_path) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Select samples
    if num_samples > 0 and num_samples < len(examples):
        # Take evenly spaced samples
        step = len(examples) // num_samples
        samples = examples[::step][:num_samples]
        print(f"Testing {len(samples)} samples (every {step}th example)")
    else:
        samples = examples
        print(f"Testing all {len(samples)} examples")

    print("=" * 70)

    # Test each sample
    passed = 0
    failed = 0

    for i, ex in enumerate(samples, 1):
        prompt = ex["prompt"]
        binary_hex = ex["binary_hex"]

        # Get expected value
        expected = extract_expected_value(prompt)
        if expected is None:
            print(f"[{i:4d}] ? SKIP - Cannot parse prompt: {prompt}")
            continue

        # Run binary
        result = run_binary(qemu_path, binary_hex)

        if not result["success"]:
            print(f"[{i:4d}] ✗ EXECUTION FAILED: {result.get('error', 'Unknown')}")
            print(f"        Prompt: {prompt}")
            failed += 1
            continue

        actual = result["return_code"]

        # Verify
        if actual == expected:
            print(f"[{i:4d}] ✓ PASS - {prompt} → {actual}")
            passed += 1
        else:
            print(f"[{i:4d}] ✗ FAIL - {prompt}")
            print(f"        Expected: {expected}, Got: {actual}")
            failed += 1

    # Summary
    total = passed + failed
    print("\n" + "=" * 70)
    print(f"RESULTS:")
    print(f"  Total tested: {total}")
    print(f"  Passed: {passed} ({100*passed//total if total > 0 else 0}%)")
    print(f"  Failed: {failed} ({100*failed//total if total > 0 else 0}%)")

    if failed == 0 and total > 0:
        print("\n✓ ALL TESTS PASSED!")
        return True
    elif total == 0:
        print("\n? NO TESTS RUN")
        return False
    else:
        print(f"\n✗ {failed} TESTS FAILED")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify returns dataset binaries with QEMU"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/processed/returns_dataset.jsonl",
        help="Path to dataset (default: dataset/processed/returns_dataset.jsonl)"
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=100,
        help="Number of samples to test, 0 for all (default: 100)"
    )

    args = parser.parse_args()

    success = verify_dataset(args.dataset, args.num_samples)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())