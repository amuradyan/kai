#!/usr/bin/env python3
"""Analyze failed generations from evaluation."""

import argparse
import json
import subprocess
from pathlib import Path
from collections import Counter


def disassemble_binary(binary_path: str) -> str:
    """Disassemble binary with objdump."""
    try:
        result = subprocess.run(
            ['riscv64-unknown-linux-gnu-objdump', '-d', binary_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error disassembling: {e}"


def analyze_failures(failures_path: Path, binaries_dir: Path):
    """Analyze failure patterns."""

    with open(failures_path, 'r') as f:
        failures = json.load(f)

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total failures: {len(failures)}\n")

    # Categorize failures
    error_types = Counter()
    for failure in failures:
        if 'error' in failure:
            error_types[failure['error']] += 1
        elif failure.get('executed') == False:
            error_types['execution_failed'] += 1
        else:
            error_types['incorrect_output'] += 1

    print("Failure categories:")
    for error_type, count in error_types.most_common():
        print(f"  {error_type}: {count}")

    # Show sample failures
    print(f"\n{'='*60}")
    print("Sample failures (first 5):")
    print(f"{'='*60}\n")

    for i, failure in enumerate(failures[:5]):
        print(f"\nFailure {i+1}:")
        print(f"  Prompt: {failure['prompt']}")
        if 'source' in failure:
            print(f"  Source: {failure['source']}")
        if 'error' in failure:
            print(f"  Error: {failure['error']}")
        if 'generated_hex' in failure:
            print(f"  Generated (first 100 chars): {failure['generated_hex'][:100]}")
        if 'expected_hex' in failure:
            print(f"  Expected (first 100 chars): {failure['expected_hex'][:100]}")

        # Try to disassemble if binary exists
        if 'index' in failure:
            binary_path = binaries_dir / f"test_{failure['index']}.bin"
            if binary_path.exists():
                print(f"\n  Disassembly (first 20 lines):")
                disasm = disassemble_binary(str(binary_path))
                lines = disasm.split('\n')[:20]
                for line in lines:
                    if line.strip():
                        print(f"    {line}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation failures")
    parser.add_argument(
        '--results-dir',
        type=str,
        default='./evaluation/results',
        help='Directory containing evaluation results',
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    failures_path = results_dir / 'failures.json'
    if not failures_path.exists():
        print(f"Error: No failures.json found at {failures_path}")
        print("Run evaluation first: python scripts/evaluation/evaluate_model.py")
        return

    analyze_failures(failures_path, results_dir)


if __name__ == '__main__':
    main()
