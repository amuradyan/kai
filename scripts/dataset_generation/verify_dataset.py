#!/usr/bin/env python3
"""
Verify the quality and correctness of the synthetic dataset.

Checks for:
- Duplicate prompts
- Duplicate source codes
- Prompt→source conflicts (same prompt, different code)
- Category distribution
- Data format validity

Returns exit code 0 if all checks pass, 1 otherwise.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def verify_dataset(dataset_path: str) -> bool:
    """Verify dataset quality. Returns True if all checks pass."""

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return False

    # Load all examples
    examples = []
    try:
        with open(dataset_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {line_num}: {e}")
                    return False
    except Exception as e:
        print(f"ERROR: Failed to read dataset: {e}")
        return False

    if not examples:
        print("ERROR: Dataset is empty")
        return False

    print(f"Loaded {len(examples)} examples from {dataset_path}")
    print(f"File size: {dataset_path.stat().st_size / (1024*1024):.1f} MB")
    print("=" * 60)

    all_checks_passed = True

    # Check 1: Category distribution
    print("\n1. Category Distribution:")
    categories = Counter(ex['category'] for ex in examples)
    for cat, count in sorted(categories.items()):
        print(f"   {cat:15s}: {count:5d}")

    expected_categories = {'constant', 'arithmetic', 'variable', 'conditional', 'loop'}
    if set(categories.keys()) != expected_categories:
        print(f"   WARNING: Expected categories {expected_categories}")
        print(f"            Got {set(categories.keys())}")

    # Check 2: Unique prompts
    print("\n2. Prompt Uniqueness:")
    prompts = [ex['prompt'] for ex in examples]
    unique_prompts = len(set(prompts))
    duplicate_prompts = len(prompts) - unique_prompts

    print(f"   Total prompts: {len(prompts)}")
    print(f"   Unique prompts: {unique_prompts}")

    if duplicate_prompts > 0:
        print(f"   ✗ FAIL: {duplicate_prompts} duplicate prompts!")
        all_checks_passed = False

        # Show duplicates
        prompt_counts = Counter(prompts)
        duplicates = [(p, c) for p, c in prompt_counts.items() if c > 1]
        print(f"   Top duplicates:")
        for prompt, count in sorted(duplicates, key=lambda x: -x[1])[:5]:
            print(f"     {count}x: {prompt[:60]}...")
    else:
        print("   ✓ PASS: All prompts are unique")

    # Check 3: Unique source codes
    print("\n3. Source Code Uniqueness:")
    sources = [ex['source_code'] for ex in examples]
    unique_sources = len(set(sources))
    duplicate_sources = len(sources) - unique_sources

    print(f"   Total source codes: {len(sources)}")
    print(f"   Unique source codes: {unique_sources}")

    if duplicate_sources > 0:
        print(f"   ✗ FAIL: {duplicate_sources} duplicate source codes!")
        all_checks_passed = False
    else:
        print("   ✓ PASS: All source codes are unique")

    # Check 4: Prompt→Source conflicts (CRITICAL)
    print("\n4. Prompt→Source Mapping:")
    prompt_to_sources = {}
    for ex in examples:
        p = ex['prompt']
        s = ex['source_code']
        if p not in prompt_to_sources:
            prompt_to_sources[p] = []
        prompt_to_sources[p].append(s)

    conflicts = {p: srcs for p, srcs in prompt_to_sources.items() if len(set(srcs)) > 1}

    if conflicts:
        print(f"   ✗ CRITICAL FAIL: {len(conflicts)} prompts map to multiple source codes!")
        print("   Examples:")
        for i, (prompt, sources) in enumerate(list(conflicts.items())[:3]):
            print(f"\n   Prompt: {prompt[:70]}...")
            for src in set(sources)[:3]:
                print(f"     → {src[:70]}...")
        all_checks_passed = False
    else:
        print("   ✓ PASS: No prompt→source conflicts")

    # Check 5: Data format
    print("\n5. Data Format:")
    required_fields = {'id', 'category', 'prompt', 'source_code', 'binary_hex', 'assembly', 'binary_size'}

    missing_fields_count = 0
    for i, ex in enumerate(examples[:10]):  # Check first 10
        missing = required_fields - set(ex.keys())
        if missing:
            print(f"   Example {i}: Missing fields {missing}")
            missing_fields_count += 1

    if missing_fields_count > 0:
        print(f"   ✗ FAIL: {missing_fields_count} examples have missing fields")
        all_checks_passed = False
    else:
        print("   ✓ PASS: All required fields present")

    # Check 6: Binary sizes
    print("\n6. Binary Sizes:")
    binary_sizes = [ex['binary_size'] for ex in examples]
    avg_size = sum(binary_sizes) / len(binary_sizes)
    min_size = min(binary_sizes)
    max_size = max(binary_sizes)

    print(f"   Average: {avg_size:.0f} bytes")
    print(f"   Min: {min_size} bytes")
    print(f"   Max: {max_size} bytes")

    if min_size < 1000:
        print(f"   WARNING: Some binaries are very small (<1KB)")
    if max_size > 10000:
        print(f"   WARNING: Some binaries are very large (>10KB)")

    # Sample examples
    print("\n7. Sample Examples:")
    sample_indices = [0, len(examples)//4, len(examples)//2, 3*len(examples)//4, len(examples)-1]

    for idx in sample_indices:
        if idx < len(examples):
            ex = examples[idx]
            print(f"\n   [{idx}] {ex['category']}")
            print(f"      Prompt: {ex['prompt'][:70]}")
            print(f"      Source: {ex['source_code'][:70].replace(chr(10), ' ')}")
            print(f"      Binary: {ex['binary_size']} bytes, Assembly: {len(ex['assembly'].split(chr(10)))} lines")

    # Final verdict
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED - Dataset is valid")
        return True
    else:
        print("✗ SOME CHECKS FAILED - Dataset has issues")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify synthetic dataset quality")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="./dataset/processed/synthetic_dataset.jsonl",
        help="Path to dataset JSONL file (default: ./dataset/processed/synthetic_dataset.jsonl)"
    )

    args = parser.parse_args()

    success = verify_dataset(args.dataset)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
