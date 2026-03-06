#!/usr/bin/env python3
"""Create focused dataset for Phase 1 curriculum training."""

import argparse
import json
import random
from pathlib import Path
from typing import List, Set


def create_phase1_subset(
    input_path: Path,
    output_path: Path,
    num_small: int = 50,
    num_large: int = 50,
    hold_out_values: Set[int] = None,
    seed: int = 42
) -> None:
    """Create Phase 1 subset with strategic value selection.

    Args:
        input_path: Path to full returns dataset
        output_path: Output path for subset
        num_small: Number of small values (1-31)
        num_large: Number of large values (>=32)
        hold_out_values: Values to exclude for validation
        seed: Random seed for reproducible selection
    """
    random.seed(seed)

    if hold_out_values is None:
        hold_out_values = set()

    # Read full dataset
    examples_by_value = {}
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            value = int(data['prompt'].split()[-1])
            examples_by_value[value] = data

    # Select small values (1-31, compressed instructions)
    small_candidates = [v for v in range(1, 32) if v not in hold_out_values and v in examples_by_value]

    if len(small_candidates) < num_small:
        print(f"Warning: Only {len(small_candidates)} small values available, requested {num_small}")
        selected_small = small_candidates
    else:
        # Strategic selection: spread across range
        selected_small = []

        # Include important boundaries
        for v in [1, 2, 31]:
            if v in small_candidates:
                selected_small.append(v)
                small_candidates.remove(v)

        # Fill remaining slots randomly
        remaining = num_small - len(selected_small)
        selected_small.extend(random.sample(small_candidates, min(remaining, len(small_candidates))))

    # Select large values (>=32, uncompressed instructions)
    large_candidates = [v for v in examples_by_value.keys() if v >= 32 and v not in hold_out_values]

    if len(large_candidates) < num_large:
        print(f"Warning: Only {len(large_candidates)} large values available, requested {num_large}")
        selected_large = large_candidates
    else:
        # Strategic selection: logarithmic spread
        selected_large = []

        # Include important values
        key_values = [32, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000, 7000, 9000, 10000]
        for v in key_values:
            if v in large_candidates and len(selected_large) < num_large:
                selected_large.append(v)
                large_candidates.remove(v)

        # Fill remaining slots randomly from what's left
        remaining = num_large - len(selected_large)
        if remaining > 0 and large_candidates:
            selected_large.extend(random.sample(large_candidates, min(remaining, len(large_candidates))))

    # Combine and sort
    selected_values = sorted(selected_small + selected_large)

    # Create subset dataset
    subset = []
    for value in selected_values:
        if value in examples_by_value:
            subset.append(examples_by_value[value])

    # Write subset to file
    with open(output_path, 'w') as f:
        for example in subset:
            f.write(json.dumps(example) + '\n')

    # Print summary
    print(f"Created Phase 1 dataset: {output_path}")
    print(f"  Small values (1-31):  {len(selected_small)}")
    print(f"  Large values (>=32):  {len(selected_large)}")
    print(f"  Total examples:       {len(subset)}")
    print(f"  Held out for validation: {sorted(hold_out_values)}")

    # Show value distribution
    print(f"\nValue distribution:")
    print(f"  Small: {sorted(selected_small)[:10]}{'...' if len(selected_small) > 10 else ''}")
    print(f"  Large: {sorted(selected_large)[:10]}{'...' if len(selected_large) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Create focused dataset for Phase 1 curriculum training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/processed/returns_dataset.jsonl"),
        help="Source dataset (default: dataset/processed/returns_dataset.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/phase1_dataset.jsonl"),
        help="Output subset (default: dataset/processed/phase1_dataset.jsonl)"
    )
    parser.add_argument(
        "--num-small",
        type=int,
        default=50,
        help="Number of small values 1-31 (default: 50)"
    )
    parser.add_argument(
        "--num-large",
        type=int,
        default=50,
        help="Number of large values >=32 (default: 50)"
    )
    parser.add_argument(
        "--hold-out",
        type=str,
        default="15,750",
        help="Comma-separated values to exclude for validation (default: 15,750)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection (default: 42)"
    )

    args = parser.parse_args()

    # Parse hold-out values
    hold_out_values = set()
    if args.hold_out:
        for v in args.hold_out.split(','):
            try:
                hold_out_values.add(int(v.strip()))
            except ValueError:
                print(f"Warning: Invalid hold-out value: {v}")

    # Check input exists
    if not args.input.exists():
        print(f"Error: Input dataset not found: {args.input}")
        return 1

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create subset
    create_phase1_subset(
        args.input,
        args.output,
        args.num_small,
        args.num_large,
        hold_out_values,
        args.seed
    )

    return 0


if __name__ == "__main__":
    exit(main())