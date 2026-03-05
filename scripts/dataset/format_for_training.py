#!/usr/bin/env python3
"""Convert synthetic dataset to instruction-tuning format for training."""

import argparse
import json
from pathlib import Path
from datasets import Dataset

def format_for_training(example):
    """Format a single example as instruction-output pair."""
    return {
        "instruction": example["prompt"],
        "output": example["binary_hex"]
    }

def main():
    parser = argparse.ArgumentParser(
        description="Format dataset for instruction tuning"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/processed/synthetic_dataset.jsonl",
        help="Input JSONL dataset file (default: synthetic_dataset.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/processed/training_dataset",
        help="Output base name without extensions (default: training_dataset)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_base = Path(args.output)

    print(f"Loading dataset from {input_path}...")

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    output_path = output_base.with_suffix(".jsonl")
    output_hf_path = Path(str(output_base) + "_hf")

    # Read and convert
    formatted_examples = []
    with open(input_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            formatted = format_for_training(example)
            formatted_examples.append(formatted)

    print(f"Loaded {len(formatted_examples)} examples")

    # Save formatted dataset
    with open(output_path, 'w') as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Saved formatted dataset to {output_path}")

    # Create Hugging Face dataset
    dataset = Dataset.from_list(formatted_examples)
    dataset.save_to_disk(str(output_hf_path))

    print(f"✅ Saved Hugging Face dataset to {output_hf_path}")
    print(f"\nDataset info:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Features: {dataset.features}")

    # Show sample
    print(f"\nSample example:")
    print(f"  Instruction: {dataset[0]['instruction']}")
    print(f"  Output length: {len(dataset[0]['output'])} characters")

if __name__ == "__main__":
    main()
