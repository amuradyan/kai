#!/usr/bin/env python3
"""Convert synthetic dataset to instruction-tuning format for training."""

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
    print("Loading synthetic dataset...")
    input_path = Path("dataset/processed/synthetic_dataset.jsonl")
    output_path = Path("dataset/processed/training_dataset.jsonl")

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
    dataset_hf_path = Path("dataset/processed/training_dataset_hf")
    dataset.save_to_disk(str(dataset_hf_path))

    print(f"✅ Saved Hugging Face dataset to {dataset_hf_path}")
    print(f"\nDataset info:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Features: {dataset.features}")

    # Show sample
    print(f"\nSample example:")
    print(f"  Instruction: {dataset[0]['instruction']}")
    print(f"  Output length: {len(dataset[0]['output'])} characters")

if __name__ == "__main__":
    main()
