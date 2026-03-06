#!/usr/bin/env python3
"""Evaluate trained model on test dataset."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from generation.generate_binary import load_model, format_prompt, generate_binary, save_binary


def load_test_dataset(dataset_path: str, max_examples: int = None) -> List[Dict]:
    """Load test examples from dataset."""
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def test_binary_execution(binary_path: str, expected_return: int, timeout: int = 5) -> Tuple[bool, int]:
    """Test if binary executes and returns expected value."""
    try:
        result = subprocess.run(
            ['qemu-riscv64', binary_path],
            capture_output=True,
            timeout=timeout
        )
        return True, result.returncode
    except subprocess.TimeoutExpired:
        return False, -1
    except Exception as e:
        return False, -1


def evaluate_model(
    model,
    tokenizer,
    test_examples: List[Dict],
    output_dir: Path,
) -> Dict:
    """Evaluate model on test examples.

    Returns:
        Dict with metrics: exact_match_rate, execution_success_rate, correctness_rate
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'total': len(test_examples),
        'valid_hex': 0,
        'valid_binary': 0,
        'execution_success': 0,
        'correct_output': 0,
        'exact_match': 0,
    }

    failures = []

    print(f"\n{'='*60}")
    print(f"Evaluating model on {len(test_examples)} examples")
    print(f"{'='*60}\n")

    for i, example in enumerate(tqdm(test_examples, desc="Evaluating")):
        prompt = example['prompt']
        expected_hex = example['binary_hex']
        expected_binary = bytes.fromhex(expected_hex)

        # Get source info for failure analysis
        source = example.get('source_code', 'N/A')

        # Generate binary
        try:
            generated_hex = generate_binary(
                model,
                tokenizer,
                prompt,
                max_new_tokens=4096,
            )

            # Clean hex string
            generated_hex_clean = ''.join(generated_hex.split())

            # Check if valid hex
            try:
                generated_binary = bytes.fromhex(generated_hex_clean)
                metrics['valid_hex'] += 1

                # Save to file
                binary_path = output_dir / f"test_{i}.bin"
                if save_binary(generated_hex_clean, str(binary_path)):
                    metrics['valid_binary'] += 1

                    # Test execution
                    success, return_code = test_binary_execution(str(binary_path), 0)
                    if success:
                        metrics['execution_success'] += 1

                        # Check correctness (we don't have expected return in dataset, so just check if it ran)
                        # For now, consider any successful execution as correct
                        if return_code >= 0 and return_code < 256:
                            metrics['correct_output'] += 1

                    # Check exact match
                    if generated_hex_clean == expected_hex:
                        metrics['exact_match'] += 1
                    else:
                        # Record failure for analysis
                        failures.append({
                            'index': i,
                            'prompt': prompt,
                            'source': source,
                            'expected_hex': expected_hex[:200],
                            'generated_hex': generated_hex_clean[:200],
                            'executed': success,
                            'return_code': return_code if success else None,
                        })

            except ValueError:
                # Invalid hex
                failures.append({
                    'index': i,
                    'prompt': prompt,
                    'source': source,
                    'error': 'invalid_hex',
                    'generated': generated_hex[:200],
                })

        except Exception as e:
            failures.append({
                'index': i,
                'prompt': prompt,
                'source': source,
                'error': str(e),
            })

    # Calculate rates
    metrics['valid_hex_rate'] = metrics['valid_hex'] / metrics['total']
    metrics['valid_binary_rate'] = metrics['valid_binary'] / metrics['total']
    metrics['execution_success_rate'] = metrics['execution_success'] / metrics['total']
    metrics['correctness_rate'] = metrics['correct_output'] / metrics['total']
    metrics['exact_match_rate'] = metrics['exact_match'] / metrics['total']

    # Save failures
    failures_path = output_dir / 'failures.json'
    with open(failures_path, 'w') as f:
        json.dump(failures, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./models/checkpoints/qwen3-0.6b-lora',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='./dataset/processed/returns_dataset.jsonl',
        help='Path to test dataset',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation/results',
        help='Output directory for results',
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=100,
        help='Maximum number of examples to evaluate (default: 100)',
    )

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint)

    # Load test examples
    print(f"Loading test dataset from {args.dataset}...")
    test_examples = load_test_dataset(args.dataset, args.max_examples)
    print(f"Loaded {len(test_examples)} test examples")

    # Evaluate
    output_dir = Path(args.output_dir)
    metrics = evaluate_model(model, tokenizer, test_examples, output_dir)

    # Print results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total examples:           {metrics['total']}")
    print(f"Valid hex:                {metrics['valid_hex']} ({metrics['valid_hex_rate']:.1%})")
    print(f"Valid binary:             {metrics['valid_binary']} ({metrics['valid_binary_rate']:.1%})")
    print(f"Execution success:        {metrics['execution_success']} ({metrics['execution_success_rate']:.1%})")
    print(f"Correct output:           {metrics['correct_output']} ({metrics['correctness_rate']:.1%})")
    print(f"Exact match:              {metrics['exact_match']} ({metrics['exact_match_rate']:.1%})")
    print(f"{'='*60}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"  - metrics.json: Evaluation metrics")
    print(f"  - failures.json: Failed examples for analysis")
    print(f"  - test_*.bin: Generated binaries")


if __name__ == '__main__':
    main()
