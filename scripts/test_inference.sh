#!/usr/bin/env bash
# Test the inference script with known good data from dataset

set -e

echo "=== Testing Inference Script ==="
echo ""

# Test 1: Save binary function with dataset example
echo "Test 1: Save binary from dataset"
echo "--------------------------------"

python3 << 'EOF'
import json
import sys
sys.path.insert(0, '/home/spectrum/playground/kai/scripts')
from generate import save_binary
import subprocess

# Read one example from the dataset
with open('dataset/processed/synthetic_dataset.jsonl', 'r') as f:
    example = json.loads(f.readline())

print(f"Prompt: {example['prompt']}")
print(f"Source: {example['source_code']}")
print(f"Binary hex length: {len(example['binary_hex'])} chars")

# Test with save_binary
result = save_binary(example['binary_hex'], "/tmp/dataset_binary_test")
print(f"\nSave result: {result}")

if result:
    print("\nTesting binary on QEMU:")
    proc = subprocess.run(
        ['qemu-riscv64', '/tmp/dataset_binary_test'],
        capture_output=True
    )
    print(f"Exit code: {proc.returncode}")

    if proc.returncode >= 0 and proc.returncode < 256:
        print(f"✅ Binary is valid and executable!")
    else:
        print(f"❌ Binary failed to execute properly")
        sys.exit(1)
else:
    print("❌ Failed to save binary")
    sys.exit(1)
EOF

echo ""
echo "Test 2: Try base model generation (expected to fail)"
echo "-----------------------------------------------------"

python scripts/generate.py \
    --base-model \
    --prompt "Write a C program that returns 42" \
    --output /tmp/base_model_test \
    --max-tokens 200 \
    --timeout 30 || echo "Expected failure: base model produces text, not hex"

echo ""
echo "Test 3: Check for trained model"
echo "--------------------------------"

if [ -d "./models/checkpoints/qwen3-0.6b-lora" ]; then
    echo "✅ Trained model found at ./models/checkpoints/qwen3-0.6b-lora"
    echo ""
    echo "Test 4: Generate with trained model"
    echo "------------------------------------"

    python scripts/generate.py \
        --prompt "Write a C program that returns 42" \
        --output /tmp/trained_model_test \
        --show-hex

    if [ -f "/tmp/trained_model_test" ]; then
        echo ""
        echo "Testing generated binary:"
        qemu-riscv64 /tmp/trained_model_test
        EXIT_CODE=$?
        echo "Exit code: $EXIT_CODE"

        if [ $EXIT_CODE -eq 42 ]; then
            echo "✅✅✅ SUCCESS! Model generated correct binary!"
        else
            echo "⚠️  Binary executed but returned $EXIT_CODE (expected 42)"
        fi
    fi
else
    echo "⚠️  No trained model found"
    echo "Run: python scripts/train.py"
fi

echo ""
echo "=== Tests Complete ==="
