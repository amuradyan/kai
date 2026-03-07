#!/usr/bin/env python3
"""Generate RISC-V binaries from natural language prompts using trained model."""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# Suppress transformers deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Set transformers logging to ERROR only (suppresses warnings)
logging.getLogger("transformers").setLevel(logging.ERROR)


def load_model(checkpoint_path: str, max_seq_length: int = 2048):
    """Load the fine-tuned model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Load the adapter config to get the base model path
    import json
    with open(f"{checkpoint_path}/adapter_config.json", "r") as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path", "./models/base/qwen3-0.6b")
    print(f"Loading from base model: {base_model_path}")

    # First load the base model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Load the tokenizer from checkpoint (which has the END_BINARY token)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Resize model embeddings to match the tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Now load the adapter weights
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
        is_trainable=False,
    )

    # Check if END_BINARY token is in the tokenizer
    if '<END_BINARY>' in tokenizer.get_vocab():
        print("✓ END_BINARY token found in checkpoint")
    else:
        print("Warning: END_BINARY token not found. This checkpoint may not support END_BINARY.")

    # Enable inference mode (faster, no grad)
    FastLanguageModel.for_inference(model)

    print("✅ Model loaded successfully")
    return model, tokenizer


def format_prompt(instruction: str) -> str:
    """Format instruction as prompt matching training format."""
    return f"""### Instruction:
{instruction}

### Response:
"""


def generate_binary(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.0,
    timeout_seconds: int = 60,
) -> str:
    """Generate binary from prompt.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: Natural language instruction
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy, deterministic)

    Returns:
        Generated hex-encoded binary string
    """
    # Format prompt
    formatted_prompt = format_prompt(prompt)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {prompt}")
    print(f"Generating binary (max {max_new_tokens} tokens)...")

    # Generate with greedy decoding (temperature=0 for deterministic output)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=False,
        )

    # Decode output (keep special tokens to find END_BINARY)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the response part (after "### Response:")
    if "### Response:" in full_output:
        response = full_output.split("### Response:")[1].strip()
    else:
        response = full_output

    # Remove END_BINARY token if present
    if "<END_BINARY>" in response:
        response = response.split("<END_BINARY>")[0]
        print("Found END_BINARY token, generation stopped correctly")
    else:
        print("Warning: END_BINARY token not found in generation")

    return response


def save_binary(hex_string: str, output_path: str):
    """Convert hex string to binary file."""
    try:
        # Remove any whitespace/newlines
        hex_clean = ''.join(hex_string.split())

        # Convert hex to bytes
        binary_data = bytes.fromhex(hex_clean)

        # Write to file
        output_file = Path(output_path)
        output_file.write_bytes(binary_data)

        # Make executable
        output_file.chmod(0o755)

        print(f"✅ Binary saved to {output_path} ({len(binary_data)} bytes)")
        return True

    except ValueError as e:
        print(f"❌ Invalid hex string: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate RISC-V binaries from natural language prompts"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/checkpoints/qwen3-0.6b-lora",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Natural language instruction (e.g., 'Write a C program that returns 42')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./generated_binary",
        help="Output binary file path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096, use ~500 for base model testing)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Generation timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--show-hex",
        action="store_true",
        help="Print generated hex string",
    )

    args = parser.parse_args()

    # Check if model exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Model not found: {args.checkpoint}")
        print("\nHave you trained the model yet?")
        print("Run: ./kai train")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(args.checkpoint)

    # Generate binary
    try:
        hex_output = generate_binary(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n\n❌ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Generation failed: {e}")
        sys.exit(1)

    # Save raw output for debugging
    raw_output_path = args.output + ".raw"
    with open(raw_output_path, 'w') as f:
        f.write(hex_output)
    print(f"\n📝 Raw output saved to: {raw_output_path}")

    # Show hex if requested
    if args.show_hex:
        print(f"\nGenerated hex ({len(hex_output)} chars):")
        print(hex_output[:200] + "..." if len(hex_output) > 200 else hex_output)

    # Save to file
    if save_binary(hex_output, args.output):
        print(f"\n✅ Generation complete!")
        print(f"\nTo test the binary on QEMU:")
        print(f"  qemu-riscv64 {args.output}")
        print(f"  echo $?  # Check return code")
    else:
        print("\n❌ Failed to save binary (invalid hex output)")
        print("\nThis is expected if the model is untrained.")
        print("The model needs to be fine-tuned first to generate valid binaries.")
        sys.exit(1)


if __name__ == "__main__":
    main()
