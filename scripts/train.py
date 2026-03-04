#!/usr/bin/env python3
"""Training script for fine-tuning Qwen3-0.6B on RISC-V binary generation."""

import argparse
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

def format_prompt(example):
    """Format example with chat template."""
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-0.6B for RISC-V binary generation")
    parser.add_argument("--test", action="store_true", help="Run small-scale test (10 examples, 10 steps)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (overrides epochs)")
    parser.add_argument("--num-examples", type=int, default=None, help="Limit number of training examples")
    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.max_steps = 10
        args.num_examples = 10
        print("🧪 Running in TEST mode: 10 examples, 10 steps")

    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='./models/base/qwen3-0.6b',
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print("Preparing model for QLoRA training...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                      # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if torch.cuda.is_available():
        print(f"VRAM after model prep: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print("Loading dataset...")
    dataset = load_from_disk("dataset/processed/training_dataset_hf")

    # Limit examples if specified
    if args.num_examples:
        dataset = dataset.select(range(args.num_examples))
        print(f"Using {args.num_examples} examples")

    # Format with prompt template
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Sample: {dataset[0]['text'][:200]}...")

    # Training arguments
    output_dir = "./models/checkpoints/qwen3-0.6b-lora-test" if args.test else "./models/checkpoints/qwen3-0.6b-lora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,    # Effective batch size = 4
        warmup_steps=10,
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=3.0 if not args.max_steps else 1.0,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps" if args.test else "epoch",
        save_steps=5 if args.test else 500,
        report_to="none",              # Disable wandb for now
    )

    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    # Train
    trainer.train()

    print("\n✅ Training complete!")

    # Save model
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if torch.cuda.is_available():
        print(f"\nFinal VRAM usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")

    print(f"\n✅ Model saved to {output_dir}")

if __name__ == "__main__":
    main()
