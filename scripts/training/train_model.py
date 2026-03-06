#!/usr/bin/env python3
"""Training script for fine-tuning Qwen3-0.6B on RISC-V binary generation."""

import argparse
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer
from torch import nn

class NonZeroWeightedTrainer(SFTTrainer):
    """Custom trainer with weighted loss for non-zero tokens."""

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

        # Get token IDs we need
        self.zero_token_id = tokenizer.encode('0', add_special_tokens=False)[0]

        # Get END_BINARY token ID (will be set after adding special token)
        end_tokens = tokenizer.encode('<END_BINARY>', add_special_tokens=False)
        self.end_token_id = end_tokens[0] if end_tokens else None

        print(f"Zero token ID: {self.zero_token_id}")
        print(f"END_BINARY token ID: {self.end_token_id}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss that weights non-zero tokens 5x."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Create weight tensor
        weights = torch.ones_like(labels, dtype=torch.float)

        # Weight all non-zero hex characters 5x
        non_zero_mask = (labels != self.zero_token_id) & (labels != -100)  # -100 is padding
        weights[non_zero_mask] = 5.0

        # Also weight END_BINARY token 5x if it exists
        if self.end_token_id is not None:
            end_mask = labels == self.end_token_id
            weights[end_mask] = 5.0

        # Compute weighted cross entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weighted_loss = (loss * shift_weights.view(-1)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

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
    parser.add_argument("--dataset", type=str, default="dataset/processed/training_dataset_hf",
                        help="Path to HuggingFace dataset directory (default: training_dataset_hf)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this training run (default: qwen3-0.6b-lora[-test])")
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Continue training from existing checkpoint")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--num-train-epochs", type=float, default=3.0,
                        help="Number of training epochs (default: 3.0)")
    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.max_steps = 10
        args.num_examples = 10
        print("🧪 Running in TEST mode: 10 examples, 10 steps")

    if args.from_checkpoint:
        print(f"Loading from checkpoint: {args.from_checkpoint}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.from_checkpoint,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        # Prepare for continued training
        print("Preparing model for continued QLoRA training...")
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
    else:
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

    # Add END_BINARY special token
    print("Adding END_BINARY special token...")
    special_tokens_dict = {'additional_special_tokens': ['<END_BINARY>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens")

    # Resize model embeddings to account for new token
    model.resize_token_embeddings(len(tokenizer))
    print(f"Token vocabulary size: {len(tokenizer)}")

    if torch.cuda.is_available():
        print(f"VRAM after model prep: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)

    # Limit examples if specified
    if args.num_examples:
        dataset = dataset.select(range(args.num_examples))
        print(f"Using {args.num_examples} examples")

    # Format with prompt template
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    print(f"Dataset size: {len(dataset)} examples")
    print(f"Sample: {dataset[0]['text'][:200]}...")

    # Training arguments
    if args.run_name:
        output_dir = f"./models/checkpoints/{args.run_name}"
    else:
        output_dir = "./models/checkpoints/qwen3-0.6b-lora-test" if args.test else "./models/checkpoints/qwen3-0.6b-lora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,    # Effective batch size = 4
        warmup_steps=10,
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=args.num_train_epochs if not args.max_steps else 1.0,
        learning_rate=args.learning_rate,
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

    print("Initializing trainer with weighted loss...")
    trainer = NonZeroWeightedTrainer(
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
