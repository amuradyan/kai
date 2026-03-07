#!/usr/bin/env python3
"""Train model with dynamic weighting: footer non-zeros 10x, other non-zeros 2x."""

import sys
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_from_disk
from transformers import TrainingArguments
from torch import nn

class DynamicWeightedTrainer(SFTTrainer):
    """Custom trainer with dynamic weighting for different regions."""

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

        # Get token IDs we need
        self.zero_token_id = tokenizer.encode('0', add_special_tokens=False)[0]

        # Get END_BINARY token ID
        end_tokens = tokenizer.encode('<END_BINARY>', add_special_tokens=False)
        self.end_token_id = end_tokens[0] if end_tokens else None

        print(f"Zero token ID: {self.zero_token_id}")
        print(f"END_BINARY token ID: {self.end_token_id}")

    def identify_footer_region(self, labels):
        """Identify footer regions (after long zero sequences)."""
        batch_size, seq_len = labels.shape
        is_footer = torch.zeros_like(labels, dtype=torch.bool)

        for b in range(batch_size):
            consecutive_zeros = 0
            in_footer = False

            for i in range(seq_len):
                if labels[b, i] == self.zero_token_id:
                    consecutive_zeros += 1
                    # After 20+ consecutive zeros, we're likely in footer padding
                    if consecutive_zeros >= 20:
                        in_footer = True
                else:
                    # Reset counter on non-zero, but stay in footer if already there
                    consecutive_zeros = 0

                if in_footer:
                    is_footer[b, i] = True

        return is_footer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss with dynamic weighting based on region."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # Identify footer regions
        is_footer = self.identify_footer_region(labels)

        # Create weight tensor
        weights = torch.ones_like(labels, dtype=torch.float)

        # Mask for valid positions (not padding)
        valid_mask = labels != -100  # -100 is padding

        # Non-zero tokens mask
        non_zero_mask = (labels != self.zero_token_id) & valid_mask

        # Apply dynamic weights:
        # - Footer non-zeros: 100x weight
        # - Regular non-zeros: 2x weight
        # - Zeros: 1x weight (default)
        footer_nonzero_mask = is_footer & non_zero_mask
        regular_nonzero_mask = ~is_footer & non_zero_mask

        weights[footer_nonzero_mask] = 100.0
        weights[regular_nonzero_mask] = 2.0

        # Also weight END_BINARY token heavily (100x)
        if self.end_token_id is not None:
            end_mask = labels == self.end_token_id
            weights[end_mask] = 100.0

        # Log statistics occasionally
        if torch.rand(1).item() < 0.01:  # 1% chance to log
            total_tokens = valid_mask.sum().item()
            footer_tokens = (is_footer & valid_mask).sum().item()
            footer_nonzeros = footer_nonzero_mask.sum().item()
            regular_nonzeros = regular_nonzero_mask.sum().item()

            print(f"\n[Dynamic Weights] Total: {total_tokens}, "
                  f"Footer region: {footer_tokens} ({footer_tokens/max(total_tokens,1)*100:.1f}%), "
                  f"Footer non-zeros (100x): {footer_nonzeros}, "
                  f"Regular non-zeros (2x): {regular_nonzeros}")

        # Compute weighted cross entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weighted_loss = (loss * shift_weights.view(-1)).mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

def format_prompt(examples):
    """Format examples for training."""
    outputs = []
    for i in range(len(examples['instruction'])):
        text = f"""### Instruction:
{examples['instruction'][i]}

### Response:
{examples['output'][i]}"""
        outputs.append(text)
    return outputs

def main():
    # Load model
    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./models/base/qwen3-0.6b",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': ['<END_BINARY>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added_toks > 0:
        print(f"Added {num_added_toks} special tokens")
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))

    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk("./dataset/processed/phase1_training_hf")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/checkpoints/dynamic-weighted",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    # Create trainer with dynamic weighting
    trainer = DynamicWeightedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_prompt,
        max_seq_length=2048,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training with dynamic weighting:")
    print("- Footer non-zeros: 100x weight")
    print("- Regular non-zeros: 2x weight")
    print("- Zeros: 1x weight")
    print("="*60 + "\n")

    trainer.train()

    # Save model
    print("Saving model...")
    model.save_pretrained("./models/checkpoints/dynamic-weighted")
    tokenizer.save_pretrained("./models/checkpoints/dynamic-weighted")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()