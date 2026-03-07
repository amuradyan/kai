#!/usr/bin/env python3
"""Train model with progressive weighting: non-zeros get higher weights further in sequence (1x to 5x)."""

import sys
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_from_disk
from transformers import TrainingArguments
from torch import nn

class ProgressiveWeightedTrainer(SFTTrainer):
    """Custom trainer with progressive weighting based on position."""

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss with progressive weighting based on position."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        batch_size, seq_len = labels.shape

        # Create base weight tensor
        weights = torch.ones_like(labels, dtype=torch.float)

        # Mask for valid positions (not padding)
        valid_mask = labels != -100  # -100 is padding

        # Non-zero tokens mask
        non_zero_mask = (labels != self.zero_token_id) & valid_mask

        # Calculate position-based weights for non-zeros
        # Create position ratios (0 to 1 across sequence)
        positions = torch.arange(seq_len, device=labels.device).float()
        position_ratios = positions / (seq_len - 1)  # 0 to 1

        # Expand position ratios to match batch size
        position_ratios = position_ratios.unsqueeze(0).expand(batch_size, -1)

        # Progressive weight formula: 1 + (position_ratio * 4) → ranges from 1x to 5x
        progressive_weights = 1.0 + (position_ratios * 4.0)

        # Apply progressive weights only to non-zero tokens
        weights[non_zero_mask] = progressive_weights[non_zero_mask]

        # END_BINARY token gets maximum weight (5x)
        if self.end_token_id is not None:
            end_mask = labels == self.end_token_id
            weights[end_mask] = 5.0

        # Log statistics occasionally
        if torch.rand(1).item() < 0.01:  # 1% chance to log
            total_tokens = valid_mask.sum().item()
            nonzero_count = non_zero_mask.sum().item()

            # Calculate average weight for non-zeros in different regions
            if nonzero_count > 0:
                # Split sequence into quarters for statistics
                quarter_len = seq_len // 4

                q1_mask = non_zero_mask & (positions < quarter_len).unsqueeze(0).expand(batch_size, -1)
                q2_mask = non_zero_mask & (positions >= quarter_len) & (positions < 2*quarter_len).unsqueeze(0).expand(batch_size, -1)
                q3_mask = non_zero_mask & (positions >= 2*quarter_len) & (positions < 3*quarter_len).unsqueeze(0).expand(batch_size, -1)
                q4_mask = non_zero_mask & (positions >= 3*quarter_len).unsqueeze(0).expand(batch_size, -1)

                q1_avg = weights[q1_mask].mean().item() if q1_mask.any() else 0
                q2_avg = weights[q2_mask].mean().item() if q2_mask.any() else 0
                q3_avg = weights[q3_mask].mean().item() if q3_mask.any() else 0
                q4_avg = weights[q4_mask].mean().item() if q4_mask.any() else 0

                print(f"\n[Progressive Weights] Total: {total_tokens}, Non-zeros: {nonzero_count}")
                print(f"  Q1 avg: {q1_avg:.1f}x, Q2 avg: {q2_avg:.1f}x, "
                      f"Q3 avg: {q3_avg:.1f}x, Q4 avg: {q4_avg:.1f}x")

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
        output_dir="./models/checkpoints/progressive-weighted",
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

    # Create trainer with progressive weighting
    trainer = ProgressiveWeightedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        formatting_func=format_prompt,
        max_seq_length=2048,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training with progressive weighting:")
    print("- Non-zeros: 1x (start) → 5x (end)")
    print("- Zeros: 1x weight")
    print("- END_BINARY: 5x weight")
    print("="*60 + "\n")

    trainer.train()

    # Save model
    print("Saving model...")
    model.save_pretrained("./models/checkpoints/progressive-weighted")
    tokenizer.save_pretrained("./models/checkpoints/progressive-weighted")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()