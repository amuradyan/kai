#!/usr/bin/env python3
"""Fine-tune Qwen3-0.6B on the compact Phase 1 dataset.

Parallel to scripts/training/train_gpt2_compact.py for fair comparison:
same compact dataset (22-hex-char variable region + <END_BINARY>), same
recipe (lr, epochs, weighted loss, prompt template), but the existing
Qwen+Unsloth+QLoRA loader instead of HF/bf16. Differences from
train_model.py: compact-pipeline defaults (dataset, run name, lr,
epochs, max_seq_length=256) and the trl 0.24 API (SFTConfig +
processing_class).
"""

import argparse
import time
import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from torch import nn


class NonZeroWeightedTrainer(SFTTrainer):
    """Weights non-zero tokens 5x during loss computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = self.processing_class
        self.zero_token_id = tokenizer.encode('0', add_special_tokens=False)[0]
        end = tokenizer.encode('<END_BINARY>', add_special_tokens=False)
        self.end_token_id = end[0] if end else None
        print(f"Zero token ID: {self.zero_token_id}")
        print(f"END_BINARY token ID: {self.end_token_id}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        weights = torch.ones_like(labels, dtype=torch.float)
        non_zero = (labels != self.zero_token_id) & (labels != -100)
        weights[non_zero] = 5.0
        if self.end_token_id is not None:
            weights[labels == self.end_token_id] = 5.0

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        weighted = (loss * shift_weights.view(-1)).mean()
        return (weighted, outputs) if return_outputs else weighted


def make_format_prompt(tokenizer):
    """Format with Qwen3's native chat template, thinking disabled.

    Per Unsloth's Qwen3.5 guidance: "wrong chat template / EOS token at
    inference time" causes degradation. Using apply_chat_template at
    both train and inference keeps the format consistent.
    """
    def format_prompt(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                enable_thinking=False,
            )
        }
    return format_prompt


def main():
    p = argparse.ArgumentParser(description="Train Qwen3-0.6B on compact Phase 1 dataset")
    p.add_argument("--test", action="store_true", help="10 examples, 10 steps")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--dataset", default="dataset/processed/phase1_compact_hf")
    p.add_argument("--run-name", default=None)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-train-epochs", type=float, default=40.0)
    p.add_argument("--base-model", default="./models/base/qwen3-0.6b",
                   help="Local Qwen3-0.6B path (matches train_model.py)")
    p.add_argument("--max-seq-length", type=int, default=256,
                   help="Compact targets fit in <100 tokens; 256 is generous")
    p.add_argument("--quantize-4bit", action="store_true",
                   help="Opt into 4-bit QLoRA. Default is bf16 LoRA per Unsloth's "
                        "Qwen3.5 guidance (4-bit quant degrades small Qwen models).")
    args = p.parse_args()

    if args.test:
        args.max_steps = 10
        args.num_examples = 10
        print("🧪 TEST mode: 10 examples, 10 steps")

    t0 = time.time()
    mode = "4-bit QLoRA" if args.quantize_4bit else "bf16 LoRA"
    print(f"[{time.time()-t0:5.1f}s] Loading {args.base_model} via Unsloth ({mode})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None if args.quantize_4bit else torch.bfloat16,
        load_in_4bit=args.quantize_4bit,
    )
    print(f"[{time.time()-t0:5.1f}s] Model + tokenizer loaded")

    print(f"[{time.time()-t0:5.1f}s] Applying QLoRA (Llama-style targets)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Unsloth Qwen3.5 recipe
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"[{time.time()-t0:5.1f}s] Adding END_BINARY special token...")
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<END_BINARY>"]}
    )
    print(f"  Added {n_added} special tokens; vocab size now {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    if torch.cuda.is_available():
        print(f"[{time.time()-t0:5.1f}s] VRAM after model prep: "
              f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print(f"[{time.time()-t0:5.1f}s] Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    if args.num_examples:
        dataset = dataset.select(range(args.num_examples))
        print(f"  Limited to {args.num_examples} examples")
    dataset = dataset.map(make_format_prompt(tokenizer), remove_columns=dataset.column_names)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Sample: {dataset[0]['text']}")

    if args.run_name:
        out = f"./models/checkpoints/{args.run_name}"
    else:
        out = (
            "./models/checkpoints/qwen3-compact-test"
            if args.test
            else "./models/checkpoints/qwen3-compact"
        )

    training_args = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_length,
    )

    print(f"[{time.time()-t0:5.1f}s] Initializing trainer with weighted loss...")
    trainer = NonZeroWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n" + "=" * 60)
    print(f"[{time.time()-t0:5.1f}s] Starting training...")
    print("=" * 60 + "\n")
    trainer.train()
    print(f"\n✅ [{time.time()-t0:5.1f}s] Training complete")

    print(f"[{time.time()-t0:5.1f}s] Saving to {out}...")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if torch.cuda.is_available():
        print(f"  Final VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Peak reserved: {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")
    print(f"\n✅ [{time.time()-t0:5.1f}s] Saved to {out}")


if __name__ == "__main__":
    main()
