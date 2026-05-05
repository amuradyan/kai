# Testing GPT-2 with the toolchain

Goal: validate that GPT-2 small (~124M) can be plugged into the kai
pipeline as a swap for Qwen3-0.6B, so we can run the FP-CPU experiment
on a low-code-prior base. This is the smoke test, not a real training
run.

## Why a parallel pipeline (and not a fork of `train_model.py`)

The current pipeline is built on **Unsloth**:

- `scripts/training/train_model.py` imports `from unsloth import FastLanguageModel`.
- `scripts/generation/generate_binary.py` does the same.

Unsloth's supported model list is Llama / Mistral / Gemma / Qwen / Phi /
Granite. **GPT-2 is not on it** — Unsloth ships hand-rewritten kernels
for those families and does not have a fallback for the GPT-2
architecture. Trying to load `gpt2` through `FastLanguageModel.from_pretrained`
either fails or silently does the wrong thing. So we need plain HF
`transformers` + `peft` for this experiment.

That's the only fundamental change. Concrete differences from the
Unsloth pipeline:

1. `from unsloth import FastLanguageModel` → `AutoModelForCausalLM.from_pretrained`.
2. LoRA `target_modules`: Llama-style names (`q_proj`, `k_proj`, `v_proj`,
   `o_proj`, `gate_proj`, `up_proj`, `down_proj`) → GPT-2 module names
   (`c_attn`, `c_proj`, `c_fc`). Wrong names here yield
   `ValueError: Target modules ... not found`.
3. `max_seq_length`: 2048 → **1024**. GPT-2 small's positional
   embeddings only go to 1024.
4. `tokenizer.pad_token` is unset on GPT-2 — alias it to `eos_token`
   before training, otherwise SFTTrainer complains.
5. No `load_in_4bit` — 124M params at bf16 fits in <300 MB; quantization
   is unnecessary.

Everything else stays: dataset format, `### Instruction:` / `### Response:`
prompt template, `<END_BINARY>` special token, the
`NonZeroWeightedTrainer` weighted-loss trick, the JSONL → HF dataset
flow.

## Known constraint: context length

The current Phase 1 examples produce 1696 hex characters per binary
(848-byte binaries × 2 hex chars). With GPT-2's BPE plus the prompt
template overhead, this lands around **~1700–1900 tokens** — over GPT-2's
1024 limit. Practical effects:

- The 10-step smoke test (`--test`) runs fine; `SFTTrainer` truncates
  per `max_seq_length` and just drops the tail.
- A real training run **will silently truncate every example**. The
  model cannot learn the full binary because it never sees the full
  binary in context. To get end-to-end learning we either shorten the
  target (drop the ELF wrapper, train on just the code section) or move
  to a longer-context base.

Worth knowing before committing to a long run.

## New scripts

Two new files alongside the existing pipeline. Inline below — copy into
place when ready to run.

- `scripts/training/train_gpt2.py`
- `scripts/generation/generate_gpt2.py`

The `kai` CLI wrapper does not need to change. Invoke the new scripts
directly with `python`.

### `scripts/training/train_gpt2.py`

```python
#!/usr/bin/env python3
"""Fine-tune GPT-2 small on RISC-V binary generation.

Parallel to scripts/training/train_model.py. Uses plain HF transformers
+ PEFT instead of Unsloth, since Unsloth does not support GPT-2.
"""

import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from torch import nn


class NonZeroWeightedTrainer(SFTTrainer):
    """Weights non-zero tokens 5x during loss computation."""

    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
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


def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }


def main():
    p = argparse.ArgumentParser(description="Train GPT-2 small for RISC-V binary generation")
    p.add_argument("--test", action="store_true", help="10 examples, 10 steps")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-examples", type=int, default=None)
    p.add_argument("--dataset", default="dataset/processed/training_dataset_hf")
    p.add_argument("--run-name", default=None)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--base-model", default="gpt2",
                   help="HF id (default: gpt2 = 124M)")
    p.add_argument("--max-seq-length", type=int, default=1024,
                   help="GPT-2 small ceiling is 1024")
    args = p.parse_args()

    if args.test:
        args.max_steps = 10
        args.num_examples = 10
        print("🧪 TEST mode: 10 examples, 10 steps")

    print(f"Loading {args.base_model} via HF transformers...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)

    print("Adding END_BINARY special token before LoRA wrap...")
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<END_BINARY>"]}
    )
    print(f"Added {n_added} special tokens")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Vocab size: {len(tokenizer)}")

    print("Applying LoRA (targets: c_attn, c_proj, c_fc)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if torch.cuda.is_available():
        model = model.cuda()
        print(f"VRAM after model prep: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    if args.num_examples:
        dataset = dataset.select(range(args.num_examples))
        print(f"Limited to {args.num_examples} examples")
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample (first 200 chars): {dataset[0]['text'][:200]}...")

    if args.run_name:
        out = f"./models/checkpoints/{args.run_name}"
    else:
        out = (
            "./models/checkpoints/gpt2-small-lora-test"
            if args.test
            else "./models/checkpoints/gpt2-small-lora"
        )

    training_args = TrainingArguments(
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
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps" if args.test else "epoch",
        save_steps=5 if args.test else 500,
        report_to="none",
    )

    print("Initializing trainer with weighted loss...")
    trainer = NonZeroWeightedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    trainer.train()
    print("\n✅ Training complete")

    print(f"Saving to {out}...")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if torch.cuda.is_available():
        print(f"Final VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Peak reserved: {torch.cuda.max_memory_reserved(0) / 1024**3:.2f} GB")
    print(f"\n✅ Saved to {out}")


if __name__ == "__main__":
    main()
```

### `scripts/generation/generate_gpt2.py`

```python
#!/usr/bin/env python3
"""Generate RISC-V binaries from prompts using a GPT-2 + PEFT checkpoint.

Parallel to scripts/generation/generate_binary.py. No Unsloth.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(checkpoint_path: str):
    print(f"Loading {checkpoint_path}...")
    with open(f"{checkpoint_path}/adapter_config.json") as f:
        adapter_config = json.load(f)
    base = adapter_config.get("base_model_name_or_path", "gpt2")
    print(f"Base model: {base}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print("✅ Model loaded")
    return model, tokenizer


def format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_binary(model, tokenizer, prompt, max_new_tokens=900):
    formatted = format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    print(f"\nPrompt: {prompt}")
    print(f"Generating up to {max_new_tokens} tokens...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "### Response:" in full:
        response = full.split("### Response:")[1].strip()
    else:
        response = full
    if "<END_BINARY>" in response:
        response = response.split("<END_BINARY>")[0]
        print("Found END_BINARY")
    else:
        print("Warning: END_BINARY not found")
    return response


def save_binary(hex_string, output_path):
    try:
        clean = ''.join(hex_string.split())
        data = bytes.fromhex(clean)
        out = Path(output_path)
        out.write_bytes(data)
        out.chmod(0o755)
        print(f"✅ Saved {output_path} ({len(data)} bytes)")
        return True
    except ValueError as e:
        print(f"❌ Invalid hex: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./models/checkpoints/gpt2-small-lora")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", default="./generated_binary")
    p.add_argument("--max-tokens", type=int, default=900)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, tok = load_model(args.checkpoint)
    hex_out = generate_binary(model, tok, args.prompt, args.max_tokens)
    Path(args.output + ".raw").write_text(hex_out)
    save_binary(hex_out, args.output)


if __name__ == "__main__":
    main()
```

## Steps to run

Project root, venv active.

1. **Verify dependencies are present** (already in `.venv` per the
   existing setup; this just confirms):

   ```bash
   source .venv/bin/activate
   python -c "import transformers, peft, trl, datasets, torch; print('imports OK')"
   ```

2. **Confirm a formatted dataset exists**. Phase 1 is the current
   experiment target:

   ```bash
   ls dataset/processed/phase1_training_hf
   ```

   If missing:

   ```bash
   python scripts/dataset/generate_phase1_dataset.py
   python scripts/dataset/format_for_training.py \
       --input dataset/processed/phase1_dataset.jsonl \
       --output dataset/processed/phase1_training
   ```

3. **Drop in the new scripts** at the paths above, then **smoke test**
   (10 examples, 10 steps; ~1 minute on the RTX 2000 Ada):

   ```bash
   python scripts/training/train_gpt2.py \
       --test \
       --dataset dataset/processed/phase1_training_hf
   ```

   Pass criterion: training completes, checkpoint at
   `models/checkpoints/gpt2-small-lora-test/`. Loss values are noise at
   this scale — what matters is the pipeline reaches the end without
   crashing.

4. **Real Phase 1 run** (only after the smoke test passes; remember the
   context-length caveat):

   ```bash
   python scripts/training/train_gpt2.py \
       --dataset dataset/processed/phase1_training_hf \
       --run-name gpt2-small-phase1 \
       --num-train-epochs 40 \
       --learning-rate 1e-4
   ```

5. **Generate and execute**:

   ```bash
   python scripts/generation/generate_gpt2.py \
       --checkpoint models/checkpoints/gpt2-small-phase1 \
       --prompt "Write a program that returns 42" \
       --output test_42_gpt2
   qemu-riscv64 test_42_gpt2; echo $?
   ```

## Things to watch for

- `ValueError: Asking to pad but the tokenizer does not have a padding
  token` → the script must alias `tokenizer.pad_token = tokenizer.eos_token`
  before the trainer is constructed. Already in the script; only an issue
  if removed.
- `ValueError: Target modules ['q_proj', ...] not found in the base
  model` → confirm LoRA targets are `c_attn` / `c_proj` / `c_fc`, not
  the Qwen-style names.
- PEFT shape-mismatch error around the embedding layer → `<END_BINARY>`
  was added *after* `get_peft_model`. Order must be: `add_special_tokens`
  → `resize_token_embeddings` → `get_peft_model`.
- `model.print_trainable_parameters()` should report ~0.3–0.5 %
  trainable on the 124M base. Much lower means LoRA targets are
  misnamed; much higher means LoRA is also wrapping the embedding layer
  (it shouldn't).
- SFTTrainer logs how many examples were truncated against
  `max_seq_length=1024`. With the current 1696-hex dataset this will be
  ~all of them. Expected at this stage; see the context-length section
  above.
