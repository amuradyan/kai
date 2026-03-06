# Curriculum Training Plan for Kai

## Context

Model: Qwen3-0.6B with QLoRA
Task: Generate RISC-V ELF64 binaries from natural language prompts ("Write a program that returns N")
Problem: First training run (10K examples, 3 epochs) learned ELF structure correctly but failed value encoding — generated 145 instead of 42.

## Root Cause

The mapping from prompt text ("42") to RISC-V immediate encoding (`02a00513`) is a precise arithmetic transformation buried in 1696 hex chars of mostly-static boilerplate. The signal is too weak for the model to pick up in 3 epochs. More data alone won't fix it.

## Solution: Two-Phase Curriculum Training

### Phase 1 — Teach the encoding mechanic

**Dataset:** ~100 examples total

- 50 small values (1–31): these use RVC compressed `li` (2-byte instruction, 10-byte .text)
- 50 large values (≥32): spread across range (e.g. 32, 100, 500, 1000, 5000, 9000...) — uncompressed `li` (4-byte instruction, 12-byte .text)

Cover both because small and large produce structurally different footers (`.text` size field differs by 1 byte at footer offset 96: `0a` vs `0c`). Model needs to see both.

**Parameters:**

- Epochs: 20–30
- Learning rate: 1e-4
- LoRA rank: 16 (same as before)
- Batch size: 4

**Gate before Phase 2:** Test on held-out values not in phase 1 training set — one small (e.g. 15), one large (e.g. 750). Model must get both correct (or very close) before proceeding.

### Phase 2 — Generalize across full range

Start from Phase 1 checkpoint, continue training on full dataset.

**Dataset:** 10,000 examples (existing, values 1–10000)

**Parameters:**

- Epochs: 5–7
- Learning rate: 5e-5 (lower to avoid overwriting phase 1 learning)
- LoRA rank: 16

## Also Needed: Footer Fix Script

A script does not yet exist that does the full pipeline:

1. Take raw model output hex
2. Strip trailing zeros
3. Graft footer from a dataset reference binary
4. Convert to binary
5. Run through QEMU and report exit code

Existing scripts (`compare_output.py`, `evaluate_model.py`, `hex_to_binary.py`) each do part of this but not the full pipeline. This is needed for evaluation during and after training.

## What NOT to Do

- Do not add assembly as an intermediate step in training data (by design decision)
- Do not pre-segment hex into header/code/footer fields in output (output stays as one blob)
- Do not rely on more data alone without curriculum structure
