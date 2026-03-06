# Tools and Scripts Guide

This guide explains all scripts in the project, organized by workflow stage.

---

## Quick Reference

```bash
# Complete workflow
python scripts/setup/download_base_model.py                    # 1. One-time setup
python scripts/dataset/generate_returns_dataset.py             # 2. Generate dataset
python scripts/dataset/format_for_training.py                  # 3. Format for training
./kai train --dataset dataset/processed/returns_training_hf    # 4. Train model
./kai generate --checkpoint models/... --prompt "..."          # 5. Generate binary
python scripts/evaluation/evaluate_model.py                    # 6. Evaluate results
```

---

## 1. Setup (One-Time)

### `scripts/setup/download_base_model.py`

**Purpose:** Download the Qwen3-0.6B base model from HuggingFace.

**What it does:** Downloads and caches the pre-trained model to `models/base/qwen3-0.6b/`.

**Usage:**

```bash
python scripts/setup/download_base_model.py
```

**When to use:** First time setting up the project, or when base model is missing.

**Output:** ~1.5GB model files in `models/base/qwen3-0.6b/`

---

## 2. Dataset Generation

### `scripts/dataset/generate_returns_dataset.py`

**Purpose:** Generate the training dataset of minimal RISC-V binaries.

**What it does:**

- Compiles 10,000 minimal static binaries (returns 1-10000)
- Uses direct syscalls, no libc (~500 bytes per binary)
- Saves to `dataset/processed/returns_dataset.jsonl`

**Usage:**

```bash
python scripts/dataset/generate_returns_dataset.py
```

**When to use:**

- Creating the dataset for the first time
- Regenerating after changing compilation flags
- Expanding to different value ranges

**Output:**

- `returns_dataset.jsonl` (~21MB, 10K examples)
- Each example contains: prompt, binary_hex

**Time:** ~10 minutes for 10K examples

---

### `scripts/dataset/generate_phase1_dataset.py`

**Purpose:** Generate strategic Phase 1 curriculum dataset directly.

**What it does:**

- Generates 81 carefully selected examples
- 30 small values (1-31, except 15): compressed RISC-V instructions
- 51 large values (32-10000, except 750): uncompressed instructions
- Automatically includes `<END_BINARY>` token
- Takes ~2 minutes (vs 10 for full dataset)

**Usage:**

```bash
python scripts/dataset/generate_phase1_dataset.py
```

**When to use:**

- Curriculum training experiments
- Quick training iterations
- Testing new loss functions
- Value encoding experiments

**Output:** `phase1_dataset.jsonl` with END_BINARY tokens

---

### `scripts/dataset/create_phase1_subset.py`

**Purpose:** Create focused dataset for Phase 1 curriculum training.

**What it does:**

- Selects 100 strategic examples from full dataset
- 50 small values (1-31): Use compressed RISC-V instructions
- 50 large values (≥32): Use uncompressed instructions
- Holds out specific values for validation (default: 15, 750)
- Ensures coverage of both instruction types

**Usage:**

```bash
# Default: 50 small + 50 large, hold out 15 and 750
python scripts/dataset/create_phase1_subset.py

# Custom parameters
python scripts/dataset/create_phase1_subset.py \
    --input dataset/processed/returns_dataset.jsonl \
    --output dataset/processed/phase1_dataset.jsonl \
    --num-small 50 \
    --num-large 50 \
    --hold-out 15,750
```

**When to use:**

- Setting up Phase 1 of curriculum training
- Teaching model the value encoding mechanic
- Creating focused dataset for difficult-to-learn patterns

**Output:** `phase1_dataset.jsonl` (100 examples)

---

### `scripts/dataset/format_for_training.py`

**Purpose:** Convert JSONL dataset to HuggingFace format for training.

**What it does:**

- Reads JSONL dataset
- Formats as instruction-tuning pairs (prompt → binary_hex)
- Saves in HuggingFace Dataset format (memory-mapped, efficient)

**Usage:**

```bash
# Default (returns dataset)
python scripts/dataset/format_for_training.py

# Custom input/output
python scripts/dataset/format_for_training.py \
    --input dataset/processed/phase1_dataset.jsonl \
    --output dataset/processed/phase1_training
```

**When to use:** After generating or modifying the dataset, before training.

**Output:**

- `returns_training.jsonl` - Human-readable format
- `returns_training_hf/` - HuggingFace Dataset (Arrow format)

---

### `scripts/dataset/verify_returns_dataset.py`

**Purpose:** Validate that generated binaries execute correctly.

**What it does:**

- Tests binaries with QEMU user-mode emulation
- Verifies exit codes match expected values
- Extracts expected value directly from prompt
- Works with simplified dataset format (prompt + binary_hex only)

**Usage:**

```bash
# Test 100 evenly-spaced samples (default)
python scripts/dataset/verify_returns_dataset.py

# Test specific number
python scripts/dataset/verify_returns_dataset.py -n 50

# Test all 10,000 binaries
python scripts/dataset/verify_returns_dataset.py -n 0

# Test custom dataset
python scripts/dataset/verify_returns_dataset.py --dataset dataset/processed/phase1_dataset.jsonl
```

**When to use:**

- After generating dataset (quality check)
- Debugging binary generation issues
- Verifying QEMU setup
- Testing Phase 1 curriculum dataset

**Output:** Pass/fail statistics with per-test results

---

## 3. Training

### `scripts/training/train_model.py`

**Purpose:** Fine-tune Qwen3-0.6B to generate RISC-V binaries.

**What it does:**

- Loads base model with QLoRA (4-bit quantization)
- Trains on instruction → binary_hex pairs
- Supports custom weighted loss via `NonZeroWeightedTrainer`
- Automatically adds `<END_BINARY>` special token
- Saves checkpoints to `models/checkpoints/`

**Weighted Loss Training:**

The script includes `NonZeroWeightedTrainer` class that:
- Weights all non-zero hex tokens 5x
- Weights `<END_BINARY>` token 5x
- Helps model focus on information-carrying bytes
- Solves footer generation and value encoding problems

**Usage:**

```bash
# Full training (3 epochs, 10K examples)
./kai train --dataset dataset/processed/returns_training_hf

# Test mode (10 examples, 10 steps)
./kai train --test

# Custom run name (avoid overwriting)
./kai train --run-name experiment-v2 --dataset dataset/processed/returns_training_hf

# Custom hyperparameters
./kai train --max-steps 1000 --num-examples 5000
```

**When to use:**

- After preparing dataset
- Experimenting with different hyperparameters
- Training from scratch or continuing from checkpoint

**Flags:**

- `--test` - Quick test run (10 examples, 10 steps)
- `--dataset PATH` - Dataset directory (HuggingFace format)
- `--run-name NAME` - Custom checkpoint name
- `--max-steps N` - Override epoch-based training
- `--num-examples N` - Limit training examples
- `--from-checkpoint PATH` - Continue training from existing checkpoint
- `--learning-rate FLOAT` - Set learning rate (default: 2e-4)
- `--num-train-epochs FLOAT` - Number of epochs (default: 3.0)

**Output:** Model checkpoint in `models/checkpoints/{run-name}/`

**Time:** ~3 hours for full training (GPU-dependent)

**Resources:** ~3.2GB VRAM peak with 4-bit quantization

---

## 4. Generation (Inference)

### `scripts/generation/generate_binary.py`

**Purpose:** Generate RISC-V binaries from natural language prompts.

**What it does:**

- Loads trained model checkpoint
- Generates hex string from prompt
- Converts to executable binary
- Saves output files

**Usage:**

```bash
# Generate binary
./kai generate \
    --checkpoint models/checkpoints/qwen3-0.6b-lora \
    --prompt "Write a program that returns 42" \
    --output my_binary

# Show generated hex
./kai generate --prompt "..." --show-hex

# Custom max tokens
./kai generate --prompt "..." --max-tokens 2048
```

**When to use:**

- Testing trained model
- Generating binaries for evaluation
- Interactive experimentation

**Flags:**

- `--checkpoint PATH` - Model checkpoint directory
- `--prompt TEXT` - Natural language instruction
- `--output PATH` - Output binary file
- `--show-hex` - Print generated hex string
- `--max-tokens N` - Generation length limit (default: 4096)

**Output:**

- Binary file (executable)
- `.raw` file with hex string

**Time:** ~10-30 seconds per generation

---

## 5. Evaluation & Analysis

### `scripts/evaluation/test_with_fixed_footer.py`

**Purpose:** Test raw model output by fixing footer and running with QEMU.

**What it does:**

- Takes raw model output (incomplete hex)
- Strips trailing zeros
- Finds cut marker position
- Grafts footer from dataset reference
- Converts to binary and runs with QEMU
- Reports exit code

**Usage:**

```bash
python scripts/evaluation/test_with_fixed_footer.py test_output.raw
```

**When to use:**

- Quick testing during training
- Model generates incomplete output (stops early)
- Debugging value encoding issues
- Testing without full binary generation

**Output:** Exit code from QEMU execution

---

### `scripts/evaluation/validate_checkpoint.py`

**Purpose:** Validate checkpoint on specific values - gate between Phase 1 and Phase 2.

**What it does:**

- Loads model checkpoint
- Generates binaries for test values
- Fixes incomplete output with dataset footer
- Runs with QEMU and checks exit codes
- Reports PASS/FAIL for each value
- Returns exit code 0 if all pass, 1 if any fail

**Usage:**

```bash
# Default: test on values 15 and 750
python scripts/evaluation/validate_checkpoint.py \
    --checkpoint models/checkpoints/phase1-mechanic

# Custom test values
python scripts/evaluation/validate_checkpoint.py \
    --checkpoint models/checkpoints/phase1-mechanic \
    --test-values 15 750 1000
```

**When to use:**

- After Phase 1 training (curriculum learning)
- Before proceeding to Phase 2
- Validating model learned specific mechanic
- Quality gate in training pipeline

**Output:**

- Per-value PASS/FAIL results
- Summary of validation
- Exit code for scripting (0 = success)

---

### `scripts/evaluation/evaluate_model.py`

**Purpose:** Systematic evaluation of model across value ranges.

**What it does:**

- Tests model on multiple prompts
- Executes binaries with QEMU
- Compares actual vs expected exit codes
- Generates evaluation report

**Usage:**

```bash
# Evaluate on specific values
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/qwen3-0.6b-lora \
    --values 1 10 42 100 1000

# Evaluate on range
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/qwen3-0.6b-lora \
    --range 1 100

# Full evaluation suite
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/qwen3-0.6b-lora \
    --full
```

**When to use:**

- After training completes
- Comparing different checkpoints
- Understanding model behavior across value ranges

**Output:**

- Success/failure statistics
- Value distribution analysis
- Error patterns

---

### `scripts/evaluation/compare_output.py`

**Purpose:** Compare generated binary against expected dataset binary.

**What it does:**

- Loads generated hex and expected hex
- Finds first divergence point
- Shows byte-level differences
- Annotates with ELF structure context

**Usage:**

```bash
python scripts/evaluation/compare_output.py \
    generated.raw \
    expected.raw
```

**When to use:**

- Debugging generation failures
- Understanding where model makes mistakes
- Analyzing partial correctness

**Output:**

- Divergence position and context
- Hex dump of differences
- Structure annotations (which ELF section)

---

### `scripts/evaluation/analyze_elf_structure.py`

**Purpose:** Annotate binary hex with ELF structure information.

**What it does:**

- Parses ELF binary
- Annotates each section with labels
- Shows headers, segments, code, metadata
- Helps understand binary layout

**Usage:**

```bash
python scripts/evaluation/analyze_elf_structure.py binary.elf
```

**When to use:**

- Understanding ELF format
- Debugging binary structure issues
- Documenting binary layout

**Output:** Annotated hex dump with section labels

---

### `scripts/evaluation/analyze_failures.py`

**Purpose:** Analyze patterns in generation failures.

**What it does:**

- Collects multiple failed generations
- Identifies common failure modes
- Shows statistics on error types
- Suggests potential fixes

**Usage:**

```bash
python scripts/evaluation/analyze_failures.py \
    --checkpoint models/checkpoints/qwen3-0.6b-lora \
    --num-samples 100
```

**When to use:**

- Understanding systematic failures
- Identifying dataset or training issues
- Prioritizing improvements

**Output:** Failure mode analysis, common patterns, recommendations

---

## 6. Utilities

### `scripts/utils/hex_to_binary.py`

**Purpose:** Convert hex string files to executable binaries.

**What it does:**

- Reads hex string from file
- Validates hex format
- Converts to binary bytes
- Makes executable (chmod +x)

**Usage:**

```bash
# Auto-detect output name (strips .raw/.hex)
python scripts/utils/hex_to_binary.py input.raw

# Custom output name
python scripts/utils/hex_to_binary.py input.raw output_binary

# Example: Reproduce experiment results
python scripts/utils/hex_to_binary.py \
    "notes/When the tail of the binary was all wrong/test_output_fixed.raw"
qemu-riscv64 "notes/When the tail of the binary was all wrong/test_output_fixed"
echo $?  # Check exit code
```

**When to use:**

- Reproducing experiment results
- Testing hex strings manually
- Converting raw model outputs to executables

**Output:** Executable binary file

---

## 7. CLI Wrapper

### `kai`

**Purpose:** Unified command-line interface for common operations.

**What it does:**

- Wrapper around training and generation scripts
- Passes through all arguments
- Simplifies common workflows

**Usage:**

```bash
# Training
./kai train [args...]
  → python scripts/training/train_model.py [args...]

# Generation
./kai generate [args...]
  → python scripts/generation/generate_binary.py [args...]
```

**When to use:** Prefer this over calling scripts directly (cleaner, shorter).

---

## Workflow Examples

### Full Pipeline: Dataset → Training → Evaluation

```bash
# 1. Generate dataset
python scripts/dataset/generate_returns_dataset.py

# 2. Validate dataset
python scripts/dataset/verify_returns_dataset.py -n 0  # Test all

# 3. Format for training
python scripts/dataset/format_for_training.py \
    --input dataset/processed/returns_dataset.jsonl \
    --output dataset/processed/returns_training

# 4. Train model
./kai train \
    --dataset dataset/processed/returns_training_hf \
    --run-name returns-v1

# 5. Generate test binary
./kai generate \
    --checkpoint models/checkpoints/returns-v1 \
    --prompt "Write a program that returns 42" \
    --output test_42

# 6. Test execution
qemu-riscv64 test_42
echo $?

# 7. Full evaluation
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/returns-v1 \
    --range 1 100
```

---

### Quick Test: Try a Single Value

```bash
# Generate
./kai generate \
    --checkpoint models/checkpoints/qwen3-0.6b-lora \
    --prompt "Write a program that returns 7" \
    --output test_7 \
    --show-hex

# Test
qemu-riscv64 test_7
echo "Exit code: $?"

# Compare to expected
python scripts/evaluation/compare_output.py \
    test_7.raw \
    <(head -7 dataset/processed/returns_dataset.jsonl | tail -1 | jq -r .binary_hex)
```

---

### Debugging: Why Did Generation Fail?

```bash
# 1. Generate with hex output
./kai generate --prompt "..." --output failed --show-hex > failed.log

# 2. Compare to dataset example
python scripts/evaluation/compare_output.py \
    failed.raw \
    dataset_example.raw

# 3. Analyze ELF structure
python scripts/evaluation/analyze_elf_structure.py failed

# 4. Try to fix by combining with dataset ending
python << EOF
model_hex = open('failed.raw').read().strip()
expected_hex = open('dataset_example.raw').read().strip()
cut_point = model_hex.find('6d6d656e74002e72697363762e61747472696275746573')
if cut_point > 0:
    combined = model_hex[:cut_point+len('6d6d656e74002e72697363762e61747472696275746573')] + expected_hex[cut_point+len('6d6d656e74002e72697363762e61747472696275746573'):]
    open('fixed.raw', 'w').write(combined)
EOF

python scripts/utils/hex_to_binary.py fixed.raw
qemu-riscv64 fixed
echo "Fixed exit code: $?"
```

---

### Two-Phase Curriculum Training

Train the model using curriculum learning to solve value encoding problems:

```bash
# Phase 1: Teach the Mechanic (100 examples, focused learning)

# 1. Create Phase 1 dataset
python scripts/dataset/create_phase1_subset.py \
    --input dataset/processed/returns_dataset.jsonl \
    --output dataset/processed/phase1_dataset.jsonl \
    --num-small 50 \
    --num-large 50 \
    --hold-out 15,750

# 2. Format Phase 1 dataset
python scripts/dataset/format_for_training.py \
    --input dataset/processed/phase1_dataset.jsonl \
    --output dataset/processed/phase1_training

# 3. Train Phase 1 (high learning rate, many epochs)
./kai train \
    --dataset dataset/processed/phase1_training_hf \
    --run-name phase1-mechanic \
    --learning-rate 1e-4 \
    --num-train-epochs 25

# 4. Validate Phase 1 (GATE - must pass before Phase 2)
python scripts/evaluation/validate_checkpoint.py \
    --checkpoint models/checkpoints/phase1-mechanic \
    --test-values 15 750

# If validation fails, adjust Phase 1 training (more epochs, different LR)
# Only proceed to Phase 2 after validation passes

# Phase 2: Generalize (10K examples, lower learning rate)

# 5. Train Phase 2 from Phase 1 checkpoint
./kai train \
    --from-checkpoint models/checkpoints/phase1-mechanic \
    --dataset dataset/processed/returns_training_hf \
    --run-name phase2-generalize \
    --learning-rate 5e-5 \
    --num-train-epochs 7

# 6. Evaluate final model
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/phase2-generalize \
    --range 1 100
```

**Why Curriculum Training?**

- Phase 1: Teaches precise value encoding on small dataset (100 examples)
- Phase 2: Generalizes learning to full range (10K examples)
- Solves problem where model learns structure but fails values
- Gate ensures mechanic is learned before generalizing
