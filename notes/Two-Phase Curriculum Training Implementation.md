# Two-Phase Curriculum Training Implementation

**Purpose**: Fix value encoding problem through curriculum learning

**Problem**: Model learned ELF structure but fails value encoding (prompt "42" → generates wrong immediate value)

**Solution**: Two-phase training - teach mechanic first, then generalize

---

## Phase 1: Teach the Mechanic

**Goal**: Learn value encoding on focused dataset

**Dataset**: ~100 examples

- 50 small values (1-31): compressed instructions, 10-byte .text
- 50 large values (≥32): uncompressed instructions, 12-byte .text

**Why both**: Footer differs (byte 96: `0a` vs `0c` for .text size)

**Training parameters**:

- Learning rate: 1e-4
- LoRA rank: 16
- Batch size: 4
- Epochs: 20-30 (~3000 steps)

**Validation gate**: Test on held-out values (15, 750) - must pass both before Phase 2

**Output**: `models/checkpoints/phase1-mechanic/`

---

## Phase 2: Generalize

**Goal**: Extend learning to full range

**Dataset**: 10,000 examples (existing returns dataset)

**Training parameters**:

- Continue from Phase 1 checkpoint
- Learning rate: 5e-5 (lower to preserve Phase 1 learning)
- Epochs: 5-7

**Output**: `models/checkpoints/phase2-generalize/`

---

## New Scripts Required

### 1. `scripts/dataset/create_phase1_subset.py`

**Purpose**: Create small focused dataset for Phase 1

**Interface**:

```bash
python scripts/dataset/create_phase1_subset.py \
    --input dataset/processed/returns_dataset.jsonl \
    --output dataset/processed/phase1_dataset.jsonl \
    --num-small 50 \
    --num-large 50 \
    --hold-out 15,750
```

**Logic**:

- Read full returns dataset
- Select values strategically:
  - Small: 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 20, 21, 23, 25, 27, 29, 30, 31, ... (avoid 15)
  - Large: 32, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000, 7000, 9000, ... (spread, avoid 750)
- Save subset as JSONL

**Parameters**:

- `--input`: Source dataset
- `--output`: Output subset
- `--num-small`: Number of small values (default: 50)
- `--num-large`: Number of large values (default: 50)
- `--hold-out`: Comma-separated values to exclude (for validation)

---

### 2. `scripts/evaluation/test_with_fixed_footer.py`

**Purpose**: Quick test of raw model output

**Interface**:

```bash
python scripts/evaluation/test_with_fixed_footer.py \
    test_output.raw
```

**Logic**:

1. Read raw hex from file
2. Strip trailing zeros
3. Find cut marker: `6d6d656e74002e72697363762e61747472696275746573`
4. Extract model's part (up to cut marker)
5. Get footer from any dataset example
6. Combine model part + dataset footer
7. Convert to binary
8. Run with QEMU
9. Print exit code

**Why any footer works**: QEMU doesn't validate section header metadata

---

### 3. `scripts/evaluation/validate_checkpoint.py`

**Purpose**: Test checkpoint on specific values (gate before Phase 2)

**Interface**:

```bash
python scripts/evaluation/validate_checkpoint.py \
    --checkpoint models/checkpoints/phase1-mechanic \
    --test-values 15 750
```

**Logic**:

1. Load checkpoint
2. For each test value:
   - Generate binary with model
   - Save raw output
   - Use test_with_fixed_footer.py logic to fix and run
   - Check exit code
3. Report PASS/FAIL for each value
4. Exit code 0 if all pass, 1 if any fail

**Parameters**:

- `--checkpoint`: Path to model checkpoint
- `--test-values`: Space-separated values to test

---

## Modifications to Existing Scripts

### `scripts/training/train_model.py`

**Add command-line arguments**:

```python
parser.add_argument("--from-checkpoint", type=str, default=None,
                    help="Continue training from existing checkpoint")
parser.add_argument("--learning-rate", type=float, default=2e-4,
                    help="Learning rate (default: 2e-4)")
parser.add_argument("--num-train-epochs", type=float, default=3.0,
                    help="Number of training epochs (default: 3.0)")
```

**Checkpoint loading logic**:

```python
if args.from_checkpoint:
    print(f"Loading from checkpoint: {args.from_checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.from_checkpoint,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    # Prepare for continued training
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
else:
    # Load base model (existing code)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='./models/base/qwen3-0.6b',
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(...)
```

**Use learning rate and epochs from args**:

```python
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,  # Use from args
    ...
)
```

---

## Complete Workflow

### Step 1: Create Phase 1 Dataset

```bash
python scripts/dataset/create_phase1_subset.py \
    --input dataset/processed/returns_dataset.jsonl \
    --output dataset/processed/phase1_dataset.jsonl \
    --num-small 50 \
    --num-large 50 \
    --hold-out 15,750
```

Output: `dataset/processed/phase1_dataset.jsonl` (100 examples)

---

### Step 2: Format Phase 1 Dataset

```bash
python scripts/dataset/format_for_training.py \
    --input dataset/processed/phase1_dataset.jsonl \
    --output dataset/processed/phase1_training
```

Output:

- `dataset/processed/phase1_training.jsonl`
- `dataset/processed/phase1_training_hf/`

---

### Step 3: Train Phase 1

```bash
./kai train \
    --dataset dataset/processed/phase1_training_hf \
    --run-name phase1-mechanic \
    --learning-rate 1e-4 \
    --num-train-epochs 25
```

**Monitor**: Training loss should decrease steadily over 25 epochs

Output: `models/checkpoints/phase1-mechanic/`

---

### Step 4: Validate Phase 1 (GATE)

```bash
python scripts/evaluation/validate_checkpoint.py \
    --checkpoint models/checkpoints/phase1-mechanic \
    --test-values 15 750
```

**Expected output**:

```
Testing value 15...
  Generated exit code: 15
  Expected exit code: 15
  ✅ PASS

Testing value 750...
  Generated exit code: 238 (750 % 256)
  Expected exit code: 238 (750 % 256)
  ✅ PASS

✅ All tests passed! Ready for Phase 2.
```

**If validation fails**: Adjust Phase 1 training (more epochs, different LR, check dataset)

---

### Step 5: Train Phase 2

```bash
./kai train \
    --from-checkpoint models/checkpoints/phase1-mechanic \
    --dataset dataset/processed/returns_training_hf \
    --run-name phase2-generalize \
    --learning-rate 5e-5 \
    --num-train-epochs 7
```

**Key**: Lower learning rate (5e-5) to preserve Phase 1 learning while generalizing

Output: `models/checkpoints/phase2-generalize/`

---

### Step 6: Evaluate Final Model

```bash
# Test on range
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/phase2-generalize \
    --range 1 100

# Test on full dataset (if evaluate_model.py supports)
python scripts/evaluation/evaluate_model.py \
    --checkpoint models/checkpoints/phase2-generalize \
    --full
```

---

## Implementation Order

1. **test_with_fixed_footer.py** - Needed by validate_checkpoint.py
2. **create_phase1_subset.py** - Dataset preparation
3. **Modify train_model.py** - Add checkpoint continuation
4. **validate_checkpoint.py** - Gating logic
5. **Run complete workflow** - Execute all steps

---

## Success Criteria

### Phase 1 Success

- **Validation gate**: Both held-out values (15, 750) return correct exit codes
- **Exact match**: Generated value == expected value
- **No hallucination**: Model generates valid RISC-V instructions

### Phase 2 Success

- **Improved accuracy**: Better than baseline across 1-10000 range
- **Generalization**: Works on values not in training set
- **Stability**: Doesn't regress on Phase 1 values

---

## Metrics to Track

**During training**:

- Training loss
- Token-level accuracy (if available)

**During evaluation**:

- **Exact match rate**: `generated_value == expected_value`
- **Close match rate**: `abs(generated_value - expected_value) <= 10`
- **Exit code accuracy**: `(generated_value % 256) == (expected_value % 256)`
- **Structure correctness**: Binary executes without errors

---

## Key Design Decisions

### Why 100 examples in Phase 1?

- Small enough to memorize mechanical pattern
- Large enough to cover compressed/uncompressed variations
- Focused signal without distribution noise

### Why lower learning rate in Phase 2?

- Preserve learned mechanic from Phase 1
- Avoid catastrophic forgetting
- Allow gradual generalization

### Why validation gate?

- Ensures Phase 1 learned the mechanic
- Prevents wasting time on Phase 2 if Phase 1 failed
- Provides checkpoint for iterating

### Why any footer works for testing?

- QEMU doesn't validate section header metadata
- Wrong .text size (10 vs 12 bytes) doesn't affect execution
- Only impacts debug tools (objdump, readelf)

---

## Troubleshooting

### Phase 1 validation fails

- **Increase epochs**: Try 30-40 instead of 25
- **Check dataset**: Verify subset was created correctly
- **Learning rate**: Try 5e-5 or 2e-4
- **Check model loading**: Verify base model loads correctly

### Phase 2 regresses

- **Lower learning rate more**: Try 1e-5 instead of 5e-5
- **Fewer epochs**: Try 3-5 instead of 7
- **Check checkpoint loading**: Verify Phase 1 weights are loaded

### Model generates invalid binaries

- **Check footer fix**: Verify test_with_fixed_footer.py works
- **Check cut marker**: Ensure model generates up to expected point
- **Inspect raw output**: Look for patterns in failures

---

## Future Improvements

**After Phase 2 works**:

- Add Phase 1.5: Mid-range dataset (100-1000 examples)
- Vary learning rate schedule (warmup, decay)
- Add regularization to prevent overfitting
- Test on out-of-distribution values (>10000)

**Longer term**:

- Extend to arithmetic operations
- Support multiple syscalls
- Generate more complex programs

---

## References

- **ELF structure**: `docs/ELF Binary Structure Guide.md`
- **Minimal binaries**: `docs/Minimal Static Binaries.md`
- **Footer investigation**: `notes/The case of long zero footers/notes.md`
- **Curriculum rationale**: `notes/# Curriculum Training Plan for Kai.md`

