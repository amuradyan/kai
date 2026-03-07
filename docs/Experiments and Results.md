# Experiments and Results

Documentation of training experiments and their outcomes.

## Experiment 1: All Non-Zeros 5x Weighting

**Branch:** `all-non-zeros-5x`
**Date:** March 2024
**Status:** In Progress

### Hypothesis
Weighting all non-zero tokens 5x will solve both the footer generation problem (stopping at 922) and the value encoding problem (wrong return values).

### Setup

#### Dataset
- **Size:** 81 Phase 1 examples
- **Small values:** 1-31 (except 15) - 30 examples with compressed instructions
- **Large values:** 32-10000 (except 750) - 51 examples with uncompressed instructions
- **Format:** Simplified (only prompt and binary_hex fields)
- **Special tokens:** `<END_BINARY>` appended to each example

#### Training Configuration
- **Model:** Qwen3-0.6B with QLoRA
- **Epochs:** 40
- **Learning rate:** 1e-4
- **Batch size:** 1 with gradient accumulation of 4
- **Total steps:** 840 (81 examples × 40 epochs / 4)
- **Custom loss:** NonZeroWeightedTrainer with 5x weight for non-zeros and END_BINARY

#### Validation
- **Hold-out values:** 15 (small), 750 (large)
- **Test method:** Generate binary, execute with QEMU, check exit code

### Implementation Details

1. **Custom Trainer Class:** `NonZeroWeightedTrainer` extends SFTTrainer
2. **Weight Calculation:**
   - Identify zero token ID from tokenizer
   - Apply 5x weight to all non-zero tokens
   - Apply 5x weight to END_BINARY token
3. **Special Token Handling:**
   - Add `<END_BINARY>` to tokenizer vocabulary
   - Resize model embeddings accordingly

### Expected Outcomes

**If successful:**
- Model generates complete 1696-character binaries (not stopping at 922)
- Correct value encoding (return values match prompts)
- Clear stopping with `<END_BINARY>` token
- Both small and large validation values work

**Potential issues:**
- Header overfitting (non-zeros in header might get memorized)
- 5x weight might be too high/low
- Need to adjust epochs or learning rate

### Results

**Status:** ✅ Completed (March 7, 2024)
**Checkpoint:** `models/checkpoints/all-non-zeros-x5`

- **Training loss:** Converged successfully over 40 epochs
- **Binary generation:**
  - Generates hex output correctly
  - Produces 4101 characters (odd number - causes fromhex error)
  - Binary works when manually padded
- **Validation results:**
  - Value 42: ✅ Works (returns 42 correctly)
  - Binary executes properly in QEMU
- **Issues:**
  - Generates odd-length hex (4101 chars)
  - Does not generate END_BINARY token
  - Continues with zeros until max token limit

### Lessons Learned

1. **Partial success:** 5x weighting preserves hex generation capability
2. **Token boundary issue:** Model stops mid-byte at token limit
3. **END_BINARY not learned:** Despite weighting, token isn't generated
4. **Footer still problematic:** Long zero sequences remain

---

## Experiment 2: Dynamic Weighting (100x Footer)

**Date:** March 7, 2024
**Status:** ❌ FAILED - Catastrophic forgetting

### Setup
- **Dataset:** Same 83 Phase 1 examples
- **Weights:**
  - Footer non-zeros: 100x
  - Regular non-zeros: 2x
  - Zeros: 1x
- **Configuration:** Batch size 1, gradient accumulation 8

### Results
**Checkpoint:** `models/checkpoints/dynamic-weighted` ⚠️ DO NOT USE

- **Complete failure:** Model forgot hex generation entirely
- **Output:** Generated Python code instead: `print("42")`
- **Task confusion:** Switched from binary generation to source code

### Lessons Learned
1. **100x too extreme:** Causes catastrophic forgetting
2. **Small dataset can't handle:** 83 examples insufficient for weighted training
3. **Mode collapse:** Model reverts to pre-training knowledge

---

## Experiment 3: Progressive Weighting (1x→5x)

**Date:** March 7, 2024
**Status:** ❌ FAILED - Worse than Experiment 2

### Setup
- **Dataset:** Same 83 Phase 1 examples
- **Weights:** Non-zeros progressively increase from 1x (start) to 5x (end)
- **Configuration:** Same as Experiment 2

### Results
**Checkpoint:** `models/checkpoints/progressive-weighted` ⚠️ DO NOT USE

- **Catastrophic failure with repetition:**
  ```python
  print("42")
  ```
  Then 272 repetitions of:
  ```
  ### Steps:
  1. Create a variable
  2. Print the variable
  ```
- **Worse than 100x:** Entered infinite repetitive loop
- **Zero hex output:** Complete loss of binary generation

### Lessons Learned
1. **Progressive doesn't help:** Still causes catastrophic forgetting
2. **New failure mode:** Repetitive collapse (272x same text)
3. **Dataset critically small:** 83 examples cannot support ANY weighted training

---

## Key Insights and Critical Findings

### Dataset Size is Critical
1. **83 examples is dangerously small** for weighted training approaches
2. **Weighted training requires larger datasets:** Need 10,000+ examples for stability
3. **Small datasets + weights = catastrophic forgetting:** Model loses core task

### Weighting Experiment Results
1. **Uniform 5x:** Partial success - generates hex but odd-length (4101 chars)
2. **Dynamic 100x:** Complete failure - generates Python code
3. **Progressive 1x→5x:** Worst failure - repetitive collapse (272x loop)

### Technical Issues Discovered
1. **Token boundary misalignment:** Model generates 4101 hex chars (odd number)
2. **END_BINARY token not learned:** Despite weighting, never generated
3. **Footer problem persists:** Long zero sequences remain problematic
4. **Gradient instability:** Non-uniform weights cause training collapse

### Failure Modes Observed
1. **Task amnesia:** Model forgets hex generation, reverts to Python
2. **Repetitive collapse:** New failure mode - infinite loops of same text
3. **Mode reversion:** Model falls back to pre-training knowledge

### Recommendations Moving Forward
1. **Use full 10,000 example dataset** before attempting weighted training
2. **Consider RLE encoding** instead of weighting for footer problem
3. **Fix token alignment issue** to avoid odd-length hex
4. **Uniform weights safer** for small datasets
5. **Monitor for catastrophic forgetting** early in training