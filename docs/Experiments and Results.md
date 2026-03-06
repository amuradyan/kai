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

[Training in progress - to be updated]

- **Training loss:**
- **Validation results:**
  - Value 15:
  - Value 750:
- **Binary completeness:**
- **END_BINARY generation:**

### Lessons Learned

[To be documented after experiment completes]

---

## Experiment 2: [Next Experiment]

*To be designed based on Experiment 1 results*

### Potential Directions
- Adjust weight multiplier (3x or 10x instead of 5x)
- Try position-aware weighting (different weights for header/code/footer)
- Implement RLE encoding for footer
- Test different learning rates
- Extend to Phase 2 training on full dataset

---

## Key Insights So Far

1. **Footer problem root cause:** Low entropy regions (90% zeros) get ignored during training
2. **Value encoding challenge:** Weak signal for specific bytes in sea of static structure
3. **Early stopping:** Model learns position 922 as valid endpoint without explicit stop token
4. **Simple solutions work:** Uniform weighting may be as effective as complex position-aware schemes
5. **Curriculum is critical:** Strategic value selection better than random sampling