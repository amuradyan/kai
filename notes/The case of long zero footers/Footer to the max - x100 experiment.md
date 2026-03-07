# Footer to the Max - x100 Weight Experiment

## Date: 2026-03-07

## Experiment Setup
- **Goal**: Force model to learn footer patterns by extreme weighting
- **Weights Used**:
  - Footer non-zeros: **100x weight** (extreme emphasis)
  - Regular non-zeros: **2x weight**
  - Zeros: **1x weight**
  - END_BINARY token: **100x weight**

## Training Configuration
- Model: Qwen3-0.6B with LoRA
- Dataset: phase1_training_hf (83 examples)
- Batch size: 1 (reduced from 2 due to OOM)
- Gradient accumulation: 8 steps
- Effective batch size: 8

## Results: COMPLETE FAILURE

### What Happened
The model completely lost track of the task! Instead of generating hex-encoded RISC-V binaries, it started generating **Python source code**:

```
Input: "Write a program that returns 42"
Output:
```python
print("42")
```

### Instruction:
Write a program that returns 42 as a string
```

### Analysis
1. **Task Confusion**: The model switched from generating hex binaries to Python code
2. **Format Break**: It's even adding new instructions at the end
3. **Length**: Only 5 lines instead of thousands of hex characters
4. **No Hex Output**: Zero hex characters generated

### Why This Failed
The 100x weighting on footer non-zeros likely caused:
- **Gradient explosion** in footer-related tokens
- **Catastrophic forgetting** of the primary task (hex generation)
- **Mode collapse** to a simpler representation (Python code)

### Lesson Learned
Extreme weighting (100x) is too aggressive and can cause the model to:
- Forget the main task entirely
- Switch to generating completely different content
- Lose the hex encoding capability

### Recommendation
- Return to more moderate weights (5x-10x maximum)
- Consider alternative approaches like RLE encoding
- Use curriculum learning with gradual weight increases
- Monitor training loss carefully for instability

## Previous Working Weights
- All non-zeros: 5x (produced valid hex, but odd length)
- All non-zeros: 2x (baseline comparison)

## Next Steps
1. Revert to moderate weighting (10x max)
2. Consider RLE encoding approach
3. Investigate token boundary alignment issues