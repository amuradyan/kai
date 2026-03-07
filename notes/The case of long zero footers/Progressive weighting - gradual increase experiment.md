# Progressive Weighting - Gradual Increase Experiment

## Date: 2026-03-07

## Experiment Setup
- **Goal**: Apply gradually increasing weights to teach footer patterns without catastrophic forgetting
- **Weight Formula**:
  - Non-zeros: 1x (start) → 5x (end of sequence)
  - Weight = 1 + (position_ratio × 4)
  - Zeros: Always 1x
  - END_BINARY token: 5x

## Training Configuration
- Model: Qwen3-0.6B with LoRA
- Dataset: phase1_training_hf (83 examples only!)
- Batch size: 1
- Gradient accumulation: 8 steps
- Total training steps: 33 (3 epochs)
- Effective batch size: 8

## Results: COMPLETE FAILURE WITH REPETITIVE COLLAPSE

### What Happened
The model not only forgot hex generation, but entered a **repetitive loop**:

```
Input: "Write a program that returns 42"
Output:
```python
print("42")
```
### Steps:
1. Create a variable
2. Print the variable

[REPEATED 272 TIMES]
```

### Statistics
- **Total output**: 1,092 lines
- **Unique pattern**: Same 3-line block repeated 272 times
- **Hex bytes generated**: 0
- **Python code**: 3 lines
- **Repetitive text**: 1,089 lines

### Analysis

1. **Task Amnesia**: Complete loss of hex generation capability
2. **Mode Collapse**: Stuck in infinite loop generating same text
3. **Wrong Domain**: Generating Python + English instructions
4. **Nonsensical Content**: Instructions don't match the code
5. **No Binary Output**: Zero hex characters generated

### Why Progressive Weighting Failed

Despite being more moderate (1x→5x) than the 100x experiment:

1. **Dataset Too Small**:
   - 83 examples insufficient for weighted training stability
   - Model overfits to weight pattern, not task

2. **Gradient Flow Issues**:
   - Even gradual increase disrupts learning
   - Late-sequence high weights interfere with early learning

3. **Catastrophic Forgetting**:
   - Model reverts to pre-training (Python/English)
   - Loses all fine-tuning for hex generation

## Comparison with Previous Experiments

| Experiment | Weights | Result |
|------------|---------|--------|
| Uniform 5x | All non-zeros: 5x | Hex generated (4101 chars, odd) |
| Dynamic 100x | Footer: 100x, Regular: 2x | Python code only |
| Progressive | 1x→5x gradual | Python + 272x repetition |

## Key Insights

1. **Any non-uniform weighting breaks the model** with this small dataset
2. **Repetitive collapse** is a new failure mode (worse than before)
3. **83 examples is critically insufficient** for weighted training

## Lessons Learned

1. **Small datasets can't handle weighted training**
   - Need at least the 10,000 example dataset
   - Or need uniform weights only

2. **Progressive weighting isn't safer**
   - Still causes catastrophic forgetting
   - Actually worse (repetitive loops) than binary weighting

3. **Model degradation patterns**:
   - First loses hex generation
   - Then loses coherent generation
   - Finally enters repetitive loops

## Recommendations

### Immediate Actions
1. **Stop weighted training on 83-example dataset**
2. **Use full 10,000 example dataset** for any weighted approaches
3. **Consider uniform weights only** for small datasets

### Alternative Approaches
1. **RLE Encoding**: Compress zeros in training data
2. **Curriculum Learning**: Start simple, increase complexity
3. **Token Alignment**: Fix the 4101-character issue directly
4. **Stopping Tokens**: Focus on END_BINARY generation only

## Next Steps
1. Convert 10,000 example dataset to HF format
2. Retry weighted training with adequate data
3. Or pivot to RLE encoding approach
4. Or fix token boundary alignment issue

## Failed Checkpoint
Location: `models/checkpoints/progressive-weighted`
Status: **DO NOT USE** - produces repetitive Python code