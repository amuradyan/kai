# Troubleshooting

Common issues and solutions for the Kai project.

## Training Issues

### TypeError: compute_loss() got unexpected keyword argument 'num_items_in_batch'
**Problem:** Custom loss function doesn't accept all SFTTrainer parameters.

**Solution:** Add `**kwargs` to the compute_loss method signature:
```python
def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    # Your loss implementation
```

### Model stops generating at position 922
**Problem:** Model treats the end of `.riscv.attributes` section as a stopping point, generating only 922 characters instead of the full 1696.

**Solution:**
1. Use `<END_BINARY>` token to mark actual end of binary
2. Apply weighted loss to emphasize footer non-zeros
3. Set `early_stopping=False` in generation config

### Low loss but wrong outputs
**Problem:** Training loss decreases but model generates incorrect values.

**Solution:** The model may be learning structure but not values. Use weighted loss to emphasize non-zero bytes which carry the actual information.

## Dataset Issues

### Dataset has unnecessary fields
**Problem:** Old dataset format includes unused fields (assembly, id, category, source_code, binary_size).

**Solution:** Use `generate_phase1_dataset.py` which generates clean format with only `prompt` and `binary_hex` fields.

### Dataset missing END_BINARY token
**Problem:** Generated dataset doesn't have `<END_BINARY>` markers.

**Solution:** Regenerate using the updated scripts:
```bash
python scripts/dataset/generate_phase1_dataset.py  # For Phase 1
# or
python scripts/dataset/generate_returns_dataset.py  # For full dataset
```

### Phase 1 subset takes too long
**Problem:** Generating full 10K dataset just to select 100 examples is inefficient.

**Solution:** Use `generate_phase1_dataset.py` which directly generates the 81 strategic examples needed.

## Generation Issues

### No END_BINARY token in output
**Problem:** Model doesn't generate the stop token.

**Solution:**
1. Verify dataset includes `<END_BINARY>`
2. Check model was trained with weighted loss
3. Ensure special token was added during training

### Wrong value encoding (e.g., 145 instead of 42)
**Problem:** Model generates structurally correct binary but with wrong return value.

**Solution:**
1. Use weighted loss to emphasize non-zero value bytes
2. Train with curriculum approach (Phase 1 focused examples)
3. Ensure sufficient epochs for small dataset

### Generation fills with zeros after position 922
**Problem:** Model generates meaningful content until position 922, then only zeros.

**Solution:** Model learned this as valid stopping point. Retrain with:
1. END_BINARY token for explicit stop signal
2. Weighted loss for footer importance
3. Dataset that includes complete binaries

## QEMU Execution Issues

### QEMU not found
**Problem:** `qemu-riscv64` command not available.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt install qemu-user

# Or use Nix
nix-shell -p qemu
```

### Binary executes but wrong exit code
**Problem:** Generated binary runs but returns wrong value.

**Solution:** This is a value encoding issue. The model needs more training with weighted loss on the value-encoding bytes.

## Memory Issues

### CUDA out of memory
**Problem:** Training fails with OOM error.

**Solution:**
1. Reduce batch size (already at 1)
2. Increase gradient accumulation steps
3. Use smaller max_seq_length
4. Ensure 4-bit quantization is working

### Embedding size mismatch
**Problem:** Warning about embedding layer being resized.

**Solution:** This is expected when adding special tokens. The warning is harmless and the system handles it automatically.

## Common Mistakes

1. **Forgetting to regenerate dataset** after modifying generation scripts
2. **Not formatting dataset** before training (need to run format_for_training.py)
3. **Using wrong dataset path** (check for `_hf` suffix for HuggingFace format)
4. **Training without weighted loss** for footer/value problems
5. **Not adding special tokens** when loading model for generation