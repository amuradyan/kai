# Triton + NixOS Compilation Issue

## Status: RESOLVED ✅

Training now works! The issue was successfully resolved by providing proper Python headers and CUDA tools to Triton.

## Root Causes

Triton (used by Unsloth for fast CUDA kernels) failed to compile on NixOS due to two issues:

### Issue 1: Python.h Not Found

Triton compiles `cuda_utils.c` with gcc during first run. Compilation failed because:

1. gcc looked for Python.h at `-I/run/current-system/sw/include/python3.12`
2. This path doesn't exist on NixOS
3. Python.h actually exists at `/nix/store/*-python3-3.12.*/include/python3.12/Python.h`
4. NixOS stores packages in `/nix/store`, not standard system paths

### Issue 2: ptxas Not Found

After fixing Python.h, Triton couldn't find `ptxas` (NVIDIA's PTX assembler):

1. Triton looks for `ptxas` using its own mechanism, not just PATH
2. Even though `ptxas` was in PATH, Triton's knobs system couldn't locate it
3. Error: `RuntimeError: Cannot find ptxas`

## What We Tried

1. ✅ `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib` - helped Triton find libcuda.so.1
2. ❌ `C_INCLUDE_PATH` alone - didn't help gcc find Python.h (wrong Python package)
3. ❌ Subprocess patching - would break other things
4. ❌ Various environment variables - incomplete without proper packages

## Solution That Worked

### 1. Use python312Full in shell.nix

Changed from `python312` to `python312Full` which includes development headers:

```nix
buildInputs = [
  python312Full  # Changed from python312
  # ... other packages
];
```

### 2. Add Python Headers to C_INCLUDE_PATH in .envrc

```bash
PYTHON_INCLUDE=$(echo /nix/store/*-python3-3.12.*/include/python3.12 | cut -d' ' -f1)
export C_INCLUDE_PATH="$PYTHON_INCLUDE:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE:${CPLUS_INCLUDE_PATH:-}"
```

### 3. Add CUDA bin to PATH in .envrc

```bash
CUDA_BIN=$(echo /nix/store/*-cuda-merged-*/bin | cut -d' ' -f1)
export PATH="$CUDA_BIN:$PATH"
```

### 4. Set TRITON_PTXAS_PATH in .envrc

This tells Triton exactly where to find ptxas:

```bash
export TRITON_PTXAS_PATH="$CUDA_BIN/ptxas"
```

## Final Working Configuration

### shell.nix
- `python312Full` instead of `python312`
- `cudaPackages.cudatoolkit` for CUDA tools
- Environment variables for library paths

### .envrc
- `C_INCLUDE_PATH` and `CPLUS_INCLUDE_PATH` pointing to Python headers
- `TRITON_LIBCUDA_PATH=/run/opengl-driver/lib`
- `TRITON_PTXAS_PATH` pointing to ptxas binary
- `PATH` including CUDA bin directory

## Test Results

```
✅ Training complete!
- 10 steps completed in ~10 seconds
- Loss decreased: 0.701 → 0.509
- Model saved to ./models/checkpoints/qwen3-0.6b-lora-test
- VRAM usage: 0.60 GB (peak 3.20 GB)
```

## Key Learnings

1. **NixOS requires explicit paths**: Standard system paths don't exist
2. **python312Full vs python312**: Only `Full` variant includes dev headers
3. **Triton needs explicit configuration**: Environment variables must point to exact paths
4. **TRITON_PTXAS_PATH is critical**: Triton doesn't use PATH alone to find CUDA tools
5. **direnv + nix-shell**: Both must export the same environment variables

## Running Training

```bash
# Test mode (10 examples, 10 steps)
python scripts/train.py --test

# Full training (2500 examples, 3 epochs)
python scripts/train.py
```

Environment is automatically loaded via direnv when entering the directory.
