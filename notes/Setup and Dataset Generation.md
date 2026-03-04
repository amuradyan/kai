# Kai: Binary Generation from Prompts - Technical Log

**Project Goal:** Train Qwen3-0.6B to generate RISC-V binaries from natural language prompts.

**Date:** March 4, 2026

---

## System Specifications

### Hardware

- **CPU:** Intel i9-13980HX (32 cores @ 5.4GHz)
- **GPU:** NVIDIA RTX 2000 Ada Generation Laptop GPU (8GB VRAM)
- **CUDA:** Version 12.7
- **RAM:** 128GB
- **Storage:** 163GB available
- **OS:** NixOS 24.11

### Why This Hardware Matters

- 8GB VRAM: Perfect for QLoRA training of 0.6B model
- CUDA 12.7: Latest PyTorch support
- 128GB RAM: Can handle large datasets entirely in memory
- RISC-V cross-compilation: Available via Nix packages

---

## Phase 1: Environment Setup

### Technology Stack Selection

**Base Model:** Qwen3-0.6B

- 600M parameters (440M non-embedding)
- 32,768 token context window
- 28 transformer layers with Grouped Query Attention
- bfloat16 precision
- Built-in "thinking mode" for reasoning tasks

**Target Architecture:** RISC-V (rv64g)

- Simpler instruction set than x86-64
- Open-source and growing ecosystem
- Easier for model to learn initially
- Can expand to x86-64 and ARM later

**Training Framework:** Unsloth

- 2x faster training, 70% less VRAM usage
- QLoRA support (4-bit quantization + LoRA adapters)
- Perfect for single 8GB GPU setup
- Built-in support for Qwen3

### Reproducible Environment with Nix + Direnv

Created `shell.nix` declaring all dependencies:

```nix
- Python 3.12 + pip + virtualenv
- GCC 13.3.0 for native builds
- CUDA 12.x toolkit + cuDNN
- RISC-V cross-compiler (riscv64-unknown-linux-gnu-gcc)
- QEMU for RISC-V binary emulation
- Binary analysis tools (binutils, gdb)
```

Created `.envrc` for automatic environment activation:

```bash
- Loads Nix environment
- Creates Python venv on first use
- Activates venv automatically on cd
- Sets LD_LIBRARY_PATH for CUDA/PyTorch
- No manual intervention required
```

Configured `~/.config/nixpkgs/config.nix` to allow unfree packages (CUDA).

### Python Dependencies

Created `requirements.txt` with:

**Core ML Stack:**

- PyTorch 2.10.0 with CUDA 12.1
- Transformers ≥4.51.0 (required for Qwen3)
- Hugging Face Datasets & Hub
- Accelerate (distributed training support)

**Efficient Training:**

- Unsloth (from git, latest optimizations)
- bitsandbytes (4-bit/8-bit quantization)
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)

**Binary Analysis:**

- Capstone (disassembly framework)
- LIEF (ELF binary parsing)
- pyelftools (ELF format utilities)

**Utilities:**

- tqdm (progress bars)
- wandb (experiment tracking)
- ipython (interactive development)

### Installation Results

Environment setup completed with:

- All dependencies installed successfully
- GPU detected: NVIDIA RTX 2000 Ada
- PyTorch CUDA support verified
- Total Python environment: ~8GB

---

## Phase 2: Model Acquisition

### Base Model Download

Downloaded Qwen3-0.6B from Hugging Face:

- Model weights: `model.safetensors` (1.5GB)
- Tokenizer: `tokenizer.json` (11MB)
- Vocabulary: 151,936 tokens
- Configuration: `config.json` with architecture specs

**Model Location:** `./models/base/qwen3-0.6b/`

### Configuration for Deterministic Output

Modified `generation_config.json` for binary code generation:

**Changes:**

- `do_sample: false` - Use greedy decoding (no randomness)
- `temperature: 0.0` - Completely deterministic
- `top_k: 1` - Only consider best token
- `top_p: 1.0` - No nucleus sampling

**Rationale:** Binary/assembly code must be exact and reproducible. Same prompt should always produce same binary.

---

## Phase 3: Dataset Generation Strategy

### Dataset Requirements Analysis

Based on research into LLM binary generation:

- LLM4Decompile: Trained on 4B tokens (assembly + C pairs)
- RevEng.AI: Used 33M function pairs
- For 0.6B model: 10K-50K examples is reasonable

**Our Approach:** Start with 2,500 synthetic examples across 5 complexity levels.

### Synthetic Data Categories

**1. Constants (500 examples)**

- Simple return statements: `int main() { return 42; }`
- Value range: 0-10,000
- Tests basic code generation

**2. Arithmetic (500 examples)**

- Binary operations: `int main() { return 5 + 3; }`
- Operations: +, -, *, /, %
- Operand ranges: a∈[1,1000], b∈[1,500]
- Tests expression evaluation

**3. Variables (500 examples)**

- Variable assignment: `int main() { int x = 100; return x; }`
- Variable names: x, y, z, num, val, result
- Value range: 0-10,000
- Tests memory operations

**4. Conditionals (500 examples)**

- If/else statements: `if (a > b) return 1; else return 0;`
- Comparison operators: >, <, >=, <=, ==, !=
- Operand ranges: a,b∈[0,200]
- Return value ranges: true∈[1,100], false∈[101,200]
- Tests control flow

**5. Loops (496 examples, 4 compilation failures)**

- For loops with accumulation
- Range: summing integers 0 to N-1 where N∈[5,10000]
- Tests iteration and state management

### Dataset Generation Implementation

Created `scripts/dataset_generation/generate_synthetic_dataset.py`:

**Key Features:**

1. **Pre-generation of unique combinations**
   - Uses `random.sample()` for guaranteed uniqueness
   - No wasted compilations from duplicates
   - Efficient: compiles exactly once per unique program

2. **Deduplication system**
   - Tracks (prompt, source) tuples
   - Prevents same prompt mapping to different code
   - Ensures training data consistency

3. **RISC-V compilation**
   - Compiler: `riscv64-unknown-linux-gnu-gcc`
   - Flags: `-O0 -march=rv64g -mabi=lp64d -static -nostdlib -e main`
   - Generates both binary and assembly for each example

4. **Data format**
   Each example stored as JSON with:

   ```json
   {
     "id": "hash of source",
     "category": "constant|arithmetic|variable|conditional|loop",
     "prompt": "Natural language description",
     "source_code": "C source code",
     "binary_hex": "Full RISC-V binary (hex encoded)",
     "assembly": "RISC-V assembly listing",
     "binary_size": "Binary size in bytes"
   }
   ```

### Dataset Generation Results

**Output:** `dataset/processed/synthetic_dataset.jsonl`

**Statistics:**

- Total examples: 2,496 (99.8% success rate)
- File size: 9.9 MB
- Average binary size: ~1,580 bytes
- Format: JSON Lines (one example per line)

**Category Breakdown:**

- Constants: 500 ✓
- Arithmetic: 500 ✓
- Variables: 500 ✓
- Conditionals: 500 ✓
- Loops: 496 (4 compilation failures)

**Quality Metrics:**

- 2,496/2,496 unique prompts (100%)
- 2,496/2,496 unique source codes (100%)
- Zero prompt→source conflicts
- All required fields present in every example

### Dataset Verification

Created `scripts/dataset_generation/verify_dataset.py` for automated quality checks:

**Verification Tests:**

1. Prompt uniqueness check
2. Source code uniqueness check
3. Prompt→source mapping validation (critical for training)
4. Category distribution analysis
5. Data format validation
6. Binary size statistics
7. Sample inspection

**Result:** All checks passed ✓

---

## Phase 4: Binary Execution Validation

### QEMU Setup for RISC-V Emulation

Added QEMU to `shell.nix` for running RISC-V binaries on x86-64 host.

Created `scripts/test_binary.py` for validation:

**Features:**

- Compiles C code to RISC-V with `-static` linking
- Executes with `qemu-riscv64`
- Captures return code
- Validates against expected output
- 5-second timeout for safety

**Purpose:** Verify that generated binaries actually execute correctly.

---

## Project Structure

```
kai/
├── dataset/
│   ├── raw/              # (Future: external C code sources)
│   └── processed/        # Generated training data
│       └── synthetic_dataset.jsonl (2,496 examples, 9.9MB)
│
├── models/
│   ├── base/
│   │   └── qwen3-0.6b/   # Base model (1.5GB)
│   └── checkpoints/      # (Future: training checkpoints)
│
├── scripts/
│   ├── dataset_generation/
│   │   ├── generate_synthetic_dataset.py
│   │   └── verify_dataset.py
│   ├── download_model.py
│   ├── install_deps.sh
│   └── test_binary.py
│
├── configs/              # (Future: training configurations)
├── evaluation/           # (Future: evaluation scripts)
│
├── .envrc                # Direnv automation
├── .gitignore            # Excludes models, venv, datasets
├── shell.nix             # Nix environment declaration
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Key Design Decisions

### 1. Why RISC-V First?

- Simpler, more regular instruction set
- Fewer special cases than x86-64
- Open specification for reference
- Good learning target for LLM
- Can expand to other architectures after validation

### 2. Why Synthetic Dataset?

- Complete control over complexity
- Known correct outputs
- Fast iteration
- No licensing concerns
- Easy to verify correctness
- Can generate arbitrary quantities

### 3. Why Unsloth over Standard PEFT?

- 2x faster training
- 70% less VRAM usage
- Fits 0.6B model + training in 8GB VRAM
- Built-in QLoRA optimizations
- Active development and Qwen3 support

### 4. Why Assembly + Binary in Dataset?

- Assembly: Human-readable, good for model interpretability
- Binary: Ultimate ground truth, enables execution validation
- Both formats train model on different representations
- Flexibility in output format during inference

### 5. Deterministic Generation (temperature=0)

- Binary code has no room for creativity
- Same input must produce same output
- Easier to debug and validate
- Reproducible results for testing

---

## Technical Achievements

1. **Reproducible Environment**
   - Nix + direnv: Zero-friction environment activation
   - No "works on my machine" issues
   - Declarative dependency management

2. **Efficient Dataset Generation**
   - Pre-generation strategy: No wasted compilations
   - Deduplication: No training data conflicts
   - Verification: Automated quality assurance

3. **Cross-Architecture Development**
   - Native x86-64 development
   - RISC-V target compilation
   - QEMU emulation for validation
   - All tools integrated seamlessly

4. **Quality Assurance**
   - 100% unique prompts and source codes
   - Automated verification script
   - Binary execution testing capability
   - Format validation

---

## Metrics Summary

**Model:**

- Parameters: 600M (0.6B)
- Context window: 32,768 tokens
- Precision: bfloat16
- Size on disk: 1.5GB

**Dataset:**

- Examples: 2,496
- Size: 9.9 MB
- Success rate: 99.8%
- Uniqueness: 100%
- Coverage: 5 complexity levels

**Environment:**

- Setup time: ~30 minutes (first time)
- Activation time: <1 second (direnv)
- Python packages: 70+ installed
- Disk usage: ~10GB total (including model)

**Quality:**

- Zero duplicate prompts
- Zero prompt→source conflicts
- Zero data format errors
- Automated verification passing

---

## Completed Implementation

Training pipeline now functional:

1. **Data Preprocessing** ✅
   - `scripts/preprocessing/format_dataset.py` converts JSONL → Hugging Face Dataset
   - Formatted for instruction tuning with prompt/response pairs
   - Train/validation split (90/10)
   - Saved to `dataset/processed/training_dataset_hf/`

2. **Training Script** ✅
   - `scripts/train.py` with Unsloth configuration
   - QLoRA hyperparameters: 4-bit quantization, LoRA rank 16, alpha 32
   - Test mode (10 examples) and full mode (2,500 examples)
   - Checkpoint management to `models/checkpoints/`
   - VRAM-efficient: ~0.6GB loaded, 3.2GB peak

3. **NixOS/Triton Compatibility** ✅
   - Fixed Python.h compilation errors (python312Full)
   - Fixed ptxas discovery (TRITON_PTXAS_PATH)
   - All environment variables in `.envrc`
   - See `notes/Triton NixOS issue.md`

## Next Steps (Not Yet Implemented)

4. **Inference & Evaluation**
   - Generate binaries from prompts
   - Execute and validate on QEMU
   - Measure accuracy (exact match, execution success)

5. **Iteration**
   - Expand dataset with more complexity levels
   - Add function calls and control flow
   - Multi-architecture support (x86-64, ARM)

---

## Lessons Learned

**What Worked Well:**

- Nix + direnv: Seamless environment management
- Pre-generation strategy: Massive time savings
- Verification automation: Caught issues early
- JSONL format: Simple, flexible, line-oriented

**Key Technical Insights:**

- Dataset quality > quantity for initial validation
- Deduplication must happen at generation time, not post-processing
- Same prompt → different code is a critical bug that poisons training
- Binary size consistency (~1,580 bytes) indicates stable compilation

**Infrastructure Wins:**

- Scripts are reusable and documented
- Verification can run in CI/CD
- Dataset generation is deterministic (with seed)
- All tools accessible via direnv (no PATH management)

---

## References

**Research:**

- Qwen3 model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- LLM4Decompile: 4B tokens of binary/source pairs
- Unsloth documentation: <https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune>

**Tools:**

- RISC-V GNU Toolchain: <https://github.com/riscv-collab/riscv-gnu-toolchain>
- QEMU User Mode: <https://www.qemu.org/docs/master/user/index.html>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers>

---

**Status:** Training pipeline complete and functional. ✅

**Total Setup Time:** ~2 hours (including research, setup, dataset generation)

**Total Implementation Time:** ~4 hours (dataset formatting, training script, NixOS/Triton fixes)

**Current State:**
- Production-ready dataset of 2,496 examples, validated and verified ✅
- Training pipeline implemented with test mode ✅
- Triton compilation issues resolved on NixOS ✅
- Ready for full-scale training runs ✅

See `notes/Triton NixOS issue.md` for details on Python.h and ptxas compilation fixes.
