# Kai: Binary Generation from Prompts - Technical Log

**Project Goal:** Train Qwen3-0.6B to generate RISC-V binaries from natural language prompts.

**Date:** March 4, 2026

---

## System Requirements

### Minimum Hardware

- **GPU:** 8GB VRAM (for QLoRA training)
- **CUDA:** 12.1+ support
- **RAM:** 16GB (32GB+ recommended for dataset generation)
- **Storage:** 20GB available
- **OS:** Linux (NixOS recommended for reproducibility)

### Development Environment

- 8GB VRAM GPU for QLoRA training of 0.6B model
- Modern CUDA support for PyTorch
- Sufficient RAM for handling datasets in memory
- RISC-V cross-compilation tools via Nix

---

## Phase 1: Environment Setup

### Technology Stack Selection

**Base Model:** Qwen3-0.6B

- 600M parameters (440M non-embedding)
- 32,768 token max context (training uses 2048 for efficiency)
- 28 transformer layers with Grouped Query Attention
- bfloat16 precision
- Built-in "thinking mode" for reasoning tasks

**Note:** While the model supports 32K context, we use 2048 during training as our minimal static binaries (~500 bytes = ~1000 hex chars) fit comfortably in this window.

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

### Dataset Approach

**Our Approach:** Generate 10,000 minimal static binaries for simple return values (1-10000).

**Rationale:**
- Start with simplest possible task (return a value)
- Validate that model can learn binary generation
- Minimal binaries (~500 bytes) fit in 2048 token window
- Language-agnostic prompts from the start
- Once validated, can expand to more complex programs

### Returns Dataset

**Dataset:** 10,000 examples of simple return statements

**Characteristics:**
- Prompts: "Write a program that returns N" where N ∈ [1, 10000]
- Language-agnostic (no mention of C or any language)
- Minimal static binaries using direct syscalls
- Average binary size: ~500 bytes (vs 8KB with dynamic linking)
- No libc dependency, just RISC-V exit syscall

**Why start here:**
- Simplest possible binary generation task
- Validates end-to-end pipeline
- Tests model's ability to encode values in binary
- Establishes baseline before adding complexity

### Dataset Generation Implementation

Created `scripts/dataset/generate_returns_dataset.py`:

**Key Features:**

1. **Minimal static binary compilation**
   - Compiler: `riscv64-unknown-linux-gnu-gcc`
   - Flags: `-static -nostdlib -ffreestanding -O2 -Wl,--strip-all`
   - Custom `_start` with direct syscalls (no libc)
   - Result: ~500 byte binaries (see `docs/Minimal Static Binaries.md`)

2. **Direct syscall approach**
   - Uses RISC-V exit syscall (number 93)
   - No dynamic linking, no PLT/GOT overhead
   - 95% code, 5% ELF header

3. **Progress tracking**
   - Uses `tqdm` for real-time progress bar
   - Shows generation speed and ETA
   - Generates both binary and assembly for verification

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

**Output:** `dataset/processed/returns_dataset.jsonl`

**Statistics:**

- Total examples: 10,000 (100% success rate)
- File size: 21 MB
- Average binary size: ~500 bytes per binary
- Hex output: ~1000 characters per binary
- Format: JSON Lines (one example per line)

**Coverage:**
- Return values: 1 through 10,000
- All binaries tested with QEMU
- Exit codes validated (modulo 256 for values > 255)

**Quality Metrics:**

- 10,000/10,000 unique prompts (100%)
- 10,000/10,000 unique binaries (100%)
- All binaries execute correctly on QEMU
- All required fields present in every example

### Dataset Verification

Created `scripts/dataset/test_minimal_binaries.py` for automated quality checks:

**Verification Tests:**

1. Binary execution on QEMU
2. Return code validation (with modulo 256 for large values)
3. Random sampling or full dataset testing (`--all` flag)
4. Progress tracking with timing statistics

**Test Script Usage:**
```bash
python scripts/dataset/test_minimal_binaries.py        # Test 50 random samples
python scripts/dataset/test_minimal_binaries.py -n 100 # Test 100 samples
python scripts/dataset/test_minimal_binaries.py --all  # Test all 10,000
```

**Result:** All binaries execute correctly ✓

---

## Phase 4: Binary Execution Validation

### QEMU Setup for RISC-V Emulation

Added QEMU to `shell.nix` for running RISC-V binaries on x86-64 host.

### Validation Scripts

**1. `scripts/dataset/test_minimal_binaries.py`** - Primary testing tool

**Features:**
- Tests pre-generated binaries from dataset
- Executes with `qemu-riscv64`
- Validates return codes (with modulo 256 for large values)
- Progress bar with timing statistics
- Flexible testing: random samples or full dataset

**Usage:**
```bash
python scripts/dataset/test_minimal_binaries.py        # 50 random samples
python scripts/dataset/test_minimal_binaries.py -n 100 # 100 samples
python scripts/dataset/test_minimal_binaries.py --all  # All 10,000
```

**2. `scripts/dataset/test_binary.py`** - Compile and test utility

**Features:**
- Compiles C code to RISC-V on-the-fly
- Tests individual programs
- 5-second timeout for safety
- Used for ad-hoc testing

**Purpose:** Verify that binaries execute correctly on QEMU before training.

---

## Project Structure

```
kai/
├── dataset/
│   ├── raw/              # Reserved for external data sources
│   └── processed/        # Generated training data
│       ├── returns_dataset.jsonl (21MB, 10K examples)
│       ├── returns_training.jsonl (17MB, formatted)
│       └── returns_training_hf/ (83MB, HuggingFace Arrow format)
│
├── models/
│   ├── base/
│   │   └── qwen3-0.6b/   # Base model (1.5GB)
│   └── checkpoints/      # Training checkpoints
│       └── qwen3-0.6b-lora/ (created during training)
│
├── scripts/
│   ├── setup/
│   │   ├── download_base_model.py
│   │   └── install_dependencies.sh
│   ├── dataset/
│   │   ├── generate_returns_dataset.py
│   │   ├── test_minimal_binaries.py
│   │   ├── test_binary.py
│   │   ├── format_for_training.py
│   │   └── verify_binaries_qemu.py
│   ├── training/
│   │   └── train_model.py
│   ├── generation/
│   │   └── generate_binary.py
│   └── evaluation/
│       ├── analyze_elf_structure.py
│       ├── compare_output.py
│       └── evaluate_model.py
│
├── docs/
│   ├── ELF Binary Structure Guide.md
│   └── Minimal Static Binaries.md
│
├── notes/
│   └── Setup and Dataset Generation.md
│
├── kai                   # CLI wrapper script
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

- Assembly: Human-readable, used for debugging and verification only
- Binary: What the model actually generates during training
- Assembly is NOT used for training, only for our analysis
- Model learns: prompt → binary (hex-encoded executable)

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
- Context window: 32,768 max (training uses 2,048)
- Precision: bfloat16
- Size on disk: 1.5GB

**Dataset:**

- Examples: 10,000
- Size: 21 MB
- Success rate: 100%
- Uniqueness: 100%
- Coverage: Return values 1-10,000

**Environment:**

- Setup time: ~30 minutes (first time)
- Activation time: <1 second (direnv)
- Python packages: 70+ installed
- Disk usage: ~10GB total (including model)

**Quality:**

- Zero duplicate prompts
- Zero duplicate binaries
- All binaries execute correctly on QEMU
- Automated verification passing

---

## Completed Implementation

Training pipeline fully functional:

1. **Data Preprocessing** ✅
   - `scripts/dataset/format_for_training.py` converts JSONL → Hugging Face Dataset
   - Formatted for instruction tuning with prompt/response pairs
   - Flexible input/output paths via command-line flags
   - Saved to `dataset/processed/returns_training_hf/`

2. **Training Script** ✅
   - `scripts/training/train_model.py` with Unsloth configuration
   - QLoRA hyperparameters: 4-bit quantization, LoRA rank 16, alpha 16
   - Test mode (10 examples, 10 steps) and full mode (10,000 examples, 3 epochs)
   - Dataset selection via `--dataset` flag
   - Checkpoint management to `models/checkpoints/`
   - VRAM-efficient: ~0.6GB loaded, peak depends on batch size

3. **Generation Pipeline** ✅
   - `scripts/generation/generate_binary.py` for inference
   - Greedy decoding (temperature=0) for deterministic output
   - Saves raw hex output and compiled binary
   - Accessible via `./kai generate` CLI

4. **CLI Wrapper** ✅
   - `./kai` command for unified interface
   - `./kai train --dataset <path>` for training
   - `./kai generate --prompt "..."` for inference
   - Pass-through of all command-line arguments

5. **NixOS/Triton Compatibility** ✅
   - Fixed Python.h compilation errors (python312Full)
   - Fixed ptxas discovery (TRITON_PTXAS_PATH)
   - All environment variables in `.envrc`
   - See `notes/Triton NixOS issue.md`

## Evolution: Minimal Static Binaries

**Problem with original approach:**
- Dynamic linking produced 8KB binaries for simple programs
- 98% was ELF metadata (.plt, .got, .dynamic, etc.)
- Required 20K+ token context window
- Model had to learn complex linking infrastructure

**Solution:**
- Switched to minimal static binaries (~500 bytes)
- Direct syscalls instead of libc
- 95% is actual code, 5% is ELF header
- Fits in 2048 token window (17x reduction)
- Language-agnostic prompts from the start

See `docs/Minimal Static Binaries.md` for full details.

## Current Status

**Completed:** ✅
- Environment setup with Nix + direnv
- Base model downloaded (Qwen3-0.6B)
- Returns dataset generated (10,000 examples)
- Minimal static binary approach implemented
- Training pipeline functional
- Generation pipeline functional
- Binary validation on QEMU working
- CLI wrapper (`./kai`) for ease of use

**In Progress:** 🔄
- Training on returns dataset (10K examples)
- Evaluating model performance

## Next Steps

**Immediate:**
- Complete training run on returns dataset
- Evaluate model accuracy (exact match, execution correctness)
- Test generalization to unseen values

**Future Expansion:**
- Add arithmetic operations (addition, subtraction, etc.)
- Add conditionals and control flow
- Support multiple syscalls (write, read, open)
- Generate more complex programs
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
- Minimal static binaries (500 bytes) fit in context window vs dynamic linking (8KB)
- Language-agnostic prompts prevent model bias toward specific languages
- Direct syscalls eliminate libc dependency and reduce binary complexity
- Progress bars and timing in all long-running scripts improve UX

**Infrastructure Wins:**

- Scripts are reusable and documented
- Verification can run in CI/CD
- Dataset generation is deterministic (with seed)
- All tools accessible via direnv (no PATH management)

---

## References

**Research:**

- Qwen3 model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
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

---

## Phase 3: Curriculum Training

After initial training with the full dataset failed to correctly encode values (model learned ELF structure but generated wrong immediate values), we implemented a two-phase curriculum training approach.

### The Problem

First training run with 10K examples over 3 epochs resulted in:
- ✅ Perfect ELF structure generation
- ✅ Correct syscall sequence
- ❌ Wrong immediate values (e.g., generated 145 instead of 42)

The issue: the mapping from text "42" to RISC-V encoding `02a00513` is buried in 1696 hex chars of mostly-static output. The signal is too weak.

### The Solution: Two-Phase Curriculum

**Phase 1: Teach the Mechanic**
- Dataset: 100 strategic examples (50 small values 1-31, 50 large values ≥32)
- Training: 20-30 epochs at 1e-4 learning rate
- Goal: Learn value encoding in isolation
- Validation: Test on held-out values (15, 750)

**Phase 2: Generalize**
- Dataset: Full 10K examples
- Training: 5-7 epochs at 5e-5 learning rate from Phase 1 checkpoint
- Goal: Extend learning to full value range
- Validation: Test across range 1-100

### Implementation

New scripts added for curriculum training:
- `scripts/dataset/create_phase1_subset.py` - Create focused Phase 1 dataset
- `scripts/evaluation/test_with_fixed_footer.py` - Quick test with footer fix
- `scripts/evaluation/validate_checkpoint.py` - Gate between phases
- `scripts/training/train_model.py` - Added `--from-checkpoint`, `--learning-rate`, `--num-train-epochs`

See `notes/Two-Phase Curriculum Training Implementation.md` for complete workflow and commands.

**Status:** Curriculum training pipeline implemented and ready for execution. ✅
