# Kai - prompt to machine code via LLMs

Can a small language model learn to generate executable binaries from natural language? This project explores that question by training Qwen3-0.6B to produce RISC-V machine code directly from prompts like "write a program that sums integers from 0 to 99".

## The Experiment

Most code generation models output source code—Python, C, JavaScript. But source code is just an intermediate representation. What if we skip the middleman and teach the model to speak directly in machine code?

We're starting with RISC-V because its instruction set is simpler and more regular than x86-64, making it easier for a model to learn. Once proven, the approach can extend to other architectures.

The project now includes a two-phase curriculum training approach to solve the value encoding problem—where the model learns ELF structure perfectly but fails to encode the correct values. Phase 1 teaches the encoding mechanic on 100 focused examples, then Phase 2 generalizes to the full dataset.

**Model:** Qwen3-0.6B (600M parameters)
**Target:** RISC-V 64-bit binaries
**Training:** QLoRA (4-bit quantization) on 8GB GPU
**Dataset:** 10,000 minimal static RISC-V binaries (prompt → hex format)

## Quick Start - Curriculum Training

For faster, more effective training using our weighted loss approach:

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
./scripts/setup/install_dependencies.sh

# 2. Generate Phase 1 dataset (83 strategic examples with END_BINARY)
python scripts/dataset/generate_phase1_dataset.py

# 3. Format for training
python scripts/dataset/format_for_training.py \
    --input dataset/processed/phase1_dataset.jsonl \
    --output dataset/processed/phase1_training

# 4. Train with weighted loss (40 epochs)
./kai train \
    --dataset dataset/processed/phase1_training_hf \
    --run-name all-non-zeros-5x \
    --num-train-epochs 40 \
    --learning-rate 1e-4

# 5. Test with validation values
python scripts/evaluation/validate_checkpoint.py \
    --checkpoint models/checkpoints/all-non-zeros-5x \
    --test-values 15 750
```

## Quick Start - Standard Training

Get from zero to a trained model in five steps:

```bash
# 1. Set up Python environment
python -m venv .venv
source .venv/bin/activate
./scripts/setup/install_dependencies.sh

# 2. Download the base model (runs in background)
python scripts/setup/download_base_model.py &

# 3. Generate training dataset (10,000 examples)
python scripts/dataset/generate_returns_dataset.py

# 4. Format dataset for training
python scripts/dataset/format_for_training.py

# 5. Train the model
./kai train --test  # Test mode: 10 examples, 10 steps
./kai train         # Full training: 10,000 examples, 3 epochs
```

**Using the Kai CLI:**

```bash
./kai train --test              # Quick training test
./kai train                     # Full training
./kai generate --prompt "..."   # Generate binaries (after training)
./kai help                      # Show help
```

## Setup

### System Requirements

**Required packages:**

- Python 3.10+ (3.12 recommended)
- GCC (for compiling binaries)
- RISC-V cross-compiler (`riscv64-unknown-linux-gnu-gcc` or `riscv64-linux-gnu-gcc`)
- QEMU with RISC-V support (`qemu-user` or `qemu-riscv64`)
- CUDA 12.1+ (for GPU training)

**Installation by distribution:**

**Ubuntu/Debian:**

```bash
sudo apt install python3 python3-venv gcc-riscv64-unknown-elf qemu-user
```

**Fedora:**

```bash
sudo dnf install python3 gcc-riscv64-linux-gnu qemu-user
```

**Arch Linux:**

```bash
sudo pacman -S python gcc riscv64-linux-gnu-gcc qemu-user
```

**NixOS (optional):**
If you prefer reproducible builds, this project includes Nix configuration:

```bash
direnv allow .  # Automatically sets up everything
```

### First Time Setup

**1. Create Python virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install Python dependencies:**

```bash
./scripts/setup/install_dependencies.sh
```

This installs PyTorch with CUDA 12.1, Transformers, Unsloth (for efficient QLoRA training), and binary analysis tools.

**3. Download the base model:**

```bash
python scripts/setup/download_base_model.py
```

Downloads Qwen3-0.6B from Hugging Face (~1.5GB). This can run in the background while you work on other steps.

### Daily Usage

Activate the virtual environment:

```bash
source .venv/bin/activate
```

If using NixOS with direnv, the environment activates automatically when you `cd` into the directory.

## Usage

### Generate Training Data

The model learns from pairs: natural language prompts and their corresponding RISC-V binaries. We use minimal static binaries that directly use syscalls without libc.

```bash
python scripts/dataset/generate_returns_dataset.py
```

This creates 10,000 minimal static binaries that return values from 1 to 10,000:

- Each binary is ~848 bytes (compared to 8KB+ for dynamically-linked binaries)
- Uses direct syscalls: `li a7, 93` (exit syscall) and `li a0, N` (return value)
- No libc dependency, no dynamic linking overhead
- Focused dataset for learning value encoding

Each binary is assembled directly to RISC-V machine code, producing both the binary (hex-encoded) and assembly listing. The dataset is saved to `dataset/processed/returns_dataset.jsonl` (~21MB).

**Test binary execution:**

```bash
python scripts/dataset/verify_binaries_qemu.py -n 10
```

Randomly samples 10 binaries from the dataset and executes them on QEMU to verify they produce correct return codes.

### Train the Model

Fine-tune Qwen3-0.6B using QLoRA to generate RISC-V binaries from prompts.

**First, format the dataset:**

```bash
python scripts/dataset/format_for_training.py
```

This converts the JSONL dataset to Hugging Face format with instruction/response pairs.

**Run a quick test (10 examples, ~10 seconds):**

```bash
./kai train --test
```

**Run full training (10,000 examples, ~2-3 hours):**

```bash
./kai train
```

Training happens on a single GPU using Unsloth's optimizations, fitting comfortably in 8GB VRAM. The model checkpoint is saved to `models/checkpoints/qwen3-0.6b-lora/`.

### Generate Binaries

Once trained, generate executable binaries from natural language:

```bash
./kai generate --prompt "Write a program that returns the sum of 10 and 20" --output my_binary
```

The model outputs hex-encoded RISC-V machine code, which is converted to an executable binary file. Test it with QEMU:

```bash
qemu-riscv64 my_binary
echo $?  # Should output 30
```

## Project Structure

```plaintext
kai/
├── dataset/
│   ├── raw/              # External datasets (future)
│   └── processed/        # Generated training data (JSONL)
│
├── models/
│   ├── base/             # Qwen3-0.6B base model
│   └── checkpoints/      # Training checkpoints
│
├── scripts/
│   ├── setup/
│   │   ├── download_base_model.py
│   │   └── install_dependencies.sh
│   ├── dataset/
│   │   ├── generate_returns_dataset.py      # Full 10K dataset generator
│   │   ├── generate_phase1_dataset.py       # Phase 1 curriculum dataset (83 examples)
│   │   ├── format_for_training.py
│   │   ├── create_phase1_subset.py          # Select from full dataset
│   │   ├── verify_returns_dataset.py        # Simplified verification for new format
│   │   └── verify_binaries_qemu.py          # [DEPRECATED - old format]
│   ├── training/
│   │   ├── train_model.py                # Main training script - supports --from-checkpoint
│   │   ├── train_dynamic_weighted.py     # [EXPERIMENTAL - FAILED] 100x weighting
│   │   └── train_progressive_weighted.py # [EXPERIMENTAL - FAILED] Progressive 1x→5x
│   ├── generation/
│   │   └── generate_binary.py
│   ├── evaluation/
│   │   ├── analyze_elf_structure.py       # Annotate binary with ELF info
│   │   ├── analyze_failures.py            # Analyze failure patterns
│   │   ├── compare_output.py              # Compare generated vs expected
│   │   ├── evaluate_model.py              # Systematic evaluation
│   │   ├── test_with_fixed_footer.py     # Quick test incomplete output
│   │   └── validate_checkpoint.py         # Phase 1/2 gate
│   └── utils/
│       └── hex_to_binary.py               # Convert hex to executable
│
├── configs/              # Training configurations (future)
│
├── .envrc                # Direnv automation
├── shell.nix             # Nix environment declaration
├── requirements.txt      # Python dependencies
└── README.md
```

## Hardware Requirements

**Minimum:**

- GPU with 8GB VRAM (for QLoRA training)
- CUDA 12.1+ support
- 20GB disk space (model + dataset + dependencies)

**This project was developed on:**

- NVIDIA RTX 2000 Ada (8GB VRAM)
- CUDA 12.7
- NixOS 24.11
- 128GB RAM (overkill, but helpful for large dataset generation)

## Why RISC-V?

RISC-V is a natural starting point for teaching models to generate machine code:

**Simplicity:** Fixed-width instructions, regular encoding, fewer special cases than x86-64

**Transparency:** Open ISA specification—no secrets, no quirks

**Tooling:** Mature GCC cross-compiler and QEMU emulator available via Nix

**Future:** Once proven on RISC-V, the approach extends to x86-64 and ARM with more training data

## Technical Approach

1. **Minimal static binaries** - Generate ~848 byte binaries with direct syscalls (no libc)
2. **Focused dataset** - 10,000 examples returning values 1-10000
3. **Curriculum training** - Two-phase approach to teach value encoding
4. **QLoRA fine-tuning** - Efficient 4-bit training with LoRA adapters
5. **Validation** - Execute generated binaries on QEMU, verify correctness

See `notes/Setup and Dataset Generation.md` for detailed technical log.

## Advanced Training Techniques

### Weighted Loss Training
We discovered the model struggles with low-entropy regions (like footers with 90% zeros). Our solution: weight all non-zero tokens 5x during training, forcing the model to pay attention to information-carrying bytes.

⚠️ **Critical Warning:** Weighted training on small datasets (<1000 examples) can cause catastrophic forgetting. The Phase 1 dataset (83 examples) is too small for non-uniform weights. Use the full 10,000 example dataset or uniform weights only.

### END_BINARY Token
Added a custom `<END_BINARY>` token to explicitly mark where binaries end. This solved the early stopping problem where the model would generate only 922 characters and fill the rest with zeros.

### Curriculum Learning
Two-phase approach:
- **Phase 1**: 81 strategic examples (30 small + 51 large values) with high learning rate
- **Phase 2**: Full 10K dataset with lower learning rate for generalization

See `notes/The case of long zero footers/` for detailed investigation.

---

**Note:** This is an experimental research project exploring whether small LLMs can learn machine code generation. It's not (yet) practical for real-world use—think of it as a proof of concept.
