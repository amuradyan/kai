# Kai - prompt to machine code via LLMs

Can a small language model learn to generate executable binaries from natural language? This project explores that question by training Qwen3-0.6B to produce RISC-V machine code directly from prompts like "write a program that sums integers from 0 to 99".

## The Experiment

Most code generation models output source code—Python, C, JavaScript. But source code is just an intermediate representation. What if we skip the middleman and teach the model to speak directly in machine code?

We're starting with RISC-V because its instruction set is simpler and more regular than x86-64, making it easier for a model to learn. Once proven, the approach can extend to other architectures.

**Model:** Qwen3-0.6B (600M parameters, 32K context)
**Target:** RISC-V 64-bit binaries
**Training:** QLoRA (4-bit quantization) on 8GB GPU
**Dataset:** Synthetic C programs → compiled RISC-V binaries

## Quick Start

Get from zero to a trained model in four steps:

```bash
# 1. Set up the environment (first time only)
direnv allow .
./scripts/install_deps.sh

# 2. Download the base model (runs in background)
python scripts/download_model.py &

# 3. Generate training dataset (2,500 examples)
python scripts/dataset_generation/generate_synthetic_dataset.py

# 4. Train the model (coming soon)
# python scripts/train.py
```

The environment automatically activates when you `cd` into the directory—no manual setup needed after the first time.

## Setup

This project uses NixOS with direnv for completely reproducible builds. Every dependency, from Python to the RISC-V compiler, is declared and versioned.

### First Time Setup

**1. Activate the environment:**

```bash
direnv allow .
```

This command triggers everything:

- Loads Nix shell with Python 3.12, GCC, CUDA, RISC-V toolchain, and QEMU
- Creates a Python virtual environment in `.venv/`
- Sets up CUDA paths for PyTorch
- Activates the venv automatically

**2. Install Python dependencies:**

```bash
./scripts/install_deps.sh
```

This installs the ML stack: PyTorch with CUDA 12.1, Transformers, Unsloth (for efficient QLoRA training), and binary analysis tools like Capstone and LIEF.

**3. Download the base model:**

```bash
python scripts/download_model.py
```

Downloads Qwen3-0.6B from Hugging Face (~1.5GB). This can run in the background while you work on other steps.

### Daily Usage

The magic of direnv: just navigate to the project.

```bash
cd ~/playground/kai
# Environment loads automatically
# Virtual environment activates
# CUDA paths configured
# RISC-V compiler available
```

Everything you need is in scope. No activation scripts, no PATH juggling.

## Usage

### Generate Training Data

The model learns from pairs: natural language prompts and their corresponding RISC-V binaries. We generate these synthetically by creating simple C programs and compiling them.

```bash
python scripts/dataset_generation/generate_synthetic_dataset.py
```

This creates 2,500 examples across five complexity levels:

- **Constants** (500): `return 42;`
- **Arithmetic** (500): `return 5 + 3;`
- **Variables** (500): `int x = 100; return x;`
- **Conditionals** (500): `if (a > b) return 1; else return 0;`
- **Loops** (500): `for (int i = 0; i < N; i++) sum += i;`

Each example is compiled to RISC-V with `riscv64-unknown-linux-gnu-gcc`, producing both the binary (hex-encoded) and assembly listing. The dataset is saved to `dataset/processed/synthetic_dataset.jsonl` (~10MB).

**Verify the dataset:**

```bash
python scripts/dataset_generation/verify_dataset.py
```

Checks for duplicate prompts, source code conflicts, and data format validity.

**Test binary execution:**

```bash
python scripts/verify_binaries_qemu.py -n 10
```

Randomly samples 10 binaries from the dataset and executes them on QEMU to verify they produce correct return codes.

### Train the Model

*(Coming soon)*

Fine-tune Qwen3-0.6B using QLoRA to generate RISC-V binaries from prompts:

```bash
python scripts/train.py
```

Training happens on a single GPU using Unsloth's optimizations, fitting comfortably in 8GB VRAM.

### Generate Binaries

*(Coming soon)*

Once trained, generate executable binaries from natural language:

```bash
python scripts/generate.py --prompt "write a program that returns the sum of 10 and 20"
```

The model outputs hex-encoded RISC-V machine code, which can be written to a file and executed on QEMU or real RISC-V hardware.

## Project Structure

```
kai/
├── dataset/
│   ├── raw/              # External C code sources (future)
│   └── processed/        # Generated training data (JSONL)
│
├── models/
│   ├── base/             # Qwen3-0.6B base model
│   └── checkpoints/      # Training checkpoints (future)
│
├── scripts/
│   ├── dataset_generation/
│   │   ├── generate_synthetic_dataset.py
│   │   └── verify_dataset.py
│   ├── download_model.py
│   ├── install_deps.sh
│   └── verify_binaries_qemu.py
│
├── configs/              # Training configurations (future)
├── evaluation/           # Evaluation scripts (future)
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

1. **Synthetic data generation** - Create diverse C programs, compile to RISC-V
2. **Instruction tuning** - Format as prompt → binary pairs
3. **QLoRA fine-tuning** - Efficient 4-bit training with LoRA adapters
4. **Deterministic generation** - Temperature 0, greedy decoding (binaries must be exact)
5. **Validation** - Execute generated binaries on QEMU, verify correctness

See `notes/Setup and Dataset Generation.md` for detailed technical log.

## Status

- ✅ Environment setup (Nix + direnv)
- ✅ Base model downloaded (Qwen3-0.6B)
- ✅ Dataset generation pipeline (2,500 examples)
- ✅ Binary validation on QEMU
- 🚧 Training pipeline (in progress)
- 🚧 Inference and evaluation (planned)

---

**Note:** This is an experimental research project exploring whether small LLMs can learn machine code generation. It's not (yet) practical for real-world use—think of it as a proof of concept.
