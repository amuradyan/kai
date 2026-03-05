# Minimal Static Binaries Approach

## The Problem

Initial dataset used dynamically-linked binaries:
- **Size**: 8416 bytes (16832 hex chars) per binary
- **Context needed**: 20K+ tokens to generate one binary
- **Complexity**: 98% is ELF metadata, dynamic linking infrastructure
- **Training difficulty**: Model must learn complex `.plt`, `.got`, `.dynamic` sections

For a simple `return 42` program:
- Actual code: ~30 bytes
- Dynamic linking overhead: ~8300 bytes

## The Solution

Use **static linking with `nostdlib`** to create minimal binaries:

### Compilation Approach

Instead of standard compilation:
```c
int main() { return 42; }
```

Use direct syscalls with custom `_start`:
```c
void _start() {
    // RISC-V exit syscall: a7=93 (exit), a0=exit_code
    register long syscall_num asm("a7") = 93;
    register long exit_code asm("a0") = 42;
    asm volatile ("ecall" : : "r"(syscall_num), "r"(exit_code) : "memory");
    __builtin_unreachable();
}
```

Compile with:
```bash
riscv64-unknown-linux-gnu-gcc \
    -static \
    -nostdlib \
    -ffreestanding \
    -O2 \
    -Wl,--build-id=none \
    -Wl,--strip-all \
    program.c -o program
```

### Results

**Size reduction**:
- Dynamic: ~8400 bytes (16832 hex chars)
- Static minimal: ~500 bytes (1000 hex chars)
- **Reduction**: 94% smaller, 17x fewer tokens

**Complexity reduction**:
- No `.interp`, `.plt`, `.got`, `.dynamic` sections
- No symbol tables, no notes
- Just: ELF header + program header + code segment
- ~5% is metadata, ~95% is actual code

### Binary Structure (Minimal)

```
Minimal ELF binary (~500 bytes):
├── ELF Header (64 bytes)
│   Magic, type, entry point, header offsets
├── Program Headers (56 bytes × 1-2)
│   PT_LOAD: Single executable segment
└── Code Segment (~380 bytes)
    Actual RISC-V instructions
```

Compare to dynamic binary:
```
Dynamic ELF binary (~8400 bytes):
├── ELF Header (64 bytes)
├── Program Headers (560 bytes, 10 headers)
├── .interp (116 bytes) - dynamic linker path
├── .dynsym (72 bytes) - dynamic symbols
├── .dynstr (261 bytes) - symbol strings
├── .plt (48 bytes) - procedure linkage table
├── .got (32 bytes) - global offset table
├── .text (188 bytes) - actual code ← only this matters
├── .dynamic (496 bytes) - dynamic linking info
├── .symtab (1512 bytes) - symbol table
├── .strtab (627 bytes) - string table
└── Section Headers (1792 bytes, 28 headers)
```

## Benefits for Training

1. **Faster training**: 17x fewer tokens per example
2. **Lower VRAM**: Can fit in 2048 context window
3. **Easier to learn**: Simpler structure, less memorization
4. **More data**: Can train on larger datasets with same resources
5. **Better generalization**: Less boilerplate to overfit on

## Trade-offs

**Pros**:
- Much smaller binaries
- Simpler structure
- Faster training
- Language-agnostic (no libc dependency)

**Cons**:
- Requires syscall knowledge (but model learns this)
- Can't use standard library functions (for now)
- Limited to programs that can be expressed as syscalls

## Future Work

Once model learns minimal binaries:
1. Add syscalls beyond exit (write, read, open, etc.)
2. Gradually introduce more complex patterns
3. Eventually support full programs with libc (if needed)

## Dataset: Returns (1-10000)

First minimal dataset:
- **Examples**: 10,000
- **Prompts**: "Write a program that returns N" (N=1 to 10000)
- **Size per binary**: ~500 bytes
- **Total dataset**: ~5MB (vs ~84MB for dynamic)
- **Language**: None specified (language-agnostic)

Generate with:
```bash
python scripts/dataset/generate_returns_dataset.py
```

Test with:
```bash
python scripts/dataset/test_minimal_binaries.py
```

## Implementation Details

### Exit Code Range

UNIX exit codes are 0-255 (8-bit). For values > 255:
- Stored value: N
- Exit code: N % 256
- Test scripts handle this automatically

### RISC-V Exit Syscall

```asm
li a7, 93       # Syscall number for exit
li a0, <value>  # Exit code
ecall           # Make syscall
```

The C inline assembly approach lets the compiler optimize the rest while ensuring correct syscall convention.

### Verification

All binaries are tested with QEMU:
```bash
qemu-riscv64 ./program
echo $?  # Check exit code
```

## Key Insight

**The goal isn't to generate C programs or assembly - it's to generate executable binaries.**

By eliminating language-specific boilerplate (dynamic linking, libc), we make the task:
- Simpler for the model
- More aligned with the actual goal
- Language-agnostic from the start

The model learns: **prompt → minimal executable binary**

Not: **prompt → C code → compiled with specific toolchain → bloated binary**
