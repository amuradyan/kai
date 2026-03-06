#!/usr/bin/env python3
"""Generate minimal static binary dataset for simple return values.

Creates 10,000 examples of programs that return numbers 1-10000.
Uses static linking with nostdlib to create minimal binaries (~500 bytes vs 8KB).
"""

import json
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm


def generate_minimal_binary(return_value: int) -> bytes:
    """Generate a minimal statically-linked RISC-V binary.

    Uses -static -nostdlib to avoid dynamic linking overhead.
    Writes custom _start function instead of main.

    Args:
        return_value: Integer to return (0-255 for valid exit codes)

    Returns:
        binary_bytes: The compiled binary as bytes
    """
    # Write minimal C program with custom _start
    # Exit syscall number for RISC-V is 93
    c_code = f"""
void _start() {{
    // RISC-V exit syscall: a7=93 (exit), a0=exit_code
    register long syscall_num asm("a7") = 93;
    register long exit_code asm("a0") = {return_value};
    asm volatile (
        "ecall"
        : /* no outputs */
        : "r"(syscall_num), "r"(exit_code)
        : "memory"
    );
    __builtin_unreachable();
}}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write source
        c_file = tmpdir / "program.c"
        c_file.write_text(c_code)

        # Compile with minimal flags
        obj_file = tmpdir / "program.o"
        binary_file = tmpdir / "program"
        asm_file = tmpdir / "program.s"

        # Compile to object
        subprocess.run([
            "riscv64-unknown-linux-gnu-gcc",
            "-c",
            "-O2",
            "-ffreestanding",
            "-nostdlib",
            str(c_file),
            "-o", str(obj_file)
        ], check=True, capture_output=True)

        # Link with minimal linker script
        subprocess.run([
            "riscv64-unknown-linux-gnu-gcc",
            "-static",
            "-nostdlib",
            "-Wl,--build-id=none",  # Remove build ID
            "-Wl,--strip-all",       # Strip symbols
            str(obj_file),
            "-o", str(binary_file)
        ], check=True, capture_output=True)

        # Read results
        binary_bytes = binary_file.read_bytes()

        return binary_bytes


def main():
    print("Generating minimal returns dataset...")
    print("=" * 80)

    output_file = Path("dataset/processed/returns_dataset.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    for i in tqdm(range(1, 10001), desc="Generating binaries", unit=" binary"):
        try:
            # Generate binary
            binary_bytes = generate_minimal_binary(i)
            binary_hex = binary_bytes.hex()

            # Create example - only keep essential fields
            # Add END_BINARY token for clear stopping signal
            example = {
                "prompt": f"Write a program that returns {i}",
                "binary_hex": binary_hex + "<END_BINARY>"
            }

            examples.append(example)

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate binary for return value {i}: {e}")
            continue

    # Save dataset
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✅ Generated {len(examples)} examples")
    print(f"📁 Saved to: {output_file}")

    # Statistics
    sizes = [len(ex['binary_hex']) // 2 for ex in examples]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)

    print(f"\nBinary size statistics:")
    print(f"  Average: {avg_size:.0f} bytes")
    print(f"  Min: {min_size} bytes")
    print(f"  Max: {max_size} bytes")
    print(f"  Hex chars per binary: {avg_size*2:.0f} (vs 16832 for dynamic linking)")

    # Show first example
    print(f"\nFirst example:")
    print(f"  Prompt: {examples[0]['prompt']}")
    print(f"  Binary size: {len(examples[0]['binary_hex']) // 2} bytes")
    print(f"  Hex (first 100 chars): {examples[0]['binary_hex'][:100]}...")


if __name__ == "__main__":
    main()
