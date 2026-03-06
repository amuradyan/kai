#!/usr/bin/env python3
"""Generate Phase 1 dataset directly with strategic values for curriculum training.

Creates ~100 examples with both small (compressed) and large (uncompressed) values.
Includes END_BINARY token for clear stopping signal.
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
    print("Generating Phase 1 dataset for curriculum training...")
    print("=" * 80)

    output_file = Path("dataset/processed/phase1_dataset.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Strategic Phase 1 values - 81 total examples
    # Small values (1-31) for compressed instructions - 30 examples (excluding 15)
    phase1_values = list(range(1, 32))

    # Expanded large values for better coverage - 51+ examples
    large_values = [
        # Near boundary values
        32, 33, 35, 40, 42, 45, 50, 55, 60, 64,
        # Small-medium values
        70, 75, 80, 85, 90, 95, 100, 110, 120, 127,
        # Powers of 2 and nearby
        128, 150, 200, 255, 256, 300, 400, 500, 512,
        # Around 1000
        600, 700, 750, 800, 900, 999, 1000, 1024, 1100,
        # Medium-large values
        1200, 1337, 1500, 2000, 2048, 2500, 3000,
        # Large values
        4000, 4096, 5000, 6000, 7000, 8000, 9000, 9999, 10000
    ]
    phase1_values.extend(large_values)

    # Remove validation hold-outs (15 from small, 750 from large)
    phase1_values = [v for v in phase1_values if v not in [15, 750]]

    print(f"Generating {len(phase1_values)} examples:")
    print(f"  Small values (1-31, except 15): {len([v for v in phase1_values if v < 32])} examples")
    print(f"  Large values (strategic, except 750): {len([v for v in phase1_values if v >= 32])} examples")
    print(f"  Hold-out for validation: 15, 750")

    examples = []

    for i in tqdm(phase1_values, desc="Generating binaries", unit=" binary"):
        try:
            # Generate binary
            binary_bytes = generate_minimal_binary(i)
            binary_hex = binary_bytes.hex()

            # Create example with END_BINARY token
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

    print(f"\n✅ Generated {len(examples)} Phase 1 examples")
    print(f"📁 Saved to: {output_file}")

    # Statistics
    sizes = [len(ex['binary_hex']) // 2 for ex in examples]  # Divide by 2 for byte count (minus END_BINARY)
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    print(f"\nBinary size statistics:")
    print(f"  Average: {avg_size:.0f} bytes")
    print(f"  Min: {min_size} bytes")
    print(f"  Max: {max_size} bytes")

    # Show first example
    print(f"\nFirst example:")
    print(f"  Prompt: {examples[0]['prompt']}")
    print(f"  Has END_BINARY: {'<END_BINARY>' in examples[0]['binary_hex']}")
    print(f"  Binary size: {len(examples[0]['binary_hex']) // 2} bytes (excluding END_BINARY)")
    print(f"  Hex (last 50 chars): ...{examples[0]['binary_hex'][-50:]}")


if __name__ == "__main__":
    main()