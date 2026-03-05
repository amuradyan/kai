#!/usr/bin/env python3
"""Compare generated binary output against expected output."""

import sys
from pathlib import Path


def compare_outputs(generated_file, expected_file):
    """Compare two hex outputs and report differences."""
    with open(generated_file, "r") as f:
        generated = f.read().strip()

    with open(expected_file, "r") as f:
        expected = f.read().strip()

    print("=" * 80)
    print("MODEL OUTPUT COMPARISON")
    print("=" * 80)

    gen_len = len(generated)
    exp_len = len(expected)

    print(f"\nExpected length: {exp_len} hex chars ({exp_len//2} bytes)")
    print(f"Generated length: {gen_len} hex chars ({gen_len//2} bytes)")
    print(f"Coverage: {gen_len/exp_len*100:.1f}%")
    print(f"Missing: {exp_len - gen_len} hex chars ({(exp_len - gen_len)//2} bytes)")

    # Find where they diverge
    diverge_at = -1
    for i in range(min(len(generated), len(expected))):
        if generated[i] != expected[i]:
            diverge_at = i
            break

    if diverge_at == -1:
        print(f"\n✅ First {gen_len} characters match perfectly!")
        print(f"   Model just stopped early (truncation issue)")
        byte_pos = gen_len // 2
        print(f"\n   Stopped at byte {byte_pos}")

        # Figure out what section this is in
        sections = [
            (0, 64, "ELF Header"),
            (64, 624, "Program Headers"),
            (624, 740, ".interp"),
            (740, 772, ".note.ABI-tag"),
            (776, 840, ".hash/.gnu.hash"),
            (840, 912, ".dynsym"),
            (912, 1173, ".dynstr"),
            (1174, 1216, ".gnu.version*"),
            (1216, 1240, ".rela.plt"),
            (1248, 1296, ".plt"),
            (1296, 1484, ".text (MAIN CODE)"),
            (1484, 1488, ".rodata"),
            (1488, 1616, ".eh_frame*"),
            (3552, 3560, ".preinit_array"),
            (3560, 3568, ".init_array"),
            (3568, 3576, ".fini_array"),
            (3576, 4072, ".dynamic"),
            (4072, 4104, ".got/.got.plt"),
            (4104, 4112, ".sdata"),
            (4112, 4130, ".comment"),
            (4130, 4232, ".riscv.attributes"),
            (4232, 5744, ".symtab"),
            (5744, 6371, ".strtab"),
            (6371, 6623, ".shstrtab"),
            (6624, 8416, "Section Headers"),
        ]

        for start, end, name in sections:
            if start <= byte_pos < end:
                progress = (byte_pos - start) / (end - start) * 100
                print(f"   Stopped in: {name}")
                print(f"   Progress through section: {progress:.1f}%")
                break
    else:
        print(
            f"\n❌ Output diverges at character {diverge_at} (byte {diverge_at//2})"
        )
        print(f"\n   Context around divergence:")
        start = max(0, diverge_at - 20)
        end = min(len(expected), diverge_at + 20)
        print(f"   Expected: ...{expected[start:end]}...")
        print(f"   Got:      ...{generated[start:end]}...")

        # Determine which section the divergence is in
        byte_pos = diverge_at // 2
        sections = [
            (0, 64, "ELF Header"),
            (64, 624, "Program Headers"),
            (624, 740, ".interp"),
            (740, 772, ".note.ABI-tag"),
            (776, 840, ".hash/.gnu.hash"),
            (840, 912, ".dynsym"),
            (912, 1173, ".dynstr"),
            (1174, 1216, ".gnu.version*"),
            (1216, 1240, ".rela.plt"),
            (1248, 1296, ".plt"),
            (1296, 1484, ".text (MAIN CODE)"),
            (1484, 1488, ".rodata"),
            (1488, 1616, ".eh_frame*"),
            (3552, 3560, ".preinit_array"),
            (3560, 3568, ".init_array"),
            (3568, 3576, ".fini_array"),
            (3576, 4072, ".dynamic"),
            (4072, 4104, ".got/.got.plt"),
            (4104, 4112, ".sdata"),
            (4112, 4130, ".comment"),
            (4130, 4232, ".riscv.attributes"),
            (4232, 5744, ".symtab"),
            (5744, 6371, ".strtab"),
            (6371, 6623, ".shstrtab"),
            (6624, 8416, "Section Headers"),
        ]

        for start, end, name in sections:
            if start <= byte_pos < end:
                print(f"\n   Divergence in section: {name}")
                break


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_output.py <generated_hex> <expected_hex>")
        print("  generated_hex: File with model-generated hex output")
        print("  expected_hex: File with expected hex output")
        sys.exit(1)

    generated_file = Path(sys.argv[1])
    expected_file = Path(sys.argv[2])

    if not generated_file.exists():
        print(f"Error: Generated file not found: {generated_file}")
        sys.exit(1)

    if not expected_file.exists():
        print(f"Error: Expected file not found: {expected_file}")
        sys.exit(1)

    compare_outputs(generated_file, expected_file)


if __name__ == "__main__":
    main()
