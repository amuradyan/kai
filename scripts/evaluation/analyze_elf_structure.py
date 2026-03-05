#!/usr/bin/env python3
"""Analyze and annotate ELF binary structure from hex string."""

import sys
from pathlib import Path


def show_section(hex_str, name, start, length, description=""):
    """Display a section of the hex string with annotation."""
    end = start + length
    hex_chunk = hex_str[start * 2 : end * 2]
    print(f"\n📦 {name} (bytes {start}-{end-1}, {length} bytes)")
    if description:
        print(f"   {description}")
    print(f"   Hex: {hex_chunk[:80]}{'...' if len(hex_chunk) > 80 else ''}")
    return end


def analyze_elf(hex_str):
    """Analyze and display ELF binary structure."""
    print("=" * 80)
    print("ELF BINARY STRUCTURE BREAKDOWN")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("PART 1: ELF HEADER (64 bytes)")
    print("=" * 80)

    pos = show_section(
        hex_str,
        "Magic + Identification",
        0,
        16,
        "7f454c46 = ELF magic, 02=64-bit, 01=little-endian, 01=v1",
    )
    pos = show_section(
        hex_str,
        "Type + Machine + Version",
        pos,
        8,
        "0200=EXEC, f300=RISC-V, 01000000=version 1",
    )
    pos = show_section(
        hex_str,
        "Entry point",
        pos,
        8,
        "Address where execution starts: 0x10510",
    )
    pos = show_section(
        hex_str,
        "Program header offset",
        pos,
        8,
        "Offset to program headers: 64 bytes",
    )
    pos = show_section(
        hex_str,
        "Section header offset",
        pos,
        8,
        "Offset to section headers: 6624 bytes (0x19e0)",
    )
    pos = show_section(
        hex_str, "Flags + Header sizes", pos, 12, "Architecture flags + header size info"
    )

    print(f"\nTotal ELF Header: {pos} bytes")

    print("\n" + "=" * 80)
    print("PART 2: PROGRAM HEADERS (10 headers × 56 bytes = 560 bytes)")
    print("=" * 80)

    ph_start = 64
    for i in range(10):
        ph_pos = ph_start + (i * 56)
        show_section(
            hex_str,
            f"Program Header {i}",
            ph_pos,
            56,
            f"Describes loadable segment or metadata",
        )

    print("\n" + "=" * 80)
    print("PART 3: LOADABLE SEGMENTS")
    print("=" * 80)

    show_section(hex_str, ".interp section", 0x270, 116, "Path to dynamic linker")
    show_section(hex_str, ".note.ABI-tag", 0x2E4, 32, "ABI identification note")
    show_section(
        hex_str, ".hash + .gnu.hash", 0x308, 64, "Symbol hash tables for dynamic linking"
    )
    show_section(hex_str, ".dynsym", 0x348, 72, "Dynamic symbol table")
    show_section(hex_str, ".dynstr", 0x390, 261, "Dynamic string table (symbol names)")
    show_section(hex_str, ".gnu.version*", 0x496, 42, "Symbol version information")
    show_section(hex_str, ".rela.plt", 0x4C0, 24, "PLT relocation entries")
    show_section(
        hex_str,
        ".plt",
        0x4E0,
        48,
        "Procedure Linkage Table (dynamic function calls)",
    )
    show_section(
        hex_str,
        ".text",
        0x510,
        188,
        "🔥 ACTUAL PROGRAM CODE - this is the main() function!",
    )
    show_section(hex_str, ".rodata", 0x5CC, 4, "Read-only data")
    show_section(
        hex_str, ".eh_frame*", 0x5D0, 128, "Exception handling / stack unwinding info"
    )

    print("\n" + "=" * 80)
    print("PART 4: DATA SEGMENTS")
    print("=" * 80)

    show_section(
        hex_str, ".preinit_array", 0xDE0, 8, "Pre-initialization function pointers"
    )
    show_section(hex_str, ".init_array", 0xDE8, 8, "Initialization function pointers")
    show_section(hex_str, ".fini_array", 0xDF0, 8, "Finalization function pointers")
    show_section(hex_str, ".dynamic", 0xDF8, 496, "Dynamic linking information")
    show_section(
        hex_str,
        ".got + .got.plt",
        0xFE8,
        32,
        "Global Offset Table (for dynamic linking)",
    )
    show_section(hex_str, ".sdata", 0x1008, 8, "Small data section")

    print("\n" + "=" * 80)
    print("PART 5: METADATA (non-loadable)")
    print("=" * 80)

    show_section(hex_str, ".comment", 0x1010, 18, "Compiler identification string")
    show_section(
        hex_str, ".riscv.attributes", 0x1022, 102, "RISC-V ISA extensions and ABI info"
    )
    show_section(hex_str, ".symtab", 0x1088, 1512, "Full symbol table (debugging)")
    show_section(hex_str, ".strtab", 0x1670, 627, "String table for symbols")
    show_section(
        hex_str,
        ".shstrtab",
        0x18E3,
        252,
        "Section header string table (section names)",
    )

    print("\n" + "=" * 80)
    print("PART 6: SECTION HEADERS (28 headers × 64 bytes = 1792 bytes)")
    print("=" * 80)

    sh_start = 0x19E0
    show_section(
        hex_str,
        "Section Headers",
        sh_start,
        1792,
        "28 entries describing all sections in the file",
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total binary size: {len(hex_str)//2} bytes ({len(hex_str)} hex chars)")
    print(f"")
    print(f"Key insight: The actual PROGRAM CODE (.text) is only 188 bytes!")
    print(f"Everything else is ELF metadata, dynamic linking, and debug info.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_elf_structure.py <hex_file>")
        print("  hex_file: File containing hex string of ELF binary")
        sys.exit(1)

    hex_file = Path(sys.argv[1])
    if not hex_file.exists():
        print(f"Error: File not found: {hex_file}")
        sys.exit(1)

    with open(hex_file, "r") as f:
        hex_str = f.read().strip()

    analyze_elf(hex_str)


if __name__ == "__main__":
    main()
