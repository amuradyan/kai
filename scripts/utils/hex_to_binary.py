#!/usr/bin/env python3
"""Convert hex string file to executable binary."""

import argparse
import sys
from pathlib import Path


def hex_to_binary(hex_file: Path, output_file: Path) -> None:
    """Convert hex string to binary executable.

    Args:
        hex_file: Path to file containing hex string
        output_file: Path to output binary file
    """
    # Read hex string
    hex_content = hex_file.read_text().strip()

    # Remove any whitespace
    hex_clean = ''.join(hex_content.split())

    # Validate hex
    if not all(c in '0123456789abcdefABCDEF' for c in hex_clean):
        print(f"❌ Invalid hex characters in {hex_file}")
        sys.exit(1)

    if len(hex_clean) % 2 != 0:
        print(f"❌ Hex string has odd length: {len(hex_clean)}")
        sys.exit(1)

    # Convert to binary
    try:
        binary_data = bytes.fromhex(hex_clean)
    except ValueError as e:
        print(f"❌ Failed to convert hex: {e}")
        sys.exit(1)

    # Write binary
    output_file.write_bytes(binary_data)

    # Make executable
    output_file.chmod(0o755)

    print(f"✅ Converted {hex_file} → {output_file}")
    print(f"   {len(hex_clean)} hex chars → {len(binary_data)} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Convert hex string file to executable binary"
    )
    parser.add_argument(
        "hex_file",
        type=Path,
        help="Input file containing hex string"
    )
    parser.add_argument(
        "output_file",
        type=Path,
        nargs='?',
        help="Output binary file (default: input without .raw/.hex extension)"
    )
    args = parser.parse_args()

    # Default output name: strip .raw or .hex extension
    if not args.output_file:
        stem = args.hex_file.stem
        if stem.endswith('.raw'):
            stem = stem[:-4]
        args.output_file = args.hex_file.parent / stem

    if not args.hex_file.exists():
        print(f"❌ File not found: {args.hex_file}")
        sys.exit(1)

    hex_to_binary(args.hex_file, args.output_file)


if __name__ == "__main__":
    main()
