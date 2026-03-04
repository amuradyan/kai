#!/usr/bin/env python3
"""
Test RISC-V binaries using QEMU emulator.

Compiles C code to RISC-V and runs it to verify the return value.
"""

import subprocess
import tempfile
import sys
from pathlib import Path


def compile_and_run(source_code: str, expected_return: int = None) -> dict:
    """
    Compile C code to RISC-V and run it with QEMU.

    Returns:
        dict with 'success', 'return_code', 'stdout', 'stderr'
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source_file = tmpdir / "test.c"
        binary_file = tmpdir / "test.elf"

        # Write source
        source_file.write_text(source_code)

        try:
            # Compile to RISC-V
            # Note: Using -static so it's self-contained for QEMU
            compile_result = subprocess.run(
                [
                    "riscv64-unknown-linux-gnu-gcc",
                    "-static",
                    str(source_file),
                    "-o", str(binary_file)
                ],
                check=True,
                capture_output=True,
                text=True
            )

            # Run with QEMU
            run_result = subprocess.run(
                ["qemu-riscv64", str(binary_file)],
                capture_output=True,
                text=True,
                timeout=5
            )

            result = {
                "success": True,
                "compiled": True,
                "return_code": run_result.returncode,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
            }

            # Check if return code matches expected
            if expected_return is not None:
                result["match"] = (run_result.returncode == expected_return)

            return result

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "compiled": False,
                "error": str(e),
                "stderr": e.stderr if hasattr(e, 'stderr') else ""
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "compiled": True,
                "error": "Execution timeout (>5s)",
            }


def main():
    """Test with example programs."""

    tests = [
        {
            "name": "Return 0",
            "code": "int main() { return 0; }",
            "expected": 0
        },
        {
            "name": "Return 42",
            "code": "int main() { return 42; }",
            "expected": 42
        },
        {
            "name": "Simple arithmetic",
            "code": "int main() { return 5 + 3; }",
            "expected": 8
        },
        {
            "name": "Variable",
            "code": "int main() { int x = 100; return x; }",
            "expected": 100
        },
        {
            "name": "Conditional",
            "code": "int main() { if (10 > 5) return 1; else return 0; }",
            "expected": 1
        },
    ]

    print("Testing RISC-V binary execution with QEMU")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        print(f"\n{test['name']}:", end=" ")
        result = compile_and_run(test['code'], test['expected'])

        if result['success']:
            if 'match' in result and result['match']:
                print(f"✓ PASS (returned {result['return_code']})")
                passed += 1
            else:
                print(f"✗ FAIL (expected {test['expected']}, got {result['return_code']})")
                failed += 1
        else:
            print(f"✗ ERROR: {result.get('error', 'Unknown error')}")
            if result.get('stderr'):
                print(f"  stderr: {result['stderr'][:200]}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
