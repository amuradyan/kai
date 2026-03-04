#!/usr/bin/env python3
"""
Generate synthetic C code dataset for binary generation training.

Creates simple C programs with increasing complexity:
1. Constants and return values
2. Simple arithmetic
3. Variables
4. If/else statements
5. Simple loops

Each example is compiled to RISC-V binary and stored with metadata.
"""

import json
import random
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


class SyntheticDatasetGenerator:
    def __init__(self, output_dir: str = "./dataset/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples = []
        self.seen_examples = set()  # Track (prompt, source) tuples for deduplication

    def _compile_to_riscv(self, source_code: str) -> Optional[Dict[str, str]]:
        """Compile C code to RISC-V binary and extract assembly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_file = tmpdir / "program.c"
            binary_file = tmpdir / "program.elf"
            asm_file = tmpdir / "program.s"

            # Write source
            source_file.write_text(source_code)

            try:
                # Compile to RISC-V binary
                subprocess.run(
                    [
                        "riscv64-unknown-linux-gnu-gcc",
                        "-O0",  # No optimization for now
                        "-march=rv64g",
                        "-mabi=lp64d",
                        str(source_file),
                        "-o", str(binary_file)
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Generate assembly
                subprocess.run(
                    [
                        "riscv64-unknown-linux-gnu-gcc",
                        "-S",
                        "-O0",
                        "-march=rv64g",
                        "-mabi=lp64d",
                        str(source_file),
                        "-o", str(asm_file)
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Read binary as hex
                with open(binary_file, "rb") as f:
                    binary_hex = f.read().hex()

                # Read assembly
                assembly = asm_file.read_text()

                # Get binary size
                binary_size = os.path.getsize(binary_file)

                return {
                    "binary_hex": binary_hex,
                    "assembly": assembly,
                    "binary_size": binary_size
                }

            except subprocess.CalledProcessError as e:
                print(f"Compilation failed: {e.stderr}")
                return None

    def _add_example(self, category: str, prompt: str, source: str, result: Dict) -> bool:
        """Add example only if unique (prompt, source) pair."""
        key = (prompt, source)
        if key in self.seen_examples:
            return False

        self.seen_examples.add(key)
        self.examples.append({
            "id": hashlib.md5(source.encode()).hexdigest()[:8],
            "category": category,
            "prompt": prompt,
            "source_code": source,
            **result
        })
        return True

    def generate_constants(self, count: int = 500) -> List[Dict]:
        """Generate simple return constant programs."""
        # Pre-generate unique values
        values = random.sample(range(0, 10000), count)
        added = 0

        for value in values:
            source = f"int main() {{ return {value}; }}"
            prompt = f"Write a C program that returns {value}"

            result = self._compile_to_riscv(source)
            if result:
                if self._add_example("constant", prompt, source, result):
                    added += 1

        print(f"Generated {added} constant examples")
        return []

    def generate_arithmetic(self, count: int = 500) -> List[Dict]:
        """Generate simple arithmetic programs."""
        ops = ["+", "-", "*", "/", "%"]
        op_names = {"+": "sum", "-": "difference", "*": "product", "/": "quotient", "%": "remainder"}

        # Pre-generate unique combinations
        combinations = set()
        while len(combinations) < count:
            a = random.randint(1, 1000)
            b = random.randint(1, 500)
            op = random.choice(ops)

            # Avoid division by zero
            if op in ["/", "%"] and b == 0:
                continue

            combinations.add((a, b, op))

        added = 0
        for a, b, op in combinations:
            source = f"int main() {{ return {a} {op} {b}; }}"
            prompt = f"Write a C program that returns the {op_names[op]} of {a} and {b}"

            result = self._compile_to_riscv(source)
            if result:
                if self._add_example("arithmetic", prompt, source, result):
                    added += 1

        print(f"Generated {added} arithmetic examples")
        return []

    def generate_variables(self, count: int = 500) -> List[Dict]:
        """Generate programs with variables."""
        var_names = ["x", "y", "z", "num", "val", "result"]

        # Pre-generate unique combinations
        combinations = set()
        while len(combinations) < count:
            value = random.randint(0, 10000)
            var_name = random.choice(var_names)
            combinations.add((value, var_name))

        added = 0
        for value, var_name in combinations:
            source = f"int main() {{ int {var_name} = {value}; return {var_name}; }}"
            prompt = f"Write a C program that stores {value} in variable '{var_name}' and returns it"

            result = self._compile_to_riscv(source)
            if result:
                if self._add_example("variable", prompt, source, result):
                    added += 1

        print(f"Generated {added} variable examples")
        return []

    def generate_conditionals(self, count: int = 500) -> List[Dict]:
        """Generate if/else statements."""
        ops = [">", "<", ">=", "<=", "==", "!="]
        op_names = {
            ">": "greater than", "<": "less than",
            ">=": "greater than or equal to", "<=": "less than or equal to",
            "==": "equal to", "!=": "not equal to"
        }

        # Pre-generate unique combinations
        combinations = set()
        while len(combinations) < count:
            a = random.randint(0, 200)
            b = random.randint(0, 200)
            op = random.choice(ops)
            true_val = random.randint(1, 100)
            false_val = random.randint(101, 200)
            combinations.add((a, b, op, true_val, false_val))

        added = 0
        for a, b, op, true_val, false_val in combinations:
            source = f"int main() {{ if ({a} {op} {b}) return {true_val}; else return {false_val}; }}"
            prompt = f"Write a C program that returns {true_val} if {a} is {op_names[op]} {b}, otherwise {false_val}"

            result = self._compile_to_riscv(source)
            if result:
                if self._add_example("conditional", prompt, source, result):
                    added += 1

        print(f"Generated {added} conditional examples")
        return []

    def generate_loops(self, count: int = 500) -> List[Dict]:
        """Generate simple loop programs."""
        # Pre-generate unique limit values
        limits = random.sample(range(5, 10000), count)
        added = 0

        for limit in limits:
            source = f"""int main() {{
    int sum = 0;
    for (int i = 0; i < {limit}; i++) {{
        sum += i;
    }}
    return sum;
}}"""

            prompt = f"Write a C program that sums integers from 0 to {limit-1}"

            result = self._compile_to_riscv(source)
            if result:
                if self._add_example("loop", prompt, source, result):
                    added += 1

        print(f"Generated {added} loop examples")
        return []

    def generate_dataset(self):
        """Generate complete dataset."""
        print("Generating synthetic dataset...")
        print("=" * 60)

        self.examples.extend(self.generate_constants(500))
        self.examples.extend(self.generate_arithmetic(500))
        self.examples.extend(self.generate_variables(500))
        self.examples.extend(self.generate_conditionals(500))
        self.examples.extend(self.generate_loops(500))

        print("=" * 60)
        print(f"Total examples generated: {len(self.examples)}")

        # Save dataset
        output_file = self.output_dir / "synthetic_dataset.jsonl"
        with open(output_file, "w") as f:
            for example in self.examples:
                f.write(json.dumps(example) + "\n")

        print(f"Dataset saved to: {output_file}")

        # Generate statistics
        self._print_statistics()

    def _print_statistics(self):
        """Print dataset statistics."""
        print("\nDataset Statistics:")
        print("-" * 60)

        categories = {}
        total_binary_size = 0

        for ex in self.examples:
            cat = ex["category"]
            categories[cat] = categories.get(cat, 0) + 1
            total_binary_size += ex["binary_size"]

        for cat, count in sorted(categories.items()):
            print(f"{cat:15s}: {count:5d} examples")

        if len(self.examples) > 0:
            print(f"\nTotal unique examples: {len(self.examples)}")
            print(f"Average binary size: {total_binary_size // len(self.examples)} bytes")
            print(f"Total dataset size: {total_binary_size / (1024*1024):.2f} MB")
        else:
            print("\nNo examples generated!")


def main():
    generator = SyntheticDatasetGenerator()
    generator.generate_dataset()


if __name__ == "__main__":
    main()
