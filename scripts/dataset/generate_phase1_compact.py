#!/usr/bin/env python3
"""Build a compact training dataset from phase1_dataset.jsonl.

The compact dataset trains the model to emit only the 22 hex characters
at byte offsets 232-242 of each binary — the only region that varies
across examples. The 1674-char ELF wrapper is removed from the training
target. At generation time, the wrapper is glued back on by
generate_gpt2_compact.py, which also recomputes the two .eh_frame CFI
length nibbles (bytes 280 and 560) deterministically from the emitted
code length.

Outputs:
- dataset/processed/phase1_compact.jsonl       (compact training set)
- dataset/processed/phase1_wrapper_template.json (constant wrapper +
                                                  variable-region offsets)

Run scripts/dataset/format_for_training.py afterwards to build the HF
dataset for training.
"""
import argparse
import json
from pathlib import Path

VAR_HEX_START = 464
VAR_HEX_END = 486            # exclusive
CFI_NIBBLE_POSITIONS = [561, 1121]
END_MARKER = "<END_BINARY>"


def strip_marker(h: str) -> str:
    return h.split(END_MARKER)[0] if END_MARKER in h else h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="dataset/processed/phase1_dataset.jsonl")
    p.add_argument("--output", default="dataset/processed/phase1_compact.jsonl")
    p.add_argument(
        "--template-output",
        default="dataset/processed/phase1_wrapper_template.json",
        help="Where to save the wrapper template + offsets",
    )
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    tpl = Path(args.template_output)

    if not inp.exists():
        print(f"❌ Input not found: {inp}")
        return 1

    print(f"Loading {inp}...")
    with open(inp) as f:
        examples = [json.loads(line) for line in f]
    print(f"  Loaded {len(examples)} examples")

    # Verify uniformity: all binaries same length, only the documented
    # regions vary.
    first = strip_marker(examples[0]["binary_hex"])
    L = len(first)
    print(f"  Binary length: {L} hex chars")

    bad_length = [e for e in examples if len(strip_marker(e["binary_hex"])) != L]
    if bad_length:
        print(f"❌ {len(bad_length)} examples have non-uniform length")
        return 1

    # Build wrapper from first example: replace variable region with
    # placeholder underscores; CFI nibbles get their own placeholder.
    wrapper_chars = list(first)
    for i in range(VAR_HEX_START, VAR_HEX_END):
        wrapper_chars[i] = "_"
    for i in CFI_NIBBLE_POSITIONS:
        wrapper_chars[i] = "?"
    wrapper = "".join(wrapper_chars)

    # Verify wrapper is consistent across all examples outside the
    # variable region and the CFI nibbles.
    skip = set(range(VAR_HEX_START, VAR_HEX_END)) | set(CFI_NIBBLE_POSITIONS)
    diffs = 0
    for e in examples:
        h = strip_marker(e["binary_hex"])
        for i, c in enumerate(h):
            if i in skip:
                continue
            if first[i] != c:
                diffs += 1
                break
    if diffs:
        print(f"❌ {diffs} examples differ from wrapper outside expected regions")
        return 1
    print(f"✅ Wrapper is consistent across all {len(examples)} examples")

    template = {
        "wrapper_with_placeholders": wrapper,
        "variable_region": [VAR_HEX_START, VAR_HEX_END],
        "cfi_nibble_positions": CFI_NIBBLE_POSITIONS,
        "comment": (
            "Underscores mark the 22-char variable region the model emits. "
            "Question marks mark the two .eh_frame CFI length nibbles, "
            "which the post-processor computes from the emitted code length."
        ),
    }
    tpl.parent.mkdir(parents=True, exist_ok=True)
    with open(tpl, "w") as f:
        json.dump(template, f, indent=2)
    print(f"✅ Wrote wrapper template to {tpl}")

    # Compact dataset: prompt unchanged, binary_hex = variable region only
    # + END_BINARY marker.
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for e in examples:
            h = strip_marker(e["binary_hex"])
            var_region = h[VAR_HEX_START:VAR_HEX_END]
            assert len(var_region) == VAR_HEX_END - VAR_HEX_START
            record = {
                "id": e.get("id"),
                "category": e.get("category"),
                "prompt": e["prompt"],
                "binary_hex": var_region + END_MARKER,
            }
            f.write(json.dumps(record) + "\n")
    print(f"✅ Wrote {len(examples)} compact examples to {out}")
    print(f"   Variable region per example: {VAR_HEX_END - VAR_HEX_START} hex chars + {END_MARKER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
