# The Case of Long Zero Footers

## 2026-03-06 - Testing with 777888

Ran the model with prompt "Write a program that returns 777888"

Value 777888 is way outside training range (1-10000).

Model output:

- Generated 922 hex chars (stopped at same point as before)
- Instruction generated: `1305c574`

Combined with dataset footer and ran:

- Exit code: **76**
- Expected: 777888 % 256 = **160**

Disassembly shows:

```asm
100e8:  6509                lui  a0,0x2
100ea:  05d00893            li   a7,93
100ee:  74c50513            addi a0,a0,1868    # 0x274c = 10060
```

**Model generated value: 10060**
**Expected value: 777888**

Same failure mode as before - generates wrong value. This time it's way off because 777888 is completely outside training distribution. Model generated something from within its known range instead.

---

## Footers Investigation

Question came up: do we have a script that fixes the footer and tests?

Answer: No. We have:

- `compare_output.py` - only compares
- `evaluate_model.py` - runs QEMU but doesn't fix
- `hex_to_binary.py` - converts but doesn't fix

Need to create one.

But first - are footers even the same across binaries?

Checked documentation (`docs/ELF Binary Structure Guide.md`):

- Footer contains section headers
- Section headers include .text size
- .text size varies based on compressed vs uncompressed instructions

Hypothesis: footers differ based on instruction encoding.

---

## Dataset Analysis

Tested values: 1, 2, 42, 100, 1000

Results:

```
Value 1:    848 bytes, footer = 774 chars
Value 2:    848 bytes, footer = 774 chars
Value 42:   848 bytes, footer = 774 chars
Value 100:  848 bytes, footer = 774 chars
Value 1000: 848 bytes, footer = 774 chars

Unique footers: 2
```

Footers are NOT identical.

Difference found at byte 96 (of the footer):

- Value 1: `0a` (10 bytes)
- Value 42: `0c` (12 bytes)

This is the .text section size in the section header.

---

## Why Footers Differ

RISC-V compressed instructions (RVC extension):

- Small values (< 32): `li a0, 1` = 2 bytes compressed
- Larger values (≥ 32): `li a0, 42` = 4 bytes uncompressed

Total .text section:

- `li a7, 93` - 4 bytes (always uncompressed)
- `li a0, N` - 2 or 4 bytes (depends on N)
- `ecall` - 4 bytes (always uncompressed)

Small values: 4 + 2 + 4 = **10 bytes**
Large values: 4 + 4 + 4 = **12 bytes**

Section header includes size field → footer varies.

---

## Observations

The footer is mostly zeros:

- Section headers (384 bytes)
- Lots of padding
- Only small variations (1-2 bytes difference)

Model output also ends in zeros:

- Generates up to cut point (~922 chars)
- Then fills with zeros (3179 trailing zeros)

Both have "long zero" sequences, but for different reasons:

- Footer zeros: ELF structure padding
- Model zeros: failed generation / no confidence

---

## Implications

For testing script:

- Can use **any** footer from dataset
- Metadata will be slightly wrong (10 vs 12 bytes)
- But execution doesn't care - QEMU doesn't validate section headers
- Binaries run fine with incorrect .text size metadata

Tools like `objdump` might show wrong size, but execution is unaffected.

---

## What We Learned

1. Model fails the same way for out-of-range values (777888 → 10060)
2. Footers are not universal (compressed vs uncompressed)
3. Footer difference is minimal (one byte in section header)
4. Wrong metadata doesn't break execution
5. We need a test script that fixes footer and runs

---

## The Long Zero Footer Problem

Started by looking at the note "When the tail of the binary was all wrong." Two distinct zero-tail problems were identified that had been getting conflated.

The model stops generating meaningful content around the 922-char mark — likely hitting a newline in training data or an uncertainty threshold — and then fills the rest of its token budget with approximately 3179 trailing zeros. This is a generation behavior failure, not an ELF issue.

The ELF footer also contains lots of zeros, but that's legitimate: section header padding is mostly zeroed out by design. The footer isn't even uniform across binaries — it varies by one byte at offset 96 of the footer, which is the `.text` section size field. Small values (N < 32) produce a 10-byte `.text` section using the RVC compressed instruction encoding; larger values (N ≥ 32) produce 12-byte `.text` using the full 4-byte uncompressed encoding. The section header records this, hence the difference.

The practical finding: QEMU doesn't validate section header metadata, so using any footer from the dataset as a replacement works fine at runtime. `objdump` would complain but execution is unaffected. A script to automate this (strip zeros, graft footer, run QEMU) was identified as missing.

## Why More Data Alone Won't Fix Value Encoding

The first training run used 10,000 examples over 3 epochs. The model learned ELF structure remarkably well — correct magic bytes, entry point, program headers, syscall sequence — but generated the wrong immediate value (145 instead of 42).

The problem is that the mapping from the text token "42" to the bytes `02 a0` in the immediate field of `addi` is a precise arithmetic transformation. It's buried in 1696 hex chars of mostly-static output with no direct supervision signal pointing at it. More examples of the same structure give more of the same weak signal. More epochs on existing data, or a stronger learning signal, is what's actually needed.

## Teaching Abstractions

The question came up: would framing the model as a compiler, or teaching it ELF and RISC-V explicitly, help? The framing idea is reasonable — Qwen3 almost certainly saw RISC-V ISA specs and ELF documentation during pretraining, and a system prompt framing could activate that knowledge. But it doesn't solve the core problem. The task isn't to *reason* about encoding, it's to *generate* exact hex bytes token by token. Understanding the format doesn't give you mechanical precision in sequence generation.

Breaking the hex into three parts (header / code / footer) in the dataset was considered. It would give a much cleaner signal by isolating the value-encoding problem to a tiny sequence. But since the output needs to stay as one blob, it would mean using segmented format only as a training supervision signal, which complicates the setup. Deferred for now.

## Curriculum Learning Plan

The approach settled on: two-phase curriculum training.

Phase 1 trains on roughly 100 examples — 50 small values (1–31, RVC compressed) and 50 large values (≥32, spread across the range). The goal is to teach the encoding mechanic in isolation before introducing the full complexity. Small and large values must both be covered because they produce structurally different binaries with different footers. 20–30 epochs at 1e-4 learning rate. Before proceeding to phase 2, the model is tested on held-out values it hasn't seen — if it gets both a small and large unseen value correct, the mechanic has transferred.

Phase 2 continues from the phase 1 checkpoint on the full 10,000 example dataset, 5–7 epochs at 5e-5 learning rate. The lower learning rate is important — the goal is generalization, not overwriting what phase 1 taught.

The assembly field in the dataset will not be used as training input. Output stays as one hex blob. The approach bets on curriculum structure rather than richer intermediate representations.
