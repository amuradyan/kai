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

_Continuing..._
