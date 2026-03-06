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

---

## Phase 1 Curriculum Training Results

Ran Phase 1 training to teach value encoding mechanic.

**Experiment settings:**

- Dataset: 80 examples (values 1-31 small, various large up to 10000)
- Excluded values: 15, 750 (held out for validation)
- Learning rate: 1e-4
- Epochs: 40 (800 steps total)
- Batch size: 1, gradient accumulation: 4

**Results with prompt "Write a program that returns 50":**

Model output:

- Generated 922 hex chars (stops at same position as before)
- Instruction at code section: `9308d0051305806b73000000`
- Then filled with zeros for remaining ~3179 chars

With footer fix applied:

- Exit code: **184**
- Expected: **50**

**Observation:**

Model learned ELF structure through code section but consistently stops at position 922. Never generates footer sections (`.comment`, `.riscv.attributes`, section headers). The generation stops at exact same point as previous experiments - after the `.text` section code. Model outputs zeros for the remainder instead of continuing with ELF structure.

---

## Investigating the 922-char Stop Point

### Investigation Steps

**1. Checked if position 922 coincides with cut marker:**

- Cut marker `6d6d656e74002e72697363762e61747472696275746573` (hex for "mment.riscv.attributes")
- Found marker starts at position 876 in hex string
- Marker is 46 chars long
- 876 + 46 = **922** - exact stop position!

**2. Suspected training data corruption:**

- Checked if binary_hex strings were broken at newlines
- Verified JSONL has complete 1696-char hex strings on single lines
- Checked `format_for_training.py` - passes hex through as-is, no decoding
- Training data is clean

**3. Looked for `0a` byte (newline) issues:**

- Found `0a` byte exists in binary but as legitimate hex chars "0a"
- Not causing line breaks in training data
- Position 922 follows `.riscv.attributes` string, not the `0a` byte

**4. Checked max_tokens setting:**

- `generate_binary.py` defaults to `max_new_tokens=4096`
- Not a token limit issue

**5. Found the culprit - `early_stopping=True`:**

- Generation config had `early_stopping=True`
- Model likely generates EOS token `<|im_end|>` at position 922
- EOS token ID: 151645 for Qwen tokenizer
- Early stopping causes immediate return with zero padding

### Root Cause Analysis

Model learned position 922 (end of `.riscv.attributes`) as acceptable stopping point because:

1. Footer is mostly zeros/padding (low entropy, easy to predict)
2. Loss landscape gives good scores even for early stopping
3. `.riscv.attributes` looks like semantic boundary
4. Model learned this is where sequences "can" end

### Fix Applied

Changed `scripts/generation/generate_binary.py` line 86:

- From: `early_stopping=True`
- To: `early_stopping=False`

This forces generation to continue even after EOS token, up to max_new_tokens limit. We'll see how it goes.

---

## Analysis of Position 922 Stop Point

The investigation revealed why the model consistently stops at position 922:

### Pattern Recognition Problem

Position 922 marks the end of `.riscv.attributes` section - a natural semantic boundary. The model learned this as a valid completion point during training.

### Loss Gradient Issue

The footer (positions 922-1696) consists of:
- Section headers (384 bytes)
- Mostly zeros and padding
- Low information density

During training, the model achieved good loss scores even when truncating at 922. The remaining ~774 characters contribute minimal loss penalty, creating weak gradient signal to continue.

### EOS Token Placement

The model learned to generate EOS token `<|im_end|>` (ID: 151645) at position 922:
- Treats cut marker boundary as sequence end
- With `early_stopping=True`, generation halts immediately
- Model wasn't failing - it was successfully placing EOS where it learned sequences could end

### Why This Specific Position

The cut marker ending at 922 created perfect conditions:
- Readable string boundary (unlike raw binary)
- Precedes low-entropy section
- Training loss rewarded early completion

### The Early Stopping Mechanism

`early_stopping` parameter controls EOS behavior:
- **True**: Stops at first EOS token
- **False**: Continues to `max_new_tokens` limit

Setting to False forces continuation past EOS. However, this may reveal secondary issue: model might output zeros or garbage after 922 if footer structure wasn't learned.

### Implications

The fix addresses the mechanical stopping but not necessarily the underlying learning problem. The model needs stronger signal to learn footer importance. Testing with `early_stopping=False` will show whether model:
- Generates meaningful footer content
- Or just produces padding/zeros when forced to continue

---

## Test Results with early_stopping=False

Ran generation with the fix applied on prompt "Write a program that returns 50".

**Command:**
```bash
python scripts/evaluation/test_with_fixed_footer.py test_50_no_early_stop.raw.raw
```

**Results:**
```
Model output: 922 hex chars
Footer added: 774 hex chars
Total binary: 848 bytes
Exit code: 100
```

**Analysis of raw output:**
- File contains 4101 total chars (max_tokens=4096)
- First 922 chars: valid hex up to end of `.riscv.attributes`
- Remaining ~3179 chars: all zeros (padding to token limit)
- Expected total: 1696 chars for complete binary
- No actual footer structure generated

**Key finding:**
The `early_stopping=False` fix **failed**. Model still only generates 922 chars of meaningful content, then pads with zeros. The change only affected the padding behavior:
- With `early_stopping=True`: stops at 922
- With `early_stopping=False`: continues to max_tokens but only outputs zeros

**Root problem remains:**
The model has learned that meaningful content ends at position 922. It never learned to generate the footer section. The footer in training data has such low entropy (mostly zeros) that the model treats it as optional padding rather than essential structure

---

## All Non-Zeros 5x Weighting Experiment

### The Hypothesis

Instead of complex position-aware weighting, try a simple approach:
- Weight ALL non-zero hex characters 5x (everywhere in the binary)
- Add `<END_BINARY>` token for explicit stop signal
- This might solve both problems: footer generation AND value encoding

### Why This Might Work

1. **Zeros are overrepresented** - In the full binary:
   - Header: ~20% zeros
   - Code: ~30% zeros
   - Footer: ~90% zeros
   - Overall: ~60-70% zeros

2. **Non-zeros carry all information**:
   - Header: ELF magic bytes, metadata
   - Code: The actual return value encoding
   - Footer: Section sizes, offsets

3. **Current failures are with non-zeros**:
   - Value encoding wrong (145 instead of 42)
   - Footer non-zeros missed (stops at 922)

### Implementation

1. **Custom Weighted Loss**:
   - Find zero token ID from tokenizer
   - Weight all non-zero tokens 5x
   - Weight `<END_BINARY>` token 5x

2. **Dataset Changes**:
   - Append `<END_BINARY>` to all binary_hex strings
   - No other format changes needed

3. **Training Plan**:
   - 40 epochs on Phase 1 dataset (100 examples)
   - Learning rate: 1e-4
   - Branch: `all-non-zeros-5x`

### Expected Outcomes

If successful:
- Model generates complete 1696-char binaries
- Correct value encoding (non-zeros in code section get more attention)
- Clear stopping with `<END_BINARY>` token

Potential issues:
- Header overfitting (non-zeros in header might get memorized)
- 5x might be too high/low (could try 3x or 10x)

### Status: Training in progress...
