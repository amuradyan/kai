# ELF Binary Structure Guide (Minimal Static Binary)

This document breaks down the hex representation of a minimal RISC-V ELF binary from our returns dataset.

## Overview

**Binary**: Program that returns 42
**Total Size**: 848 bytes (1696 hex characters)
**Approach**: Static linking with `-nostdlib`, direct syscalls

The binary is organized into these parts:

1. **ELF Header** (64 bytes) - File identification and metadata
2. **Program Headers** (168 bytes) - 3 headers describing loadable segments
3. **Code Segment** (288 bytes) - Actual executable code + exception handling
4. **Metadata** (284 bytes) - Compiler info, ISA attributes, section names
5. **Section Headers** (384 bytes) - 6 headers describing file sections

**Key insight**: Only **12 bytes** (1.4%) is actual program logic. The rest is ELF structure.

Compare to dynamic linking: 8416 bytes with 188 bytes code (2.2%), but 98% was linking infrastructure we eliminated.

---

## Part 1: ELF Header (bytes 0-63)

The ELF header identifies the file format and provides essential metadata.

### Bytes 0-15: Magic + Identification

```
7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
```

- `7f 45 4c 46` - ELF magic number (`\x7fELF`)
- `02` - 64-bit format (ELF64)
- `01` - Little-endian byte order
- `01` - ELF version 1
- `00 00 00...` - OS/ABI (UNIX System V) + padding

### Bytes 16-23: Type + Machine + Version

```
02 00 f3 00 01 00 00 00
```

- `02 00` - File type: EXEC (executable)
- `f3 00` - Machine: RISC-V (0x00f3)
- `01 00 00 00` - ELF version 1 (again)

### Bytes 24-31: Entry Point Address

```
e8 00 01 00 00 00 00 00
```

- `0x00000000000100e8` (little-endian) - Virtual address where execution starts
- This points directly to `_start` in the `.text` section

### Bytes 32-39: Program Header Table Offset

```
40 00 00 00 00 00 00 00
```

- `0x0000000000000040` (64 bytes) - Program headers start immediately after this header

### Bytes 40-47: Section Header Table Offset

```
d0 01 00 00 00 00 00 00
```

- `0x00000000000001d0` (464 bytes) - Section headers near end of file

### Bytes 48-63: Flags + Header Sizes

```
05 00 00 00 40 00 38 00 03 00 40 00 06 00 05 00
```

- `05 00 00 00` - Flags: RVC (compressed instructions), double-float ABI
- `40 00` - ELF header size: 64 bytes
- `38 00` - Program header entry size: 56 bytes
- `03 00` - Number of program headers: 3
- `40 00` - Section header entry size: 64 bytes
- `06 00` - Number of section headers: 6
- `05 00` - Section name string table index: 5

---

## Part 2: Program Headers (bytes 64-231)

Program headers describe segments loaded into memory. There are 3 headers, each 56 bytes.

### Program Header Structure

Each header has:
- Bytes 0-3: Type (PT_LOAD, PT_GNU_STACK, etc.)
- Bytes 4-7: Flags (Read/Write/Execute permissions)
- Bytes 8-15: Offset in file
- Bytes 16-23: Virtual address in memory
- Bytes 24-31: Physical address
- Bytes 32-39: Size in file
- Bytes 40-47: Size in memory
- Bytes 48-55: Alignment

### Header 0 (bytes 64-119): RISCV_ATTRIBUTES

```
03 00 00 70 04 00 00 00 32 01 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
66 00 00 00 00 00 00 00 66 00 00 00 00 00 00 00
01 00 00 00 00 00 00 00
```

- Type: `0x70000003` = RISCV_ATTRIBUTES
- Flags: `0x04` = Read-only
- Offset: `0x132` (306 bytes into file)
- Size: `0x66` (102 bytes)
- Contains ISA feature flags (rv64gc, extensions, etc.)

### Header 1 (bytes 120-175): PT_LOAD (Executable)

```
01 00 00 00 05 00 00 00 00 00 00 00 00 00 00 00
00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00
20 01 00 00 00 00 00 00 20 01 00 00 00 00 00 00
00 10 00 00 00 00 00 00
```

- Type: `0x01` = PT_LOAD (loadable segment)
- Flags: `0x05` = Read + Execute
- VirtAddr: `0x10000`
- Offset: `0x00`
- Size (file): `0x120` (288 bytes)
- Size (memory): `0x120` (288 bytes)
- Alignment: `0x1000` (4KB page)

**This segment contains `.text` (our code) and `.eh_frame` (exception handling).**

### Header 2 (bytes 176-231): GNU_STACK

```
51 e5 74 64 06 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
10 00 00 00 00 00 00 00
```

- Type: `0x6474e551` = PT_GNU_STACK
- Flags: `0x06` = Read + Write
- Size: 0 (no actual stack in file, allocated at runtime)
- Alignment: `0x10` (16 bytes)

Marks stack as non-executable for security.

---

## Part 3: Code Segment (bytes 232-519)

### .text (bytes 232-243): 🔥 THE ACTUAL PROGRAM CODE

Located at file offset `0xe8` (232 bytes), mapped to virtual address `0x100e8`.

```
93 08 d0 05 13 05 a0 02 73 00 00 00
```

This is RISC-V machine code - the entire program:

```asm
100e8:  93 08 d0 05     li   a7, 93        # Load syscall number 93 (exit)
100ec:  13 05 a0 02     li   a0, 42        # Load exit code 42
100f0:  73 00 00 00     ecall              # Make syscall
```

**That's it. 12 bytes. 3 instructions.**

1. Load 93 into register a7 (syscall number for `exit`)
2. Load 42 into register a0 (argument: exit code)
3. Execute syscall

Instruction encoding:
- `li a7, 93` = `05d00893` → stored as `93 08 d0 05` (little-endian)
- `li a0, 42` = `02a00513` → stored as `13 05 a0 02`
- `ecall` = `00000073` → stored as `73 00 00 00`

### .eh_frame (bytes 244-283): Exception Handling

```
00 00 00 00 10 00 00 00 00 00 00 00 01 7a 52 00
01 7c 01 01 1b 0c 02 00 10 00 00 00 18 00 00 00
d4 ff ff ff 0c 00 00 00 00 00 00 00
```

40 bytes (0x28) of DWARF exception handling metadata.

Not strictly necessary for this simple program, but included by compiler for consistency.

---

## Part 4: Metadata (bytes 284-519)

### .comment (bytes 288-305): Compiler Version

```
00 47 43 43 3a 20 28 47 4e 55 29 20 31 33 2e 33 2e 30 00
```

ASCII string: `GCC: (GNU) 13.3.0`

Null-terminated string identifying the compiler.

### .riscv.attributes (bytes 306-407): ISA Feature Flags

```
41 65 00 00 00 72 69 73 63 76 00 01 5b 00 00 00
04 10 05 72 76 36 34 69 32 70 31 5f 6d 32 70 30
5f 61 32 70 31 5f 66 32 70 32 5f 64 32 70 32 5f
63 32 70 30 5f 7a 69 63 73 72 32 70 30 5f 7a 69
66 65 6e 63 65 69 32 70 30 5f 7a 6d 6d 75 6c 31
70 30 5f 7a 61 61 6d 6f 31 70 30 5f 7a 61 6c 72
73 63 31 70 30 00
```

102 bytes describing RISC-V architecture features:

Starts with vendor tag, then ASCII strings:
- `riscv` - Architecture name
- `rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0`

This encodes:
- `rv64i2p1` - Base 64-bit integer ISA version 2.1
- `m2p0` - Multiply/divide extension
- `a2p1` - Atomic instructions
- `f2p2` - Single-precision floating point
- `d2p2` - Double-precision floating point
- `c2p0` - Compressed instructions (RVC)
- `zicsr`, `zifencei` - CSR and fence instructions
- `zmmul`, `zaamo`, `zalrsc` - Additional extensions

### .shstrtab (bytes 408-461): Section Name String Table

```
00 2e 73 68 73 74 72 74 61 62 00 2e 74 65 78 74
00 2e 65 68 5f 66 72 61 6d 65 00 2e 63 6f 6d 6d
65 6e 74 00 2e 72 69 73 63 76 2e 61 74 74 72 69
62 75 74 65 73 00
```

54 bytes of null-terminated ASCII strings:

- `\0` (index 0)
- `.shstrtab` (index 1)
- `.text` (index 11)
- `.eh_frame` (index 17)
- `.comment` (index 27)
- `.riscv.attributes` (index 36)

Section headers reference these by index.

---

## Part 5: Section Headers (bytes 464-847)

Section headers describe the 6 sections in the file. Each header is 64 bytes.

### Section Header Structure

- Bytes 0-3: Name offset (into `.shstrtab`)
- Bytes 4-7: Type (PROGBITS, STRTAB, etc.)
- Bytes 8-15: Flags (ALLOC, WRITE, EXECINSTR, etc.)
- Bytes 16-23: Virtual address
- Bytes 24-31: File offset
- Bytes 32-39: Section size
- Bytes 40-47: Entry size (for tables)
- Bytes 48-55: Link, info, alignment

### Header 0 (bytes 464-527): NULL Section

```
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

All zeros. Required by ELF spec (index 0 is always NULL).

### Header 1 (bytes 528-591): .text

```
0b 00 00 00 01 00 00 00 06 00 00 00 00 00 00 00
e8 00 01 00 00 00 00 00 e8 00 00 00 00 00 00 00
0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

- Name: index `0x0b` → ".text"
- Type: `0x01` = PROGBITS (program data)
- Flags: `0x06` = ALLOC + EXECINSTR (allocated in memory, executable)
- Address: `0x100e8`
- Offset: `0xe8` (232 bytes)
- Size: `0x0c` (12 bytes) ← **THE ACTUAL PROGRAM**
- Alignment: 2 bytes

### Header 2 (bytes 592-655): .eh_frame

```
11 00 00 00 01 00 00 00 02 00 00 00 00 00 00 00
f8 00 01 00 00 00 00 00 f8 00 00 00 00 00 00 00
28 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

- Name: index `0x11` → ".eh_frame"
- Type: PROGBITS
- Flags: `0x02` = ALLOC (allocated in memory, read-only)
- Address: `0x100f8`
- Offset: `0xf8` (248 bytes)
- Size: `0x28` (40 bytes)

### Header 3 (bytes 656-719): .comment

```
1b 00 00 00 01 00 00 00 30 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 20 01 00 00 00 00 00 00
12 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
```

- Name: index `0x1b` → ".comment"
- Type: PROGBITS
- Flags: `0x30` = MERGE + STRINGS (mergeable strings)
- Address: 0 (not allocated in memory)
- Offset: `0x120` (288 bytes)
- Size: `0x12` (18 bytes)

### Header 4 (bytes 720-783): .riscv.attributes

```
24 00 00 00 03 00 00 70 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 32 01 00 00 00 00 00 00
66 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

- Name: index `0x24` → ".riscv.attributes"
- Type: `0x70000003` = RISCV_ATTRIBUTES
- Offset: `0x132` (306 bytes)
- Size: `0x66` (102 bytes)

### Header 5 (bytes 784-847): .shstrtab

```
01 00 00 00 03 00 00 00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 98 01 00 00 00 00 00 00
36 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
```

- Name: index `0x01` → ".shstrtab"
- Type: `0x03` = STRTAB (string table)
- Offset: `0x198` (408 bytes)
- Size: `0x36` (54 bytes)

---

## Summary

**Size Breakdown:**

| Component | Bytes | % of Total | Purpose |
|-----------|-------|------------|---------|
| ELF Header | 64 | 7.5% | File identification |
| Program Headers | 168 | 19.8% | Memory layout |
| **`.text` (code)** | **12** | **1.4%** | **Actual program logic** |
| `.eh_frame` | 40 | 4.7% | Exception handling |
| `.comment` | 18 | 2.1% | Compiler version |
| `.riscv.attributes` | 102 | 12.0% | ISA features |
| `.shstrtab` | 54 | 6.4% | Section names |
| Section Headers | 384 | 45.3% | Section metadata |
| **Total** | **848** | **100%** | |

**Key Insights:**

1. **Only 12 bytes (1.4%) is actual program logic** - 3 RISC-V instructions
2. **No dynamic linking overhead** - Eliminated `.plt`, `.got`, `.dynamic`, `.interp`, `.dynsym`, `.dynstr`
3. **Minimal metadata** - Just compiler version and ISA features
4. **Simple structure** - 6 sections vs 28 in dynamic binary
5. **17x smaller** - 848 bytes vs 8416 bytes for equivalent dynamic binary

**For the model to generate this:**

- Learn ELF header format (always same structure)
- Learn minimal program headers (3 standard headers)
- Generate RISC-V instructions for the actual logic (12 bytes)
- Add standard metadata (always similar)
- Maintain correct offsets and sizes

**Comparison to Dynamic Linking:**

| Aspect | Minimal Static | Dynamic Linked | Improvement |
|--------|----------------|----------------|-------------|
| Total size | 848 bytes | 8416 bytes | **10x smaller** |
| Hex output | 1696 chars | 16832 chars | **10x fewer tokens** |
| Code section | 12 bytes (1.4%) | 188 bytes (2.2%) | Simpler |
| Total sections | 6 | 28 | **4.7x fewer** |
| Program headers | 3 | 10 | **3.3x fewer** |
| Has dynamic linking | No | Yes | **Eliminated complexity** |

**Why This Matters for Training:**

1. **Fits in context window**: 1696 hex chars fits in 2048 token window
2. **Less to memorize**: No dynamic linking infrastructure to learn
3. **More signal**: Higher ratio of actual code to boilerplate
4. **Language-agnostic**: No libc dependency, pure syscalls
5. **Faster training**: 10x smaller examples = 10x more data in same memory

---

## Appendix: Dynamic Linking Comparison

Our previous approach used dynamic linking with libc, resulting in 8416-byte binaries.

**What we eliminated:**

- **`.interp`** (116 bytes) - Dynamic linker path
- **`.dynsym`** (72 bytes) - Dynamic symbol table
- **`.dynstr`** (261 bytes) - Symbol name strings
- **`.plt`** (48 bytes) - Procedure linkage table (function call stubs)
- **`.got`** (32 bytes) - Global offset table (runtime addresses)
- **`.dynamic`** (496 bytes) - Dynamic linking metadata
- **`.symtab`** (1512 bytes) - Full symbol table for debugging
- **`.strtab`** (627 bytes) - Symbol string table
- **Multiple NOTE sections** (~200 bytes) - ABI tags, build IDs

**Why we don't need them:**

- We make direct syscalls instead of calling libc functions
- No shared libraries to link at runtime
- Static linking resolves everything at compile time
- Stripped debugging symbols with `--strip-all`

**Trade-off**: Can't use standard library functions (printf, malloc, etc.) - but for simple programs that just return values, we don't need them.

---

## Reading Hex Tips

- **Every 2 hex chars = 1 byte**: `7f454c46` = 4 bytes
- **Little-endian**: Multi-byte values are reversed
  - `e8 00 01 00 00 00 00 00` = `0x00000000000100e8`
  - Read: right-to-left for each byte group
- **ASCII strings**: Readable text in hex
  - `47 43 43` = "GCC"
  - `2e 74 65 78 74 00` = ".text\0"
- **Use tools**: `readelf`, `objdump`, `hexdump` to inspect binaries
- **Null bytes**: `00` for padding, null terminators, unused fields

---

**For complete understanding**, see also:
- `docs/Minimal Static Binaries.md` - Why we use this approach
- `scripts/dataset/generate_returns_dataset.py` - How we generate these binaries
- ELF specification: https://refspecs.linuxfoundation.org/elf/elf.pdf
