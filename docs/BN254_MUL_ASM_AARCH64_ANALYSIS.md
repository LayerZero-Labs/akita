# BN254 Mul AArch64 Assembly Analysis

This report dumps the entire `BN254` field multiply symbol assembly and annotates it manually with a latency/throughput model.

## Source and symbol

- Assembly file: `target/release/deps/codegen_probe-855aaea603d2760f.s`
- Symbol analyzed: `_bn254_mul_field`
- Symbol span in `.s`: lines `860..1130`
- Target architecture: `arm64`

## Instruction census (exact for this symbol)

Counted directly from `_bn254_mul_field` instruction lines:

- `mul`: `36`
- `umulh`: `32`
- Total multiply-class ops: `68`
- `adds`: `42`
- `adcs`: `8`
- `adc`: `4`
- `cinc`: `42`
- `cmn`: `8`
- `cmp`: `8`
- branches (`b*`): `9` total (`b.hs`/`b.ne`/`b.lo`/`b.eq`/`b`)

## Bench numbers used for cycle mapping

From current `field_mul_isolated/*` and chain benches (median values):

- `fp128_mul_single`: `2.3145 ns`
- `bn254_mul_single`: `9.3428 ns`
- `fp128_pair_passthrough`: `1.8122 ns`
- `bn254_pair_passthrough`: `3.0869 ns`
- `fp128_mul_8way_independent`: `15.996 ns` total (8 muls)
- `bn254_mul_8way_independent`: `75.039 ns` total (8 muls)
- `fp128_8way_passthrough`: `4.6349 ns` total
- `bn254_8way_passthrough`: `8.4336 ns` total
- `mul_chain_2048` per-op:
  - `Fp128`: `11.924 us / 2048 = 5.822 ns`
  - `BN254`: `22.147 us / 2048 = 10.814 ns`

At `3.5 GHz`:

- `BN254 mul_chain` latency proxy: `10.814 ns ~= 37.85 cycles`
- `BN254 8-way net throughput proxy`:
  - `(75.039 - 8.4336) / 8 = 8.3257 ns ~= 29.14 cycles/op`

## How to read isolated bench subtraction

Recommended subtraction pairs:

1. **Single-op estimate (rough):**
   - `mul_single - black_box_only`
2. **Throughput estimate (better):**
   - `(mul_8way_independent - 8way_passthrough) / 8`

Using current medians:

- Single rough:
  - `Fp128`: `2.3145 - 0.5329 = 1.7816 ns`
  - `BN254`: `9.3428 - 1.0944 = 8.2484 ns`
  - Ratio: `~4.63x`
- 8-way net (preferred):
  - `Fp128`: `(15.996 - 4.6349)/8 = 1.4201 ns/op`
  - `BN254`: `(75.039 - 8.4336)/8 = 8.3257 ns/op`
  - Ratio: `~5.86x`

Notes:

- `pair_passthrough` is useful as a control but tends to over-subtract for tiny types; use it qualitatively, not as the primary net estimate.
- For latency claims, prefer `mul_chain_2048` (or `16384`) dependent chains.

## Manual phase annotation

The symbol is a fully inlined `Fp256<MontBackend<FrConfig, 4>>::mul` path on AArch64:

1. **Setup / loads** (`ldp` prologue, constants in `x17/x11/x9/x10/x12`)
2. **CIOS round i=0** using first multiplier limb (`x0`) and reduction constant `INV`
3. **CIOS round i=1** using second multiplier limb (`x2`)
4. **CIOS round i=2** using third multiplier limb (`x2` after `ldp x2, x1, [x1,#16]`)
5. **CIOS round i=3** using fourth multiplier limb (`x1`)
6. **Final conditional subtraction** (`cmp`/`b.hs` path and carry-corrected subtract chain)
7. **Store reduced 256-bit result** (`stp` x4-limb output)

Notable structural facts:

- Four unrolled Montgomery CIOS rounds are clearly visible.
- Carry propagation is explicit (`adds`/`adcs`/`adc` + `cinc`), creating serial chains.
- Final compare/subtract path is branchful and can extend the tail latency.

## Latency + throughput picture (manual model)

### Latency (single dependent op)

- The measured chain benchmark gives `~37.85 cycles/op` for BN254 mul on this machine.
- The asm shape supports this:
  - 4 serialized CIOS rounds with carry-dependent accumulation
  - final conditional subtract tail
- Manual estimate from dependency structure: **~37-43 cycles latency**, centered near measured `~38 cycles`.

### Throughput (independent ops)

- Static multiply pressure is high: `68` multiply-class instructions/call.
- Measured 8-way independent net gives `~29.14 cycles/op`.
- This indicates substantial overlap across independent operations and a backend that can keep multiply/add pipelines busy.
- Practical throughput on this system is therefore around **~29 cycles/op** for independent BN254 muls.

### Why chain ratio is smaller than naive instruction ratio

- Chain benchmarks measure **latency**, not pure instruction count.
- BN254 has more work, but OoO overlap and scheduling reduce the ratio compared with raw op-count intuition.
- Your current chain numbers (`~37.8 cycles BN254` vs `~20.4 cycles Fp128`) yielding `~1.86x` are consistent with this.

## Entire raw `_bn254_mul_field` assembly

```asm
_bn254_mul_field:
	.cfi_startproc
	stp	x24, x23, [sp, #-48]!
	.cfi_def_cfa_offset 48
	stp	x22, x21, [sp, #16]
	stp	x20, x19, [sp, #32]
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	ldp	x16, x15, [x0]
	ldp	x14, x13, [x0, #16]
	ldp	x0, x2, [x1]
	umulh	x9, x0, x16
	mul	x12, x0, x16
	mov	x17, #-268435457
	movk	x17, #62867, lsl #32
	movk	x17, #49889, lsl #48
	mul	x3, x12, x17
	mov	x11, #-268435455
	movk	x11, #62867, lsl #32
	movk	x11, #17377, lsl #48
	umulh	x4, x3, x11
	mul	x5, x3, x11
	umulh	x10, x0, x15
	mul	x6, x0, x15
	adds	x6, x9, x6
	cinc	x10, x10, hs
	mov	x9, #28817
	movk	x9, #31161, lsl #16
	movk	x9, #59464, lsl #32
	movk	x9, #10291, lsl #48
	umulh	x7, x3, x9
	mul	x19, x3, x9
	adds	x6, x19, x6
	cinc	x7, x7, hs
	umulh	x19, x0, x14
	mul	x20, x0, x14
	adds	x20, x10, x20
	cinc	x19, x19, hs
	mov	x10, #22621
	movk	x10, #33153, lsl #16
	movk	x10, #17846, lsl #32
	movk	x10, #47184, lsl #48
	umulh	x21, x3, x10
	mul	x22, x3, x10
	adds	x20, x20, x22
	cinc	x21, x21, hs
	cmn	x5, x12
	adcs	x4, x6, x4
	adcs	x5, x20, x7
	cinc	x6, x21, hs
	mul	x7, x0, x13
	adds	x20, x19, x7
	mov	x12, #41001
	movk	x12, #57649, lsl #16
	movk	x12, #20082, lsl #32
	movk	x12, #12388, lsl #48
	umulh	x21, x3, x12
	mul	x3, x3, x12
	adds	x3, x20, x3
	cinc	x20, x21, hs
	adds	x3, x3, x6
	cinc	x6, x20, hs
	cmn	x19, x7
	umulh	x0, x0, x13
	adc	x0, x6, x0
	umulh	x6, x2, x16
	mul	x7, x2, x16
	adds	x4, x4, x7
	cinc	x6, x6, hs
	mul	x7, x4, x17
	umulh	x19, x7, x11
	mul	x20, x7, x11
	umulh	x21, x2, x15
	mul	x22, x2, x15
	adds	x5, x5, x22
	cinc	x21, x21, hs
	adds	x5, x5, x6
	cinc	x6, x21, hs
	umulh	x21, x7, x9
	mul	x22, x7, x9
	adds	x5, x5, x22
	cinc	x21, x21, hs
	umulh	x22, x2, x14
	mul	x23, x2, x14
	adds	x3, x3, x23
	cinc	x22, x22, hs
	adds	x3, x3, x6
	cinc	x6, x22, hs
	umulh	x22, x7, x10
	mul	x23, x7, x10
	adds	x3, x3, x23
	cinc	x22, x22, hs
	cmn	x20, x4
	adcs	x5, x5, x19
	adcs	x3, x3, x21
	cinc	x4, x22, hs
	umulh	x19, x2, x13
	mul	x2, x2, x13
	adds	x0, x2, x0
	cinc	x2, x19, hs
	adds	x19, x0, x6
	umulh	x20, x7, x12
	mul	x7, x7, x12
	adds	x7, x19, x7
	cinc	x19, x20, hs
	adds	x4, x7, x4
	cinc	x7, x19, hs
	cmn	x0, x6
	adc	x0, x7, x2
	ldp	x2, x1, [x1, #16]
	umulh	x6, x2, x16
	mul	x7, x2, x16
	adds	x5, x5, x7
	cinc	x6, x6, hs
	mul	x7, x5, x17
	umulh	x19, x7, x11
	mul	x20, x7, x11
	umulh	x21, x2, x15
	mul	x22, x2, x15
	adds	x3, x3, x22
	cinc	x21, x21, hs
	adds	x3, x3, x6
	cinc	x6, x21, hs
	umulh	x21, x7, x9
	mul	x22, x7, x9
	adds	x3, x3, x22
	cinc	x21, x21, hs
	umulh	x22, x2, x14
	mul	x23, x2, x14
	adds	x4, x4, x23
	cinc	x22, x22, hs
	adds	x4, x4, x6
	cinc	x6, x22, hs
	umulh	x22, x7, x10
	mul	x23, x7, x10
	adds	x4, x4, x23
	cinc	x22, x22, hs
	cmn	x20, x5
	adcs	x3, x3, x19
	adcs	x4, x4, x21
	cinc	x5, x22, hs
	umulh	x19, x2, x13
	mul	x2, x2, x13
	adds	x0, x2, x0
	cinc	x2, x19, hs
	adds	x19, x0, x6
	umulh	x20, x7, x12
	mul	x7, x7, x12
	adds	x7, x19, x7
	cinc	x19, x20, hs
	adds	x5, x7, x5
	cinc	x7, x19, hs
	cmn	x0, x6
	adc	x0, x7, x2
	umulh	x2, x1, x16
	mul	x16, x1, x16
	adds	x16, x3, x16
	cinc	x2, x2, hs
	mul	x17, x16, x17
	umulh	x3, x17, x11
	mul	x6, x17, x11
	umulh	x7, x1, x15
	mul	x15, x1, x15
	adds	x15, x4, x15
	cinc	x4, x7, hs
	adds	x15, x15, x2
	cinc	x2, x4, hs
	umulh	x4, x17, x9
	mul	x7, x17, x9
	adds	x15, x15, x7
	cinc	x4, x4, hs
	umulh	x7, x1, x14
	mul	x14, x1, x14
	adds	x14, x5, x14
	cinc	x5, x7, hs
	adds	x14, x14, x2
	cinc	x2, x5, hs
	umulh	x5, x17, x10
	mul	x7, x17, x10
	adds	x7, x14, x7
	cinc	x5, x5, hs
	cmn	x6, x16
	adcs	x14, x15, x3
	adcs	x15, x7, x4
	cinc	x16, x5, hs
	umulh	x3, x1, x13
	mul	x13, x1, x13
	adds	x13, x13, x0
	cinc	x0, x3, hs
	adds	x1, x13, x2
	umulh	x3, x17, x12
	mul	x17, x17, x12
	adds	x17, x1, x17
	cinc	x1, x3, hs
	adds	x16, x17, x16
	cinc	x17, x1, hs
	cmn	x13, x2
	adc	x13, x17, x0
	cmp	x13, x12
	b.hs	LBB8_3
	mov	x12, x13
LBB8_2:
	mov	x10, x16
	mov	x9, x15
	b	LBB8_11
LBB8_3:
	b.ne	LBB8_10
	cmp	x16, x10
	b.lo	LBB8_2
	b.ne	LBB8_10
	cmp	x15, x9
	b.hs	LBB8_8
	mov	x9, x15
	b	LBB8_11
LBB8_8:
	cmp	x14, x11
	b.hs	LBB8_10
	cmp	x15, x9
	b.eq	LBB8_11
LBB8_10:
	mov	x9, #-4026531841
	movk	x9, #2668, lsl #32
	movk	x9, #48158, lsl #48
	cmp	x14, x11
	add	x14, x14, x9
	cset	w9, lo
	subs	x9, x15, x9
	ngc	x10, xzr
	mov	x11, #36719
	movk	x11, #34374, lsl #16
	movk	x11, #6071, lsl #32
	movk	x11, #55244, lsl #48
	adds	x9, x9, x11
	cinc	x10, x10, hs
	cmp	x10, #0
	cset	w10, eq
	subs	x10, x16, x10
	ngc	x11, xzr
	mov	x12, #42915
	movk	x12, #32382, lsl #16
	movk	x12, #47689, lsl #32
	movk	x12, #18351, lsl #48
	adds	x10, x10, x12
	cinc	x11, x11, hs
	cmp	x11, #0
	cset	w11, eq
	mov	x12, #24535
	movk	x12, #7886, lsl #16
	movk	x12, #45453, lsl #32
	movk	x12, #53147, lsl #48
	sub	x11, x13, x11
	add	x12, x11, x12
LBB8_11:
	stp	x14, x9, [x8]
	stp	x10, x12, [x8, #16]
	ldp	x20, x19, [sp, #32]
	ldp	x22, x21, [sp, #16]
	ldp	x24, x23, [sp], #48
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	ret
	.cfi_endproc
```
