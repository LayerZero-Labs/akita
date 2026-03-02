# Fp128 Mul AArch64 Assembly Analysis

Fresh rebuild was used to generate this assembly:

- Command: `cargo clean && cargo rustc --bench codegen_probe --release -- --emit asm`
- Assembly file: `target/release/deps/codegen_probe-1c486cb16985754a.s`
- Symbol analyzed: `_fp128_mul_field`
- Symbol span in `.s`: lines `97..141`
- Target architecture: `arm64`

## Instruction census (exact for `_fp128_mul_field`)

- `mul`: `7`
- `umulh`: `6`
- Total multiply-class ops: `13`
- `adds`: `9`
- `adcs`: `3`
- `adc`: `3`
- `cset`: `5`
- `cinc`: `2`
- `csel`: `2`
- `ldp`: `2`
- `stp`: `1`

This is a compact, spill-free kernel.

## Fresh benchmark anchors

From fresh run medians:

- `field_mul_only/mul_chain_2048`:
  - Fp128: `11.948 us` total -> `5.8340 ns/op`
  - BN254: `22.225 us` total -> `10.8521 ns/op`
  - ratio `BN254/Fp128 ~= 1.86x`
- `field_mul_isolated` (Fp128 only):
  - `fp128_mul_single`: `2.4427 ns`
  - `fp128_pair_passthrough`: `1.8375 ns`
  - `fp128_black_box_only`: `0.55984 ns`
  - `fp128_mul_8way_independent`: `16.247 ns` total
  - `fp128_8way_passthrough`: `4.8632 ns` total

Derived estimates:

- `single - black_box_only`: `1.8829 ns` (`~6.59 cycles @3.5GHz`)
- `(8way_independent - 8way_passthrough)/8`: `1.4230 ns/op` (`~4.98 cycles @3.5GHz`)
- chain-latency proxy: `5.8340 ns/op` (`~20.42 cycles @3.5GHz`)

Interpretation:

- Throughput can approach ~5 cycles/op in independent streams.
- Dependent latency stays around ~20 cycles/op, matching the carry-chain structure in codegen.

## Manual phase annotation

`_fp128_mul_field` naturally splits into 4 stages:

1. **Load limbs**
   - `ldp x9,x10,[x0]` and `ldp x11,x12,[x1]`
2. **2x2 schoolbook + row carries**
   - 4 low products + 4 highs arranged with `adds/adcs/adc`
3. **Solinas fold-1 with `c=8207`**
   - `mul/umulh` by `c` on upper carry limb path
4. **Fold-2 + canonicalize**
   - add `c` once, then choose reduced/non-reduced using final carry condition (`csel`)

The emitted code is already branchless in the hot path (`csel` only, no branch mispredict risk).

## Entire raw `_fp128_mul_field` assembly

```asm
_fp128_mul_field:
	.cfi_startproc
	ldp	x9, x10, [x0]
	ldp	x11, x12, [x1]
	mul	x13, x11, x9
	umulh	x14, x11, x9
	mul	x15, x12, x9
	umulh	x9, x12, x9
	umulh	x16, x11, x10
	mul	x11, x11, x10
	umulh	x17, x12, x10
	mul	x10, x12, x10
	adds	x11, x11, x14
	cset	w12, hs
	adds	x9, x9, x16
	cset	w14, hs
	adds	x9, x9, x10
	cinc	x10, x14, hs
	adds	x11, x11, x15
	adcs	x9, x9, x12
	adc	x10, x17, x10
	mov	w12, #8207
	mul	x14, x9, x12
	umulh	x9, x9, x12
	umulh	x15, x10, x12
	mul	x10, x10, x12
	adds	x9, x9, x11
	cset	w11, hs
	adds	x9, x9, x10
	cinc	x10, x11, hs
	adds	x11, x14, x13
	cset	w13, hs
	adcs	x14, x9, xzr
	adc	x10, x15, x10
	mul	x10, x10, x12
	adds	x10, x11, x10
	cset	w11, hs
	adc	x9, x9, x13
	adds	x12, x10, x12
	adcs	x11, x14, x11
	csel	x9, x11, x9, hs
	csel	x10, x12, x10, hs
	stp	x10, x9, [x8]
	ret
	.cfi_endproc
```

## AArch64 carry-path attempt result

I implemented and benchmarked an AArch64-specific inline-asm carry helper path (`adds/adc` helper blocks for 2/3/4-term limb sums), then reverted it.

Observed outcome (after thermal cooldown rerun):

- It did **not** improve this kernel.
- Primary reason: helper-level asm barriers reduced global scheduling quality and caused overall slowdown versus LLVM's current fully inlined codegen.

Conclusion: micro-asm helpers are not the right granularity here.

## Optimization opportunities spotted

These are the only realistic next knobs I see from the current asm:

1. **Whole-kernel AArch64 asm (not helper-level asm)**
   - There are several `cset` + `cinc` carry extractions.
   - If we pursue arch-specific optimization, it should be a single end-to-end kernel so NZCV stays live across the whole sequence.
   - This is still the main remaining codegen-level opportunity, but with higher implementation complexity.

2. **Register-pressure-aware scheduling for fold boundaries**
   - The kernel is short already; gains here are likely small.
   - Might improve by interleaving one fold multiply earlier, but LLVM already does a reasonable schedule.

3. **Target-specific build flags**
   - Ensure release benches use `target-cpu=native` for final local tuning runs.
   - This can help instruction selection/scheduling but is environment-specific.

No obvious algorithmic simplification is left at Rust level without moving to explicit architecture-specific code.
