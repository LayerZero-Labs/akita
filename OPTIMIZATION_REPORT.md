# Hachi Prover Optimization Report

**Target**: `HACHI_MODE=onehot HACHI_NUM_VARS=32`
**Machine**: Apple Silicon (AArch64), NEON SIMD, ~10 threads (default Rayon)
**Date**: 2026-03-20

---

## 1. `mat_vec_mul_ntt_digits_i8` (commit_w level 0)

### Summary

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| `mat_vec_mul_ntt_digits_i8` L0 | **273 ms** | **23 ms** | **11.9×** |
| `commit_w` L0 total | 276 ms | 26 ms | 10.6× |

### Root Cause

Two structural bottlenecks:

1. **Parallelism starvation**: Column tiling created only 5 tiles for Rayon across 10 threads.
2. **Iteration overhead on zeros**: 75% of digit planes are zero (one-hot), but all 256 blocks × 3276 planes per tile were iterated.

### Fix: Block-Parallel Fast Path

When `n_a <= 2` and `num_blocks >= 16`, parallelize over blocks instead of column tiles.

- **256-way parallelism** (blocks) vs 5-way (tiles)
- Each block is independent — no shared accumulator, no reduction
- Near-perfect linear scaling: 1T=272ms → 10T=23ms

### What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| Precomputed Unit NTT Table | No change | NEON NTT already ~240 instructions; table lookup + clone cost matches NTT cost |
| Pre-scan nonzero indices | Slightly slower | Vec allocation + extra scan pass outweighs branch-predictor-friendly zero-skip |
| Lazy Montgomery reduction | Not viable | i32 limbs with ~2^30 primes overflow after 2 unreduced accumulations |

### File Changed

`src/protocol/commitment/utils/linear.rs` — Added `mat_vec_mul_digits_i8_block_parallel`.

---

## 2. `OneHotPoly::decompose_fold` (compute_z_pre)

### Summary

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| `onehot_sparse_accumulate` | **260 ms** | **23 ms** | **11.3×** |
| `OneHotPoly::decompose_fold` total | 262 ms | 23.5 ms | 11.1× |

### Root Cause

The function uses `cfg_fold_reduce!` over `num_blocks = 4096` blocks, accumulating into
`Vec<[i32; D]>` of size `inner_width = 16384` (4 MB per accumulator). Three problems:

1. **Massive per-task accumulators**: Rayon's `fold` creates one 4 MB identity per leaf task.
   With ~20 leaves: 80 MB of allocations + zero-filling.
2. **Expensive tree reduce**: Merging 4 MB vectors element-by-element through O(log N) tree levels.
3. **Branch-heavy inner loop**: `accum_onehot_coeff` has an unpredictable branch for negacyclic
   wrap (`target < D`), causing ~50% misprediction on the 42-iteration sparse challenge loop.

### Critical Dimensions (level 0)

| Parameter | Value |
|-----------|-------|
| `inner_width` | 16,384 |
| `num_blocks` | 4,096 |
| `block_len` | 16,384 |
| `num_digits` | 1 |
| `D` (ring degree) | 64 |
| `total_entries` | 16,777,216 (2^24) |
| `avg_entries/block` | 4,096 |
| Challenge weight | 42 (SplitRing, half_weight=21) |

### Fix: Rotation Table + Position-Partitioned Parallelism

Two complementary optimizations:

#### A. Precomputed Rotation Table (eliminates branches)

Instead of calling `accum_onehot_coeff` per entry (42 iterations with unpredictable branch),
precompute a dense `[i32; D] × D` rotation table per challenge:

```
table[ci][t] = coefficients of challenge * X^ci  in  Z[X]/(X^D+1)
```

The table is 16 KB (fits entirely in L1 cache). Each entry accumulation becomes a
**branchless 64-element vector addition** (`for k in 0..D { dst[k] += rot[k]; }`),
which LLVM auto-vectorizes to 16 NEON `vaddq_s32` instructions.

**Per-entry cost**: 42 branchy scalar ops → 16 NEON vector ops.

#### B. Dense Rotation via Negacyclic Shift (fast table fill)

Initial table fill used the same scatter approach (D × 42 = 2688 ops, branch-heavy).
Replaced with a two-phase approach:

1. **Scatter** the sparse challenge into a dense `[i32; D]` buffer (42 writes).
2. **Derive rotations** via `copy_from_slice` + negate (memcpy-based, NEON-friendly).

Table fill cost: **9 ms → 1 ms** per thread.

#### C. Position-Partitioned Parallelism (eliminates reduce)

Instead of parallelizing over blocks (requiring per-thread 4 MB accumulators + reduce),
partition output positions across threads:

- Each thread owns a contiguous range of ~1,638 output positions (420 KB accumulator).
- For each block, binary search on sorted entries to find the relevant range.
- No reduce phase needed — results are concatenated directly.

**Benefits**:
- Accumulator fits in L2 cache (420 KB vs 4 MB)
- Zero reduce overhead
- 16,384-way parallelism over output positions

### Optimization Progression

| Step | `onehot_sparse_accumulate` | Speedup vs baseline |
|------|---------------------------|---------------------|
| Baseline (fold-reduce, branchy scatter) | 260 ms | 1.0× |
| + Rotation table + explicit chunking | 29 ms | 9.0× |
| + Position-partitioned (no reduce) | 29 ms | 9.0× (same, smaller accumulators) |
| + Dense rotation via negacyclic shift | **23 ms** | **11.3×** |

### What Didn't Help

| Approach | Result | Why |
|----------|--------|-----|
| Branchless `accum_onehot_coeff` (no table) | ~140 ms est. | 42 scalar ops/entry vs 16 NEON ops/entry with table |
| Precompute all 4096 tables in shared array | Slower | 64 MB table array causes L2/DRAM misses; local table computation stays in L1 |
| Atomic scatter to shared output | ~3.4 s est. | 16M × 42 atomic fetch_add at ~5 ns each |
| Mutex per output position | ~400 ms est. | 16M lock/unlock at ~25 ns each |
| CSR inverted index | ~140 ms est. | 48 MB index build + 16M random challenge accesses |

### File Changed

`src/protocol/hachi_poly_ops/mod.rs` — Replaced `cfg_fold_reduce!` in
`decompose_fold_sparse_onehot` with `sparse_onehot_accumulate` + `fill_rotated_challenge`.

---

---

## 3. `ring_switch_build_w` sub-spans (compute_r_split_eq + build_w_coeffs)

### Summary

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| `challenge_fold_row` | **19.6 ms** | **2.2 ms** | **8.9×** |
| `A_row` | **20.5 ms** | **2.3 ms** | **8.9×** |
| `build_w_coeffs` | **14.6 ms** | **6.8 ms** | **2.1×** |
| `compute_r_split_eq` total | **70.3 ms** | **32 ms** | **2.2×** |
| `ring_switch_build_w` total | **84.9 ms** | **39 ms** | **2.2×** |

### Root Causes

1. **`challenge_fold_row` / `A_row`**: Sequential accumulation of 4096 blocks into a single `[F; D]`
   quotient buffer via `add_sparse_ring_product_high_half`. Each call iterates all D=64 coefficients
   but only `pos` of them contribute (50% wasted iteration due to `if deg >= D` branch).
2. **`build_w_coeffs`**: Sequential decomposition of 16,384 `z_pre_centered` elements via
   `balanced_decompose_centered_i32_i8_into`.
3. **D/B NTT rows**: Sequential execution of independent NTT matrix-vector products.

### Fixes

#### A. Parallel fold-reduce for challenge_fold_row / A_row

Replaced sequential accumulation loops with `cfg_fold_reduce!` over blocks.
Each leaf task gets a `Vec<F>` of size D (512 bytes) — trivially small accumulators.
With 4096-way parallelism, gives ~10× speedup.

#### B. Branchless inner loop (`add_sparse_ring_product_high_half`)

Changed inner loop from iterating all D coefficients with branch to starting at `D - pos`:
```
// Before: for (s, &r_s) in rc.iter().enumerate() { if p+s >= D { ... } }
// After:  for s in (D - p)..D { quotient[p+s-D] += c * rc[s]; }
```
Eliminates branch, halves iteration count on average.

#### C. Parallel decomposition in `build_w_coeffs`

Replaced sequential `z_pre_centered` and `r` decomposition loops with `cfg_iter!`-based
parallel decomposition, then sequential flat copy to output.

#### D. Concurrent D/B/A NTT rows via `rayon::join`

All three NTT computations (D_rows, B_rows, A_rows) are independent. Using nested
`rayon::join` overlaps their execution, saving ~7ms of sequential scheduling overhead.

### Files Changed

- `src/protocol/quadratic_equation.rs` — Parallel `challenge_fold_row` / `A_row`, concurrent NTTs
- `src/protocol/ring_switch.rs` — Parallel `build_w_coeffs` decomposition

---

## 4. `decompose_w_hat`

### Summary

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| `decompose_w_hat` L0 | **32.3 ms** | **4.6 ms** | **7.0×** |

### Root Cause

Sequential iteration over 4096 ring elements, each calling
`balanced_decompose_pow2_i8(43, 3)` — pure i128 arithmetic with no data dependencies
between elements.

### Fix

One-line change: `pre_folded.iter()` → `cfg_iter!(pre_folded)`, enabling Rayon
parallelism over independent ring element decompositions.

### File Changed

`src/protocol/quadratic_equation.rs` — `cfg_iter!` in `decompose_w_hat` span.

---

## 5. `BalancedDigitPoly::decompose_fold` (compute_z_pre, levels ≥ 1)

### Summary

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Level 1 `decompose_fold` | **17.4 ms** | **18-21 ms** | ~1× (neutral) |

### Analysis

Level 1 dimensions: `inner_width=2048`, `num_blocks=256`, `block_len=2048`, `num_digits=1`, `D=64`.
Applied position-partitioned parallelism (same technique as OneHotPoly). However, the original
`cfg_fold_reduce!` already had sufficient parallelism (256 blocks) with modest 512KB accumulators,
so the position-partitioned approach was neutral in performance but improved peak memory usage.

### Fix

Replaced `cfg_fold_reduce!` with position-partitioned `balanced_digit_decompose_fold_partitioned`.
Also fixed an overflow bug when `num_threads > inner_width`.

### File Changed

`src/protocol/hachi_poly_ops/mod.rs`

---

## 6. `QuadraticEquation::new_verifier` (single-threaded)

### Summary

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Level 0 `new_verifier` | **6.5 ms** | **6.2 ms** | **1.05×** |

### Analysis

99% of `new_verifier` time is in `sample_sparse_challenges`, which samples 4096
SplitRing challenges. Each challenge requires:
- 4 `append_bytes` calls (12 hash state updates)
- 1 `challenge_bytes` call producing 274 bytes (5 Blake2b512 chain operations)

The hash computation is the fundamental single-threaded bottleneck.

### Fix

Batched the 4 per-challenge `append_bytes` calls into a single call with a pre-allocated
buffer, reducing hash update overhead by ~25% per challenge. Also enabled buffer reuse
across all 4096 challenges.

### What Wasn't Attempted (protocol change required)

| Approach | Why Not |
|----------|---------|
| PRG-seeded expansion (1 hash → ChaCha expand all 4096 challenges) | Changes Fiat-Shamir protocol |
| Parallel challenge sampling | User requirement: keep verifier single-threaded |
| Reduce challenge bytes (shorter challenges) | Changes security parameters |

### File Changed

`src/protocol/challenges/sparse.rs` — Buffer reuse and batched absorb in `sample_one`.

---

## 7. Other fixes

- **`inner_ajtai_onehot_wide` overflow**: Fixed `num_threads > num_blocks` causing
  subtraction overflow in `commit_inner` for OneHotPoly.

---

## 8. `OneHotPoly::commit_inner_witness` (initial commit, A*s computation)

### Summary

| nv | Before (ms) | After (ms) | Speedup |
|----|------------|------------|---------|
| 20 | 0.37 | 0.55 | ~1× (fallback path) |
| 24 | 1.20 | 1.32 | ~1× (fallback path) |
| 28 | 10.9 | 10.9 | ~1× (fallback path) |
| 32 | **258** | **105-119** | **2.2-2.5×** |
| 36 | **6,990** | **2,890-3,660** | **1.9-2.4×** |

### Root Cause

The original block-by-block approach processes each block independently, where each
block's `inner_ajtai_onehot_wide` reads ~4MB from the shared 16MB A matrix row (25%
column density at D=64, ONEHOT_K=256). Three problems:

1. **Redundant A-column widening**: The same A column is widened (Fp128 → Fp128x8i32,
   512 ops per ring element) separately by every block that references it. With ~100
   blocks per thread referencing each column, that's ~100× redundant widening.
2. **Super-linear scaling**: Total L3 bandwidth grows as `num_blocks × 4MB`. At nv=32
   (4096 blocks × 4MB = 16GB aggregate), L3 is saturated. At nv=36 (65K blocks ×
   4MB = 256GB), it's catastrophic.
3. **Random A access pattern**: Within each block, entries are sorted by pos_in_block
   but only 25% of columns are accessed — sparse random reads defeat simple prefetching
   when the A row doesn't fit in L2.

### Fix: Column-Sweep with L2-Tiled Block Accumulators

Replaced the block-by-block iteration with a two-level tiled column-sweep:

**Outer level (parallelism)**: Rayon threads partition blocks evenly.

**Inner level (cache locality)**: Within each thread, blocks are processed in tiles of
~1024 blocks (2MB of accumulators fits in L2). For each tile:

1. **Bucket entries by A-column** — O(entries_per_tile) scatter into `col_entries[col]`
   vector of `(block_local_idx, coeff_idx)` pairs.
2. **Sequential column sweep** — Iterate A columns in order. For each non-empty column,
   widen the A ring element exactly once, then scatter-accumulate into all referencing
   blocks' `WideCyclotomicRing` accumulators via `shift_accumulate_into`.
3. **Reduce** — Convert wide accumulators back to `CyclotomicRing`.

**Fallback**: When `blocks_per_thread ≤ 128`, the column-sweep bucketing overhead
exceeds its benefit; fall back to the original `inner_ajtai_onehot_wide` path.

### Why It Works

| Property | Block-by-block (old) | Column-sweep (new) |
|----------|---------------------|-------------------|
| A column widenings per thread | blocks × entries/block = ~1.68M | columns = ~16K (**102× less**) |
| A data read per thread | ~1.6GB from L3 | ~16MB from L3 (once per tile) |
| Accumulator locality | 840KB–13MB (depends on blocks) | Tiled to ≤ 2MB (always L2) |
| Entries per column (amortized widen) | 1 | ~100–1600 |

The key insight: A column reads are sequential and dense (every column in order),
maximizing hardware prefetcher effectiveness. Accumulator writes are random but
L2-resident within each tile. The combination eliminates the memory bandwidth
bottleneck that dominated the original approach.

### What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| Pre-widen A matrix into `a_wide` array | **+35% regression** | Doubled memory footprint (2048B vs 1024B per element); memory bandwidth, not compute, is the bottleneck |
| Branchless `shift_accumulate_into` (split loops) | **+13% regression** | LLVM already optimizes the branch well; manual splitting prevents vectorization |
| NTT-domain approach (pointwise mul-acc) | Not implemented | 5 CRT primes × 64 slots = 320 Montgomery mul-acc per entry (560 NEON ops) vs 128 NEON ops for coefficient-domain shift — 4.4× more compute |
| Double-tiling (A columns + blocks) | Neutral | Hardware prefetcher already handles sequential A access; column tiling adds loop overhead without benefit |
| Tile size sweep (2^19 to 2^23) | 2^21 optimal | Smaller tiles increase col_entries rebuild overhead; larger tiles cause L2 thrashing on accumulators |

### Scaling Analysis

The commit_inner_witness cost should scale roughly linearly with num_blocks (and thus
with 2^nv). Before/after scaling factors:

| nv delta | Blocks ratio | Old scaling | New scaling |
|----------|-------------|-------------|-------------|
| 28→32 | 16× | 24× (super-linear) | 10× (sub-linear, amortization) |
| 32→36 | 16× | 27× (super-linear) | 28× (linear + L3 contention) |

The super-linear scaling at nv=36 is reduced from ~2.4× to ~1.7× overhead vs ideal
linear, primarily from L3 contention when many tiles re-read the 16MB A row.

### Files Changed

`src/protocol/hachi_poly_ops/mod.rs` — New `onehot_column_sweep_ajtai` function,
updated `commit_inner` and `commit_inner_witness` methods on `OneHotPoly`.

---

## Current Bottleneck Decomposition (prove, onehot nv32)

After all optimizations (rounds 1-6):

| Component | Time (ms) | % of prove | vs Previous |
|-----------|-----------|------------|-------------|
| `ring_switch_build_w` L0 | 39 | 8% | 87→39 (2.2×) |
| `decompose_w_hat` L0 | 5 | 1% | 33→5 (6.6×) |
| **`OneHotPoly::decompose_fold`** L0 | **24** | **5%** | 260→24 (10.8×) |
| **`commit_w` L0** | **26** | **5%** | 276→26 (10.6×) |
| `compute_v` (mat_vec_mul_ntt_single_i8) | 14 | 3% | unchanged |
| `stage1_sumcheck` (all levels) | ~108 | 22% | unchanged |
| `ring_switch_finalize` L0 | 13 | 3% | unchanged |
| `compute_z_pre` L1 (BalancedDigitPoly) | 20 | 4% | 17→20 (neutral) |
| Other (L2-L5, labrador tail) | ~230 | 47% | reduced slightly |
| **Total prove** | **~490** | **100%** | 564→490 (1.15×) |

### Cumulative Improvement from All Optimization Rounds

| Metric | Original (pre-opt) | After Round 1-2 | After Round 3-6 | Total Speedup |
|--------|-------------------|------------------|-----------------|---------------|
| `commit_w` L0 | 276 ms | 26 ms | 26 ms | **10.6×** |
| `OneHotPoly::decompose_fold` | 260 ms | 24 ms | 24 ms | **10.8×** |
| `ring_switch_build_w` L0 | 87 ms | 87 ms | 39 ms | **2.2×** |
| `decompose_w_hat` L0 | 33 ms | 33 ms | 5 ms | **6.6×** |
| **Total prove** | **~850 ms** | **~564 ms** | **~490 ms** | **~1.7×** |

## Overall Commit Performance (nv=32, onehot)

| Component | Before (ms) | After (ms) | Speedup |
|-----------|------------|------------|---------|
| `commit_inner_witness` (A*s) | 258 | 105-119 | **2.2-2.5×** |
| `mat_vec_mul_ntt_single_i8` (B) | ~14 | ~13 | ~1× |
| **Total commit** | **~310** | **~125** | **~2.5×** |

## Recommendations for Further Optimization

### Commitment path
1. **`commit_inner_witness` at large nv (36+)** — The column-sweep tiling still
   re-reads the 16MB A row once per tile (~7 tiles for nv=36). A shared read-only
   prefetch of A into L3 before the parallel sweep could reduce L3 miss latency.
2. **`mat_vec_mul_ntt_single_i8` (13ms)** — Already block-parallelized; NTT is the
   bottleneck. Batched NTT or different CRT prime selection could help.

### Proving path
3. **`stage1_sumcheck` (108ms, 22%)** — The `fuse_compact_to_round2` step (57ms) dominates.
   This is a large bivariate-to-univariate reduction and may benefit from better cache tiling.
4. **`compute_r_split_eq` NTT rows (32ms, 7%)** — D/B/A NTT matrix-vector multiplies are
   already parallelized internally; further gains require algorithmic improvements (e.g.,
   batched NTT, different CRT prime selection).
5. **`ring_switch_finalize` (13ms, 3%)** — Already runs `compute_m_evals_x` and
   `build_w_evals_compact` in parallel. The sequential `EqPolynomial::evals` and gadget
   setup are small fixed costs.
