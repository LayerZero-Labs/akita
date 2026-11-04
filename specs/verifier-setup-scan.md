# Verifier direct setup scan

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-04 |
| Status | active |
| Scope | Byte-preserving speedups of the direct setup-contribution scan (`relation_setup_scan`) |

## Summary

Under `SetupContributionMode::Direct`, every ordinary fold's Stage-2 relation
check evaluates the setup matrix's contribution to the relation weight by
scanning a prefix of the expanded setup. In binary-source verification at
N = 2^22 the six ordinary suffix folds spend about 10 ms single-threaded and
2.7 ms parallel in this scan, the largest verifier cost after the binary root.
This note records what the scan computes, a cost model, measured spans, and
the speedups that keep every transcript, schedule, setup, and proof byte
unchanged.

## What the scan computes

A fold's plan (`SetupContributionPlan`) projects every group's A, B, and D
setup roles onto one base ring dimension `d` (64 for every fold measured
here). Base ring `b` of the shared matrix, for `b < required`, carries
coefficients `c_{b,0..d}`. The scan returns

    S = Σ_{b < required} w(b) · Σ_{j < d} c_{b,j} · π_j

- `π` is the coefficient functional. For a quotient-lift fold it is the base
  powers `α^j`; a role of dimension `r·d` contributes the extra lane scale
  `α^{d·(b mod r)}`. For a reduced-evaluation fold it is the prepared reduced
  functional (`ReducedRoleCoefficientState`), which is shared by all roles
  when they have the base dimension.
- `w(b)` adds, over groups and over the D, B, and A roles whose packed
  segment covers `b`, the product of a row weight (`plan.d_weights()`,
  `PhysicalBWeightTerm::row_weight`, `a_row_weights`) with a role
  column-equality value (`DirectScanWeights::e`, `t`, `z`) and, for projected
  roles, the lane scale. A physical B ring sums one term per logical
  commitment slice that maps onto it.

`DirectScan::new` (span `relation_setup_weights`) materializes `e`, `t`, `z`
and the packed segment partition once per fold. `DirectScan::evaluate_direct`
(span `relation_setup_scan`) is the scan.

## Cost model

Per base ring the scan performs `d` widening `E × F` multiply-accumulates for
`⟨c_b, π⟩`, one reduction, one `E × E` multiplication by `w(b)`, and the
multiplications that form `w(b)`: one per active role and B term, plus one
lane-scale product per projected base ring.

The setup is materialized once in `AkitaExpandedSetup`; the scan derives no
setup entry. Each fold reads the prefix `[0, required)` of the same matrix.
Over the six folds at 2^22 that is 97,688 ring reads (6.25 M field
coefficients, 100 MB) against 52,632 distinct rings.

For fp128 (`E = F = Prime128Offset275`) a coefficient costs one 128 × 128-bit
schoolbook product, eight 64-bit multiply instructions, accumulated into
four 128-bit slots. Single-threaded throughput is 1.3 ns per coefficient
(12 GB/s of setup), which is compute-bound and close to the two-multiplier
port limit of the core. The `d` coefficient products per ring are a lower
bound for a uniformly random setup: every coefficient carries an independent
weight `w(b)·π_j`, so contracting the tensor structure of `w` or `π` cannot
reduce the count below one wide product per coefficient.

## Measured profile

Machine: Apple M4 Max (12 performance and 4 efficiency cores), shared;
the one-minute load is recorded with every measurement. Binary-source runs use
`scripts/binary-source-full-size.sh` fixtures (proof digests match
`scripts/binary-source-full-size-digests.tsv`) and time the verify phase only.
The ordinary schedule is the `profile` example with `onehot_fp128` at
`AKITA_NUM_VARS=32`, which reports the setup-scan phase single- and
multi-threaded. Every figure is the median of three runs at the base commit
`84c2176e8`.

| Configuration | Parallel (ms) | One thread (ms) | Load |
| --- | ---: | ---: | --- |
| binary source 2^22 `word-prime` | 2.71 | 10.04 | 8.5–9.6 |
| binary source 2^22 `word` | 2.81 | 9.94 | 8.1–8.5 |
| binary source 2^23 `word` | 4.27 | 17.77 | 8.1–9.8 |
| binary source 2^23 `word-prime` | 4.08 | 17.08 | 9.6–9.9 |
| ordinary `onehot_fp128`, nv 32 | 2.44 | 10.85 | 16–19 |

Fold geometry at 2^22 `word-prime` (all `d = 64`):

| Fold | Mode | Groups | Rings | Scan, one thread (ms) | `⟨c_b, π⟩` alone (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | quotient lift, A ratio 8 | 2 (fused) | 52,632 | 5.66 | 4.42 |
| 1 | reduced | 1 | 24,576 | 2.53 | 2.07 |
| 2 | reduced | 1 | 12,288 | 1.22 | 1.00 |
| 3 | reduced | 1 | 4,096 | 0.43 | 0.33 |
| 4 | reduced | 1 | 2,048 | 0.22 | 0.18 |
| 5 | reduced | 1 | 2,048 | 0.20 | 0.17 |

The last column times a bare `dot_base` loop over the same rings. It is the
multiplication floor: 8.2 ms of the 10.2 ms. Adding the per-ring weight
multiplication through `WeightedRowBatch` raises it to 8.8 ms. The remaining
1.2–1.4 ms forms the weights: per-ring closures, `Result` plumbing, bounds
checks, and up to four reduced multiplications per ring. Every B segment at
2^22 has one term.

The ordinary schedule has two quotient-lift single-group folds (65,536 and
14,104 rings, A ratio 8) and four reduced folds. Its large fold costs 8.2 ms
against a 5.45 ms floor, because each physical B ring there sums eight
logical-slice terms.

Parallelization is a Rayon fold over jobs of `SETUP_SCAN_JOB_RINGS = 2048`
rings; folds run in sequence. Folds 3–5 at 2^22 have one or two jobs, so they
run serially. A job on an efficiency core takes about 3.5 times as long, and
the slowest job sets each fold's latency. Parallel throughput is 37 GB/s,
far below the memory system, so the parallel scan is latency- and
balance-bound rather than bandwidth-bound.

## Candidate speedups

All candidates keep the scan's exact field value: they reorder exact field
sums, and delayed reduction is used only where the `jolt-field` contract
declares it exact.

1. **Slice-wise weight assembly.** Build each job's ring weights segment by
   segment with one contiguous pass per role, forming a projected role's
   weighted equality value once per role ring rather than per base ring. One
   job evaluator then serves single-group, fused, and shared-functional
   reduced scans. Expected gain: 0.5–1 ms single-threaded at 2^22, more on the
   ordinary quotient-lift folds.
2. **Transposed contraction.** Evaluate `Σ_j π_j · Σ_b w(b)·c_{b,j}` per job,
   accumulating the inner sums as unreduced `E × F` products
   (`MulBaseUnreduced::mul_base_unreduced`, exact within the accumulator
   headroom for every implementor). This drops one reduction and one
   `E × E` multiplication per ring. Expected gain: 0.4–0.6 ms single-threaded
   at 2^22.
3. **Smaller scan jobs.** Jobs of 256–512 rings give every fold enough jobs
   to balance, and shorten the tail an efficiency core adds. Expected gain:
   0.4–1 ms parallel at 2^22, nothing single-threaded.

Not pursued here; see [What remains](#what-remains):

- a faster fp128 widening dot kernel, which belongs in `jolt-field`;
- contracting the B role's slice terms through the tensor structure of `t`;
- fusing the six folds' scans into one pass, which saves memory traffic but
  no multiplications, and which the sequential fold verification does not
  allow without restructuring.

## Results

Candidates 1 and 2 landed; candidate 3 did not. Every figure is the median of
three interleaved runs of the base binary (`84c2176e8`) and the final binary,
with the one-minute load range over those runs. Single-threaded figures
repeat within about 0.2 ms. Parallel figures on this shared machine vary by
up to 2× between runs at loads above about 7, so treat their differences as
indicative only.

| Configuration | Parallel (ms) | One thread (ms) | Load |
| --- | --- | --- | --- |
| binary source 2^22 `word-prime` | 2.53 → 2.09 | 10.03 → 8.59 | 6.7–7.9 |
| binary source 2^22 `word` | 2.76 → 2.48 | 9.96 → 8.73 | 7.6–8.1 |
| binary source 2^23 `word` | 8.77 → 5.29 | 16.99 → 14.76 | 11.8–13.2 |
| binary source 2^23 `word-prime` | 3.10 → 2.82 | 16.58 → 14.52 | 6.2–8.1 |
| ordinary `onehot_fp128`, nv 32 | 3.60 → 2.18 | 10.94 → 9.23 | 11.7–17.1 |

The single-threaded scan is 12–16% faster across all five configurations.
Per commit, at binary source 2^22 `word-prime`, one thread:

| Commit | Before (ms) | After (ms) | Load |
| --- | ---: | ---: | --- |
| slice-wise weight assembly | 10.16 | 9.57 | 8.0–13.1 |
| column-first contraction | 9.49 | 8.69 | 7.1–7.4 |

The column-first contraction stages rings in tiles. Tiles of 4, 8, and 16
rings measured 8.87, 9.10, and 10.34 ms in one interleaved run at load
10–13; the code uses 4.

Smaller jobs (candidate 3) gave no parallel gain distinguishable from noise.
In nine interleaved parallel runs at 2^22 `word-prime` (load 8–11), jobs of
2048, 256, and a fold-dependent size (about 64 jobs per fold, clamped to
256–2048) measured 2.27, 2.46, and 2.26 ms. One earlier run at load 6.5 had
favored 256-ring jobs, 1.91 against 2.27 ms. The column-first contraction
also changes the trade-off. Each job now ends with 64 reductions and 64
`E × E` multiplications, so 256-ring jobs cost 0.3 ms single-threaded (8.64
against 8.96 ms). Choosing the size from the Rayon pool's thread count
would avoid that cost; the verifier's `parallel` feature enables
`dep:rayon`, so the thread count is available under that feature. Without a
measurable parallel gain from smaller jobs, this was not pursued.
Carrying the unreduced column sums through the Rayon fold, one finish per
split, measured slower in both modes. The job size stays 2048.

## What remains

After these changes the single-threaded scan at 2^22 takes 8.6 ms, near the
8.2 ms per-ring dot floor in the fold table above. Further byte-preserving
gains need a different kernel or a different scan structure:

- **A faster widening fp128 dot kernel.** The `E × F` product dominates. A
  lazier carry layout in `Fp128ProductAccum`, or a NEON kernel, belongs in
  `jolt-field`; every caller of `mul_base_unreduced` would benefit.
- **Tensor-structured B contraction.** On the ordinary quotient-lift folds
  every physical B ring sums eight logical-slice terms. Contracting those
  terms through the tensor structure of `t` would remove most of that weight
  work. It does not remove any coefficient product.
- **Fusing the folds' scans.** The folds read overlapping prefixes of one
  matrix: 97,688 ring reads over 52,632 distinct rings at 2^22. One pass that
  serves every fold would save memory traffic and scheduling but no
  multiplications. It needs every fold's weights and functional before the
  first fold's check, which changes the verifier's control flow.
- **Parallel balance.** The parallel scan reaches about four times the
  single-threaded speed on 16 cores, because the folds run in sequence and
  the small folds have one or two jobs. A balance fix needs either fold
  fusion or a job size chosen from the thread pool (see Results).
- **Weight preparation.** `DirectScan::new` (span `relation_setup_weights`)
  is now a visible share of the per-fold setup cost. It is outside the scan.
- **The binary root.** `binary_direct_a_weights` dominates binary-source
  verification and is outside this note.

A protocol change could cut further. Each fold now evaluates its own linear
functional of the setup. If the folds' setup claims were deferred and batched
with a random linear combination into one claim, the verifier would scan the
largest prefix once, 52,632 rings instead of 97,688 at 2^22. Each scan's
`d` products per ring would also become shared. Alternatively, the verifier
could check the claims against a commitment to the setup with an opening
proof, and stop scanning the setup. Both change the transcript and what the
verifier checks, so neither is implemented here.
