# Verifier direct setup scan

| Field | Value |
| --- | --- |
| Author(s) | Quang Dao |
| Created | 2026-11-04 |
| Status | implemented |
| PR | [#327](https://github.com/LayerZero-Labs/akita/pull/327) |
| Supersedes | |
| Superseded-by | |
| Book-chapter | book/src/how/verifying/setup_contribution.md |
| Scope | Byte-preserving speedups of the direct setup-contribution scan (`relation_setup_scan`) |

## Summary

Under `SetupContributionMode::Direct`, every ordinary fold's Stage-2 relation
check evaluates the setup matrix's contribution to the relation weight by
scanning a prefix of the expanded setup. On the CI profile benchmarks
(`.github/workflows/profile-bench.yml`) this scan is the largest phase of
single-threaded verification: 44–77% of it on an Apple M4 Max at the base
commit. This note records what the scan computes, a cost model, measured
spans, and the speedups that keep every transcript, schedule, setup, and
proof byte unchanged.

## What the scan computes

A fold's plan (`SetupContributionPlan`) projects every group's A, B, and D
setup roles onto one base ring dimension `d` (64 or 128 in the profiles
measured here). Base ring `b` of the shared matrix, for `b < required`,
carries coefficients `c_{b,0..d}`. The scan returns

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
In `onehot_fp128` at 33 variables the six folds read 110,592 rings against
65,536 distinct ones.

For fp128 (`E = F = Prime128OffsetA7F7`) a coefficient costs one
128 × 128-bit schoolbook product accumulated into an unreduced
`Fp128ProductAccum`. The `d` coefficient products per ring are a lower bound
for a uniformly random setup: every coefficient carries an independent
weight `w(b)·π_j`, so contracting the tensor structure of `w` or `π` cannot
reduce the count below one wide product per coefficient.

## Measured profile

The thirteen CI profiles run in seven shards. The `profile` example reports
each verification's relation phases (`setup_scan`, `structured_groups`,
`coefficient_functional_preparation`, `quotient_tail`) single- and
multi-threaded; `AKITA_PROFILE_SPAN_CLOSES=1` adds one `relation_setup_scan`
span per fold with its geometry.

Local figures come from an Apple M4 Max (12 performance and 4 efficiency
cores), shared; the one-minute load is recorded with every measurement.
Every figure is the median of three interleaved runs of the base and the
candidate binaries, built with `parallel,profile-ci,transcript-keccak`.

Fold geometry and single-threaded scan spans at the base commit
(`b05545e69`), M4:

| Profile | Fold | Mode | `d` | Rings | A ratio | Scan, one thread (ms) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `onehot_fp128`, nv 33 | 0 | quotient lift | 128 | 65,536 | 4 | 12.35 |
| | 1 | quotient lift | 64 | 24,576 | 2 | 2.46 |
| | 2–5 | reduced | 64 | 12,288 + 4,096 + 2 × 2,048 | 1 | 2.01 |
| `multi_chunk_w2r2`, nv 29 | 0 | quotient lift | 128 | 16,384 | 4 | 3.05 |
| | 1 | quotient lift | 64 | 24,576 | 1 | 2.33 |
| | 2–5 | reduced | 64 | 10,240 + 3 × 2,048 | 1 | 1.64 |
| `multi_chunk_w8r2`, nv 29 | 0 | quotient lift | 64 | 32,768 | 4 | 3.76 |
| | 1 | quotient lift | 64 | 14,336 | 1 | 1.63 |
| | 2–6 | reduced | 64 | 24,576 + 6,665 + 5,120 + 2 × 2,048 | 1 | 3.95 |

In these three profiles and in `dense_fp128`, every plan has one group whose
segments cover every base ring below `required`, so the scan never skips a
ring.

### The CI runners differ

The profile-benchmark shards do not run on one CPU model. In run
`41858592471`, the `7-distributed` shard (the three `multi_chunk` profiles)
ran on an AMD EPYC 9V74 (Zen 4), and every other shard on an Intel Xeon
6973P-C. The same base scan code costs very different amounts on the two:
the `onehot_fp128` nv 33 quotient-lift folds took 74 ms single-threaded on
the Xeon 6 runner, against about 34 ms on Zen 4 runners in other runs of the
same code. Comparisons are therefore valid only within one runner model, and
a change can win on one runner and lose on the other.

## Changes

All changes keep the scan's exact field value: they reorder exact field
sums, and delayed reduction is used only where the `jolt-field` contract
declares it exact (`Unreduced::SUM_IS_EXACT`).

### Slice-wise weight assembly

Each job of `SETUP_SCAN_JOB_RINGS = 2048` base rings builds its ring weights
segment by segment, one contiguous pass per role (and per B slice term),
forming a projected role's weighted equality value once per role ring rather
than per base ring. One job evaluator (`evaluate_weighted_scan`) serves the
single-group, fused multi-group, and shared-functional reduced scans; the
per-ring role dispatch and the separate fused-group scan are gone.

A job assembles its weights in blocks of `WEIGHT_BLOCK_RINGS = 128` rings.
The block's weight slots live on the stack and stay in the L1 cache while
every role pass reads and writes them. The job allocates nothing. A slot's
size depends on the outer sum below: a canonical field element where every
product is reduced (16 bytes at fp128, so a block takes 2 KiB), or a 64-byte
delayed product sum for the quartic fp32 and quadratic fp64 extensions
(8 KiB a block).

### Per-ring contraction

After a block's weights are assembled, the job adds `w(b)·⟨c_b, π⟩` for
every nonzero-weight ring into one product sum (`ProductSum`).
`⟨c_b, π⟩` is `MulBaseUnreduced::dot_base`, whose base-field products are
summed exactly within that function's own capacity contract.

The outer sum is delayed, and reduced once per job, only where the field
declares `SUM_IS_EXACT`: in the pinned `jolt-field`, the quartic fp32 and
quadratic fp64 extensions. Every other field, including fp128 and therefore
every fp128 CI profile, uses `CanonicalProductSum`, which reduces each
product.

The delayed sums are exact. Each one adds products of reduced field
elements, and the two delayed accumulators document a headroom of at least
`2^61` products. Every accumulator stays far below that:

- **Weight slot.** A slot collects, for each group whose segment covers its
  ring, one D product, one product per B slice term, and one A product: at
  most `groups × (2 + B slice terms)` products. Segments of one group are
  disjoint, so each group contributes at most once.
- **Job sum.** `evaluate_weighted_scan` adds one product per ring of its job,
  at most `SETUP_SCAN_JOB_RINGS`. This covers the single-group, fused, and
  shared-functional reduced scans.
- **General reduced path.** `evaluate_groups_reduced_with`, which handles
  mixed dimensions or distinct functionals, keeps one scalar per functional
  class with the same per-ring bound as a weight slot, and a job term that
  adds at most one product per class per ring: at most
  `SETUP_SCAN_JOB_RINGS × classes`, with `classes ≤ 3 × groups`.

Group, B slice term, and class counts are lengths of in-memory arrays, so
every bound above is many orders of magnitude below `2^61`.

### Rejected: column-first contraction

The contraction can also run column first, `Σ_j π_j · Σ_b w(b)·c_{b,j}`,
accumulating each column as unreduced `E × F` products over tiles of four
rings and applying `π` once per column. Against a per-ring dot product this
saves one reduction and one `E × E` multiplication per ring. It is what the
branch first shipped, and it is the cause of the `7-distributed` shard's
regression.

- **M4.** Column-first is 5–11% faster than the per-ring contraction in the
  single-threaded scan of every CI profile, for every field.
- **Zen 4, CI.** On the `7-distributed` shard the `multi_chunk_w2r2`
  single-threaded quotient-lift scan went from 12.6 ms at the base commit to
  14.3–15.6 ms, and the reduced folds from 3.71 to 3.96–4.19 ms. The same
  run made the Xeon 6 shards faster, which hid the problem in the other ten
  profiles.
- **Zen 4, standalone fold.** A reproduction of one fp128 identity fold
  (`d = 64`, 10,240 rings, D, B, and A roles) on an AMD EPYC 9554P measured
  the base scan at 1.94 ms. Column-first in tiles of 2, 4, 8, and 16 rings
  measured 2.64–3.38 ms. The per-ring contraction measured 2.25 ms with heap
  weight slots and 2.08 ms with the 128-ring stack blocks.
- **Zen 4, profile.** The `profile` example built with
  `-C target-cpu=x86-64-v3` on the same EPYC 9554P (a shared 4-vCPU virtual
  machine at load 7–12, so these are minima over five runs) gave these sums of
  the single-threaded `relation_setup_scan` spans:

  | Profile | Base (ms) | Column-first (ms) | Per-ring (ms) |
  | --- | ---: | ---: | ---: |
  | `multi_chunk_w2r2`, nv 29 | 17.1 | 31.7 | 16.9 |
  | `multi_chunk_w4r2`, nv 29 | 15.2 | 39.2 | 14.7 |
  | `multi_chunk_w8r2`, nv 29 | 21.1 | 35.3 | 25.4 |

  The W8R2 row ran at the highest load; an earlier three-run pass at load
  10 gave medians of 29.1, 46.4, and 25.9 ms.

`llvm-mca` predicts equal throughput per product for the two inner loops on
`znver4` and `graniterapids`, so the x86 cost is not in the multiply chain;
the cause was not isolated further. The Xeon 6 runner was not measured
outside CI.

## Results

Every figure is the median of three interleaved runs of the base binary
(`b05545e69`) and the final binary on the M4, with the one-minute load range
over those runs. Single-threaded figures repeat within a few percent.
Parallel figures on this shared machine vary by up to 2× at loads above
about 7, so treat their differences as indicative only.

| Profile | Verify, parallel (ms) | Verify, one thread (ms) | Scan, one thread (ms) | Load |
| --- | --- | --- | --- | --- |
| `dense_fp32` 30:1 | 7.5 → 6.8 | 17.6 → 14.7 | 8.99 → 6.28 | 4.5–9.3 |
| `onehot_fp32` 31:1 | 6.7 → 6.5 | 13.5 → 11.7 | 7.29 → 5.37 | 6.5–10.0 |
| `dense_fp64` 29:1 | 5.7 → 5.4 | 12.7 → 10.6 | 8.06 → 6.04 | 9.0–10.4 |
| `onehot_fp64` 32:1 | 5.6 → 5.3 | 14.4 → 12.2 | 9.06 → 6.86 | 3.4–7.2 |
| `dense_fp128` 28:1 | 5.6 → 5.3 | 17.2 → 16.5 | 12.00 → 11.48 | 5.3–9.2 |
| `onehot_fp128` 33:1 direct | 6.1 → 6.1 | 21.7 → 21.1 | 16.61 → 16.02 | 7.2–26.4 |
| `onehot_fp128` 33:1 recursive | 5.6 → 5.5 | 11.4 → 11.0 | 7.45 → 6.98 | 9.2–19.1 |
| `onehot_fp128_multi_group` 31:4 direct | 6.0 → 5.8 | 17.6 → 17.0 | 11.76 → 11.35 | 7.9–11.9 |
| `onehot_fp128_multi_group_recursive` 31:4 | 6.2 → 6.1 | 12.2 → 11.9 | 7.55 → 7.24 | 5.2–11.1 |
| `…_multi_group_recursive_multi_chunk_w8r2` 29:4 | 8.6 → 8.5 | 14.3 → 14.1 | 6.31 → 6.00 | 6.9–9.3 |
| `onehot_fp128_multi_chunk_w2r2` 29:1 | 4.8 → 4.6 | 10.5 → 10.3 | 7.11 → 7.01 | 9.5–17.2 |
| `onehot_fp128_multi_chunk_w4r2` 29:1 | 5.8 → 6.2 | 12.0 → 11.8 | 6.99 → 6.76 | 11.7–13.8 |
| `onehot_fp128_multi_chunk_w8r2` 29:1 | 6.6 → 6.5 | 15.9 → 16.0 | 9.57 → 9.25 | 12.7–14.9 |

The single-threaded scan is 1–30% faster on every profile; the fp32 and
fp64 profiles gain most. Against the column-first contraction the final scan
is 5–11% slower on the M4 and, on Zen 4, about as fast as the base instead
of 1.7–2.6 times slower.

## What remains

- **Confirmation on the CI runners.** The profile benchmark on
  `0ff08a8a9` (run 41953778461) settles it: single-threaded verification
  is faster on all 13 profiles, by 1.6–7.7%, including the three
  `multi_chunk` profiles on the Zen 4 shard (−1.6%, −3.7%, −3.9%).
  Multi-threaded differences are within ±8% and mixed.
- **The M4's 5–11%.** Keeping the column-first contraction on `aarch64`
  alone would recover it, at the cost of a second contraction path selected
  by target architecture. That is worth doing only with measurements on more
  than one Arm core.
- **A faster widening fp128 dot kernel.** The `E × F` product dominates. A
  lazier carry layout in `Fp128ProductAccum`, or a NEON kernel, belongs in
  `jolt-field`; every caller of `dot_base` would benefit.
- **Tensor-structured B contraction.** On quotient-lift folds with several
  commitment slices every physical B ring sums one term per slice.
  Contracting those terms through the tensor structure of `t` would remove
  most of that weight work. It does not remove any coefficient product.
- **Fusing the folds' scans.** The folds read overlapping prefixes of one
  matrix: 110,592 ring reads over 65,536 distinct rings in `onehot_fp128` at
  nv 33. One pass that serves every fold would save memory traffic and
  scheduling but no multiplications. It needs every fold's weights and
  functional before the first fold's check, which changes the verifier's
  control flow.
- **Parallel balance.** Folds run in sequence, and the smallest reduced
  folds have one job each, so they run serially. Smaller jobs did not give a
  parallel gain distinguishable from noise on the shared M4, and each job's
  final reduction makes them cost single-threaded time.
- **Weight preparation.** `DirectScan::new` (span `relation_setup_weights`)
  is a visible share of the per-fold setup cost. It is outside the scan.

A protocol change could cut further. Each fold now evaluates its own linear
functional of the setup. If the folds' setup claims were deferred and batched
with a random linear combination into one claim, the verifier would scan the
largest prefix once. Alternatively, the verifier could check the claims
against a commitment to the setup with an opening proof, and stop scanning
the setup. Both change the transcript and what the verifier checks, so
neither is implemented here.
