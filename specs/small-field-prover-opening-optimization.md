# Spec: Small-Field Prover Opening Optimization

| Field       | Value                                      |
|-------------|--------------------------------------------|
| Author(s)   | @quangvdao                                 |
| Created     | 2026-05-17                                 |
| Status      | implemented; retrospective                 |
| PR          | #85 (`quang/sumcheck-extension-opening-opt`) |

## Summary

PR #85 optimizes the fp32 and fp64 prover paths introduced by the general-field
opening work. The main problem was not proof size or verifier logic, but the
extra prover-side cost of preparing and proving root extension-opening
reductions for small fields, especially one-hot workloads. This PR keeps the
protocol, transcript, verifier behavior, and proof bytes stable while moving
tensor-opening work to better backend boundaries, reducing repeated sparse
materialization, and adding focused benchmarks for the remaining hot paths.

## Intent

### Goal

Make small-field dense and one-hot proving materially faster by specializing
prover-side tensor opening, sparse root projection, and extension-opening
reduction at existing backend and sumcheck abstraction boundaries.

The key surfaces modified are:

- `akita-prover::AkitaPolyOps`: new tensor column-partial and sparse
  same-point linear-combination hooks.
- `akita-prover::backend::dense`: dense tensor-opening methods that stream from
  ring storage instead of materializing a flat field vector first.
- `akita-prover::backend::onehot`: one-hot tensor column partials, batched
  same-point partial preparation, sparse tensor witness linear combinations,
  cached tensor-projected sparse root polynomials, and weighted tensor-basis
  precomputation.
- `akita-prover::backend::sparse_ring`: validated packed sparse-ring
  coefficients, sorted constructors, borrowed row decomposition, and a
  position-bucketed sparse column sweep for dense tiles.
- `akita-prover::protocol::flow`: grouping root extension-opening claims by
  point, passing caller-owned input claims into the batched reduction prover,
  and using sparse tensor-factor terms for one-hot same-point batches.
- `akita-sumcheck::extension_opening_reduction`: lazy tensor equality factors,
  sparse witness sorted constructors, coefficient-hoisted sparse accumulation,
  dense round parallelization/folding, and stronger focused tests.
- `akita-field`: scalar and packed `Fp2` multiplication use the same
  three-multiply Karatsuba formula.
- `akita-pcs` benches: focused Criterion lenses for extension-opening
  reduction and one-hot root-projection commitment.

### Invariants

1. Proof bytes and verifier behavior must not change for this optimization.
   The latest small-field one-hot profiles still produce `46,000` bytes for
   `onehot_fp32_d64` and `45,904` bytes for `onehot_fp64_d32`, with verifier
   success in the profile runs.
2. Transcript and protocol semantics must not change. The PR changes how the
   prover computes witnesses, factors, and root projections, not the sequence of
   claims, challenges, or verifier checks.
3. The extension-opening tensor factor must respect the fact that coordinate
   projection is only base-field linear. The lazy tensor factor is checked
   against dense-factor round polynomials by
   `sparse_tensor_factor_matches_dense_factor_rounds`.
4. Dense tensor-opening overrides must be equivalent to the flat-table
   reference implementation. This is protected by
   `dense_tensor_opening_methods_match_flat_reference`.
5. Backend grouping by point must preserve claim order when control returns to
   the protocol flow. Same-point fast paths are optimization boundaries, not new
   public batching semantics.
6. Sparse witness constructors must validate the ordering, range, uniqueness,
   and nonzero contracts they rely on. This is covered by
   `sparse_witness_sorted_constructor_combines_without_sorting` and the
   `from_sorted_unique_entries` rejection cases.
7. Sparse-ring constructors must not trust malformed compact coefficients.
   `SparseRingCoeff` keeps fields private, and sparse-ring tests cover sorted
   rejection and packed-vs-tuple constructor equivalence.
8. The one-hot root projection cache must cache only deterministic projections
   of immutable one-hot indices, keyed by tensor width. A request for another
   valid width must compute the corresponding projection instead of reusing an
   incompatible cached value.
9. `Fp2` Karatsuba multiplication must remain algebraically identical for every
   configured non-residue. Scalar and packed multiplication tests must pass.
10. Optimization code must remain maintainable: rejected local wins that add
    duplicate representations, hidden state machines, or architecture-specific
    clutter are not part of the PR.

### Non-Goals

1. No protocol change to avoid root extension-opening reductions.
2. No proof-size change or verifier-side optimization.
3. No new public commitment/proving API variant for each dense, one-hot,
   same-point, or sparse special case.
4. No compatibility shim for old internal helper shapes; this repository does
   not promise internal backward compatibility.
5. No CRT prime-count cutover for the Q64 path. The tested three-prime Q64
   commitment path did not improve prover time enough to include here.
6. No speculative SIMD rewrite for sparse-ring or sparse-sumcheck kernels.
7. No guarantee that one-hot fp32/fp64 proving beats the matching fp128 profile
   in this PR. The PR materially reduces the gap, but the remaining gap appears
   to require a larger root extension-opening design or protocol change.

## Evaluation

### Acceptance Criteria

- [x] Dense fp32/fp64 tensor-opening preparation streams from dense ring
      storage and matches the flat reference.
- [x] One-hot root extension-opening preparation computes tensor column
      partials without repeated per-head evaluation for the profiled layouts.
- [x] Same-point one-hot root openings build one sparse tensor witness for the
      coefficient-weighted batch instead of materializing one sparse witness per
      claim and normalizing a separate linear combination.
- [x] Sparse extension-opening terms can use a lazy tensor equality factor for
      the first reduction rounds and still match dense-factor round
      polynomials.
- [x] Sparse witness and sparse-ring fast constructors validate their stated
      sorted/unique/packed contracts.
- [x] One-hot tensor-projected sparse roots are cached across commit/prove
      reuse without changing proof bytes or verifier behavior.
- [x] Sparse-ring root-projection commitment has a focused benchmark and a
      faster dense-tile column-sweep path.
- [x] `Fp2` scalar and packed multiplication use a consistent Karatsuba
      formula and pass field tests.
- [x] Focused Criterion benches exist for extension-opening reduction and
      one-hot root-projection commitment, with 10-sample bounded runs.
- [x] CI profile benchmark comments are generated successfully for the PR head.
- [x] Final GitHub CI is green on PR #85 after the spec is added.

### Testing Strategy

Focused tests added or exercised by this PR:

- `cargo test -q -p akita-field fp2_mul`
- `cargo test -q -p akita-field packed_fp2_mul`
- `cargo test -q -p akita-field ring_subfield`
- `cargo test -q -p akita-prover dense_tensor_opening_methods_match_flat_reference`
- `cargo test -q -p akita-prover tensor_column_partials`
- `cargo test -q -p akita-prover tensor_packed_sparse_linear_combination_matches_individual_witnesses`
- `cargo test -q -p akita-prover sparse_ring`
- `cargo test -q -p akita-sumcheck extension_opening_reduction`
- `cargo test -q -p akita-sumcheck sparse_tensor_factor_matches_dense_factor_rounds`

Final local validation before opening the latest PR update included:

- `cargo fmt -q`
- `cargo test -q -p akita-sumcheck extension_opening_reduction`
- `cargo bench -q -p akita-pcs --bench extension_opening_reduction --no-run`
- `cargo bench -q -p akita-pcs --bench onehot_root_projection_commit --no-run`
- `cargo clippy -q -p akita-sumcheck -p akita-pcs --benches -- -D warnings`
- `RUSTDOCFLAGS="-D warnings" cargo doc -q --workspace --all-features --no-deps`
- `git diff --check`

Additional broader validation performed during the optimization pass included
native-release tests and profiles on both local ARM/NEON and `leopard`
x86/AVX. The main profile configurations were:

```text
RUSTFLAGS="-C target-cpu=native" \
AKITA_MODE=onehot_fp32_d64 \
AKITA_NUM_VARS=30 \
AKITA_NUM_POLYS=4 \
AKITA_PROFILE_TRACE=0 \
cargo run --release -p akita-pcs --example profile
```

and the analogous `onehot_fp64_d32`, `dense_fp32_d64`, `dense_fp64_d32`,
`onehot_d64`, and `onehot_d32` profiles.

### Performance

The initial overnight target was to make fp32/fp64 proving faster than the
matching fp128 setup where possible. Dense small-field proving reached that
shape in the measured `nv26`, single-polynomial profiles. One-hot proving did
not beat fp128, but improved substantially while keeping proof bytes stable.

Representative same-branch one-hot profile movement for `AKITA_NUM_VARS=30`,
`AKITA_NUM_POLYS=4`:

| Host | Mode | Starting prove | Latest accepted prove | Proof bytes |
| --- | --- | ---: | ---: | ---: |
| local ARM/NEON | `onehot_fp32_d64` | 25.284s | 2.564s | 46,000 |
| local ARM/NEON | `onehot_fp64_d32` | 22.213s | 2.115s | 45,904 |
| leopard x86/AVX | `onehot_fp32_d64` | 11.945s | 2.39s | 46,000 |
| leopard x86/AVX | `onehot_fp64_d32` | 10.443s | 2.19s | 45,904 |

Representative dense profile movement for `AKITA_NUM_VARS=26`,
`AKITA_NUM_POLYS=1`:

| Host | Mode | Starting prove | Latest accepted prove | Readout |
| --- | --- | ---: | ---: | --- |
| leopard x86/AVX | `dense_fp32_d64` | 1.928s | 1.272s | faster than `full_d64` comparator |
| leopard x86/AVX | `dense_fp64_d32` | 3.175s | 2.442s | faster than later `full_d32` comparator samples |
| local ARM/NEON | `dense_fp32_d64` | 5.358s | 1.799s | faster than `full_d64` comparator |
| local ARM/NEON | `dense_fp64_d32` | 9.068s | 2.814s | faster than `full_d32` comparator |

Focused benchmark lenses:

- `extension_opening_reduction`: isolates the one-hot sparse tensor reduction
  used by root extension openings. This is the fast loop for future EOR work.
- `onehot_root_projection_commit`: decomposes root projection, transformed-root
  commitment, and scheme-level small-field commitment.

The remaining one-hot gap is concentrated in
`prove_prepared_root_extension_opening_reduction`, especially sparse witness
accumulation and folding. On recent leopard traces this was about `1.19s` for
`onehot_fp32_d64` and `0.86s` for `onehot_fp64_d32` inside roughly
`2.4s`/`2.2s` prove spans.

## Design

### Architecture

The PR keeps the proof system shape intact and moves work to better prover-side
owners.

```text
root opening claims
        |
        v
protocol/flow.rs
  - group claims by opening point
  - ask backend for tensor partials / sparse same-point witness
  - pass transcript-bound input claim into reduction prover
        |
        v
AkitaPolyOps backend hooks
  - DensePoly streams tensor openings from ring storage
  - OneHotPoly builds sparse tensor witnesses and cached sparse roots
        |
        v
akita-sumcheck extension-opening reduction
  - dense table path for dense/fallback cases
  - sparse witness path for one-hot root openings
  - lazy tensor equality factor for early sparse rounds
        |
        v
unchanged proof transcript and verifier checks
```

`AkitaPolyOps` is the main abstraction boundary. The default implementations
remain conservative, but dense and one-hot backends now provide storage-aware
implementations. This avoids forcing protocol code to know whether the source
polynomial is dense, one-hot, or already transformed into a sparse ring root.

The extension-opening reduction keeps one public mathematical contract: prove a
degree-two sumcheck for `sum_x witness(x) * factor(x)`. The implementation now
allows the transparent factor to be dense or represented lazily as a tensor
equality factor during early sparse rounds. The lazy representation rejoins the
dense factor table path after the bounded lazy phase.

Sparse constructors were tightened instead of bypassed. When a caller can
produce sorted or sorted-unique sparse entries, it may use a stronger
constructor that validates and consumes that invariant directly. Generic callers
still use constructors that sort or combine as needed.

The one-hot root projection cache is attached to `OneHotPoly`, the owner of the
immutable indices that determine the projection. `RootTensorProjectionPoly`
stores sparse projected roots by shared ownership so commit/prove can reuse the
same deterministic projection without threading borrowed lifetimes through the
prover API.

The field arithmetic change is intentionally small: `Fp2` multiplication uses
the standard three-base-multiply Karatsuba formula in scalar and packed paths.
Rejected fp4 and CRT arithmetic rewrites are not included.

### Alternatives Considered

1. Auto-densify sparse witnesses once folded support became large.
   Correctness was fine, but full profiles were flat or slower. The dense
   materialization did not remove the transparent-factor work and increased
   table movement.

2. Full-support sparse accumulate/fold fast paths.
   Direct indexing and one-multiply sparse fold formulas were measured and
   rejected. The control-flow simplification was not the dominant cost.

3. Deferred sparse fold fusion or sparse round pair caches.
   These removed an apparent repeated scan but added pending-fold state or large
   transient `(w0, w1)` caches. Focused benchmarks regressed sharply.

4. Chunked one-hot sparse EOR witnesses.
   A root-specific fixed-slot representation was prototyped and tested against
   the generic sparse path. It added a new witness type and protocol dispatch
   but lost to the existing sorted sparse vector in focused benchmarks.

5. Tensor-factor micro-specialization.
   Flattened low-state storage, incremental low-state folding, fused factor
   pair projection, fixed-width projection, and lazy cutoff sweeps were all
   measured. None produced a robust focused benchmark win over the kept
   lazy-factor design with cutoff `12`.

6. Sparse fold merge rewrites.
   Rayon flattening, direct-fill sparse fold, serial in-place fold, append-based
   merging, and chunk-count tuning were rejected. The current sparse fold needs
   parallelism; avoiding one copy was not enough to beat the measured path.

7. Sparse-ring grouping/counting variants.
   Unconditional counting and grouped column accumulation were rejected. The
   kept sparse-ring change is narrower: a position-bucketed tile path only when
   a tile is dense enough in block positions to justify the bucket array.

8. Q64 three-prime CRT cutover.
   This improved a commitment kernel in isolation, but did not improve the
   active prover metric on clean dense fp64 profiles. It is left for a separate
   commitment-focused design if needed.

9. RingSubfieldFp4 arithmetic rewrites.
   Product hoisting and a quadratic-tower multiply attempt either did not move
   the target profile or regressed it significantly. The PR keeps only the
   clearly standard `Fp2` Karatsuba improvement.

10. One-hot hot-position and active-list micro-optimizations.
    Direct hot-position arithmetic, local sorted insert, and active coefficient
    lists were removed after measurement. The only kept local precompute is the
    weighted tensor basis because it is small, local, and reduced the intended
    construction span.

## Documentation

- Add this retrospective spec under `specs/` so PR #85 satisfies the large-PR
  spec workflow.
- Keep the PR description focused on the actual diff and latest validation.
- The profile examples and CI benchmark comment remain the operational
  documentation for the current profile commands and benchmark report format.
- No README change is required because this PR does not change the public
  prover/verifier API or user-facing profile knobs.

## Execution

Implemented order, condensed:

1. Establish native-release baselines on local ARM/NEON and leopard x86/AVX for
   dense and one-hot fp32/fp64, plus fp128 comparators.
2. Move tensor column partials and tensor-packed openings to backend hooks.
3. Teach one-hot to compute tensor partials and sparse same-point tensor
   witnesses directly.
4. Add sparse witness and sparse-ring constructors that encode sortedness and
   packed coefficient contracts at the data-structure boundary.
5. Introduce lazy tensor equality factors for sparse extension-opening
   reduction and validate them against dense round polynomials.
6. Hoist sparse round coefficient multiplication and parallelize large dense
   accumulation/folding work through the existing Rayon feature.
7. Add one-hot root projection caching and position-bucketed sparse-ring column
   sweeps for dense root-projection commitment tiles.
8. Standardize `Fp2` multiplication on the Karatsuba formula across scalar and
   packed paths.
9. Add the focused Criterion benches that let future work measure EOR and
   one-hot root-projection commitment without running the full prover.
10. Remove measured losers instead of carrying speculative fast paths.
11. Fix PR review and CI fallout, including restoring non-x86 parallel tensor
    factor projection and committing the declared bench sources.
12. Add this retrospective spec.

Residual risk and follow-up:

- One-hot small-field prove remains slower than fp128 because it still pays the
  root extension-opening reduction. The remaining plausible win is a larger
  one-hot/root-specific EOR design or a protocol change that avoids the current
  large sparse reduction.
- ARM timings were sometimes noisy under local machine load. The accepted
  changes were checked with focused tests and repeated profiles, and leopard
  was used as the clean x86 acceptance host.
- The new focused benches should be used before any future sparse fold,
  tensor-factor, or sparse-ring kernel change is accepted into production code.

## References

- PR #85: <https://github.com/LayerZero-Labs/akita/pull/85>
- Predecessor extension-opening spec: `specs/extension-field-opening-batching.md`
- Field-role scaffolding spec: `specs/general-field-support.md`
- Focused bench: `crates/akita-pcs/benches/extension_opening_reduction.rs`
- Focused bench: `crates/akita-pcs/benches/onehot_root_projection_commit.rs`
- Profile command shape:

```text
RUSTFLAGS="-C target-cpu=native" \
AKITA_MODE=onehot_fp32_d64 \
AKITA_NUM_VARS=30 \
AKITA_NUM_POLYS=4 \
AKITA_PROFILE_TRACE=0 \
cargo run --release -p akita-pcs --example profile
```
