# Spec: Sumcheck prover kernel cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-07-06 |
| Status        | active |
| PR            | |
| Supersedes    | `specs/akita-sumcheck-unification.md` (deleted) |
| Book-chapter  | how/proving/sumcheck-stages.md |

## Summary

Akita's sumcheck **protocol** is already unified: stages 1–3 use standard
`akita-sumcheck` drivers on the wire, and stage 2 proves a fused inner product
over witness times `RelationWeightPolynomial` plus a batched virtual range term
(see [`relation-weight-polynomial.md`](relation-weight-polynomial.md) and the
book chapter above).

The **prover implementation** is not unified.
It is a cartesian product of modes (`round_batching` × prefix-x × prefix-y ×
sparse × compact/full × fuse-next-round) spread across `dense_terms.rs`,
`x_prefix.rs`, `y_prefix.rs`, and duplicated fold helpers.

This spec replaces the abandoned
[`akita-sumcheck-unification.md`](akita-sumcheck-unification.md) plan (descriptor
algebra, new `akita-protocol` crate, Tier-A generic kernel).
Those ideas were never implemented and are superseded by a smaller cutover: **one
pair-scan kernel**, **geometry-driven iterators**, and **`round_batching` as the
only special-case fast path**.

## Intent

### Goal

Collapse stage 1 and stage 2 prover round computation into a single fused
**pair-scan** loop per round, with fold logic shared across witness and
relation-weight tables, while preserving byte-identical round messages and
prover performance.

### Non-goals

- No new `akita-protocol` crate and no Jolt-style descriptor algebra in this
  cutover.
- No "Tier-A slow generic kernel" that production must fall back to.
  The **reference** implementation is the full-iterator pair scan; fast paths
  must match it round-for-round.
- No change to sumcheck wire format, transcript labels, round counts, or
  verifier drivers (`akita-sumcheck` + `akita-verifier/src/stages/`).
- No stage 3 rewrite (setup product sumcheck keeps its factored-product
  machinery; only shared vocabulary where natural).
- No GPU backend work.

### Invariants

Each names the test or contract that protects it.

- **Soundness preservation.**
  Round polynomials and final oracles are unchanged.
  Protected by: existing E2E prove/verify, `mixed_d_per_level_e2e`,
  `transcript_hardening*`, stage prover unit tests.

- **Byte-identical fast paths.**
  `round_batching` and compact-digit accumulation compute the same round
  messages as the reference pair scan.
  A fast path is an alternative computation of the same message, never a
  different protocol.
  Protected by: new equivalence tests (see Testing Strategy).

- **Verifier boundary unchanged.**
  Prover optimizations (`round_batching`, pair iterators, compact `i8` witness)
  do not appear on the wire or in `akita-verifier`.
  Protected by: grep + book contract in sumcheck-stages.md.

- **Single equation source (verifier + book).**
  Stage 2 `expected_output_claim` and module docs agree on:

  ```text
  w(r) * RelationWeightPolynomial(r)
    + gamma * eq(stage1_point, r) * w(r) * (w(r) + 1)
  ```

  Protected by: [`relation-weight-polynomial.md`](relation-weight-polynomial.md),
  `akita-verifier/src/stages/stage2.rs`,
  `akita-verifier/src/protocol/relation_weight.rs`.

- **Verifier no-panic.**
  Verifier-reachable code returns `AkitaError`, never panics
  ([`docs/verifier-contract.md`](../docs/verifier-contract.md)).

- **Performance parity.**
  Stage-2 prover wall time does not regress on the canonical profile workload
  (see Performance).

- **Boolean hypercube only.**
  All rounds bind `{0,1}` variables; no univariate skip or centered-integer
  domain.

## Problem statement

### What went wrong

The prover organized code around **layout artifacts** (`x` rounds, `y` rounds,
`prefix`, `sparse`) instead of around the **mathematical operation**:

```text
for each active fold pair (idx0, idx1) with eq weight:
  (w0, w1)  from witness  (compact i8 or field E)
  (p0, p1)  from RelationWeightPolynomial evals (field E)
  accumulate virtual:  gamma * eq * w * (w + 1)
  accumulate relation: w * R   (via accumulate_relation_coeffs[_signed])
```

`dense_terms.rs` and `y_prefix.rs` already share this inner loop.
They differ only in **which pairs are visited** (full split-eq blocking vs live
columns only).
Copy-pasting the loop into six modules created ~11k lines under
`protocol/sumcheck/` and a test matrix that tracks mode combinations instead of
invariants.

### What the abandoned unification spec got right

From the deleted `akita-sumcheck-unification.md`, keep:

1. **Diagnosis:** mode cartesian product is unmaintainable.
2. **Three orthogonal axes** (do not conflate):
   - **Proof format:** regular `SumcheckProof` vs eq-factored `EqFactoredSumcheckProof`.
   - **Batching:** which claims/instances share one sumcheck driver.
   - **Prover compute:** Gruen split-eq, compact digits, `round_batching`, pair iterators.
     Compute optimizations never change proof bytes.
3. **Byte-identical fast-path contract.**
4. **Boolean hypercube only.**

### What the abandoned unification spec got wrong

Reject and do not revive without a new spec:

1. **`akita-protocol` crate** and descriptor algebra (`Source`/`Term`/`Expr`/…).
2. **Tier-A generic kernel** as a mandatory slow reference separate from the hot
   loop.
   The reference **is** the pair scan with a full iterator.
3. **Stale stage-2 pilot** (separate `alpha`, `m`, `TraceWeight` sources).
   Superseded by [`relation-weight-polynomial.md`](relation-weight-polynomial.md).
4. **`plan_level` / `LevelProtocolPlan`** as a prerequisite for kernel cutover.
   Schedule derivation can stay in fold orchestration until a later need arises.

## Target model

### Layer separation

```text
┌─────────────────────────────────────────────────────────────┐
│ Protocol (book + relation-weight spec + verifier stages)     │
│   Stage 1: eq-factored range check, claim 0                  │
│   Stage 2: ⟨w,R⟩ + γ·(virtual range)                         │
│   Stage 3: setup product + carried witness (optional)        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Hypercube geometry (per fold level, immutable for instance)  │
│   μ′ = col_bits + ring_bits                                  │
│   live witness length = live_x_cols * y_len                  │
│   flat index: idx = x * y_len + y (column-major)             │
│   bind order: ring variables first, then column variables    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Prover kernel (this spec)                                    │
│   PairScan: one accumulate_pair + PairIterator per round     │
│   fold_table: witness + RelationWeightPolynomial in lockstep │
│   round_batching: optional fast path for rounds 0–1          │
└─────────────────────────────────────────────────────────────┘
```

`x` and `y` are **geometry labels** inside `HypercubeGeom`, not module names.
Mixed ring dimension per level changes `(col_bits, ring_bits, live_x_cols)` at
each fold; it does not require new `x_prefix` / `y_prefix` modules when indexing
is expressed as bind-order over flat `μ′`.

### Stage 2 mathematics (canonical)

Intermediate folds (`gamma != 0`):

```text
relation_weight_claim + gamma * s_claim
  = sum_z [
      w(z) * RelationWeightPolynomial(z)
    + gamma * eq(stage1_point, z) * w(z) * (w(z) + 1)
  ]
```

Terminal folds (`gamma = 0`): virtual term omitted; relation inner product only.

`RelationWeightPolynomial` is materialized on the prover and evaluated in closed
form on the verifier via `PreparedRelationWeightPolynomial` (see
`akita-verifier/src/protocol/relation_weight.rs`).

### The pair-scan kernel

**Per-pair accumulation** (already in `akita_stage2/mod.rs`):

```rust
// Relation: quadratic in witness at fold pair
rel[0] += w0 * p0;
rel[1] += w0 * (p1 - p0) + dw * p0;
rel[2] += dw * (p1 - p0);

// Virtual: w*(w+1) with Gruen eq weights (compact i64 or field)
```

**Witness encoding** (two arms inside one function, not two modules):

| Encoding | When | Virtual term |
|----------|------|--------------|
| `Compact(i8)` | Before / early binds | `i64` unreduced via `MulU64Accum` |
| `Field(E)` | After compact→full transition | field arithmetic |

**Relation weight** is always field `E` at `pair_flat(idx0, idx1)`.

### Pair iterators (replace prefix modules)

One `PairIterator` implementation per **visit pattern**, not per stage half:

| Iterator | Replaces | When selected |
|----------|----------|---------------|
| `SplitEqBlockedPairs` | `dense_terms` | Full active width; Gruen `e_first` × `e_second` blocking |
| `LiveColumnPairs` | `y_prefix`, stage-1 `sparse_y` (y-phase) | `live_x_cols < current_x_len`, binding ring axis |
| `LiveRowPairs` | `x_prefix` (x-phase) | Same sparsity, binding column axis |

All iterators call the same `accumulate_pair(step, w0, w1, p0, p1, …)`.

**Dense is not "field inner product over the full hypercube."**
It is the same field-weight × compact-witness scan with a full pair iterator.

### Fold (replace four relation fold helpers)

```rust
fn fold_table_column_major(
    evals: &mut [E],
    live_cols: usize,
    y_len: usize,
    bind: BindAxis,  // Ring | Column
    r: E,
) -> (next_live_cols, next_y_len);
```

Witness compact fold uses the same geometry with `CompactPairFoldLut`.

Delete as separate concepts:

- `fold_relation_weight_prefix_y`
- `fold_relation_weight_x_column_major`
- `fold_relation_weight_for_round` dispatch tree
- (keep `fold_relation_weight_through_initial_batch` logic inside `round_batching`
  until batch path folds in one shot)

### Round dispatch (two branches, not a cartesian product)

```text
each round r:
  if using_initial_round_batch(r):
    round_batching.reconstruct_round_poly(r)   // prover-only grid
  else:
    pair_scan(PairIterator::for_round(geom, r), witness, weight, split_eq)
  fold_tables(geom.bind_axis(r), witness, weight, challenge)
```

Fuse-fold-and-scan (today's `fuse_full_prefix_x_and_compute_round`) may remain as
an optional **iterator that yields post-fold pairs in one pass** or as a
post-fold cache of the next round polynomial.
It must not be a third parallel module tree.

### `round_batching` (retain, isolate)

`crates/akita-prover/src/protocol/sumcheck/round_batching/` stays.

It is **prover-only**: builds a transient bivariate grid for the first two
**ring-axis** binds when `ring_bits >= 2` and `b ∈ {4, 8}`, then reconstructs
ordinary round messages.

Wire format unchanged (`SumcheckProof` / `EqFactoredSumcheckProof`).

Stage 2 grid uses `relation_weight_evals` directly (no split `alpha × m`).

See [`relation-weight-polynomial.md`](relation-weight-polynomial.md) §1a for
naming and grid contracts.

### Stage 1 (share machinery, different virt polynomial)

Stage 1 proves `0 = sum eq(tau0, z) * Q(S(z))` with eq-factored messages.
It is **not** the same summand as stage 2, but shares:

- `HypercubeGeom` / `FoldSchedule`
- `PairIterator` visit patterns
- `round_batching` for stage 1 (`Stage1RoundBatchState`)
- Gruen `split_eq` driver

Virt accumulation uses `RangeAffineFromSPrecomp` instead of `w*(w+1)` alone.

Target: `akita_stage1/range_scan.rs` (or shared `pair_scan` module) instead of
`x_prefix.rs`, `sparse_y.rs`, `dense_terms` analogues.

`akita_stage1_tree.rs` (large `b`) should delegate to the same pair-scan backend
at leaves.

### Stage 3 (out of scope for this cutover)

Stage 3 uses `FactoredProductTerm` (setup λ × ring), not witness hypercube pair
scan.
Only align naming and cross-links in docs.

### Verifier and wire format

No changes required for kernel cutover.

The verifier never imports `round_batching`, prefix flags, or compact witness
layout.
It replays `akita-sumcheck` drivers and evaluates stage oracles in
`akita-verifier/src/stages/`.

## Vocabulary

Use consistently in code, specs, and book:

| Term | Meaning |
|------|---------|
| **Pair scan** | Canonical prover kernel: one loop over active fold pairs |
| **Pair iterator** | Geometry-driven pair visit order (`SplitEqBlocked`, `LiveColumn`, `LiveRow`) |
| **Initial round batch** / **`round_batching`** | Prover-only two-round y-axis grid skip (rounds 0–1) |
| **Sparse column fold** | Skip zero-padded x slots (`live_x_cols < 2^remaining_col_bits`) |
| **Setup prefix** | Stage 3 setup slot absorb (unrelated to sumcheck pair scan) |

Do not use **prefix** alone for multiple concepts.

Rename over time:

- `prefix_r_stage1` → `cached_stage1_point_for_y_batch`
- `use_prefix_x_round` → iterator kind, not a boolean on the prover

## Crate map (unchanged boundaries)

| Crate / path | Role |
|--------------|------|
| `akita-sumcheck` | Generic sumcheck drivers, proof types, traits |
| `akita-verifier/src/stages/` | Stage 1/2/3 verifier instances + oracles |
| `akita-prover/src/protocol/sumcheck/` | Stage provers + **pair-scan kernel** (this spec) |
| `akita-types::RelationWeightPolynomial` | Materialized relation-weight table |
| `akita-verifier/src/protocol/relation_weight.rs` | Prepared evaluator at final point |

No `akita-protocol` crate.

## Implementation plan

Full cutover in sequenced PRs on the active branch.
Each slice compiles, tests green, deletes replaced code before the next slice
depends on it.

### Slice 1 — Stage 2 pair scan extraction

1. Add `akita_stage2/pair_scan.rs`:
   - `accumulate_pair` (compact + field arms)
   - `PairIterator` trait + `SplitEqBlockedPairs`, `LiveColumnPairs`,
     `LiveRowPairs`
   - `scan_round(iterator, witness, &RelationWeightPolynomial, &GruenSplitEq)`
     → `(NormRoundTerms, rel_coeffs)`
2. Rewire `round_flow.rs` to call `scan_round` when not batching.
3. Delete scan bodies from `dense_terms.rs`, `y_prefix.rs`, `x_prefix.rs`
   (keep fuse helpers temporarily if needed).
4. Equivalence tests: new scan vs old modules on representative fixtures before
   deletion.

**Tests:** `cargo test -p akita-prover akita_stage2`

### Slice 2 — Unified fold + round dispatch

1. Add `fold_table_column_major` on witness + `RelationWeightPolynomial`.
2. Replace `fold_relation_weight_for_round` and parallel witness fold branches
   in `ingest_challenge`.
3. Collapse `round_flow` to: batch path | `scan_round` + `fold_tables`.
4. Remove dead flags: `use_prefix_x_round`, `use_prefix_y_round` as public
   dispatch inputs.

**Tests:** `trace_prefix` tests, fused round-2 transition tests.

### Slice 3 — Stage 1 pair scan share

1. Extract stage-1 virt accumulation into shared iterator machinery.
2. Point `akita_stage1/round_flow.rs` at shared iterators + range precomp.
3. Delete `akita_stage1/x_prefix.rs`, `sparse_y.rs` scan duplication.
4. Fix `can_use_stage2_initial_round_batch` misname on stage-1 prover
   (`can_use_stage1_initial_round_batch`).

**Tests:** `cargo test -p akita-prover akita_stage1`

### Slice 4 — Cleanup and test decomposition

1. Delete empty modules (`dense_terms`, `y_prefix`, `x_prefix` when fully migrated).
2. Split `round_batching/tests.rs` and `akita_stage2/tests.rs` below 1k lines;
   delete reference grid builders that duplicate `reconstruct_round0/1`.
3. Remove `bridge_relation_weight_from_split` from production
   (`ring_switch/evals.rs`); materialize directly.
4. Delete `akita-types::PreparedRelationWeightPolynomial` stub (verifier owns
   real type).

**Tests:** full workspace `cargo test`; profile parity check.

### Slice 5 — Docs

1. Expand `book/src/how/proving/sumcheck-stages.md` with prover-only optimization
   boundary paragraph.
2. Cross-link this spec from `relation-weight-polynomial.md` and
   `akita-sumcheck` crate docs.

**Tests:** `./scripts/check-doc-guardrails.sh`

```text
Slice 1 (stage2 pair_scan)
  └─→ Slice 2 (fold + dispatch)
        └─→ Slice 3 (stage1 share)
              └─→ Slice 4 (cleanup)
                    └─→ Slice 5 (docs)
```

## Evaluation

### Acceptance criteria

- [ ] Stage 2 round computation uses one `pair_scan` entry point (batching excepted).
- [ ] `dense_terms.rs`, `y_prefix.rs` deleted or reduced to thin re-exports
  during migration only (gone by Slice 4).
- [ ] `fold_relation_weight_for_round` and separate x/y fold helpers deleted.
- [ ] `round_flow.rs` has at most two top-level round branches (batch | scan).
- [ ] Equivalence: `round_batching` path matches pair scan on same fixtures.
- [ ] Equivalence: each `PairIterator` matches prior module output on fixtures.
- [ ] E2E and transcript hardening tests unchanged and green.
- [ ] Stage-2 profile workload within noise of pre-cutover baseline.
- [ ] Book + this spec cross-linked; unification spec deleted.

### Testing strategy

**Existing (must stay green):**

- `cargo test -p akita-pcs` E2E, `ring_switch`, `transcript_hardening*`
- `cargo test -p akita-prover` stage1/stage2/round_batching
- `cargo test -p akita-verifier`
- `mixed_d_per_level_e2e`

**New:**

- `pair_scan_matches_legacy_dense` — full-width instances, all rounds
- `pair_scan_matches_legacy_prefix_y` — sparse `live_x_cols`
- `pair_scan_matches_legacy_prefix_x` — x-phase sparse
- `round_batching_matches_pair_scan` — extend `trace_prefix` coverage
- Optional: `fold_table_matches_legacy_relation_folds` per axis

Run with and without `parallel` feature on stage-2 tests.

### Performance

Canonical profile (from AGENTS.md):

```bash
AKITA_MODE=onehot_fp128_d64 AKITA_NUM_VARS=32 cargo run --release --example profile
```

Compare stage-2 sumcheck timing before/after each slice.
No regression expected: hot loop is the same instructions, only reorganized.

If a slice regresses, treat as blocking until the iterator or batch path is
fixed.

## Relationship to other specs

| Spec | Relationship |
|------|----------------|
| [`relation-weight-polynomial.md`](relation-weight-polynomial.md) | Owns stage-2 **protocol** identity and `RelationWeightPolynomial` API |
| [`setup-product-sumcheck.md`](setup-product-sumcheck.md) | Stage 3; unchanged by this cutover |
| [`packed-sumcheck.md`](packed-sumcheck.md) | Orthogonal SIMD/packing; may compose later |
| [`runtime-ring-cutover.md`](runtime-ring-cutover.md) | Per-level `col_bits`/`ring_bits`; geometry inputs to `HypercubeGeom` |
| [`akita-polyops-cutover.md`](akita-polyops-cutover.md) | Witness source boundary; sumcheck kernel consumes compact/field tables |

## Alternatives considered

- **Keep editing monolithic stage drivers.**
  Rejected: mode product grows with every schedule change.

- **Implement full descriptor + `akita-protocol` crate (old unification spec).**
  Rejected: never shipped; oversized for current needs; stale after
  relation-weight cutover.

- **Tier-A slow generic kernel + Tier-B fast paths.**
  Rejected: reference kernel is the full pair scan; slow generic IP is the wrong
  abstraction (field witness × field weight), not what production runs today.

- **Delete all fast paths including `round_batching`.**
  Rejected: measurable regression on stage entry; spec §1a in relation-weight
  marks batching non-negotiable.

- **Verifier-only unification.**
  Already done for stage 2 via `RelationWeightPolynomial` oracle; prover cutover
  is the remaining work.

## References

- Book: [`book/src/how/proving/sumcheck-stages.md`](../book/src/how/proving/sumcheck-stages.md)
- Protocol: [`specs/relation-weight-polynomial.md`](relation-weight-polynomial.md)
- Prover: `crates/akita-prover/src/protocol/sumcheck/`
- Verifier: `crates/akita-verifier/src/stages/`, `protocol/relation_weight.rs`
- Drivers: `crates/akita-sumcheck/`
- Deleted (superseded by this spec): `specs/akita-sumcheck-unification.md`
