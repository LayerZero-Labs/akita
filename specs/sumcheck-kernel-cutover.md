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
│   μ′ = Boolean width of zero-extended live witness stream     │
│   live witness length = actual emitted coefficient count      │
│   flat index: idx in 0..live_len                              │
│   optional local views: segment-specific col/coeff coordinates│
│   bind order: schedule-defined over flat Boolean addresses    │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Prover kernel (this spec)                                    │
│   PairScan: one accumulate_pair + FlatPairStream per round   │
│   FusedFoldScan: fold + next-round scan in one flat pass     │
│   fold: WitnessPolynomial + RelationWeightPolynomial         │
│   round_batching: optional fast path for rounds 0–1          │
└─────────────────────────────────────────────────────────────┘
```

`x` and `y` are not protocol concepts. They are temporary storage labels for
the current uniform scalar-level next-witness stream. Mixed role dimensions
already invalidate `y` as a row-family abstraction, and future heterogeneous
witness layouts may invalidate one global `ring_len` entirely. The kernel API
must be phrased in terms of flat live ranges, segment geometry, bind axes, and
row-family evaluators. Any current `x_prefix` / `y_prefix` name is legacy prover
plumbing to be deleted during this cutover.

For the current uniform implementation, a local scan may still view a segment as
`idx = local_col * local_ring_len + local_coeff`. That view is an optimization
input, not the global protocol shape. The iterator must map every visited local
point to a flat address and read zero when the address is outside the live
range. This is the mixed-dimension landing zone: different row families may have
different local ring dimensions, but they all embed into one flat
`RelationWeightPolynomial` / witness vector.

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

**WitnessPolynomial storage** (two prover storage arms for one flat polynomial,
not two modules):

```rust
enum WitnessPolynomial<'a, E> {
    CompactDigits(&'a [i8]),
    FieldEvals(&'a [E]),
}
```

Stage 2 starts from a live flat `Vec<i8>` of balanced digits and later folds to
a live flat `Vec<E>`. This is the same witness polynomial from the protocol
equation; the enum is only the prover storage representation. The compact arm
must keep optimized field-times-small arithmetic (`MulU64Accum` /
small-signed accumulation).

**Relation weight** is always field `E` at `pair_flat(idx0, idx1)`.
If either `idx0` or `idx1` lies outside the live range, both the witness and
relation weight at that address are implicit zero. The kernel must not require
relation-weight storage for padded Boolean-domain slots.

**Range binding** is an optional summand, not part of pair identity:

```rust
enum RangeBindingTerm<'a, E> {
    Absent,
    Present { split_eq: &'a GruenSplitEq<E>, gamma: E },
}
```

Shape can differ, but the contract is fixed: the range term computes
`gamma * eq(stage1_point, z) * w(z) * (w(z) + 1)` over the same flat witness
addresses as the relation term. It must not use a separate global `x/y` scan.

### Flat pair streams (replace prefix modules)

One `FlatPairStream` implementation per **flat-address visit pattern**, not per
stage half and not per global row/column axis:

| Stream | Replaces | When selected |
|--------|----------|---------------|
| `BlockedFlatPairs` | `dense_terms` | Full active width; stream order is chosen to preserve current Gruen table locality |
| `EmbeddedLocalAxisPairs` | `y_prefix`, `x_prefix`, stage-1 sparse-axis scans | A local segment view is useful for the current bind, but every local address maps to `Option<flat_live_address>` |
| `FusedFoldPairs` | `fuse_full_prefix_x_and_compute_round` and analogous fused paths | The prover can fold the current live flat vectors and emit the next round's flat pairs in one memory pass |

All streams emit only flat pair addresses:

```rust
struct PairStep {
    idx0: usize,
    idx1: usize,
}
```

Do not put a global `row`, `column`, `x`, `y`, or `eq_weight` in `PairStep`.
The relation summand only needs `(idx0, idx1)` and the live relation-weight
accessor. Gruen equality weights belong to the fused range-term context, not to
the generic pair identity.

All streams call the same `accumulate_pair(step, w0, w1, p0, p1, …)`. Local
axis streams may use current uniform coordinates internally, but they must map
those coordinates to flat addresses before touching witness or relation-weight
storage.

**Dense is not "field inner product over the full hypercube."**
It is the same field-weight × compact-witness scan with a full flat pair stream.

### Fold (replace relation and witness fold helpers)

```rust
fn fold_witness_polynomial(
    witness: WitnessPolynomial<'_, E>,
    schedule: &FoldSchedule,
    round: usize,
    r: E,
) -> WitnessPolynomial<'_, E>;

fn fold_relation_weight(
    relation_weight: &RelationWeightPolynomial<E>,
    schedule: &FoldSchedule,
    round: usize,
    r: E,
) -> RelationWeightPolynomial<E>;
```

The schedule maps local segment coordinates to flat addresses. A uniform
column-major view may be an internal fast path, but it is not the fold API.
Witness compact fold uses the same flat schedule with `CompactPairFoldLut`.

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
    pair_scan(
      FlatPairStream::for_round(schedule, r),
      WitnessPolynomial,
      RelationWeightPolynomial,
      RangeBindingTerm
    )
  fold_polynomials(schedule, witness, relation_weight, challenge)
```

### Fused fold-scan kernel

The efficient fuse-fold-and-scan path is required in the final architecture. It
is not an exception to pair scan, and it must not keep the current
`fuse_full_prefix_x_and_compute_round` / `x_prefix` vocabulary.

The target is a first-class `FusedFoldScan` kernel:

```rust
fn fused_fold_scan(
    schedule: &FoldSchedule,
    round: usize,
    challenge: E,
    witness: WitnessPolynomial<'_, E>,
    relation_weight: &RelationWeightPolynomial<E>,
    range: &RangeBindingTerm<'_, E>,
) -> (WitnessPolynomial<'_, E>, RelationWeightPolynomial<E>, RoundMessageCache<E>);
```

Shape can differ, but the contract should not:

- It consumes live flat witness and relation-weight vectors.
- It folds both vectors by the same flat schedule.
- It emits the next live flat witness and relation-weight vectors.
- While data is hot in cache, it also scans the next round's `FusedFoldPairs`
  and computes the next round message.
- It preserves compact-digit lookup tables and field-times-small arithmetic
  wherever the input witness is still `CompactDigits`.
- It uses local segment coordinates only inside embedding helpers that emit
  flat addresses.

This is how the current efficient memory pipeline survives the cleanup. The old
function names and module boundaries do not survive.

### `round_batching` (retain, isolate)

`crates/akita-prover/src/protocol/sumcheck/round_batching/` stays.

It is **prover-only**: builds a transient bivariate grid for the first two binds
of the current inner storage axis when that axis has at least two variables and
`b ∈ {4, 8}`, then reconstructs ordinary round messages.

Wire format unchanged (`SumcheckProof` / `EqFactoredSumcheckProof`).

Stage 2 grid uses live `relation_weight_evals` directly (no split `alpha × m`).
It scans only `0..live_len`; padded columns are implicit zeroes.

See [`relation-weight-polynomial.md`](relation-weight-polynomial.md) §1a for
naming and grid contracts.

### Stage 1 (share machinery, different virt polynomial)

Stage 1 proves `0 = sum eq(tau0, z) * Q(S(z))` with eq-factored messages.
It is **not** the same summand as stage 2, but shares:

- `HypercubeGeom` / `FoldSchedule`
- `FlatPairStream` visit patterns
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
| **Pair scan** | Canonical prover kernel: one loop over active flat fold pairs |
| **Flat pair stream** | Flat-address pair source (`BlockedFlatPairs`, `EmbeddedLocalAxisPairs`, `FusedFoldPairs`) |
| **WitnessPolynomial** | Live flat witness polynomial storage: compact balanced digits before folding, field evals after folding |
| **RangeBindingTerm** | Optional Stage 2 `gamma * eq * w * (w+1)` summand over the flat witness |
| **Initial round batch** / **`round_batching`** | Prover-only two-round local-axis grid skip (rounds 0–1) |
| **Live-range fold** | Fold that skips or zero-extends padded flat addresses |
| **Setup prefix** | Stage 3 setup slot absorb (unrelated to sumcheck pair scan) |

Forbidden final vocabulary in this area:

- `prefix` as a generic sumcheck-kernel concept;
- `x_prefix`, `y_prefix`, `prefix_r_stage1`, `use_prefix_x_round`,
  `use_prefix_y_round`;
- `LiveColumnPairs`, `LiveRowPairs`;
- global `x`, `y`, `row`, or `column` in pair-stream APIs.

Full cutover means deleting those names from live code, not renaming them
later.

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

### Slice 1 — Stage 2 flat scan extraction

1. Add the flat kernel surface:
   - `accumulate_pair` (compact + field arms);
   - `FlatPairStream` trait with `BlockedFlatPairs` and
     `EmbeddedLocalAxisPairs`;
   - `WitnessPolynomial`;
   - `RangeBindingTerm`;
   - `scan_round(stream, &WitnessPolynomial, &RelationWeightPolynomial,
     &RangeBindingTerm)` → `(NormRoundTerms, rel_coeffs)`.
2. Rewire `round_flow.rs` to call `scan_round` when not batching or fused.
3. Delete scan bodies from `dense_terms.rs`, `y_prefix.rs`, `x_prefix.rs` once
   equivalent streams are in place.
4. Equivalence tests: new flat streams vs old modules on representative
   fixtures before deletion.

**Tests:** `cargo test -p akita-prover akita_stage2`

### Slice 2 — Unified witness/range fold + round dispatch

1. Add `fold_witness_polynomial` and `fold_relation_weight` over the same flat
   `FoldSchedule`.
2. Route the Stage 2 relation summand and range-binding summand through the
   same flat witness access.
3. Replace `fold_relation_weight_for_round` and parallel witness fold branches
   in `ingest_challenge`.
4. Collapse `round_flow` to: initial batch | flat scan | fused fold-scan.
5. Delete `use_prefix_x_round`, `use_prefix_y_round`, and similarly named
   dispatch flags. Replace them with flat stream selection.

**Tests:** `trace_prefix` tests, fused round-2 transition tests.

### Slice 3 — First-class fused fold-scan

1. Add `FusedFoldScan` over `WitnessPolynomial`, `RelationWeightPolynomial`,
   `RangeBindingTerm`, and `FoldSchedule`.
2. Replace `fuse_full_prefix_x_and_compute_round` with the flat fused kernel.
3. Preserve the current efficient memory behavior:
   - fold live flat vectors;
   - emit next live flat vectors;
   - compute the next round message while data is hot;
   - keep compact lookup tables and field-times-small arithmetic.
4. Delete the old fused function name and any x/y-shaped fused dispatch.
5. Add equivalence tests against current fused fixtures before deletion.

**Tests:** fused round-2 transition tests, trace-prefix fused tests,
`cargo test -p akita-prover round_batching`.

### Slice 4 — Stage 1 and range flat-scan share

1. Move shared flat-stream and `WitnessPolynomial` primitives to
   `sumcheck/kernel/` if Stage 1 also depends on them.
2. Extract Stage 1 range accumulation into shared flat-stream machinery.
3. Point `akita_stage1/round_flow.rs` at shared streams + range precomp.
4. Delete `akita_stage1/x_prefix.rs`, `sparse_y.rs` scan duplication.
5. Add fixtures where live length is not a rectangular global `x/y` grid and
   padded witness data cannot affect the Stage 1 range check or Stage 2
   range-binding term.
6. Fix `can_use_stage2_initial_round_batch` misname on stage-1 prover
   (`can_use_stage1_initial_round_batch`).

**Tests:** `cargo test -p akita-prover akita_stage1`

### Slice 5 — Cleanup and test decomposition

1. Delete empty modules (`dense_terms`, `y_prefix`, `x_prefix` when fully migrated).
2. Split `round_batching/tests.rs` and `akita_stage2/tests.rs` below 1k lines;
   delete reference grid builders that duplicate `reconstruct_round0/1`.
3. Keep `bridge_relation_weight_from_split` deleted from production and shared
   types; `ring_switch/evals.rs` materializes relation weights directly.
4. Delete `akita-types::PreparedRelationWeightPolynomial` stub (verifier owns
   real type).

**Tests:** full workspace `cargo test`; profile parity check.

### Slice 6 — Docs

1. Expand `book/src/how/proving/sumcheck-stages.md` with prover-only optimization
   boundary paragraph.
2. Cross-link this spec from `relation-weight-polynomial.md` and
   `akita-sumcheck` crate docs.
3. Grep live code and docs for forbidden vocabulary from this spec.

**Tests:** `./scripts/check-doc-guardrails.sh`

```text
Slice 1 (stage2 pair_scan)
  └─→ Slice 2 (fold + dispatch)
        └─→ Slice 3 (fused fold-scan)
              └─→ Slice 4 (stage1/range share)
                    └─→ Slice 5 (cleanup)
                          └─→ Slice 6 (docs)
```

## Evaluation

### Acceptance criteria

- [ ] Stage 2 round computation uses one `pair_scan` entry point (batching excepted).
- [ ] `RelationWeightPolynomial` stores exactly one length, the live
  coefficient range, and treats padded Boolean-domain entries as implicit
  zeroes.
- [ ] `dense_terms.rs`, `y_prefix.rs` deleted or reduced to thin re-exports
  during migration only (gone by Slice 5).
- [ ] `fold_relation_weight_for_round`, `fuse_full_prefix_x_and_compute_round`,
  and separate x/y fold helpers deleted or replaced by `FusedFoldScan` /
  flat-stream names.
- [ ] `FusedFoldScan` preserves current fused fold+next-round performance
  without carrying x/y-shaped names or APIs.
- [ ] Stage 1 range scans and Stage 2 range-binding scans use flat
  `WitnessPolynomial` addressing, not a global `x/y` split.
- [ ] `round_flow.rs` has at most two top-level round branches (batch | scan).
- [ ] Equivalence: `round_batching` path matches pair scan on same fixtures.
- [ ] Equivalence: each `FlatPairStream` matches prior module output on fixtures.
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
- `pair_scan_matches_legacy_local_axis_inner` — current sparse inner-axis case
- `pair_scan_matches_legacy_local_axis_outer` — current sparse outer-axis case
- `fused_fold_scan_matches_legacy_fused_path` — preserves current fused path
  round messages and next folded vectors
- `range_scan_ignores_padded_witness` — Stage 1 and Stage 2 range paths cannot
  be changed by out-of-live witness advice
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
