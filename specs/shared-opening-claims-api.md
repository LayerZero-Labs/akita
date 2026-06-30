# Spec: SharedOpeningClaims API


| Field        | Value                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| Author(s)    |                                                                                                             |
| Created      | 2026-06-29                                                                                                  |
| Status       | proposed                                                                                                    |
| PR           |                                                                                                             |
| Supersedes   | `specs/shared-opening-input-api.md`; partial overlap with `specs/single-point-opening-batch.md` (API layer) |
| Book-chapter | book/src/usage/commitment-api.md                                                                            |




## Summary

Replace the parallel prover/verifier opening-batch structs
(`ProverOpeningBatch`, `VerifierOpeningBatch`, `CommitmentGroup`,
`ProverCommitmentGroup`) and the separate `OpeningBatchShape` **/**
`OpeningGroupShape` summary types with two layered claim types:

- `SharedOpeningClaims` — all public data both prover and verifier share
(point, per-group evaluations, commitments, routing metadata).
- `ProverOpeningClaims` — `SharedOpeningClaims` plus prover-only witnesses
(polynomials and commitment hints).

There is **one** batch description type for real protocol inputs. Count
queries, transcript shape absorption, instance-descriptor digests, and schedule
lookup all read from `SharedOpeningClaims` methods — not from a detached shape
struct.

Input types use **private fields and accessor methods only**.

## Intent



### Goal

Introduce a single, layered public input model for batched single-point openings
where verifier input is literally the public subset of prover input, and
`OpeningBatchShape` **is deleted** (no alias, no “derived shape” return type).

Primary types (in `crates/akita-types/src/proof/opening_batch.rs`; prover
extension in `crates/akita-prover/src/lib.rs` or `opening_claims.rs`):

```rust
/// Shared public opening claims: one point and commitment groups in transcript order.
pub struct SharedOpeningClaims<'a, F, C> {
    point: OpeningPoints<'a, F>,
    groups: Vec<CommitmentGroupClaims<'a, F, C>>,
}

/// One commitment group's public claims.
pub struct CommitmentGroupClaims<'a, F, C> {
    point_vars: PointVariableSelection,
    evaluations: Cow<'a, [F]>,
    commitment: C,
}

/// Prover-side opening claims: shared public claims plus witness polynomials and hints.
pub struct ProverOpeningClaims<'a, PointF, P, CommitF, const D: usize> {
    shared: SharedOpeningClaims<'a, PointF, RingCommitment<CommitF, D>>,
    hints: Vec<AkitaCommitmentHint<CommitF, D>>,
    polynomials: Vec<&'a [&'a P]>,
}
```

**All fields are private.** Callers construct through validated constructors and
read through accessor methods only.

**Removed types (no aliases, no deprecation wrappers):**


| Removed                 | Replaced by                               |
| ----------------------- | ----------------------------------------- |
| `VerifierOpeningBatch`  | `SharedOpeningClaims`                     |
| `CommitmentGroup`       | `CommitmentGroupClaims`                   |
| `ProverOpeningBatch`    | `ProverOpeningClaims`                     |
| `ProverCommitmentGroup` | accessors on `ProverOpeningClaims`        |
| `OpeningBatchShape`     | methods on `SharedOpeningClaims`          |
| `OpeningGroupShape`     | `CommitmentGroupClaims` + group accessors |




### Design decisions (detailed)



#### 1. Private fields, method-only access

**Decision:** `SharedOpeningClaims`, `CommitmentGroupClaims`, and
`ProverOpeningClaims` expose **no public fields**. All construction goes through
named constructors; all reads go through accessor methods.

**Rules:**


| Rule                             | Detail                                                                                                      |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| No `pub` fields                  | On all three claim types                                                                                    |
| Construct via constructors       | `CommitmentGroupClaims::new(...)`, `SharedOpeningClaims::from_groups(...)`, `ProverOpeningClaims::new(...)` |
| Read via methods                 | `point()`, `shared()`, `group_evaluations(g)`, `num_total_polynomials()`, …                                 |
| No struct literals at call sites | Tests use constructors                                                                                      |




#### 2. `SharedOpeningClaims` — shared public batch

**Decision:** The public batch type is `SharedOpeningClaims`. No verifier alias.

**Why:**

- Both prover and verifier bind the **same** public claims (point, evaluations,
commitments, routing). “Shared” + “Claims” states that contract directly.
- Prover wraps it in `ProverOpeningClaims`; verifier takes `SharedOpeningClaims`
at the PCS boundary.
- Call sites: `batched_verify(..., shared: SharedOpeningClaims)`,
`prover.shared()`.



#### 3. `ProverOpeningClaims` wraps `SharedOpeningClaims`

**Decision:** Prover claims own private `shared: SharedOpeningClaims<...>` plus
parallel `hints` and `polynomials`. Access via `shared()`, `hints()`,
`group_polys(g)`.

**Alignment invariant:**

```text
shared.num_groups() == hints.len() == polynomials.len()
∀ g: group_polys(g).len() == group_evaluations(g).len()
```



#### 4. Hints on `ProverOpeningClaims` only

Hints stay out of `SharedOpeningClaims` — prover-only, one per commitment group,
accessed via `hints()` / `group_hint(g)`.

#### 5. `CommitmentGroupClaims` per group

One commitment, many evaluations, plus `point_vars` for transcript routing.
Per-group count: `num_evaluations()` only (no batch-level duplicate names).

#### 6. One batch polynomial count: `num_total_polynomials()`

**Decision:** `SharedOpeningClaims` exposes a **single** batch-wide polynomial
count method. Do **not** expose separate `num_claims()` or `num_polynomials()`.

```rust
impl SharedOpeningClaims<'_, F, C> {
    /// Total polynomials opened across all commitment groups (sum of group sizes).
    pub fn num_total_polynomials(&self) -> usize;
}
```

**Why one name:**

- In the current opening-batch model, “claims”, “openings”, and “polynomials”
are the same count — one claimed evaluation per committed polynomial per group.
- `num_claims()` vs `num_polynomials()` duplicated the same integer under two
protocol synonyms and invited callers to wonder which to use.
- `num_total_polynomials()` is explicit about what is being counted and matches
prover-side `flat_polys().len()` / verifier-side `flat_evaluations().len()`.

**Mapping from old APIs:**


| Old                                    | New                              |
| -------------------------------------- | -------------------------------- |
| `OpeningBatchShape::num_claims()`      | `shared.num_total_polynomials()` |
| `OpeningBatchShape::num_polynomials()` | `shared.num_total_polynomials()` |
| `VerifierOpeningBatch::num_claims()`   | `shared.num_total_polynomials()` |


Per-group sizes remain available via `group_sizes()` (evaluations per commitment
group). `CommitmentGroupClaims::num_evaluations()` remains for one group.

#### 7. Delete `OpeningBatchShape` entirely

**Decision:** Remove `OpeningBatchShape` and `OpeningGroupShape`. Do **not**
keep them as derived return types, stored fields, or test-only summaries.

**Replacement map:**


| Current API                                              | After                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------- |
| `OpeningBatchShape::new(nv, k)` (tests)                  | `SharedOpeningClaims::fixture(nv, &[k])`                            |
| `shape.num_vars()`                                       | `shared.num_vars()`                                                 |
| `shape.num_claims()` / `num_polynomials()`               | `shared.num_total_polynomials()`                                    |
| `shape.num_commitment_groups()`                          | `shared.num_groups()`                                               |
| `shape.num_polys_per_commitment_group()`                 | `shared.group_sizes()`                                              |
| `digest_opening_batch(&shape)`                           | `shared.opening_batch_digest()`                                     |
| `CallSection::from_opening_batch(&shape, basis)`         | `CallSection::from_shared_opening_claims(&shared, basis)`           |
| `AkitaScheduleLookupKey::new_from_opening_batch(&shape)` | `AkitaScheduleLookupKey::from_shared_opening_claims(&shared)`       |
| `Cfg::get_params_for_prove(&shape)`                      | `Cfg::get_params_for_prove(&shared)`                                |
| `sample_public_row_coefficients(&shape, t)`              | `sample_public_row_coefficients(shared.num_total_polynomials(), t)` |
| `batched_eval_target_from_opening_batch(&shape, …)`      | `batched_eval_target(shared.num_total_polynomials(), …)`            |


`SharedOpeningClaims` **count / digest / transcript methods:**

```rust
impl SharedOpeningClaims<'_, F, C> {
    pub fn num_vars(&self) -> usize;              // point().len()
    pub fn num_groups(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize; // sole batch-wide poly/claim count
    pub fn group_sizes(&self) -> Vec<usize>;     // per-group evaluation counts

    pub fn opening_batch_digest(&self) -> DescriptorDigest;
    pub fn append_to_transcript<TranscriptF, T>(&self, transcript: &mut T) -> Result<(), AkitaError>;
}
```

**Schedule-only tests:** `SharedOpeningClaims::fixture(num_vars, group_sizes)`.

`RingRelationInstance`**:** store `num_total_polynomials: usize` (and `num_vars`
if needed) instead of `OpeningBatchShape`.

`ProverOpeningClaims::num_vars()` **vs** `shared.num_vars()`**:** prover poly-aware
padding via `num_vars<PolyF>()` after validation; verifier uses `shared.num_vars()`.

#### 8. Pass decomposed variables into internal functions

Public APIs take `SharedOpeningClaims` / `ProverOpeningClaims`. Orchestration
reads counts from `shared`, then internal helpers take `&SharedOpeningClaims` or
explicit primitives — **never** a shape type.

```rust
input.validate::<Cfg::Field>()?;
let shared = prover.shared();
let schedule = Cfg::get_params_for_prove(shared)?;

fn prepare_fold_inner(
    shared: &SharedOpeningClaims<'_, E, RingCommitment<F, D>>,
    polys: &[&P],
    hints: &[AkitaCommitmentHint<F, D>],
    ...
) -> Result<PreparedFold<...>, AkitaError>
```

Prefer `(num_total_polynomials, num_vars, group_sizes)` when a helper needs only
scalars.

#### 9. `point_vars` on each `CommitmentGroupClaims`

Unchanged — transcript and descriptor bind coordinate order per group.

### Invariants


| Invariant                   | Enforcement                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------- |
| Encapsulation               | No public fields; constructor + accessor API only                                   |
| Single batch poly count     | Only `num_total_polynomials()` at batch level — no `num_claims` / `num_polynomials` |
| Single shared padded point  | `shared.num_vars()` consistent across groups                                        |
| Group alignment             | `shared.num_groups()`; each group nonempty                                          |
| Prover alignment            | `group_polys(g).len() == group_evaluations(g).len()`; hints 1:1 with groups         |
| No duplicate shape type     | `OpeningBatchShape` / `OpeningGroupShape` absent                                    |
| Prover/verifier consistency | Same `SharedOpeningClaims` transcript binding                                       |
| No verifier panic           | Validation returns `AkitaError`                                                     |




### Non-Goals

- Multi-group folded root proving (still rejected; `GROUPED_ROOT_*` unchanged).
- Multipoint batches (removed; unchanged).
- Changing proof wire format or transcript labels.
- Separate batch-level `num_claims()` alias for `num_total_polynomials()`.



## Evaluation



### Acceptance Criteria

- [ ] `SharedOpeningClaims`, `CommitmentGroupClaims`, `ProverOpeningClaims` with private fields, constructors, accessors, `check()`, `validate(limits)`.
- [ ] Batch-level count API is `num_total_polynomials()` **only** — no `num_claims()` or `num_polynomials()` on `SharedOpeningClaims`.
- [ ] `OpeningBatchShape` **and** `OpeningGroupShape` **deleted**; zero references in `crates/`.
- [ ] PCS traits use `SharedOpeningClaims` / `ProverOpeningClaims`.
- [ ] All PCS e2e, transcript-hardening, recursion tests pass.



### Testing Strategy

- Port unit tests to `SharedOpeningClaims`; assert `num_total_polynomials()` matches group sum.
- Grep gate: no `OpeningBatchShape`, no `num_claims()` on shared claims type.
- Run full workspace `cargo test`.



## Design



### Architecture

```text
                    ┌─────────────────────────────┐
                    │   AkitaCommitmentScheme     │
                    │  batched_prove / _verify    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ProverOpeningClaims                      SharedOpeningClaims
   shared(), hints(), group_polys(g)        num_total_polynomials(), group_sizes(),
                                            opening_batch_digest(), append_to_transcript
              │
              │ Cfg::get_params_for_prove(shared)
              ▼
        Schedule / descriptor / transcript
```



### Public API surface



#### `SharedOpeningClaims`

```rust
impl<'a, F, C> SharedOpeningClaims<'a, F, C> {
    pub fn from_groups(
        point: impl Into<OpeningPoints<'a, F>>,
        groups: Vec<CommitmentGroupClaims<'a, F, C>>,
    ) -> Result<Self, AkitaError>;

    pub fn fixture(num_vars: usize, evaluations_per_group: &[usize]) -> Result<Self, AkitaError>;
    pub fn with_padded_point(...) -> Result<Self, AkitaError>;

    pub fn check(&self) -> Result<(), AkitaError>;
    pub fn validate(&self, limits: OpeningBatchLimits) -> Result<(), AkitaError>;

    pub fn point(&self) -> &[F];
    pub fn num_vars(&self) -> usize;
    pub fn num_groups(&self) -> usize;
    pub fn num_total_polynomials(&self) -> usize;
    pub fn group_sizes(&self) -> Vec<usize>;

    pub fn groups(&self) -> &[CommitmentGroupClaims<'a, F, C>];
    pub fn group(&self, index: usize) -> Result<&CommitmentGroupClaims<'a, F, C>, AkitaError>;
    pub fn group_evaluations(&self, index: usize) -> Result<&[F], AkitaError>;
    pub fn group_point_vars(&self, index: usize) -> Result<&PointVariableSelection, AkitaError>;
    pub fn group_commitment(&self, index: usize) -> Result<&C, AkitaError>;
    pub fn flat_evaluations(&self) -> Vec<F>;

    pub fn opening_batch_digest(&self) -> DescriptorDigest;
    pub fn append_to_transcript<TranscriptF, T>(&self, transcript: &mut T) -> Result<(), AkitaError>;
}
```



#### `ProverOpeningClaims`

```rust
impl<'a, PointF, P, CommitF, const D: usize> ProverOpeningClaims<'a, PointF, P, CommitF, D> {
    pub fn new(
        shared: SharedOpeningClaims<'a, PointF, RingCommitment<CommitF, D>>,
        hints: Vec<AkitaCommitmentHint<CommitF, D>>,
        polynomials: Vec<&'a [&'a P]>,
    ) -> Result<Self, AkitaError>;

    pub fn from_groups(...) -> Result<Self, AkitaError>;

    pub fn validate<PolyF>(&self) -> Result<(), AkitaError>
    where
        PolyF: FieldCore,
        P: RootPolyShape<PolyF, D> + RootOpeningSource<PolyF, D>;

    pub fn num_vars<PolyF>(&self) -> Result<usize, AkitaError>
    where P: RootPolyShape<PolyF, D>;

    pub fn shared(&self) -> &SharedOpeningClaims<'a, PointF, RingCommitment<CommitF, D>>;
    pub fn hints(&self) -> &[AkitaCommitmentHint<CommitF, D>];
    pub fn group_hint(&self, index: usize) -> Result<&AkitaCommitmentHint<CommitF, D>, AkitaError>;
    pub fn group_polys(&self, index: usize) -> Result<&'a [&'a P], AkitaError>;
    pub fn flat_polys(&self) -> Vec<&'a P>;
    pub fn commitments(&self) -> Vec<&RingCommitment<CommitF, D>>;

    pub fn append_to_transcript<T>(&self, transcript: &mut T) -> Result<(), AkitaError>;
}
```



#### PCS traits

```rust
fn batched_verify<T: Transcript<F>>(
    ...
    shared: SharedOpeningClaims<'_, Self::ExtField, &Self::Commitment>,
    ...
) -> Result<(), AkitaError>;

fn batched_prove<'a, T, P, B>(
    ...
    prover: ProverOpeningClaims<'a, Self::ExtField, P, F, D>,
    ...
) -> Result<Self::BatchedProof, AkitaError>;
```



### Function migration map (selected)


| Before                                      | After                                                               |
| ------------------------------------------- | ------------------------------------------------------------------- |
| `shape.num_claims()`                        | `shared.num_total_polynomials()`                                    |
| `claims.num_claims()`                       | `shared.num_total_polynomials()`                                    |
| `sample_public_row_coefficients(&shape, t)` | `sample_public_row_coefficients(shared.num_total_polynomials(), t)` |
| `OpeningBatchShape::new(nv, k)`             | `SharedOpeningClaims::fixture(nv, &[k])?`                           |
| `prove_input` / `verify_input`              | `ProverOpeningClaims` / `SharedOpeningClaims`                       |




### Before / after samples



#### Construction

```rust
let group = CommitmentGroupClaims::new(
    PointVariableSelection::prefix(point.len(), point.len())?,
    openings,
    commitment.clone(),
    point.len(),
)?;

let shared = SharedOpeningClaims::from_groups(point, vec![group.clone()])?;

let prover = ProverOpeningClaims::new(
    SharedOpeningClaims::from_groups(point, vec![group])?,
    vec![hint],
    vec![&polys],
)?;
```



#### Verifier root replay

```rust
let openings = shared.flat_evaluations();
shared.append_to_transcript::<F, T>(transcript)?;
let row_coefficients = sample_public_row_coefficients::<F, E, T>(
    shared.num_total_polynomials(),
    transcript,
)?;
```



#### Prove orchestration

```rust
pub fn batched_prove(..., prover: ProverOpeningClaims<'a, ...>, ...) -> ... {
    prover.validate::<Cfg::Field>()?;
    let shared = prover.shared();
    validate_batched_inputs(
        expanded.as_ref(),
        shared.point(),
        &shared.group_sizes(),
        true,
    )?;
    let schedule = Cfg::get_params_for_prove(shared)?;
    ...
}
```



## Documentation

- Update book stub: `SharedOpeningClaims` / `ProverOpeningClaims`.
- Update `specs/single-point-opening-batch.md` API bullets.
- Remove `OpeningBatchShape` from architecture docs if mentioned.



## Execution

1. Implement `SharedOpeningClaims` + `CommitmentGroupClaims` (`num_total_polynomials()` only at batch level).
2. Implement `ProverOpeningClaims` + `fixture()`.
3. Migrate config, schedule, descriptor to `&SharedOpeningClaims`.
4. Switch PCS traits; migrate verifier then prover protocol.
5. Delete `OpeningBatchShape`, old batch structs.
6. Test/bench migration; grep gates.



## References

- `specs/single-point-opening-batch.md`
- `specs/multi-group-batching.md`
- `crates/akita-types/src/proof/opening_batch.rs`

