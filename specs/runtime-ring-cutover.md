# Spec: Runtime Ring-Dimension Full Cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-29 |
| Status        | active |
| PR            | single draft PR from `quang/runtime-ring-full-cutover` |
| Supersedes    | collapsed incomplete attempt in PR #227 and closed child runtime-ring PR stack |
| Superseded-by | |
| Book-chapter  | book/src/how/architecture.md |

## Summary

This cutover removes the cyclotomic ring dimension `D` from Akita's high-level
PCS API, proof/claim storage, setup wrapper types, transcript boundary, and
prover/verifier orchestration. Ring dimension becomes schedule-derived shape
metadata used to validate flat field-element vectors and to enter
const-generic arithmetic kernels. The implementation may remain internally
const-generic at kernel and backend leaf boundaries, but the protocol shape is
field vectors plus schedule-owned dimensions, not `CyclotomicRing<F, D>` at the
API boundary.

## Intent

### Goal

Build the full runtime ring-dimension cutover in one PR based on current
`main`, so that the normal PCS entry points, proof objects, verifier claims,
prover claims, setup wrappers, transcript absorption, and protocol orchestration
do not carry a compile-time root `D`.

### Single-PR Contract

This work intentionally uses one large PR. The previous attempt split the work
into many stacked partial PRs; that made it hard to see whether the cutover was
actually complete and encouraged compatibility layers that hid unfinished
surfaces. This PR must instead keep all implementation slices in one branch.

Rules:

- The branch starts from `layerzero/main`, not from the collapsed old stack.
- PR #227 and its child stack are historical references only. They may be used
  for tests, examples, and lessons, but their architecture is not the baseline.
- Do not introduce backward-compatible aliases, wrapper APIs, bridge traits, or
  deprecated old paths.
- Do not make a D-free API that immediately reconstructs the old D-typed API
  as its normal implementation path.
- The implementation may use multiple commits and internal milestones, but the
  public review artifact remains one PR until this full cutover lands or is
  intentionally abandoned.

### Current Mainline Surfaces To Remove

The current `main` baseline still bakes `D` into these high-level surfaces:

- `AkitaCommitmentScheme<const D, Cfg>`
- `CommitmentProver<F, D>`
- `CommitmentVerifier<F, D>`
- `AkitaProverSetup<F, D>`
- `AkitaCommitmentHint<F, D>`
- `ProverOpeningBatch<'a, PointF, P, CommitF, D>`
- `ProverCommitmentGroup<'a, P, F, D>`
- `RingCommitment<F, D>` as a protocol-facing commitment object
- `FlatDigitBlocks<D>` as protocol/hint storage
- `FlatRingVec<F>` methods that reconstruct protocol-level
  `RingCommitment<F, D>` or absorb by pretending to be a typed ring commitment
- top-level prover and verifier functions such as
  `akita_prover::batched_prove::<..., D>` and
  `akita_verifier::batched_verify::<..., D>`

These names are allowed to remain only when they are leaf arithmetic types,
backend/kernel traits, tests for low-level arithmetic, or temporary in-flight
code inside this PR before the final state. They must not remain as normal
PCS/protocol surfaces at merge time.

### Target Model

`D` is a view over protocol data. It is not a storage type, not a proof type,
not a claim type, not the high-level scheme type, and not a root-level dispatch
parameter.

The protocol stores and transmits field-element vectors. The schedule determines
how those vectors are interpreted at each level:

- level ring dimension
- expected number of ring elements
- expected field-element length
- setup view dimension and prefix geometry
- transcript absorption length
- verifier-side shape validation

`CyclotomicRing<F, D>` remains the right representation for arithmetic once a
specific operation has selected a concrete `D`. It should live at boundaries
such as NTT transforms, digit decomposition kernels, ring-switch arithmetic,
tensor/opening kernels, and small local views over already validated buffers.

### Naming

Use the following naming direction for new canonical containers:

- `RingVec<F>`: owned protocol storage for one or more ring-shaped rows as raw
  field coefficients. It is the platonic owned object, not merely a "flat"
  workaround.
- `RingView<'a, F>` or a more specific borrowed view type: borrowed validated
  access to a `RingVec` under schedule-provided shape.
- Keep `CyclotomicRing<F, D>` for leaf arithmetic. Do not delete it as part of
  this cutover.

`FlatRingVec<F>` should not survive as the main public/protocol name. If a
low-level helper still needs "flat" in its name, it should be clearly private or
implementation-local.

### Invariants

Schedule ownership:

- Every proof and claim vector that represents ring-shaped data is validated
  against schedule-derived field lengths before verifier arithmetic.
- The verifier never trusts a stored `ring_dim` tag inside a proof object when
  the schedule already determines that dimension.
- Root commitments are interpreted using the root schedule level. Recursive
  fold commitments are interpreted using their own fold levels. There is no
  special "root D" dispatch for the whole proof.

Storage:

- Protocol-facing commitment, hint, opening, and proof storage is D-free.
- Flat coefficient buffers have checked shape conversions at operation
  boundaries.
- Verifier-reachable malformed lengths return `AkitaError` or
  `SerializationError`, never panic.
- Unbounded allocation from attacker-controlled serialized lengths remains
  guarded by shape contexts and maximum-size checks.

Transcript:

- Transcript absorption is defined over canonical field-element encodings with
  schedule-derived lengths.
- Prover and verifier absorb identical bytes for equivalent claims.
- Absorption must not reconstruct `RingCommitment<F, D>` as the protocol
  authority.

Kernel dispatch:

- Dispatch to const-generic `D` is allowed when entering a concrete arithmetic
  operation.
- Dispatch should be coarse enough to avoid per-row or per-coefficient dynamic
  overhead.
- Dispatch must be local enough that high-level PCS orchestration is D-free.

Setup:

- `AkitaProverSetup<F>` owns an expanded setup with `gen_ring_dim` and
  shape metadata.
- Setup validation checks that the selected schedule can be viewed against the
  generated setup envelope.
- The first full cutover may require all active schedule dimensions to divide
  the setup generation dimension. Arbitrary unrelated dimensions are out of
  scope.

Mixed-D:

- The PR must support multiple ring dimensions across folds when the generated
  schedule provides them and they are valid under the setup envelope.
- It is acceptable for initial generated schedules to remain uniform-D, but the
  protocol/storage/API shape must not prevent mixed-D schedules.
- General per-role dimensions inside one fold, such as separate `d_a`, `d_b`,
  and `d_d`, are out of scope.

### Non-Goals

- Do not redesign cyclotomic arithmetic or remove `CyclotomicRing<F, D>`.
- Do not require arbitrary dimensions unrelated to the setup envelope.
- Do not introduce per-role ring dimensions inside one fold.
- Do not solve all planner optimization work for choosing mixed-D schedules.
- Do not split this into a new PR stack.

## Evaluation

### Acceptance Criteria

- [ ] `AkitaCommitmentScheme` is no longer const-generic over `D`.
- [ ] `CommitmentProver<F, D>` and `CommitmentVerifier<F, D>` are removed or
      replaced by D-free API surfaces.
- [ ] `AkitaProverSetup<F, D>` is replaced by `AkitaProverSetup<F>`.
- [ ] Protocol-facing commitments use `RingVec<F>` or an equivalent D-free
      owned field-vector type, not `RingCommitment<F, D>`.
- [ ] Protocol-facing prover and verifier opening batches are D-free.
- [ ] `AkitaCommitmentHint` and digit-block storage no longer carry a
      compile-time `D`.
- [ ] Transcript absorption of commitments and ring-shaped proof data uses
      flat field coefficients under schedule-derived shape.
- [ ] Top-level `akita_prover::batched_prove` and
      `akita_verifier::batched_verify` are not const-generic over a root `D`.
- [ ] Root polynomial inputs do not force `D` through PCS orchestration. Any
      remaining `DensePoly<F, D>`, `OneHotPoly<F, D>`, or `SparseRingPoly<F, D>`
      usage is confined to implementation views or kernel-entry conversions.
- [ ] Verifier-reachable shape mismatches return errors, not panics.
- [ ] Uniform-D existing E2E tests still pass.
- [ ] A mixed-D-per-level fixture proves and verifies through the normal
      public PCS API, not a special test-only typed path.
- [ ] `rg "AkitaCommitmentScheme<.*const D|CommitmentProver<.*,.*D|CommitmentVerifier<.*,.*D|AkitaProverSetup<.*,.*D|AkitaCommitmentHint<.*,.*D|ProverOpeningBatch<.*,.*D" crates`
      has no protocol-facing hits at merge time.

### Testing Strategy

Existing tests that must continue passing:

- scheme unit tests under `crates/akita-pcs/src/scheme/tests`
- E2E tests under `crates/akita-pcs/tests`
- verifier rejection tests under `crates/akita-verifier`
- setup serialization/deserialization tests
- generated schedule and runtime fallback tests under `crates/akita-config`
- doc guardrails via `./scripts/check-doc-guardrails.sh`

New or updated tests:

- A compile-facing API test that constructs `AkitaCommitmentScheme<Cfg>` without
  a const `D` parameter.
- A prover/verifier E2E test using the normal public API and a schedule whose
  fold levels use different valid dimensions under one setup envelope.
- A transcript equivalence test proving that flat commitment absorption matches
  the old canonical coefficient order for a uniform-D schedule.
- Malformed proof/claim tests for commitment vector length, hint digit length,
  recursive fold commitment length, root commitment length, and terminal/direct
  witness length.
- A grep-style or small Rust compile test that prevents reintroducing the
  removed high-level D surfaces.

### Performance

Expected performance:

- Uniform-D proofs should not materially regress. The steady-state hot path
  should still enter monomorphized kernels.
- Runtime shape validation should happen at proof/protocol boundaries and level
  transitions, not inside inner arithmetic loops.
- NTT and backend caches may remain per concrete `D`; caches for different `D`
  are distinct.
- Memory may temporarily increase during the cutover only when replacing a
  typed view with an owned flat vector. The final implementation should borrow
  validated views where possible.

Verification commands:

```bash
cargo fmt -q
cargo clippy --all --message-format=short -q -- -D warnings
cargo test
./scripts/check-doc-guardrails.sh
```

Focused commands while iterating:

```bash
cargo check -p akita-types -q
cargo check -p akita-prover --tests -q
cargo check -p akita-verifier --tests -q
cargo check -p akita-pcs --tests -q
cargo test -p akita-pcs mixed_d -- --nocapture
```

## Design

### Architecture

The final architecture is:

```text
public PCS API
  AkitaCommitmentScheme<Cfg>
  AkitaProverSetup<F>
  RingVec<F> commitments
  D-free prover/verifier batches
          |
          v
schedule-bound protocol context
  validates level shapes
  maps proof vectors to level-local ring dimensions
  chooses setup views
  defines transcript lengths
          |
          v
operation dispatch boundary
  dispatch by level ring_d
  borrow RingView / typed local slices
          |
          v
leaf arithmetic
  CyclotomicRing<F, D>
  NTT/cache kernels
  dense/one-hot/sparse/root kernels
  ring-switch kernels
```

The schedule is the authority for protocol shape. Proof objects should contain
values, not dimension policy. Dimension policy enters through config and
schedule resolution, then is checked against setup metadata and proof vector
lengths.

### Mainline Cutover Points

#### Scheme API

Replace:

```rust
AkitaCommitmentScheme<const D: usize, Cfg>
CommitmentProver<F, D>
CommitmentVerifier<F, D>
```

with:

```rust
AkitaCommitmentScheme<Cfg>
// D-free inherent methods or D-free traits.
```

The normal API should obtain ring dimensions from `Cfg::runtime_schedule` and
the proof/opening shape. It must not choose a single root `D` and propagate it
through the whole proof.

#### Setup

Replace `AkitaProverSetup<F, D>` with `AkitaProverSetup<F>`.

The setup seed already stores `gen_ring_dim`. The setup wrapper should validate
that any selected runtime schedule can view the setup at each level. The setup
does not need a compile-time `D` to own the expanded matrix.

Backend-prepared setup may still be indexed by concrete `D`. The public setup
wrapper should not be.

#### Commitments

Replace protocol-facing `RingCommitment<F, D>` with `RingVec<F>` or a
commitment newtype over `RingVec<F>`.

The verifier claim path should receive flat commitment coefficients and a
schedule-derived expected shape. It should not receive a typed ring commitment.

`RingCommitment<F, D>` may remain as an internal arithmetic helper while
transitioning, but the final public/protocol API should not expose it.

#### Hints and Digit Blocks

Replace `FlatDigitBlocks<D>` with runtime-shaped digit-block storage:

```rust
struct DigitBlocks {
    digits: Vec<i8>,
    block_sizes: Vec<usize>,
}
```

The ring dimension is supplied by the schedule or operation context when the
digits are interpreted. If storing a dimension locally is useful for debug
validation, it must not be a protocol authority and must not replace schedule
validation.

`AkitaCommitmentHint<F, D>` becomes `AkitaCommitmentHint<F>`.

#### Prover Claims

Replace `ProverOpeningBatch<'a, PointF, P, CommitF, D>` with a D-free batch.
The commitment component should be a flat commitment plus hint. Polynomial
source objects should expose their logical field length/arity without requiring
the high-level batch to carry `D`.

Root polynomial implementations may still have typed storage initially, but
their D must be consumed at a root operation boundary, not at the PCS API.

#### Verifier Claims

`VerifierOpeningBatch` should remain D-free and should point at D-free
commitments. The verifier should derive all commitment row counts and field
lengths from the opening shape and schedule.

#### Transcript

Commitment absorption should use:

- opening batch shape
- flat commitment coefficient vectors
- shared opening point
- claimed values

The transcript contract should say exactly how many field elements are absorbed
for each commitment at each proof location. That count comes from the schedule.

Do not implement transcript absorption by converting a flat commitment back
into `RingCommitment<F, D>`.

#### Prover Orchestration

Replace top-level `batched_prove::<..., D>` with D-free orchestration that:

1. resolves the runtime schedule,
2. validates setup compatibility,
3. validates root commitment/prover claim shape,
4. absorbs D-free claims,
5. dispatches each level's arithmetic by that level's ring dimension,
6. emits D-free proof storage.

#### Verifier Orchestration

Replace top-level `batched_verify::<..., D>` with D-free replay that:

1. resolves or receives the same runtime schedule,
2. validates proof shape before interpreting buffers,
3. absorbs D-free claims,
4. dispatches root/fold/ring-switch arithmetic by the level's schedule
   dimension,
5. returns errors for malformed inputs.

### Alternatives Considered

#### Keep one root D and dispatch from it

Rejected. It treats root dimension as the proof's type even when later folds
have different dimensions. This is the conceptual source of the old failure.

#### Add D-free wrappers over the old typed API

Rejected. It hides incomplete work and creates two contracts to maintain. This
repo does not need backward compatibility for this migration.

#### Split into stacked PRs

Rejected for this attempt. The old stack grew without a clear completion signal.
This PR can have internal milestones, but review should see the full diff and
the final contract together.

#### Delete `CyclotomicRing<F, D>`

Rejected. `CyclotomicRing<F, D>` is still the right arithmetic type once a
kernel has selected a concrete dimension. The cutover removes it from protocol
shape, not from arithmetic.

## Documentation

This PR must update:

- this spec, including acceptance checkboxes before merge,
- `book/src/how/architecture.md` for the new API and type ownership model,
- verifier contract docs if new shape validation boundaries are added,
- any usage or profiling docs that name `AkitaCommitmentScheme<D, Cfg>` or
  `AkitaProverSetup<F, D>`.

The old collapsed runtime-ring branch should not become durable documentation.
If its lessons need to be preserved, fold them into this spec or the book in
plain terms.

## Execution

This list is an implementation order inside the single PR, not a PR split.

1. Land this spec and create the branch/PR as the single cutover vehicle.
2. Introduce the final D-free storage names: `RingVec<F>` and borrowed view
   helpers. Convert proof-facing `FlatRingVec` usages or rename it fully.
3. Convert `FlatDigitBlocks<D>` and `AkitaCommitmentHint<F, D>` to runtime-shaped
   storage.
4. Convert `AkitaProverSetup<F, D>` to `AkitaProverSetup<F>` and push concrete
   D selection into prepared-backend setup/cache access.
5. Convert protocol-facing commitments from `RingCommitment<F, D>` to D-free
   commitment storage.
6. Convert prover opening batches and verifier claims to D-free commitment
   storage.
7. Convert transcript absorption to flat coefficient absorption with
   schedule-derived expected lengths.
8. Remove `AkitaCommitmentScheme<const D, Cfg>`, `CommitmentProver<F, D>`, and
   `CommitmentVerifier<F, D>` from the high-level API.
9. Convert top-level prover orchestration to D-free schedule-owned dispatch.
10. Convert top-level verifier orchestration to D-free schedule-owned dispatch.
11. Push remaining root polynomial D exposure downward to operation boundaries.
12. Add mixed-D-per-level normal-API E2E coverage.
13. Run full format, clippy, tests, doc guardrails, and final grep audits.

Likely risk areas:

- root polynomial trait bounds currently assume `RootPolyShape<F, D>`;
- compute stacks currently store `PreparedSetup<D>`;
- setup-prefix prover registry is currently D-typed;
- hints mix prover-private data with proof/protocol-adjacent storage;
- transcript tests may pin typed serialization details;
- verifier no-panic requirements make shape validation order important.

## References

- PR #227: collapsed incomplete runtime-ring attempt, historical only.
- Closed child PR stack from the previous attempt, historical only.
- `docs/documentation.md`
- `book/src/how/architecture.md`
- `docs/verifier-contract.md`
- `book/src/how/verification.md`
