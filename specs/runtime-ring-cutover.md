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

Kernel dispatch (normative contract — read this before writing any cutover
code; every prior failure of this migration traces to leaving this boundary
implicit):

Every function on the prove/verify path belongs to exactly one of two roles.

- **Orchestration**: reads schedule types (`ExecutionSchedule`, `LevelParams`,
  `ValidatedScheduleContext`, `RingDimPlan`), drives transcript flow between
  operations, moves D-free storage (`RingVec<F>`, `Commitment<F>`, flat digit
  blocks). Orchestration functions MUST NOT have `const D` in their signature
  or trait bounds.
- **Kernel**: performs ring/field arithmetic or checked shape conversion on
  already-validated buffers. Kernels MAY be const-generic over `D`. Kernels
  receive dimensions and sizes as plain values or const parameters; they MUST
  NOT read any schedule type.

**Discriminator rule (decidable — apply it mechanically, no judgment
required): if a function reads a schedule type, it is orchestration and must
be D-free. If a function needs `const D`, it must receive extracted numbers,
never schedule types. A function that today needs both must be split into an
orchestration half and a kernel half.**

The bridge between the roles is the **operation adapter**: a D-free function
that

1. accepts schedule-derived inputs (`LevelParams`, dims from `RingDimPlan`),
2. extracts the ring dimension of the specific data this one operation
   touches,
3. invokes `akita_types::dispatch_ring_dim_result!(ring_d, |D| kernel::<D>(…))`
   exactly once,
4. converts any D-typed kernel output back to D-free storage inside the
   dispatch arm,
5. returns a D-free type.

Correct in-tree examples: `commit_w`, `ring_switch_build_w`,
`ring_switch_finalize` (crates/akita-prover/src/protocol/ring_switch/).
Copy their shape.

**The dispatch unit is the operation — not the fold level, and not the
proof.** Rationale: the planned per-role ring-dimension work (see Mixed-D
below) requires two different ring dimensions to be live within ONE fold
level. Any dispatch coarser than one operation makes that impossible without
restructuring. Dispatch happens O(operations per level) times per proof —
tens of dispatches total, which is noise. It must never happen per row or per
coefficient; that granularity belongs inside the monomorphized kernel.

Forbidden patterns — each one has already been built, reviewed, and reverted
at least once in this migration. If the code you are about to write matches
either sketch, stop; you are reproducing a known failure, not making
progress:

```rust
// F1: THE FACADE (killed PR #227). A "D-free" entry that recovers D and
// forwards to a typed orchestration function. The typed spine survives; the
// cutover is cosmetic. Detection: the dispatch arm calls a function that
// reads schedule types (i.e. orchestration), not a kernel.
pub fn batched_prove(...) -> ... {
    dispatch_ring_dim_result!(root_d, |D| typed_batched_prove::<..., D>(...))
}

// F2: LEVEL MONOMORPHIZATION (added and reverted in commits 7ec52460 /
// 247b6e7d). Dispatching once per fold level around the whole fold body.
// Looks clean, satisfies every API grep, and is a dead end: it compiles the
// ~250-line fold body 4x, forces every schedule-aware helper below it to
// carry const D or re-dispatch (double dispatch plus `ring_dim == D`
// consistency gates), and cannot express per-role dims within one level.
dispatch_ring_dim_result!(level_d, |D| prove_fold::<..., D>(...))
```

A function is a kernel — and may sit inside a dispatch arm — only if it
satisfies the discriminator rule above. "It does a lot of arithmetic" does
not make `prove_fold` a kernel; it reads the schedule, so it is
orchestration.

Progress on this contract is measured by
`scripts/ring-cutover-progress.sh`: the number of `const D` functions in the
orchestration spine files must decrease monotonically and reach zero at
merge (`--merge-gate`). A slice that adds a D-free path is complete only
when the typed path it replaces is DELETED — "a D-free way exists" is the
#227 failure restated.

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
- Per-role dimension *execution* inside one fold (separate `d_a`, `d_b`,
  `d_d`) is out of scope for this PR, but it is the motivating end state of
  this cutover (see `specs/mixed-row-ring-dimensions.md`, proposed, on branch
  `quang/mixed-row-ring-dimensions`): per-matrix ring dimension shrinks
  matrix descriptions toward module rank 1, speeds up the matvec, sparsifies
  the challenge set, and shortens setup offloading. This cutover exists to
  make that work possible.
- **Shape requirement (litmus test — normative).** The dispatch architecture
  must admit `d_a != d_b` within one fold level by changing ONLY the ring
  dimensions fed to individual operation adapters, with no restructuring of
  orchestration. Any design that monomorphizes an entire fold level (or the
  whole proof) on one `D` fails this test and is wrong **even if every
  acceptance grep passes**. When evaluating a design, ask: "could this level
  run its A-row operations at D=128 and its B/D-row operations at D=32 by
  passing different dims to different adapters?" If the answer requires
  restructuring, the dispatch boundary is in the wrong place.

### Non-Goals

- Do not redesign cyclotomic arithmetic or remove `CyclotomicRing<F, D>`.
- Do not require arbitrary dimensions unrelated to the setup envelope.
- Do not *implement* per-role ring dimensions inside one fold — but the
  dispatch architecture must satisfy the per-role shape requirement in
  §Invariants/Mixed-D. Reading this non-goal as "one D per level is the
  final model" is the mistake that keeps regenerating level-monomorphized
  designs (forbidden pattern F2).
- Do not solve all planner optimization work for choosing mixed-D schedules.
- Do not split this into a new PR stack.

## Evaluation

### Acceptance Criteria

- [x] `AkitaCommitmentScheme` is no longer const-generic over `D`.
- [x] `CommitmentProver<F, D>` and `CommitmentVerifier<F, D>` are removed or
      replaced by D-free API surfaces.
- [x] `AkitaProverSetup<F, D>` is replaced by `AkitaProverSetup<F>`.
- [x] Protocol-facing commitments use `RingVec<F>` or an equivalent D-free
      owned field-vector type, not `RingCommitment<F, D>`.
- [x] Protocol-facing prover and verifier opening batches are D-free.
- [x] `AkitaCommitmentHint` and digit-block storage no longer carry a
      compile-time `D`.
- [x] Transcript absorption of commitments and ring-shaped proof data uses
      flat field coefficients under schedule-derived shape.
- [x] Top-level `akita_prover::batched_prove` and
      `akita_verifier::batched_verify` are not const-generic over a root `D`.
- [x] Root polynomial inputs do not force `D` through PCS orchestration. Any
      remaining `DensePoly<F, D>`, `OneHotPoly<F, D>`, or `SparseRingPoly<F, D>`
      usage is confined to implementation views or kernel-entry conversions.
- [ ] Verifier-reachable shape mismatches return errors, not panics.
- [x] Uniform-D existing E2E tests still pass.
- [ ] A mixed-D-per-level fixture proves and verifies through the normal
      public PCS API, not a special test-only typed path.
- [x] `rg "AkitaCommitmentScheme<.*const D|CommitmentProver<.*,.*D|CommitmentVerifier<.*,.*D|AkitaProverSetup<.*,.*D|AkitaCommitmentHint<.*,.*D|ProverOpeningBatch<.*,.*D" crates`
      has no protocol-facing hits at merge time.
- [x] `scripts/ring-cutover-progress.sh --merge-gate` passes: zero `const D`
      in the prover orchestration spine
      (`crates/akita-prover/src/protocol/core.rs` and
      `crates/akita-prover/src/protocol/core/{prove,fold,root_fold,suffix}.rs`)
      and zero hits for the banned #227 bridge names.
- [x] No function reads a schedule type (`ExecutionSchedule`, `LevelParams`,
      `ValidatedScheduleContext`, `RingDimPlan`) and also has `const D`
      (discriminator rule; enforced by review over the spine diff).
- [x] Every dispatch arm in orchestration calls a kernel or an operation
      adapter, never another orchestration function (no F1/F2 patterns).
- [x] For each D-free replacement added during the cutover, the typed path it
      replaced is deleted in the same slice — the API-surface greps above are
      satisfiable by a facade; these structural criteria are the ones that
      cannot be gamed.

The verifier spine (`verify_fold`, `verify_root_inner`, `prepare_fold_data`)
is per-level monomorphized today. That is accepted as *transitional* for this
PR — its entry is D-free and uniform-D behavior is correct — but it is
forbidden pattern F2 as an end state and must be decomposed to operation
adapters when per-role dims land (tracked in
`specs/mixed-row-ring-dimensions.md`). Do not copy the verifier spine's shape
into the prover.

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
See "Root Polynomial Inputs" below — typed poly storage is the single
structural reason the prover entry cannot lose its `const D`, and it has its
own cutover step.

#### Root Polynomial Inputs

This is the load-bearing blocker for a D-free `batched_prove`, and the reason
both prior attempts stalled at the entry signature. Today callers construct
`DensePoly<F, D>`, `OneHotPoly<F, D>`, `SparseRingPoly<F, D>` — `D` is baked
into the *type constructor of the input data*. `batched_prove` is generic
over `P: RootProvePoly<F, D>`, so as long as inputs carry `D` in their type,
any "D-free" entry can only be a facade that reads `D` back out of the type
(forbidden pattern F1). **You cannot cut the entry before cutting the input
representation. Do not try.**

The cutover:

- A polynomial is flat field coefficients plus arity metadata. `num_vars` is
  already stored on the poly independent of `D` (that is what the
  `RootPolyMeta<F>` / `RootPolyShape<F, D>` split established). Make the
  storage types D-free: `DensePoly<F>`, `OneHotPoly<F, I>`,
  `SparseRingPoly<F>`.
- `RootPolyShape<F, D>` and the view traits (`RootCommitSource`,
  `RootOpeningSource`, `RootTensorSource`, `DirectRootWitnessSource`) become
  kernel-entry view constructors reached through operation adapters that
  dispatch on the schedule's dimension for that operation.
- Orchestration bounds use `RootPolyMeta<F>` only. The
  `RootProvePoly<F, D>`-style bounds on orchestration collapse into
  runtime-supported bundles following the pattern already proven in-tree by
  `RuntimeRingSwitchProveBackend<F>` (a supertrait over all supported
  dimensions).
- This is the same thesis the rest of the cutover applies to commitments:
  the protocol object is a field-coefficient vector; `D` is a view selected
  at the operation, not a property of the data.

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

#### Dispatch once per fold level (monomorphized `prove_fold`)

Rejected as an end state — this is forbidden pattern F2, and it is the design
agents repeatedly converge on because it looks like a clean compromise: one
dispatch site, kernels compose typed inside, matches the current verifier.
It fails three ways: (1) it monomorphizes the entire fold body 4x for no
benefit — the orchestration between kernels is bookkeeping; (2) it forces
every schedule-aware helper below it to either carry `const D` or re-dispatch
at runtime, producing double dispatch plus `ring_dimension == D` consistency
gates (the state the prover was in mid-2026-07); (3) it is structurally
incapable of per-role dimensions within one level, which is the motivating
end state of this whole cutover. If a proposed design contains
`dispatch_ring_dim_result!(level_d, |D| <any function that reads schedule
types>::<D>(…))`, it is this alternative and must not be built.

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
11. Cut over root polynomial storage to D-free types and collapse
    `RootProvePoly<F, D>`-style orchestration bounds into runtime-supported
    bundles (see "Root Polynomial Inputs"). Only after this can steps that
    remove `const D` from `prepare_root`/`prove_root`/`prove`/`batched_prove`
    succeed without a facade.
12. Add mixed-D-per-level normal-API E2E coverage.
13. Run full format, clippy, tests, doc guardrails,
    `scripts/ring-cutover-progress.sh --merge-gate`, and final grep audits.

Live sequencing, per-slice burn-down state, and the concrete remaining-work
inventory are maintained in the branch worklog
(`RUNTIME-RING-CUTOVER-PLAN-NEVER-COMMIT.md`, never committed).

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
- `specs/mixed-row-ring-dimensions.md` (proposed, branch
  `quang/mixed-row-ring-dimensions`): per-role ring dimensions inside one
  fold — the motivating end state this cutover must not preclude. Its shape
  requirement is normative here (§Invariants/Mixed-D).
- `docs/documentation.md`
- `book/src/how/architecture.md`
- `docs/verifier-contract.md`
- `book/src/how/verification.md`
