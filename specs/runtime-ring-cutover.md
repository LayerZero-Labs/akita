# Spec: Runtime Ring-Dimension Full Cutover

| Field         | Value |
|---------------|-------|
| Author(s)     | Quang Dao |
| Created       | 2026-06-29 |
| Status        | implemented |
| PR            | #249 (`quang/runtime-ring-full-cutover`) |
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

### Removed Mainline Surfaces (pre-cutover)

The pre-cutover `main` baseline baked `D` into these high-level surfaces. All are
removed at merge:

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
  `ValidatedScheduleContext`, `validate_schedule_ring_dims`), drives transcript flow between
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

1. accepts schedule-derived inputs (`RingLevelContext` or extracted numbers
   from `LevelParams` / `CommitmentRingDims` — not bare `lp.ring_dimension`
   reads for role-specific data),
2. extracts the ring dimension of the **specific role** this operation
   touches (`d_a`, `d_b`, or `d_d` from [`CommitmentRingDims`]),
3. invokes `akita_types::dispatch_ring_dim_result!(role_d, |D| kernel::<D>(…))`
   exactly once for that operation,
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

Progress on this contract is maintained by code review and the mixed-D E2E /
rejection tests below. **Prover and verifier discriminator violations**
(`const D` functions taking schedule types) must stay zero; orchestration must
not reintroduce banned #227 bridge names or F2 level-wrap dispatch in suffix
paths. A slice that adds a D-free path is complete only when the typed path it
replaces is DELETED — "a D-free way exists" is the #227 failure restated.

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
- Per-role dimension *planner emission* inside one fold (schedules that set
  `d_a ≠ d_b` or `d_d` on `LevelParams`) may remain deferred until setup/planner
  work lands (see `specs/mixed-row-ring-dimensions.md`, proposed). **Per-role
  operation dispatch and D-free orchestration on both prover and verifier are
  in scope for this cutover** — the architecture must be complete before
  divergent schedules are turned on.
- **Shape requirement (litmus test — normative).** The dispatch architecture
  must admit `d_a != d_b` within one fold level by changing ONLY the ring
  dimensions fed to individual operation adapters, with no restructuring of
  orchestration. Any design that monomorphizes an entire fold level (or the
  whole proof) on one `D` fails this test and is wrong **even if every
  acceptance grep passes**. When evaluating a design, ask: "could this level
  run its A-row operations at D=128 and its B/D-row operations at D=32 by
  passing different dims to different adapters?" If the answer requires
  restructuring, the dispatch boundary is in the wrong place.

**Per-role authority (`CommitmentRingDims`).** Normative role mapping:

| Role | Field | Data |
|------|-------|------|
| A (`d_a`, `inner`) | fold / ring-switch | `z`, row coefficients, opening-point pack, EOR, stage2 witness geometry, `w_len` divisibility |
| B (`d_b`, `outer`) | commitment | sent commitment rows, COMMIT segment of `y`, `commit_w`, next-witness commitment decode |
| D (`d_d`, `opening`) | opening digits | `e_hat`, `v = D·e_hat`, D-block segment of `y`, D-matrix matvec |

Orchestration obtains `CommitmentRingDims` from `LevelParams::role_dims` /
`RingLevelContext::role_dims` at each fold entry. **Role-specific buffers use
`dims.d_a()`, `dims.d_b()`, or `dims.d_d()` — not bare
`LevelParams::ring_dimension`.** Witness-borrow paths that still call
`uniform_dim()` are deferred follow-on work (see Deferred below).

`RingRelationInstance` and relation helpers (`generate_relation_rhs`,
`relation_claim_from_rows_extension`) must treat `v`, commitment rows, and
`row_coefficient_rings` under their respective role dimensions, not a single
stored `ring_dim`.

`validate_level_dispatch::<D>(lp)` is insufficient for per-role work; kernels
enter through `validate_role_dispatch` keyed on the matching `d_a` / `d_b` /
`d_d`.

### Non-Goals

- Do not redesign cyclotomic arithmetic or remove `CyclotomicRing<F, D>`.
- Do not require arbitrary dimensions unrelated to the setup envelope.
- Do not *emit* divergent per-role schedules from the planner until setup
  matrices and SIS tables support them — but do **not** treat "one D per fold
  level" as the final orchestration model. Reading the deferred planner work as
  permission to keep level-monomorphized verifier/prover spines (forbidden
  pattern F2) is the mistake this spec rejects.
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
- [x] Verifier-reachable shape mismatches return errors, not panics.
- [x] Uniform-D existing E2E tests still pass.
- [x] A mixed-D-per-level fixture proves and verifies through the normal
      public PCS API, not a special test-only typed path.
- [x] **Per-role operation dispatch (litmus):** prover and verifier fold paths
      admit `d_a`, `d_b`, `d_d` from `CommitmentRingDims` with separate
      `dispatch_ring_dim_result!` per operation (EOR, relation build, ring
      switch, stage2, stage3, relation claim). No `uniform_dim()` fused path
      remains on the prove/verify hot path.
- [x] Zero **prover** and **verifier** discriminator violations (`const D` +
      schedule types), zero banned #227 bridge names, and no F2 level-wrap in
      suffix orchestration. Prover orchestration spine has zero `const D`.
- [x] No function in `crates/akita-prover/src` or `crates/akita-verifier/src`
      reads a schedule type and also has `const D` (discriminator rule).
- [x] Every dispatch arm in orchestration calls a kernel or an operation
      adapter, never another orchestration function (no F1/F2 patterns).
      **Verifier `verify_suffix` must not wrap `prepare_fold_data` +
      `verify_fold` in one per-level dispatch** (F2).
- [x] `rg "AkitaCommitmentScheme<.*const D|..."` (high-level D surfaces) has
      no protocol-facing hits at merge time.
- [x] For each D-free replacement added during the cutover, the typed path it
      replaced is deleted in the same slice where applicable (API-surface greps
      alone are not sufficient; structural criteria below are normative).

**Completed slices (2026-07-03):** D-free PCS API, proof storage, prover spine
`const D` = 0, mixed-D-per-level E2E, root poly step 11, council audit fixes
(grind dispatch hoist, `RingView::append_flat_to_transcript` → `Result`),
slices 0–4 (authority, per-role dispatch, verifier F2 teardown, planner
`role_dims`, regression locks).

**Deferred (follow-on, not merge blockers):** divergent per-role planner
emission (`d_a ≠ d_b ≠ d_d` within one fold level), nested ring-switch views
when `d_d ≠ d_a`, and removing the last witness borrow `uniform_dim()` gate.

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
- **`akita-verifier` mixed-D rejection tests** mirroring `mixed_d_per_level_e2e`
  malformed matrix (not only PCS integration).
- Per-role adapter test: `CommitmentRingDims { d_a:128, d_b:64, d_d:32 }` with
  nesting `d_d | d_b | d_a`; each operation adapter dispatches on its role dim
  (planner may still reject divergent dims until Slice 3).
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
  ValidatedScheduleContext / validate_schedule_ring_dims
  RingLevelContext.role_dims per fold level
  validates level shapes
  maps proof vectors to role-local ring dimensions (d_a, d_b, d_d)
  chooses setup views
  defines transcript lengths
          |
          v
operation dispatch boundary (per operation, per role)
  dispatch_ring_dim_result!(dims.d_a() | d_b() | d_d(), ...)
  borrow RingView / typed local slices
  D-free orchestration between dispatches
          |
          v
leaf arithmetic (const D kernels only)
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

- A polynomial is flat field coefficients plus arity metadata. `nuposition_index_bits` is
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

1. resolves the runtime schedule and validates ring dimensions via `validate_schedule_ring_dims`,
2. validates setup compatibility,
3. validates root commitment/prover claim shape,
4. absorbs D-free claims,
5. at each fold level, runs D-free orchestration (`prove_fold`, `prepare_suffix`,
   `RingRelationProver::new`) with **per-operation, per-role** kernel dispatch,
6. emits D-free proof storage.

#### Verifier Orchestration

Replace top-level `batched_verify::<..., D>` with D-free replay that:

1. resolves or receives the same runtime schedule and validates ring dimensions via `validate_schedule_ring_dims`,
2. validates proof shape before interpreting buffers,
3. absorbs D-free claims,
4. at each fold level, runs D-free orchestration (`verify_fold`, `prepare_fold_replay`,
   suffix loop) with **per-operation, per-role** kernel dispatch matching the
   prover operation map,
5. returns errors for malformed inputs.

**Forbidden:** `dispatch_ring_dim_result!(level_d, |D| { prepare_fold_data::<D>;
verify_fold::<D>(...) })` in `verify_suffix` (F2). The verifier must mirror the
prover: D-free fold orchestration, typed kernels only inside single-role dispatch
arms.

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

Slices 0–12 and slices 0–4 (per-role authority, prover finish, verifier F2
teardown, planner `role_dims`, regression locks) all landed in PR #249. The
execution list below is retained as historical implementation order.

### Completed (slices 0–12 and 0–4)

1. Land this spec and create the branch/PR as the single cutover vehicle.
2. Introduce the final D-free storage names: `RingVec<F>` and borrowed view
   helpers.
3. Convert `FlatDigitBlocks<D>` and `AkitaCommitmentHint<F, D>` to runtime-shaped
   storage.
4. Convert `AkitaProverSetup<F, D>` to `AkitaProverSetup<F>`.
5. Convert protocol-facing commitments to D-free storage.
6. Convert prover opening batches and verifier claims to D-free storage.
7. Convert transcript absorption to flat coefficient absorption.
8. Remove high-level `const D` PCS API surfaces.
9. Convert top-level prover orchestration to D-free schedule-owned dispatch.
10. Convert top-level verifier **entry** to D-free (`batched_verify`); F2 spine
    teardown is Slice 2.
11. Cut over root polynomial storage (`RuntimeRootProvePoly`).
12. Add mixed-D-per-level normal-API E2E coverage.

All slice-0–4 items (authority types, prover per-role finish, verifier F2
teardown, planner `role_dims`, regression locks) completed in PR #249. Divergent
per-role planner emission remains deferred to `specs/mixed-row-ring-dimensions.md`.

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
