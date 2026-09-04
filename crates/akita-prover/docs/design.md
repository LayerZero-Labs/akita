# Design and Implementation Plan: Composable Commitment Execution


| Field        | Value                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| Author       | Omid Bodaghi                                                                                                    |
| Revised      | 2026-09-04                                                                                                      |
| Status       | Proposed                                                                                                        |
| Rebase base  | `origin/codex/trusted-schedule-artifacts` at `c02ed7928`                                                       |
| Scope        | Prover-side commitment execution in `akita-prover`, plus direct callers in `akita-pcs`, `akita-setup`, and Jolt |


## 1. Decision

Akita will separate commitment execution into three replaceable stages:

1. inner commitment with matrix A;
2. outer commitment with matrix B; and
3. commitment compression.

A caller can assign each stage to a backend. A backend that owns both the
inner and outer stages can also implement one explicit fused operation. The
fused operation performs A, outer decomposition and slicing, and B without a
host round trip between A and B.

Source types will implement one Akita-owned, ring-dimension-free source
contract. A source can use a standard `PolynomialRepresentation`, an optional
backend-specific external inner commitment operation, or both. Adding a source
that uses a standard polynomial type will not require a new backend type for
commitment execution.

This work changes prover execution and prover-private state ownership. It does
not change the protocol, schedules, commitment bytes, proof bytes, transcript,
setup derivation, or verifier. The public result becomes
`CommitOutput<F, S>`, where `S` is prover-private commitment state selected by
the execution configuration. `S` is not a protocol type and has no universal
serialization, cloning, host-memory, or `AkitaCommitmentHint` bound.

The CPU compatibility configuration can still use
`AkitaCommitmentHint<F>` as its portable state or export that value at an
explicit persistence boundary. A GPU configuration can instead return a
checked reference to resident inner and compression state. It does not need to
construct, download, or return an `AkitaCommitmentHint`. The opening and
ring-relation paths consume `S` through operation-specific state adapters, not
by reading fields from a mandatory hint.

## 2. Requirements



### R1. A source owns its inner-commit representation

A source type inside or outside this repository implements one Akita-owned
source trait. It does not need to define a new backend merely to translate its
data into an existing commitment representation.

This requirement is deliberately limited to commitment execution. A source
that participates in a complete proof still implements the existing
`RootOpeningSource` and, where applicable, `RootTensorSource` capabilities at
the supported dimensions, and the selected opening/tensor backends still
implement their source-typed kernels. Generalizing those boundaries is
separate work. The commit-only external-wrapper acceptance test in section
10.2 therefore proves only the commitment-side extensibility claim.

The source has two possible paths:

- The standard path exposes `DenseType`, `ShortNormType`, or `OneHotType`
through a bounded borrowed representation. Backends implement algorithms for
these polynomial types, not for every concrete source type.
- The external-operation path optionally exposes a source-owned inner
commitment operation for a named backend kind. This preserves downstream
hand-tuned algorithms such as Jolt's packed one-hot trace commit.

When both paths exist, the standard polynomial path is the reference and can be
forced in tests.

### R2. Outer commitment and compression are independently routable

The caller chooses the inner, outer, and compression operations when it builds
a `CommitmentExecutor`. Each selected operation carries prepared state for the
same expanded setup. Executor construction rejects setup or capability
mismatches before commitment arithmetic begins.

Changing a route must not change `u`, the compressed payload, the public
commitment, the transcript, or the proof. Prover-private state representation
may change with the route.

### R3. Inner and outer commitment can be one backend operation

Fusion is explicit. The caller supplies a `FusedInnerOuterOperation`. The
composite makes one call to that operation for A, outer decomposition and
slicing, and B. A remote or GPU implementation must perform these steps in one
transport submission, with no host-visible inner rows or digit planes between
A and B.

After B finishes, the fused result returns `u` and a checked backend-state
reference. The reference identifies the resident inner image and binds it to
the setup, plan, backend instance, state store, and generation. It does not expose
host inner rows. An implementation may export portable state later only when a
selected consumer or setup-prefix persistence policy explicitly requires that
transfer.

Rust's trait system cannot prove that an implementation used one transport
submission. A recording fake transport and backend-specific integration tests
enforce that operational part of the contract.

### R4. Trusted schedule ownership remains outside execution routing

`TrustedScheduleCatalog` remains the single validated schedule authority used
by setup, commitment, proving, and verification. `AkitaCommitmentScheme` owns
that catalog. The low-level root commitment API receives it explicitly,
resolves `AkitaScheduleLookupKey` strictly, and rejects a missing row; it never
invokes planner search or a compiled-table fallback.

Catalog decoding, family/policy validation, row-digest validation, and setup
sizing remain preprocessing or scheme-construction responsibilities. A
`CommitmentExecutor` receives only the already-resolved commitment profile and
does not own, decode, synthesize, or cache schedule rows. Executor route
metadata does not enter `OpeningScheduleSelection`, the catalog digest, or any
other authenticated schedule identity.

## 3. Non-negotiable compatibility gates

This repository permits breaking Rust APIs, but this project may not remove a
supported capability or change protocol output.

### 3.1 Byte and proof-size invariants

For identical setup, parameters, sources, and prover randomness:

- `GroupCommitPhaseParams` is unchanged;
- the uncompressed outer image `u` is coefficient-for-coefficient identical;
- the compression chain plan is unchanged;
- the schedule-authenticated `RingRelationMode` is unchanged;
- the `Commitment` and `CommittedGroup` values are identical;
- canonical serialized commitment bytes are identical;
- prover-private state never enters a commitment, proof, transcript, schedule
key, or verifier input;
- when a route explicitly exports the legacy portable representation, the
resulting `AkitaCommitmentHint` is identical to today's value;
- transcript events and challenges are identical;
- the final proof value and canonical serialized proof bytes are identical;
- `AkitaBatchedProofShape` and `proof.size()` are identical; and
- the verifier accepts the same valid proofs and rejects the same invalid
proofs.

Backend identity, source path, route, and fusion are execution metadata.
They must not enter a protocol message, descriptor digest, transcript label,
challenge, schedule lookup, or proof-size formula.

No stage in this project may change code in the verifier or change serialized
proof/commitment types to make a differential test pass. A mismatch is a bug in
the new execution path.

### 3.2 Supported paths that must remain available

The implementation must preserve all of these paths:

- dense sources, including cached digit planes and the exact-i16 paths;
- row-major one-hot sources and the multi-source column sweep, at every stored
hot-position index width;
- mixed `MultilinearPolynomial` dispatch;
- recursive witness commitment, with and without tensor packing;
- terminal inner-only commitment;
- setup-prefix commitment and persisted prover setup-prefix state;
- delegated CPU backends that currently forward supported compute kernels;
- sliced and unsliced B commitments;
- every compression-chain length selected by current profiles;
- both `RingRelationMode::QuotientLift` and
  `RingRelationMode::ReducedEvaluation`, including the reduced mode's absence
  of polynomial-modulus quotient rows;
- external trusted schedule artifacts, strict missing-row rejection, and the
  single catalog shared by setup, commitment, proving, and verification;
- all supported field and ring-dimension combinations;
- the exact set of fields for which each CPU stage operation is constructible
(see section 7.1);
- direct and recursive setup-contribution modes;
- base-field and extension-field opening flows;
- subring coefficient packing;
- `parallel` and non-`parallel` execution;
- `disk-persistence`;
- `logging-transcript`;
- `response-model-diagnostics`; and
- both supported transcript feature selections.

Opening, folding, tensor projection, ring switching, proof serialization, and
verification keep their mathematical behavior and verifier-visible data. Their
prover-only input plumbing changes from a concrete hint to generic commitment
state. The CPU adapters reuse today's algorithms. A state-consuming GPU
operation may perform the same derivation against resident state and return the
same canonical downstream witness material without ever constructing a hint.

### 3.3 Performance and memory invariants

The production CPU route must continue to reach each current optimized kernel.
In particular:

- dense coefficient sources borrow existing storage rather than copying the
full polynomial;
- cached dense digits stay cached and use the current cached-digit kernel;
- one-hot sources keep the column sweep and its bounded scratch policy, and
their hot-position indices reach the sweep at their stored width, with no
widening, no re-encoding, and no copy;
- recursive packed digits stay packed until the current kernel decodes a tile;
- B slice construction preserves today's policy: one reusable buffer builds
each slice, `commit_outer_slices` retains the group's completed slice inputs,
and one batched `digit_rows` call consumes them;
- compression keeps `MAX_COMPRESSION_RHS_BATCH`; quotient-lift execution keeps
  the current cyclic/negacyclic quotient construction, while reduced execution
  keeps the current negacyclic-only path and allocates no quotient rows; and
- routing adds no witness-sized persistent copy beyond the one retained inner
image required by later proving; the image may remain in backend-owned
memory and need not also exist as host rows.

Before a legacy path is removed, representative root-commit benchmarks must
show no regression under the profiling procedure in the Akita Book. Run the
base and candidate on the same pinned host, release build, feature set, input,
and warm/cold-cache condition, with at least 10 measured repetitions after
warm-up. Record every sample, median time, peak resident memory, and retained
backend cache bytes. A candidate median more than 3% slower, or any unexplained
peak-memory or retained-cache increase, blocks the cutover until it is fixed or
the user explicitly revises this requirement. Record first-commit and
warm-cache results separately so lazy dense-cache behavior cannot hide a
regression.

## 4. Current commitment contract

The current root path is in
`[api/commitment.rs](../src/api/commitment.rs)` and
`[api/commitment/inner_outer.rs](../src/api/commitment/inner_outer.rs)`.
For a same-shape group it does this:

```text
source polynomials
    │ balanced inner decomposition
    ▼
A × source digits = inner rows t_i
    │ validate t_i; split D_A rows into D_B subcolumns;
    │ balanced outer decomposition; polynomial-major slice packing
    ▼
B × outer digits = uncompressed outer image u
    │ compression maps; quotient rows only in QuotientLift mode
    ▼
terminal payload
    │
    ├── CommittedGroup(profile, Commitment(payload))
    └── prover-private state S
          ├── inner-image state owned by the inner/fused operation
          └── compression state owned by the compression operation
```

`compute_inner_outer_commitment` currently calls a source-typed
`RootCommitKernel`, brings its result back to the API layer, decomposes and
packs it, and calls `DigitRowsComputeBackend` for B. The root `commit()` then
calls the compression executor and constructs an `AkitaCommitmentHint`. That
root path currently uses `RingRelationMode::QuotientLift`. Recursive
commitment passes the schedule-authenticated relation mode into the shared
compression executor; `ReducedEvaluation` uses negacyclic products only and
retains no quotient images.
This concrete assembly is the coupling this design removes from the universal
commitment boundary.

The same concepts are repeated in
`[protocol/ring_switch/commit.rs](../src/protocol/ring_switch/commit.rs)` for
recursive witnesses and in
`[api/setup_prefix.rs](../src/api/setup_prefix.rs)` for setup prefixes. These
copies are a migration risk because geometry, validation, and output assembly
can drift.

The current source/backend extension point is a product of three axes:
concrete source view, const ring dimension, and backend. The downstream-style
test in
`[akita-pcs/tests/commitment_contract.rs](../../akita-pcs/tests/commitment_contract.rs)`
shows the cost: its source needs a `CommitView` GAT and dimension-specialized
`RootCommitKernel` implementations, the source-specific algorithm is attached
to a backend type rather than the source representation, and the call site is
coupled through `RuntimeCommitBackendFor`. Rust's orphan rules do permit a
downstream implementation for `CpuBackend` when a local source-view type is a
trait argument; the forwarding backend in that test is an ownership/design
choice, not a coherence requirement.

The redesign must solve those problems without changing the mathematical
pipeline above.

## 5. Source contract



### 5.1 Source metadata and admission

Introduce a `CommitmentSource<F>` trait that is object-safe and has no const
ring-dimension parameter. The exact field names can change during
implementation, but the contract must contain these concepts:

```rust,ignore
pub trait CommitmentSource<F: Field>: Send + Sync {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError>;

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding;

    /// O(1), side-effect-free declaration. It must not build a lazy cache.
    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError>;

    /// Materialize or borrow only the representation selected by request
    /// compilation.
    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError>;

    /// O(1), side-effect-free external-operation capability declaration.
    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        Ok(None)
    }

    /// Prepare only the external operation selected by request compilation.
    fn prepare_external_inner_commitment(
        &self,
        selected: ExternalInnerCommitmentCapability,
        plan: &CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
        let _unused = (selected, plan);
        Err(AkitaError::InvalidInput(
            "source did not provide the selected external inner commitment capability".into(),
        ))
    }
}
```

`BackendKindId` is an open, process-local identifier, such as a `TypeId` plus
a diagnostic name. It is not a closed Akita-owned enum: downstream code can
define a backend family without an Akita source change. The identifier is
execution metadata and is never serialized or transcript-bound. The executor
builder issues opaque `BackendInstanceId` values; callers do not construct raw
numeric identities.

`CommitSourceDescriptor` contains only O(1) shape and structural data:

- `num_vars`;
- total and live coefficient lengths;
- committed source class, including one-hot chunk size where applicable; and
- a stable source-family name for diagnostics.

Admission remains at the protocol boundary and initially preserves today's
behavior exactly. The code checks source class and centered reach before any
stage operation runs. Dense-like sources perform the same pre-scan they perform
today. Structurally bounded one-hot and packed-digit sources return their exact
constant or stored bounds. This plan does not fuse admission into source
streaming in the first cutover; doing so would change traversal and failure
timing and should be evaluated separately.

The `CommitInnerPlan` used to select a standard polynomial representation or
external inner commitment operation must already exist. Therefore source
resolution happens while compiling a request, after the commitment profile is
known. Executor construction must not attempt to resolve sources from
descriptor-only data.

Capability discovery and materialization are separate. Discovery is O(1), is
side-effect-free, and can advertise more than one lossless representation. The
request compiler intersects those offerings with the selected inner
operation's preferences and declared reductions. It completes setup, profile,
source, and route validation before it calls `represent_as` or
`prepare_external_inner_commitment`. Only the chosen method may build or borrow
a lazy cache. This preserves the current first-commit versus warm-cache
behavior and avoids building a witness-sized dense digit cache for a route that
will not use it.

`AvailablePolynomialTypes` is a small immutable list of `PolynomialType`
descriptors, not borrowed witness data. It includes the storage semantic tag
and any exact plan key or bounded reduction cost needed for selection.
`PolynomialTypeSelection` is an opaque token issued by the request compiler
from one advertised entry; a caller cannot use it to request an unadvertised
form.

At the high-level PCS boundary, `AkitaCommitmentScheme` continues to own one
validated `TrustedScheduleCatalog`. Its constructor validates the catalog's
family and policy binding, and `AkitaCommitmentScheme::commit` continues to
run `validate_config_policy::<Cfg>()` before entering `akita-prover`. The
lower-level commitment API does not duplicate those configuration or catalog
validation gates; it receives the scheme's already-validated catalog.

For a root commitment inside `akita-prover`, preserve today's validation order:

1. reject an empty group, mixed `num_vars`, or source/setup layout overflow;
2. resolve the trusted catalog or explicit commitment profile; trusted-catalog
   resolution builds the same `AkitaScheduleLookupKey` from the final group and ordered
   `PrecommittedGroupProfiles` in `PrecommittedGroupContext`;
3. for `WithPrecommittedGroups`, run the existing
  `ensure_prover_schedule_fits_setup` check on the complete selected schedule;
4. validate the frozen precommit profile and setup geometry/capacity;
5. obtain `Cfg::committed_source_contract()`;
6. validate the declared source class; and
7. validate centered reach under the selected inner digit parameters.

Only after this per-call schedule/profile resolution does the executor compile
its immutable request. A missing trusted-catalog row is an error and never
starts planner search or a compiled-table fallback. The already-resolved
profile is the input to plan construction; catalog lookup, catalog identity,
and precommitted-group policy do not move into the executor or become
executor-construction state.

Descriptor consistency, route capability checks, and selected-representation
materialization follow those existing admission gates. The mode-specific plan
constructors in section 6.1 document and test the corresponding current order
for recursive, terminal, and setup-prefix calls. New errors must occur before
arithmetic and before transcript mutation, but must not move an existing
root-admission failure behind source materialization.

`CommitmentSource::descriptor` and commitment-plan construction use fallible,
checked shape arithmetic. They do not call the default prover-only
`RootPolyShape::num_vars` overflow path. `RootPolyShape` remains unchanged for
opening and tensor code outside this project; this design does not claim to
remove that helper's documented prover-side panic globally.

### 5.2 Standard polynomial types

The first release has three Akita-owned semantic polynomial types:

```rust,ignore
pub enum PolynomialType {
    Dense(DenseType),
    ShortNorm(ShortNormType),
    OneHot(OneHotType),
}

pub enum PolynomialRepresentation<'a, F: Field> {
    Dense(DenseRepresentation<'a, F>),
    ShortNorm(ShortNormRepresentation<'a>),
    OneHot(OneHotRepresentation<'a>),
}

pub enum DenseRepresentation<'a, F: Field> {
    Coefficients(&'a dyn DenseCoefficientSource<F>),
    PredecomposedDigits(PredecomposedDigitPlanes<'a>),
}
```

These three types are all required by today's implementation:

- `DenseType` covers general field-coefficient polynomials. A dense source can
advertise raw coefficients and, when available for the exact plan key,
predecomposed cached digits. `DenseType` records which of those two physical
forms is offered; cached digits remain a dense optimization rather than a
short-norm polynomial.
- `ShortNormType` covers bounded signed coefficients in the existing packed
representation used by recursive witnesses. They may still require balanced
decomposition when the inner plan has more than one digit.
- `OneHotType` covers structurally one-hot data and preserves the one-hot
column sweep. Its representation exposes the complete stored hot-position
slice, chunk size `K`, and variable count.

The semantic type and physical representation are separate deliberately.
There are three polynomial types but four current physical paths: dense
coefficients, dense predecomposed digits, packed short-norm coefficients, and
one-hot positions. Predecomposed digits and packed short-norm coefficients must
never be treated as interchangeable.

No source receives a backend-owned sink or controls backend traversal. Dense
coefficients and short-norm coefficients retain their bounded pull/borrow
semantics. `OneHotRepresentation` is whole-slice rather than ranged, for the
reason given below. A source with only an external inner commitment operation
is not required to fabricate a standard `PolynomialRepresentation`.

A source must borrow, never fill, on every path that has a borrowable
representation today: dense coefficient blocks, cached dense digit planes, and
recursive packed coefficients. A filling accessor is a copy and violates the
memory invariants in section 3.3.

Each registered inner operation declares the polynomial types, physical forms,
and bounded lossless reductions it supports. Every request must resolve to one
supported path before materialization. The CPU operation must implement all
three types and all four current physical forms so existing fast paths do not
regress. A reduction is selected and reported while compiling the execution
plan; it may not appear silently during execution.

The accessor contracts are object-safe and return borrowed storage. Their
final spelling can change, but they must expose these exact semantics:

```rust,ignore
pub trait DenseCoefficientSource<F: Field>: Send + Sync {
    /// Canonical logical coefficients in source order. Plan-owned padding is
    /// represented logically and must not require a full copied buffer.
    fn coefficients(&self) -> &[F];
}

pub struct ShortNormRepresentation<'a> {
    coefficients: PackedSignedCoefficientSlice<'a>,
}

pub struct OneHotRepresentation<'a> {
    positions: UnitPositionSlice<'a>,
    chunk_size: usize,
    num_vars: usize,
}
```

`PredecomposedDigitPlanes` records `D`, digit count, log basis, logical ring
count, physical length, and exact signed bounds. Its borrowed bytes use
`[ring][digit][coefficient]` order. Request compilation accepts it only when
all key fields equal the selected plan.

`PackedSignedCoefficientSlice` records the borrowed encoded bytes, logical and
commitment-aligned coefficient lengths, signed bit width, block encoding, load
padding, and exact signed bounds. Its checked constructor makes the current
packed encoding public as a polynomial representation without exposing mutable
storage.
The CPU operation recovers a monomorphic view once, then uses the current
bounded tile decoder. It does not decode the whole witness or confuse packed
source coefficients with already-decomposed planes.

Materialization calls an object-safe accessor at most once per selected source
before entering a const-D hot kernel. It converts the returned borrowed value
to a concrete enum variant and monomorphic view. No trait-object call occurs
inside a per-block, per-ring, per-row, or per-position loop.

Repeated facts have one authority. Request compilation checks that dense and
short-norm coefficient lengths, one-hot `chunk_size`/`num_vars`, and exact
bounds agree with `CommitSourceDescriptor` and the selected plan before
arithmetic. A source-family diagnostic name is never used for type safety.

#### Unit positions keep their stored index width

The one-hot payload is one flat slice plus two scalars. `OneHotPoly` stores
`Vec<Option<I>>` for `I` in `u8`, `u16`, `u32`, `usize`, and the current view
exposes exactly `indices()`, `onehot_k()`, and `num_vars()`. The sweep derives
its own blocking from the plan; it never asks the source for a block range.

Two properties of that storage are load-bearing and must survive erasure:

- **The width.** Narrow indices are a deliberate footprint choice. Widening
`u8` or `u16` positions to a common type would multiply traffic in the
hottest loop of the commit path.
- **The** `Option`**.** `None` denotes an all-zero chunk. An accessor that returns
bare positions cannot represent one.

Therefore `OneHotRepresentation` carries the width in its returned
`UnitPositionSlice` rather than in a type parameter and borrows the stored slice
unchanged:

```rust,ignore
/// Borrowed one-hot chunk indices at their stored width.
/// `None` denotes an all-zero chunk.
pub enum UnitPositionSlice<'a> {
    U8(&'a [Option<u8>]),
    U16(&'a [Option<u16>]),
    U32(&'a [Option<u32>]),
    Usize(&'a [Option<usize>]),
}
```

`OneHotType` and `OneHotRepresentation` deliberately have no `F` parameter.
A field bound on either would encode one backend's arithmetic requirement into
an Akita-owned representation class and force every other backend to inherit
it. Backend requirements belong on backend implementations; see section 7.1.

Adding a width to `UnitPositionSlice` is a breaking change to every backend
that consumes the kind, so the four widths above are the frozen set for this
release. A source storing indices in any other width converts at construction,
not while `represent_as` is borrowing the selected representation.

#### Where source validation happens

Today `OneHotPoly::new` range-checks each index against `K` at construction,
and `source_view::<D>()` validates the ring dimension before releasing a view.
Under this contract the per-index range check stays at source construction, and
the ring-dimension check moves into `represent_as`, which receives the resolved
`CommitInnerPlan`. Neither check is dropped, and neither is re-run per block.

### 5.3 External inner commitment operations

The external-operation path preserves a source's hand-written algorithm for
a particular backend family.
`external_inner_commitment_capability` advertises support without doing work.
After selection, `prepare_external_inner_commitment` returns a
`PreparedExternalInnerCommitment` containing a type-checked family identifier,
borrowed payload, and borrowed operation object. The source itself does not
have to implement the operation. It can return distinct adapter objects for
different backend kinds. This avoids assuming that one Rust trait
implementation on the source can represent several backend-specific
algorithms.

The external inner commitment operation receives:

- the resolved `CommitInnerPlan`, including the runtime ring dimension;
- the sources in original group order;
- a backend-owned, type-erased external-operation context; and
- the setup identity already checked by executor construction.

It returns canonical `CommitInnerWitness<F>` values in source order. It has no
access to the transcript and cannot implement B or compression. It dispatches
the runtime ring dimension inside its own code.

A source with only an external-operation path is supported on the backend
kinds for which it supplies external inner commitment operations. Request
compilation rejects any other route before arithmetic. A source with both
standard and external paths is tested for equality.
`ExternalInnerCommitmentPolicy::Forbid` forces the standard polynomial path for
differential testing and incident diagnosis.

An external selection used by a fused route must also declare that it can
encode its work into that fused backend's one command/request builder. Request
compilation rejects an external selection without this capability. The fused
operation must not call an ordinary external inner commitment operation as one
submission and then submit B separately, because that would satisfy the Rust
signature but violate R3.

The external-operation API uses checked existential erasure rather than a general
`CommitmentSource::as_any` hook:

- `ExternalInnerCommitmentCapability` is an opaque token returned by
side-effect-free discovery. It contains an open `BackendKindId`, a
`TypeId`-backed external
family identity, an algorithm identity, and a diagnostic name.
- `PreparedExternalInnerCommitment` contains that token, a borrowed erased
payload, and a borrowed `ExternalInnerCommitmentOperation<F>` object. Only
its checked constructor can pair those values.
- The request compiler groups selections only when backend kind, family type,
algorithm identity, and required context type all agree. It never groups or
downcasts based on a string.
- `ExternalInnerCommitmentOperation::commit_group` validates every token and
uses safe `Any::downcast_ref`; a mismatch returns `AkitaError` before backend
work. Erased payloads are not exposed through the general source interface.
- A fused-capable selection additionally contains a
`ExternalFusedInnerCommitmentEncoder`. The encoder appends A work to the fused
backend's in-progress command builder but does not submit it. The fused
operation then appends decomposition, slicing, and B work and performs the
one submission. Request compilation rejects an external selection whose
encoder and fused command-context `TypeId`s do not agree.

All of these types are public and object-safe because downstream source and
backend crates must be able to implement them. Their data fields remain
private; checked constructors and read-only accessors preserve the type and
identity invariants. This is safe type erasure, not unchecked downcasting.

### 5.4 Mapping current sources


| Current source           | Standard/external mapping                                                                              | Production path that must remain                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DensePoly`              | `DenseType`, as coefficients or exact-key predecomposed digits; only the selected form is materialized | cached-digit, exact-i16, or dense NTT kernel selected by current CPU policy; no unused lazy cache build                                                         |
| `OneHotPoly`             | `OneHotType` at its stored index width                                                                 | `column_sweep_ajtai_onehot_multi` with the current scratch bound, monomorphic in that width                                                                     |
| `RecursiveWitnessFlat`   | `ShortNormType` with packed signed coefficients                                                        | current packed recursive commitment kernel; live length excludes commitment-only padding and multi-digit plans still decompose source coefficients exactly once |
| `MultilinearPolynomial`  | its dense or one-hot member's polynomial type                                                          | current ordering and group semantics; no protocol-level group merge                                                                                             |
| downstream dense wrapper | `DenseType` with coefficients                                                                          | no downstream forwarding backend                                                                                                                                |
| Jolt packed trace        | external CPU inner commitment, with an optional standard `OneHotType` reference if practical           | Jolt's existing hand-tuned packed trace commit                                                                                                                  |


This contract does not combine groups that the protocol currently commits
separately. In particular, Jolt's dense, row-major one-hot, and packed-trace
groups keep their current group boundaries and transcript absorptions.

## 6. Stage contracts and executor



### 6.1 Canonical plans

Compile one checked commitment execution plan through a mode-specific
constructor. Root, recursive, terminal, and setup-prefix callers do not begin
with the same protocol type and must retain their current admission and
padding semantics:

- `CommitmentExecutionPlan::for_root` consumes the already-resolved
  `GroupCommitPhaseParams`, the config's committed-source contract, and root
  source metadata; it does not receive `GroupContext` or a catalog;
- `for_recursive` consumes checked `CommittedGroupParams`, fold level,
  tensor-packing choice, and commitment-aligned physical length; source
  encoding, payload mode, and `RingRelationMode` are read only from those
  params rather than passed as duplicate authorities;
- `for_terminal` consumes `TerminalFoldParams` and produces an inner-only
plan; and
- `for_setup_prefix` consumes the exact `SetupPrefixSlotId` and derives its
  frozen commitment profile, natural length, and padded prefix length through
  the existing checked accessors rather than accepting duplicate values.

These constructors yield the same canonical arithmetic subplans. They do not
force root source-class/reach admission onto recursive or setup-owned sources
that do not perform that check today. Stage plans are borrowed views of the
result; a stage must not re-derive geometry independently.

- `CommitInnerPlan`: the existing public plan, extended with `D_A` and live
block count while retaining `n_a`, positions per block,
inner digit count, and inner log basis.
- `OuterCommitPlan`: `n_b`, `D_B`, outer digit settings, and the exact
`CommitmentSliceGeometry`.
- `CompressionChainPlan`: the current canonical compression plan without
modification.
- `UncompressedCommitPlan`: the checked inner and outer views together.

`RingRelationMode` remains separate from `CompressionChainPlan`, as it is in
the current code. A full commitment execution plan carries both values when
compression runs. Root and setup-prefix construction preserve today's
`QuotientLift` behavior. Recursive construction copies
`CommittedGroupParams::ring_relation_mode`; terminal and uncompressed modes do
not run commitment compression. No backend may infer the mode from whether
quotient storage happens to be present.

`CommittedSourceEncoding` and the descriptor's committed source class are
independent. The former is protocol-selected physical source encoding
(`CanonicalCoefficientTable` or `TensorSubfieldProjection`); the latter is the
SIS/admission class such as balanced signed digits or structurally unit one-hot.
Root plans retain the existing `CanonicalCoefficientTable` requirement.
Recursive plan construction validates `CommittedSourceEncoding`, performs or
selects the existing tensor packing when requested, and then resolves a
polynomial representation for the resulting physical commitment source. The source descriptor does not
duplicate or reinterpret the schedule's encoding field.

The complete plan also binds group layout and source count, committed-source
contract when applicable, live and physical source extents, setup descriptor
and capacity, expected inner witness count and row shape, expected `u` ring
dimension and coefficient length, SIS modulus profile, and compression source
length. The compression plan is derived before arithmetic from checked
`slice_count * n_b * D_B` geometry. The executor verifies that returned `u`
matches that exact precomputed length before compression.

Use `akita_error::checked` and existing checked layout constructors for all
generic `usize` arithmetic. Do not add local copies of checked multiplication,
sum, range, division, alignment, or power-of-two formulas.

### 6.2 Backend state and generic commit results

Witness-sized stage results are backend-owned state, not mandatory host
vectors. Introduce session-issued, store-owned, generation-checked references:

```rust,ignore
pub struct BackendStateRef<K> { /* private store, owner, slot, generation */ }
pub enum InnerImage {}
pub enum CompressionState {}
pub enum CompositeCommitmentState {}

pub struct InnerCommitOutput {
    image: BackendStateRef<InnerImage>,
}

pub struct UncompressedCommitmentOutput<F: Field> {
    image: BackendStateRef<InnerImage>,
    u: RingVec<F>,
}

pub struct CompressionStageOutput<F: Field> {
    terminal_payload: RingVec<F>,
    state: BackendStateRef<CompressionState>,
}

pub struct CommitOutput<F: Field, S> {
    pub committed_group: CommittedGroup<F>,
    pub prover_state: S,
}
```

`CompressionStageOutput` binds the requested `RingRelationMode` into its state
metadata. In `QuotientLift` mode the state retains the checked packed witness
and exactly one quotient image per compression map. In `ReducedEvaluation`
mode it retains the checked packed witness and an explicit reduced-mode marker,
with no quotient allocation. The terminal payload has the same shape in both
modes.

`BackendStateRef<K>` has no public raw-ID constructor and gives callers no
`Any` downcast. The state store and owning operation jointly issue it after
checking the setup descriptor, plan identity, group layout, ring dimensions,
`RingRelationMode` when applicable, backend instance, store identity, and
generation. A foreign, stale, wrong-kind, wrong-plan, or cross-mode reference
returns `AkitaError` before a consumer performs
arithmetic or any further transcript mutation that depends on it. Cloning a
built-in reference clones a lease; dropping the last lease
releases the slot according to the selected cache/state policy.

The marker `K` identifies the semantic state kind, not its representation. A
CPU slot can own `Vec<CommitInnerWitness<F>>`; a GPU slot can own device buffers
and event/fence state. Neither representation is exposed through the generic
reference.

`CommitOutput<F, S>` deliberately places no bound on `S`. The API that moves a
state between threads can require `Send`; an API that shares it can require
`Sync`; a persistence API can require a portable-export operation. The root
commit operation itself does not require `S: AkitaSerialize`, `S: Clone`, or
`S = AkitaCommitmentHint<F>`. Useful configurations include:

- `S = BackendStateRef<CompositeCommitmentState>` for resident CPU, GPU, or
mixed-owner state;
- `S = AkitaCommitmentHint<F>` for the portable CPU compatibility path; and
- `S = ()` for commit-only use or for a proving route that explicitly
recomputes private state from the still-available source polynomials.

The standard proof stacks use one `akita-prover` type,
`ProverOpeningState<F>`, as `S`. Its private representation can own a portable
hint, a composite resident reference, or an explicit no-retained-state marker.
This gives tiered and heterogeneous stacks one Rust type while preserving
operation-specific capability checks. It is not
`Option<AkitaCommitmentHint<F>>`: mixed inner/compression ownership, lifecycle
binding, and recomputation policy are first-class states rather than
route-dependent `None` handling. A side-effect-free observation may report
that portable state is already present; it must never trigger a transfer
implicitly.

The composite state record holds the inner-image reference, an optional
compression-state reference, the execution mode, the applicable
`RingRelationMode`, and the full setup/plan/group binding. It can refer to
different physical owners. It never becomes a protocol message.

Every type named by a public operation signature is itself public and
nameable. In particular, `ResolvedCommitSource<'a, F>` is an opaque checked
value with accessors for its selected standard polynomial representation or
external inner commitment, source order, and descriptor. Only the request
compiler constructs it. External operations can inspect and execute the
selected representation but cannot forge an admitted source. Plans,
references, and stage outputs similarly use checked constructors and accessors
without acquiring protocol serialization.

### 6.3 Object-safe operations

Define object-safe stage traits whose implementations dispatch `D_A` or `D_B`
inside their method bodies:

```rust,ignore
pub trait InnerCommitOperation<F: Field>: Send + Sync {
    fn commit_inner(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError>;
}

pub trait OuterCommitOperation<F: Field>: Send + Sync {
    fn commit_outer(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &OuterCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError>;
}

pub enum InnerImageInput<'a, F: Field> {
    Owned(&'a BackendStateRef<InnerImage>),
    HostRows(&'a [RingVec<F>]),
}

pub trait CompressionOperation<F: Field>: Send + Sync {
    fn compress(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
        u: RingVec<F>,
    ) -> Result<CompressionStageOutput<F>, AkitaError>;
}

pub trait FusedInnerOuterOperation<F: Field>: Send + Sync {
    fn commit_inner_outer(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &UncompressedCommitPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError>;
}
```

These signatures are normative at the level of ownership and data flow, not
final spelling. Two details are mandatory:

1. Split and fused operations receive the same resolved source representation.
  The fused trait must not bypass admission or receive a different raw
   polynomial representation list.
2. The fused operation returns an inner-image state reference and `u`. It does
  not return host inner rows and does not construct a hint.

`StateRegistrationContext` is issued by the executor for exactly one call. It
is the checked public factory that lets an external operation register
backend-owned state and receive a `BackendStateRef<K>`. Registration supplies
the operation's builder-issued owner identity, immutable shape/binding
metadata, retained-byte accounting, and cleanup policy. The context checks
these facts against the active plan. External code never chooses store IDs,
slots, generations, or semantic marker kinds directly.

An external owner must still recover its own private buffer. The builder gives
each registered owner an unforgeable `StateOwnerCapability<K>`.
`StateRegistrationContext::reserve_owned` validates that capability and returns
a pending deposit containing a new `OwnerStateToken`. The operation uses that
opaque token as a key in its private table, installs its buffer, and seals the
deposit with its cleanup callback; dropping an unsealed deposit cancels the
reservation. The token is not a device pointer and its contents are not
publicly inspectable. Later,
`BackendStateRef<K>::resolve_for_owner(&StateOwnerCapability<K>)` checks owner,
kind, store, generation, setup, and plan binding before returning that opaque
token. A different owner cannot resolve it. The matched external operation uses
the token to find its buffer in its own table. Cleanup invokes the owner-
registered release callback for the same token. This mechanism uses neither
`Any` nor a cross-owner downcast.

Cross-owner transfer is a separate optional capability:

```rust,ignore
pub trait InnerImageExportOperation<F: Field>: Send + Sync {
    fn export_inner_rows(
        &self,
        plan: &CommitInnerPlan,
        image: &BackendStateRef<InnerImage>,
    ) -> Result<Vec<RingVec<F>>, AkitaError>;
}
```

The executor requires this capability only when the selected split route moves
an inner image to an outer operation that cannot consume the owned reference,
or when an explicit portable export needs host rows. A fused GPU route and a
same-owner split route do not acquire this bound.

Each CPU operation object owns or borrows `CpuBackend` and its prepared setup.
A GPU or remote operation owns its equivalent queue, allocator, transport, and
prepared setup. The object-safe boundary does not require a common wire format.

The executor's private stage registration stores operation identity, open
backend-kind identity, setup descriptor, polynomial-type preferences and
reductions, external-operation context, dimension capabilities, and resource
control. Keeping this metadata out of the operation traits is what lets the
four arithmetic capabilities remain independent.

### 6.4 Executor construction and per-request resolution

`CommitmentExecutor<F, StatePolicy>` contains private validated registrations
for:

- one inner operation;
- one outer operation;
- one compression operation;
- an optional explicit fused inner/outer operation;
- optional inner-image transfer and state-consumer registrations;
- `ExternalInnerCommitmentPolicy`; and
- one state policy whose associated `State` becomes the `S` in
`CommitOutput<F, S>`.

Provide a direct CPU constructor from `(&CpuBackend, &CpuPreparedSetup<F>, &AkitaExpandedSetup<F>, state_policy)`. It registers the
same prepared CPU inner, outer, compression, resource, and state operations as
the CPU stack constructor. Commit-only callers, setup-prefix generation, and
profiling code use this constructor without fabricating opening, tensor, or
ring-switch contexts. This is constructor assembly on the canonical executor,
not a second commitment API or routing wrapper.

The executor builder always requires the three individual stages because the existing
terminal-inner path needs inner alone and uncompressed recursive paths may stop
before compression. Supplying a fused operation changes only full
inner-plus-outer execution. Commit-only construction does not require a later
state consumer. A `ProverComputeStack` that will open the output joins the
executor with the selected state-consumer registrations and rejects an
incomplete end-to-end route before proving starts.

Executor construction checks that every stage was prepared from the same setup
descriptor. It validates operation identity, open backend-kind identity,
resource declaration, and the internal coherence of advertised dimensions,
polynomial representations, reductions, and external-operation contexts. It records whether fusion was
explicitly selected. It does not inspect sources, validate a selected plan, or
choose a polynomial type because those values do not exist yet.

For each call, `compile_commitment_request` receives the checked commitment
plan, executor, and actual sources. Before arithmetic it:

1. checks source count, arity, live length, class, and centered reach;
2. obtains only side-effect-free polynomial-type and external-operation
  capability declarations using the actual `CommitInnerPlan`;
3. applies `ExternalInnerCommitmentPolicy`, operation preferences, and backend
  capabilities;
4. resolves one declared lossless polynomial representation or external path
  per source;
5. validates the selected dimension and every repeated descriptor fact;
6. rejects an unsupported source/backend pair before materialization;
7. borrows or builds only the selected representation;
8. records source order, selected polynomial type or external family, encoding,
  and any host transfer; and
9. returns `ResolvedCommitSource` values used unchanged by split or fused
  execution.

Every execution emits one sanitized `tracing::debug!` route summary inside the
executor's call span. It records the selected inner, outer, and compression
operations, whether fusion was selected, each source's path/encoding,
declared reductions, and transfer edges with byte counts. It does not log
payloads, raw handles, or unredacted owner identities. This uses the existing
tracing infrastructure; it does not add a public diagnostics API. Recording
fake operations remain the normative mechanism for kernel-selection and
transport-order tests.

There is no fallback after a stage starts. A runtime error is returned as an
error; it does not select another source path or route.

Fusion is never inferred from equal backend IDs. An executor with the same
backend in `inner` and `outer`, but no explicitly selected fused operation,
executes two stage calls and reports `fused = false`.

The initial transfer representation is explicit host data. A split inner stage
returns an owned state reference. If the next stage has the same owner and
accepts that state kind, it receives `InnerImageInput::Owned`. Otherwise, the
compiled route must contain an `InnerImageExportOperation` edge and it receives
validated `HostRows`. The outer stage returns host `u`; its size is checked
against `MAX_COMPRESSION_INPUT_BYTES` before compression. A device-to-host
transfer is not free, even when bounded, so route diagnostics record its bytes.
Fusion removes the A-to-B transfer and does not add a post-B inner-row download.

Every registration carries either a type-erased resource controller adapted
from `ComputeBackendSetup` or an explicit no-resources declaration. A and B
NTT requirements gain a `CommitmentNttStage::{Inner, Outer}` discriminator.
The executor routes A to the inner registration and B to the outer
registration, or routes both to the fused registration when fusion is
selected. It preserves `ensure_ntt_slot`, retained-versus-streamed policy,
planned bytes, compression cache diagnostics, cache release, and physical
owner deduplication. Appendix A.7 through A.9 specifies this lifecycle and its
stack integration.

### 6.5 Composite execution

One canonical executor owns public-output and private-state assembly:

```text
executor.execute_full(plan, resolved_sources):
  if the executor explicitly selects fusion:
      inner_outer = fused.commit_inner_outer(state_ctx, uncompressed_plan, sources)
  else:
      inner = inner.commit_inner(state_ctx, inner_plan, sources)
      inner_input = same-owner reference or declared checked export
      u = outer.commit_outer(state_ctx, outer_plan, inner_input)
      inner_outer = { inner, u }

  validate the inner-state binding and u ring dimension/logical row count

  compression = compression.compress(state_ctx, compression_plan, relation_mode, u)
  validate the compression-state binding and terminal payload shape

  components = { inner_image, compression_state, plan_binding }
  state = state_policy.bind_full(components, state_dispatch)
  return CommitOutput { committed_group, prover_state: state }
```

The executor also exposes two intentional protocol modes:

- inner plus outer without compression, for an existing uncompressed recursive
payload; and
- inner only, for `commit_terminal_w`.

These modes share the canonical inner and inner-plus-outer functions. They do
not duplicate geometry or commitment arithmetic.

All host-visible boundaries validate exact source count, row count, ring
dimension, slice count, relation mode, and payload shape. Resident state is
created only by a checked owner that records the same shape facts. Any later
export validates the exported rows, witnesses, and mode-appropriate relation
data before use: quotient count and dimensions in `QuotientLift`, and the
absence of quotient rows in `ReducedEvaluation`. The existing checks in
`validate_commit_inner_shape`, `commit_outer_slices`, and compression execution
move behind these boundaries; they are not weakened or deleted.

### 6.6 State consumption is capability-specific

Do not add one supertrait that requires every `S` to expose host rows,
compression witnesses, terminal binding data, cloning, and serialization.
Such a bound would recreate the `AkitaCommitmentHint` requirement under a new
name and would make a resident-only GPU implementation impossible.

Instead, add independent operations at the actual consumer boundaries:

- `OuterCompressionStateOperation<F, S>` obtains the first-map source,
  compression witness, and mode-appropriate relation data needed by
  ring-relation construction;
- `InnerRelationStateOperation<F, S>` derives the canonical inner-relation
material used by `ring_switch::coeffs`, either in place or through a checked
export;
- `TerminalBindingStateOperation<F, S>` produces the one canonical field
segment that the terminal transcript already absorbs;
- `PortableCommitmentStateExport<F, S>` and the matching import adapter
explicitly convert to/from `AkitaCommitmentHint<F>` for compatibility and
setup-prefix persistence; and
- `RecomputeCommitmentStateOperation<F, S>` is an explicit fallback for a
state such as `()`, using the still-bound source group and the same checked
commitment plan.

These traits are schematic names for distinct capabilities; their final
methods must take the canonical checked plan/binding types and return the
existing canonical downstream witness carriers. They must not expose device
pointers or add serialization to resident state. A proving entry point adds
only the operation bounds required by the path it actually executes. Executor
and stack construction validate registrations and owner compatibility, but
cannot choose schedule-dependent transitions.

`SelectedProverOpeningData::from_committed_claims` first resolves the exact
committed profiles to an `OpeningScheduleSelection`. `batched_prove` then
resolves that selection through the same trusted catalog, applies
`effective_batched_schedule`, validates nonterminal execution and setup
capacity, and derives `NttExecutionRequirements`. Only after that effective
schedule exists may proving compile every per-level state consumer, transfer,
or recomputation edge. This full route validation and the existing NTT
prewarming complete before `bind_transcript_instance_descriptor`; no
schedule-dependent fallback is allowed after transcript binding.

The standard `ProverOpeningState<F>` uses a private representation, so external
operations never match on it. A built-in `CommitmentStateDispatcher<F>` reads
the private semantic component, validates its binding, and selects an
operation from a builder-populated `(owner_id, state_kind)` registry. A
resident inner operation receives `&BackendStateRef<InnerImage>`; a resident
compression operation receives `&BackendStateRef<CompressionState>`. The
builder issues every registry key, rejects duplicates, and proves that the
operation owner matches the reference owner. A portable variant goes directly
to the built-in CPU implementation. A no-retained-state variant goes only to
the explicitly registered recomputation path. Thus a third-party GPU neither
inspects a private enum nor downcasts another owner's payload.

Portable construction works at the component boundary, before an `S` exists.
Register independent `PortableInnerStateExport<F>` and
`PortableCompressionStateExport<F>` operations. The former yields the checked
canonical rows; the latter yields the checked compression witness plus either
quotient rows or an explicit reduced-evaluation marker. The portable state
policy consumes `CommitmentStateComponents`, invokes those owner-dispatched
exporters, and constructs exactly one `AkitaCommitmentHint` in the current
mode-specific encoding. The resident policy instead binds the same components
without export. This removes the circular requirement to export a hint from an
already-created generic state.

The CPU compatibility adapter uses the existing hint APIs without changing
their encoding. Quotient-lift export/import uses
`new_with_outer_compression` or `singleton_with_outer_compression` together
with `outer_compression_witness` and `outer_compression_quotients`. Reduced
recursive export/import uses `singleton_with_reduced_outer_compression` and
`reduced_outer_compression_witness`; it must reject any retained quotient row.

Each exporter offers a consuming path when the caller owns the last lease and
a borrowing path when shared state must remain usable. The consuming path moves
CPU vectors and releases device state as soon as transfer completes. The
borrowing path may temporarily overlap resident and host copies; it must report
the transient bytes, release the temporary promptly, and pass the peak-memory
gate. No policy retains both copies persistently. Setup-prefix persistence uses
the consuming path when ownership is available.

The built-in CPU state operations call the current hint/row algorithms and are
differentially tested against them. A GPU operation can consume its own
`BackendStateRef` and return the same downstream material without ever forming
an `AkitaCommitmentHint`. An adapter may still export a byte-identical
`AkitaCommitmentHint` for compatibility or persistence, but that is
an optional capability, not the commitment contract.

## 7. CPU and fused implementations



### 7.1 CPU inner operation

The CPU inner operation groups resolved standard sources by polynomial type
and, for `OneHotType`, by stored index width. Sources using external inner
commitment operations are grouped by external family. Original source order is
preserved in the result.

- `DenseType` coefficients reach the current dense coefficient kernels, while
its predecomposed representation reaches the cached dense kernel.
- `ShortNormType` packed coefficients reach the recursive packed kernel without
expanding the full witness or skipping required multi-digit decomposition.
- `OneHotType` sends all sources of one width through one multi-source column
sweep.
- external families receive one group call per family and the CPU external
operation context.

Do not move opening traits into this source contract. Existing
`RootOpeningSource`, fold, batch, tensor, and coefficient-packing traits remain
in place.

#### Dispatch structure

Request compilation resolves standard versus external execution, polynomial
type, physical representation, and external family before a hot kernel begins. The CPU operation
performs one ring-dimension dispatch per committed group with the existing
`dispatch_for_field!` mechanism. Runtime-to-const dispatch moves from
`api/commitment/inner_outer.rs` and setup-prefix callers into the prepared CPU
operation implementations. The separate API-layer dispatch currently used
only to call the const-`D` centered-reach method disappears because
`CommitmentSource::committed_centered_reach` is D-free. External backends may
use any equivalent checked internal dispatch. Each
`OneHotType` subgroup performs at most one match on `UnitPositionSlice` to
recover its concrete index type. Everything below those outer matches is
monomorphic and compiles to the same code as today. No `dyn` call may appear
inside a per-block, per-row, or per-position loop.

The sweep is monomorphic in the index type, so one call handles one width.
Today a committed group is a slice of one source type, so its width is fixed —
`MultilinearPolynomial` fixes it too — and the single-width case is the norm.
The operation must still handle a multi-width group defensively: one sweep per
width, results reassembled into original source order.

#### The sweep signature changes

`column_sweep_ajtai_onehot_multi` currently takes `&[OneHotView<'_, F, D, I>]`,
and `OneHotView` borrows a concrete `OneHotPoly<F, I>`. A downstream one-hot
source that is not an `OneHotPoly` cannot construct that view, so R1 is not
satisfied for the kind until the kernel accepts a view built from the selected
one-hot representation alone:

```rust,ignore
pub struct OneHotSource<'a, I: OneHotIndex> {
    pub indices: &'a [Option<I>],
    pub chunk_size: usize,
    pub num_vars: usize,
}
```

The commitment sweep takes `&[OneHotSource<'_, I>]`. `OneHotView`
remains unchanged for opening and fold kernels, which still use it. The
one-hot commitment adapter adds a conversion/accessor that constructs
`OneHotSource` from the same borrowed fields. This is a signature
refactor with no change to the sweep's algorithm, scratch policy, or output,
and it is a required Stage 2 work item rather than a free reuse of the existing
kernel.

#### Backend arithmetic bounds stay on the implementation

The column sweep requires more of `F` than the stage trait declares:
`WithCommitAccumulator` for its deferred-reduction accumulator, on top of the
`Unreduced` and `Wide: AdditiveGroup + From<F>` that `commit()` already
requires today. `InnerCommitOperation<F>` stays at `F: Field`; the CPU
implementation adds `CanonicalEncoding` and its algorithm-specific bounds,
which are erased at the trait-object boundary:

```rust,ignore
impl<F> InnerCommitOperation<F> for PreparedCpuInnerWithOneHot<'_, F>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator,
{
    // Includes the common paths. OneHotType resolves D and stored index
    // width, then calls the current column sweep.
}
```

Putting this bound on `PolynomialRepresentation` or `OneHotType` instead is not
an option. Rust applies enum bounds to every variant, so a bound placed there
propagates through `CommitmentSource<F>` to `commit()` and to every source and
backend, including ones that never touch a one-hot representation.

`WithCommitAccumulator` is implemented for `Fp32<P>`, `Fp64<P>`, and
`Fp128<P>`, and the three in-tree preset protocol fields are aliases of those.
A test-only compile assertion instantiates the one-hot adapter's bound at each
in-tree preset field.

Do not put this bound on the only constructible CPU inner operation. Today
dense and recursive CPU commitment require only `Field + CanonicalEncoding`,
and `CommitmentConfig::Field` is an open associated type. Provide a common
coefficient/packed CPU inner adapter at the current weaker bound and a full
adapter at the stronger bound that contains the common paths plus
`OneHotType`. In-tree preset executors register the full adapter. A custom
field that lacks `WithCommitAccumulator` can register the common adapter and
retains its current dense and recursive commitment paths; it receives a
plan-time unsupported-polynomial-type error only if it requests one-hot
execution.

These adapters are concrete over `CpuBackend` and `CpuPreparedSetup<F>`; a
`DigitRowsComputeBackend` bound alone cannot reach the current source-typed A
kernels or the one-hot matrix view. Stage 2 first refactors those CPU A entry
points to accept the resolved polynomial representations, then implements the CPU
operation objects over those entry points. Delegating CPU backends register or
delegate the resulting operation objects. They do not rely on a fictitious
generic `B: DigitRowsComputeBackend` inner adapter.

### 7.2 CPU outer operation

The CPU outer operation owns the current logic now split across
`compute_outer_commitment`, `decompose_commit_blocks_into`, and
`commit_outer_slices`. It:

1. validates inner shapes;
2. projects D_A rows into D_B subcolumns;
3. decomposes with the profile's outer digits;
4. packs canonical polynomial-major slices with exact zero padding;
5. calls the current optimized digit-row kernel for B; and
6. returns D-erased `u`.

There must be one production implementation of slice ordering and geometry.

### 7.3 CPU compression operation

The CPU compression operation wraps the current
`execute_compression_chains` path. It returns the terminal payload and stores
the compression witness and `CompressionRelationOutput` equivalent in a
checked CPU-owned `CompressionState` slot. It passes the plan's explicit
`RingRelationMode` to the current executor: quotient-lift mode retains one
quotient per map, while reduced-evaluation mode uses the negacyclic-only kernel
and retains none. It keeps every current compression map, batch bound, shape
check, and cache policy. The portable CPU state adapter can later assemble the
same mode-specific `AkitaCommitmentHint`; normal resident execution does not
have to do so at commit time.

Do not remove the current `DigitRowsComputeBackend` to
`CompressionComputeBackend` relationship globally as part of the first
cutover. `DigitRowsComputeBackend` is also used by opening and ring-relation
code. The new stage traits already let a future outer-only backend compile
without implementing the new compression operation. A broader compute-trait
cleanup needs its own blast-radius review.

### 7.4 Production fused operation

A production fused operation receives the same checked plan and resolved
sources as the split route. It performs the same source grouping, A product,
D_A-to-D_B projection, balanced decomposition, slice packing, and B product.

For a device backend, inner rows remain device-resident until all required B
slices finish. The implementation then returns:

- a checked `BackendStateRef<InnerImage>` for those resident rows; and
- canonical host `u`.

A remote implementation can use any private request encoding, but one call to
`commit_inner_outer` must correspond to one recorded transport submission.
Retries must be explicit and idempotent; a hidden split into an A response and
a later B request violates R3.

## 8. Caller migration



### 8.1 Root `commit()`

Change the result to `CommitOutput<F, S> { committed_group, prover_state }`.
Resolve the same profile and admission contract, compile one commitment
request, call the composite executor, and return the state selected by its
state policy. Root commitment does not require `S` to be a hint or serializable.

The low-level root API accepts both the scheme's
`&TrustedScheduleCatalog` and `&CommitmentExecutor` directly. The catalog
resolves `GroupContext::Scheduler`; the executor begins only after that
resolution. A caller that already owns a `ProverComputeStack` passes
`stack.commitment()`; a commit-only caller constructs or reuses an executor
without fabricating opening, tensor, and ring-switch contexts. Recursive
proving obtains the per-level executor from `LevelProveStacks`. These are two
uses of the same executor type, not two root commit APIs. The executor never
becomes a second catalog owner or a planner fallback.

The public Rust signature changes in the cutover because the repository has no
backward-compatibility promise. The source set, validation, commitment bytes,
proof bytes, and verifier behavior remain. Update
`AkitaCommitmentScheme::commit` and all in-repository callers atomically so no
temporary API reintroduces a mandatory hint.

### 8.2 Recursive `commit_w`

Keep tensor packing and generalize `NextWitnessStateOutput` over its private
state. Replace the copied inner, outer, and compression arithmetic with the
canonical executor mode selected by `payload_mode`.

- compressed mode returns the same outer payload and generic state containing
or referencing both inner and compression material;
- uncompressed mode returns the same `u` and generic inner state; and
- packed and unpacked recursive sources preserve the same physical witness.



### 8.3 Terminal `commit_terminal_w`

Use the executor's inner-only mode. Keep
`NextWitnessState::TerminalInnerState`, retain generic inner state, and obtain
the canonical `TerminalTFieldsMessage` through the terminal-binding operation
before the transcript needs it. This message contains exactly the field segment
already absorbed today; it is not a hint and does not enter the proof as a new
field. Do not run B or compression.

### 8.4 Setup-prefix commitment

Keep `SetupPrefixSlot<F>`, `SetupPrefixPublicCommitment`, and the existing
serialized `AkitaCommitmentHint<F>` format unchanged. Backend-resident state
lives in `akita-prover`; it must not move into `akita-types` or make the setup
artifact depend on a backend crate.

`AkitaCommitmentScheme::setup_prover` continues to pass its owned
`TrustedScheduleCatalog` to `akita_setup::new_prover_setup`. Setup-prefix slot
enumeration remains `setup_prefix_slot_ids_from_catalog`; neither the executor
nor a backend may infer required slots from capacity alone. A loaded persisted
registry is checked against the exact slot IDs required by that same catalog,
and the disk registry filename remains namespaced by `catalog_digest`, field,
and capacity. This prevents a valid registry for one trusted artifact from
being reused under another.

Build a dense source over the exact shared-matrix prefix and use the canonical
full executor. `commit_setup_prefix` no longer exposes `const D` or a generic
backend/prepared pair: it accepts the checked `SetupPrefixSlotId` and a
prepared commitment executor, derives the profile, natural/padded lengths, and
runtime dimension through `CommitmentExecutionPlan::for_setup_prefix`, and
performs the CPU `dispatch_for_field!` inside the selected operation. The
caller-side dispatch in `akita-setup/src/recursive_prefixes.rs` is deleted.

Because `commit_setup_prefix` returns `SetupPrefixSlot<F>` directly, a
setup-prefix persistence workflow either selects the portable CPU state policy
or invokes portable export inside this function before constructing and
returning the slot. Export is not deferred until registry insertion. Loading a
slot imports its hint as generic prover-opening state. A resident route with no
portable export is valid for normal in-memory commitment but is rejected before
setup-prefix artifact creation.

Keep `AkitaProverSetup::prefix_slots`, `SetupPrefixProverRegistry`, registry
insertion and validation, `AkitaProverSetup::to_verifier_setup`, and
`SetupPrefixVerifierRegistry::replace_from_prover_registry` unchanged. Verifier
derivation strips prover-only hint material and copies the slot ID and public
commitment; portable export is required by the persisted prover slot and its
later proving use, not by verifier arithmetic.

The first cutover retains the existing validated
`setup_slot.hint.clone()` in `ProverOpeningData::new_recursive_suffix_fold`.
That is an existing portable setup-prefix cost, not a resident-state path. Do
not add registry-wide `Arc` storage or custom serialization in this project,
and do not introduce another hint clone. Preserve all current checks for
natural length, power-of-two prefix, setup capacity, live-block geometry, slot
identity, and hint validation. `SetupPrefixVerifierSlot` and all verifier
serialization remain unchanged.

### 8.5 Downstream/Jolt integration

Update the downstream contract test first. `ContractRootPoly` implements the
new source trait and uses the CPU executor directly; delete its forwarding
`ContractCommitBackend` only after output parity passes.

Migrate Jolt without changing its protocol group boundaries:

- dense objects use `DenseRepresentation::Coefficients` or
`DenseRepresentation::PredecomposedDigits`;
- row-major one-hot objects use `OneHotType`;
- `TracePackedOneHot` uses its current hand-tuned code through an external CPU
inner commitment operation; and
- opening adapters receive the generic commitment state supported by their
selected state operations. The CPU compatibility adapter may continue to
use `AkitaCommitmentHint` internally.

Revise `crates/akita-prover/docs/jolt-backend.md` with the final API names and
document the selected state representation and lifetime.

## 9. End-to-end implementation stages

Each stage has a narrow exit gate. Do not start deleting a legacy path until
the replacement for that path passes its differential tests.

### Stage 0: Freeze the baseline

Files:

- `crates/akita-pcs/tests/commitment_contract.rs`
- `crates/akita-prover/src/api/commitment/tests.rs`
- current end-to-end and transcript tests

Actions:

1. Inventory all callers and feature-gated commitment paths.
2. Add deterministic fixtures that expose the current `u`, public commitment,
  hint-derived inner/compression material, transcript events, full proof
   bytes, and proof size.
3. Pin the exact checked-in trusted schedule artifacts, catalog digests,
   resolved row identities, and setup-prefix registry filenames used by every
   fixture. Do not regenerate a different baseline through planner search.
4. Record CPU timing, peak RSS, and retained cache bytes for representative
  dense, one-hot, recursive, and setup-prefix commitments.
5. Confirm the current CI feature graphs and path-specific workflows.

Exit gate: baseline fixtures pass on the untouched implementation and the
benchmark record is attached to the first implementation PR.

### Stage 1: Introduce plans, sources, and resolved requests

Files:

- new modules under `compute/commitment/`
- built-in source implementations under `backend/`

Actions:

1. Add the D-free source descriptor and side-effect-free, multi-type
  capability discovery.
2. Add `represent_as` with distinct dense coefficients, dense predecomposed
  digits, short-norm packed coefficients, and one-hot representations.
3. Add the type-checked external inner commitment selection, batching,
  context, and fused-encoding contracts from section 5.3.
4. Add `for_root`, `for_recursive`, `for_terminal`, and `for_setup_prefix`
  checked plan constructors and their shared arithmetic plan views. Ensure
   recursive construction derives source encoding, payload mode, and
   `RingRelationMode` only from `CommittedGroupParams`.
5. Add per-call request compilation, recorded source resolution, and delayed
  materialization of only the selected representation or external operation.
6. Preserve each caller's current admission, structural checks, padding, and
  failure order.
7. Implement source adapters for dense, one-hot, recursive, and multilinear
  sources without changing the active production path.

Exit gate: each new source adapter materializes the same canonical input as its
current `CommitView`; invalid classes, bounds, sizes, and digit settings return
typed errors before arithmetic.

### Stage 2: Add CPU stage operations

Files:

- `compute/commitment/stages.rs`
- CPU stage modules under `compute/cpu/`
- `api/commitment/inner_outer.rs`
- `api/commitment/compression.rs`

Actions:

1. Refactor the current CPU A entry points to consume the resolved polynomial
  representations, then implement concrete CPU inner operation adapters over
   those optimized kernels.
2. Retarget `column_sweep_ajtai_onehot_multi` from `OneHotView` to
  `OneHotSource`, keeping its algorithm, scratch policy, and output
   unchanged. Keep `OneHotView` for opening/fold and add a commitment-only
   conversion to `OneHotSource`.
3. Move the one production outer decomposition/slicing path behind the CPU
  outer operation.
4. Wrap current compression execution in the CPU compression operation,
   preserving quotient-lift cyclic/negacyclic products and the reduced mode's
   negacyclic-only, zero-quotient path.
5. Add exact boundary shape validation.
6. Separate the common coefficient/packed adapter from the
  `WithCommitAccumulator` one-hot adapter. Add compile assertions for the
   one-hot adapter at each in-tree preset field without closing the open
   `CommitmentConfig::Field` contract.
7. Run new versus legacy differential tests for every built-in source class and
  supported fast path, including one group per stored index width and one
   mixed dense/one-hot `MultilinearPolynomial` group.

Exit gate: new CPU stages equal legacy inner witnesses, `u`, and compression
output when exported through the CPU state adapter. The portable adapter
reconstructs a byte-identical mode-specific legacy hint. Kernel-selection and
call-count tests prove cached digits, column sweep, packed digits, exact-i16,
and compression batching still run; prove that one-hot indices reach the sweep
at their stored width; and prove reduced compression performs no cyclic
product or quotient allocation.

### Stage 3: Add registration, resources, and the composite executor

Files:

- `compute/commitment/registration.rs`
- `compute/commitment/executor.rs`
- test support for recording stage operations

Actions:

1. Add the executor builder and private validated registrations for separate
  inner, outer, and compression operations.
2. Add `BackendStateStore`, owner-issued typed state references, generation
  checks, lease/release behavior, and composite state binding.
3. Add the generic state policy and CPU resident, portable-hint, and explicit
  no-retained-state policies. The last policy is usable for proving only with
   an explicit recomputation operation.
4. Add the independent state-consumer, portable export/import, and
  recomputation contracts, plus the executor-issued
   `StateRegistrationContext` required by external operations.
5. Add the optional inner-image export edge and validate it only for routes
  that require a cross-owner host transfer.
6. Add the object-safe resource controller, builder-issued backend/resource
  identities, and an explicit no-resources declaration.
7. Add `CommitmentNttStage::{Inner, Outer}` and executor-local resource
  routing without changing the still-legacy `ProverComputeStack` yet.
8. Compile source choices per request after the mode-specific plan is known;
  complete validation before materializing a lazy cache or preparing an
   external inner commitment operation.
9. Add split full, uncompressed, and inner-only execution modes.
10. Keep public commitment assembly and state-policy binding in the executor.
11. Add fake operations B and C and prove independent outer/compression
  routing, shared-owner deduplication, streamed policy, planned bytes, and
   release behavior through the executor-local resource API.
12. Add negative tests for setup mismatch, stale/foreign/wrong-kind state,
  cross-mode state reuse, incoherent capability metadata,
   unsupported polynomial type, representation, or encoding;
   external-operation-only source mismatch; forged external family/context
   mismatch; wrong result count; wrong ring dimension; and wrong slice or
   payload length.

Exit gate: all-CPU and mixed-stage routes return identical outputs. A backend
can implement the new outer operation without implementing the new compression
operation.

### Stage 4: Add and prove explicit fusion

Files:

- fused stage contract and recording fake remote backend
- commitment executor tests

Actions:

1. Add `FusedInnerOuterOperation` over the same resolved sources as split
  execution.
2. Require explicit fused selection on the executor builder.
3. Add a fake remote implementation that performs correct arithmetic and
  records requests/responses.
4. Record ordered `SubmitFused`, `DeviceAComplete`, `DeviceBBegin`, and
  `HostResult` events and assert one logical submission for A through B.
5. Assert that no `HostInnerRows` event occurs before `DeviceBBegin`.
6. Compare fused `u` with the new split CPU route and, while the legacy path is
  retained for differential testing, with today's split inner-then-outer
   result. Assert that the GPU result contains only an owner-issued inner-state
   reference, with no host-row or hint materialization event.
7. Add recording state-consumer operations for the fake remote state. Compare
  their derived inner/compression relation material with the CPU portable
   path, but leave production opening integration for the atomic Stage 5
   cutover.
8. Treat a retry as a new transport attempt. The default acceptance test
  permits no retry; a backend-specific idempotent retry test records every
   attempt separately.

Exit gate: the default transport attempt count is one, ordered event and output
parity checks pass, and a same-instance split executor remains visibly
unfused.

### Stage 5: Cut over root and recursive prover state atomically

Files:

- `api/commitment.rs`
- `akita-pcs/src/scheme/`
- `types/opening_data.rs`
- `protocol/ring_relation.rs`
- `protocol/ring_relation/compression_witness.rs`
- `protocol/ring_relation_witness.rs`
- `protocol/ring_switch/coeffs.rs`
- `compute/stack.rs`
- `compute/requirements.rs`
- `protocol/core.rs`
- `protocol/core/prove.rs`
- `protocol/core/root_fold.rs`
- `protocol/core/fold/`
- `protocol/core/suffix.rs`
- `protocol/ring_switch/commit.rs`
- `akita-pcs/examples/profile/workload/multi_group.rs`
- `akita-pcs/examples/profile/workload/single_group.rs`
- in-repository root commitment callers and tests

Actions:

1. Change root `commit()` to accept the existing `&TrustedScheduleCatalog` and
   a `&CommitmentExecutor` directly, resolve the catalog or explicit profile
   before plan construction, and route execution through the composite. Do not
   perform a second root API migration through `ProverComputeStack`, and do not
   move catalog ownership or schedule search into the executor.
2. Change the root result to `CommitOutput<F, S>` and make
  `AkitaCommitmentScheme::commit` generic over the selected private state.
   Add no `AkitaCommitmentHint`, `Clone`, or serialization bound to root
   commitment.
3. Generalize `ProverOpeningData`, `SelectedProverOpeningData`, and
  `ProverGroupInput` over `S`; replace `group_hint()` with the selected
   state-consumer boundary. Replace derived `Clone`/`Debug` bounds with moves,
   `Arc`, or redacted manual implementations. Delete the internal
   `CommitmentWithHint<F>` tuple alias and pass the named `CommitOutput` fields
   directly. Preserve the existing setup-prefix hint clone named in section
   8.4; do not add a second clone.
4. Add outer-compression and inner-relation state operations. Refactor
  `ring_relation.rs`, `compression_witness.rs`, `RingRelationGroupWitness`,
   and `ring_switch::coeffs` to consume their canonical derived carriers rather
   than an `AkitaCommitmentHint`.
5. Add portable hint export/import and explicit recomputation. Put each bound
  only on the consumer that uses it, and preflight the complete root
   commit-to-opening route before transcript mutation.
6. Replace the stack's single commit `OperationCtx<C>` with a
  `CommitmentExecutor`. Propagate `S` and the state-access operations through
   `ProverComputeStack`, `LevelProveStacks`, `UniformProverStack`,
   `TieredProveStacks`, `batched_prove`, `prove`, root-fold, and all fold helper
   signatures in the same change.
7. Generalize `NextWitnessStateOutput` and `SuffixProverState` over `S`. Add
  the independent terminal-binding operation and keep its bound separate from
   every other state capability.
8. Move compressed/uncompressed `commit_w` and inner-only
  `commit_terminal_w` to the executor. Use their dedicated checked plan
   constructors, preserve exact padding/tensor semantics, pass the
   schedule-owned `RingRelationMode` into compressed execution, preserve
   terminal `raw_field_segment_bytes`, and remove their direct
   `RuntimeCommitBackendFor` bounds. Reduced suffixes must retain no quotient
   image and must not execute the cyclic product path.
9. Preserve the trusted-catalog proving order:
   `SelectedProverOpeningData::from_committed_claims` resolves exact profiles
   to `OpeningScheduleSelection`; `batched_prove` resolves that selection,
   applies `effective_batched_schedule`, validates nonterminal execution and
   setup capacity, and derives `NttExecutionRequirements`. Then configure every
   fold-level/tier transition with the consumer, transfer, or explicit
   recomputation needed for the preceding level's state. Complete transition
   validation and NTT prewarming before
   `bind_transcript_instance_descriptor` mutates the transcript.
10. Add the stage discriminator to canonical `NttExecutionRequirements`, then
  route prewarming, planned cache metrics, streamed policy, and
   `ReleaseRootNttAfterFold` through the executor with physical-owner
   deduplication. Land this with the stack change so no revision has an
   unroutable A/B requirement.
11. Update callers atomically. A commit-only test must succeed with `S = ()`;
  a root proving test must succeed both with portable CPU state and with the
   fake GPU's resident state without constructing a hint. Run recursive
   compressed, uncompressed, terminal, and tier-transition tests in the same
   revision.
12. Keep the old implementation callable only from differential tests during
  this stage.
13. Run root and recursive prove/verify tests for all supported profiles and
  feature graphs.
14. Migrate the two profiling workload modules to the new executor API, then
  collect the candidate measurements against Stage 0's legacy baseline. The
    Stage 5 performance gate is not complete until this migrated harness runs.

Exit gate: every root and recursive workflow compiles without a legacy commit
context and passes output, proof, verifier, transcript, performance, and memory
gates.

### Stage 6: Cut over setup-prefix persistence and finish integration coverage

Files:

- `api/setup_prefix.rs`
- `api/setup.rs`
- `akita-setup/src/recursive_prefixes.rs`
- `akita-setup/src/lib.rs`
- setup-prefix and recursive end-to-end tests

Actions:

1. Build setup-prefix plans through the dedicated checked constructor,
  preserving current validation order and physical padding semantics. Drive
   required slot enumeration from the same `TrustedScheduleCatalog` through
   `setup_prefix_slot_ids_from_catalog`, not from executor capabilities.
2. Move setup-prefix commitment to full execution. Remove its public `const D`
   and backend/prepared parameters, accept the checked `SetupPrefixSlotId` plus
   the bare executor built by the CPU constructor, derive profile and lengths
   from that ID, remove the caller-side `dispatch_for_field!`, and perform
   explicit state-to-hint export inside `commit_setup_prefix` before it returns
   the existing portable prover-slot format. Leave the verifier slot unchanged.
3. Compare public payloads, state-derived canonical relation inputs, proofs,
  transcript events, prewarm keys, planned bytes, and release counts with
   legacy paths. Separately prove byte-identical portable hints when the CPU
   portable-state adapter is selected.
4. Exercise the existing CPU portable setup artifact across a byte-identical
  write/read round trip. Exercise resident state with portable export and
   prove its slot bytes, hint bytes, disk round trip, and later proof equal the
   CPU artifact.
5. Prove a resident configuration without portable export fails during
  request compilation, before commitment arithmetic and before it creates a
   partial artifact.
6. Preserve `AkitaProverSetup::prefix_slots`, prover-registry insertion and
  validation, and `to_verifier_setup` conversion through
   `replace_from_prover_registry`; test that verifier-slot derivation remains
   byte-identical.
7. Keep `AkitaCommitmentScheme::setup_prover` and
   `akita_setup::new_prover_setup` threaded with the same validated catalog.
   Validate loaded registry coverage against that catalog and preserve the
   `catalog_digest` component of disk registry filenames.
8. Migrate `akita-setup/src/lib.rs::ntt_caches_rebuilt_correctly_from_disk`
  away from direct `RootCommitKernel`/`RootCommitSource` use while preserving
   its fresh-versus-disk-prepared cache check.

Exit gate: all recursive, terminal, setup-prefix, coefficient-packing, and
disk-persistence tests pass with byte-identical outputs.

### Stage 7: Migrate downstream extensibility and Jolt

Files:

- `akita-pcs/tests/commitment_contract.rs`
- Jolt's Akita adapter and packed-trace source
- `crates/akita-prover/docs/jolt-backend.md`

Actions:

1. Replace the contract test's forwarding backend with a source-only standard
  polynomial representation.
2. Prove the downstream source produces the same commitment and state-derived
  relation material as its dense reference. Under the portable CPU policy,
   also compare the exported hint.
3. Move Jolt's packed trace kernel to an external CPU inner commitment
  selection without changing its body or group boundaries.
4. Pin the downstream Jolt checkout and adapter API used as evidence, and run
  its exact compatibility, proof-byte, performance, and memory workflows.
5. Update the Jolt design note to match the shipped API.

Exit gate: a downstream source needs no forwarding backend, the packed
external-operation path equals its reference, and the Jolt integration shows
no proof-size, transcript, feature, performance, or memory regression.

### Stage 8: Remove superseded commit-only surfaces

Actions:

1. Remove legacy root-commit dispatch only after every caller has moved.
2. Delete `RootCommitKernel`, `RootCommitSource`, `CommitBackendFor`,
  `RuntimeCommitSource`, and `RuntimeCommitBackendFor` only after workspace
   and pinned-downstream searches find no caller.
3. Keep `RootPolyMeta`, `RootPolyShape`, and all opening capabilities that
  remain in use. Remove only the commitment implementations from `DenseView`
   and `MultilinearPolynomialView`; keep those views and `OneHotBatchView` for
   their opening roles. Keep `SparseRingBlockEntry` unchanged in the first
   cutover.
4. Remove temporary legacy differential hooks after split/fused, commitment,
  proof-byte, transcript, and verifier parity tests pass.
5. Search the workspace and downstream integration for stale symbols.
6. Fold durable architecture prose into the Akita Book and update spec status.

Do not delete `CommitOutput`, the portable `AkitaCommitmentHint` codec and
setup-prefix prover-slot persistence, `ProverOpeningData`, or
the existing opening/tensor/ring-switch algorithms in this stage. Remove only
their universal dependency on the concrete hint.

Exit gate: no duplicate production commitment implementation remains, all
supported paths compile and test, and documentation guardrails pass.

## 10. Test matrix and acceptance criteria

### 10.1 Arithmetic and layout

- [ ] While the legacy path is retained, the new split route returns the same
      `u`, public commitment, and canonical commitment bytes.
- [ ] Split and fused production routes return identical `u`; when both states
      are intentionally exported through the portable CPU adapter, their
      canonical inner and compression material is identical.
- [ ] Tests cover one and multiple polynomials, one and multiple slices,
      partial A blocks, partial final slices, and `D_A > D_B`.
- [ ] Result ordering is `[polynomial][block][A row]` for inner rows and
      `[slice][B row]` for `u`.
- [ ] Every boundary rejects the wrong count, ring dimension, or physical
      width with `AkitaError`.
- [ ] Root, recursive, terminal, and setup-prefix plan constructors preserve
      their distinct current validation order, admission rules, payload mode,
      source encoding, relation mode, tensor-packing choice, and padding
      semantics.

### 10.2 Source extensibility

These criteria cover commitment extensibility only. A source used in an
end-to-end opening proof continues to satisfy the existing opening and tensor
source/kernel contracts described under R1.

- [ ] A downstream dense wrapper implements only `CommitmentSource` and commits
      on the CPU executor.
- [ ] The downstream wrapper's commitment and state-derived relation material
      equal its dense reference; portable exports agree when requested.
- [ ] Dense, cached-digit, one-hot, recursive packed-digit, multilinear, and
      external packed-trace sources retain their current algorithms.
- [ ] A one-hot source commits identically at every stored index width, and the
      sweep receives `u8`, `u16`, `u32`, and `usize` indices unwidened.
- [ ] An all-zero one-hot chunk survives `OneHotRepresentation` as `None` and
      commits to the same rows as today.
- [ ] A one-hot-shaped source that is not `OneHotPoly` commits through
      `OneHotType` with no forwarding backend.
- [ ] Every supported protocol field satisfies the CPU inner operation's
      bounds, proven by a compile assertion rather than by a runtime error.
- [ ] Standard and external inner commitment paths agree wherever both exist.
- [ ] `ExternalInnerCommitmentPolicy::Forbid` is observable and never silently
      falls back.
- [ ] External-operation-only sources fail during request compilation on an
      unsupported backend kind.
- [ ] Capability discovery is side-effect-free; an unused dense digit cache is
      not built, while first-use and warm-cache kernel counters match today.
- [ ] A source that offers both coefficients and cached digits resolves to the
      best supported form for the selected inner operation without losing the
      alternative form.

### 10.3 Routing and fusion

- [ ] Inner on A, outer on B, and compression on C equals all-CPU output.
- [ ] Setup mismatch is rejected before arithmetic.
- [ ] Fused execution uses exactly one recorded transport submission for A
      through B.
- [ ] The fused operation receives the same admitted, resolved source list as
      the split operation.
- [ ] Equal inner/outer backend IDs do not imply fusion.
- [ ] Route and backend metadata never enter protocol serialization or the
      transcript.
- [ ] A fused resident route records no host-inner-row transfer and constructs
      no `AkitaCommitmentHint` during commitment.
- [ ] Foreign, stale, wrong-generation, wrong-kind, and wrong-plan state
      references fail before any further transcript mutation can depend on
      them.
- [ ] State bound to one `RingRelationMode` cannot be consumed under the other
      mode.
- [ ] Wrong owner, setup, profile, group layout, source count, or public
      commitment binding fails at the same boundary.
- [ ] Dropping the executor before the output does not invalidate resident
      state; clones remain usable, `close(self)` releases only its lease, and
      physical cleanup occurs exactly once after the last lease.
- [ ] A retry cannot alias state from a failed attempt or a reused slot.
- [ ] A mixed route can bind GPU inner state and CPU compression state without
      copying the inner image or confusing their owners.

### 10.4 Protocol compatibility and proof size

- [ ] `CommitOutput<F, S>` contains the unchanged `committed_group` plus a
      prover-private `S`, with no universal hint or serialization bound.
- [ ] `AkitaCommitmentHint` remains supported as the CPU portable state/export,
      and that export is byte-identical to today's mode-specific value.
- [ ] Quotient-lift state retains one correctly shaped quotient image per map;
      reduced-evaluation state retains none and runs no cyclic-product kernel.
- [ ] `singleton_with_reduced_outer_compression` and
      `reduced_outer_compression_witness` round-trip the reduced portable state
      without adding quotient bytes.
- [ ] Root, recursive, terminal, and setup-prefix commitments have identical
      verifier-visible outputs across old/new and split/fused paths.
- [ ] Resident GPU state, portable CPU state, and explicit recomputation produce
      identical canonical ring-relation inputs and proofs.
- [ ] GPU commit to CPU state consumption, portable CPU commit to a registered
      alternate consumer, and a fold-level tier change all follow precompiled
      transfer/consumption routes with no mid-proof fallback.
- [ ] Compressed and uncompressed recursive states retain the correct
      components; terminal state produces exactly today's
      `raw_field_segment_bytes` transcript input without constructing a hint.
- [ ] `LoggingTranscript` event sequences are identical.
- [ ] Deterministic end-to-end proofs are byte-identical and verify.
- [ ] Each proof comparison starts from commitments and prover states produced
      by the compared executors and then runs otherwise identical deterministic
      opening/proving flows with the same transcript fixture, RNG inputs, and
      canonical serialization mode.
- [ ] `AkitaBatchedProofShape` and `proof.size()` are identical.
- [ ] No verifier dependency, verifier code, proof serializer, or proof-size
      formula changes as part of the backend refactor.
- [ ] `SetupPrefixSlot` and `SetupPrefixVerifierSlot` bytes are unchanged. CPU
      portable prover slots retain a validated disk round trip; resident state
      with export produces the same slot and later proof; resident state
      without explicit portable export is rejected before commitment
      arithmetic begins.
- [ ] The same trusted catalog resolves setup, commitment, proving, and
      verification. Missing rows reject without runtime planner search.
- [ ] `OpeningScheduleSelection`, effective-schedule resolution, transcript
      binding, catalog digest, and canonical artifact bytes are unchanged.
- [ ] Setup-prefix required IDs and loaded-registry coverage come from that
      catalog, and disk registry names remain bound to `catalog_digest`.

### 10.5 Feature, performance, and operational coverage

- [ ] Default, no-default, parallel, non-parallel, disk-persistence,
      response-model-diagnostics, logging-transcript, Blake2b transcript, and
      Keccak transcript builds retain their current behavior.
- [ ] Current CI Clippy feature graphs pass.
- [ ] Current nextest shards and path-specific workflows pass.
- [ ] Root, recursive, setup-prefix, and Jolt benchmarks show no unexplained
      time or peak-memory regression.
- [ ] Reduced recursive suffix benchmarks show no quotient allocation or cyclic
      compression work.

## 11. Validation order for every implementation PR

Run the current cheap repository-wide gates before expensive compilation. For
code stages, then run focused tests for the changed crate and source path. The
workflow files are the source of truth for exact commands and feature graphs.
Before merging a cutover stage:

1. run legacy/new split, split/fused, and public-commitment differential tests;
2. run source-specific dense, one-hot, recursive, multilinear, and setup-prefix
  tests;
3. run transcript parity tests under `logging-transcript`;
4. run proof-shape and serialized proof-size tests;
5. run `scripts/check-external-schedule-artifacts.sh` and
   `cargo test --release -p akita-config --test trusted_schedule_artifact`;
6. run `scripts/generate-schedule-artifacts.sh` in a clean worktree and require
   zero diff in `artifacts/schedules`, including catalog digests. Match the
   temporary-output comparison and remaining checks in
   `.github/workflows/ci.yml`'s `test-schedule-artifact-drift` job;
7. run the current CI-fidelity nextest and Clippy commands from
   `.github/workflows/ci.yml`;
8. run disk-persistence and full downstream Jolt compatibility workflows when
   their callers change. Also run `.github/workflows/jolt-verifier-profile-smoke.yml`,
   recognizing that this in-tree smoke does not replace the full manual Jolt
   prove/verify workflow;
9. run the recorded performance and memory comparisons through the cases and
   baseline policy in `.github/workflows/profile-bench.yml`; and
10. run `./scripts/check-doc-guardrails.sh` when documentation changes.

## 12. Risks and controls


| Risk                                                                  | Control                                                                                                                                                           |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fused source handling bypasses admission                              | Compile one `ResolvedCommitSource` list after admission and pass that exact list to both split and fused operations.                                              |
| Polynomial representation is selected without the real plan           | Resolve sources per request only after `CommitInnerPlan` exists.                                                                                                  |
| External source operation supports only one backend kind accidentally | Return a backend-keyed selection object; do not require the source itself to be the sole external operation object.                                               |
| Trait claims fusion but makes two remote calls                        | Recording transport tests plus backend integration telemetry; the type system alone is not the control.                                                           |
| Fused execution loses prover state                                    | Retain the inner image behind an owner-issued checked reference; validate a compatible state consumer before proving starts.                                      |
| A generic state abstraction secretly requires host rows               | Keep export, inner-relation, compression, terminal, portable export/import, and recomputation capabilities independent; add only the bound used by each caller.   |
| External operations cannot inspect private heterogeneous state        | Dispatch internally by builder-issued owner and semantic kind, then pass the external operation a typed component reference.                                      |
| Portable state construction is circular or doubles memory             | Export inner and compression components before `S` exists; prefer the consuming path and explicitly budget transient overlap for shared exports.                  |
| Resident state is stale, foreign, or released too early               | Bind store, owner, setup, plan, kind, generation, and leases; test every rejection before any further transcript mutation can depend on it.                       |
| Compression state is reused under the wrong relation mode             | Bind `RingRelationMode` into the plan and state metadata; require quotient rows only for `QuotientLift` and reject them for `ReducedEvaluation`.                  |
| Executor construction becomes a second schedule authority             | Keep the validated `TrustedScheduleCatalog` on `AkitaCommitmentScheme`; resolve before plan construction and reject missing rows without planner fallback.         |
| A setup-prefix route cannot persist its state                         | Require explicit export to the existing portable hint format; keep setup-prefix bytes unchanged and reject unsupported routes before creating an artifact.        |
| Setup-prefix persistence is reused across catalog revisions           | Derive required slot IDs from the trusted catalog, validate loaded coverage, and retain `catalog_digest` in the registry filename.                                  |
| Slice or subcolumn ordering changes                                   | Legacy/new split and split/fused differential fixtures covering multiple polynomials, slicing, padding, and `D_A > D_B`.                                          |
| A route changes proof size                                            | Keep route metadata outside protocol types and compare proof shapes, serialized proofs, and `size()`.                                                             |
| Source abstraction disables a fast path                               | Kernel-selection counters and before/after benchmarks for every current source class.                                                                             |
| Polynomial-representation erasure widens or copies one-hot indices    | Carry the stored width in `UnitPositionSlice` and borrow the slice unchanged; assert unwidened arrival at the sweep and benchmark the one-hot commit.             |
| A backend's field bound leaks into the source contract                | Keep arithmetic bounds on backend implementations; a bound on `PolynomialRepresentation` would propagate through `CommitmentSource` to every source and backend.  |
| An in-tree preset field silently loses one-hot commitment             | Compile assertion instantiating the full CPU inner adapter at every in-tree preset field; the weaker common adapter remains constructible for open custom fields. |
| Global compute-trait cleanup breaks opening code                      | Keep the first cutover scoped to new commitment stage traits; review wider trait changes separately.                                                              |
| Temporary dual paths become permanent wrapper slop                    | Permit the legacy path only for differential tests during migration and delete it after all callers cut over.                                                     |




## 13. Non-goals and deferred work

This project does not:

- require one universal state representation or one universal checkpoint
format;
- require every backend to export an `AkitaCommitmentHint`;
- define or persist device-handle checkpoints; setup-prefix persistence keeps
the existing portable hint encoding;
- define a common remote wire protocol;
- fuse commitment compression into the same device request;
- change opening, fold, tensor, ring-switch, or sum-check mathematics or their
verifier-visible outputs; prover-only state plumbing and operation bounds do
change;
- combine protocol commitment groups that are separate today;
- change schedule selection, SIS pricing, proof shape, or transcript binding;
- change verifier behavior or its no-panic contract; or
- globally redesign `DigitRowsComputeBackend` and
`CompressionComputeBackend` beyond what the commitment adapters require.

A future all-stage device operation can be proposed after fused A-to-B ships.
It may keep all private state resident. It must supply the state-consuming
operations needed by the selected proving flow and must pass the same
commitment/proof/transcript/verifier gates.

## 14. Completion definition

The design is implemented only when all stages through cleanup are complete,
all acceptance criteria are checked, and no supported commitment or proof flow
has regressed.

The final code has one production implementation of each concept:

- one request compiler for source admission and resolution;
- one inner operation per backend;
- one outer operation per backend;
- one compression operation per backend;
- one optional explicit fused inner/outer operation per backend;
- one explicit `RingRelationMode` input to compression, with no quotient
  allocation in reduced mode;
- one composite assembler for the public commitment and selected generic
prover state;
- independent state-consumer, portable export/import, and recomputation
operations, required only where used.

`TrustedScheduleCatalog` remains the single schedule authority outside that
execution boundary. The final implementation introduces no runtime planner
fallback, alternate catalog cache, or setup-prefix slot enumeration in the
executor.

That boundary provides source extensibility, stage routing, and real A-to-B
fusion while leaving Akita's protocol bytes and supported feature set intact.

## Appendix A. Commitment trait migration review



### A.1 Status and review scope

This appendix records the proposed trait-level cutover for detailed review
before implementation. It distinguishes commitment-only traits that the new
executor supersedes from shared compute and proof traits that must remain.

The public reusable object is `CommitmentExecutor`, not a separate routing
wrapper. A per-call `CommitmentExecutionPlan` contains the resolved route. The
main design and this appendix use that terminology consistently.

No item in this appendix authorizes changing protocol messages, proof shape,
transcript order, commitment bytes, setup-prefix hint bytes, supported source
representations, supported ring dimensions, or cache lifecycle behavior. The
live prover-state representation and prover-only trait bounds do change.

### A.2 Current commitment trait graph

The current source/backend contract is effectively:

```text
CompressionComputeBackend<F>  --extends--> ComputeBackendSetup<F>
DigitRowsComputeBackend<F>    --extends--> ComputeBackendSetup<F>
DigitRowsComputeBackend<F>    --extends--> CompressionComputeBackend<F>
CyclicRowsComputeBackend<F>   --extends--> DigitRowsComputeBackend<F>

RootCommitKernel<S, F, D>     --extends--> ComputeBackendSetup<F>
RootCommitSource<F, D>        --extends--> RootPolyShape<F, D>
RootCommitSource<F, D>        --defines--> CommitView<'a>

CommitBackendFor<F, P, D>
    = DigitRowsComputeBackend<F>
      + RootCommitKernel<P::CommitView<'a>, F, D>

RuntimeCommitSource<F>
    = RootPolyMeta<F>
      + RootCommitSource at D = 64, 128, 256, 512, 1024, and 2048

RuntimeCommitBackendFor<F, P>
    = DigitRowsComputeBackend
      + RootCommitKernel for P at every supported D
```

This gives compile-time proof that one backend implements both the source's A
kernel and the B digit-row operation at every runtime-supported dimension. It
also couples those operations to one backend type and one prepared setup.

### A.3 Proposed commitment trait graph

The proposed public commitment boundary is:

```text
CommitmentSource<F>
          |
          v
CommitmentExecutor<F, StatePolicy>
|-- InnerCommitOperation<F>
|-- OuterCommitOperation<F>
|-- CompressionOperation<F>
`-- optional FusedInnerOuterOperation<F>

Each registered stage also carries:
|-- backend instance identity
|-- setup identity
|-- supported-dimension and polynomial-type capability metadata
|-- explicit resource control, or an explicit no-resources declaration
|-- checked state-registration ownership
`-- only the transfer/state-consumer capabilities selected routes need

CommitOutput<F, StatePolicy::State>
`-- unchanged CommittedGroup<F> + prover-private state
```

There is no common operation supertrait. There is also no public routing
object in addition to the executor. The executor owns reusable stage
selection, validates registration, and compiles one immutable per-call
`CommitmentExecutionPlan` after the commitment profile and actual sources are
known.

The operation traits are deliberately minimal and object-safe:

```rust,ignore
pub trait InnerCommitOperation<F: Field>: Send + Sync {
    fn commit_inner(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &CommitInnerPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<InnerCommitOutput, AkitaError>;
}

pub trait OuterCommitOperation<F: Field>: Send + Sync {
    fn commit_outer(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &OuterCommitPlan,
        inner: InnerImageInput<'_, F>,
    ) -> Result<RingVec<F>, AkitaError>;
}

pub trait CompressionOperation<F: Field>: Send + Sync {
    fn compress(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
        u: RingVec<F>,
    ) -> Result<CompressionStageOutput<F>, AkitaError>;
}

pub trait FusedInnerOuterOperation<F: Field>: Send + Sync {
    fn commit_inner_outer(
        &self,
        state: &mut StateRegistrationContext<'_, F>,
        plan: &UncompressedCommitPlan,
        sources: &[ResolvedCommitSource<'_, F>],
    ) -> Result<UncompressedCommitmentOutput<F>, AkitaError>;
}
```

Backend identity, setup identity, resource ownership, supported dimensions,
and polynomial-representation capabilities are registration data. They are not supertraits of
the arithmetic operations. This prevents an inner-only or outer-only backend
from implementing unrelated operations merely to satisfy a bundle.

Because these traits are public extension points, `ResolvedCommitSource`, all
plan types, `StateRegistrationContext`, `BackendStateRef`,
`InnerCommitOutput`, `UncompressedCommitmentOutput`, `InnerImageInput`, and
`CompressionStageOutput` are public execution types with checked factories and
sufficient read-only/consuming accessors for external implementations. They are
not protocol messages and implement no protocol serialization merely because
they cross this API boundary.

State capabilities are independent traits, not supertraits of the four
arithmetic operations. The minimum set is the inner-image export edge,
outer-compression consumption, inner-relation consumption, terminal binding,
portable hint export/import, and explicit recomputation. A selected proving
path requires only the subset it uses. In particular, neither
`FusedInnerOuterOperation` nor its state type has an
`AkitaCommitmentHint`/serialization bound.

Their conceptual signatures are:

```rust,ignore
pub trait OuterCompressionStateOperation<F: Field, S>: Send + Sync {
    fn prepare_outer_compression(
        &self,
        binding: &CommitmentStateBinding,
        state: &S,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<PreparedOuterCompressionMaterial<F>, AkitaError>;
}

pub trait InnerRelationStateOperation<F: Field, S>: Send + Sync {
    fn prepare_inner_relation(
        &self,
        binding: &CommitmentStateBinding,
        state: &S,
        plan: &CheckedCommitmentPlan,
    ) -> Result<PreparedInnerRelationMaterial<F>, AkitaError>;
}

pub trait TerminalBindingStateOperation<F: Field, S>: Send + Sync {
    fn terminal_t_fields(
        &self,
        binding: &CommitmentStateBinding,
        state: &S,
        plan: &CommitInnerPlan,
    ) -> Result<TerminalTFieldsMessage<F>, AkitaError>;
}

pub trait PortableCommitmentStateExport<F: Field, S>: Send + Sync {
    fn export_hint(
        &self,
        binding: &CommitmentStateBinding,
        state: &S,
        plan: &CheckedCommitmentPlan,
    ) -> Result<AkitaCommitmentHint<F>, AkitaError>;

    fn into_hint(
        &self,
        binding: CommitmentStateBinding,
        state: S,
        plan: &CheckedCommitmentPlan,
    ) -> Result<AkitaCommitmentHint<F>, AkitaError>;
}
```

The four generic traits above are the high-level proving boundary. The standard
dispatcher implements them for `S = ProverOpeningState<F>`. External resident
backends implement component operations whose inputs are typed references:

```rust,ignore
pub trait ResidentInnerRelationOperation<F: Field>: Send + Sync {
    fn prepare_inner_relation(
        &self,
        binding: &CommitmentStateBinding,
        image: &BackendStateRef<InnerImage>,
        plan: &CheckedCommitmentPlan,
    ) -> Result<PreparedInnerRelationMaterial<F>, AkitaError>;
}

pub trait ResidentOuterCompressionOperation<F: Field>: Send + Sync {
    fn prepare_outer_compression(
        &self,
        binding: &CommitmentStateBinding,
        state: &BackendStateRef<CompressionState>,
        plan: &CompressionChainPlan,
        relation_mode: RingRelationMode,
    ) -> Result<PreparedOuterCompressionMaterial<F>, AkitaError>;
}
```

The same owner-keyed registration pattern applies to portable component export
and terminal binding. The dispatcher, not external code, unwraps the private
standard state and passes the correctly typed reference to the owner-matched
operation.

The external operation retains its builder-issued
`StateOwnerCapability<InnerImage>` or
`StateOwnerCapability<CompressionState>`. It calls
`resolve_for_owner` on the typed reference to obtain the opaque key for its own
private buffer table. Resolution validates owner, semantic kind, store,
generation, setup, and plan; the key exposes no pointer, store slot, or payload
type and cannot resolve under another capability. This is the only public
resident-payload lookup mechanism.

The returned carrier ownership is:

```rust,ignore
pub enum PreparedCompressionRelation<F: Field> {
    QuotientLift { quotients: Vec<RingVec<F>> },
    ReducedEvaluation,
}

pub struct PreparedOuterCompressionMaterial<F: Field> {
    group_index: usize,
    plan: CompressionChainPlan,
    relation_mode: RingRelationMode,
    terminal_payload: RingVec<F>,
    b_source: RingVec<F>,
    witness: CompressionChainWitness,
    relation: PreparedCompressionRelation<F>,
}

pub struct PreparedInnerRelationMaterial<F: Field> {
    t_hat: DigitBlocks,
    recomposed_inner_rows: RingVec<F>,
}
```

`PreparedOuterCompressionMaterial` owns the group index, the exact compression
plan and relation mode, the terminal payload association, the complete B source
in canonical `[slice][B row][D_B coefficient]` order, the checked
`CompressionChainWitness`, and mode-appropriate relation data. Construction
checks that the first retained stage recomposes to the supplied B source, the
witness plan equals the requested plan, and the terminal payload equals the
public commitment payload bound to this group. `QuotientLift` requires one
correctly dimensioned quotient `RingVec` per map in map order;
`ReducedEvaluation` requires zero quotient images. The state binding must match
the requested mode before any carrier is constructed.

`PreparedInnerRelationMaterial` owns `t_hat` and the recomposed inner rows.
`t_hat` uses the existing `DigitBlocks` layout produced by
`decompose_commit_blocks_into`: one value whose blocks are ordered
`[polynomial][live block]`, and whose internal order is
`[A row][D_A / D_B subcolumn][outer digit][D_B coefficient]`. Recombined rows
use ring dimension `D_A` and order `[polynomial][live block][A row]`; their count is
`polynomial_count * live_blocks * n_a`. Construction checks all dimensions,
counts, padding, and plan identity. These two carriers move into
`RingRelationGroupWitness`/the prepared ring-switch group; they are not cloned
or retained alongside a hint.

These are prover-execution carriers, not new proof fields. A CPU implementation
reuses the current functions. A GPU implementation may derive them directly in
resident memory and transfer only these canonical next-stage inputs.

Keep field bounds at the concrete consumer implementation:

- the built-in outer-compression materializer requires
`F: Field + CanonicalEncoding + AkitaSerialize`, matching
`compression_witness.rs`;
- the built-in inner-relation/ring-switch materializer requires
`F: Field + CanonicalEncoding + Ring + AkitaSerialize`, matching
`ring_switch::coeffs`;
- portable hint serialization requires `F: Field + AkitaSerialize`; and
- terminal byte emission carries the exact bound required by
`raw_field_segment_bytes` at that call site.

These bounds do not appear on `CommitOutput<F, S>`, `S`,
`CommitmentSource<F>`, or an unrelated state operation. A custom resident
operation can use its own strictly necessary field bounds.

The import side for a deserialized hint and the explicit recomputation path are
constructors/operations on the built-in prover-state policy; they are not
required methods on every state type. Recompute reruns the canonical checked
commitment from the bound sources, compares the resulting public commitment to
the binding, and only then exposes the new state. No operation may silently
switch to export or recomputation after transcript mutation.

### A.4 Disposition of every affected current trait


| Current trait                          | Disposition                                   | Detailed reason                                                                                                                                                                                                                                                                                       |
| -------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ComputeBackendSetup<F>`               | Keep unchanged in the first cutover           | It owns prepared-setup validation, exact NTT slot construction, streamed-versus-resident policy, physical cache-owner identity, planned bytes, and cache release. Opening, tensor, ring-switch, setup, profiling, and compatibility adapters still need it. It stops being a public `commit()` bound. |
| `CompressionComputeBackend<F>`         | Keep as a low-level capability                | `execute_compression_chains` uses it for root commitment compression, and ring-relation proving uses it again for the opening/D compression witness. A commitment-only `CompressionOperation` cannot replace the latter.                                                                              |
| `DigitRowsComputeBackend<F>`           | Keep unchanged in the first cutover           | It is used outside outer commitment by opening, fold, and ring-relation code. A new outer-only backend can implement `OuterCommitOperation` directly, so global trait decoupling is not required for stage composition.                                                                               |
| `CyclicRowsComputeBackend<F>`          | Keep unchanged                                | Ring-switch relation construction still requires its distinct cyclic-product operation.                                                                                                                                                                                                               |
| `RootCommitKernel<S, F, D>`            | Replace, then delete                          | The backend/source/ring-dimension product is the coupling the new source and inner-operation contracts remove. Keep it temporarily only for differential migration and delete it after every commitment producer has moved.                                                                           |
| `RootCommitSource<F, D>`               | Replace, then delete                          | Its commit view and centered-reach method move to the D-free commitment source contract. It does not own opening or tensor behavior, so those traits remain separate.                                                                                                                                 |
| `RootPolyMeta<F>`                      | Keep unchanged                                | `ProverOpeningData`, root-group validation, one-hot classification, proof orchestration, and response-model diagnostics use its D-free metadata independently of commitment execution.                                                                                                                |
| `RootPolyShape<F, D>`                  | Keep unchanged                                | Opening, tensor projection, coefficient packing, root-group checks, and recursive live-prefix semantics still need dimension-specific ring counts.                                                                                                                                                    |
| `CommitBackendFor<F, P, D>`            | Delete after migration                        | It is a marker combining `RootCommitKernel` and `DigitRowsComputeBackend`; the executor and registered operations supersede it.                                                                                                                                                                       |
| `RuntimeCommitSource<F>`               | Delete after migration                        | It is the supported-dimension ladder over `RootCommitSource`. The D-free source contract plus plan-time dimension validation supersedes it.                                                                                                                                                           |
| `RuntimeCommitBackendFor<F, P>`        | Delete after all commitment producers migrate | Root commitment, recursive `w`, terminal `w`, setup-prefix generation, `akita-setup`, and high-level proving bounds currently use it. Removing it before those callers migrate would remove supported features.                                                                                       |
| `OpeningFoldKernel`                    | Keep unchanged                                | It is outside the commitment-stage redesign.                                                                                                                                                                                                                                                          |
| `OpeningBatchKernel`                   | Keep unchanged                                | It is outside the commitment-stage redesign.                                                                                                                                                                                                                                                          |
| `TensorProjectionKernel`               | Keep unchanged                                | It is outside the commitment-stage redesign.                                                                                                                                                                                                                                                          |
| `TensorProjectionBatchKernel`          | Keep unchanged                                | It is outside the commitment-stage redesign.                                                                                                                                                                                                                                                          |
| `SubringCoefficientPackingBatchKernel` | Keep unchanged                                | It is outside the commitment-stage redesign.                                                                                                                                                                                                                                                          |
| `RingSwitchRelationKernel`             | Keep unchanged                                | It is outside the commitment-stage redesign.                                                                                                                                                                                                                                                          |
| `RootOpeningSource`                    | Keep unchanged                                | Commitment representation must not absorb opening-source behavior.                                                                                                                                                                                                                                    |
| `RootTensorSource`                     | Keep unchanged                                | Commitment representation must not absorb tensor-source behavior.                                                                                                                                                                                                                                     |


The following current data types are affected even though they are not backend
traits:


| Current type                                                                       | Disposition                                                           | Detailed reason                                                                                                                                                                              |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CommitInnerPlan`                                                                  | Keep and extend                                                       | This remains the single inner arithmetic plan. Add runtime `D_A` and live-block count; do not introduce a transposed `InnerCommitPlan` alongside it.                                         |
| `CommitInnerWitness<F>`                                                            | Keep                                                                  | It remains the canonical host result for existing optimized and external CPU inner algorithms. Resident and fused operations are not required to construct it.                               |
| `TrustedScheduleCatalog` and `OpeningScheduleSelection`                            | Keep unchanged                                                        | The scheme-owned catalog remains the only schedule authority. Selection and effective-schedule resolution occur before schedule-dependent state-route planning; neither type moves into the executor. |
| `RingRelationMode`                                                                 | Keep unchanged                                                        | The authenticated schedule selects `QuotientLift` or `ReducedEvaluation`. Commitment compression and retained state must receive and bind the selected mode without modifying the protocol enum. |
| `CompressionRelationOutput<F>`                                                     | Keep its semantic split                                               | CPU state may wrap or move this data, but it must preserve `QuotientLift { quotients }` versus `ReducedEvaluation` with no quotient image.                                                    |
| `RelationQuotientPlan` and `RelationQuotientLayout`                                | Keep unchanged                                                        | Witness layout remains schedule-mode-aware; reduced relations continue to carry no quotient metadata or rows.                                                                                |
| `CommitOutput<F>`                                                                  | Generalize to `CommitOutput<F, S>`                                    | The public commitment stays unchanged; the prover-private field becomes route-selected state rather than a mandatory hint.                                                                   |
| `CommitmentWithHint<F>`                                                            | Delete during the state cutover                                       | This private tuple alias hard-codes the old hint. Callers use `CommitOutput<F, S>` or its named fields directly; do not replace it with another tuple alias.                                 |
| `AkitaCommitmentHint<F>`                                                           | Keep as a portable state/export and the setup-prefix persisted format | CPU and disk workflows retain it, but resident operations do not implement or return it unless an explicit export is selected.                                                               |
| `ProverOpeningData<...>`, `SelectedProverOpeningData<...>`, and `ProverGroupInput` | Generalize over state                                                 | They bind one private state value to each committed group. They must not expose `group_hint()` as the universal accessor or regain unconditional `S: Clone + Debug` through derives.         |
| `PreparedProverGroup<'a, P>`                                                       | Keep unchanged                                                        | It stores only borrowed polynomial sources. The surrounding opening-data types gain generic state; this carrier does not.                                                                    |
| `RingRelationGroupWitness<F>`                                                      | Replace its concrete `hint` field                                     | It carries either generic state until a selected state operation consumes it, or the canonical derived inner-relation material. Do not copy resident state into an intermediate hint vector. |
| `NextWitnessStateOutput<F>` and `SuffixProverState<F>`                             | Generalize over state                                                 | Recursive state ownership remains valid across fold levels and tier selection without requiring host rows.                                                                                   |
| `SetupPrefixSlot<F>`                                                               | Keep unchanged                                                        | It is an `akita-types` portable artifact. Load wraps its hint into prover state; save explicitly exports portable state before insertion.                                                    |
| `SetupPrefixSlotId`                                                                | Keep unchanged                                                        | Required IDs continue to come from `setup_prefix_slot_ids_from_catalog`; plan construction derives profile and prefix lengths from the checked ID.                                            |
| `SetupPrefixProverRegistry<F>` and `SetupPrefixVerifierRegistry<F>`                | Keep unchanged                                                        | The prover registry remains the public field on `AkitaProverSetup`; verifier derivation continues to strip the hint and copy the existing public slot data.                                  |
| `DenseView<'a, F, D>`                                                              | Keep for opening; remove its commit role                              | Dense coefficient and predecomposed-digit representations replace its `RootCommitKernel` use, while existing opening kernels still consume the view.                                         |
| `MultilinearPolynomialView<'a, F, D, I>`                                           | Keep for opening; remove its commit dispatcher                        | The source contract replaces its dense/one-hot `RootCommitKernel` dispatch. Its opening implementations remain outside this redesign.                                                        |
| `OneHotBatchView<'a, F, D, I>`                                                     | Keep unchanged                                                        | It is an opening batch view and is not replaced by commitment representation.                                                                                                                |
| `RecursiveFoldSource<F>`                                                           | Keep unchanged and out of scope                                       | It carries an already-committed setup prefix or recursive witness into opening/tensor operations and does not implement `RootCommitSource`.                                                  |
| `SparseRingBlockEntry`                                                             | Keep unchanged in the first cutover                                   | One-hot opening/fold and CPU sweep code still use it. Visibility cleanup is separate and must not remove a supported public surface in this project.                                         |
| `CommittedSourceEncoding`                                                          | Keep unchanged                                                        | It is schedule-owned physical encoding, distinct from the source descriptor's SIS/admission class, and remains part of recursive/root plan validation.                                       |


The first cutover intentionally leaves the existing
`DigitRowsComputeBackend: CompressionComputeBackend` relationship intact. It
is broader than the new stage model needs, but removing it would change bounds
through non-commitment code. A later cleanup can decouple those low-level
traits after a separate blast-radius review and after adding explicit
compression bounds at every real consumer.

### A.5 Replacement for `RootCommitSource`

The proposed source trait is D-free and object-safe:

```rust,ignore
pub trait CommitmentSource<F>: Send + Sync
where
    F: Field,
{
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError>;

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: CanonicalEncoding;

    fn available_polynomial_types(
        &self,
        plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError>;

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError>;

    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        Ok(None)
    }

    fn prepare_external_inner_commitment(
        &self,
        selected: ExternalInnerCommitmentCapability,
        plan: &CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
        let _unused = (selected, plan);
        Err(AkitaError::InvalidInput(
            "source did not provide the selected external inner commitment capability".into(),
        ))
    }
}
```

Compared with `RootCommitSource`, this trait has:

- no const `D` parameter;
- no associated `CommitView<'a>`;
- no `Clone` bound;
- no requirement that a source implement six dimension-specialized versions;
- side-effect-free discovery of multiple standard polynomial types and
external inner commitment choices;
- materialization of only the selected representation or external operation;
- no backend arithmetic bound on `F`, so a source stays portable across
backends whose kernels have different field requirements; and
- the same mandatory centered-reach admission answer used today.

It does not inherit `RootPolyMeta`. `RootPolyMeta: Clone` is unsuitable for a
trait-object boundary, and its opening/proving consumers remain independent.
A type that is both committed and opened implements both traits. Its
`CommitmentSource` implementation derives descriptor values from the same
stored shape data rather than maintaining a second authority.

Do not add a general `as_any` hook to `CommitmentSource`. The checked external
inner commitment selection owns the narrow erased payload boundary described
in section 5.3; family and context `TypeId`s are validated before safe
downcasting. Strings are diagnostics only.

`available_polynomial_types` may report several choices for one source. It
never builds a lazy cache. The executor selects a supported form first and
calls exactly one of `represent_as` or
`prepare_external_inner_commitment` afterward. `DenseType` distinguishes
coefficients from exact-plan predecomposed planes, while `ShortNormType`
selects packed signed source coefficients as specified in section 5.2;
treating those physical representations as interchangeable is an
execution-plan error.

### A.6 Where concrete backend bounds move

The current full constraint appears at the high-level call:

```rust,ignore
B: RuntimeCommitBackendFor<F, P>
```

After the cutover, bounds apply where concrete prepared operations are built:

```rust,ignore
impl<F> InnerCommitOperation<F> for PreparedCpuCommonInner<'_, F>
where
    F: Field + CanonicalEncoding,
{
    // Dense coefficients, predecomposed planes, packed signed coefficients,
    // and external families supported by the CPU context.
}

impl<F> InnerCommitOperation<F> for PreparedCpuInnerWithOneHot<'_, F>
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator,
{
    // Common paths plus the one-hot column sweep at stored index width.
}

impl<F, B> OuterCommitOperation<F> for PreparedOuter<'_, F, B>
where
    F: Field + CanonicalEncoding,
    B: DigitRowsComputeBackend<F>,
{
    // Call the one canonical checked outer implementation.
}

impl<F, B> CompressionOperation<F> for PreparedCompression<'_, F, B>
where
    F: Field + CanonicalEncoding,
    B: CompressionComputeBackend<F>,
{
    // Call execute_compression_chains.
}
```

The two constructible CPU adapter forms preserve the current dense,
cached-digit, exact-i16, one-hot, multilinear, recursive packed-digit, and
approved external algorithms. The common form does not acquire the
one-hot-only `WithCommitAccumulator` bound. The full form includes those
common paths and adds `OneHotType`; one executor still registers exactly one
inner operation. Both forms are concrete over `CpuBackend` and
`CpuPreparedSetup<F>` because `DigitRowsComputeBackend` alone cannot reach the
current optimized A kernels. A custom backend can instead implement any new
operation trait directly. It does not need to claim support for unrelated
stages.

The state policy has an associated output type and no universal state bound:

```rust,ignore
pub trait CommitmentStatePolicy<F: Field> {
    type State;

    fn bind(
        &self,
        components: CommitmentStateComponents,
        dispatcher: &CommitmentStateDispatcher<F>,
    ) -> Result<Self::State, AkitaError>;
}
```

`CommitmentStateComponents` is a checked consuming carrier for the binding,
inner reference, and optional compression reference. A resident policy moves
it into `ProverOpeningState`; a portable policy consumes it through the
component exporters and builds one hint. No policy gets raw store access.

The root entry point therefore becomes conceptually:

```rust,ignore
pub fn commit<Cfg, P, SP>(
    polys: &[P],
    expanded: &AkitaExpandedSetup<Cfg::Field>,
    schedules: &TrustedScheduleCatalog,
    executor: &CommitmentExecutor<'_, Cfg::Field, SP>,
    context: GroupContext<'_>,
) -> Result<CommitOutput<Cfg::Field, SP::State>, AkitaError>
where
    Cfg: CommitmentConfig,
    P: CommitmentSource<Cfg::Field>,
    SP: CommitmentStatePolicy<Cfg::Field>,
    Cfg::Field: Field + CanonicalEncoding + Ring + Unreduced + 'static,
    <Cfg::Field as Unreduced>::Wide: From<Cfg::Field>,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>;
```

There is no generic backend `B` on `commit()`. Unsupported source, dimension,
or stage combinations become checked request-compilation errors before
arithmetic or transcript mutation. There is also no unconditional bound on
`SP::State`; downstream APIs state their own actual needs.

### A.7 Setup and resource lifecycle

The four operation traits deliberately do not inherit `ComputeBackendSetup`,
but the executor must retain all current operational guarantees. Stage
registration therefore carries an object-safe resource control interface:

```rust,ignore
pub trait CommitmentResourceControl<F>: Send + Sync
where
    F: Field + CanonicalEncoding,
{
    fn ensure_ntt_slot(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<(), AkitaError>;

    fn requirement_is_cached(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<bool, AkitaError>;

    fn cache_owner_id(&self) -> NttCacheOwnerId;

    fn planned_ntt_cache_entry_bytes(
        &self,
        requirement: CommitmentNttRequirement,
    ) -> Result<usize, AkitaError>;

    fn release_built_ntt_slots(&self) -> Result<usize, AkitaError>;

    fn compression_cache_bytes(&self) -> Option<usize> {
        None
    }
}
```

A CPU adapter captures `&B` and `&B::PreparedSetup` and implements these
methods through the existing `ComputeBackendSetup` contract. This adapter is a
real associated-type-erasure and lifecycle boundary, not a forwarding alias.
`CommitmentNttRequirement` contains the exact `NttCacheKey`, routing extent,
and `CommitmentNttStage`; the enclosing routed requirement retains fold-level
information.

External code does not construct the private numeric field of
`NttCacheOwnerId`. The normal builder method accepts a
`ComputeBackendSetup`/prepared pair and obtains its owner through
`ntt_cache_owner_id`. A direct resource implementation either receives an
opaque owner token from the builder or uses a safe public `from_owner(&T)`
constructor added during the cutover. The raw integer remains private. The
builder likewise issues `BackendInstanceId`; strings never establish identity.

A stage that owns no such resources registers an explicit
`StageResources::None`. Missing resource registration must never be silently
interpreted as a no-cache policy. Executor construction validates that every
registered stage has:

- the same setup descriptor as the explicit expanded setup;
- a stable backend instance identity;
- internally coherent advertised dimension, polynomial-type,
external-operation-context, and reduction capabilities; and
- either a resource controller or an explicit no-resources declaration.

Per-call request compilation, not executor construction, checks the selected
plan's dimensions and the actual sources against those capabilities. It
performs that check before materializing the selected representation or
preparing the external inner commitment operation.

Physical resource owners are deduplicated by `NttCacheOwnerId`, preserving the
current behavior when several stages share one prepared cache.

State lifetime is separate from NTT-cache lifetime. A resident output owns an
`Arc` lease to its state owner/store, so dropping the executor or proof stack
does not invalidate a live `CommitOutput`. After the producing fence completes,
state is immutable and safe to share. Cloning a built-in state reference adds a
lease and never consumes the first opening; dropping the last lease performs
best-effort cleanup. `close(self)` closes only that lease and reports a cleanup
error only when it releases the last lease. Other clones remain valid. Calling
close twice is prevented by ownership; dropping an already-closed lease is a
deterministic no-op. A force-close that invalidates sibling leases is not part
of the public API. Retry/request IDs are bound to the generation, so an
abandoned attempt cannot alias a later slot.

Stage-state metadata binds the setup descriptor, exact checked profile and
group geometry, source count, applicable `RingRelationMode`, owner, store,
marker kind, and generation. After the executor forms the public value, the
composite state binding additionally binds the emitted `CommittedGroup`.
Cross-mode state reuse is rejected before transcript mutation. Handles and
backend IDs have redacted
`Debug`; they never serialize or enter transcripts. Registration reports
retained bytes and cleanup policy so memory planning distinguishes device
state, portable state, and NTT caches. The CPU resident route must not keep host
rows twice.

`ProverOpeningData` and `SelectedProverOpeningData` must not recover their
current `Debug + Clone` derives by adding those bounds to generic `S`. Use
redacted/manual debug and ownership by move or `Arc` where sharing is real. Do
not build the current intermediate `Vec<AkitaCommitmentHint>` on a resident
path.

The portable setup-prefix path is the explicit exception: the first cutover
keeps the current `setup_slot.hint.clone()` in
`ProverOpeningData::new_recursive_suffix_fold`. It is existing behavior and
does not justify changing registry storage or serialization in this project.

### A.8 Stack trait migration

The stack integration is required for feature parity. Passing an executor only
to the root `commit()` function would lose per-fold routing for recursive and
terminal commitments.

#### `OperationCtx`

Keep `OperationCtx` for opening, tensor, and ring-switch. Commitment adapters
may also use it internally to capture an existing backend and prepared setup,
but the stack's commitment slot is no longer one `OperationCtx<C>`.

#### `ProverComputeStack`

The current shape is:

```rust,ignore
ProverComputeStack<'a, F, C, O, T, R> {
    commit: OperationCtx<'a, F, C>,
    opening: OperationCtx<'a, F, O>,
    tensor: OperationCtx<'a, F, T>,
    ring_switch: OperationCtx<'a, F, R>,
}
```

The proposed shape is:

```rust,ignore
ProverComputeStack<'a, F, SP, O, T, R> {
    commitment: CommitmentExecutor<'a, F, SP>,
    opening: OperationCtx<'a, F, O>,
    tensor: OperationCtx<'a, F, T>,
    ring_switch: OperationCtx<'a, F, R>,
}
```

Its `commitment()` accessor returns the complete executor. The executor exposes
the full, uncompressed, and inner-only modes needed respectively by root or
compressed recursive commitment, uncompressed recursive commitment, and
terminal commitment. Stack construction also validates the state-consumer
operations required by its opening and ring-switch route. A tier cannot start a
transcript and only then discover that it cannot consume the preceding tier's
state.

#### `LevelProveStacks`

Keep per-fold stack selection, but remove the single `Commit` associated
backend type. Every selected stack already contains its complete commitment
executor. This preserves different commitment routes or backend instances at
different fold levels.

#### `UniformProverStack`

Preserve the common single-backend capability even if it is no longer a
trivial four-backend alias. Provide:

```rust,ignore
CommitmentExecutor::cpu(
    backend,
    prepared,
    expanded,
    state_policy,
)
```

for commit-only, setup-prefix, and profiling callers that own a CPU
backend/prepared pair but do not need a complete proof stack. The full stack
constructor consumes that same canonical executor:

```rust,ignore
ProverComputeStack::new(
    commitment_executor,
    opening,
    tensor,
    ring_switch,
    expanded,
)
```

and a CPU convenience constructor equivalent in effect to today's call:

```rust,ignore
pub fn cpu_uniform(
    backend: &'a CpuBackend,
    prepared: &'a CpuPreparedSetup<F>,
    expanded: &AkitaExpandedSetup<F>,
) -> Result<Self, AkitaError>
```

A custom backend builds a `CommitmentExecutor` from its registered operations
and can still use one backend and prepared resource owner for every stage.

#### `TieredProveStacks`

Keep tiered stack selection. Each tier contains its own executor, so recursive
commitments continue to follow the selected fold-level tier. The compiled
level plan records whether the next tier consumes resident state in place,
uses a declared transfer/import, uses portable state, or explicitly recomputes.
This choice is fixed before the current level mutates the transcript.

#### `ReleaseRootNttAfterFold`

Keep the explicit root-release policy. Extend it to release and deduplicate
every cache owner reachable through the root executor's selected inner, outer,
compression, and fused registrations, as well as the existing opening,
tensor, and ring-switch contexts.

NTT release does not release a live commitment-state lease. State has its own
last-owner/explicit-close lifecycle. Tests must prove the root NTT cache can be
released while a root commitment state remains valid for opening.

### A.9 Stage-specific NTT routing

Today both A and B matrix requirements carry only
`NttOperationCluster::Commit`. Once A and B can use different backends, that
tag cannot identify the correct resource owner. Sending every requirement to
both owners would waste memory and could reject a valid stage-specific
backend.

Add a commitment-stage discriminator:

```rust,ignore
pub enum CommitmentNttStage {
    Inner,
    Outer,
}
```

The canonical NTT requirement plan must then route:

- A requirements to the selected inner operation;
- B requirements to the selected outer operation;
- both A and B requirements to the fused operation when fusion is selected;
- compression-private cache accounting to the compression operation; and
- physically shared owners only once.

`prewarm_ntt_requirements`, `planned_ntt_cache_metrics`, streamed-versus-cached
decisions, and `ReleaseRootNttAfterFold` must migrate in the same stage. Tests
must cover split stages with distinct owners, split stages with one shared
owner, fused routing, streamed requirements, tiered stacks, planned bytes, and
release deduplication.

### A.10 Combined proving traits

The following are not superseded by the commitment executor and remain:

- `RootProvePoly`;
- `RootProveBackend`;
- `ProveFlowBackendFor`;
- `RootProveFlowBackend`;
- `RuntimeRecursiveWitnessProveBackend`;
- `RecursiveProveBackend`;
- `RuntimeOpeningProveBackendFor`;
- `RuntimeCoefficientPackingBackendFor`;
- `RuntimeTensorBackendFor`;
- `RuntimeRingSwitchProveBackend`;
- `SuffixOpeningProveBackend`; and
- `SuffixTensorProveBackend`.

`ProveStackFor` changes only on its commit side: its `C` backend parameter and
commit-side bound disappear because the stack contains a validated executor.
Its opening, tensor, and ring-switch arithmetic requirements remain. Its state
type and operation bounds are explicit: compressed flows require outer-
compression and inner-relation consumption, terminal flows require terminal
binding, persisted setup-prefix production requires portable export, and a
no-retained-state flow requires recomputation. These are separate bounds; do
not replace them with one `CommitmentStateBackend` bundle.

Direct `RuntimeCommitBackendFor<..., RecursiveWitnessFlat>` bounds disappear
from batched proving, fold proving, suffix proving, and ring-switch commitment
entry points only after those functions obtain the executor from their
selected stack.

### A.11 Required call-site migration before deletion

Do not delete the old commitment-only traits until all of these paths use the
executor:

- root `api::commitment::commit`;
- `AkitaCommitmentScheme::commit`;
- `types::opening_data::{ProverOpeningData, ProverGroupInput}` and every
constructor/accessor that currently requires `AkitaCommitmentHint`;
- ring-relation outer-compression reconstruction and
`compression_witness::from_outer_hint`;
- `RingRelationGroupWitness` and `ring_switch::coeffs` inner-row consumption;
- recursive `commit_w`, in compressed and uncompressed modes;
- `commit_terminal_w` inner-only execution;
- `SuffixProverState` and the terminal raw-field transcript binding;
- `api::setup_prefix::commit_setup_prefix`;
- `api::setup::{AkitaProverSetup::prefix_slots, AkitaProverSetup::to_verifier_setup}` and both setup-prefix registries;
- `akita-setup` recursive-prefix construction;
- `akita-setup/src/lib.rs::ntt_caches_rebuilt_correctly_from_disk`;
- setup-prefix serialization and disk-persistence tests;
- `batched_prove`, root-fold, suffix, and fold-level commit bounds;
- `akita-pcs/examples/profile/workload/{multi_group,single_group}.rs`;
- the private `CommitmentWithHint<F>` alias and every constructor that accepts
it;
- downstream commitment contract tests;
- dense, cached-digit, exact-i16, one-hot, multilinear, and recursive source
implementations, including removal of only the commit roles from
`DenseView` and `MultilinearPolynomialView`;
- delegating CPU commitment support; and
- Jolt's packed-trace external-operation integration.

Only then may workspace searches for `RootCommitKernel`, `RootCommitSource`,
`CommitBackendFor`, `RuntimeCommitSource`, and `RuntimeCommitBackendFor` become
empty.

### A.12 Production fused operation

The production fused operation returns resident inner state and `u`:

```rust,ignore
UncompressedCommitmentOutput {
    image: BackendStateRef<InnerImage>,
    u,
}
```

It must not construct or return a hint or host inner rows. The compression
operation similarly retains its witness and mode-specific relation data as
backend state: quotient images in `QuotientLift`, none in
`ReducedEvaluation`. The selected state policy binds those pieces. Only an
explicit portable-export operation may later construct the existing hint, and
a direct resident state consumer need not construct one at all.

### A.13 Feature-preservation review checklist

Reviewers should reject the cutover if any proposed change loses or weakens:

- every supported ring dimension from 64 through 2048;
- dense coefficient, cached-digit, exact-i16, one-hot, multilinear, recursive
packed-digit, downstream standard, or approved external-operation sources;
- recursive compressed, recursive uncompressed, or terminal-inner modes;
- quotient-lift and reduced-evaluation relations, including zero quotient
  rows and no cyclic compression work in reduced mode;
- setup-prefix generation, portable hint serialization, or disk persistence;
- trusted schedule artifact admission, strict row lookup, catalog-digest
  identity, and catalog-derived setup-prefix coverage;
- per-fold and tiered commitment routing;
- exact NTT prewarming and planned memory reporting;
- streamed cache policies and explicit root-cache release;
- cache-owner deduplication across shared prepared state;
- source class and centered-reach admission;
- canonical state-derived inner and compression relation material, including
a byte-identical `AkitaCommitmentHint` when portable export is selected;
- public commitment bytes, proof shape, serialized proof size, or transcript
order; or
- current opening, tensor, coefficient-packing, ring-switch, and verifier
behavior.



### A.14 Review decisions

The trait cutover should proceed only after reviewers accept all of these
decisions:

1. Delete only the five commitment-specific surfaces after migration:
  `RootCommitKernel`, `RootCommitSource`, `CommitBackendFor`,
   `RuntimeCommitSource`, and `RuntimeCommitBackendFor`.
2. Keep the shared setup, digit-row, compression, cyclic-row, opening, tensor,
  and proving traits in the first cutover.
3. Use four independent minimal operation traits with no common supertrait.
4. Use `CommitmentExecutor` plus a per-call `CommitmentExecutionPlan`; do not
  add a separate public `CommitmentStageRouting` wrapper.
5. Carry backend identity, setup identity, capabilities, and explicit resource
  control in stage registration.
6. Evolve the stack and NTT requirement routing so tiering, prewarming,
  reporting, streaming, and release remain supported.
7. Keep production fused output as resident inner state plus `u`, with no host
  rows or hint.
8. Leave `DigitRowsComputeBackend: CompressionComputeBackend` unchanged during
  this implementation and review any global decoupling separately.
9. Make capability discovery O(1) and side-effect-free, allow a source to
  advertise multiple polynomial types, and materialize only the selected
   representation or external operation.
10. Keep predecomposed digit planes distinct from packed signed source
  coefficients, including exact plan-key and layout validation.
11. Use distinct checked plan constructors for root, recursive, terminal, and
  setup-prefix modes while sharing one set of arithmetic subplans.
12. Make every type in a public operation signature publicly nameable and
  constructible through checked APIs, without turning execution data into a
    protocol message.
13. Make `CommitOutput` generic over prover state. Add no universal hint,
  serialization, `Clone`, or host-row bound.
14. Keep `SetupPrefixSlot` and its portable hint bytes unchanged; resident
  state stays in `akita-prover` and crosses persistence only through explicit
    portable export/import.
15. Plan state consumption and transfers before the proof transcript starts;
  validate each produced handle before the next transcript mutation can
    depend on it, and bind it to owner, setup, plan, public commitment, store,
    kind, and generation.
16. Scope source/backend decoupling to commitment execution; opening and tensor
  source/kernel products remain separate work.
17. Reuse and extend the existing `CommitInnerPlan`; do not add a second
  `InnerCommitPlan` type with transposed naming.
18. Preserve setup-prefix prover/verifier registries and bytes. Bare
  setup-prefix callers construct a CPU executor without a full proof stack,
    and `commit_setup_prefix` exports portable state before returning its slot.
19. Emit one sanitized route summary through existing tracing infrastructure;
  do not add a separate public diagnostics subsystem.
20. Keep `TrustedScheduleCatalog` owned by `AkitaCommitmentScheme` and passed
    explicitly to low-level schedule resolution. The executor accepts only an
    already-resolved profile and never runs planner search.
21. Pass `RingRelationMode` explicitly to compression and bind it into retained
    state. Preserve quotient images only for `QuotientLift`; preserve the
    zero-quotient negacyclic-only path for `ReducedEvaluation`.
22. Derive setup-prefix slot IDs and persisted-registry coverage from the same
    trusted catalog, and keep disk registry names bound to `catalog_digest`.
