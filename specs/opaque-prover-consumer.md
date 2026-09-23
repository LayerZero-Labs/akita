# Spec: Opaque Prover Consumer

| Field | Value |
|---|---|
| Author(s) | |
| Created | 2026-09-18 |
| Status | proposed |
| PR | |
| Supersedes | Opaque recursive-witness carrier proposal |
| Superseded-by | |
| Book-chapter | |

## Summary

Make Akita protocol orchestration independent of CPU witness representations.
Protocol code owns the transcript, validated plans, canonical layouts, and
verifier-visible messages. A selected prover consumer owns or opaquely carries
every private implementation value and exposes only typed handles with public
metadata.

This is an API-opacity boundary, not an in-process security boundary. An
in-process CPU handle may directly own private CPU state behind private fields;
a GPU, process, enclave, or remote implementation may instead store state in a
resident registry and return an owner-bound generational key. Both models
implement the same public contracts.

The cutover preserves CPU algorithms, packed representations, application
diagnostics, proof bytes, and transcript order. Physical routing is internal to
the owning backend; generic proving uses one coherent opaque handle family.

## Naming rule

Every protocol-level value that transports an opaque private resource rather
than the object named by the type must use a `*_handle` name. Internal identity
components retain precise `*_id` and `*_key` names; for example, a
`GpuWitnessHandle` contains a private `resident_key`, while protocol code names
the complete carrier `witness_handle`.

Examples:

- `witness_handle`, never `witness`, when the value is `WitnessHandle`;
- `build_handle`, never `build`, when the value is `WitnessBuildHandle`;
- `relation_handle`, never `relation`, when the value is `RelationHandle`;
- `session_handle`, never `session`, when the value is a session handle;
- `commitment_material_handle`, never `material`, for retained private state;
- `opening_handle`, never `opening`, for private opening state;
- `fold_handle`, never `fold`, for an accepted private fold.

Names without the suffix denote real values. For example,
`RingRelationInstance<F>` is public verifier-visible data and is correctly
called `instance`; a `PrivateRecursiveWitness` inside the CPU backend is an
actual private witness and is correctly called `witness_state`.

Generic type parameters follow the same rule: `WitnessHandle`, `BuildHandle`,
`RelationHandle`, and `MaterialHandle`, not `H`, `W`, `S`, or `M` when a more
precise name is available.

## End-state ownership

Akita protocol owns:

- transcript and challenge order;
- validated plans and canonical layouts;
- proof-visible commitments and relation instances;
- opening messages and round polynomials;
- public claims;
- terminal encoded proof bytes;
- public handle metadata;
- proof orchestration and error handling.

The selected consumer owns or opaquely carries:

- source representations after import;
- accepted `z` and `z_i` fold responses;
- E, T, R, and compression witness material;
- complete recursive witnesses;
- commitment-domain witness representations;
- packed relation tables;
- Stage 1 and Stage 2 mutable prover states;
- tensor evaluation tables;
- extension-opening-reduction state;
- terminal-fold coefficients.

Diagnostics remain available as an explicit, feature-gated aggregate
declassification. They are not security inputs.

## Module structure

```text
crates/akita-prover/src/
|-- compute/
|   |-- opaque/
|   |   |-- mod.rs
|   |   |-- diagnostics.rs
|   |   |-- handles.rs
|   |   |-- kernels.rs
|   |   |-- messages.rs
|   |   `-- scope.rs
|   |-- recursive_primitives/
|   |   |-- formulas.rs
|   |   |-- geometry.rs
|   |   `-- lookup_constants.rs
|   |-- backend.rs
|   |-- operation_plans.rs
|   |-- runtime_capabilities.rs
|   `-- stack.rs
|-- backend/
|   |-- cpu/
|   |   |-- opaque/
|   |   |   |-- fold.rs
|   |   |   |-- witness.rs
|   |   |   |-- commitment.rs
|   |   |   |-- relation.rs
|   |   |   `-- opening.rs
|   |   `-- recursive/
|   |       |-- witness/
|   |       |-- commitment/
|   |       |-- digit_range/
|   |       |-- relation_range_image/
|   |       |-- two_round_prefix/
|   |       `-- opening/
|   `-- gpu/                       # future accelerator crate or module
|       |-- opaque/
|       |   |-- consumer.rs
|       |   `-- registry.rs
|       `-- recursive/
|           |-- witness/
|           |-- commitment/
|           |-- digit_range/
|           |-- relation_range_image/
|           |-- two_round_prefix/
|           `-- opening/
`-- protocol/
    `-- public orchestration only
```

The current concrete implementations under `backend/recursive` are CPU
implementations. Move them mechanically under `backend/cpu/recursive`; do not
leave them at a backend-neutral layer.

Only code that is genuinely independent of storage and execution belongs in
`compute/recursive_primitives`, such as canonical geometry, pure formulas, and
constant lookup-table generation. Shared code must not contain host `Vec`
state, Rayon, SIMD dispatch, CPU sessions, device buffers, stream policy, or
backend cache ownership.

## Public metadata and handle contracts

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceMetadata {
    num_polynomials: usize,
    num_vars: usize,
}

impl SourceMetadata {
    pub const fn num_polynomials(self) -> usize {
        self.num_polynomials
    }

    pub const fn num_vars(self) -> usize {
        self.num_vars
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecursiveWitnessManifest {
    logical_len: usize,
    commitment_domain_len: usize,
    commitment_ring_dimension: usize,
}

impl RecursiveWitnessManifest {
    pub fn try_new(
        logical_len: usize,
        commitment_domain_len: usize,
        commitment_ring_dimension: usize,
    ) -> Result<Self, AkitaError> {
        if logical_len == 0
            || commitment_domain_len < logical_len
            || commitment_ring_dimension == 0
            || !commitment_ring_dimension.is_power_of_two()
        {
            return Err(AkitaError::InvalidInput(
                "invalid recursive-witness manifest".into(),
            ));
        }
        Ok(Self {
            logical_len,
            commitment_domain_len,
            commitment_ring_dimension,
        })
    }

    pub const fn logical_len(self) -> usize {
        self.logical_len
    }

    pub const fn commitment_domain_len(self) -> usize {
        self.commitment_domain_len
    }

    pub const fn commitment_ring_dimension(self) -> usize {
        self.commitment_ring_dimension
    }
}

pub trait OpaqueSourceHandle: Send + Sync {
    fn metadata(&self) -> SourceMetadata;
}

pub trait AcceptedFoldHandle: Send + Sync + 'static {
    fn metadata(&self) -> AcceptedFoldMetadata;
}

pub trait RecursiveWitnessHandle: Send + Sync + 'static {
    fn manifest(&self) -> RecursiveWitnessManifest;
}
```

`commitment_domain_len` is immutable planned geometry. It does not imply that
the committed representation has already been materialized.

No handle exposes coefficient, digit, view, packed-storage, energy, downcast,
serialization, or `Any` accessors. Linear handles such as builds, accepted
folds, commitment material, openings, and sessions do not implement `Clone` or
`Copy`.

## Consumer type family

```rust
pub trait OpaqueProverConsumer<F, E>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type PreparedOpeningHandle: Send + 'static;
    type CommitmentMaterialHandle: Send + 'static;

    type AcceptedFoldHandle: AcceptedFoldHandle;
    type AcceptedTerminalFoldHandle: Send + 'static;

    type WitnessBuildHandle: Send + 'static;
    type WitnessHandle: RecursiveWitnessHandle;

    type RelationHandle: Send + 'static;
    type Stage1SessionHandle: Send + 'static;
    type Stage2SessionHandle: Send + 'static;

    type WitnessOpeningHandle: Send + 'static;
    type WitnessEorSessionHandle: Send + 'static;
}
```

This is one logical handle family. Its implementation may route commitment,
opening, tensor, and ring-switch work internally. A single reusable `CpuBackend`
owns configuration and private resources; generic proving does not compose
physical routes. Cross-owner reuse requires an explicit validated import.

## Proof-scoped ownership

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProofScopeId(u128);

pub trait ProofScopeConsumer {
    fn begin_scope(&self) -> Result<ProofScopeId, AkitaError>;
    fn finish_scope(&self, scope: ProofScopeId) -> Result<(), AkitaError>;
    fn abort_scope_best_effort(&self, scope: ProofScopeId);
}

pub struct ProofScope<'a, Consumer: ProofScopeConsumer> {
    consumer: &'a Consumer,
    scope_id: ProofScopeId,
    finished: bool,
}

impl<Consumer: ProofScopeConsumer> ProofScope<'_, Consumer> {
    pub fn begin(consumer: &Consumer) -> Result<ProofScope<'_, Consumer>, AkitaError> {
        Ok(ProofScope {
            consumer,
            scope_id: consumer.begin_scope()?,
            finished: false,
        })
    }

    pub const fn id(&self) -> ProofScopeId {
        self.scope_id
    }

    pub fn finish(mut self) -> Result<(), AkitaError> {
        self.consumer.finish_scope(self.scope_id)?;
        self.finished = true;
        Ok(())
    }
}

impl<Consumer: ProofScopeConsumer> Drop for ProofScope<'_, Consumer> {
    fn drop(&mut self) {
        if !self.finished {
            self.consumer.abort_scope_best_effort(self.scope_id);
        }
    }
}
```

CPU handles normally release their direct state through `Drop`. GPU, process,
and remote consumers use the proof scope for bulk cleanup on rejection, error,
device loss, or early caller return.

Every private value is internally bound to consumer identity, proof scope,
setup digest, fold level, and a unique operation identity. Metadata equality is
never sufficient to establish identity.

## Source ingestion

A `'static` source handle would force the CPU path to copy or heap-own borrowed
input. Source retention therefore uses a lifetime-bearing associated type.

```rust
pub trait RetainSourceGroupKernel<F, Polynomial>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    type SourceHandle<'a>: OpaqueSourceHandle + 'a
    where
        Self: 'a,
        Polynomial: 'a;

    fn retain_source_group<'a>(
        &'a self,
        prepared: Option<&'a Self::PreparedSetup>,
        context: &ProofContext,
        polynomials: &[&'a Polynomial],
        plan: &ValidatedSourceGroupPlan,
    ) -> Result<Self::SourceHandle<'a>, AkitaError>;
}
```

Root ingestion uses `RootProverGroupOpening::retain_for_proof` once for each
ordered group and fold level. The resulting `RetainedProverGroupOpening`
resource serves opening preparation and every fold retry. CPU ingestion calls
`RetainSourceGroupKernel` and retains references to the original polynomials;
`OpaqueFoldKernel` dispatches through the selected opening backend's
`FoldHandleBackend` family. The proof context binds owner, setup, scope, level,
and group. Recursive private witnesses reuse their existing consumer handle.

The CPU source handle may borrow the original representation without a copy.
The GPU implementation uploads once and returns a scope-bound resident handle.
Dense, one-hot, sparse-ring, and packed recursive sources retain their
representation-specific ingestion paths; no common dense `Vec<F>` conversion
is introduced.

## Fold kernels

```rust
pub enum FoldProbeOutcome<FoldHandle> {
    Rejected,
    Accepted {
        fold_handle: FoldHandle,
        diagnostics: FoldProbeDiagnostics,
    },
}

pub trait OpaqueFoldKernel<F, SourceHandle>:
    FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
    SourceHandle: OpaqueSourceHandle,
{
    fn probe_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        source_handle: &SourceHandle,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<
        FoldProbeOutcome<Self::AcceptedFold>,
        AkitaError,
    >;

}

pub trait OpaqueTerminalFoldKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn probe_terminal_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<
        FoldProbeOutcome<Self::AcceptedTerminalFoldHandle>,
        AkitaError,
    >;

    fn encode_terminal_fold(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        terminal_fold_handle: Self::AcceptedTerminalFoldHandle,
        plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError>;
}
```

The implementation retains per-chunk admission, checked `z = sum_i z_i`,
canonical chunk ranges, response bounds, terminal coding checks, and operation
binding. Encoding consumes the terminal-fold handle.

Prepared group openings follow the same naming rule:

```rust
pub struct PreparedGroupOpening<E: Field, OpeningHandle> {
    scalar_openings: Vec<E>,
    opening_handle: OpeningHandle,
}

impl<E: Field, OpeningHandle>
    PreparedGroupOpening<E, OpeningHandle>
{
    pub fn scalar_openings(&self) -> &[E] {
        &self.scalar_openings
    }

    pub fn into_opening_handle(self) -> OpeningHandle {
        self.opening_handle
    }
}
```

## Diagnostics

The diagnostic feature is preserved. Diagnostic aggregates are explicit and
feature-gated rather than exposed as witness accessors.

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct FoldProbeDiagnostics {
    observed_l2_sq: Option<u128>,

    #[cfg(feature = "response-model-diagnostics")]
    source_l2_sq: Option<u128>,
}

impl FoldProbeDiagnostics {
    pub const fn observed_l2_sq(self) -> Option<u128> {
        self.observed_l2_sq
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub const fn source_l2_sq(self) -> Option<u128> {
        self.source_l2_sq
    }
}
```

With `response-model-diagnostics`, the consumer reports the exact source energy
and observed response energy used by the current calibration logs. Protocol
code continues to derive the challenge-scaled conditional mean. Without the
feature, there is no source scan, allocation, logging, or device transfer.

## Two-phase recursive witness construction

Transcript-owned fold grinding occurs between the two phases.

```rust
pub struct RecursiveWitnessBuildStart<
    F: Field,
    E: Field,
    BuildHandle,
> {
    public_groups: Vec<PreparedRelationGroupPublic<F, E>>,
    relation_rhs: RingVec<F>,
    opening_payload: RingVec<F>,
    opening_payload_ring_dimension: usize,
    v: RingVec<F>,
    build_handle: BuildHandle,
}

impl<F: Field, E: Field, BuildHandle>
    RecursiveWitnessBuildStart<F, E, BuildHandle>
{
    pub fn public_groups(&self) -> &[PreparedRelationGroupPublic<F, E>] {
        &self.public_groups
    }

    pub fn relation_rhs(&self) -> &RingVec<F> {
        &self.relation_rhs
    }

    pub fn opening_payload(&self) -> &RingVec<F> {
        &self.opening_payload
    }

    pub const fn opening_payload_ring_dimension(&self) -> usize {
        self.opening_payload_ring_dimension
    }

    pub fn v(&self) -> &RingVec<F> {
        &self.v
    }

    pub fn into_build_handle(self) -> BuildHandle {
        self.build_handle
    }
}

pub struct RecursiveWitnessBuildOutput<F: Field, WitnessHandle> {
    instance: RingRelationInstance<F>,
    witness_handle: WitnessHandle,
}

impl<F: Field, WitnessHandle>
    RecursiveWitnessBuildOutput<F, WitnessHandle>
{
    pub fn new(
        instance: RingRelationInstance<F>,
        witness_handle: WitnessHandle,
    ) -> Self {
        Self {
            instance,
            witness_handle,
        }
    }

    pub fn into_instance_and_witness_handle(
        self,
    ) -> (RingRelationInstance<F>, WitnessHandle) {
        (self.instance, self.witness_handle)
    }
}
```

The output contains a real public relation instance and an opaque witness
handle. It does not contain a recursive witness representation.

```rust
pub struct RecursiveWitnessFoldInput<FoldHandle> {
    fold_handle: FoldHandle,
    challenges: GroupFoldChallenges,
}

impl<FoldHandle> RecursiveWitnessFoldInput<FoldHandle> {
    pub fn into_fold_handle_and_challenges(
        self,
    ) -> (FoldHandle, GroupFoldChallenges) {
        (self.fold_handle, self.challenges)
    }
}

pub trait OpaqueRecursiveWitnessBuildKernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    #[allow(clippy::too_many_arguments)]
    fn begin_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        scope_id: ProofScopeId,
        prepared_opening_handles: Vec<Self::PreparedOpeningHandle>,
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation_rhs_layout: &RelationRhsLayout,
        group_commitments: &[RingVec<F>],
    ) -> Result<
        RecursiveWitnessBuildStart<
            F,
            E,
            Self::WitnessBuildHandle,
        >,
        AkitaError,
    >;

    fn finish_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<
            RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>,
        >,
        relation: &RingRelationInstance<F>,
        plan: &ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<Self::WitnessHandle, AkitaError>;
}
```

This replaces the current protocol-visible intermediate
`RingRelationWitness`. Complete private witness construction happens inside the
consumer before the final witness handle is returned. The protocol constructs
and validates the public `RingRelationInstance` once, then lends that canonical
instance to the consumer. A consumer may check backend-private material against
the instance, but must not construct a parallel public relation statement.

## Commitment lifecycle

Commitment consumes the old witness handle and returns a new handle. This
prevents stale manifests and aliased mutable lifecycle state.

```rust
pub struct WitnessCommitmentOutput<
    F: Field,
    WitnessHandle,
    MaterialHandle,
> {
    commitment: Option<RingVec<F>>,
    witness_handle: WitnessHandle,
    commitment_material_handle: MaterialHandle,
}

impl<F: Field, WitnessHandle, MaterialHandle>
    WitnessCommitmentOutput<F, WitnessHandle, MaterialHandle>
{
    pub fn new(
        commitment: Option<RingVec<F>>,
        witness_handle: WitnessHandle,
        commitment_material_handle: MaterialHandle,
    ) -> Self {
        Self {
            commitment,
            witness_handle,
            commitment_material_handle,
        }
    }

    pub fn commitment(&self) -> Option<&RingVec<F>> {
        self.commitment.as_ref()
    }

    pub fn into_commitment_parts(
        self,
    ) -> (Option<RingVec<F>>, WitnessHandle, MaterialHandle) {
        (
            self.commitment,
            self.witness_handle,
            self.commitment_material_handle,
        )
    }
}

pub trait OpaqueWitnessCommitKernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn commit_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: Self::WitnessHandle,
        plan: &ValidatedRecursiveWitnessCommitPlan,
    ) -> Result<
        WitnessCommitmentOutput<
            F,
            Self::WitnessHandle,
            Self::CommitmentMaterialHandle,
        >,
        AkitaError,
    >;
}
```

The consumer's prepared state owns the selected commitment route and resources.
An external executor must not request a raw witness source from the handle.

## Relation witness and Stage 1/2

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelationWitnessMetadata {
    witness_len: usize,
    column_bits: usize,
    coefficient_bits: usize,
}

pub struct PreparedRelationHandle<RelationHandle> {
    relation_handle: RelationHandle,
    metadata: RelationWitnessMetadata,
}

impl<RelationHandle> PreparedRelationHandle<RelationHandle> {
    pub const fn metadata(&self) -> RelationWitnessMetadata {
        self.metadata
    }

    pub fn into_relation_handle(self) -> RelationHandle {
        self.relation_handle
    }
}

pub trait OpaqueRelationWitnessKernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn prepare_relation_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        plan: &ValidatedRelationWitnessPlan,
    ) -> Result<
        PreparedRelationHandle<Self::RelationHandle>,
        AkitaError,
    >;
}
```

Session methods use `&mut SessionHandle`. This lets the CPU handle directly own
the existing session without an arena or mutex. Stage 2 consumes the relation
handle after Stage 1 finishes.

```rust
pub trait OpaqueStage1Kernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn begin_stage1(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        relation_handle: &Self::RelationHandle,
        plan: &ValidatedStage1Plan<E>,
    ) -> Result<Self::Stage1SessionHandle, AkitaError>;

    fn stage1_round_polynomial(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<Stage1RoundPolynomial<E>, AkitaError>;

    fn bind_stage1_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn stage1_public_transition(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: Stage1Step,
    ) -> Result<Stage1PublicTransition<E>, AkitaError>;

    fn bind_stage1_batch_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        transition: Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_stage1(
        &self,
        session_handle: Self::Stage1SessionHandle,
    ) -> Result<Stage1FinalClaims<E>, AkitaError>;
}

pub trait OpaqueStage2Kernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn begin_stage2(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        relation_handle: Self::RelationHandle,
        plan: ValidatedRelationSessionPlan<'_, E>,
    ) -> Result<Self::Stage2SessionHandle, AkitaError>;

    fn stage2_input_claim(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<E, AkitaError>;

    fn stage2_num_rounds(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<usize, AkitaError>;

    fn stage2_round_polynomial(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        previous_claim: E,
    ) -> Result<UniPoly<E>, AkitaError>;

    fn bind_stage2_challenge(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_stage2(
        &self,
        session_handle: Self::Stage2SessionHandle,
    ) -> Result<RelationWitnessFinalClaims<E>, AkitaError>;
}
```

The CPU backend retains its low-basis range prover, range tree, compact digit
source, prefix caches, deferred compact prefixes, fused relation/range scans,
and physical-L2 fusion. Those are implementation details under
`backend/cpu/recursive`.

## Witness opening and EOR

```rust
pub struct PreparedWitnessOpening<E: Field, OpeningHandle> {
    messages: Vec<E>,
    folded_messages: Vec<E>,
    opening_handle: OpeningHandle,
}

impl<E: Field, OpeningHandle>
    PreparedWitnessOpening<E, OpeningHandle>
{
    pub fn messages(&self) -> &[E] {
        &self.messages
    }

    pub fn folded_messages(&self) -> &[E] {
        &self.folded_messages
    }

    pub fn into_opening_handle(self) -> OpeningHandle {
        self.opening_handle
    }
}

pub trait OpaqueWitnessOpeningKernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn prepare_witness_opening(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        plan: &ValidatedWitnessOpeningPlan<'_, E>,
    ) -> Result<
        PreparedWitnessOpening<E, Self::WitnessOpeningHandle>,
        AkitaError,
    >;

    fn begin_witness_eor(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        opening_handle: Self::WitnessOpeningHandle,
        plan: &ValidatedWitnessEorPlan<'_, E>,
    ) -> Result<Self::WitnessEorSessionHandle, AkitaError>;

    fn eor_round_polynomial(
        &self,
        session_handle: &mut Self::WitnessEorSessionHandle,
        round: usize,
        previous_claim: E,
    ) -> Result<UniPoly<E>, AkitaError>;

    fn bind_eor_challenge(
        &self,
        session_handle: &mut Self::WitnessEorSessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_witness_eor(
        &self,
        session_handle: Self::WitnessEorSessionHandle,
    ) -> Result<WitnessEorFinalClaims<E>, AkitaError>;
}
```

`begin_witness_eor` consumes the opening handle so a tensor table cannot remain
accidentally live after it has moved into the EOR state.

## CPU implementation

The CPU implementation uses direct opaque ownership. It does not introduce a
global state arena, a global mutex, generation lookup, or a witness-sized copy.

```rust
pub struct CpuAcceptedFoldHandle<F: Field> {
    metadata: AcceptedFoldMetadata,
    binding: OperationBinding,
    fold_state: PrivateAcceptedFold<F>,
}

struct PrivateAcceptedFold<F: Field> {
    global: DecomposeFoldWitness<F>,
    chunks: Vec<DecomposeFoldWitness<F>>,
}

pub struct CpuWitnessHandle {
    manifest: RecursiveWitnessManifest,
    binding: OperationBinding,
    witness_state: PrivateRecursiveWitness,
}

struct PrivateRecursiveWitness {
    logical: RecursiveWitnessFlat,
    committed: Option<RecursiveWitnessFlat>,
}

pub struct CpuCommitmentMaterialHandle<F: Field> {
    binding: OperationBinding,
    commitment_material_state: PrivateCommitmentMaterial<F>,
}

pub struct CpuRelationHandle {
    metadata: RelationWitnessMetadata,
    binding: OperationBinding,
    relation_state: ConsumerRelationWitness,
}

pub struct CpuStage1SessionHandle<E: Field> {
    binding: OperationBinding,
    session_state: DigitRangeSession<E>,
}

pub struct CpuStage2SessionHandle<E: Field> {
    binding: OperationBinding,
    session_state: ConsumerStage2Session<E>,
}

pub struct CpuWitnessOpeningHandle<E: Field> {
    binding: OperationBinding,
    opening_state: OpaqueWitnessOpeningState<E>,
}

pub struct CpuWitnessEorSessionHandle<E: Field> {
    binding: OperationBinding,
    session_state: CpuExtensionOpeningSession<E>,
}
```

These public handle types have private fields and private constructors. Protocol
code can move them but cannot inspect their CPU state.

Commitment consumes and reconstructs the witness handle:

```rust
fn commit_witness(
    &self,
    prepared: Option<&Self::PreparedSetup>,
    witness_handle: CpuWitnessHandle,
    plan: &ValidatedRecursiveWitnessCommitPlan,
) -> Result<
    WitnessCommitmentOutput<
        F,
        CpuWitnessHandle,
        CpuCommitmentMaterialHandle<F>,
    >,
    AkitaError,
> {
    let prepared = prepared.ok_or_else(|| {
        AkitaError::InvalidInput(
            "recursive commitment requires prepared setup".into(),
        )
    })?;

    witness_handle.binding.validate(
        self.id,
        prepared,
        plan.operation_context(),
    )?;

    let CpuWitnessHandle {
        binding,
        witness_state,
        ..
    } = witness_handle;

    let committed = commit_private_witness(
        witness_state,
        &self.routes.commitment,
        plan,
    )?;

    Ok(WitnessCommitmentOutput::new(
        committed.public_commitment,
        CpuWitnessHandle {
            manifest: committed.manifest,
            binding: binding.clone(),
            witness_state: committed.witness_state,
        },
        CpuCommitmentMaterialHandle {
            binding,
            commitment_material_state:
                committed.commitment_material_state,
        },
    ))
}
```

## GPU implementation

A GPU, remote, or process consumer may keep private state resident and return
generational keys.

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ResidentKey {
    consumer: ConsumerId,
    scope: ProofScopeId,
    slot: u32,
    generation: u32,
}

pub struct GpuWitnessHandle {
    resident_key: ResidentKey,
    manifest: RecursiveWitnessManifest,
}

struct GpuRecursiveWitnessState {
    logical_buffer: DeviceBuffer<PackedWord>,
    committed_buffer: Option<DeviceBuffer<PackedWord>>,
    ready_event: GpuEvent,
    manifest: RecursiveWitnessManifest,
}

struct GpuRelationState {
    packed_buffer: DeviceBufferLease<PackedWord>,
    column_bits: usize,
    coefficient_bits: usize,
    ready_event: GpuEvent,
}

struct GpuStage1SessionState<E> {
    stream: GpuStream,
    device_state: DeviceBuffer<Stage1DeviceState<E>>,
    pending_round: Option<Stage1PendingRound<E>>,
    ready_event: GpuEvent,
}

struct GpuStage2SessionState<E> {
    stream: GpuStream,
    witness_buffer: DeviceBufferLease<PackedWord>,
    folded_state: DeviceBuffer<E>,
    relation_weights: DeviceBuffer<E>,
    pending_round_polynomial: Option<UniPoly<E>>,
    ready_event: GpuEvent,
}
```

Use separate registries by resource kind. A registry lock validates and clones
an entry reference, then is released before any kernel launch, device wait, or
copy. Mutable session serialization is per session, never global.

```rust
pub struct GpuProverConsumer<F, E> {
    id: ConsumerId,
    devices: Vec<GpuDevice>,
    routing: GpuRoutingTable,
    witnesses: RwLock<GenerationalArena<Arc<GpuRecursiveWitnessState>>>,
    relations: RwLock<GenerationalArena<Arc<GpuRelationState>>>,
    stage1_sessions: RwLock<
        GenerationalArena<Arc<Mutex<GpuStage1SessionState<E>>>>,
    >,
    stage2_sessions: RwLock<
        GenerationalArena<Arc<Mutex<GpuStage2SessionState<E>>>>,
    >,
}
```

One interactive round returns only a small public polynomial. Binding the
challenge enqueues the next in-place fold on the same stream and need not wait
synchronously. The next operation observes stream ordering.

The transfer boundary is:

```text
host -> GPU once:
    source upload
    public prepared setup

host -> GPU per interactive round:
    one transcript challenge

GPU -> host:
    public commitments
    opening payloads
    round-polynomial coefficients
    public claims
    terminal proof bytes
    optional diagnostic aggregates

never downloaded:
    z and z_i
    packed recursive witness
    E/T/R/compression witness material
    relation digit tables
    tensor evaluation tables
    mutable sumcheck tables
```

Unsupported field, ring, or device combinations fail during preflight before
transcript mutation. There is no silent CPU fallback.

## Explicit resource release

Long-lived witness handles support explicit release. All other linear handles
are consumed by their normal lifecycle methods or reclaimed with the proof
scope.

```rust
pub trait OpaqueResourceReleaseKernel<F, E>:
    OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn release_witness_handle(
        &self,
        witness_handle: Self::WitnessHandle,
    ) -> Result<(), AkitaError>;
}
```

## Protocol flow after cutover

```rust
let proof_scope = ProofScope::begin(consumer)?;

let build_start = consumer.begin_recursive_witness(
    Some(prepared),
    proof_scope.id(),
    prepared_opening_handles,
    commitment_material_handles,
    level,
    &opening_batch,
    relation_rhs_layout,
    group_commitments,
)?;

build_start.opening_payload().append_flat_to_transcript(
    ABSORB_OPENING_PAYLOAD,
    build_start.opening_payload_ring_dimension(),
    transcript,
)?;

let gamma = prepare_gamma(transcript, &opening_batch)?;
let row_coefficient_rings =
    prepare_row_coefficient_rings(transcript, &opening_batch)?;

let fold_inputs = grind_fold_handles(
    consumer,
    source_handles,
    transcript,
    level,
)?;

// The protocol constructs and validates the canonical instance from the
// transcript-derived challenges, public openings, and relation RHS here.
let witness_handle = consumer.finish_recursive_witness(
    Some(prepared),
    build_start.into_build_handle(),
    fold_inputs,
    &instance,
    &witness_plan,
)?;

let commitment_output = consumer.commit_witness(
    Some(prepared),
    witness_handle,
    &commit_plan,
)?;

let (
    public_commitment,
    witness_handle,
    commitment_material_handle,
) = commitment_output.into_commitment_parts();

let prepared_relation_handle = consumer.prepare_relation_witness(
    Some(prepared),
    &witness_handle,
    &relation_plan,
)?;

let relation_metadata = prepared_relation_handle.metadata();
let relation_handle = prepared_relation_handle.into_relation_handle();

let mut stage1_session_handle = consumer.begin_stage1(
    Some(prepared),
    &relation_handle,
    &stage1_plan,
)?;

let stage1_output = drive_stage1_transcript(
    consumer,
    &mut stage1_session_handle,
    transcript,
)?;

let stage1_final_claims =
    consumer.finish_stage1(stage1_session_handle)?;

let mut stage2_session_handle = consumer.begin_stage2(
    Some(prepared),
    relation_handle,
    stage2_plan,
)?;

let stage2_output = drive_stage2_transcript(
    consumer,
    &mut stage2_session_handle,
    transcript,
)?;

let stage2_final_claims =
    consumer.finish_stage2(stage2_session_handle)?;

consumer.release_witness_handle(witness_handle)?;
proof_scope.finish()?;
```

Every local carrying an opaque object is named `*_handle`. Public values such as
`instance`, `public_commitment`, `stage1_output`, and final claims remain named
for the actual value they contain.

## Remove or privatize

Remove from crate-root and production public exports:

- `RecursiveWitnessFlat`;
- `DecomposeFoldWitness`;
- `OpaqueRecursiveWitness`;
- `OpaqueWitnessOpeningState`;
- `RingRelationWitness`;
- `CommittedRecursiveWitness`;
- `ConsumerRelationWitness`;
- `ConsumerStage2Session`;
- `CpuExtensionOpeningSession`;
- `SuffixWitnessView`;
- `SuffixWitnessBatchView`;
- `CpuRecursiveWitnessAssemblyState`;
- raw commitment and compression witness material types;
- relation quotient witness types.

Make raw constructors and accessors `pub(crate)` or private:

- `RecursiveWitnessFlat::from_i8_digits`;
- `RecursiveWitnessFlat::view`;
- `RecursiveWitnessFlat::packed_digits`;
- `RecursiveWitnessFlat::to_i8_digits`;
- `DecomposeFoldWitness::from_parts`;
- `DecomposeFoldWitness::from_coefficient_parts`;
- `DecomposeFoldWitness::centered_coeffs_flat`;
- `DecomposeFoldWitness::z_folded_rings_trusted`.

Replace protocol-visible carrier contracts:

- `RelationWitnessSession`;
- `RecursiveWitnessInputs`;
- `RecursiveWitnessKernel`;
- `RecursiveWitnessAssemblyFinish` returning `RingRelationWitness`;
- `RecursiveWitnessCommitmentState`;
- any associated session type that exposes a concrete CPU prover.

`DigitRangeProver` and `LowBasisRangeCheckProver` stop being crate-root
production exports. They may remain available from a CPU test or benchmark
module for named standalone scenarios.

## Add

Add representation-free contracts:

- `OpaqueProverConsumer`;
- `OpaqueSourceHandle`;
- `AcceptedFoldHandle`;
- `RecursiveWitnessHandle`;
- `RecursiveWitnessManifest`;
- `SourceMetadata`;
- `RelationWitnessMetadata`;
- `ProofScope` and `ProofScopeId`;
- `RecursiveWitnessBuildStart`;
- `RecursiveWitnessBuildOutput` with `witness_handle`;
- `WitnessCommitmentOutput` with `witness_handle` and
  `commitment_material_handle`;
- `PreparedRelationHandle` with `relation_handle`;
- `PreparedWitnessOpening` with `opening_handle`;
- opaque fold, build, commitment, relation, Stage 1, Stage 2, and EOR kernels;
- feature-gated diagnostic aggregate accessors.

Add private CPU handle types for every private lifecycle stage, using direct
ownership. Add resident GPU handle types, per-kind generational registries,
device leases, events, routing, transfer accounting, and proof-scope cleanup
when accelerator work begins.

## Migration sequence

1. Record CPU time, peak memory, retained cache, proof-byte, transcript, and
   diagnostic baselines.
2. Add the representation-free handle, message, scope, and diagnostics types.
3. Move concrete recursive implementation modules under
   `backend/cpu/recursive` without changing algorithms.
4. Wrap current CPU values in direct opaque handles; do not introduce arenas.
5. Cut accepted folds over and remove protocol access to
   `DecomposeFoldWitness`.
6. Merge recursive assembly and complete witness construction behind the
   two-phase build handle.
7. Make recursive commitment consume and return the witness handle.
8. Move relation preparation and Stage 1/2 state fully behind handles.
9. Move witness opening and EOR state fully behind handles.
10. Preserve and compare response-model diagnostic events field for field.
11. Remove raw exports and constructors after all call sites migrate.
12. Add the GPU consumer skeleton and then replace operations incrementally.
13. Run parity, safety, memory, and performance gates.

## Required validation

Correctness and compatibility:

- deterministic CPU proof bytes remain unchanged;
- transcript events remain unchanged;
- single- and multi-group proofs pass;
- single- and multi-chunk folds pass;
- terminal and recursive paths pass;
- evaluation-trace and coefficient-packing openings pass;
- quotient-lift and reduced-evaluation modes pass;
- direct and recursive setup-contribution modes pass.

Handle safety:

- cross-consumer and cross-scope handles are rejected;
- stale and consumed handles are rejected;
- multi-handle consumption validates atomically before mutation;
- prepared-setup, level, and operation-context mismatches are rejected;
- abort cleanup releases every resident object;
- failed consuming transitions enter an explicit failed state or roll back
  atomically.

Diagnostics:

- `response-model-diagnostics` retains exact source L2, observed response L2,
  conditional mean, accepted nonce, and attempt count;
- disabled builds perform no diagnostic source scan;
- future GPU reductions match CPU aggregate values exactly.

Performance:

- no new persistent witness-sized CPU copy;
- no global mutex or arena lookup in CPU round paths;
- packed digits remain packed;
- existing fused and prefix algorithms remain active;
- heterogeneous and tiered routing remain available;
- median CPU regression remains below the repository's existing three-percent
  threshold;
- peak memory and retained-cache usage do not increase;
- GPU transfer accounting confirms that private buffers are never downloaded;
- independent proofs may use independent GPU streams.

## Security boundary

Private Rust fields and owner-bound handles prevent ordinary protocol code from
depending on a consumer representation. They do not protect against malicious
unsafe code in the same process or crate. A stronger threat model uses the same
logical API through a separate crate, process, GPU runtime, enclave, or remote
service, with authenticated session-scoped capability identifiers where the
transport can be forged.

## Public session plans

`ValidatedStage1Plan` exposes its complete mathematical inputs through public
getters. `ValidatedRelationSessionPlan` carries `RelationWeightDescription` and
`Stage2OpeningDescription`, with public getters for all remaining inputs.
These descriptions contain public factors, evaluations, opening semantics,
layouts, and challenges. Each consumer compiles its own mutable Stage 2
representation; CPU compilation builds its relation-weight oracle and prepared
linear terms inside the CPU backend.

The reusable default commitment state is `PortableCommitmentHandle`, whose
row and compression accessors and codec are backend-only. Prover setup-prefix
slots and their persistence registry also live in the backend. No raw private
commitment representation is exported from `akita-types`.
