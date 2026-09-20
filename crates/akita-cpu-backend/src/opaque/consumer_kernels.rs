//! Private execution contracts used after consumer binding validation.
use crate::opaque::{
    ComputeBackendSetup, CpuWitnessBuildOutput, FoldHandleBackend, FoldProbeOutcome,
    PreparedRelationWitness, PreparedWitnessOpening, ProofScopeId, ProverHandleFamily,
    RecursiveWitnessAssemblyFinish, RecursiveWitnessAssemblyStart, RecursiveWitnessBuildStart,
    RecursiveWitnessFoldInput, RecursiveWitnessPublicInputs, RelationWitnessFinalClaims,
    Stage1FinalClaims, Stage1PublicTransition, Stage1RoundPolynomial, Stage1Step, Stage1Transition,
    ValidatedFoldProbePlan, ValidatedRelationSessionPlan, ValidatedRelationWitnessPlan,
    ValidatedStage1Plan, ValidatedTerminalFoldProbePlan, ValidatedTerminalZEncodingPlan,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Interactive public-artifact boundary for a consumer-owned relation witness.
pub(crate) trait RelationWitnessSession<E: Field>: Send {
    fn num_rounds(&self) -> usize;
    fn input_claim(&self) -> E;
    fn round_polynomial(
        &mut self,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError>;
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError>;
    fn finish(self) -> Result<RelationWitnessFinalClaims<E>, AkitaError>;
}

/// Backend-associated construction of the consumer-owned Stage 2 session.
pub(crate) trait RecursiveWitnessRelationKernel<H, F, E>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type Session: RelationWitnessSession<E>;

    fn begin_relation_session(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: H,
        plan: ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self::Session, AkitaError>;
}

/// Start consumer-owned Stage 1 without granting the consumer transcript access.
pub(crate) trait RecursiveWitnessStage1Kernel<H, F, E>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type Session: Send + 'static;

    fn begin_stage1(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedStage1Plan<E>,
    ) -> Result<Self::Session, AkitaError>;

    fn stage1_round_polynomial(
        &self,
        session: &mut Self::Session,
        step: Stage1Step,
        round: usize,
        previous_local_claim: E,
    ) -> Result<Stage1RoundPolynomial<E>, AkitaError>;

    fn bind_stage1_challenge(
        &self,
        session: &mut Self::Session,
        step: Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn stage1_public_transition(
        &self,
        session: &mut Self::Session,
        step: Stage1Step,
    ) -> Result<Stage1PublicTransition<E>, AkitaError>;

    fn bind_stage1_batch_challenge(
        &self,
        session: &mut Self::Session,
        transition: Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_stage1(&self, session: Self::Session) -> Result<Stage1FinalClaims<E>, AkitaError>;
}

/// Consumer-owned preparation of E, T, R, and compression witness state.
///
/// Akita transports `State` and `Witness` but can inspect only the public
/// artifacts returned alongside them.
pub(crate) trait RecursiveWitnessAssemblyKernel<F, E, OpeningH, FoldH, CommitH>:
    ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type State: Send + 'static;
    type Witness: Send + 'static;

    #[allow(clippy::too_many_arguments)]
    fn begin_recursive_witness_assembly(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        prepared_group_openings: &[OpeningH],
        commitment_material: Vec<CommitH>,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        relation_rhs_layout: &akita_types::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessAssemblyStart<F, E, Self::State>, AkitaError>;

    fn finish_recursive_witness_assembly(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        state: Self::State,
        folds: Vec<RecursiveWitnessFoldInput<FoldH>>,
    ) -> Result<RecursiveWitnessAssemblyFinish<F, Self::Witness>, AkitaError>;
}

/// Backend-associated preparation of the consumer-owned Stage 1/2 witness.
pub(crate) trait RecursiveRelationWitnessKernel<H, F>: ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
{
    type RelationWitness: Send + 'static;

    fn prepare_relation_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedRelationWitnessPlan,
    ) -> Result<PreparedRelationWitness<Self::RelationWitness>, AkitaError>;
}

pub(crate) trait CpuWitnessBuildKernel<F, E>:
    ProverHandleFamily<F, E> + ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    #[allow(clippy::too_many_arguments)]
    fn begin_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        scope_id: ProofScopeId,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        relation_rhs_layout: &akita_types::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError>;

    fn finish_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        public_inputs: RecursiveWitnessPublicInputs<'_, F>,
        plan: &crate::opaque::ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<CpuWitnessBuildOutput<F, Self::WitnessHandle>, AkitaError>;
}

pub(crate) trait CpuWitnessOpeningKernel<F, E>:
    ProverHandleFamily<F, E> + ComputeBackendSetup<F>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type WitnessOpeningHandle;
    type WitnessEorSessionHandle;
    fn prepare_witness_opening(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::opaque::ValidatedWitnessOpeningPlan<'_, E>,
    ) -> Result<PreparedWitnessOpening<E, Self::WitnessOpeningHandle>, AkitaError>;

    fn begin_witness_eor(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        opening_handle: Self::WitnessOpeningHandle,
        plan: &crate::opaque::ValidatedWitnessEorPlan<'_, E>,
    ) -> Result<Self::WitnessEorSessionHandle, AkitaError>;
}

/// Fold probing over an opaque recursive-witness handle.
pub(crate) trait RecursiveWitnessFoldKernel<H, F, const D: usize>:
    FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
{
    fn probe_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<<Self as FoldHandleBackend<F>>::AcceptedFold>, AkitaError>;
}

/// Terminal fold probing and canonical encoding over an opaque witness handle.
pub(crate) trait RecursiveWitnessTerminalFoldKernel<H, F, const D: usize>:
    FoldHandleBackend<F>
where
    F: Field + CanonicalEncoding,
{
    fn probe_terminal_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &H,
        plan: &ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<<Self as FoldHandleBackend<F>>::AcceptedTerminalFold>, AkitaError>;

    fn encode_terminal_recursive_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        fold: &<Self as FoldHandleBackend<F>>::AcceptedTerminalFold,
        plan: &ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError>;
}
