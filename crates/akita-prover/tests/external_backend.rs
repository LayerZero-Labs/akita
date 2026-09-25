//! External implementation fixture: the real entrypoint must require no CPU types.

use akita_error::AkitaError;
use akita_prover::backend::*;
use akita_types::*;
use jolt_field::{CanonicalEncoding, Field};
use jolt_poly::UnivariatePoly;
use std::marker::PhantomData;

pub struct ExternalBackend<F, E>(PhantomData<(F, E)>);
#[derive(Clone)]
pub struct Handle;
pub struct ExternalProofSession;
impl CommitmentHandleMetadata for Handle {
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::try_new(1, 0).expect("fixture public shape")
    }
}
impl AcceptedFoldHandle for Handle {
    fn metadata(&self) -> AcceptedFoldMetadata {
        AcceptedFoldMetadata::try_new(1, 1, 1).expect("fixture public shape")
    }
}
impl RecursiveWitnessHandle for Handle {
    fn manifest(&self) -> RecursiveWitnessManifest {
        RecursiveWitnessManifest::try_new(1, 1, 1).expect("fixture public shape")
    }
}
impl<F: Field + CanonicalEncoding> CommitmentRelationMaterial<F> for Handle {
    fn metadata(&self) -> CommitmentMaterialMetadata {
        CommitmentMaterialMetadata::try_new(1, 1, false).expect("fixture public shape")
    }
}
impl<F: Field + CanonicalEncoding, E: Field> ProverHandleFamily<F, E> for ExternalBackend<F, E> {
    type CommitmentHandle = Handle;
    type EorPreparationHandle = ();
    type EorSessionHandle = ();
    type PreparedOpeningHandle = ();
    type CommitmentMaterialHandle = Handle;
    type AcceptedFoldHandle = Handle;
    type AcceptedTerminalFoldHandle = ();
    type WitnessBuildHandle = ();
    type WitnessHandle = Handle;
    type RelationHandle = ();
    type Stage1SessionHandle = ();
    type Stage2SessionHandle = ();
}
impl<F, E> ProofScopeConsumer for ExternalBackend<F, E> {
    type ProofSessionHandle = ExternalProofSession;
    fn finish_scope(&self, _session: &Self::ProofSessionHandle) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidProof)
    }
    fn abort_scope_best_effort(&self, _session: &Self::ProofSessionHandle) {}
}
impl<F: Field + CanonicalEncoding, E: Field> OpaqueProverConsumer<F, E> for ExternalBackend<F, E> {}
impl<F: Field + CanonicalEncoding, E: Field> TerminalCommitmentMaterialKernel<F, Handle>
    for ExternalBackend<F, E>
{
    fn terminal_message(
        &self,
        _material: &Handle,
    ) -> Result<TerminalTFieldsMessage<F>, AkitaError> {
        Err(AkitaError::InvalidProof)
    }
    fn consume_terminal_row(&self, _material: Handle) -> Result<RingVec<F>, AkitaError> {
        Err(AkitaError::InvalidProof)
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> ProofAdmission<F, E> for ExternalBackend<F, E> {
    fn begin_proof(
        &self,
        setup: &AkitaSetupDescriptor,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<Self::ProofSessionHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn proof_context(
        &self,
        session: &Self::ProofSessionHandle,
        level: u32,
    ) -> Result<ProofContext, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn validate_commitment(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        handle: &Self::CommitmentHandle,
        parameters: &GroupCommitPhaseParams,
        commitment: &Commitment<F>,
    ) -> Result<Self::CommitmentMaterialHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueOpeningKernel<F, E> for ExternalBackend<F, E> {
    fn prepare_opening(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        source: OpeningSource<'_, Self::CommitmentHandle, Self::WitnessHandle>,
        plan: &akita_prover::backend::ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn probe_opening_fold(
        &self,
        context: &ProofContext,
        opening: &Self::PreparedOpeningHandle,
        plan: &akita_prover::backend::ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFoldHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueEorKernel<F, E> for ExternalBackend<F, E> {
    fn prepare_eor(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        layout: &OpeningClaimsLayout,
        groups: &[EorGroupRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<PreparedEor<E, Self::EorPreparationHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn begin_eor(
        &self,
        preparation: Self::EorPreparationHandle,
        eta: &[E],
        coefficients: &[E],
    ) -> Result<(E, Self::EorSessionHandle), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn bind_eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn finish_eor(&self, session: Self::EorSessionHandle) -> Result<Vec<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueRecursiveWitnessBuildKernel<F, E>
    for ExternalBackend<F, E>
{
    fn begin_recursive_witness(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        relation_rhs_layout: &akita_types::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn finish_recursive_witness(
        &self,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        relation: &akita_types::RingRelationInstance<F>,
        plan: &akita_prover::backend::ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<Self::WitnessHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueWitnessCommitKernel<F, E>
    for ExternalBackend<F, E>
{
    fn commit_witness(
        &self,
        witness_handle: Self::WitnessHandle,
        plan: &akita_prover::backend::ValidatedRecursiveWitnessCommitPlan,
    ) -> Result<
        WitnessCommitmentOutput<F, Self::WitnessHandle, Self::CommitmentMaterialHandle>,
        AkitaError,
    > {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn advance_witness_level(
        &self,
        witness_handle: &mut Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueRelationWitnessKernel<F, E>
    for ExternalBackend<F, E>
{
    fn prepare_relation_witness(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &akita_prover::backend::ValidatedRelationWitnessPlan,
    ) -> Result<PreparedRelationHandle<Self::RelationHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueStage1Kernel<F, E> for ExternalBackend<F, E> {
    fn begin_stage1(
        &self,
        relation_handle: &Self::RelationHandle,
        plan: &akita_prover::backend::ValidatedStage1Plan<E>,
    ) -> Result<Self::Stage1SessionHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn stage1_round_polynomial(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: akita_prover::backend::Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_prover::backend::Stage1RoundPolynomial<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn bind_stage1_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: akita_prover::backend::Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn stage1_public_transition(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: akita_prover::backend::Stage1Step,
    ) -> Result<akita_prover::backend::Stage1PublicTransition<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn bind_stage1_batch_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        transition: akita_prover::backend::Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn finish_stage1(
        &self,
        session_handle: Self::Stage1SessionHandle,
    ) -> Result<akita_prover::backend::Stage1FinalClaims<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueStage2Kernel<F, E> for ExternalBackend<F, E> {
    fn begin_stage2(
        &self,
        relation_handle: Self::RelationHandle,
        plan: akita_prover::backend::ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self::Stage2SessionHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn stage2_input_claim(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<E, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn stage2_num_rounds(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<usize, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn stage2_round_polynomial(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        previous_claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn bind_stage2_challenge(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn finish_stage2(
        &self,
        session_handle: Self::Stage2SessionHandle,
    ) -> Result<akita_prover::backend::RelationWitnessFinalClaims<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueTerminalFoldKernel<F, E>
    for ExternalBackend<F, E>
{
    fn probe_terminal_fold(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &akita_prover::backend::ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedTerminalFoldHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn encode_terminal_fold(
        &self,
        terminal_fold_handle: Self::AcceptedTerminalFoldHandle,
        plan: &akita_prover::backend::ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueWitnessOpeningKernel<F, E>
    for ExternalBackend<F, E>
{
    fn prepare_native_witness_opening(
        &self,
        _witness_handle: &Self::WitnessHandle,
        _plan: &akita_prover::backend::ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn terminal_native_witness_opening(
        &self,
        _opening_handle: Self::PreparedOpeningHandle,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueResourceReleaseKernel<F, E>
    for ExternalBackend<F, E>
{
    fn release_witness_handle(
        &self,
        witness_handle: Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

#[allow(unused_variables)]
impl<F: Field + CanonicalEncoding, E: Field> OpaqueStage3Kernel<F, E> for ExternalBackend<F, E> {
    type Stage3SessionHandle = ();
    fn begin_stage3(
        &self,
        request: Stage3Request<'_, F, E, Self::ProofSessionHandle>,
    ) -> Result<(E, Self::Stage3SessionHandle), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn stage3_round_polynomial(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn bind_stage3_challenge(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
    fn finish_stage3(&self, session: Self::Stage3SessionHandle) -> Result<E, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external fixture rejects this operation".into(),
        ))
    }
}

/// This body instantiates the actual prover for a backend in an external crate.
/// Admission rejects unsupported plans without requiring CPU preparation or sources.
#[allow(clippy::too_many_arguments)]
pub fn prove_with_external_backend<'a, Cfg>(
    expanded: &AkitaSetupDescriptor,
    prefixes: &akita_prover::SetupPrefixProverRegistry<Cfg::Field, Handle>,
    schedules: &akita_config::TrustedScheduleCatalog<Cfg>,
    opening: akita_prover::SelectedProverOpeningData<'a, Cfg::ExtField, Handle, Cfg::Field>,
    transcript_session: &[u8],
) -> Result<Vec<u8>, AkitaError>
where
    Cfg: akita_config::CommitmentConfig,
    Cfg::Field: CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::Unreduced
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + 'static,
    <Cfg::Field as jolt_field::Unreduced>::Wide: From<Cfg::Field> + jolt_field::AdditiveGroup,
    Cfg::ExtField: FpExtEncoding<Cfg::Field>
        + jolt_field::ExtField<Cfg::Field>
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::Ring
        + jolt_field::MulBaseUnreduced<Cfg::Field>
        + akita_serialization::AkitaSerialize
        + 'static,
{
    let backend = ExternalBackend::<Cfg::Field, Cfg::ExtField>(PhantomData);
    akita_prover::batched_prove::<Cfg, _>(
        expanded,
        prefixes,
        schedules,
        &backend,
        opening,
        transcript_session,
        BasisMode::Lagrange,
    )
}

#[test]
fn opaque_messages_retain_external_handles_without_storage_requirements() {
    type F = jolt_field::Prime128OffsetA7F7;
    let opening = PreparedGroupOpening::new(vec![F::default()], Box::new(73u32));
    let (messages, handle) = opening.into_parts();
    assert_eq!(messages.len(), 1);
    assert_eq!(*handle, 73);
    assert_eq!(RecursiveWitnessHandle::manifest(&Handle).logical_len(), 1);
    assert_eq!(
        <Handle as CommitmentRelationMaterial<F>>::metadata(&Handle).source_count(),
        1
    );
}

#[test]
fn scope_guard_finishes_once_and_aborts_after_failure_or_early_return() {
    use std::cell::Cell;
    struct Lifecycle {
        finished: Cell<usize>,
        aborted: Cell<usize>,
        fail: Cell<bool>,
    }
    impl ProofScopeConsumer for Lifecycle {
        type ProofSessionHandle = ExternalProofSession;
        fn finish_scope(&self, _: &Self::ProofSessionHandle) -> Result<(), AkitaError> {
            if self.fail.get() {
                return Err(AkitaError::InvalidProof);
            }
            self.finished.set(self.finished.get() + 1);
            Ok(())
        }
        fn abort_scope_best_effort(&self, _: &Self::ProofSessionHandle) {
            self.aborted.set(self.aborted.get() + 1);
        }
    }
    let backend = Lifecycle {
        finished: Cell::new(0),
        aborted: Cell::new(0),
        fail: Cell::new(false),
    };
    ProofScope::admitted(&backend, ExternalProofSession)
        .finish()
        .unwrap();
    assert_eq!((backend.finished.get(), backend.aborted.get()), (1, 0));
    drop(ProofScope::admitted(&backend, ExternalProofSession));
    assert_eq!(backend.aborted.get(), 1);
    backend.fail.set(true);
    assert!(ProofScope::admitted(&backend, ExternalProofSession)
        .finish()
        .is_err());
    assert_eq!((backend.finished.get(), backend.aborted.get()), (1, 2));
}
