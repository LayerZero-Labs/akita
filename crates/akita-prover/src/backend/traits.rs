use super::{
    AcceptedFoldHandle, FoldProbeOutcome, PreparedGroupOpening, PreparedRelationHandle,
    RecursiveWitnessBuildStart, RecursiveWitnessFoldInput, RecursiveWitnessHandle,
    RecursiveWitnessPublicInputs, WitnessCommitmentOutput,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Coherent opaque handles for a backend implementation.
pub trait ProverHandleFamily<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    type CommitmentHandle: super::CommitmentHandleMetadata + Clone + Send + Sync + 'static;
    type EorPreparationHandle: Send;
    type EorSessionHandle: Send;
    type PreparedOpeningHandle: Send + 'static;
    type CommitmentMaterialHandle: crate::backend::CommitmentRelationMaterial<F> + Send + 'static;
    type AcceptedFoldHandle: AcceptedFoldHandle;
    type AcceptedTerminalFoldHandle: Send + 'static;
    type WitnessBuildHandle: Send + 'static;
    type WitnessHandle: RecursiveWitnessHandle;
    type RelationHandle: Send + 'static;
    type Stage1SessionHandle: Send + 'static;
    type Stage2SessionHandle: Send + 'static;
}

/// An identity-bearing consumer that operates on handles within a proof scope.
pub trait OpaqueProverConsumer<F, E>: ProverHandleFamily<F, E> + super::ProofScopeConsumer
where
    F: Field + CanonicalEncoding,
    E: Field,
{
}

/// Admission and canonical encoding of the terminal public response.
pub trait OpaqueTerminalFoldKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn probe_terminal_fold(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::backend::ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedTerminalFoldHandle>, AkitaError>;

    fn encode_terminal_fold(
        &self,
        terminal_fold_handle: Self::AcceptedTerminalFoldHandle,
        plan: &crate::backend::ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError>;
}

/// Two-phase construction of a complete private recursive witness.
pub trait OpaqueRecursiveWitnessBuildKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    #[allow(clippy::too_many_arguments)]
    fn begin_recursive_witness(
        &self,
        context: &super::ProofContext,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        relation_rhs_layout: &akita_types::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError>;

    fn finish_recursive_witness(
        &self,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        public_inputs: RecursiveWitnessPublicInputs<'_, F>,
        plan: &crate::backend::ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<Self::WitnessHandle, AkitaError>;
}

/// Commitment transition that consumes and replaces a witness handle.
pub trait OpaqueWitnessCommitKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn commit_witness(
        &self,
        witness_handle: Self::WitnessHandle,
        plan: &crate::backend::ValidatedRecursiveWitnessCommitPlan,
    ) -> Result<
        WitnessCommitmentOutput<F, Self::WitnessHandle, Self::CommitmentMaterialHandle>,
        AkitaError,
    >;

    /// Transfer a fully consumed current-level witness to the next fold level.
    fn advance_witness_level(
        &self,
        witness_handle: &mut Self::WitnessHandle,
    ) -> Result<(), AkitaError>;
}

/// Prepare private packed relation state from a recursive witness handle.
pub trait OpaqueRelationWitnessKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn prepare_relation_witness(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::backend::ValidatedRelationWitnessPlan,
    ) -> Result<PreparedRelationHandle<Self::RelationHandle>, AkitaError>;
}

/// Stage 1 interactive kernel over a private mutable session.
pub trait OpaqueStage1Kernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn begin_stage1(
        &self,
        relation_handle: &Self::RelationHandle,
        plan: &crate::backend::ValidatedStage1Plan<E>,
    ) -> Result<Self::Stage1SessionHandle, AkitaError>;

    fn stage1_round_polynomial(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: crate::backend::Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<crate::backend::Stage1RoundPolynomial<E>, AkitaError>;

    fn bind_stage1_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: crate::backend::Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn stage1_public_transition(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: crate::backend::Stage1Step,
    ) -> Result<crate::backend::Stage1PublicTransition<E>, AkitaError>;

    fn bind_stage1_batch_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        transition: crate::backend::Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_stage1(
        &self,
        session_handle: Self::Stage1SessionHandle,
    ) -> Result<crate::backend::Stage1FinalClaims<E>, AkitaError>;
}

/// Stage 2 interactive kernel over a private mutable session.
pub trait OpaqueStage2Kernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn begin_stage2(
        &self,
        relation_handle: Self::RelationHandle,
        plan: crate::backend::ValidatedRelationSessionPlan<'_, F, E>,
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
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError>;

    fn bind_stage2_challenge(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError>;

    fn finish_stage2(
        &self,
        session_handle: Self::Stage2SessionHandle,
    ) -> Result<crate::backend::RelationWitnessFinalClaims<E>, AkitaError>;
}

/// Witness opening and extension-opening-reduction lifecycle.
pub trait OpaqueWitnessOpeningKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    /// Prepare a native-ring opening without exposing the witness storage.
    fn prepare_native_witness_opening(
        &self,
        _witness_handle: &Self::WitnessHandle,
        _plan: &crate::backend::ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "consumer does not support native recursive-witness opening".into(),
        ))
    }

    /// Consume a retained native opening as terminal evaluation-trace rows.
    fn terminal_native_witness_opening(
        &self,
        _opening_handle: Self::PreparedOpeningHandle,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "consumer does not support terminal native witness opening".into(),
        ))
    }
}

/// Explicit early release for a long-lived recursive witness handle.
pub trait OpaqueResourceReleaseKernel<F, E>: OpaqueProverConsumer<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    fn release_witness_handle(&self, witness_handle: Self::WitnessHandle)
        -> Result<(), AkitaError>;
}
