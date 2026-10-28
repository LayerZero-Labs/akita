use super::{
    AcceptedFoldHandle, FoldProbeOutcome, PreparedGroupOpening, PreparedRelationHandle,
    RecursiveWitnessBuildStart, RecursiveWitnessFoldInput, RecursiveWitnessHandle,
    WitnessCommitmentOutput,
};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};
use jolt_poly::UnivariatePoly;

/// Coherent opaque handles for a backend implementation.
pub trait ProverHandleFamily<F, E>
where
    F: Field + CanonicalEncoding,
    E: Field,
{
    /// Identity of the instance that owns this handle family.
    fn instance_identity(&self) -> super::BackendInstanceIdentity
    where
        Self: 'static;

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
    /// Extension seam: prepare a native group's public payload and opaque private state.
    /// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
    fn prepare_root_group(
        &self,
        _session: &Self::ProofSessionHandle,
        _context: &super::ProofContext,
        _handle: &Self::CommitmentHandle,
        _request: &dyn std::any::Any,
    ) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError> {
        Err(AkitaError::InvalidInput(
            "backend does not prepare native root groups".into(),
        ))
    }

    /// Extension seam: construct one admitted native root after the response challenge.
    /// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
    #[allow(clippy::too_many_arguments)]
    fn build_root_group(
        &self,
        _session: &Self::ProofSessionHandle,
        _context: &super::ProofContext,
        _preparation: &dyn std::any::Any,
        _challenges: &akita_challenges::Challenges,
        _logical_len: usize,
        _destination_ring_dimension: usize,
    ) -> Result<Option<Self::WitnessHandle>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "backend does not build native root groups".into(),
        ))
    }

    /// Extension seam: import a coefficient table into the admitted root scope.
    /// Native root producers attach through https://github.com/LayerZero-Labs/akita/issues/285.
    fn import_root_witness(
        &self,
        _session: &Self::ProofSessionHandle,
        _context: &super::ProofContext,
        _digits: Vec<i8>,
        _destination_ring_dimension: usize,
    ) -> Result<Self::WitnessHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "backend does not import root witnesses".into(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_recursive_witness(
        &self,
        session: &Self::ProofSessionHandle,
        context: &super::ProofContext,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_params::CommittedGroupParams,
        opening_batch: &akita_params::OpeningClaimsLayout,
        relation_rhs_layout: &akita_params::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError>;

    fn finish_recursive_witness(
        &self,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        relation: &akita_types::RingRelationInstance<F>,
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
    /// Extension seam: range-check an admitted coefficient table independently of its native ring.
    /// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
    fn begin_root_digit_range(
        &self,
        _witness: &Self::WitnessHandle,
        _domain: akita_params::FlatBooleanDomain,
        _equality_point: &[E],
        _range: akita_params::DigitRangePlan,
    ) -> Result<Self::Stage1SessionHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "backend does not range-check native root witnesses".into(),
        ))
    }

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
    /// Extension seam: combine direct public linear weights with the ordinary range-image relation.
    /// Tracking issue: https://github.com/LayerZero-Labs/akita/issues/285.
    /// Sparse negative-binary weights optionally add per-address `w(w+1)` predicates.
    #[allow(clippy::too_many_arguments)]
    fn begin_root_relation(
        &self,
        _witness: &Self::WitnessHandle,
        _weights: Vec<E>,
        _negative_binary_weights: Vec<(usize, E)>,
        _claim: E,
        _stage1_point: &[E],
        _range_image_evaluation: E,
        _batching: E,
    ) -> Result<Self::Stage2SessionHandle, AkitaError> {
        Err(AkitaError::InvalidInput(
            "backend does not compile native root relations".into(),
        ))
    }

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
    ) -> Result<UnivariatePoly<E>, AkitaError>;

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
