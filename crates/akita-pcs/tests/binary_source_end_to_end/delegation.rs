//! Test-only forwarding of ordinary backend capabilities to the real CPU backend.
#![allow(clippy::too_many_arguments)]
use super::*;

impl ProofScopeConsumer for FaultBackend {
    type ProofSessionHandle = <NativeBackend as ProofScopeConsumer>::ProofSessionHandle;
    fn finish_scope(&self, session: &Self::ProofSessionHandle) -> Result<(), AkitaError> {
        self.inner.finish_scope(session)
    }
    fn abort_scope_best_effort(&self, session: &Self::ProofSessionHandle) {
        self.inner.abort_scope_best_effort(session)
    }
}

impl ProverHandleFamily<F, E> for FaultBackend {
    type CommitmentHandle = <NativeBackend as ProverHandleFamily<F, E>>::CommitmentHandle;
    type EorPreparationHandle = <NativeBackend as ProverHandleFamily<F, E>>::EorPreparationHandle;
    type EorSessionHandle = <NativeBackend as ProverHandleFamily<F, E>>::EorSessionHandle;
    type PreparedOpeningHandle = <NativeBackend as ProverHandleFamily<F, E>>::PreparedOpeningHandle;
    type CommitmentMaterialHandle =
        <NativeBackend as ProverHandleFamily<F, E>>::CommitmentMaterialHandle;
    type AcceptedFoldHandle = <NativeBackend as ProverHandleFamily<F, E>>::AcceptedFoldHandle;
    type AcceptedTerminalFoldHandle =
        <NativeBackend as ProverHandleFamily<F, E>>::AcceptedTerminalFoldHandle;
    type WitnessBuildHandle = <NativeBackend as ProverHandleFamily<F, E>>::WitnessBuildHandle;
    type WitnessHandle = <NativeBackend as ProverHandleFamily<F, E>>::WitnessHandle;
    type RelationHandle = <NativeBackend as ProverHandleFamily<F, E>>::RelationHandle;
    type Stage1SessionHandle = <NativeBackend as ProverHandleFamily<F, E>>::Stage1SessionHandle;
    type Stage2SessionHandle = <NativeBackend as ProverHandleFamily<F, E>>::Stage2SessionHandle;
    fn instance_identity(&self) -> akita_prover::backend::BackendInstanceIdentity
    where
        Self: 'static,
    {
        self.inner.instance_identity()
    }
}

impl OpaqueProverConsumer<F, E> for FaultBackend {}

impl ProofAdmission<F, E> for FaultBackend {
    fn prepare_executor<Cfg>(
        &self,
        setup: &AkitaSetupDescriptor,
        schedules: &TrustedScheduleCatalog<Cfg>,
        plan: &FoldSchedule,
        layout: &OpeningClaimsLayout,
    ) -> Result<Self::ProofSessionHandle, AkitaError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        self.inner
            .prepare_executor::<Cfg>(setup, schedules, plan, layout)
    }
    fn begin_fold(
        &self,
        session: &Self::ProofSessionHandle,
        requirements: &akita_prover::backend::FoldExecutionRequirements<'_>,
    ) -> Result<(), AkitaError> {
        self.inner.begin_fold(session, requirements)
    }
    fn proof_context(
        &self,
        session: &Self::ProofSessionHandle,
        level: u32,
    ) -> Result<ProofContext, AkitaError> {
        self.inner.proof_context(session, level)
    }
    fn validate_commitment(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        handle: &Self::CommitmentHandle,
        parameters: &GroupCommitPhaseParams,
        commitment: &Commitment<F>,
    ) -> Result<Self::CommitmentMaterialHandle, AkitaError> {
        self.inner
            .validate_commitment(session, context, handle, parameters, commitment)
    }
}

impl OpaqueOpeningKernel<F, E> for FaultBackend {
    fn prepare_openings(
        &self,
        session: &Self::ProofSessionHandle,
        requests: &[GroupOpeningRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<Vec<PreparedGroupOpening<E, Self::PreparedOpeningHandle>>, AkitaError> {
        self.inner.prepare_openings(session, requests)
    }
    fn probe_opening_fold(
        &self,
        context: &ProofContext,
        opening: &Self::PreparedOpeningHandle,
        plan: &akita_prover::backend::ValidatedFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedFoldHandle>, AkitaError> {
        self.inner.probe_opening_fold(context, opening, plan)
    }
}

impl OpaqueEorKernel<F, E> for FaultBackend {
    fn prepare_eor(
        &self,
        session: &Self::ProofSessionHandle,
        context: &ProofContext,
        layout: &OpeningClaimsLayout,
        groups: &[EorGroupRequest<'_, E, Self::CommitmentHandle, Self::WitnessHandle>],
    ) -> Result<PreparedEor<E, Self::EorPreparationHandle>, AkitaError> {
        self.inner.prepare_eor(session, context, layout, groups)
    }
    fn begin_eor(
        &self,
        preparation: Self::EorPreparationHandle,
        eta: &[E],
        coefficients: &[E],
    ) -> Result<(E, Self::EorSessionHandle), AkitaError> {
        self.inner.begin_eor(preparation, eta, coefficients)
    }
    fn eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.inner.eor_round(session, round, claim)
    }
    fn bind_eor_round(
        &self,
        session: &mut Self::EorSessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner.bind_eor_round(session, round, challenge)
    }
    fn finish_eor(&self, session: Self::EorSessionHandle) -> Result<Vec<E>, AkitaError> {
        self.inner.finish_eor(session)
    }
}

impl OpaqueRecursiveWitnessBuildKernel<F, E> for FaultBackend {
    fn prepare_root_group(
        &self,
        _session: &Self::ProofSessionHandle,
        _context: &akita_prover::backend::ProofContext,
        _handle: &Self::CommitmentHandle,
        _request: &dyn std::any::Any,
    ) -> Result<(Vec<F>, Box<dyn std::any::Any + Send + Sync>), AkitaError> {
        self.prepare_fault_root_group(_session, _context, _handle, _request)
    }
    fn build_root_group(
        &self,
        _session: &Self::ProofSessionHandle,
        _context: &akita_prover::backend::ProofContext,
        _preparation: &dyn std::any::Any,
        _challenges: &akita_challenges::Challenges,
        _logical_len: usize,
        _destination_ring_dimension: usize,
    ) -> Result<Option<Self::WitnessHandle>, AkitaError> {
        self.build_fault_root_group(
            _session,
            _context,
            _preparation,
            _challenges,
            _logical_len,
            _destination_ring_dimension,
        )
    }
    fn import_root_witness(
        &self,
        _session: &Self::ProofSessionHandle,
        _context: &akita_prover::backend::ProofContext,
        _digits: Vec<i8>,
        _destination_ring_dimension: usize,
    ) -> Result<Self::WitnessHandle, AkitaError> {
        self.inner
            .import_root_witness(_session, _context, _digits, _destination_ring_dimension)
    }
    fn begin_recursive_witness(
        &self,
        session: &Self::ProofSessionHandle,
        context: &akita_prover::backend::ProofContext,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_params::CommittedGroupParams,
        opening_batch: &akita_params::OpeningClaimsLayout,
        relation_rhs_layout: &akita_params::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError> {
        self.inner.begin_recursive_witness(
            session,
            context,
            prepared_opening_handles,
            commitment_material_handles,
            level,
            opening_batch,
            relation_rhs_layout,
            group_commitments,
        )
    }
    fn finish_recursive_witness(
        &self,
        build_handle: Self::WitnessBuildHandle,
        fold_inputs: Vec<RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        relation: &akita_types::RingRelationInstance<F>,
        plan: &akita_prover::backend::ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<Self::WitnessHandle, AkitaError> {
        self.inner
            .finish_recursive_witness(build_handle, fold_inputs, relation, plan)
    }
}

impl OpaqueWitnessCommitKernel<F, E> for FaultBackend {
    fn commit_witness(
        &self,
        witness_handle: Self::WitnessHandle,
        plan: &akita_prover::backend::ValidatedRecursiveWitnessCommitPlan,
    ) -> Result<
        WitnessCommitmentOutput<F, Self::WitnessHandle, Self::CommitmentMaterialHandle>,
        AkitaError,
    > {
        self.inner.commit_witness(witness_handle, plan)
    }
    fn advance_witness_level(
        &self,
        witness_handle: &mut Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        self.inner.advance_witness_level(witness_handle)
    }
}

impl OpaqueRelationWitnessKernel<F, E> for FaultBackend {
    fn prepare_relation_witness(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &akita_prover::backend::ValidatedRelationWitnessPlan,
    ) -> Result<PreparedRelationHandle<Self::RelationHandle>, AkitaError> {
        self.inner.prepare_relation_witness(witness_handle, plan)
    }
}

impl OpaqueStage1Kernel<F, E> for FaultBackend {
    fn begin_root_digit_range(
        &self,
        _witness: &Self::WitnessHandle,
        _domain: akita_params::FlatBooleanDomain,
        _equality_point: &[E],
        _range: akita_params::DigitRangePlan,
    ) -> Result<Self::Stage1SessionHandle, AkitaError> {
        self.inner
            .begin_root_digit_range(_witness, _domain, _equality_point, _range)
    }
    fn begin_stage1(
        &self,
        relation_handle: &Self::RelationHandle,
        plan: &akita_prover::backend::ValidatedStage1Plan<E>,
    ) -> Result<Self::Stage1SessionHandle, AkitaError> {
        self.inner.begin_stage1(relation_handle, plan)
    }
    fn stage1_round_polynomial(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: akita_prover::backend::Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_prover::backend::Stage1RoundPolynomial<E>, AkitaError> {
        self.inner
            .stage1_round_polynomial(session_handle, step, round, previous_claim)
    }
    fn bind_stage1_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: akita_prover::backend::Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner
            .bind_stage1_challenge(session_handle, step, round, challenge)
    }
    fn stage1_public_transition(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: akita_prover::backend::Stage1Step,
    ) -> Result<akita_prover::backend::Stage1PublicTransition<E>, AkitaError> {
        self.inner.stage1_public_transition(session_handle, step)
    }
    fn bind_stage1_batch_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        transition: akita_prover::backend::Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner
            .bind_stage1_batch_challenge(session_handle, transition, challenge)
    }
    fn finish_stage1(
        &self,
        session_handle: Self::Stage1SessionHandle,
    ) -> Result<akita_prover::backend::Stage1FinalClaims<E>, AkitaError> {
        self.inner.finish_stage1(session_handle)
    }
}

impl OpaqueStage2Kernel<F, E> for FaultBackend {
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
        self.begin_fault_root_relation(
            _witness,
            _weights,
            _negative_binary_weights,
            _claim,
            _stage1_point,
            _range_image_evaluation,
            _batching,
        )
    }
    fn begin_stage2(
        &self,
        relation_handle: Self::RelationHandle,
        plan: akita_prover::backend::ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self::Stage2SessionHandle, AkitaError> {
        self.inner.begin_stage2(relation_handle, plan)
    }
    fn stage2_input_claim(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<E, AkitaError> {
        self.inner.stage2_input_claim(session_handle)
    }
    fn stage2_num_rounds(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<usize, AkitaError> {
        self.inner.stage2_num_rounds(session_handle)
    }
    fn stage2_round_polynomial(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        previous_claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.inner
            .stage2_round_polynomial(session_handle, round, previous_claim)
    }
    fn bind_stage2_challenge(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner
            .bind_stage2_challenge(session_handle, round, challenge)
    }
    fn finish_stage2(
        &self,
        session_handle: Self::Stage2SessionHandle,
    ) -> Result<akita_prover::backend::RelationWitnessFinalClaims<E>, AkitaError> {
        self.inner.finish_stage2(session_handle)
    }
}

impl OpaqueStage3Kernel<F, E> for FaultBackend {
    type Stage3SessionHandle = <NativeBackend as OpaqueStage3Kernel<F, E>>::Stage3SessionHandle;
    fn begin_stage3(
        &self,
        request: Stage3Request<'_, F, E, Self::ProofSessionHandle>,
    ) -> Result<(E, Self::Stage3SessionHandle), AkitaError> {
        self.inner.begin_stage3(request)
    }
    fn stage3_round_polynomial(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        claim: E,
    ) -> Result<UnivariatePoly<E>, AkitaError> {
        self.inner.stage3_round_polynomial(session, round, claim)
    }
    fn bind_stage3_challenge(
        &self,
        session: &mut Self::Stage3SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.inner.bind_stage3_challenge(session, round, challenge)
    }
    fn finish_stage3(&self, session: Self::Stage3SessionHandle) -> Result<E, AkitaError> {
        self.inner.finish_stage3(session)
    }
}

impl OpaqueTerminalFoldKernel<F, E> for FaultBackend {
    fn probe_terminal_fold(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &akita_prover::backend::ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<FoldProbeOutcome<Self::AcceptedTerminalFoldHandle>, AkitaError> {
        self.inner.probe_terminal_fold(witness_handle, plan)
    }
    fn encode_terminal_fold(
        &self,
        terminal_fold_handle: Self::AcceptedTerminalFoldHandle,
        plan: &akita_prover::backend::ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError> {
        self.inner.encode_terminal_fold(terminal_fold_handle, plan)
    }
}

impl OpaqueWitnessOpeningKernel<F, E> for FaultBackend {
    fn prepare_native_witness_opening(
        &self,
        _witness_handle: &Self::WitnessHandle,
        _plan: &akita_prover::backend::ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError> {
        self.inner
            .prepare_native_witness_opening(_witness_handle, _plan)
    }
    fn terminal_native_witness_opening(
        &self,
        _opening_handle: Self::PreparedOpeningHandle,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
        self.inner.terminal_native_witness_opening(_opening_handle)
    }
}

impl OpaqueResourceReleaseKernel<F, E> for FaultBackend {
    fn release_witness_handle(
        &self,
        witness_handle: Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        self.inner.release_witness_handle(witness_handle)
    }
}

impl SuccessorExportKernel<F, E> for FaultBackend {
    type ExportPacket = <NativeBackend as SuccessorExportKernel<F, E>>::ExportPacket;
    fn export_successor(
        &self,
        session: &Self::ProofSessionHandle,
        witness: &Self::WitnessHandle,
        material: &Self::CommitmentMaterialHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
    ) -> Result<Self::ExportPacket, AkitaError> {
        self.inner
            .export_successor(session, witness, material, plan)
    }
}

impl SuccessorImportKernel<F, E> for FaultBackend {
    type ImportPacket = <NativeBackend as SuccessorImportKernel<F, E>>::ImportPacket;
    fn import_successor(
        &self,
        session: &Self::ProofSessionHandle,
        plan: &ValidatedSuccessorHandoffPlan<'_, F>,
        packet: Self::ImportPacket,
    ) -> Result<ImportedSuccessor<Self::WitnessHandle, Self::CommitmentMaterialHandle>, AkitaError>
    {
        self.inner.import_successor(session, plan, packet)
    }
}

impl TerminalCommitmentMaterialKernel<F, M> for FaultBackend {
    fn terminal_message(
        &self,
        material: &M,
    ) -> Result<akita_prover::backend::TerminalTFieldsMessage<F>, AkitaError> {
        self.inner.terminal_message(material)
    }
    fn consume_terminal_row(&self, material: M) -> Result<akita_types::RingVec<F>, AkitaError> {
        self.inner.consume_terminal_row(material)
    }
}
