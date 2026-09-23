//! Direct CPU implementation of the opaque protocol kernels.
mod admission;
mod backend;
pub(crate) mod capabilities;
pub(crate) mod compression;
pub(crate) mod consumer_kernels;
mod contracts;
mod decompose_fold;
pub(crate) mod eor;
mod eor_plans;
mod fold_kernels;
mod handles;
pub(crate) mod lifecycle;
mod openings;
pub(crate) mod operation_plans;
mod owned;
mod owned_commit;
mod owned_prefix;
pub(crate) mod plans;
mod prepared_opening;
pub(crate) mod recursive;
pub(crate) mod relation_weights;
mod source;
#[doc(hidden)]
pub mod standalone;
pub(crate) mod sumcheck;

#[cfg(test)]
pub(crate) use crate::arithmetic::CyclicRowsComputeBackend;
pub(crate) use crate::arithmetic::{tensor_pack_recursive_witness, RingSwitchRelationView};
pub(crate) use crate::arithmetic::{
    CompressionComputeBackend, CompressionRowsProducts, ComputeBackendSetup,
    CpuCompressionOperation, CpuInnerCommitOperation, CpuOuterCommitOperation, CpuPreparedSetup,
    DigitRowsComputeBackend, NttCacheOwnerId, NttExecutionRequirements, NttOperationCluster,
    OperationCtx, RoutedNttRequirement,
};
pub use crate::arithmetic::{PreparedCrtNttProfile, PreparedNttCacheMetric};
pub(crate) use crate::sources::commit_onehot_sources;
pub(crate) use crate::sources::poly::centered_reach_of_field_coeffs;
pub use crate::sources::poly::{
    evaluate_root_polynomial, RootPolyMeta, RootPolyShape, RootPolynomialEvaluator,
};
pub(crate) use crate::sources::poly::{
    OpeningProveBackendFor, RingSwitchProveBackend, RootOpeningSource,
};
pub(crate) use crate::sources::{DensePoly, OneHotIndex, OneHotPoly, OneHotSource};
use akita_error::AkitaError;
pub(crate) use akita_prover::backend::*;
pub use backend::CpuBackend;
pub(crate) use capabilities::{
    RuntimeCoefficientPackingBackendFor, RuntimeFoldRelationBackend, RuntimeOpeningProveBackendFor,
    RuntimeRingSwitchProveBackend, RuntimeRootProvePoly,
};
pub use contracts::SubringCoefficientPackingBatchKernel;
pub(crate) use contracts::{
    CpuWitnessBuildOutput, FoldHandleBackend, FoldResponseKernel, PreparedRelationWitness,
    RingSwitchRelationKernel, TerminalFoldResponseKernel,
};
pub use decompose_fold::DecomposeFoldWitness;
pub(crate) use eor_plans::{
    PreparedWitnessOpening, ValidatedWitnessEorPlan, ValidatedWitnessOpeningPlan,
};
pub(crate) use fold_kernels::{FoldRelationKernel, FoldRelationOutput, RelationQuotientRow};
pub use fold_kernels::{OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput};
pub(crate) use handles::{CpuWitnessBuildHandle, OperationBinding};
use jolt_field::{CanonicalEncoding, Field};
pub(crate) use lifecycle::{BackendIdentity, ScopeLease};
pub use operation_plans::{
    CommitInnerPlan, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldPlan,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
pub(crate) use operation_plans::{RingSwitchRelationPlan, ValidatedFoldRelationPlan};
pub use owned::{CommitOutput, CommitmentHandle, CpuSource, SourceHandle};
pub(crate) use prepared_opening::PreparedGroupOpeningKernel;
pub(crate) use recursive::OpaqueRecursiveWitness;
pub(crate) use recursive::{build_w_evals_compact, PreparedProverLinearTerms};
pub(crate) use recursive::{
    cpu_extension_opening_session_from_witnesses, prepare_recursive_witness_opening,
};
#[cfg(test)]
pub(crate) use recursive::{
    pad_compact_witness, LowBasisRangeCheckProver, StructuredLinearSegment, StructuredLinearTerm,
    StructuredLinearWeights,
};
pub(crate) use recursive::{
    CpuPreparedOpeningHandle, CpuRelationHandle, CpuStage1SessionHandle, CpuStage2SessionHandle,
    CpuWitnessHandle, CpuWitnessOpeningHandle, RecursiveWitnessFlat,
};
pub(crate) use relation_weights::RelationWeightDescription;
impl<F, E> crate::opaque::ProverHandleFamily<F, E> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding + Send + Sync + 'static,
    E: Field + Send + Sync + 'static,
{
    type CommitmentHandle = CommitmentHandle<F, E>;
    type EorPreparationHandle = eor::CpuEorPreparation<F, E>;
    type EorSessionHandle = eor::CpuEorSession<E>;
    type PreparedOpeningHandle = crate::opaque::CpuPreparedOpeningHandle<F, E>;
    type CommitmentMaterialHandle = CpuCommitmentMaterialHandle<F>;
    type AcceptedFoldHandle = CpuAcceptedFoldHandle<F>;
    type AcceptedTerminalFoldHandle = CpuAcceptedTerminalFoldHandle<F>;
    type WitnessBuildHandle = CpuWitnessBuildHandle<F, E>;
    type WitnessHandle = crate::opaque::CpuWitnessHandle;
    type RelationHandle = crate::opaque::CpuRelationHandle;
    type Stage1SessionHandle = crate::opaque::CpuStage1SessionHandle<E>;
    type Stage2SessionHandle = crate::opaque::CpuStage2SessionHandle<E>;
}
impl<F, E> crate::opaque::OpaqueProverConsumer<F, E> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding + Send + Sync + 'static,
    E: Field + Send + Sync + 'static,
{
}
impl<F, E> crate::opaque::OpaqueResourceReleaseKernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn release_witness_handle(
        &self,
        witness_handle: Self::WitnessHandle,
    ) -> Result<(), AkitaError> {
        self.validate_binding(&witness_handle.operation_binding())?;
        drop(witness_handle);
        Ok(())
    }
}

impl<F, E> crate::opaque::OpaqueRecursiveWitnessBuildKernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn begin_recursive_witness(
        &self,
        context: &crate::opaque::ProofContext,
        prepared_opening_handles: &[Self::PreparedOpeningHandle],
        commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        relation_rhs_layout: &akita_types::RelationRhsLayout,
        group_commitments: &[akita_types::RingVec<F>],
    ) -> Result<crate::opaque::RecursiveWitnessBuildStart<F, E, Self::WitnessBuildHandle>, AkitaError>
    {
        let binding = self.binding(context)?;
        let (schedule, root_layout) = self.owner().proof_plan(context.scope_id())?;
        let expected = if context.fold_level() == 0 {
            &schedule.root.params
        } else {
            &schedule
                .recursive_folds
                .get(context.fold_level() as usize - 1)
                .ok_or(AkitaError::InvalidProof)?
                .params
        };
        if expected != level || (context.fold_level() == 0 && &root_layout != opening_batch) {
            return Err(AkitaError::InvalidInput(
                "witness assembly does not match the admitted fold".into(),
            ));
        }
        level.validate_opening_batch(opening_batch)?;
        if prepared_opening_handles.len() != opening_batch.num_groups()
            || commitment_material_handles.len() != opening_batch.num_groups()
            || group_commitments.len() != opening_batch.num_groups()
        {
            return Err(AkitaError::InvalidInput(
                "witness build group count mismatch".into(),
            ));
        }
        for (group_index, (opening, material)) in prepared_opening_handles
            .iter()
            .zip(&commitment_material_handles)
            .enumerate()
        {
            let opening_binding = opening.operation_binding();
            let material_binding = material.binding();
            material.validate_public_commitment(
                group_commitments
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            let committed_id = match opening.source::<openings::RetainedOpeningSource<F, E>>()? {
                openings::RetainedOpeningSource::Commitment(committed) => committed.commitment_id,
                openings::RetainedOpeningSource::Witness(witness) => {
                    witness.operation_binding().operation_id()
                }
            };
            if material.commitment_id() != Some(committed_id) {
                return Err(AkitaError::InvalidInput(
                    "opening and commitment material do not refer to the same committed source"
                        .into(),
                ));
            }
            self.validate_binding(&opening_binding)?;
            self.validate_binding(&material_binding)?;
            binding.validate_lineage(&opening_binding)?;
            binding.validate_lineage(&material_binding)?;
            opening_binding.validate_group(group_index, opening_batch.num_groups())?;
            material_binding.validate_group(group_index, opening_batch.num_groups())?;
        }
        let opening_bindings = prepared_opening_handles
            .iter()
            .map(|opening| opening.operation_binding())
            .collect();
        crate::opaque::witness_build::begin_cpu_recursive_witness(
            self,
            self.prepared::<F>()?,
            binding,
            opening_bindings,
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
        fold_inputs: Vec<crate::opaque::RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>>,
        public_inputs: crate::opaque::RecursiveWitnessPublicInputs<'_, F>,
        plan: &crate::opaque::ValidatedRecursiveWitnessPlan<'_, F>,
    ) -> Result<Self::WitnessHandle, AkitaError> {
        self.validate_binding(&build_handle.binding)?;
        let parent_binding = build_handle.binding;
        let level = build_handle.level.clone();
        let layout = build_handle.opening_batch.clone();
        for (group_index, input) in fold_inputs.iter().enumerate() {
            input
                .fold_handle()
                .validate_challenges(input.challenges().ambient_a())?;
            let binding = input.fold_handle().binding();
            self.validate_binding(&binding)?;
            binding.validate_lineage(&parent_binding)?;
            binding.validate_computation(
                build_handle
                    .opening_bindings
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?,
            )?;
            binding.validate_group(group_index, build_handle.opening_batch.num_groups())?;
        }
        let output = crate::opaque::witness_build::finish_cpu_recursive_witness(
            self,
            self.prepared::<F>()?,
            build_handle,
            fold_inputs,
            public_inputs,
            plan,
        )?;
        let (instance, mut witness_handle) = output.into_instance_and_witness_handle();
        let witness_layout = instance.segment_layout(&level, None)?;
        let geometry = level.relation_address_geometry(
            &layout,
            E::DEGREE,
            plan.commitment_ring_dimension(),
            witness_layout.live_coeff_len(),
        )?;
        witness_handle.relation_plan = Some(std::sync::Arc::new(
            akita_types::RelationRangeImagePlan::new(
                akita_types::RelationWitnessGeometry::for_level(&level, &layout, E::DEGREE)?,
                geometry,
                akita_types::DigitRangePlan::new(
                    akita_error::checked::pow2(level.open().digits.log_basis as usize)
                        .ok_or(AkitaError::InvalidProof)?,
                )?,
                witness_layout,
                &layout,
            )?,
        ));
        witness_handle.set_operation_binding(self.next_binding(parent_binding)?);
        Ok(witness_handle)
    }
}

impl<F, E> crate::opaque::OpaqueRelationWitnessKernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn prepare_relation_witness(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::opaque::ValidatedRelationWitnessPlan,
    ) -> Result<crate::opaque::PreparedRelationHandle<Self::RelationHandle>, AkitaError> {
        self.validate_extension::<E>()?;
        let parent = witness_handle.operation_binding();
        self.validate_binding(&parent)?;
        let prepared = crate::opaque::consumer_kernels::RecursiveRelationWitnessKernel::prepare_relation_witness(
            self,
            Some(self.prepared::<F>()?),
            witness_handle,
            plan,
        )?;
        let (mut relation_handle, column_bits, coefficient_bits) = prepared.into_parts();
        relation_handle.relation_plan = witness_handle.relation_plan.clone();
        let metadata = crate::opaque::RelationWitnessMetadata::try_new(
            plan.witness_len(),
            column_bits,
            coefficient_bits,
        )?;
        relation_handle.set_operation_binding(self.next_binding(parent)?);
        Ok(crate::opaque::PreparedRelationHandle::new(
            relation_handle,
            metadata,
        ))
    }
}

impl<F, E> crate::opaque::OpaqueStage1Kernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn begin_stage1(
        &self,
        relation_handle: &Self::RelationHandle,
        plan: &crate::opaque::ValidatedStage1Plan<E>,
    ) -> Result<Self::Stage1SessionHandle, AkitaError> {
        self.validate_extension::<E>()?;
        let parent = relation_handle.operation_binding();
        self.validate_binding(&parent)?;
        if let Some(expected) = &relation_handle.relation_plan {
            let (schedule, _) = self.owner().proof_plan(parent.scope_id())?;
            let level = if parent.fold_level() == 0 {
                &schedule.root.params
            } else {
                &schedule
                    .recursive_folds
                    .get(parent.fold_level() as usize - 1)
                    .ok_or(AkitaError::InvalidProof)?
                    .params
            };
            if plan.digit_range() != expected.digit_range_plan()
                || plan.domain() != expected.digit_witness_domain()
                || plan.witness_len() != expected.witness_layout().live_coeff_len()
                || plan.physical() != akita_types::PhysicalResponsePlan::new(level, expected)?
            {
                return Err(AkitaError::InvalidInput(
                    "Stage 1 policy differs from the retained relation computation".into(),
                ));
            }
        } else {
            #[cfg(not(test))]
            return Err(AkitaError::InvalidInput(
                "relation handle has no admitted Stage 1 policy".into(),
            ));
        }
        let session_state =
            crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel::<
                crate::opaque::CpuRelationHandle,
                F,
                E,
            >::begin_stage1(self, Some(self.prepared::<F>()?), relation_handle, plan)?;
        let binding = self.next_binding(parent)?;
        let lease = self.binding_lease(&binding)?;
        Ok(crate::opaque::CpuStage1SessionHandle {
            binding,
            lease,
            session_state,
        })
    }

    fn stage1_round_polynomial(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: crate::opaque::Stage1Step,
        round: usize,
        previous_claim: E,
    ) -> Result<crate::opaque::Stage1RoundPolynomial<E>, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease(),
        )?;
        crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel::<
            crate::opaque::CpuRelationHandle,
            F,
            E,
        >::stage1_round_polynomial(
            self,
            &mut session_handle.session_state,
            step,
            round,
            previous_claim,
        )
    }

    fn bind_stage1_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: crate::opaque::Stage1Step,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease(),
        )?;
        crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel::<
            crate::opaque::CpuRelationHandle,
            F,
            E,
        >::bind_stage1_challenge(
            self,
            &mut session_handle.session_state,
            step,
            round,
            challenge,
        )
    }

    fn stage1_public_transition(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        step: crate::opaque::Stage1Step,
    ) -> Result<crate::opaque::Stage1PublicTransition<E>, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease(),
        )?;
        crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel::<
            crate::opaque::CpuRelationHandle,
            F,
            E,
        >::stage1_public_transition(self, &mut session_handle.session_state, step)
    }

    fn bind_stage1_batch_challenge(
        &self,
        session_handle: &mut Self::Stage1SessionHandle,
        transition: crate::opaque::Stage1Transition,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease(),
        )?;
        crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel::<
            crate::opaque::CpuRelationHandle,
            F,
            E,
        >::bind_stage1_batch_challenge(
            self,
            &mut session_handle.session_state,
            transition,
            challenge,
        )
    }

    fn finish_stage1(
        &self,
        session_handle: Self::Stage1SessionHandle,
    ) -> Result<crate::opaque::Stage1FinalClaims<E>, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease(),
        )?;
        crate::opaque::consumer_kernels::RecursiveWitnessStage1Kernel::<
            crate::opaque::CpuRelationHandle,
            F,
            E,
        >::finish_stage1(self, session_handle.session_state)
    }
}

impl<F, E> crate::opaque::OpaqueStage2Kernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn begin_stage2(
        &self,
        relation_handle: Self::RelationHandle,
        plan: crate::opaque::ValidatedRelationSessionPlan<'_, F, E>,
    ) -> Result<Self::Stage2SessionHandle, AkitaError> {
        self.validate_extension::<E>()?;
        let parent = relation_handle.operation_binding();
        self.validate_binding(&parent)?;
        if let Some(expected) = &relation_handle.relation_plan {
            let (schedule, _) = self.owner().proof_plan(parent.scope_id())?;
            let parameters = if parent.fold_level() == 0 {
                &schedule.root.params
            } else {
                &schedule
                    .recursive_folds
                    .get(parent.fold_level() as usize - 1)
                    .ok_or(AkitaError::InvalidProof)?
                    .params
            };
            if plan.relation().relation_plan != expected.as_ref()
                || plan.relation().parameters != parameters
                || plan.physical_l2().map(|request| request.plan)
                    != akita_types::PhysicalResponsePlan::new(parameters, expected)?.as_ref()
            {
                return Err(AkitaError::InvalidInput(
                    "Stage 2 policy differs from the retained relation computation".into(),
                ));
            }
        } else {
            #[cfg(not(test))]
            return Err(AkitaError::InvalidInput(
                "relation handle has no admitted Stage 2 policy".into(),
            ));
        }
        let mut session_handle = crate::opaque::consumer_kernels::RecursiveWitnessRelationKernel::begin_relation_session(
            self,
            Some(self.prepared::<F>()?),
            relation_handle,
            plan,
        )?;
        let binding = self.next_binding(parent)?;
        let lease = self.binding_lease(&binding)?;
        session_handle.set_operation_binding(binding, lease);
        Ok(session_handle)
    }

    fn stage2_input_claim(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<E, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease()?,
        )?;
        Ok(crate::opaque::consumer_kernels::RelationWitnessSession::input_claim(session_handle))
    }

    fn stage2_num_rounds(
        &self,
        session_handle: &Self::Stage2SessionHandle,
    ) -> Result<usize, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease()?,
        )?;
        Ok(crate::opaque::consumer_kernels::RelationWitnessSession::num_rounds(session_handle))
    }

    fn stage2_round_polynomial(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        previous_claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease()?,
        )?;
        crate::opaque::consumer_kernels::RelationWitnessSession::round_polynomial(
            session_handle,
            round,
            previous_claim,
        )
    }

    fn bind_stage2_challenge(
        &self,
        session_handle: &mut Self::Stage2SessionHandle,
        round: usize,
        challenge: E,
    ) -> Result<(), AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease()?,
        )?;
        crate::opaque::consumer_kernels::RelationWitnessSession::bind_challenge(
            session_handle,
            round,
            challenge,
        )
    }

    fn finish_stage2(
        &self,
        session_handle: Self::Stage2SessionHandle,
    ) -> Result<crate::opaque::RelationWitnessFinalClaims<E>, AkitaError> {
        self.validate_leased_binding(
            &session_handle.operation_binding(),
            session_handle.scope_lease()?,
        )?;
        crate::opaque::consumer_kernels::RelationWitnessSession::finish(session_handle)
    }
}

impl<F, E> crate::opaque::OpaqueWitnessOpeningKernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn prepare_native_witness_opening(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::opaque::ValidatedRecursiveGroupOpeningPlan<'_, E>,
    ) -> Result<crate::opaque::PreparedGroupOpening<E, Self::PreparedOpeningHandle>, AkitaError>
    {
        self.validate_extension::<E>()?;
        let parent = witness_handle.operation_binding();
        self.validate_binding(&parent)?;
        let (schedule, _) = self.owner().proof_plan(parent.scope_id())?;
        let terminal = &schedule.terminal;
        if parent.fold_level() as usize != schedule.recursive_folds.len() + 1
            || plan.ring_dimension() != terminal.d_a()
            || plan.positions_per_block() != terminal.blocks.positions_per_block
            || plan.live_blocks() != terminal.blocks.live_blocks
            || plan.opening_method() != akita_types::OpeningMethod::EvaluationTrace
        {
            return Err(AkitaError::InvalidInput(
                "native terminal opening differs from the admitted terminal fold".into(),
            ));
        }
        let opening = akita_types::dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            plan.ring_dimension(),
            |D| {
                crate::opaque::prepare_recursive_witness_opening::<F, E, crate::opaque::CpuBackend, D>(
                    self,
                    Some(self.prepared::<F>()?),
                    witness_handle,
                    plan,
                )
            }
        )?;
        let (scalar_openings, mut opening_handle) = opening.into_parts();
        opening_handle.set_operation_binding(self.next_binding(parent)?);
        opening_handle.mark_terminal_native();
        Ok(crate::opaque::PreparedGroupOpening::new(
            scalar_openings,
            opening_handle,
        ))
    }

    fn terminal_native_witness_opening(
        &self,
        opening_handle: Self::PreparedOpeningHandle,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
        let binding = opening_handle.operation_binding();
        self.validate_binding(&binding)?;
        if !opening_handle.is_terminal_native() {
            return Err(AkitaError::InvalidInput(
                "opening was not prepared for terminal publication".into(),
            ));
        }
        let (schedule, _) = self.owner().proof_plan(binding.scope_id())?;
        if binding.fold_level() as usize != schedule.recursive_folds.len() + 1 {
            return Err(AkitaError::InvalidInput(
                "terminal opening belongs to a nonterminal fold".into(),
            ));
        }
        crate::opaque::PreparedGroupOpeningKernel::terminal_evaluation_trace_opening(
            self,
            Some(self.prepared::<F>()?),
            opening_handle,
        )
    }
}

impl<F, E> crate::opaque::OpaqueTerminalFoldKernel<F, E> for crate::opaque::CpuBackend
where
    F: Field
        + CanonicalEncoding
        + akita_serialization::AkitaSerialize
        + jolt_field::PseudoMersenne
        + jolt_field::Ring
        + jolt_field::Unreduced
        + Send
        + Sync
        + 'static,
    <F as jolt_field::Unreduced>::Wide: From<F> + jolt_field::AdditiveGroup,
    E: jolt_field::ExtField<F>
        + akita_types::FpExtEncoding<F>
        + akita_serialization::AkitaSerialize
        + jolt_field::Ring
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,
{
    fn probe_terminal_fold(
        &self,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::opaque::ValidatedTerminalFoldProbePlan<'_>,
    ) -> Result<crate::opaque::FoldProbeOutcome<Self::AcceptedTerminalFoldHandle>, AkitaError> {
        self.validate_extension::<E>()?;
        let binding = witness_handle.operation_binding();
        self.validate_binding(&binding)?;
        let (schedule, _) = self.owner().proof_plan(binding.scope_id())?;
        let terminal = &schedule.terminal;
        let shape = terminal
            .response_shape
            .layout
            .groups
            .first()
            .ok_or(AkitaError::InvalidProof)?;
        if binding.fold_level() as usize != schedule.recursive_folds.len() + 1
            || plan.ring_dimension() != terminal.d_a()
            || plan.num_positions_per_block() != terminal.blocks.positions_per_block
            || plan.num_digits() != terminal.inner.digits.num_digits
            || plan.log_basis() != terminal.inner.digits.log_basis
            || plan.challenges().num_claims() != 1
            || plan.challenges().num_live_blocks_per_claim() != terminal.blocks.live_blocks
            || plan.coordinate_count() != shape.z_coords
            || plan.linf_cap() != shape.z_linf_cap
            || plan.l2_sq_cap() != terminal.response_l2_sq_cap()
            || plan.rice_low_bits() != shape.z_rice_low_bits
            || plan.payload_bytes() != shape.z_payload_bytes
        {
            return Err(AkitaError::InvalidInput(
                "terminal fold probe differs from the admitted policy".into(),
            ));
        }
        akita_types::dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            plan.ring_dimension(),
            |D| {
                let binding = witness_handle.operation_binding();
                self.validate_binding(&binding)?;
                let outcome = crate::opaque::consumer_kernels::RecursiveWitnessTerminalFoldKernel::<
                    _,
                    F,
                    D,
                >::probe_terminal_recursive_witness(
                    self,
                    Some(self.prepared::<F>()?),
                    witness_handle,
                    plan,
                )?;
                Ok(match outcome {
                    crate::opaque::FoldProbeOutcome::Rejected => {
                        crate::opaque::FoldProbeOutcome::Rejected
                    }
                    crate::opaque::FoldProbeOutcome::Accepted {
                        mut fold_handle,
                        diagnostics,
                    } => {
                        fold_handle.bind(self.next_binding(binding)?);
                        #[cfg(feature = "response-model-diagnostics")]
                        let diagnostics = diagnostics.with_source_l2_sq(
                            crate::opaque::fold::response_model_diagnostics_enabled()
                                .then(|| witness_handle.source_l2_sq::<F>())
                                .flatten(),
                        );
                        crate::opaque::FoldProbeOutcome::Accepted {
                            fold_handle,
                            diagnostics,
                        }
                    }
                })
            }
        )
    }

    fn encode_terminal_fold(
        &self,
        terminal_fold_handle: Self::AcceptedTerminalFoldHandle,
        plan: &crate::opaque::ValidatedTerminalZEncodingPlan,
    ) -> Result<Vec<u8>, AkitaError> {
        self.validate_extension::<E>()?;
        let binding = terminal_fold_handle.binding();
        self.validate_binding(&binding)?;
        akita_types::dispatch_for_field!(
            akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
            F,
            plan.ring_dimension(),
            |D| {
                crate::opaque::consumer_kernels::RecursiveWitnessTerminalFoldKernel::<_, F, D>::encode_terminal_recursive_witness(
                    self,
                    Some(self.prepared::<F>()?),
                    &terminal_fold_handle,
                    plan,
                )
            }
        )
    }
}

pub(crate) mod witness_build;
pub(crate) use witness_build::CpuCommitmentMaterialHandle;
pub(crate) use witness_build::{
    balanced_decompose_centered_i32_i8_into, CpuCommitmentMaterial,
    CpuRecursiveWitnessAssemblyState, OpaqueCompressionState, OpaqueInnerRelationState,
    PreparedOpeningWitness, PreparedRingSwitchGroup, PublicPreparedRelationOpening,
};
#[cfg(test)]
pub(crate) use witness_build::{cpu_recursive_witness_build, RingRelationWitness};
#[cfg(test)]
pub(crate) use witness_build::{
    fold_coefficient_packing_group, materialize_coefficient_packing_d_input,
    materialize_compression_witness, multi_group_quotient_calls, quotient_decomposition_calls,
    reset_multi_group_quotient_calls, reset_quotient_decomposition_calls, CompressionSourceId,
    CompressionSourceWitness, CompressionWitnessMaterialization, RelationDQuotientWitness,
    RingRelationGroupWitness,
};

pub(crate) mod fold;
pub(crate) use fold::aggregate_decompose_fold_witnesses;
pub use fold::CpuFoldResponses;
pub(crate) use fold::{CpuAcceptedFold, CpuAcceptedTerminalFold};
pub(crate) use fold::{CpuAcceptedFoldHandle, CpuAcceptedTerminalFoldHandle};

#[cfg(test)]
mod kernel_tests;

#[cfg(test)]
mod opening_tests;
#[cfg(test)]
mod relation_tests;
