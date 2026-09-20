use crate::sources::packed_digits::PackedSignedDigitWriter;
#[cfg(feature = "response-model-diagnostics")]
use crate::sources::packed_digits::PackedSignedDigits;
use crate::opaque::RecursiveWitnessFlat;
use crate::commitment::{InnerRelationMaterial, OuterCompressionMaterial};
use crate::opaque::{OperationCtx, RuntimeRingSwitchProveBackend};
use crate::kernels::linear::decompose_commit_blocks_into;
use crate::protocol::validate_chunked_witness_cfg;
use crate::validation::validate_i8_setup_log_basis;
use akita_algebra::balanced_decompose_coefficients_pow2_i8_into;
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    dispatch_for_field, emit_witness_e_planes, emit_witness_t_planes, r_decomp_levels,
    CommitmentRingDims, CommittedGroupParams, CompressionWitnessSpan, DigitBlocks,
    PackedNegativeBinary, RingRelationInstance, RingRole, RingVec, WitnessLayout,
    WitnessUnitLayout,
};
use jolt_field::{CanonicalEncoding, Field, Ring};

#[cfg(test)]
#[path = "commitment_binding_tests.rs"]
mod commitment_binding_tests;

/// Commitment-consumer state retained for recursive T construction.
/// Its coefficient rows are private to this consumer module.
#[derive(Clone)]
pub(crate) struct OpaqueInnerRelationState<F: Field> {
    material: crate::commitment::InnerRelationStateMaterial<F>,
    source_count: usize,
}

/// Retained compression state whose witness and quotient material is visible
/// only to the recursive-witness consumer.
pub(crate) struct OpaqueCompressionState<F: Field> {
    material: crate::commitment::PortableCompressionState<F>,
}

/// CPU consumer's private common material for setup-prefix and recursive state.
pub struct CpuCommitmentMaterialHandle<F: Field> {
    binding: crate::opaque::OperationBinding,
    inner: crate::commitment::InnerRelationStateMaterial<F>,
    compression: Option<crate::commitment::PortableCompressionState<F>>,
    source_count: usize,
    commitment_id: Option<u128>,
    public_commitment: Option<RingVec<F>>,
}

impl<F: Field> CpuCommitmentMaterialHandle<F> {
    fn validate_terminal(&self, backend: &crate::opaque::CpuBackend) -> Result<(), AkitaError> {
        backend.validate_binding(&self.binding)?;
        let (schedule, _) = backend.owner().proof_plan(self.binding.scope_id())?;
        if self.binding.fold_level() as usize != schedule.recursive_folds.len() + 1
            || self.commitment_id.is_none()
            || self.public_commitment.is_some()
            || self.compression.is_some()
        {
            return Err(AkitaError::InvalidInput(
                "material was not committed for terminal publication".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn bind_commitment(&mut self, id: u128, public: Option<RingVec<F>>) {
        self.commitment_id = Some(id);
        self.public_commitment = public;
    }

    pub(crate) fn validate_public_commitment(
        &self,
        public: &RingVec<F>,
    ) -> Result<(), AkitaError> {
        let expected = self.public_commitment.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("material has no admitted public commitment".into())
        })?;
        if expected.coeffs() != public.coeffs()
            || (public.ring_dim() != 0 && public.ring_dim() != expected.ring_dim())
        {
            return Err(AkitaError::InvalidInput(
                "public commitment differs from retained material".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn commitment_id(&self)->Option<u128> {self.commitment_id}

    pub(crate) fn binding(&self) -> crate::opaque::OperationBinding {
        self.binding
    }
    pub(crate) fn bind(
        &mut self,
        binding: crate::opaque::OperationBinding,
    ) {
        self.binding = binding;
    }
}

impl<F> CpuCommitmentMaterialHandle<F>
where
    F: Field + CanonicalEncoding + AkitaSerialize + 'static,
{
    pub(crate) fn from_state<S>(
        state: S,
        plan: &crate::commitment::CommitmentExecutionPlan,
        source_count: usize,
    ) -> Result<Self, AkitaError>
    where
        S: crate::commitment::InnerRelationState<
                F,
                Material = crate::commitment::InnerRelationStateMaterial<F>,
            > + crate::commitment::OuterCompressionState<
                F,
                Material = crate::commitment::PortableCompressionState<F>,
            >,
    {
        let inner = state.inner_relation_material(plan.inner(), source_count)?;
        inner.validate_relation_material(plan.inner(), source_count)?;
        let compression = match (plan.compression(), plan.relation_mode()) {
            (Some(compression), Some(mode)) => {
                let material = state.outer_compression_material(compression, mode)?;
                material.validate_compression_material(compression, mode)?;
                Some(material)
            }
            (None, None) => None,
            _ => {
                return Err(AkitaError::InvalidSetup(
                    "commitment plan has inconsistent compression metadata".into(),
                ));
            }
        };
        Ok(Self {
            binding: crate::opaque::OperationBinding::legacy_unscoped(),
            commitment_id: None,
            public_commitment: None,
            inner,
            compression,
            source_count,
        })
    }
}

impl<F> crate::opaque::CommitmentRelationMaterial<F> for CpuCommitmentMaterialHandle<F>
where
    F: Field + CanonicalEncoding + Send + 'static,
{
    fn metadata(&self) -> crate::opaque::CommitmentMaterialMetadata {
        crate::opaque::CommitmentMaterialMetadata::try_new(self.inner.ring_dimension(),self.source_count,self.compression.is_some())
            .expect("CPU material has validated public geometry")
    }
}

pub(crate) type CpuCommitmentMaterial<F> = CpuCommitmentMaterialHandle<F>;

macro_rules! impl_cpu_terminal_commitment_material {
    ($backend:ty) => {
        impl<F> crate::opaque::TerminalCommitmentMaterialKernel<F, CpuCommitmentMaterial<F>>
            for $backend
        where
            F: Field + CanonicalEncoding + AkitaSerialize + Send + 'static,
        {
            fn terminal_message(
                &self,
                material: &CpuCommitmentMaterial<F>,
            ) -> Result<crate::commitment::TerminalTFieldsMessage, AkitaError> {
                material.validate_terminal(self)?;
                crate::commitment::InnerRelationMaterial::terminal_message(&material.inner)
            }

            fn consume_terminal_row(
                &self,
                material: CpuCommitmentMaterial<F>,
            ) -> Result<RingVec<F>, AkitaError> {
                material.validate_terminal(self)?;
                crate::commitment::InnerRelationMaterial::into_terminal_row(material.inner)
            }
        }
    };
}

impl_cpu_terminal_commitment_material!(crate::opaque::CpuBackend);

impl<F: Field> OpaqueCompressionState<F> {
    pub(crate) fn new(
        material: crate::commitment::PortableCompressionState<F>,
    ) -> Self {
        Self { material }
    }

    fn into_material(self) -> crate::commitment::PortableCompressionState<F> {
        self.material
    }
}

impl<F: Field> OpaqueInnerRelationState<F> {
    pub(crate) fn new(
        material: crate::commitment::InnerRelationStateMaterial<F>,
        source_count: usize,
    ) -> Self {
        Self {
            material,
            source_count,
        }
    }

    pub(crate) const fn ring_dimension(&self) -> usize {
        self.material.ring_dimension()
    }

    pub(crate) const fn source_count(&self) -> usize {
        self.source_count
    }

    fn rows(&self) -> &[RingVec<F>] {
        self.material.rows()
    }
}

mod coefficient_packing;
mod compression_witness;
mod d_rows;
mod relation_quotient;
#[cfg(test)]
pub(crate) use coefficient_packing::{
    fold_coefficient_packing_group, materialize_coefficient_packing_d_input,
};
pub(crate) use compression_witness::{
    materialize_compression_witness, CompressionSourceId, CompressionSourceWitness,
    CompressionWitnessMaterialization,
};
use relation_quotient::{compute_multi_group_relation_quotient, RelationQuotientOutput};
#[cfg(test)]
pub(crate) use relation_quotient::{
    multi_group_quotient_calls, reset_multi_group_quotient_calls,
};

enum ConsumerOpeningKind<F: Field> {
    EvaluationTrace {
        e_folded: RingVec<F>,
        ring_multiplier_point: akita_types::RingMultiplierOpeningPoint<F>,
    },
    CoefficientPacking {
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    },
}

/// Opaque consumer state joining prepared opening rows to their E digits.
pub(crate) struct PreparedOpeningWitness<F: Field> {
    e_hat: DigitBlocks,
    kind: ConsumerOpeningKind<F>,
}

fn decompose_opening_rows<F: Field + CanonicalEncoding, const D: usize>(
    pre_folded_e: &[&[akita_algebra::CyclotomicRing<F, D>]],
    role_subcolumns: usize,
    depth_open: usize,
    log_basis: u32,
) -> Result<DigitBlocks, AkitaError> {
    let q = (-F::one())
        .to_u128_checked()
        .expect("Akita field element must fit in u128")
        + 1;
    let params = BalancedDecomposePow2Params::new(depth_open, log_basis, q);
    let total_rows: usize = pre_folded_e.iter().map(|rows| rows.len()).sum();
    if role_subcolumns == 0 || !total_rows.is_multiple_of(role_subcolumns) {
        return Err(AkitaError::InvalidSetup(
            "E rows do not form complete native-role subcolumn groups".into(),
        ));
    }
    let planes_per_semantic = role_subcolumns
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("E digit block width overflow".into()))?;
    let mut e_hat =
        DigitBlocks::zeroed(vec![planes_per_semantic; total_rows / role_subcolumns], D)?;
    let mut offset = 0usize;
    for folded_rows in pre_folded_e {
        for row in *folded_rows {
            row.balanced_decompose_pow2_i8_into_with_params(
                &mut e_hat.typed_planes_mut::<D>()?[offset..offset + depth_open],
                &params,
            );
            offset += depth_open;
        }
    }
    Ok(e_hat)
}

impl<F: Field + CanonicalEncoding> PreparedOpeningWitness<F> {
    pub(crate) fn evaluation_trace<const D: usize, E: Field>(
        point: &akita_types::PreparedOpeningPoint<F, E>,
        folded_by_claim: &[RingVec<F>],
        role_subcolumns: usize,
        depth_open: usize,
        log_basis: u32,
    ) -> Result<Self, AkitaError> {
        let typed = folded_by_claim
            .iter()
            .map(RingVec::as_ring_slice::<D>)
            .collect::<Result<Vec<_>, _>>()?;
        let e_hat = decompose_opening_rows::<F, D>(&typed, role_subcolumns, depth_open, log_basis)?;
        let e_folded = RingVec::from_coeffs(
            folded_by_claim
                .iter()
                .flat_map(|block| block.coeffs().iter().copied())
                .collect(),
        );
        Ok(Self {
            e_hat,
            kind: ConsumerOpeningKind::EvaluationTrace {
                e_folded,
                ring_multiplier_point: point.ring_multiplier_point.clone(),
            },
        })
    }

    pub(crate) fn coefficient_packing<const D: usize>(
        level: &CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        geometry: &akita_types::RelationWitnessGeometry,
        group_index: usize,
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    ) -> Result<Self, AkitaError> {
        let e_hat = coefficient_packing::materialize_coefficient_packing_d_input::<F, D>(
            level,
            opening_batch,
            geometry,
            group_index,
            &partials_by_claim,
        )?;
        Ok(Self {
            e_hat,
            kind: ConsumerOpeningKind::CoefficientPacking { partials_by_claim },
        })
    }

    fn e_hat(&self) -> &DigitBlocks {
        &self.e_hat
    }

    pub(crate) fn into_relation_group(
        self,
        fold: crate::opaque::CpuAcceptedFold<F>,
        challenges: akita_types::GroupFoldChallenges,
        inner_relation: crate::opaque::OpaqueInnerRelationState<F>,
        role_dims: CommitmentRingDims,
    ) -> Result<
        (
            akita_types::RingRelationGroupOpening<F>,
            RingRelationGroupWitness<F>,
        ),
        AkitaError,
    > {
        match (self.kind, challenges) {
            (
                ConsumerOpeningKind::EvaluationTrace {
                    e_folded,
                    ring_multiplier_point,
                },
                akita_types::OpeningFamily::EvaluationTrace(challenges),
            ) => Ok((
                akita_types::RingRelationGroupOpening::evaluation_trace(
                    challenges,
                    ring_multiplier_point,
                ),
                RingRelationGroupWitness::from_parts(
                    fold,
                    self.e_hat,
                    e_folded,
                    inner_relation,
                    role_dims,
                ),
            )),
            (
                ConsumerOpeningKind::CoefficientPacking { partials_by_claim },
                akita_types::OpeningFamily::SubringCoefficientPacking(challenges),
            ) => {
                let product = coefficient_packing::fold_coefficient_packing_group(
                    challenges.geometry(),
                    &partials_by_claim,
                    challenges.canonical(),
                )?;
                Ok((
                    akita_types::RingRelationGroupOpening::coefficient_packing(challenges),
                    RingRelationGroupWitness::from_coefficient_packing_parts(
                        fold,
                        self.e_hat,
                        product,
                        inner_relation,
                        role_dims,
                    ),
                ))
            }
            _ => Err(AkitaError::InvalidSetup(
                "prepared opening material and fold challenges disagree".into(),
            )),
        }
    }
}

pub(crate) type PublicPreparedRelationOpening<F, E> = akita_types::OpeningFamily<
    akita_types::PreparedOpeningPoint<F, E>,
    akita_types::PreparedSubringCoefficientPackingPoint<E>,
>;
type PreparedGroupWitnessOutput<F, E> = (
    PreparedOpeningWitness<F>,
    PublicPreparedRelationOpening<F, E>,
    Vec<E>,
);

pub(crate) fn prepare_group_opening_witness<F, E, const D: usize>(
    handle: &crate::opaque::CpuPreparedOpeningHandle<F, E>,
    level: &CommittedGroupParams,
    opening_batch: &akita_types::OpeningClaimsLayout,
    geometry: &akita_types::RelationWitnessGeometry,
    group_index: usize,
    group_dims: CommitmentRingDims,
) -> Result<PreparedGroupWitnessOutput<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: jolt_field::ExtField<F>,
{
    let (opening, public) = handle.relation_opening::<D>(
        level,
        opening_batch,
        geometry,
        group_index,
        group_dims,
    )?;
    Ok((opening, public, handle.scalar_openings().to_vec()))
}

pub(crate) fn prepare_opening_relation_rows<F, RB, const D: usize>(
    ring_switch_ctx: &OperationCtx<'_, F, RB>,
    opening_batch: &akita_types::OpeningClaimsLayout,
    openings: &[PreparedOpeningWitness<F>],
    has_preceding_groups: bool,
    d_row_len: usize,
    log_basis: u32,
    relation_mode: akita_types::RingRelationMode,
) -> Result<(RingVec<F>, RelationDQuotientWitness<F>), AkitaError>
where
    F: Field + CanonicalEncoding,
    RB: crate::opaque::RingSwitchProveBackend<F, D> + crate::opaque::DigitRowsComputeBackend<F>,
{
    let concatenated = has_preceding_groups
        .then(|| {
            coefficient_packing::concatenate_group_d_inputs(
                opening_batch,
                &openings
                    .iter()
                    .map(PreparedOpeningWitness::e_hat)
                    .collect::<Vec<_>>(),
            )
        })
        .transpose()?;
    let e_hat = concatenated
        .as_ref()
        .or_else(|| openings.first().map(PreparedOpeningWitness::e_hat))
        .ok_or(AkitaError::InvalidProof)?;
    if d_row_len == 0 {
        let quotients = match relation_mode {
            akita_types::RingRelationMode::QuotientLift => {
                RelationDQuotientWitness::QuotientLift(RingVec::from_coeffs(Vec::new()))
            }
            akita_types::RingRelationMode::ReducedEvaluation => {
                RelationDQuotientWitness::ReducedEvaluation
            }
        };
        return Ok((RingVec::from_coeffs(Vec::new()), quotients));
    }
    match d_rows::compute_relation_d_rows::<F, RB, D>(
        ring_switch_ctx,
        d_row_len,
        log_basis,
        e_hat,
        relation_mode,
    )? {
        d_rows::RelationDRows::QuotientLift { reduced, quotients } => Ok((
            RingVec::from_ring_elems(&reduced),
            RelationDQuotientWitness::QuotientLift(RingVec::from_ring_elems(&quotients)),
        )),
        d_rows::RelationDRows::ReducedEvaluation { reduced } => Ok((
            RingVec::from_ring_elems(&reduced),
            RelationDQuotientWitness::ReducedEvaluation,
        )),
    }
}

pub(crate) struct PreparedRelationPayload<F: Field + CanonicalEncoding> {
    inner: Vec<crate::opaque::OpaqueInnerRelationState<F>>,
    relation_rhs: RingVec<F>,
    opening_payload: RingVec<F>,
    opening_payload_ring_dimension: usize,
    compression: Option<CompressionWitnessMaterialization<F>>,
}

/// CPU consumer state retained between public payload publication and the
/// transcript-owned fold grind. No protocol module can inspect its E/T/R or
/// compression material.
pub(crate) struct CpuRecursiveWitnessAssemblyState<F: Field + CanonicalEncoding> {
    level: CommittedGroupParams,
    opening_batch: akita_types::OpeningClaimsLayout,
    group_openings: Vec<PreparedOpeningWitness<F>>,
    d_quotients: RelationDQuotientWitness<F>,
    inner_relation: Vec<crate::opaque::OpaqueInnerRelationState<F>>,
    compression: Option<CompressionWitnessMaterialization<F>>,
}

macro_rules! impl_cpu_recursive_witness_assembly_kernel {
    ($backend:ty) => {
        impl<F, E>
            crate::opaque::consumer_kernels::RecursiveWitnessAssemblyKernel<
                F,
                E,
                crate::opaque::CpuPreparedOpeningHandle<F, E>,
                crate::opaque::CpuAcceptedFold<F>,
                CpuCommitmentMaterial<F>,
            > for $backend
        where
            F: Field + CanonicalEncoding + AkitaSerialize + Send + 'static,
            E: jolt_field::ExtField<F>,
        {
            type State = CpuRecursiveWitnessAssemblyState<F>;
            type Witness = RingRelationWitness<F>;

            fn begin_recursive_witness_assembly(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                prepared_group_openings: &[crate::opaque::CpuPreparedOpeningHandle<F, E>],
                commitment_material: Vec<CpuCommitmentMaterial<F>>,
                level: &CommittedGroupParams,
                opening_batch: &akita_types::OpeningClaimsLayout,
                relation_rhs_layout: &akita_types::RelationRhsLayout,
                group_commitments: &[RingVec<F>],
            ) -> Result<crate::opaque::RecursiveWitnessAssemblyStart<F, E, Self::State>, AkitaError> {
                let prepared = prepared.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "recursive witness assembly requires prepared setup".into(),
                    )
                })?;
                let ctx = OperationCtx::new(
                    self,
                    prepared,
                    crate::opaque::ComputeBackendSetup::prepared_expanded_setup(self, prepared),
                )?;
                if prepared_group_openings.len() != opening_batch.num_groups()
                    || commitment_material.len() != opening_batch.num_groups()
                {
                    return Err(AkitaError::InvalidSize {
                        expected: opening_batch.num_groups(),
                        actual: prepared_group_openings.len().min(commitment_material.len()),
                    });
                }
                let geometry = akita_types::RelationWitnessGeometry::for_level(
                    level,
                    opening_batch,
                    E::DEGREE,
                )?;
                let mut group_openings = Vec::with_capacity(opening_batch.num_groups());
                let mut public_groups = Vec::with_capacity(opening_batch.num_groups());
                for (group_index, opening) in prepared_group_openings.iter().enumerate() {
                    let group_dims = level.group_role_dims_geometry(opening_batch, group_index)?;
                    let (opening, public_kind, scalar_openings) = dispatch_for_field!(
                        ProtocolDispatchSlot::Role(RingRole::Opening),
                        F,
                        group_dims.d_d(),
                        |D_D| prepare_group_opening_witness::<F, E, D_D>(
                            opening,
                            level,
                            opening_batch,
                            &geometry,
                            group_index,
                            group_dims,
                        )
                    )?;
                    group_openings.push(opening);
                    public_groups.push(crate::opaque::PreparedRelationGroupPublic::new(
                        public_kind,
                        scalar_openings,
                    ));
                }
                crate::arithmetic::requirements::warm_relation_ntt_cache(
                    self, prepared, level,
                )?;
                let dims = level.role_dims();
                let (v, d_quotients) = dispatch_for_field!(
                    ProtocolDispatchSlot::Role(RingRole::Opening),
                    F,
                    dims.d_d(),
                    |D_D| prepare_opening_relation_rows::<F, $backend, D_D>(
                        &ctx,
                        opening_batch,
                        &group_openings,
                        level.has_preceding_groups(),
                        level.open().matrix.output_rank(),
                        level.shared_d_digit_log_basis(),
                        level.ring_relation_mode,
                    )
                )?;
                let payload = prepare_relation_payload(
                    &ctx,
                    level,
                    opening_batch,
                    relation_rhs_layout,
                    group_commitments,
                    commitment_material,
                    &v,
                )?;
                let PreparedRelationPayload {
                    inner,
                    relation_rhs,
                    opening_payload,
                    opening_payload_ring_dimension,
                    compression,
                } = payload;
                Ok(crate::opaque::RecursiveWitnessAssemblyStart {
                    groups: public_groups,
                    relation_rhs,
                    opening_payload,
                    opening_payload_ring_dimension,
                    v,
                    state: CpuRecursiveWitnessAssemblyState {
                        level: level.clone(),
                        opening_batch: opening_batch.clone(),
                        group_openings,
                        d_quotients,
                        inner_relation: inner,
                        compression,
                    },
                })
            }

            fn finish_recursive_witness_assembly(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                state: Self::State,
                folds: Vec<crate::opaque::RecursiveWitnessFoldInput<crate::opaque::CpuAcceptedFold<F>>>,
            ) -> Result<crate::opaque::RecursiveWitnessAssemblyFinish<F, Self::Witness>, AkitaError> {
                let _ = prepared.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "recursive witness assembly requires prepared setup".into(),
                    )
                })?;
                let CpuRecursiveWitnessAssemblyState {
                    level,
                    opening_batch,
                    group_openings,
                    d_quotients,
                    inner_relation,
                    compression,
                } = state;
                if folds.len() != opening_batch.num_groups()
                    || group_openings.len() != folds.len()
                    || inner_relation.len() != folds.len()
                {
                    return Err(AkitaError::InvalidProof);
                }
                let mut relation_group_openings = Vec::with_capacity(folds.len());
                let mut group_witnesses = Vec::with_capacity(folds.len());
                for (group_index, ((fold, opening), inner_relation)) in folds
                    .into_iter()
                    .zip(group_openings)
                    .zip(inner_relation)
                    .enumerate()
                {
                    let (fold_handle, challenges) = fold.into_fold_handle_and_challenges();
                    let group_dims = level.group_role_dims_geometry(&opening_batch, group_index)?;
                    let source_count = opening_batch.group_layout(group_index)?.num_polynomials();
                    if inner_relation.ring_dimension() != group_dims.d_a()
                        || inner_relation.source_count() != source_count
                    {
                        return Err(AkitaError::InvalidInput(
                            "inner-relation state shape does not match its commitment group".into(),
                        ));
                    }
                    let (public, witness) = opening.into_relation_group(
                        fold_handle,
                        challenges,
                        inner_relation,
                        group_dims,
                    )?;
                    relation_group_openings.push(public);
                    group_witnesses.push(witness);
                }
                Ok(crate::opaque::RecursiveWitnessAssemblyFinish {
                    group_openings: relation_group_openings,
                    witness: RingRelationWitness::from_groups(
                        group_witnesses,
                        d_quotients,
                        compression,
                    ),
                })
            }
        }
    };
}

impl_cpu_recursive_witness_assembly_kernel!(crate::opaque::CpuBackend);

macro_rules! impl_cpu_opaque_recursive_witness_build {
    ($backend:ty) => {
        impl<F, E> crate::opaque::consumer_kernels::CpuWitnessBuildKernel<F, E> for $backend
        where
            F: Field + CanonicalEncoding + AkitaSerialize + Ring + Send + Sync + 'static,
            E: jolt_field::ExtField<F> + akita_types::FpExtEncoding<F> + Send + Sync + 'static,
        {
            fn begin_recursive_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                scope_id: crate::opaque::ProofScopeId,
                prepared_opening_handles: &[Self::PreparedOpeningHandle],
                commitment_material_handles: Vec<Self::CommitmentMaterialHandle>,
                level: &CommittedGroupParams,
                opening_batch: &akita_types::OpeningClaimsLayout,
                relation_rhs_layout: &akita_types::RelationRhsLayout,
                group_commitments: &[RingVec<F>],
            ) -> Result<
                crate::opaque::RecursiveWitnessBuildStart<
                    F,
                    E,
                    Self::WitnessBuildHandle,
                >,
                AkitaError,
            > {
                let prepared = prepared.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "recursive witness construction requires prepared setup".into(),
                    )
                })?;
                let consumer_id = u64::try_from(scope_id.raw() >> 64).map_err(|_| {
                    AkitaError::InvalidInput("proof-scope consumer identifier overflow".into())
                })?;
                let setup_digest = akita_types::setup_seed_digest(
                    &crate::opaque::ComputeBackendSetup::prepared_expanded_setup(self, prepared)
                        .descriptor
                        .setup_seed,
                )
                .map_err(|err| AkitaError::InvalidSetup(format!("setup identity: {err}")))?;
                let start = crate::opaque::consumer_kernels::RecursiveWitnessAssemblyKernel::begin_recursive_witness_assembly(
                    self,
                    Some(prepared),
                    prepared_opening_handles,
                    commitment_material_handles,
                    level,
                    opening_batch,
                    relation_rhs_layout,
                    group_commitments,
                )?;
                let crate::opaque::RecursiveWitnessAssemblyStart {
                    groups,
                    relation_rhs,
                    opening_payload,
                    opening_payload_ring_dimension,
                    v,
                    state,
                } = start;
                let build_handle = crate::opaque::CpuWitnessBuildHandle {
                    binding: crate::opaque::OperationBinding::new(
                        consumer_id,
                        scope_id,
                        setup_digest,
                        0,
                        self.owner().next_operation_id()?,
                    ),
                    opening_bindings: Vec::new(),
                    assembly_state: state,
                    public_groups: groups.clone(),
                    relation_rhs: relation_rhs.clone(),
                    v: v.clone(),
                    level: level.clone(),
                    opening_batch: opening_batch.clone(),
                };
                Ok(crate::opaque::RecursiveWitnessBuildStart::new(
                    groups,
                    opening_payload,
                    opening_payload_ring_dimension,
                    build_handle,
                ))
            }

            fn finish_recursive_witness(
                &self,
                prepared: Option<&Self::PreparedSetup>,
                build_handle: Self::WitnessBuildHandle,
                fold_inputs: Vec<
                    crate::opaque::RecursiveWitnessFoldInput<Self::AcceptedFoldHandle>,
                >,
                public_inputs: crate::opaque::RecursiveWitnessPublicInputs<'_, F>,
                plan: &crate::opaque::ValidatedRecursiveWitnessPlan<'_, F>,
            ) -> Result<
                crate::opaque::CpuWitnessBuildOutput<F, Self::WitnessHandle>,
                AkitaError,
            > {
                let prepared = prepared.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "recursive witness construction requires prepared setup".into(),
                    )
                })?;
                for (group_index, fold_input) in fold_inputs.iter().enumerate() {
                    let group_params =
                        build_handle.level.group_params_geometry(&build_handle.opening_batch, group_index)?;
                    fold_input.fold_handle().validate_for_build(
                        &build_handle.binding,
                        &group_params,
                        build_handle
                            .opening_batch
                            .group_layout(group_index)?
                            .num_polynomials(),
                        build_handle.level.witness_chunk.num_chunks,
                    )?;
                }
                let crate::opaque::CpuWitnessBuildHandle {
                    binding,
                    assembly_state,
                    public_groups,
                    relation_rhs,
                    v,
                    level,
                    opening_batch,
                    ..
                } = build_handle;
                let finish = <$backend as crate::opaque::consumer_kernels::RecursiveWitnessAssemblyKernel<
                    F,
                    E,
                    crate::opaque::CpuPreparedOpeningHandle<F, E>,
                    crate::opaque::CpuAcceptedFold<F>,
                    CpuCommitmentMaterial<F>,
                >>::finish_recursive_witness_assembly(
                    self, Some(prepared), assembly_state, fold_inputs,
                )?;
                let crate::opaque::RecursiveWitnessAssemblyFinish {
                    group_openings,
                    witness,
                } = finish;
                let dims = level.role_dims();
                RingRelationInstance::check_v_shape_for_level(&v, &level)?;
                let instance = RingRelationInstance::new(
                    group_openings,
                    public_inputs.extension_degree,
                    opening_batch.clone(),
                    public_inputs.gamma.to_vec(),
                    public_inputs.row_coefficient_rings.clone(),
                    relation_rhs,
                    dims,
                )?;
                crate::protocol::validate_prepared_relation_groups(
                    &public_groups,
                    &level,
                    &opening_batch,
                    &instance,
                )?;
                if instance.segment_layout(&level, None)?.live_coeff_len() != plan.logical_len() {
                    return Err(AkitaError::InvalidInput(
                        "recursive witness plan disagrees with its relation instance".into(),
                    ));
                }
                let expanded =
                    crate::opaque::ComputeBackendSetup::prepared_expanded_setup(self, prepared);
                let ctx = OperationCtx::new(self, prepared, expanded)?;
                let witness_handle = cpu_recursive_witness_build(
                    &instance,
                    witness,
                    &ctx,
                    &ctx,
                    &level,
                    plan.commitment_ring_dimension(),
                    binding,
                )?;
                Ok(crate::opaque::CpuWitnessBuildOutput::new(
                    instance,
                    witness_handle,
                ))
            }
        }
    };
}

impl_cpu_opaque_recursive_witness_build!(crate::opaque::CpuBackend);

fn prepare_relation_payload<F, B>(
    ring_switch_ctx: &OperationCtx<'_, F, B>,
    level: &CommittedGroupParams,
    opening_batch: &akita_types::OpeningClaimsLayout,
    layout: &akita_types::RelationRhsLayout,
    commitments: &[RingVec<F>],
    materials: Vec<CpuCommitmentMaterial<F>>,
    v: &RingVec<F>,
) -> Result<PreparedRelationPayload<F>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
    B: crate::opaque::CompressionComputeBackend<F>,
{
    if commitments.len() != opening_batch.num_groups()
        || materials.len() != opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidSize {
            expected: opening_batch.num_groups(),
            actual: commitments.len().min(materials.len()),
        });
    }
    let mut inner = Vec::with_capacity(materials.len());
    let mut outer = Vec::with_capacity(materials.len());
    for material in materials {
        inner.push(crate::opaque::OpaqueInnerRelationState::new(
            material.inner,
            material.source_count,
        ));
        outer.push(
            material
                .compression
                .map(crate::opaque::OpaqueCompressionState::new),
        );
    }
    let order = if level.has_preceding_groups() {
        opening_batch.root_group_order()?
    } else {
        (0..opening_batch.num_groups()).collect()
    };
    if level.payload_mode.is_compressed() {
        let sources = order
            .iter()
            .enumerate()
            .map(|(relation_group_index, &group_index)| {
                let (planned_group_index, plan) =
                    layout.group_compression_plan(relation_group_index)?;
                let commitment = commitments
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?;
                if planned_group_index != group_index
                    || commitment.coeff_len() != plan.terminal_coefficients()
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover received a malformed compressed commitment".into(),
                    ));
                }
                let material = outer
                    .get_mut(group_index)
                    .and_then(Option::take)
                    .ok_or(AkitaError::InvalidProof)?
                    .into_material();
                CompressionSourceWitness::from_outer_state(
                    group_index,
                    plan,
                    material,
                    commitment.coeffs().to_vec(),
                    level.ring_relation_mode,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (compression, report) = materialize_compression_witness(
            ring_switch_ctx,
            layout,
            sources,
            v,
            level.ring_relation_mode,
        )?;
        let opening = compression.source(CompressionSourceId::Opening)?;
        let opening_ring_dimension = opening
            .witness()
            .plan()
            .maps()
            .last()
            .ok_or(AkitaError::InvalidProof)?
            .ring_dimension();
        let opening_payload = RingVec::from_coeffs_with_ring_dim(
            opening.terminal.coefficients().to_vec(),
            opening_ring_dimension,
        )?;
        let group_payloads = (0..layout.groups.len())
            .map(|relation_group_index| {
                let (group_index, _) = layout.group_compression_plan(relation_group_index)?;
                Ok(compression
                    .source(CompressionSourceId::Outer { group_index })?
                    .terminal
                    .coefficients())
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let relation_rhs = akita_types::assemble_compressed_relation_rhs(
            layout,
            &group_payloads,
            opening.terminal.coefficients(),
        )?;
        tracing::info!(
            sources = report.sources,
            maps = report.maps,
            batches = report.batches.len(),
            source_bytes = report.source_bytes,
            terminal_bytes = report.terminal_bytes,
            retained_bytes = report.retained_packed_witness_bytes,
            peak_scratch_bytes = report.executor_peak_scratch_bytes,
            "materialized compression witness"
        );
        return Ok(PreparedRelationPayload {
            inner,
            relation_rhs,
            opening_payload,
            opening_payload_ring_dimension: opening_ring_dimension,
            compression: Some(compression),
        });
    }

    let mut coefficients = Vec::new();
    for &group_index in &order {
        let commitment = commitments
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let dims = level.group_role_dims_geometry(opening_batch, group_index)?;
        let params = level.group_params_geometry(opening_batch, group_index)?;
        if !commitment.can_decode_vec(dims.d_b())
            || commitment.coeff_len() / dims.d_b() != params.logical_b_rows_len()?
        {
            return Err(AkitaError::InvalidInput(
                "batched prover received a malformed raw commitment".into(),
            ));
        }
        coefficients.extend_from_slice(commitment.coeffs());
    }
    let commitment_rows = RingVec::from_coeffs(coefficients);
    Ok(PreparedRelationPayload {
        inner,
        relation_rhs: akita_types::assemble_relation_rhs(layout, v, &commitment_rows)?,
        opening_payload: v.clone(),
        opening_payload_ring_dimension: level.role_dims().d_d(),
        compression: None,
    })
}

type GroupFoldedOpening<F> =
    akita_types::OpeningFamily<RingVec<F>, akita_types::CoefficientPackingFoldProduct<F>>;
