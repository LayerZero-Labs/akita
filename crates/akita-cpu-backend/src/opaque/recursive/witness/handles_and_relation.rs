use crate::sources::packed_digits::PackedSignedDigits;
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};
use std::sync::Arc;

enum OpaquePreparedGroupOpeningKind<F: Field, E: Field> {
    EvaluationTrace {
        point: akita_types::PreparedOpeningPoint<F, E>,
        folded_by_claim: Vec<akita_types::RingVec<F>>,
    },
    CoefficientPacking {
        point: akita_types::PreparedSubringCoefficientPackingPoint<E>,
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
    },
}

/// Consumer-private prepared opening state. Its witness-derived rows never
/// appear in a protocol-facing carrier.
#[doc(hidden)]
pub struct CpuPreparedOpeningHandle<
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F>,
> {
    binding: crate::opaque::OperationBinding,
    kind: OpaquePreparedGroupOpeningKind<F, E>,
    scalar_openings: Vec<E>,
    source: crate::opaque::openings::PreparedOpeningSource<F, E, Cfg>,
    #[cfg(feature = "response-model-diagnostics")]
    source_l2_sq: Option<u128>,
}

impl<F, E, Cfg> CpuPreparedOpeningHandle<F, E, Cfg>
where
    F: Field + CanonicalEncoding,
    E: Field,
    Cfg: akita_config::CommitmentConfig<Field = F>,
{
    pub(crate) const fn is_terminal_native(&self) -> bool {
        matches!(
            self.source,
            crate::opaque::openings::PreparedOpeningSource::TerminalNative
        )
    }

    pub(in crate::opaque) fn source(
        &self,
    ) -> Result<&crate::opaque::openings::RetainedOpeningSource<F, E, Cfg>, AkitaError> {
        match &self.source {
            crate::opaque::openings::PreparedOpeningSource::Retained(source) => Ok(source),
            crate::opaque::openings::PreparedOpeningSource::TerminalNative => Err(
                AkitaError::InvalidInput("terminal opening has no retained proving source".into()),
            ),
        }
    }
    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn set_source_l2_sq(&mut self, source_l2_sq: Option<u128>) {
        self.source_l2_sq = source_l2_sq;
    }
    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) const fn source_l2_sq(&self) -> Option<u128> {
        self.source_l2_sq
    }
    pub(in crate::opaque) fn evaluation_trace(
        binding: crate::opaque::OperationBinding,
        source: crate::opaque::openings::PreparedOpeningSource<F, E, Cfg>,
        point: akita_types::PreparedOpeningPoint<F, E>,
        folded_by_claim: Vec<akita_types::RingVec<F>>,
        scalar_openings: Vec<E>,
    ) -> Self {
        Self {
            binding,
            kind: OpaquePreparedGroupOpeningKind::EvaluationTrace {
                point,
                folded_by_claim,
            },
            scalar_openings,
            source,
            #[cfg(feature = "response-model-diagnostics")]
            source_l2_sq: None,
        }
    }

    pub(in crate::opaque) fn coefficient_packing(
        binding: crate::opaque::OperationBinding,
        source: crate::opaque::openings::PreparedOpeningSource<F, E, Cfg>,
        point: akita_types::PreparedSubringCoefficientPackingPoint<E>,
        partials_by_claim: Vec<crate::opaque::SubringCoefficientPackingPartials<F>>,
        scalar_openings: Vec<E>,
    ) -> Self {
        Self {
            binding,
            kind: OpaquePreparedGroupOpeningKind::CoefficientPacking {
                point,
                partials_by_claim,
            },
            scalar_openings,
            source,
            #[cfg(feature = "response-model-diagnostics")]
            source_l2_sq: None,
        }
    }

    pub(in crate::opaque) fn into_terminal_evaluation_rows(
        self,
    ) -> Result<Vec<akita_types::RingVec<F>>, AkitaError> {
        match self.kind {
            OpaquePreparedGroupOpeningKind::EvaluationTrace {
                folded_by_claim, ..
            } => Ok(folded_by_claim),
            OpaquePreparedGroupOpeningKind::CoefficientPacking { .. } => {
                Err(AkitaError::InvalidProof)
            }
        }
    }

    pub(crate) fn scalar_openings(&self) -> &[E] {
        &self.scalar_openings
    }

    pub(crate) const fn operation_binding(&self) -> crate::opaque::OperationBinding {
        self.binding
    }

    pub(crate) fn relation_opening<const D: usize>(
        &self,
        level: &akita_types::CommittedGroupParams,
        opening_batch: &akita_types::OpeningClaimsLayout,
        geometry: &akita_types::RelationWitnessGeometry,
        group_index: usize,
        group_dims: akita_types::CommitmentRingDims,
    ) -> Result<
        (
            crate::opaque::PreparedOpeningWitness<F>,
            crate::opaque::PublicPreparedRelationOpening<F, E>,
        ),
        AkitaError,
    >
    where
        F: CanonicalEncoding,
    {
        let group = level.group_params_geometry(opening_batch, group_index)?;
        match &self.kind {
            OpaquePreparedGroupOpeningKind::EvaluationTrace {
                point,
                folded_by_claim,
            } => {
                if group.opening_method() != akita_types::OpeningMethod::EvaluationTrace
                    || point.ring_multiplier_point.position_len() != group.num_positions_per_block()
                    || point.ring_multiplier_point.fold_len() != group.num_live_blocks()
                    || folded_by_claim.len()
                        != opening_batch.group_layout(group_index)?.num_polynomials()
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover EvaluationTrace point layout mismatch".into(),
                    ));
                }
                let opening = crate::opaque::PreparedOpeningWitness::evaluation_trace::<D>(
                    folded_by_claim,
                    group_dims.d_a() / group_dims.d_d(),
                    group.num_digits_open(),
                    group.log_basis_open(),
                )?;
                Ok((
                    opening,
                    akita_types::OpeningFamily::EvaluationTrace(point.clone()),
                ))
            }
            OpaquePreparedGroupOpeningKind::CoefficientPacking {
                point,
                partials_by_claim,
            } => {
                if point.num_positions_per_block() != group.num_positions_per_block()
                    || point.num_live_blocks() != group.num_live_blocks()
                    || geometry.group_opening_method(group_index)? != group.opening_method()
                {
                    return Err(AkitaError::InvalidInput(
                        "batched prover coefficient-packing point layout mismatch".into(),
                    ));
                }
                let opening = crate::opaque::PreparedOpeningWitness::coefficient_packing::<D>(
                    level,
                    opening_batch,
                    geometry,
                    group_index,
                    partials_by_claim.clone(),
                )?;
                Ok((
                    opening,
                    akita_types::OpeningFamily::SubringCoefficientPacking(point.clone()),
                ))
            }
        }
    }
}

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[allow(unreachable_pub)]
pub(crate) struct RecursiveWitnessFlat {
    pub(super) digits: PackedSignedDigits,
    pub(super) live_coeff_len: usize,
    pub(super) committed_coeff_len: Option<usize>,
    pub(super) commitment_ring_dim: Option<usize>,
}

/// Opaque CPU-consumer handle for a complete recursive witness.
///
/// Protocol orchestration can transport this handle and query public shape
/// metadata, but the packed coefficient representation remains private to the
/// recursive-witness adapter.
pub struct CpuWitnessHandle {
    pub(crate) pending_successor: Option<u32>,
    pub(in crate::opaque::recursive) relation_plan:
        Option<Arc<akita_types::RelationRangeImagePlan>>,
    pub(in crate::opaque::recursive) manifest: crate::opaque::RecursiveWitnessManifest,
    pub(in crate::opaque::recursive) binding: crate::opaque::OperationBinding,
    pub(in crate::opaque::recursive) logical: RecursiveWitnessFlat,
    pub(in crate::opaque::recursive) committed: Option<RecursiveWitnessFlat>,
}

pub(crate) type OpaqueRecursiveWitness = CpuWitnessHandle;

impl CpuWitnessHandle {
    pub(crate) fn snapshot(&self) -> Self {
        Self {
            pending_successor: self.pending_successor,
            relation_plan: self.relation_plan.clone(),
            manifest: self.manifest,
            binding: self.binding,
            logical: self.logical.clone(),
            committed: self.committed.clone(),
        }
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn source_l2_sq<F: Field>(&self) -> Option<u128> {
        crate::opaque::RootPolyMeta::<F>::exact_integer_coeff_l2_sq(
            self.committed.as_ref().unwrap_or(&self.logical),
        )
    }

    pub(crate) const fn operation_binding(&self) -> crate::opaque::OperationBinding {
        self.binding
    }

    pub(crate) fn set_operation_binding(&mut self, binding: crate::opaque::OperationBinding) {
        self.binding = binding;
    }

    pub(crate) fn initialize_relation_plan<F>(
        &mut self,
        relation: &akita_types::RingRelationInstance<F>,
        level: &akita_types::CommittedGroupParams,
    ) -> Result<(), AkitaError>
    where
        F: Field + CanonicalEncoding,
    {
        if self.relation_plan.is_some() {
            return Err(AkitaError::InvalidInput(
                "recursive witness relation plan is already initialized".into(),
            ));
        }
        let opening_batch = relation.opening_batch();
        let witness_layout = relation.segment_layout(level, None)?;
        if witness_layout.live_coeff_len() != self.manifest.logical_len() {
            return Err(AkitaError::InvalidInput(
                "recursive witness manifest disagrees with its relation instance".into(),
            ));
        }
        let geometry = level.relation_address_geometry(
            opening_batch,
            relation.extension_degree(),
            self.manifest.commitment_ring_dimension(),
            witness_layout.live_coeff_len(),
        )?;
        let digit_range = akita_types::DigitRangePlan::new(
            akita_error::checked::pow2(level.open().digits.log_basis as usize)
                .ok_or(AkitaError::InvalidProof)?,
        )?;
        self.relation_plan = Some(Arc::new(akita_types::RelationRangeImagePlan::new(
            akita_types::RelationWitnessGeometry::for_level(
                level,
                opening_batch,
                relation.extension_degree(),
            )?,
            geometry,
            digit_range,
            witness_layout,
            opening_batch,
        )?));
        Ok(())
    }
}

/// Consumer-owned state retained between recursive opening preparation and EOR.
pub(crate) struct CpuWitnessOpeningHandle<E: Field> {
    pub(super) source_operation: u128,
    pub(super) point: Vec<E>,
    pub(super) tensor_evals: Vec<E>,
    pub(super) witness_len: usize,
    pub(super) ring_dimension: usize,
}

pub struct CpuRelationHandle {
    pub(crate) relation_plan: Option<Arc<akita_types::RelationRangeImagePlan>>,
    pub(super) binding: crate::opaque::OperationBinding,
    pub(super) packed: PackedSignedDigits,
}

pub(crate) type OpaqueWitnessOpeningState<E> = CpuWitnessOpeningHandle<E>;
pub(crate) type ConsumerRelationWitness = CpuRelationHandle;
macro_rules! impl_bound_handle {
    ($handle:ident $(<$field:ident>)?) => {
        impl$(<$field: Field>)? $handle$(<$field>)? {
            pub(crate) const fn operation_binding(
                &self,
            ) -> crate::opaque::OperationBinding {
                self.binding
            }

            pub(crate) fn set_operation_binding(
                &mut self,
                binding: crate::opaque::OperationBinding,
            ) {
                self.binding = binding;
            }
        }
    };
}

impl_bound_handle!(CpuRelationHandle);
impl CpuRelationHandle {
    pub(crate) fn len(&self) -> usize {
        self.packed.len()
    }
}

impl<F, B>
    crate::opaque::consumer_kernels::RecursiveRelationWitnessKernel<OpaqueRecursiveWitness, F> for B
where
    F: Field + CanonicalEncoding,
    B: crate::opaque::ComputeBackendSetup<F>,
{
    type RelationWitness = ConsumerRelationWitness;

    fn prepare_relation_witness(
        &self,
        prepared: Option<&Self::PreparedSetup>,
        witness: &OpaqueRecursiveWitness,
        plan: &crate::opaque::ValidatedRelationWitnessPlan,
    ) -> Result<crate::opaque::PreparedRelationWitness<Self::RelationWitness>, AkitaError> {
        prepared.ok_or_else(|| {
            AkitaError::InvalidInput("relation witness preparation requires prepared setup".into())
        })?;
        if witness.logical.live_coeff_len() != plan.witness_len() {
            return Err(AkitaError::InvalidInput(
                "relation witness plan has a different operation context".into(),
            ));
        }
        let (packed, column_bits, coefficient_bits) = crate::opaque::build_w_evals_compact(
            witness.logical.packed_digits().clone(),
            plan.coefficient_count(),
            plan.extension_degree(),
            plan.opening_source_len(),
        )?;
        Ok(crate::opaque::PreparedRelationWitness::new(
            ConsumerRelationWitness {
                relation_plan: witness.relation_plan.clone(),
                binding: witness.binding.for_operation(0),
                packed,
            },
            column_bits,
            coefficient_bits,
        ))
    }
}
