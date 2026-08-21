//! Whole-group root operations.

use super::*;
use crate::compute::{
    ComputeBackendSetup, DigitRowsComputeBackend, OperationCtx,
    RuntimeCoefficientPackingBackendFor, RuntimeOpeningProveBackendFor, RuntimeRootProvePoly,
    RuntimeTensorBackendFor, RuntimeTensorSource, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::PreparedProverGroup;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::{
    coefficient_packing_scalar_opening, OpeningFamily, OpeningMethod,
    PreparedSubringCoefficientPackingPoint, SubringCoefficientPackingGeometry,
};

pub(crate) struct PreparedEvaluationTraceGroup<F: FieldCore, E: FieldCore> {
    pub(crate) point: PreparedOpeningPoint<F, E>,
    pub(crate) folded_by_claim: Vec<RingVec<F>>,
}

pub(crate) struct PreparedCoefficientPackingGroup<F: FieldCore, E: FieldCore> {
    pub(crate) point: PreparedSubringCoefficientPackingPoint<E>,
    pub(crate) partials_by_claim: Vec<SubringCoefficientPackingPartials<F>>,
}

pub(crate) struct PreparedGroupOpening<F: FieldCore, E: FieldCore> {
    pub(crate) kind:
        OpeningFamily<PreparedEvaluationTraceGroup<F, E>, PreparedCoefficientPackingGroup<F, E>>,
    pub(crate) scalar_openings: Vec<E>,
}

pub(crate) trait RootProverGroupMeta<F: FieldCore> {
    fn num_polynomials(&self) -> usize;
    fn num_vars(&self) -> Result<usize, AkitaError>;
    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128>;
}

pub(crate) trait RootProverGroupOpening<F, E, B>: RootProverGroupMeta<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: FpExtEncoding<F> + ExtField<F> + AkitaSerialize,
    B: ComputeBackendSetup<F> + DigitRowsComputeBackend<F>,
{
    #[allow(clippy::too_many_arguments)]
    fn prepare_opening(
        &self,
        ctx: &OperationCtx<'_, F, B>,
        ring_dimension: usize,
        protocol_point: &[E],
        basis: BasisMode,
        num_positions_per_block: usize,
        num_live_blocks: usize,
        alpha_bits: usize,
        opening_method: OpeningMethod,
    ) -> Result<PreparedGroupOpening<F, E>, AkitaError>;

    fn probe_fold(
        &self,
        ctx: &OperationCtx<'_, F, B>,
        challenges: &crate::protocol::fold_grind::GroupFoldChallenges,
        root_params: &CommittedGroupParams,
        params: &akita_types::GroupOpenPhaseParams,
    ) -> Result<crate::protocol::fold_grind::FoldProbeOutput<F>, AkitaError>;
}

pub(crate) trait RootProverGroupTensor<F, E, B>: RootProverGroupMeta<F>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F> + MulBaseUnreduced<F>,
    B: ComputeBackendSetup<F>,
{
    fn prepare_extension_opening(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
        point: &[E],
    ) -> Result<PreparedExtensionOpeningGroup<E>, AkitaError>;

    fn extension_opening_terms(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
        claim_coefficients: &[E],
        tail_point: &[E],
        eta: &[E],
    ) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError>;
}

impl<F, P> RootProverGroupMeta<F> for PreparedProverGroup<'_, P>
where
    F: FieldCore,
    P: crate::compute::RootPolyMeta<F>,
{
    fn num_polynomials(&self) -> usize {
        self.polynomial_refs().len()
    }

    fn num_vars(&self) -> Result<usize, AkitaError> {
        let first = self.polynomial_refs().first().ok_or_else(|| {
            AkitaError::InvalidInput("prepared prover group must be nonempty".to_string())
        })?;
        let num_vars = crate::compute::RootPolyMeta::num_vars(*first);
        if self
            .polynomial_refs()
            .iter()
            .any(|poly| crate::compute::RootPolyMeta::num_vars(*poly) != num_vars)
        {
            return Err(AkitaError::InvalidInput(
                "opening polynomial groups must have uniform arity".to_string(),
            ));
        }
        Ok(num_vars)
    }

    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        self.polynomial_refs().iter().try_fold(0u128, |sum, poly| {
            crate::compute::RootPolyMeta::<F>::exact_integer_coeff_l2_sq(*poly)
                .and_then(|energy| sum.checked_add(energy))
        })
    }
}

impl<F, E, P, B> RootProverGroupOpening<F, E, B> for PreparedProverGroup<'_, P>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + AkitaSerialize,
    P: RuntimeRootProvePoly<F>,
    B: ComputeBackendSetup<F>
        + DigitRowsComputeBackend<F>
        + RuntimeOpeningProveBackendFor<F, P>
        + RuntimeCoefficientPackingBackendFor<F, P, E>,
{
    fn prepare_opening(
        &self,
        ctx: &OperationCtx<'_, F, B>,
        ring_dimension: usize,
        protocol_point: &[E],
        basis: BasisMode,
        num_positions_per_block: usize,
        num_live_blocks: usize,
        alpha_bits: usize,
        opening_method: OpeningMethod,
    ) -> Result<PreparedGroupOpening<F, E>, AkitaError> {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| {
                if let OpeningMethod::SubringCoefficientPacking {
                    challenge_subring_dimension,
                } = opening_method
                {
                    let geometry = SubringCoefficientPackingGeometry::try_new(
                        E::EXT_DEGREE,
                        D,
                        challenge_subring_dimension,
                    )?;
                    let source_num_vars = self.num_vars()?;
                    let polys = self.polynomial_refs();
                    let first = polys.first().ok_or_else(|| {
                        AkitaError::InvalidInput("opening group must be nonempty".into())
                    })?;
                    let num_live_positions =
                        <P as crate::compute::RootPolyShape<F, D>>::num_live_ring_elems(*first);
                    if polys.iter().any(|poly| {
                        <P as crate::compute::RootPolyShape<F, D>>::num_live_ring_elems(*poly)
                            != num_live_positions
                            || <P as crate::compute::RootPolyShape<F, D>>::num_vars(*poly)
                                != source_num_vars
                    }) || num_live_positions.div_ceil(num_positions_per_block) != num_live_blocks
                    {
                        return Err(AkitaError::InvalidInput(
                            "coefficient-packing source shape disagrees within its group".into(),
                        ));
                    }
                    let point = PreparedSubringCoefficientPackingPoint::new(
                        geometry,
                        basis,
                        num_live_positions,
                        num_positions_per_block,
                        source_num_vars,
                        protocol_point,
                    )?;
                    let batch =
                        <P as crate::compute::RootOpeningSource<F, D>>::opening_batch(polys)?;
                    let partials_by_claim =
                        SubringCoefficientPackingBatchKernel::coefficient_packing_partials_batch(
                            ctx.backend(),
                            Some(ctx.prepared()),
                            batch,
                            SubringCoefficientPackingPlan { point: &point },
                        )?;
                    if partials_by_claim.len() != polys.len() {
                        return Err(AkitaError::InvalidSize {
                            expected: polys.len(),
                            actual: partials_by_claim.len(),
                        });
                    }
                    let scalar_openings = partials_by_claim
                        .iter()
                        .map(|partials| {
                            coefficient_packing_scalar_opening::<F, E>(
                                geometry,
                                point.num_live_blocks(),
                                std::slice::from_ref(partials),
                                &[E::one()],
                                point.live_block_weights(),
                                point.tail_weights(),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    return Ok(PreparedGroupOpening {
                        kind: OpeningFamily::SubringCoefficientPacking(
                            PreparedCoefficientPackingGroup {
                                point,
                                partials_by_claim,
                            },
                        ),
                        scalar_openings,
                    });
                }
                let (point, (folded_rings, folded_by_claim)) =
                    prepare_and_evaluate_opening_group::<F, E, P, B, D>(
                        ctx.backend(),
                        Some(ctx.prepared()),
                        self.polynomial_refs(),
                        protocol_point,
                        basis,
                        num_positions_per_block,
                        num_live_blocks,
                        alpha_bits,
                    )?;
                let inner_point = &protocol_point[..protocol_point.len().min(alpha_bits)];
                let scalar_openings = folded_rings
                    .iter()
                    .map(|folded_ring| {
                        scalar_opening_from_folded_ring::<F, E, D>(
                            folded_ring,
                            &point,
                            inner_point,
                            basis,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok::<_, AkitaError>(PreparedGroupOpening {
                    kind: OpeningFamily::EvaluationTrace(PreparedEvaluationTraceGroup {
                        point,
                        folded_by_claim: folded_by_claim
                            .iter()
                            .map(|rows| RingVec::from_ring_elems(rows).into_compact())
                            .collect(),
                    }),
                    scalar_openings,
                })
            }
        )
    }

    fn probe_fold(
        &self,
        ctx: &OperationCtx<'_, F, B>,
        challenges: &crate::protocol::fold_grind::GroupFoldChallenges,
        root_params: &CommittedGroupParams,
        params: &akita_types::GroupOpenPhaseParams,
    ) -> Result<crate::protocol::fold_grind::FoldProbeOutput<F>, AkitaError> {
        let ring_dimension = params.inner_commit_matrix_params().ring_dimension();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| {
                let point_indices = (0..self.num_polynomials()).collect::<Vec<_>>();
                let (witness, coefficients) =
                    crate::protocol::fold_grind::fold_probe_witness_kernel::<F, P, B, D>(
                        ctx.backend(),
                        Some(ctx.prepared()),
                        challenges.ambient_a(),
                        self.polynomial_refs(),
                        &point_indices,
                        root_params,
                        params,
                    )?;
                Ok::<_, AkitaError>(crate::protocol::fold_grind::FoldProbeOutput {
                    witness,
                    coefficients,
                    challenges: challenges.clone(),
                })
            }
        )
    }
}

impl<F, E, P, B> RootProverGroupTensor<F, E, B> for PreparedProverGroup<'_, P>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F>,
    E: ExtField<F> + FpExtEncoding<F> + MulBaseUnreduced<F>,
    P: RuntimeRootProvePoly<F> + RuntimeTensorSource<F>,
    B: ComputeBackendSetup<F> + RuntimeTensorBackendFor<F, P, E>,
{
    fn prepare_extension_opening(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
        point: &[E],
    ) -> Result<PreparedExtensionOpeningGroup<E>, AkitaError> {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| prepare_extension_opening_group::<F, E, P, B, D>(
                backend,
                prepared,
                self.polynomial_refs(),
                point,
            )
        )
    }

    fn extension_opening_terms(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
        claim_coefficients: &[E],
        tail_point: &[E],
        eta: &[E],
    ) -> Result<Vec<ExtensionOpeningReductionTerm<E>>, AkitaError> {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| build_extension_opening_reduction_terms::<F, E, P, B, D>(
                backend,
                prepared,
                self.polynomial_refs(),
                claim_coefficients,
                tail_point,
                eta,
            )
        )
    }
}
