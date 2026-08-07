//! Whole-group root operations.

use super::*;
use crate::compute::{
    tensor_root_projection, ComputeBackendSetup, DigitRowsComputeBackend, OperationCtx,
    RuntimeOpeningProveBackendFor, RuntimeRootProvePoly, RuntimeTensorBackendFor,
};
use crate::{PreparedProverGroup, RootTensorProjectionPoly};
use akita_challenges::Challenges;
use akita_field::unreduced::ReduceTo;
use akita_field::AdditiveGroup;
use akita_types::LevelParamsLike;

pub(crate) struct PreparedGroupOpening<F: FieldCore, E: FieldCore> {
    pub(in crate::protocol) point: PreparedOpeningPoint<F, E>,
    pub(in crate::protocol) folded_by_claim: Vec<RingVec<F>>,
    pub(in crate::protocol) scalar_openings: Vec<E>,
}

pub(crate) trait RootProverGroupMeta<F: FieldCore> {
    fn num_polynomials(&self) -> usize;
    fn num_vars(&self) -> Result<usize, AkitaError>;
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
    ) -> Result<PreparedGroupOpening<F, E>, AkitaError>;

    fn probe_fold(
        &self,
        ctx: &OperationCtx<'_, F, B>,
        challenges: &Challenges,
        root_params: &CommittedGroupParams,
        params: &(impl LevelParamsLike + ?Sized),
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

    fn tensor_project(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
    ) -> Result<Vec<RootTensorProjectionPoly<F>>, AkitaError>;

    fn extension_opening_terms(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
        row_coefficients: &[E],
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
}

impl<F, E, P, B> RootProverGroupOpening<F, E, B> for PreparedProverGroup<'_, P>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide + 'static,
    <F as HasWide>::Wide: From<F> + ReduceTo<F> + AdditiveGroup,
    E: FpExtEncoding<F> + ExtField<F> + AkitaSerialize,
    P: RuntimeRootProvePoly<F>,
    B: ComputeBackendSetup<F> + DigitRowsComputeBackend<F> + RuntimeOpeningProveBackendFor<F, P>,
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
    ) -> Result<PreparedGroupOpening<F, E>, AkitaError> {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| {
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
                    point,
                    folded_by_claim: folded_by_claim
                        .iter()
                        .map(|rows| RingVec::from_ring_elems(rows).into_compact())
                        .collect(),
                    scalar_openings,
                })
            }
        )
    }

    fn probe_fold(
        &self,
        ctx: &OperationCtx<'_, F, B>,
        challenges: &Challenges,
        root_params: &CommittedGroupParams,
        params: &(impl LevelParamsLike + ?Sized),
    ) -> Result<crate::protocol::fold_grind::FoldProbeOutput<F>, AkitaError> {
        let ring_dimension = params.inner_commit_matrix_params().ring_dimension();
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| {
                let point_indices = (0..self.num_polynomials()).collect::<Vec<_>>();
                let (witness, centered) =
                    crate::protocol::fold_grind::fold_probe_witness_kernel::<F, P, B, D>(
                        ctx.backend(),
                        Some(ctx.prepared()),
                        challenges,
                        self.polynomial_refs(),
                        &point_indices,
                        root_params,
                        params,
                    )?;
                Ok::<_, AkitaError>(crate::protocol::fold_grind::FoldProbeOutput {
                    witness,
                    centered_per_chunk: centered
                        .into_iter()
                        .map(|chunk| chunk.into_iter().map(|row| row.to_vec()).collect())
                        .collect(),
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
    P: RuntimeRootProvePoly<F>,
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

    fn tensor_project(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
    ) -> Result<Vec<RootTensorProjectionPoly<F>>, AkitaError> {
        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            ring_dimension,
            |D| {
                self.polynomial_refs()
                    .iter()
                    .map(|poly| tensor_root_projection::<F, P, E, B, D>(backend, prepared, *poly))
                    .collect()
            }
        )
    }

    fn extension_opening_terms(
        &self,
        backend: &B,
        prepared: Option<&B::PreparedSetup>,
        ring_dimension: usize,
        row_coefficients: &[E],
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
                row_coefficients,
                tail_point,
                eta,
            )
        )
    }
}
