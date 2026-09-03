#[cfg(test)]
use super::fold::fold_onehot_block_ring;
use super::fold::{fold_onehot_block, fold_onehot_block_subfield};
use super::*;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, ComputeBackendSetup, CpuBackend,
    DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel,
    OpeningFoldOutput, OpeningFoldPlan, RootCommitKernel, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};

/// Borrowed single-polynomial view over one-hot chunk storage.
///
/// One view type backs the commit and opening-fold kernels; the kernel trait it
/// is passed to selects the operation. `D` is the kernel
/// dispatch dimension: the underlying polynomial stores flat logical data,
/// and the view fixes the ring dimension the kernels operate at.
#[derive(Debug, Clone, Copy)]
pub struct OneHotView<'a, F: Field, const D: usize, I: OneHotIndex = usize> {
    pub(super) poly: &'a OneHotPoly<F, I>,
}

impl<'a, F: Field, const D: usize, I: OneHotIndex> OneHotView<'a, F, D, I> {
    /// Per-chunk hot positions. `None` denotes an all-zero chunk.
    pub fn indices(&self) -> &'a [Option<I>] {
        &self.poly.indices
    }

    /// Number of field-evaluation slots in each one-hot chunk.
    pub fn onehot_k(&self) -> usize {
        self.poly.onehot_k
    }

    /// Number of variables in the logical multilinear polynomial.
    pub fn num_vars(&self) -> usize {
        self.poly.num_vars
    }
}

/// Same-point batch view over several one-hot polynomials.
///
/// `D` is the kernel dispatch dimension, as in [`OneHotView`].
#[derive(Debug, Clone, Copy)]
pub struct OneHotBatchView<'a, F: Field, const D: usize, I: OneHotIndex = usize> {
    polys: &'a [&'a OneHotPoly<F, I>],
}

impl<'a, F: Field, const D: usize, I: OneHotIndex> OneHotBatchView<'a, F, D, I> {
    /// Validated semantic views in source order.
    pub fn views(&self) -> impl ExactSizeIterator<Item = OneHotView<'a, F, D, I>> + '_ {
        self.polys.iter().map(|&poly| OneHotView { poly })
    }
}

impl<F: Field, I: OneHotIndex> OneHotPoly<F, I> {
    fn source_view<const D: usize>(&self) -> Result<OneHotView<'_, F, D, I>, AkitaError> {
        self.validate_ring_dimension(D)?;
        Ok(OneHotView { poly: self })
    }
}

impl<F, I> RootPolyMeta<F> for OneHotPoly<F, I>
where
    F: Field,
    I: OneHotIndex,
{
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.onehot_k)
    }
}

impl<F, const D: usize, I> RootPolyShape<F, D> for OneHotPoly<F, I>
where
    F: Field,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        (1usize << self.num_vars).div_ceil(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.onehot_k)
    }
}

impl<F, const D: usize, I> RootCommitSource<F, D> for OneHotPoly<F, I>
where
    F: Field,
    I: OneHotIndex,
{
    type CommitView<'a>
        = OneHotView<'a, F, D, I>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        self.source_view()
    }

    /// A unit one-hot source stores hot *positions*, so every coefficient it
    /// commits is `0` or `1` by construction and no scan is possible or needed.
    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: jolt_field::CanonicalEncoding,
    {
        Ok((0, 1))
    }
}

impl<F, const D: usize, I> RootOpeningSource<F, D> for OneHotPoly<F, I>
where
    F: Field,
    I: OneHotIndex,
{
    type OpeningView<'a>
        = OneHotView<'a, F, D, I>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = OneHotBatchView<'a, F, D, I>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        self.source_view()
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        for poly in polys {
            poly.validate_ring_dimension(D)?;
        }
        Ok(OneHotBatchView { polys })
    }
}

impl<F, const D: usize, I> RootCommitKernel<OneHotView<'_, F, D, I>, F, D> for CpuBackend
where
    F: Field + CanonicalEncoding + Unreduced + WithCommitAccumulator,
    I: OneHotIndex,
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<OneHotView<'_, F, D, I>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        let active_a_cols = plan
            .num_positions_per_block
            .checked_mul(plan.num_digits_inner)
            .ok_or_else(|| AkitaError::InvalidSetup("active A width overflow".into()))?;
        let a_view = self
            .prepared_expanded_setup(prepared)
            .shared_matrix
            .ring_view::<D>(plan.n_a, active_a_cols)?;
        let rows = column_sweep_ajtai_onehot_multi::<F, D, I>(
            &a_view,
            &sources,
            plan.n_a,
            active_a_cols,
            plan.num_digits_inner,
            self.commit_scratch_bytes_per_worker(),
        )?;
        Ok(rows
            .into_iter()
            .map(CommitInnerWitness::from_rows::<D>)
            .collect())
    }
}

impl<F, const D: usize, I> OpeningFoldKernel<OneHotView<'_, F, D, I>, F, D> for CpuBackend
where
    F: Field + CanonicalEncoding + Unreduced,
    I: OneHotIndex,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotView<'_, F, D, I>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        // Count-only validation: building (and caching) every block here
        // would defeat the lazy per-block folds below.
        let num_live_blocks = source
            .poly
            .num_live_blocks_for(D, plan.num_positions_per_block())?;
        plan.validate::<D>(num_live_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.poly.evaluate_and_fold::<D>(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            ),
            OpeningFoldPlan::Subfield {
                multipliers,
                num_positions_per_block,
            } => source
                .poly
                .evaluate_and_fold_subfield::<D>(multipliers, num_positions_per_block)?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotView<'_, F, D, I>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        Ok(source.poly.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        ))
    }
}

impl<F, const D: usize, I> OpeningBatchKernel<OneHotBatchView<'_, F, D, I>, F, D> for CpuBackend
where
    F: Field + CanonicalEncoding + Unreduced,
    I: OneHotIndex,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotBatchView<'_, F, D, I>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        let DecomposeFoldBatchPlan::Sparse {
            challenges,
            num_positions_per_block,
            num_digits,
            log_basis,
        } = plan;
        match OneHotPoly::decompose_fold_batched::<D>(
            source.polys,
            challenges,
            num_positions_per_block,
            num_digits,
            log_basis,
        ) {
            Some(witness) => Ok(BatchDecomposeFoldOutcome::Fused(witness)),
            None => Ok(BatchDecomposeFoldOutcome::FallbackPerPoly),
        }
    }
}

pub(in crate::backend) trait PackingWeightAccessor<E: Field>: Sync {
    fn point(&self) -> &akita_types::PreparedSubringCoefficientPackingPoint<E>;

    fn weight(&self, position: usize, low_index: usize) -> Result<E, AkitaError>;
}

struct DirectPackingWeights<'a, E: Field> {
    point: &'a akita_types::PreparedSubringCoefficientPackingPoint<E>,
}

impl<E: Field> PackingWeightAccessor<E> for DirectPackingWeights<'_, E> {
    fn point(&self) -> &akita_types::PreparedSubringCoefficientPackingPoint<E> {
        self.point
    }

    #[inline(always)]
    fn weight(&self, position: usize, low_index: usize) -> Result<E, AkitaError> {
        let position_weight = *self
            .point
            .position_weights()
            .get(position)
            .ok_or(AkitaError::InvalidProof)?;
        let packing_weight = *self
            .point
            .packing_weights()
            .get(low_index)
            .ok_or(AkitaError::InvalidProof)?;
        Ok(position_weight * packing_weight)
    }
}

impl<E: Field> PackingWeightAccessor<E>
    for crate::backend::coefficient_packing::FusedPackingWeights<'_, E>
{
    fn point(&self) -> &akita_types::PreparedSubringCoefficientPackingPoint<E> {
        self.point()
    }

    #[inline(always)]
    fn weight(&self, position: usize, low_index: usize) -> Result<E, AkitaError> {
        let index = akita_error::checked::mul_add(
            position,
            self.point().geometry().subring_embedding_stride(),
            low_index,
        )
        .ok_or(AkitaError::InvalidProof)?;
        self.values()
            .get(index)
            .copied()
            .ok_or(AkitaError::InvalidProof)
    }
}

#[tracing::instrument(skip_all, name = "coefficient_packing_onehot_partials")]
pub(in crate::backend) fn onehot_coefficient_packing_partials<F, E, const D: usize, I, W>(
    source: OneHotBatchView<'_, F, D, I>,
    weights: &W,
) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + jolt_field::MulBaseUnreduced<F>,
    I: OneHotIndex,
    W: PackingWeightAccessor<E>,
{
    let point = weights.point();
    let plan = SubringCoefficientPackingPlan { point };
    for poly in source.polys {
        plan.validate::<D>(RootPolyMeta::<F>::num_vars(*poly))?;
    }
    let geometry = point.geometry();
    if E::DEGREE != geometry.extension_degree() || D != geometry.a_ring_dimension() {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing field or ring dimension mismatch".into(),
        ));
    }
    let expected_field_len = point.num_live_positions().checked_mul(D).ok_or_else(|| {
        AkitaError::InvalidInput("coefficient-packing one-hot length overflow".into())
    })?;
    let output_len = point
        .num_live_blocks()
        .checked_mul(geometry.partial_base_field_width())
        .ok_or_else(|| {
            AkitaError::InvalidInput("coefficient-packing output length overflow".into())
        })?;
    let stride = geometry.subring_embedding_stride();
    let stride_mask = stride - 1;
    let stride_shift = stride.trailing_zeros();
    let ring_mask = D - 1;
    let ring_shift = D.trailing_zeros();
    let s = geometry.challenge_subring_dimension();
    let num_blocks = point.num_live_blocks();

    source
        .polys
        .iter()
        .map(|poly| {
            let actual_field_len = poly
                .indices
                .len()
                .checked_mul(poly.onehot_k)
                .ok_or_else(|| AkitaError::InvalidInput("one-hot source length overflow".into()))?;
            // One-hot roots authenticate their complete Boolean domain.
            // Unlike recursive witness storage, they cannot discard a
            // padded suffix merely because the opening plan names a live
            // prefix.
            if actual_field_len != expected_field_len {
                return Err(AkitaError::InvalidSize {
                    expected: expected_field_len,
                    actual: actual_field_len,
                });
            }
            if poly.onehot_k == 0 {
                return Err(AkitaError::InvalidSetup(
                    "coefficient-packing one-hot chunk size must be nonzero".into(),
                ));
            }
            debug_assert!(poly.onehot_k.is_power_of_two());
            let block_coordinates = cfg_into_iter!(0..num_blocks)
                .map(|block_index| {
                    let first_position = block_index
                        .checked_mul(point.num_positions_per_block())
                        .ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "coefficient-packing one-hot block offset overflow".into(),
                        )
                    })?;
                    let end_position = first_position
                        .checked_add(point.num_positions_per_block())
                        .ok_or_else(|| {
                            AkitaError::InvalidInput(
                                "coefficient-packing one-hot block end overflow".into(),
                            )
                        })?
                        .min(point.num_live_positions());
                    let first_field = first_position.checked_mul(D).ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "coefficient-packing one-hot field offset overflow".into(),
                        )
                    })?;
                    let end_field = end_position.checked_mul(D).ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "coefficient-packing one-hot field end overflow".into(),
                        )
                    })?;
                    let first_chunk = first_field / poly.onehot_k;
                    let end_chunk = end_field.div_ceil(poly.onehot_k).min(poly.indices.len());
                    let mut block = vec![F::zero(); geometry.partial_base_field_width()];
                    for chunk_index in first_chunk..end_chunk {
                        let Some(hot_index) = poly
                            .indices
                            .get(chunk_index)
                            .copied()
                            .ok_or(AkitaError::InvalidProof)?
                        else {
                            continue;
                        };
                        let field_index = chunk_index
                            .checked_mul(poly.onehot_k)
                            .and_then(|base| base.checked_add(hot_index.as_usize()))
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("one-hot source index overflow".into())
                            })?;
                        if field_index < first_field || field_index >= end_field {
                            continue;
                        }
                        let position_in_block = (field_index >> ring_shift) - first_position;
                        let coefficient_index = field_index & ring_mask;
                        let low_index = coefficient_index & stride_mask;
                        let subring_index = coefficient_index >> stride_shift;
                        let value = weights.weight(position_in_block, low_index)?;
                        let extension_coordinates = value.ext_coords();
                        if extension_coordinates.len() != geometry.extension_degree() {
                            return Err(AkitaError::InvalidSetup(
                                "coefficient-packing extension encoding width mismatch".into(),
                            ));
                        }
                        for (coordinate_block, coordinate) in block
                            .chunks_exact_mut(s)
                            .zip(extension_coordinates.iter().copied())
                        {
                            *coordinate_block
                                .get_mut(subring_index)
                                .ok_or(AkitaError::InvalidProof)? += coordinate;
                        }
                    }
                    Ok(block)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let mut coordinates = Vec::new();
            coordinates.try_reserve_exact(output_len).map_err(|_| {
                AkitaError::InvalidInput(
                    "coefficient-packing one-hot output allocation failed".into(),
                )
            })?;
            for block in block_coordinates {
                coordinates.extend(block);
            }
            SubringCoefficientPackingPartials::new(geometry, point.num_live_blocks(), coordinates)
        })
        .collect()
}

impl<F, E, const D: usize, I>
    SubringCoefficientPackingBatchKernel<OneHotBatchView<'_, F, D, I>, F, E, D> for CpuBackend
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + jolt_field::MulBaseUnreduced<F>,
    I: OneHotIndex,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotBatchView<'_, F, D, I>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_len =
            crate::backend::coefficient_packing::FusedPackingWeights::<E>::required_len(
                plan.point,
            )?;
        let should_prepare = source
            .polys
            .iter()
            .flat_map(|poly| poly.indices.iter())
            .filter(|index| index.is_some())
            .take(fused_len)
            .count()
            == fused_len;
        if should_prepare {
            let weights =
                crate::backend::coefficient_packing::FusedPackingWeights::new(plan.point)?;
            onehot_coefficient_packing_partials(source, &weights)
        } else {
            let weights = DirectPackingWeights { point: plan.point };
            onehot_coefficient_packing_partials(source, &weights)
        }
    }
}

impl<F, I: OneHotIndex> OneHotPoly<F, I>
where
    F: Field + CanonicalEncoding + Unreduced,
{
    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let (num_rings, num_live_blocks) = self
            .view_layout(D, num_positions_per_block)
            .expect("valid one hot fold layout");
        cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let ring_start = block_idx * num_positions_per_block;
                let ring_end = (ring_start + num_positions_per_block).min(num_rings);
                fold_onehot_block::<F, I, D>(self, ring_start..ring_end, scalars)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let num_live_blocks = self
            .num_live_blocks_for(D, num_positions_per_block)
            .expect("valid one hot fold layout");
        cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let materialized = self
                    .materialize_block_range(D, num_positions_per_block, block_idx..block_idx + 1)
                    .expect("in-range single block build");
                fold_onehot_block_ring(materialized.block(0), scalars, num_positions_per_block)
            })
            .collect()
    }

    pub(crate) fn fold_blocks_subfield<const D: usize>(
        &self,
        multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        let (num_rings, num_live_blocks) = self.view_layout(D, num_positions_per_block)?;
        cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let ring_start = block_idx * num_positions_per_block;
                let ring_end = (ring_start + num_positions_per_block).min(num_rings);
                fold_onehot_block_subfield(self, ring_start..ring_end, multipliers)
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold<const D: usize>(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks::<D>(position_weights, num_positions_per_block),
            live_block_weights,
        )
    }

    pub(crate) fn evaluate_and_fold_subfield<const D: usize>(
        &self,
        multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        crate::backend::poly_helpers::fused_evaluate_and_fold_subfield(
            self.fold_blocks_subfield::<D>(multipliers, num_positions_per_block)?,
            multipliers,
        )
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F> {
        self.view_layout(D, num_positions_per_block)
            .expect("OneHotPoly::decompose_fold: invalid block layout");
        Self::decompose_fold_batched_onehot::<D>(
            &[self],
            challenges,
            num_positions_per_block,
            num_digits,
        )
        .unwrap_or_else(|| {
            super::decompose_fold::finish_decompose_fold(
                vec![[0i32; D]; num_positions_per_block],
                num_digits,
            )
        })
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_batched")]
    pub(crate) fn decompose_fold_batched<const D: usize>(
        polys: &[&Self],
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Option<DecomposeFoldWitness<F>> {
        let first = polys.first()?;
        first
            .num_live_blocks_for(D, num_positions_per_block)
            .expect(
            "OneHotPoly::decompose_fold_batched: invalid num_positions_per_block for first polynomial",
        );
        Self::decompose_fold_batched_onehot::<D>(
            polys,
            challenges,
            num_positions_per_block,
            num_digits,
        )
    }
}
