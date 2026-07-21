use super::fold::{fold_onehot_block, fold_onehot_block_ring};
use super::*;
use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    BatchDecomposeFoldOutcome, CommitInnerPlan, CpuBackend, DecomposeFoldBatchPlan,
    DecomposeFoldPlan, OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan,
    RootCommitKernel, RootCommitSource, RootOpeningSource, RootPolyMeta, RootPolyShape,
    RootTensorSource, TensorPackedWitness, TensorProjectionBatchKernel, TensorProjectionKernel,
};
use akita_field::MulBaseUnreduced;

/// Inner (low) coordinate count for the factorized one-hot column-partials
/// fast path. The high opening coordinates split into `inner_bits` low bits
/// (a small, reusable, cache-resident `low_eq` table) and the remaining high
/// bits (the parallelizable outer blocks). This is a pure performance knob:
/// the computed partials are independent of its value.
const ONEHOT_TENSOR_PARTIALS_INNER_BITS: usize = 12;

/// Borrowed single-polynomial view over one-hot chunk storage.
///
/// One view type backs the commit, opening-fold, and tensor-projection kernels;
/// the kernel trait it is passed to selects the operation. `D` is the kernel
/// dispatch dimension: the underlying polynomial stores flat logical data,
/// and the view fixes the ring dimension the kernels operate at.
#[derive(Debug, Clone, Copy)]
pub struct OneHotView<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    poly: &'a OneHotPoly<F, I>,
}

/// Same-point batch view over several one-hot polynomials.
///
/// `D` is the kernel dispatch dimension, as in [`OneHotView`].
#[derive(Debug, Clone, Copy)]
pub struct OneHotBatchView<'a, F: FieldCore, const D: usize, I: OneHotIndex = usize> {
    polys: &'a [&'a OneHotPoly<F, I>],
}

impl<F, I> RootPolyMeta<F> for OneHotPoly<F, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    fn num_ring_elems(&self) -> usize {
        self.total_ring_elems
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn onehot_chunk_size(&self) -> Option<usize> {
        Some(self.onehot_k)
    }
}

impl<F, const D: usize, I> RootPolyShape<F, D> for OneHotPoly<F, I>
where
    F: FieldCore,
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
    F: FieldCore,
    I: OneHotIndex,
{
    type CommitView<'a>
        = OneHotView<'a, F, D, I>
    where
        Self: 'a;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        Ok(OneHotView { poly: self })
    }
}

impl<F, const D: usize, I> RootOpeningSource<F, D> for OneHotPoly<F, I>
where
    F: FieldCore,
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
        Ok(OneHotView { poly: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        Ok(OneHotBatchView { polys })
    }
}

impl<F, const D: usize, I> RootTensorSource<F, D> for OneHotPoly<F, I>
where
    F: FieldCore,
    I: OneHotIndex,
{
    type TensorView<'a>
        = OneHotView<'a, F, D, I>
    where
        Self: 'a;

    type TensorBatchView<'a>
        = OneHotBatchView<'a, F, D, I>
    where
        Self: 'a;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        Ok(OneHotView { poly: self })
    }

    fn tensor_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::TensorBatchView<'a>, AkitaError> {
        Ok(OneHotBatchView { polys })
    }
}

impl<F, const D: usize, I> RootCommitKernel<OneHotView<'_, F, D, I>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn commit_inner(
        &self,
        prepared: &Self::PreparedSetup,
        source: OneHotView<'_, F, D, I>,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError> {
        source.poly.commit_inner::<_, D>(self, prepared, plan)
    }
}

impl<F, const D: usize, I> OpeningFoldKernel<OneHotView<'_, F, D, I>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotView<'_, F, D, I>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let blocks = source.poly.blocks_for(D, plan.num_positions_per_block())?;
        plan.validate(blocks.num_live_blocks())?;
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
            OpeningFoldPlan::Ring {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.poly.evaluate_and_fold_ring::<D>(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            ),
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
    F: FieldCore + CanonicalField + HasWide,
    I: OneHotIndex,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotBatchView<'_, F, D, I>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        match plan {
            DecomposeFoldBatchPlan::Sparse {
                challenges,
                num_positions_per_block,
                num_digits,
                log_basis,
            } => match OneHotPoly::decompose_fold_batched::<D>(
                source.polys,
                challenges,
                num_positions_per_block,
                num_digits,
                log_basis,
            ) {
                Some(witness) => Ok(BatchDecomposeFoldOutcome::Fused(witness)),
                None => Ok(BatchDecomposeFoldOutcome::FallbackPerPoly),
            },
            DecomposeFoldBatchPlan::Tensor {
                tensor,
                num_positions_per_block,
                num_digits,
                log_basis,
            } => match OneHotPoly::decompose_fold_tensor_batched::<D>(
                source.polys,
                tensor,
                num_positions_per_block,
                num_digits,
                log_basis,
            )? {
                Some(witness) => Ok(BatchDecomposeFoldOutcome::Fused(witness)),
                None => Ok(BatchDecomposeFoldOutcome::Unsupported),
            },
        }
    }
}

impl<F, E, const D: usize, I> TensorProjectionKernel<OneHotView<'_, F, D, I>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt + HasWide,
    E: ExtField<F>,
    I: OneHotIndex,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotView<'_, F, D, I>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source.poly.tensor_extension_column_partials(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotView<'_, F, D, I>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(match source.poly.tensor_packed_extension_sparse_evals()? {
            Some(witness) => TensorPackedWitness::Sparse(witness),
            None => TensorPackedWitness::Dense(source.poly.tensor_packed_extension_evals()?),
        })
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotView<'_, F, D, I>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        source.poly.tensor_packed_extension_root_poly::<E, D>()
    }
}

impl<F, E, const D: usize, I> TensorProjectionBatchKernel<OneHotBatchView<'_, F, D, I>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + HasWide,
    E: ExtField<F>,
    I: OneHotIndex,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotBatchView<'_, F, D, I>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        OneHotPoly::tensor_extension_column_partials_batch(source.polys, logical_point)
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: OneHotBatchView<'_, F, D, I>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        OneHotPoly::tensor_packed_extension_sparse_linear_combination(source.polys, coeffs)
    }
}

impl<F, I: OneHotIndex> OneHotPoly<F, I>
where
    F: FieldCore + CanonicalField + HasWide,
{
    pub(crate) fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self
            .blocks_for(D, num_positions_per_block)
            .expect("OneHotPoly::fold_blocks: invalid num_positions_per_block for this polynomial");
        let num_live_blocks = blocks.num_live_blocks();
        match blocks.as_ref() {
            OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_live_blocks)
                .map(|i| fold_onehot_block(flat.block(i), scalars, num_positions_per_block))
                .collect(),
            OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_live_blocks)
                .map(|i| fold_onehot_block(flat.block(i), scalars, num_positions_per_block))
                .collect(),
        }
    }

    pub(crate) fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let blocks = self.blocks_for(D, num_positions_per_block).expect(
            "OneHotPoly::fold_blocks_ring: invalid num_positions_per_block for this polynomial",
        );
        let num_live_blocks = blocks.num_live_blocks();
        match blocks.as_ref() {
            OneHotBlocks::SingleChunk(flat) => cfg_into_iter!(0..num_live_blocks)
                .map(|i| fold_onehot_block_ring(flat.block(i), scalars, num_positions_per_block))
                .collect(),
            OneHotBlocks::MultiChunk(flat) => cfg_into_iter!(0..num_live_blocks)
                .map(|i| fold_onehot_block_ring(flat.block(i), scalars, num_positions_per_block))
                .collect(),
        }
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

    pub(crate) fn evaluate_and_fold_ring<const D: usize>(
        &self,
        live_block_weights: &[CyclotomicRing<F, D>],
        position_weights: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>) {
        crate::backend::poly_helpers::fused_evaluate_and_fold_ring(
            self.fold_blocks_ring::<D>(position_weights, num_positions_per_block),
            live_block_weights,
        )
    }

    pub(crate) fn evaluate_extension<E>(&self, point: &[E]) -> Result<E, AkitaError>
    where
        E: ExtField<F>,
    {
        if point.len() != self.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.num_vars,
                actual: point.len(),
            });
        }
        let low_vars = self.onehot_k.trailing_zeros() as usize;
        if low_vars > point.len() {
            return Err(AkitaError::InvalidPointDimension {
                expected: low_vars,
                actual: point.len(),
            });
        }
        let low_weights =
            akita_types::basis_weights(&point[..low_vars], akita_types::BasisMode::Lagrange)?;
        let high_weights =
            akita_types::basis_weights(&point[low_vars..], akita_types::BasisMode::Lagrange)?;
        Ok(self
            .indices
            .iter()
            .enumerate()
            .filter_map(|(chunk_idx, hot_idx)| {
                hot_idx.map(|hot_idx| high_weights[chunk_idx] * low_weights[hot_idx.as_usize()])
            })
            .fold(E::zero(), |acc, weight| acc + weight))
    }

    pub(crate) fn tensor_extension_column_partials<E>(
        &self,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        if logical_point.len() != self.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: self.num_vars,
                actual: logical_point.len(),
            });
        }
        let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
        if split_bits > self.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        let low_vars = self.onehot_k.trailing_zeros() as usize;
        if low_vars > logical_point.len() {
            return Err(AkitaError::InvalidPointDimension {
                expected: low_vars,
                actual: logical_point.len(),
            });
        }
        if split_bits <= low_vars {
            let head_mask = width - 1;
            let low_tail_weights = akita_types::basis_weights(
                &logical_point[split_bits..low_vars],
                akita_types::BasisMode::Lagrange,
            )?;
            let high_weights = akita_types::basis_weights(
                &logical_point[low_vars..],
                akita_types::BasisMode::Lagrange,
            )?;
            let mut partials = vec![E::zero(); width];
            for (chunk_idx, hot_idx) in self.indices.iter().copied().enumerate() {
                let Some(raw) = hot_idx else {
                    continue;
                };
                let raw = raw.as_usize();
                let head = raw & head_mask;
                let low_tail = raw >> split_bits;
                partials[head] += high_weights[chunk_idx] * low_tail_weights[low_tail];
            }
            return Ok(partials);
        }

        let mut point = logical_point.to_vec();
        let mut partials = Vec::with_capacity(width);
        for head in 0..width {
            for (bit, coord) in point.iter_mut().enumerate().take(split_bits) {
                *coord = if ((head >> bit) & 1) == 0 {
                    E::zero()
                } else {
                    E::one()
                };
            }
            partials.push(self.evaluate_extension::<E>(&point)?);
        }
        Ok(partials)
    }

    pub(crate) fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let Some(first) = polys.first() else {
            return Ok(Vec::new());
        };
        if logical_point.len() != first.num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: first.num_vars,
                actual: logical_point.len(),
            });
        }
        let (split_bits, width) = akita_types::tensor_opening_split::<F, E>()?;
        if split_bits > first.num_vars {
            return Err(AkitaError::InvalidInput(
                "extension-opening tensor split exceeds polynomial arity".to_string(),
            ));
        }
        let _span = tracing::info_span!(
            "onehot_tensor_extension_column_partials_batch",
            num_polys = polys.len(),
            num_vars = first.num_vars,
            onehot_k = first.onehot_k,
            split_bits,
            width
        )
        .entered();
        let low_vars = first.onehot_k.trailing_zeros() as usize;
        let can_share_weights = split_bits <= low_vars
            && low_vars <= logical_point.len()
            && polys
                .iter()
                .all(|poly| poly.num_vars == first.num_vars && poly.onehot_k == first.onehot_k);
        if !can_share_weights {
            return polys
                .iter()
                .map(|poly| poly.tensor_extension_column_partials(logical_point))
                .collect();
        }

        // Non-power-of-two `onehot_k` would let `raw >> split_bits` escape
        // `low_tail_weights`; the sparse fast path only covers the (production)
        // power-of-two case, so fall back to the per-poly reference otherwise.
        if !first.onehot_k.is_power_of_two() {
            return polys
                .iter()
                .map(|poly| poly.tensor_extension_column_partials(logical_point))
                .collect();
        }

        // Factor the high `hi_vars = num_vars - low_vars` opening coordinates
        // into an inner/outer tensor split so the chunk weight
        // `high_weights[(j << inner_bits) | i] == high_eq[j] * low_eq[i]`. This
        // avoids materializing the full `2^hi_vars` weight table and lets the
        // sparse kernel turn the per-chunk multiply into a per-chunk add.
        let hi_vars = first.num_vars - low_vars;
        let inner_bits = hi_vars.min(ONEHOT_TENSOR_PARTIALS_INNER_BITS);
        let low_tail_weights = akita_types::basis_weights(
            &logical_point[split_bits..low_vars],
            akita_types::BasisMode::Lagrange,
        )?;
        let low_eq = akita_types::basis_weights(
            &logical_point[low_vars..low_vars + inner_bits],
            akita_types::BasisMode::Lagrange,
        )?;
        let high_eq = akita_types::basis_weights(
            &logical_point[low_vars + inner_bits..],
            akita_types::BasisMode::Lagrange,
        )?;
        let out = polys
            .iter()
            .map(|poly| {
                poly.tensor_column_partials_from_shared_eq::<E>(
                    split_bits,
                    width,
                    inner_bits,
                    &low_eq,
                    &high_eq,
                    &low_tail_weights,
                )
            })
            .collect::<Vec<_>>();
        Ok(out)
    }

    pub(crate) fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::ExtField<F>,
    {
        let field_elems = self.direct_field_evals()?;
        akita_types::tensor_packed_witness_evals::<F, E>(self.num_vars, &field_elems)
    }

    pub(crate) fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        Ok(Some(self.tensor_packed_sparse_witness::<E>()?))
    }

    pub(crate) fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError>
    where
        E: ExtField<F>,
    {
        let _span = tracing::info_span!(
            "OneHotPoly::tensor_packed_sparse_witness_linear_combination",
            num_polys = polys.len()
        )
        .entered();
        if polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: polys.len(),
                actual: coeffs.len(),
            });
        }
        let first = polys.first().ok_or_else(|| {
            AkitaError::InvalidInput(
                "onehot sparse witness linear combination requires at least one polynomial"
                    .to_string(),
            )
        })?;
        let (width, total_evals) = first.tensor_packing_shape::<E>()?;
        let table_len = total_evals / width;
        let basis = (0..width)
            .map(|head| {
                let mut coords = vec![F::zero(); width];
                coords[head] = F::one();
                E::from_base_slice(&coords)
            })
            .collect::<Vec<_>>();
        let weighted_basis = coeffs
            .iter()
            .copied()
            .map(|coeff| {
                basis
                    .iter()
                    .copied()
                    .map(|basis_elem| basis_elem * coeff)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let capacity = polys.iter().map(|poly| poly.indices.len()).sum();
        let same_chunk_layout = polys.iter().all(|poly| {
            poly.onehot_k == first.onehot_k && poly.indices.len() == first.indices.len()
        });
        if same_chunk_layout && first.onehot_k >= width && first.onehot_k.is_multiple_of(width) {
            let tails_per_chunk = first.onehot_k / width;
            #[cfg(feature = "parallel")]
            let target_ranges = rayon::current_num_threads().max(1) * 4;
            #[cfg(not(feature = "parallel"))]
            let target_ranges = 1usize;
            let range_len = (first.indices.len() / target_ranges).max(1 << 12);
            let ranges = (0..first.indices.len())
                .step_by(range_len)
                .map(|start| (start, (start + range_len).min(first.indices.len())))
                .collect::<Vec<_>>();
            let chunks = cfg_into_iter!(ranges)
                .map(|(start, end)| {
                    let mut entries = Vec::with_capacity((end - start) * polys.len());
                    let mut local = Vec::with_capacity(polys.len());
                    for chunk_idx in start..end {
                        local.clear();
                        for (poly_idx, poly) in polys.iter().enumerate() {
                            if coeffs[poly_idx] == E::zero() {
                                continue;
                            }
                            let Some(raw) = poly.indices[chunk_idx] else {
                                continue;
                            };
                            let field_pos = poly.hot_field_position(
                                chunk_idx,
                                raw,
                                "tensor-packed witness batch",
                            )?;
                            let tail = field_pos / width;
                            let head = field_pos % width;
                            debug_assert_eq!(tail / tails_per_chunk, chunk_idx);
                            local.push((tail % tails_per_chunk, weighted_basis[poly_idx][head]));
                        }
                        local.sort_unstable_by_key(|(local_tail, _)| *local_tail);
                        for &(local_tail, value) in &local {
                            let tail = chunk_idx * tails_per_chunk + local_tail;
                            if let Some((last_tail, last_value)) = entries.last_mut() {
                                if *last_tail == tail {
                                    *last_value += value;
                                    if *last_value == E::zero() {
                                        entries.pop();
                                    }
                                    continue;
                                }
                            }
                            if value != E::zero() {
                                entries.push((tail, value));
                            }
                        }
                    }
                    Ok(entries)
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            let mut entries = Vec::with_capacity(capacity);
            for chunk in chunks {
                entries.extend(chunk);
            }
            return Ok(Some(
                SparseExtensionOpeningWitness::from_sorted_unique_entries(table_len, entries)?,
            ));
        }

        let mut cursors = vec![0usize; polys.len()];
        let mut next_entries = Vec::with_capacity(polys.len());
        for (poly_idx, (poly, &coeff)) in polys.iter().zip(coeffs).enumerate() {
            let (poly_width, poly_total_evals) = poly.tensor_packing_shape::<E>()?;
            if poly_width != width || poly_total_evals != total_evals {
                return Err(AkitaError::InvalidSize {
                    expected: total_evals,
                    actual: poly_total_evals,
                });
            }
            if coeff == E::zero() {
                next_entries.push(None);
                continue;
            }
            next_entries
                .push(poly.next_tensor_packed_sparse_position(&mut cursors[poly_idx], width)?);
        }

        let mut entries = Vec::with_capacity(capacity);
        while let Some(tail) = next_entries
            .iter()
            .filter_map(|entry| entry.map(|(tail, _)| tail))
            .min()
        {
            let mut value = E::zero();
            for (poly_idx, poly) in polys.iter().enumerate() {
                while matches!(next_entries[poly_idx], Some((entry_tail, _)) if entry_tail == tail)
                {
                    let (_, head) = next_entries[poly_idx].expect("entry checked above");
                    value += weighted_basis[poly_idx][head];
                    next_entries[poly_idx] =
                        poly.next_tensor_packed_sparse_position(&mut cursors[poly_idx], width)?;
                }
            }
            entries.push((tail, value));
        }
        Ok(Some(SparseExtensionOpeningWitness::from_sorted_entries(
            table_len, entries,
        )?))
    }

    pub(crate) fn tensor_packed_extension_root_poly<E, const D: usize>(
        &self,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        F: CanonicalField + FromPrimitiveInt,
        E: FpExtEncoding<F>,
    {
        Ok(self.tensor_packed_sparse_ring_poly::<E, D>()?.into())
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold")]
    pub(crate) fn decompose_fold<const D: usize>(
        &self,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> DecomposeFoldWitness<F> {
        let blocks = self.blocks_for(D, num_positions_per_block).expect(
            "OneHotPoly::decompose_fold: invalid num_positions_per_block for this polynomial",
        );
        match blocks.as_ref() {
            OneHotBlocks::SingleChunk(blocks) => self.decompose_fold_onehot::<SingleChunkEntry, D>(
                blocks,
                challenges,
                num_positions_per_block,
                num_digits,
            ),
            OneHotBlocks::MultiChunk(blocks) => self.decompose_fold_onehot::<MultiChunkEntry, D>(
                blocks,
                challenges,
                num_positions_per_block,
                num_digits,
            ),
        }
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
        let first_blocks = first.blocks_for(D, num_positions_per_block).expect(
            "OneHotPoly::decompose_fold_batched: invalid num_positions_per_block for first polynomial",
        );
        match first_blocks.as_ref() {
            OneHotBlocks::SingleChunk(_) => Self::decompose_fold_batched_single_chunk_onehot::<D>(
                polys,
                challenges,
                num_positions_per_block,
                num_digits,
            ),
            OneHotBlocks::MultiChunk(_) => Self::decompose_fold_batched_multi_chunk_onehot::<D>(
                polys,
                challenges,
                num_positions_per_block,
                num_digits,
            ),
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::decompose_fold_tensor_batched")]
    pub(crate) fn decompose_fold_tensor_batched<const D: usize>(
        polys: &[&Self],
        tensor: &TensorChallengeSet,
        num_positions_per_block: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Result<Option<DecomposeFoldWitness<F>>, AkitaError> {
        Self::decompose_fold_batched_tensor_onehot::<D>(
            polys,
            tensor,
            num_positions_per_block,
            num_digits,
        )
    }

    #[tracing::instrument(skip_all, name = "OneHotPoly::commit_inner")]
    pub(crate) fn commit_inner<B, const D: usize>(
        &self,
        backend: &B,
        prepared: &B::PreparedSetup,
        plan: CommitInnerPlan,
    ) -> Result<CommitInnerWitness<F>, AkitaError>
    where
        B: CommitmentComputeBackend<F>,
    {
        let blocks = self.blocks_for(D, plan.num_positions_per_block)?;
        let t = backend.onehot_commit_rows::<D>(
            prepared,
            OneHotCommitRowsPlan {
                n_a: plan.n_a,
                num_positions_per_block: plan.num_positions_per_block,
                num_digits_inner: plan.num_digits_inner,
                blocks: blocks.commit_plan_blocks(),
            },
        )?;

        let decomposed_inner_rows = crate::kernels::linear::decompose_commit_blocks_into::<F, D>(
            &t,
            plan.num_digits_outer,
            plan.log_basis_outer,
        )?;

        CommitInnerWitness::from_parts(t, decomposed_inner_rows)
    }
}
