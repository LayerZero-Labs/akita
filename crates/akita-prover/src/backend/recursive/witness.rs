//! Recursive witness helpers for later Akita prove levels.
//!
//! Recursive levels do not operate on a caller-provided polynomial anymore.
//! Instead they carry a flat digit witness `w` that is re-chunked under the
//! current ring dimension `D` on demand. [`RecursiveWitnessFlat`] owns the
//! D-agnostic digit buffer, while [`SuffixWitnessView`] provides the
//! zero-copy D-specific operations used by recursive folding and handoff paths.

#![allow(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::parallel::*;
use akita_field::{AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt};

use crate::backend::poly_helpers::{
    balanced_tight_digit_fold_partitioned, build_decompose_fold_witness,
};
use crate::compute::{CommitInnerPlan, CpuBackend, RootCommitKernel};
use akita_types::{
    tensor_column_partials_from_base_evals, tensor_packed_witness_evals, FpExtEncoding,
    WitnessLayout,
};
use std::{marker::PhantomData, sync::Arc};

use crate::{CommitInnerWitness, DecomposeFoldWitness};

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecursiveWitnessFlat {
    digits: Arc<[i8]>,
    commitment_digits: Option<Arc<[i8]>>,
    commitment_ring_dim: Option<usize>,
    known_balanced_log_basis: Option<u32>,
}

impl RecursiveWitnessFlat {
    pub fn from_i8_digits(digits: Vec<i8>) -> Self {
        Self {
            digits: digits.into(),
            commitment_digits: None,
            commitment_ring_dim: None,
            known_balanced_log_basis: None,
        }
    }

    pub(crate) fn from_witness_layout(
        digits: Vec<i8>,
        layout: &WitnessLayout,
        log_basis: u32,
    ) -> Result<Self, AkitaError> {
        let expected = layout.live_coeff_len();
        if digits.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: digits.len(),
            });
        }
        Ok(Self {
            digits: digits.into(),
            commitment_digits: None,
            commitment_ring_dim: None,
            known_balanced_log_basis: Some(log_basis),
        })
    }

    pub(crate) fn align_for_commitment_ring_dim(
        mut self,
        ring_dim: usize,
    ) -> Result<Self, AkitaError> {
        if ring_dim == 0 || !ring_dim.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "recursive witness commitment ring dimension must be a power of two".into(),
            ));
        }
        let committed_len =
            akita_types::witness_commitment_domain_len(self.digits.len(), ring_dim)?;
        self.commitment_digits = if committed_len == self.digits.len() {
            Some(Arc::clone(&self.digits))
        } else {
            let mut committed = Vec::with_capacity(committed_len);
            committed.extend_from_slice(&self.digits);
            committed.resize(committed_len, 0);
            Some(committed.into())
        };
        self.commitment_ring_dim = Some(ring_dim);
        Ok(self)
    }

    pub fn as_i8_digits(&self) -> &[i8] {
        &self.digits
    }

    pub(crate) fn shared_i8_digits(&self) -> Arc<[i8]> {
        Arc::clone(&self.digits)
    }

    pub fn live_coeff_len(&self) -> usize {
        self.digits.len()
    }

    pub(crate) fn committed_coeff_len(&self) -> Result<usize, AkitaError> {
        self.commitment_digits
            .as_ref()
            .map(|digits| digits.len())
            .ok_or(AkitaError::InvalidProof)
    }

    pub fn is_empty(&self) -> bool {
        self.digits.is_empty()
    }

    pub fn view<F: FieldCore, const D: usize>(
        &self,
    ) -> Result<SuffixWitnessView<'_, F, D>, AkitaError> {
        let digits = match (&self.commitment_digits, self.commitment_ring_dim) {
            (Some(digits), Some(ring_dim)) if ring_dim == D => digits.as_ref(),
            (Some(_), Some(_)) => return Err(AkitaError::InvalidProof),
            (None, None) => self.digits.as_ref(),
            _ => return Err(AkitaError::InvalidProof),
        };
        SuffixWitnessView::from_recursive_witness(
            digits,
            self.digits.len(),
            self.known_balanced_log_basis,
        )
    }
}

impl AsRef<[i8]> for RecursiveWitnessFlat {
    fn as_ref(&self) -> &[i8] {
        self.as_i8_digits()
    }
}

/// D-specific zero-copy view over a flat recursive witness digit buffer.
#[derive(Debug, Clone, Copy)]
pub struct SuffixWitnessView<'a, F: FieldCore, const D: usize> {
    coeffs: &'a [[i8; D]],
    live_ring_elems: usize,
    padded_ring_elems: usize,
    known_balanced_log_basis: Option<u32>,
    _marker: PhantomData<F>,
}

impl<'a, F: FieldCore, const D: usize> SuffixWitnessView<'a, F, D> {
    pub fn from_i8_digits(digits: &'a [i8]) -> Result<Self, AkitaError> {
        Self::from_recursive_witness(digits, digits.len(), None)
    }

    fn from_recursive_witness(
        digits: &'a [i8],
        live_coeff_len: usize,
        known_balanced_log_basis: Option<u32>,
    ) -> Result<Self, AkitaError> {
        let (coeffs, remainder) = digits.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: digits.len(),
            });
        }
        if live_coeff_len > digits.len() {
            return Err(AkitaError::InvalidSize {
                expected: digits.len(),
                actual: live_coeff_len,
            });
        }

        Ok(Self {
            coeffs,
            live_ring_elems: live_coeff_len.div_ceil(D),
            padded_ring_elems: coeffs.len().next_power_of_two().max(1),
            known_balanced_log_basis,
            _marker: PhantomData,
        })
    }

    #[inline]
    fn block_elem(
        &self,
        block_idx: usize,
        col_idx: usize,
        num_positions_per_block: usize,
    ) -> Option<&'a [i8; D]> {
        block_idx
            .checked_mul(num_positions_per_block)
            .and_then(|base| base.checked_add(col_idx))
            .and_then(|index| self.coeffs.get(index))
    }

    pub fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }

    #[inline]
    fn num_live_blocks(&self, num_positions_per_block: usize) -> Result<usize, AkitaError> {
        if num_positions_per_block == 0 || self.coeffs.is_empty() {
            return Err(AkitaError::InvalidInput(
                "recursive witness requires positive exact block geometry".into(),
            ));
        }
        Ok(self.live_ring_elems.div_ceil(num_positions_per_block))
    }

    #[inline]
    pub(crate) fn num_vars(&self) -> usize {
        let total = self
            .padded_ring_elems
            .checked_mul(D)
            .expect("recursive witness ring elems * D overflow");
        total.trailing_zeros() as usize
    }
}

impl<'a, F, const D: usize> SuffixWitnessView<'a, F, D>
where
    F: FieldCore + CanonicalField,
{
    pub(crate) fn base_evals(&self) -> Result<Vec<F>, AkitaError> {
        let expected_len = self.padded_ring_elems.checked_mul(D).ok_or_else(|| {
            AkitaError::InvalidInput("recursive base evals length overflow".to_string())
        })?;
        let mut base_evals = Vec::with_capacity(expected_len);
        for coeffs in self.coeffs {
            base_evals.extend(coeffs.iter().copied().map(F::from_i8));
        }
        base_evals.resize(expected_len, F::zero());
        Ok(base_evals)
    }

    pub(crate) fn tensor_packed_extension_evals<E>(&self) -> Result<Vec<E>, AkitaError>
    where
        E: ExtField<F>,
    {
        let num_vars = self.num_vars();
        let base_evals = self.base_evals()?;
        tensor_packed_witness_evals::<F, E>(num_vars, &base_evals)
    }

    pub(crate) fn tensor_extension_column_partials<E>(
        &self,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        let num_vars = self.num_vars();
        if logical_point.len() != num_vars {
            return Err(AkitaError::InvalidPointDimension {
                expected: num_vars,
                actual: logical_point.len(),
            });
        }
        let base_evals = self.base_evals()?;
        tensor_column_partials_from_base_evals::<F, E>(num_vars, &base_evals, logical_point)
    }

    pub(crate) fn tensor_extension_column_partials_batch<E>(
        polys: &[&Self],
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: akita_field::MulBaseUnreduced<F>,
    {
        polys
            .iter()
            .map(|poly| poly.tensor_extension_column_partials(logical_point))
            .collect()
    }

    pub(crate) fn tensor_packed_extension_sparse_linear_combination<E>(
        polys: &[&Self],
        coeffs: &[E],
    ) -> Result<
        Option<crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>>,
        AkitaError,
    >
    where
        E: ExtField<F>,
    {
        if polys.len() != coeffs.len() {
            return Err(AkitaError::InvalidSize {
                expected: polys.len(),
                actual: coeffs.len(),
            });
        }
        let mut witnesses = Vec::with_capacity(polys.len());
        for poly in polys {
            let Some(witness) = poly.tensor_packed_extension_sparse_evals::<E>()? else {
                return Ok(None);
            };
            witnesses.push(witness);
        }
        Ok(Some(
            crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness::linear_combination(
                coeffs.iter().copied().zip(witnesses.iter()),
            )?,
        ))
    }

    pub(crate) fn tensor_packed_extension_sparse_evals<E>(
        &self,
    ) -> Result<
        Option<crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness<E>>,
        AkitaError,
    >
    where
        E: ExtField<F>,
    {
        Ok(None)
    }

    #[cfg(test)]
    pub(crate) fn fold_blocks(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let num_live_blocks = self.num_live_blocks(num_positions_per_block).unwrap();
        cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (col_idx, &scalar) in scalars.iter().take(num_positions_per_block).enumerate() {
                    let Some(ring) = self.block_elem(block_idx, col_idx, num_positions_per_block)
                    else {
                        break;
                    };
                    for (coeff, &d) in acc.iter_mut().zip(ring.iter()) {
                        if d != 0 {
                            *coeff += scalar * F::from_i8(d);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn fold_blocks_ring(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Vec<CyclotomicRing<F, D>> {
        let num_live_blocks = self.num_live_blocks(num_positions_per_block).unwrap();
        cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (col_idx, scalar) in scalars.iter().take(num_positions_per_block).enumerate() {
                    let Some(digits) = self.block_elem(block_idx, col_idx, num_positions_per_block)
                    else {
                        break;
                    };
                    let ring = CyclotomicRing::<F, D>::from_coefficients(
                        digits.map(|digit| F::from_i8(digit)),
                    );
                    ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
                }
                acc
            })
            .collect()
    }

    pub(crate) fn evaluate_and_fold(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let num_live_blocks = self.num_live_blocks(num_positions_per_block)?;
        let folded = cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let mut acc = [F::zero(); D];
                for (col_idx, &scalar) in position_weights
                    .iter()
                    .take(num_positions_per_block)
                    .enumerate()
                {
                    let Some(ring) = self.block_elem(block_idx, col_idx, num_positions_per_block)
                    else {
                        break;
                    };
                    for (coeff, &d) in acc.iter_mut().zip(ring.iter()) {
                        if d != 0 {
                            *coeff += scalar * F::from_i8(d);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect::<Vec<_>>();
        let eval = folded
            .iter()
            .zip(live_block_weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        Ok((eval, folded))
    }

    pub(crate) fn evaluate_and_fold_ring(
        &self,
        live_block_weights: &[CyclotomicRing<F, D>],
        position_weights: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let num_live_blocks = self.num_live_blocks(num_positions_per_block)?;
        let folded = cfg_into_iter!(0..num_live_blocks)
            .map(|block_idx| {
                let mut acc = CyclotomicRing::<F, D>::zero();
                for (col_idx, scalar) in position_weights
                    .iter()
                    .take(num_positions_per_block)
                    .enumerate()
                {
                    let Some(digits) = self.block_elem(block_idx, col_idx, num_positions_per_block)
                    else {
                        break;
                    };
                    let ring = CyclotomicRing::<F, D>::from_coefficients(
                        digits.map(|digit| F::from_i8(digit)),
                    );
                    ring.mul_accumulate_sparse_rhs_into(scalar, &mut acc);
                }
                acc
            })
            .collect::<Vec<_>>();
        let eval = folded
            .iter()
            .zip(live_block_weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + (*f_i * *s_i)
            });
        Ok((eval, folded))
    }

    #[tracing::instrument(skip_all, name = "SuffixWitnessView::decompose_fold")]
    pub(crate) fn decompose_fold(
        &self,
        challenges: &[SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        _log_basis: u32,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        let num_live_blocks = self.num_live_blocks(num_positions_per_block)?;
        if challenges.len() != num_live_blocks {
            return Err(AkitaError::InvalidSize {
                expected: num_live_blocks,
                actual: challenges.len(),
            });
        }
        if num_digits != 1 {
            return Err(AkitaError::InvalidSetup(
                "recursive digit witness decomposition requires one tight digit".into(),
            ));
        }

        let q = (-F::one()).to_canonical_u128() + 1;
        let coeffs = self.coeffs;
        let coeff_accum =
            balanced_tight_digit_fold_partitioned::<D>(coeffs, challenges, num_positions_per_block);
        Ok(build_decompose_fold_witness::<F, D>(coeff_accum, q))
    }
}

// ===========================================================================
// Source-typed prove views + CpuBackend kernels for [`RecursiveWitnessFlat`].
// ===========================================================================

use crate::backend::RootTensorProjectionPoly;
use crate::compute::{
    BatchDecomposeFoldOutcome, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootTensorSource, TensorPackedWitness,
    TensorProjectionBatchKernel, TensorProjectionKernel,
};
use crate::protocol::extension_opening_reduction::SparseExtensionOpeningWitness;
use akita_field::MulBaseUnreduced;

fn padded_ring_elems_for_digits<const D: usize>(digits: &[i8]) -> usize {
    digits.len().div_ceil(D).next_power_of_two().max(1)
}

/// Same-point batch view over several [`RecursiveWitnessFlat`] suffix witnesses.
#[derive(Debug, Clone, Copy)]
pub struct SuffixWitnessBatchView<'a, F: FieldCore, const D: usize> {
    polys: &'a [&'a RecursiveWitnessFlat],
    _marker: PhantomData<F>,
}

impl<F, const D: usize> RootPolyShape<F, D> for RecursiveWitnessFlat
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        padded_ring_elems_for_digits::<D>(&self.digits)
    }
}

/// D-free polynomial metadata for the recursive suffix witness (H2 boundary).
///
/// The recursive suffix witness is genuinely D-erased: it owns a flat `Vec<i8>`
/// digit buffer (one digit per field-element coefficient) and is re-chunked
/// under the level's ring dimension only inside D-typed kernels. The D-free
/// `RootPolyMeta` is what the PCS-facing `ProverOpeningData::to_opening_shape`
/// requires, so it must expose `num_vars` without a const `D`.
///
/// `num_vars` is the witness's logical variable count `log2(coeff_count)`, where
/// `coeff_count` is the digit buffer length rounded up to the next power of two.
/// The suffix opening point is sized by the schedule's `recursive_opening_num_vars`,
/// and `to_opening_shape` validates the point length against this value. On uniform-D
/// presets this matches the former typed `RootPolyShape::<F, D>::num_vars` =
/// `log2(n_ring · D)` when the padded ring layout is a power of two. Per the cutover
/// mandate, `num_vars` here is derived from the witness's own logical length, never
/// from a const `D`.
///
/// `num_ring_elems` is not on the suffix `to_opening_shape` path (only
/// `num_vars` is consumed there); the D-keyed ring-element count is recovered
/// inside kernels via the D-typed `RootPolyShape`/`SuffixWitnessView`. The
/// D-free value reported here is the flat coefficient count, consistent with
/// `num_vars`.
impl<F> RootPolyMeta<F> for RecursiveWitnessFlat
where
    F: FieldCore,
{
    fn num_ring_elems(&self) -> usize {
        self.digits.len().max(1)
    }

    fn num_vars(&self) -> usize {
        let coeff_count = self.digits.len().next_power_of_two().max(1);
        coeff_count.trailing_zeros() as usize
    }

    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        self.digits.iter().try_fold(0u128, |sum, &digit| {
            let magnitude = u128::from(digit.unsigned_abs());
            magnitude
                .checked_mul(magnitude)
                .and_then(|square| sum.checked_add(square))
        })
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for RecursiveWitnessFlat
where
    F: FieldCore,
{
    type CommitView<'v>
        = SuffixWitnessView<'v, F, D>
    where
        Self: 'v;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        self.view::<F, D>()
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for RecursiveWitnessFlat
where
    F: FieldCore,
{
    type OpeningView<'v>
        = SuffixWitnessView<'v, F, D>
    where
        Self: 'v;

    type OpeningBatchView<'v>
        = SuffixWitnessBatchView<'v, F, D>
    where
        Self: 'v;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        self.view::<F, D>()
    }

    fn opening_batch<'v>(polys: &'v [&'v Self]) -> Result<Self::OpeningBatchView<'v>, AkitaError> {
        Ok(SuffixWitnessBatchView {
            polys,
            _marker: PhantomData,
        })
    }
}

impl<F, const D: usize> RootTensorSource<F, D> for RecursiveWitnessFlat
where
    F: FieldCore,
{
    type TensorView<'v>
        = SuffixWitnessView<'v, F, D>
    where
        Self: 'v;

    type TensorBatchView<'v>
        = SuffixWitnessBatchView<'v, F, D>
    where
        Self: 'v;

    fn tensor_view(&self) -> Result<Self::TensorView<'_>, AkitaError> {
        self.view::<F, D>()
    }

    fn tensor_batch<'v>(polys: &'v [&'v Self]) -> Result<Self::TensorBatchView<'v>, AkitaError> {
        Ok(SuffixWitnessBatchView {
            polys,
            _marker: PhantomData,
        })
    }
}

impl<F, const D: usize> RootCommitKernel<SuffixWitnessView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn commit_inner_group(
        &self,
        prepared: &Self::PreparedSetup,
        sources: Vec<SuffixWitnessView<'_, F, D>>,
        plan: CommitInnerPlan,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        sources
            .into_iter()
            .map(|source| {
                let num_live_blocks = source.num_live_blocks(plan.num_positions_per_block)?;
                let rows = self.recursive_witness_commit_rows(
                    prepared,
                    source.coeffs,
                    plan.n_a,
                    plan.num_positions_per_block,
                    num_live_blocks,
                    plan.num_digits_inner,
                    plan.log_basis_inner,
                    source.known_balanced_log_basis,
                )?;
                Ok(CommitInnerWitness::from_rows(rows))
            })
            .collect()
    }
}

impl<F, const D: usize> OpeningFoldKernel<SuffixWitnessView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F, D>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let num_positions_per_block = plan.num_positions_per_block();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source.num_live_blocks(num_positions_per_block)?;
        plan.validate(num_live_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.evaluate_and_fold(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            )?,
            OpeningFoldPlan::Ring {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.evaluate_and_fold_ring(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            )?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        source.decompose_fold(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        )
    }
}

impl<F, const D: usize> OpeningBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: SuffixWitnessBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

impl<F, E, const D: usize> TensorProjectionKernel<SuffixWitnessView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F>,
{
    fn column_partials(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<E>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        source.tensor_extension_column_partials(logical_point)
    }

    fn packed_witness(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessView<'_, F, D>,
    ) -> Result<TensorPackedWitness<E>, AkitaError> {
        Ok(TensorPackedWitness::Dense(
            source.tensor_packed_extension_evals()?,
        ))
    }

    fn root_projection(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessView<'_, F, D>,
    ) -> Result<RootTensorProjectionPoly<F>, AkitaError>
    where
        E: FpExtEncoding<F>,
    {
        let _ = source;
        Err(AkitaError::InvalidInput(
            "recursive suffix witnesses are not tensor-projected root polynomials".to_string(),
        ))
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F>,
{
    fn column_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        logical_point: &[E],
    ) -> Result<Vec<Vec<E>>, AkitaError>
    where
        E: MulBaseUnreduced<F>,
    {
        let polys = source
            .polys
            .iter()
            .map(|witness| witness.view::<F, D>())
            .collect::<Result<Vec<_>, _>>()?;
        let refs = polys.iter().collect::<Vec<_>>();
        SuffixWitnessView::tensor_extension_column_partials_batch(&refs, logical_point)
    }

    fn sparse_linear_combination(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        coeffs: &[E],
    ) -> Result<Option<SparseExtensionOpeningWitness<E>>, AkitaError> {
        let polys = source
            .polys
            .iter()
            .map(|witness| witness.view::<F, D>())
            .collect::<Result<Vec<_>, _>>()?;
        let refs = polys.iter().collect::<Vec<_>>();
        SuffixWitnessView::tensor_packed_extension_sparse_linear_combination(&refs, coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128OffsetA7F7 as F;

    #[test]
    fn suffix_opening_views_borrow_flat_digit_buffer() {
        const D: usize = 16;
        let digits: Vec<i8> = (0..64).map(|idx| (idx % 5) as i8 - 2).collect();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits.clone());
        let opening: SuffixWitnessView<'_, F, D> = witness.opening_view().expect("opening view");
        let tensor: SuffixWitnessView<'_, F, D> = witness.tensor_view().expect("tensor view");
        assert_eq!(
            opening.num_ring_elems(),
            <RecursiveWitnessFlat as RootPolyShape<F, D>>::num_ring_elems(&witness)
        );
        assert_eq!(
            tensor.num_ring_elems(),
            <RecursiveWitnessFlat as RootPolyShape<F, D>>::num_ring_elems(&witness)
        );

        let polys = [&witness];
        let batch = <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_batch(&polys)
            .expect("opening batch");
        assert_eq!(batch.polys.len(), 1);
    }

    #[test]
    fn commitment_padding_does_not_create_live_blocks() {
        const D: usize = 64;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 70 * D])
            .align_for_commitment_ring_dim(D)
            .expect("commitment alignment");
        let view = witness.view::<F, D>().expect("aligned view");

        assert_eq!(view.live_ring_elems, 70);
        assert!(view.coeffs.len() >= view.live_ring_elems);
        assert_eq!(view.num_live_blocks(10).expect("live blocks"), 7);
    }

    #[test]
    fn suffix_root_projection_is_rejected() {
        const D: usize = 16;
        type E = akita_field::FpExt4<F>;
        let digits: Vec<i8> = (0..64).map(|idx| (idx % 5) as i8 - 2).collect();
        let witness = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = witness.tensor_view().expect("tensor view");
        let err = TensorProjectionKernel::<SuffixWitnessView<'_, F, D>, F, E, D>::root_projection(
            &CpuBackend::DEFAULT,
            None,
            view,
        )
        .expect_err("suffix witnesses must not tensor-project");
        assert!(matches!(err, AkitaError::InvalidInput(_)));
    }

    #[test]
    fn logical_rows_are_contiguous_for_partial_final_fold() {
        let digits: Vec<i8> = (0..20).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w
            .view::<akita_field::Prime128OffsetA7F7, 2>()
            .expect("view");
        let num_live_blocks = 4;
        let num_positions_per_block = (w.live_coeff_len() / 2).div_ceil(num_live_blocks);

        let row = |block_idx: usize| -> Vec<[i8; 2]> {
            (0..num_positions_per_block)
                .filter_map(|col_idx| {
                    view.block_elem(block_idx, col_idx, num_positions_per_block)
                        .copied()
                })
                .collect()
        };

        assert_eq!(row(0), vec![[0, 1], [2, 3], [4, 5]]);
        assert_eq!(row(1), vec![[6, 7], [8, 9], [10, 11]]);
        assert_eq!(row(2), vec![[12, 13], [14, 15], [16, 17]]);
        assert_eq!(row(3), vec![[18, 19]]);
    }

    fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
        CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
            F::from_u64(offset + idx as u64 + 1)
        }))
    }

    #[test]
    fn ring_fold_matches_dense_multiplication_reference() {
        const D: usize = 4;
        let digits = vec![1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12];
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let scalars = vec![ring::<D>(10), ring::<D>(20)];
        let got = view.fold_blocks_ring(&scalars, 2);

        let expected = (0..2)
            .map(|block_idx| {
                (0..2).fold(CyclotomicRing::<F, D>::zero(), |acc, col_idx| {
                    let Some(digits) = view.block_elem(block_idx, col_idx, 2) else {
                        return acc;
                    };
                    let coeff = CyclotomicRing::from_coefficients(digits.map(F::from_i8));
                    acc + coeff * scalars[col_idx]
                })
            })
            .collect::<Vec<_>>();

        assert_eq!(got, expected);
    }

    #[test]
    fn fused_evaluation_uses_physical_order_with_partial_final_fold() {
        const D: usize = 4;
        let digits = (0..24).map(|idx| idx as i8 - 12).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let num_positions_per_block = 4;
        let live_block_weights = vec![F::from_u64(2), F::from_u64(5)];
        let position_weights = vec![
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(13),
            F::from_u64(17),
        ];

        let expected_folded = view.fold_blocks(&position_weights, num_positions_per_block);
        let expected_eval = expected_folded
            .iter()
            .zip(live_block_weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + f_i.scale(s_i)
            });
        let (eval, folded) = view
            .evaluate_and_fold(
                &live_block_weights,
                &position_weights,
                num_positions_per_block,
            )
            .unwrap();

        assert_eq!(folded, expected_folded);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn fused_ring_evaluation_uses_physical_order_with_partial_final_fold() {
        const D: usize = 4;
        let digits = (0..24).map(|idx| idx as i8 - 12).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let num_positions_per_block = 4;
        let live_block_weights = vec![ring::<D>(2), ring::<D>(5)];
        let position_weights = vec![ring::<D>(7), ring::<D>(11), ring::<D>(13), ring::<D>(17)];

        let expected_folded = view.fold_blocks_ring(&position_weights, num_positions_per_block);
        let expected_eval = expected_folded
            .iter()
            .zip(live_block_weights.iter())
            .fold(CyclotomicRing::<F, D>::zero(), |acc, (f_i, s_i)| {
                acc + (*f_i * *s_i)
            });
        let (eval, folded) = view
            .evaluate_and_fold_ring(
                &live_block_weights,
                &position_weights,
                num_positions_per_block,
            )
            .unwrap();

        assert_eq!(folded, expected_folded);
        assert_eq!(eval, expected_eval);
    }

    #[test]
    fn suffix_witness_decompose_fold_is_deterministic() {
        const D: usize = 16;
        let digits = (0..48).map(|idx| (idx % 7) as i8 - 3).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<F, D>().expect("view");
        let challenges = vec![
            SparseChallenge {
                positions: vec![0, 2].into(),
                coeffs: vec![1, -1].into(),
            },
            SparseChallenge {
                positions: vec![1, 3].into(),
                coeffs: vec![2, 1].into(),
            },
        ];

        let once = view.decompose_fold(&challenges, 2, 1, 0).unwrap();
        let twice = view.decompose_fold(&challenges, 2, 1, 0).unwrap();
        assert_eq!(once, twice);
    }
}
