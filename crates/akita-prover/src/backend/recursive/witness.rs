//! Recursive witness helpers for later Akita prove levels.
//!
//! Recursive levels do not operate on a caller-provided polynomial anymore.
//! Instead they carry a flat digit witness `w` that is re-chunked under the
//! current ring dimension `D` on demand. [`RecursiveWitnessFlat`] owns the
//! D-agnostic packed digit buffer, while [`SuffixWitnessView`] provides the
//! D-specific operations used by recursive folding and handoff paths.

#![allow(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod tensor;

use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_error::AkitaError;
use jolt_field::solinas::parallel::*;
use jolt_field::{CanonicalEncoding, ExtField, Field, Ring};

use crate::backend::packed_digits::{PackedSignedDigitView, PackedSignedDigits};
use crate::backend::poly_helpers::{
    build_decompose_fold_witness, packed_tight_digit_fold_partitioned,
};
use crate::compute::{CommitInnerPlan, CpuBackend, RootCommitKernel};
use akita_types::WitnessLayout;
use std::marker::PhantomData;

use crate::{CommitInnerWitness, DecomposeFoldWitness};

/// D-agnostic owner for the recursive witness vector `w`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecursiveWitnessFlat {
    digits: PackedSignedDigits,
    live_coeff_len: usize,
    committed_coeff_len: Option<usize>,
    commitment_ring_dim: Option<usize>,
}

impl RecursiveWitnessFlat {
    pub fn from_i8_digits(digits: Vec<i8>) -> Self {
        let live_coeff_len = digits.len();
        Self {
            digits: PackedSignedDigits::from_i8_digits_auto(digits),
            live_coeff_len,
            committed_coeff_len: None,
            commitment_ring_dim: None,
        }
    }

    pub(crate) fn from_witness_layout(
        digits: PackedSignedDigits,
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
        if !digits.bounds().fits_balanced_log_basis(log_basis) {
            return Err(AkitaError::InvalidInput(
                "recursive witness contains digits outside its declared balanced basis".into(),
            ));
        }
        Ok(Self {
            digits,
            live_coeff_len: expected,
            committed_coeff_len: None,
            commitment_ring_dim: None,
        })
    }

    pub(crate) fn from_tensor_packed_i8_digits(
        digits: Vec<i8>,
        live_coeff_len: usize,
    ) -> Result<Self, AkitaError> {
        if live_coeff_len > digits.len() {
            return Err(AkitaError::InvalidSize {
                expected: digits.len(),
                actual: live_coeff_len,
            });
        }
        Ok(Self {
            digits: PackedSignedDigits::from_i8_digits_auto(digits),
            live_coeff_len,
            committed_coeff_len: None,
            commitment_ring_dim: None,
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
        self.committed_coeff_len = Some(committed_len);
        self.commitment_ring_dim = Some(ring_dim);
        Ok(self)
    }

    pub fn to_i8_digits(&self) -> Vec<i8> {
        self.digits.decode()
    }

    pub(crate) fn packed_digits(&self) -> PackedSignedDigits {
        self.digits.clone()
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn digit(&self, index: usize) -> Option<i8> {
        self.digits.get(index)
    }

    pub(crate) fn digits(&self) -> impl ExactSizeIterator<Item = i8> + '_ {
        self.digits.iter()
    }

    pub fn live_coeff_len(&self) -> usize {
        self.live_coeff_len
    }

    pub(crate) fn committed_coeff_len(&self) -> Result<usize, AkitaError> {
        self.committed_coeff_len.ok_or(AkitaError::InvalidProof)
    }

    pub fn is_empty(&self) -> bool {
        self.digits.is_empty()
    }

    pub fn view<F: Field, const D: usize>(
        &self,
    ) -> Result<SuffixWitnessView<'_, F, D>, AkitaError> {
        let physical_len = match (self.committed_coeff_len, self.commitment_ring_dim) {
            (Some(committed_len), Some(ring_dim)) if ring_dim == D => committed_len,
            (Some(_), Some(_)) => return Err(AkitaError::InvalidProof),
            (None, None) => self.digits.len(),
            _ => return Err(AkitaError::InvalidProof),
        };
        if !physical_len.is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: physical_len,
            });
        }
        SuffixWitnessView::from_recursive_witness(
            self.digits.zero_padded(physical_len)?,
            self.live_coeff_len,
        )
    }
}

/// D-specific view over a packed recursive witness digit buffer.
#[derive(Debug, Clone, Copy)]
pub struct SuffixWitnessView<'a, F: Field, const D: usize> {
    digits: PackedSignedDigitView<'a>,
    live_coeff_len: usize,
    live_ring_elems: usize,
    padded_ring_elems: usize,
    _marker: PhantomData<F>,
}

impl<'a, F: Field, const D: usize> SuffixWitnessView<'a, F, D> {
    fn from_recursive_witness(
        digits: PackedSignedDigitView<'a>,
        live_coeff_len: usize,
    ) -> Result<Self, AkitaError> {
        if live_coeff_len > digits.len() {
            return Err(AkitaError::InvalidSize {
                expected: digits.len(),
                actual: live_coeff_len,
            });
        }

        Ok(Self {
            digits,
            live_coeff_len,
            live_ring_elems: live_coeff_len.div_ceil(D),
            padded_ring_elems: (digits.len() / D).next_power_of_two().max(1),
            _marker: PhantomData,
        })
    }

    #[inline]
    fn block_elem(
        &self,
        block_idx: usize,
        col_idx: usize,
        num_positions_per_block: usize,
    ) -> Option<[i8; D]> {
        block_idx
            .checked_mul(num_positions_per_block)
            .and_then(|base| base.checked_add(col_idx))
            .and_then(|index| self.ring_elem(index))
    }

    #[inline]
    fn ring_elem(&self, index: usize) -> Option<[i8; D]> {
        (index < self.padded_ring_elems)
            .then(|| self.digits.decode_array(index * D).ok())
            .flatten()
    }

    #[inline]
    fn digit(&self, index: usize) -> Option<i8> {
        self.digits.get(index)
    }

    pub fn num_ring_elems(&self) -> usize {
        self.padded_ring_elems
    }

    #[inline]
    fn num_live_blocks(&self, num_positions_per_block: usize) -> Result<usize, AkitaError> {
        if num_positions_per_block == 0 || self.digits.len() == 0 {
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
    F: Field + CanonicalEncoding,
{
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
                    for (coeff, &digit) in acc.iter_mut().zip(ring.iter()) {
                        if digit != 0 {
                            *coeff += scalar * F::from_i8(digit);
                        }
                    }
                }
                CyclotomicRing::from_coefficients(acc)
            })
            .collect::<Vec<_>>();
        Ok(crate::backend::poly_helpers::fused_evaluate_and_fold_base(
            folded,
            live_block_weights,
        ))
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
        Ok(
            crate::backend::poly_helpers::fused_evaluate_and_fold_materialized(
                folded,
                live_block_weights,
            ),
        )
    }

    pub(crate) fn evaluate_and_fold_subfield(
        &self,
        multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let position_weights = multipliers.materialize_position_rings::<D>()?;
        let live_block_weights = multipliers.materialize_fold_rings::<D>()?;
        self.evaluate_and_fold_ring(
            &live_block_weights,
            &position_weights,
            num_positions_per_block,
        )
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

        let q = (-F::one())
            .to_u128_checked()
            .expect("Akita field element must fit in u128")
            + 1;
        let coeff_accum = packed_tight_digit_fold_partitioned::<F, D>(
            self.digits,
            self.live_ring_elems,
            challenges,
            num_positions_per_block,
        );
        Ok(build_decompose_fold_witness::<F, D>(coeff_accum, q))
    }
}

// ===========================================================================
// Source-typed prove views + CpuBackend kernels for [`RecursiveWitnessFlat`].
// ===========================================================================

use crate::compute::{
    BatchDecomposeFoldOutcome, DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningBatchKernel,
    OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootCommitSource, RootOpeningSource,
    RootPolyMeta, RootPolyShape, RootTensorSource, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan, TensorProjectionBatchKernel,
    TensorProjectionKernel,
};
use jolt_field::MulBaseUnreduced;

fn padded_ring_elems_for_live_len<const D: usize>(live_coeff_len: usize) -> usize {
    live_coeff_len.div_ceil(D).next_power_of_two().max(1)
}

/// Same-point batch view over several [`RecursiveWitnessFlat`] suffix witnesses.
#[derive(Debug, Clone, Copy)]
pub struct SuffixWitnessBatchView<'a, F: Field, const D: usize> {
    polys: &'a [&'a RecursiveWitnessFlat],
    _marker: PhantomData<F>,
}

impl<F, const D: usize> RootPolyShape<F, D> for RecursiveWitnessFlat
where
    F: Field,
{
    fn num_ring_elems(&self) -> usize {
        padded_ring_elems_for_live_len::<D>(self.live_coeff_len)
    }

    fn num_live_ring_elems(&self) -> usize {
        self.live_coeff_len.div_ceil(D)
    }
}

/// D-free polynomial metadata for the recursive suffix witness (H2 boundary).
///
/// The recursive suffix witness is genuinely D-erased. It owns packed signed
/// digits and decodes D-sized rings only inside D-typed kernels. The D-free
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
impl<F> RootPolyMeta<F> for RecursiveWitnessFlat
where
    F: Field,
{
    fn num_vars(&self) -> usize {
        let coeff_count = self.live_coeff_len.next_power_of_two().max(1);
        coeff_count.trailing_zeros() as usize
    }

    #[cfg(feature = "response-model-diagnostics")]
    fn exact_integer_coeff_l2_sq(&self) -> Option<u128> {
        (0..self.live_coeff_len).try_fold(0u128, |sum, index| {
            let digit = self.digit(index)?;
            let magnitude = u128::from(digit.unsigned_abs());
            magnitude
                .checked_mul(magnitude)
                .and_then(|square| sum.checked_add(square))
        })
    }
}

impl<F, const D: usize> RootCommitSource<F, D> for RecursiveWitnessFlat
where
    F: Field,
{
    type CommitView<'v>
        = SuffixWitnessView<'v, F, D>
    where
        Self: 'v;

    fn commit_view(&self) -> Result<Self::CommitView<'_>, AkitaError> {
        self.view::<F, D>()
    }

    /// A recursive witness is already stored as signed `i8` digits, so its exact
    /// reach is the largest stored magnitude on each side. It is never
    /// field-wide, and recursive levels commit against `log_basis` rather than a
    /// declared source bound, so this is only ever a lower-cost restatement of an
    /// already-small range.
    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError>
    where
        F: jolt_field::CanonicalEncoding,
    {
        let bounds = self.digits.bounds();
        Ok((
            u128::from(bounds.negative_abs_max()),
            u128::from(bounds.positive_max()),
        ))
    }
}

impl<F, const D: usize> RootOpeningSource<F, D> for RecursiveWitnessFlat
where
    F: Field,
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
    F: Field,
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
    F: Field + CanonicalEncoding,
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
                let rows = self.recursive_packed_witness_commit_rows::<F, D>(
                    prepared,
                    source.digits,
                    plan.n_a,
                    plan.num_positions_per_block,
                    num_live_blocks,
                    plan.num_digits_inner,
                    plan.log_basis_inner,
                )?;
                Ok(CommitInnerWitness::from_rows(rows))
            })
            .collect()
    }
}

impl<F, const D: usize> OpeningFoldKernel<SuffixWitnessView<'_, F, D>, F, D> for CpuBackend
where
    F: Field + CanonicalEncoding,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let num_positions_per_block = plan.num_positions_per_block();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".to_string(),
            ));
        }
        let num_live_blocks = source.num_live_blocks(num_positions_per_block)?;
        plan.validate::<D>(num_live_blocks)?;
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
            OpeningFoldPlan::Subfield {
                multipliers,
                num_positions_per_block,
            } => source.evaluate_and_fold_subfield(multipliers, num_positions_per_block)?,
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
    F: Field + CanonicalEncoding,
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
    F: Field + CanonicalEncoding + Ring,
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
    ) -> Result<Vec<E>, AkitaError> {
        source.tensor_packed_extension_evals()
    }
}

impl<F, E, const D: usize> TensorProjectionBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D>
    for CpuBackend
where
    F: Field + CanonicalEncoding,
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
}

pub(crate) fn suffix_witness_coefficient_packing_partials<F, E, const D: usize>(
    witness: &RecursiveWitnessFlat,
    plan: SubringCoefficientPackingPlan<'_, E>,
    fused_weights: &[E],
) -> Result<SubringCoefficientPackingPartials<F>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let view = witness.view::<F, D>()?;
    if view.live_ring_elems != plan.point.num_live_positions() {
        return Err(AkitaError::InvalidSize {
            expected: plan.point.num_live_positions(),
            actual: view.live_ring_elems,
        });
    }
    let coordinates =
        crate::backend::coefficient_packing::coefficient_packing_partials_from_position_source::<
            F,
            E,
            _,
            D,
        >(
            plan,
            fused_weights,
            view.num_vars(),
            |position| view.ring_elem(position).ok_or(AkitaError::InvalidProof),
            |position, coefficient_index, source| {
                let flat_index = position * D + coefficient_index;
                if flat_index < view.live_coeff_len {
                    F::from_i8(source[coefficient_index])
                } else {
                    F::zero()
                }
            },
        )?;
    SubringCoefficientPackingPartials::new(
        plan.point.geometry(),
        plan.point.num_live_blocks(),
        coordinates,
    )
}

impl<F, E, const D: usize>
    SubringCoefficientPackingBatchKernel<SuffixWitnessBatchView<'_, F, D>, F, E, D> for CpuBackend
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + akita_types::FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        let fused_weights =
            crate::backend::coefficient_packing::prepare_packing_position_weights(plan.point)?;
        source
            .polys
            .iter()
            .map(|witness| {
                suffix_witness_coefficient_packing_partials::<F, E, D>(
                    witness,
                    plan,
                    &fused_weights,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128OffsetA7F7 as F;

    #[test]
    fn suffix_opening_views_share_packed_digit_storage() {
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
    fn recursive_owner_keeps_only_exact_packed_digits() {
        const D: usize = 64;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![-4, -1, 0, 3, 2]);
        assert_eq!(witness.digits.bit_width(), 3);
        assert_eq!(witness.digits.encoded_bytes().len(), 2);

        let aligned = witness
            .align_for_commitment_ring_dim(D)
            .expect("commitment alignment");
        assert_eq!(aligned.committed_coeff_len().unwrap(), D);
        assert_eq!(aligned.digits.encoded_bytes().len(), 2);
        assert_eq!(aligned.to_i8_digits(), [-4, -1, 0, 3, 2]);
    }

    #[test]
    fn commitment_padding_does_not_create_live_blocks() {
        const D: usize = 64;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 70 * D])
            .align_for_commitment_ring_dim(D)
            .expect("commitment alignment");
        let view = witness.view::<F, D>().expect("aligned view");

        assert_eq!(view.live_ring_elems, 70);
        assert!(view.padded_ring_elems >= view.live_ring_elems);
        assert_eq!(view.num_live_blocks(10).expect("live blocks"), 7);
    }

    #[test]
    fn tensor_view_uses_commitment_domain_after_commitment_padding() {
        const D: usize = 64;
        let witness = RecursiveWitnessFlat::from_i8_digits(vec![1; 70 * D])
            .align_for_commitment_ring_dim(D)
            .expect("commitment alignment");

        let committed: SuffixWitnessView<'_, F, D> = witness.commit_view().expect("commit view");
        let tensor: SuffixWitnessView<'_, F, D> = witness.tensor_view().expect("tensor view");

        assert_eq!(committed.digits.len(), tensor.digits.len());
        assert_eq!(tensor.live_ring_elems, 70);
        assert_eq!(tensor.num_vars(), 13);
    }

    #[test]
    fn logical_rows_are_contiguous_for_partial_final_fold() {
        let digits: Vec<i8> = (0..20).collect();
        let w = RecursiveWitnessFlat::from_i8_digits(digits);
        let view = w.view::<jolt_field::Prime128OffsetA7F7, 2>().expect("view");
        let num_live_blocks = 4;
        let num_positions_per_block = (w.live_coeff_len() / 2).div_ceil(num_live_blocks);

        let row = |block_idx: usize| -> Vec<[i8; 2]> {
            (0..num_positions_per_block)
                .filter_map(|col_idx| view.block_elem(block_idx, col_idx, num_positions_per_block))
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
