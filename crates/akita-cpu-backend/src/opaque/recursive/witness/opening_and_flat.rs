impl<F, E> crate::opaque::consumer_kernels::CpuWitnessOpeningKernel<F, E> for crate::opaque::CpuBackend
where
    F: Field + CanonicalEncoding + Send + Sync + 'static,
    E: ExtField<F>
        + jolt_field::Unreduced
        + jolt_field::Fold
        + jolt_field::MulBaseUnreduced<F>
        + Send
        + Sync
        + 'static,

{
    type WitnessOpeningHandle = crate::opaque::CpuWitnessOpeningHandle<E>;
    type WitnessEorSessionHandle = CpuExtensionOpeningSession<E>;
    fn prepare_witness_opening(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        plan: &crate::opaque::ValidatedWitnessOpeningPlan<'_, E>,
    ) -> Result<crate::opaque::PreparedWitnessOpening<E, Self::WitnessOpeningHandle>, AkitaError>
    {
        if witness_handle.logical.live_coeff_len() != plan.witness_len() {
            return Err(AkitaError::InvalidInput(
                "recursive witness opening plan disagrees with its witness length".into(),
            ));
        }
        let (partials, tensor_evals) = akita_types::dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Inner),
            F,
            plan.ring_dimension(),
            |D| {
                let view = witness_handle.logical.view::<F, D>()?;
                Ok::<_, AkitaError>((
                    view.tensor_extension_column_partials::<E>(plan.point())?,
                    view.tensor_packed_extension_evals::<E>()?,
                ))
            }
        )?;
        let opening = akita_types::derive_tensor_extension_opening_claim_from_partials::<F, E>(
            plan.point(),
            &partials,
        )?;
        Ok(crate::opaque::PreparedWitnessOpening::new(
            vec![opening],
            partials,
            OpaqueWitnessOpeningState {
                source_operation: witness_handle.binding.operation_id(),
                point: plan.point().to_vec(),
                tensor_evals,
                witness_len: witness_handle.logical.live_coeff_len(),
                ring_dimension: plan.ring_dimension(),
            },
        ))
    }

    fn begin_witness_eor(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        witness_handle: &Self::WitnessHandle,
        opening_handle: Self::WitnessOpeningHandle,
        plan: &crate::opaque::ValidatedWitnessEorPlan<'_, E>,
    ) -> Result<Self::WitnessEorSessionHandle, AkitaError> {
        if opening_handle.ring_dimension != plan.ring_dimension()
            || opening_handle.witness_len != witness_handle.logical.live_coeff_len()
            || opening_handle.source_operation != witness_handle.binding.operation_id()
        {
            return Err(AkitaError::InvalidInput(
                "recursive witness EOR plan disagrees with its opening geometry".into(),
            ));
        }
        let (split_bits,_)=akita_types::tensor_opening_split::<F,E>()?;
        if opening_handle.point.get(split_bits..)!=Some(plan.tail_point()) {
            return Err(AkitaError::InvalidInput("EOR point differs from the retained opening".into()));
        }
        let [coefficient] = plan.claim_coefficients() else {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: plan.claim_coefficients().len(),
            });
        };
        cpu_witness_eor_session_from_witnesses::<F, E>(
            vec![opening_handle.tensor_evals],
            core::slice::from_ref(coefficient),
            plan.tail_point(),
            plan.eta(),
            plan.extra_point().to_vec(),
            plan.input_claim(),
        )
    }
}

impl RecursiveWitnessFlat {
    #[doc(hidden)]
    pub(crate) fn from_i8_digits(digits: Vec<i8>) -> Self {
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

    #[cfg(test)]
    pub(crate) fn to_i8_digits(&self) -> Vec<i8> {
        self.digits.decode()
    }

    pub(crate) fn packed_digits(&self) -> &PackedSignedDigits {
        &self.digits
    }

    #[cfg(feature = "response-model-diagnostics")]
    pub(crate) fn digit(&self, index: usize) -> Option<i8> {
        self.digits.get(index)
    }

    pub(crate) fn digits(&self) -> impl ExactSizeIterator<Item = i8> + '_ {
        self.digits.iter()
    }

    pub(crate) fn live_coeff_len(&self) -> usize {
        self.live_coeff_len
    }

    pub(crate) fn commitment_physical_len(&self) -> Result<usize, AkitaError> {
        match self.committed_coeff_len {
            Some(committed_len) => Ok(committed_len),
            None => self
                .digits
                .len()
                .max(1)
                .checked_next_power_of_two()
                .ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "recursive witness commitment extent overflows usize".into(),
                    )
                }),
        }
    }

    pub(crate) fn packed_representation_parts(&self) -> (&[u8], u8, u8, u8) {
        let bounds = self.digits.bounds();
        (
            self.digits.encoded_bytes(),
            self.digits.bit_width(),
            bounds.negative_abs_max(),
            bounds.positive_max(),
        )
    }

    #[cfg(test)]
    pub(crate) fn committed_coeff_len(&self) -> Result<usize, AkitaError> {
        self.committed_coeff_len.ok_or(AkitaError::InvalidProof)
    }

    #[doc(hidden)]
    #[allow(unreachable_pub)]
    pub(crate) fn view<F: Field, const D: usize>(
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
#[derive(Clone, Copy)]
pub(crate) struct SuffixWitnessView<'a, F: Field, const D: usize> {
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
        Ok(crate::sources::poly_helpers::fused_evaluate_and_fold_base(
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
            crate::sources::poly_helpers::fused_evaluate_and_fold_materialized(
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

    pub(crate) fn decompose_fold_chunked(
        &self,
        challenges: &[SparseChallenge],
        chunk_ranges: &[std::ops::Range<usize>],
        num_positions_per_block: usize,
        num_digits: usize,
    ) -> Result<Vec<DecomposeFoldWitness<F>>, AkitaError> {
        let num_live_blocks = self.num_live_blocks(num_positions_per_block)?;
        if challenges.len() != num_live_blocks || num_digits != 1 {
            return Err(AkitaError::InvalidSetup(
                "recursive chunked fold plan disagrees with tight witness geometry".into(),
            ));
        }
        let mut accumulators = vec![vec![[0i32; D]; num_positions_per_block]; chunk_ranges.len()];
        let mut chunk = 0usize;
        for ring_index in 0..self.live_ring_elems {
            let block = ring_index / num_positions_per_block;
            while chunk + 1 < chunk_ranges.len() && block >= chunk_ranges[chunk].end {
                chunk += 1;
            }
            if chunk_ranges[chunk].contains(&block) {
                let ring = self.ring_elem(ring_index).ok_or(AkitaError::InvalidProof)?;
                sparse_mul_acc(
                    &ring,
                    &challenges[block],
                    &mut accumulators[chunk][ring_index % num_positions_per_block],
                );
            }
        }
        let q = (-F::one())
            .to_u128_checked()
            .expect("Akita field element must fit in u128")
            + 1;
        Ok(accumulators
            .into_iter()
            .map(|coefficients| build_decompose_fold_witness::<F, D>(coefficients, q))
            .collect())
    }
}

// ===========================================================================
// Source-typed prove views + CpuBackend kernels for [`RecursiveWitnessFlat`].
// ===========================================================================

use crate::arithmetic::coefficient_packing::{
    coefficient_packing_partials_from_position_source, FusedPackingWeights,
};
use crate::opaque::aggregate_decompose_fold_witnesses;
use crate::opaque::{OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput};
use crate::opaque::{
    DecomposeFoldBatchPlan, DecomposeFoldPlan, OpeningFoldPlan, RootOpeningSource, RootPolyMeta,
    RootPolyShape, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use jolt_field::MulBaseUnreduced;

fn padded_ring_elems_for_live_len<const D: usize>(live_coeff_len: usize) -> usize {
    live_coeff_len.div_ceil(D).next_power_of_two().max(1)
}

/// Same-point batch view over several [`RecursiveWitnessFlat`] suffix witnesses.
#[derive(Clone)]
pub(crate) struct SuffixWitnessBatchView<'a, F: Field, const D: usize> {
    polys: Vec<&'a RecursiveWitnessFlat>,
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
            polys: polys.to_vec(),
            _marker: PhantomData,
        })
    }
}

impl<F, const D: usize> RootPolyShape<F, D> for OpaqueRecursiveWitness
where
    F: Field,
{
    fn num_ring_elems(&self) -> usize {
        <RecursiveWitnessFlat as RootPolyShape<F, D>>::num_ring_elems(
            self.committed.as_ref().unwrap_or(&self.logical),
        )
    }

    fn num_live_ring_elems(&self) -> usize {
        <RecursiveWitnessFlat as RootPolyShape<F, D>>::num_live_ring_elems(
            self.committed.as_ref().unwrap_or(&self.logical),
        )
    }
}

impl<F> RootPolyMeta<F> for OpaqueRecursiveWitness
where
    F: Field,
{
    fn num_vars(&self) -> usize {
        <RecursiveWitnessFlat as RootPolyMeta<F>>::num_vars(
            self.committed.as_ref().unwrap_or(&self.logical),
        )
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
        prepared: Option<&Self::PreparedSetup>,
        source: SuffixWitnessBatchView<'_, F, D>,
        plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<crate::opaque::CpuFoldResponses<F>, AkitaError> {
        let challenges_per_poly = plan.challenges_per_poly(source.polys.len())?;
        let (num_positions_per_block, num_digits, log_basis) = plan.scalar_params();
        match plan {
            DecomposeFoldBatchPlan::Sparse { challenges, .. } => {
                Ok(crate::opaque::CpuFoldResponses::sparse(
                    aggregate_decompose_fold_witnesses::<F, D>(
                        source
                            .polys
                            .iter()
                            .zip(challenges.chunks_exact(challenges_per_poly))
                            .map(|(poly, poly_challenges)| {
                                <Self as OpeningFoldKernel<
                                    SuffixWitnessView<'_, F, D>, F, D,
                                >>::decompose_fold(
                                    self,
                                    prepared,
                                    poly.opening_view()?,
                                    DecomposeFoldPlan {
                                        challenges: poly_challenges,
                                        num_positions_per_block,
                                        num_digits,
                                        log_basis,
                                    },
                                )
                            }),
                    )?,
                ))
            }
            DecomposeFoldBatchPlan::SparseChunked {
                challenges,
                chunk_ranges,
                ..
            } => {
                let mut by_chunk = (0..chunk_ranges.len())
                    .map(|_| Vec::with_capacity(source.polys.len()))
                    .collect::<Vec<_>>();
                for (poly, poly_challenges) in source
                    .polys
                    .iter()
                    .zip(challenges.as_slice().chunks_exact(challenges_per_poly))
                {
                    let view =
                        <RecursiveWitnessFlat as RootOpeningSource<F, D>>::opening_view(poly)?;
                    let chunks = view.decompose_fold_chunked(
                        poly_challenges,
                        chunk_ranges,
                        num_positions_per_block,
                        num_digits,
                    )?;
                    for (chunk, witness) in by_chunk.iter_mut().zip(chunks) {
                        chunk.push(Ok(witness));
                    }
                }
                crate::opaque::CpuFoldResponses::chunked::<D>(
                    by_chunk
                        .into_iter()
                        .map(aggregate_decompose_fold_witnesses::<F, D>)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        }
    }
}
