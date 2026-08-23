//! Schedule-bound packed dense witness used after commitment.

use super::poly::DensePoly;
use crate::backend::coefficient_packing::partials_from_position_source;
use crate::backend::packed_digits::PackedSignedDigits;
use crate::backend::poly_helpers::{
    build_decompose_fold_witness, packed_digit_decompose_fold_partitioned,
};
use crate::compute::{
    BatchDecomposeFoldOutcome, CpuBackend, DecomposeFoldBatchPlan, DecomposeFoldPlan,
    OpeningBatchKernel, OpeningFoldKernel, OpeningFoldOutput, OpeningFoldPlan, RootOpeningSource,
    RootPolyMeta, RootPolyShape, SubringCoefficientPackingBatchKernel,
    SubringCoefficientPackingPartials, SubringCoefficientPackingPlan,
};
use crate::DecomposeFoldWitness;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_field::parallel::*;
use akita_field::{CanonicalField, ExtField, FieldCore};

/// Schedule-bound dense opening witness selected after commitment.
///
/// Fast signed-128 reconstruction spans retain only packed digits. Wider spans
/// keep canonical coefficients and their commitment cache until a faster
/// packed opening kernel is available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDenseWitness<F: FieldCore> {
    pub(super) num_vars: usize,
    pub(super) ring_d: usize,
    pub(super) num_digits: usize,
    pub(super) log_basis: u32,
    pub(super) storage: PreparedDenseStorage<F>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum PreparedDenseStorage<F: FieldCore> {
    Packed(PackedSignedDigits),
    Canonical(DensePoly<F>),
}

impl<F: FieldCore> PreparedDenseWitness<F> {
    #[inline]
    fn num_ring_elems_at(&self, ring_d: usize) -> usize {
        (1usize << self.num_vars).div_ceil(ring_d)
    }

    fn validate_view<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_d != D {
            return Err(AkitaError::InvalidInput(format!(
                "prepared dense witness uses ring dimension {} but opening requested D={D}",
                self.ring_d
            )));
        }
        match &self.storage {
            PreparedDenseStorage::Packed(digits) => {
                let expected = self
                    .num_ring_elems_at(D)
                    .checked_mul(self.num_digits)
                    .and_then(|planes| planes.checked_mul(D))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("prepared dense length overflow".into())
                    })?;
                if digits.len() != expected {
                    return Err(AkitaError::InvalidSize {
                        expected,
                        actual: digits.len(),
                    });
                }
                let digit_span = self
                    .num_digits
                    .checked_mul(self.log_basis as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("prepared dense digit span overflow".into())
                    })?;
                if digit_span > 126
                    || digits.bit_width() != self.log_basis as u8
                    || !digits.bounds().fits_balanced_log_basis(self.log_basis)
                {
                    return Err(AkitaError::InvalidInput(
                        "prepared dense digits do not match their fast reconstruction basis".into(),
                    ));
                }
            }
            PreparedDenseStorage::Canonical(poly) => {
                let actual = poly.ring_coeffs::<D>()?.len();
                let expected = self.num_ring_elems_at(D);
                if actual != expected {
                    return Err(AkitaError::InvalidSize { expected, actual });
                }
            }
        }
        Ok(())
    }
}

impl<F: FieldCore + CanonicalField> PreparedDenseWitness<F> {
    fn reconstruct_ring<const D: usize>(
        &self,
        ring_index: usize,
    ) -> Result<CyclotomicRing<F, D>, AkitaError> {
        debug_assert!(ring_index < self.num_ring_elems_at(D));
        if let PreparedDenseStorage::Canonical(poly) = &self.storage {
            return poly
                .ring_coeffs::<D>()?
                .get(ring_index)
                .cloned()
                .ok_or(AkitaError::InvalidProof);
        }
        let PreparedDenseStorage::Packed(digits) = &self.storage else {
            unreachable!("canonical storage returned above")
        };
        let ring_width = self
            .num_digits
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("prepared dense ring width overflow".into()))?;
        let start = ring_index.checked_mul(ring_width).ok_or_else(|| {
            AkitaError::InvalidInput("prepared dense ring offset overflow".into())
        })?;
        let end = start.checked_add(ring_width).ok_or_else(|| {
            AkitaError::InvalidInput("prepared dense ring extent overflow".into())
        })?;
        let digit_view = digits.view().slice(start..end)?;
        let mut digits = digit_view.iter();
        let digit_span = self
            .num_digits
            .checked_mul(self.log_basis as usize)
            .ok_or_else(|| AkitaError::InvalidInput("prepared dense digit span overflow".into()))?;
        debug_assert!(digit_span <= 126);
        let mut signed = [0i128; D];
        let mut shift = 0usize;
        for _ in 0..self.num_digits {
            for value in &mut signed {
                let digit = digits.next().ok_or(AkitaError::InvalidProof)?;
                *value += i128::from(digit) << shift;
            }
            shift += self.log_basis as usize;
        }
        Ok(CyclotomicRing::from_coefficients(std::array::from_fn(
            |index| F::from_i128(signed[index]),
        )))
    }

    fn fold_blocks<const D: usize>(
        &self,
        scalars: &[F],
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.validate_view::<D>()?;
        let num_rings = self.num_ring_elems_at(D);
        cfg_into_iter!(0..num_rings.div_ceil(num_positions_per_block))
            .map(|block_index| {
                let start = block_index * num_positions_per_block;
                let end = (start + num_positions_per_block).min(num_rings);
                let mut accumulator = CyclotomicRing::<F, D>::zero();
                for (ring_index, &scalar) in (start..end).zip(scalars) {
                    self.reconstruct_ring::<D>(ring_index)?
                        .scale_accumulate_into(&mut accumulator, scalar);
                }
                Ok(accumulator)
            })
            .collect()
    }

    fn fold_blocks_ring<const D: usize>(
        &self,
        scalars: &[CyclotomicRing<F, D>],
        num_positions_per_block: usize,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.validate_view::<D>()?;
        let num_rings = self.num_ring_elems_at(D);
        cfg_into_iter!(0..num_rings.div_ceil(num_positions_per_block))
            .map(|block_index| {
                let start = block_index * num_positions_per_block;
                let end = (start + num_positions_per_block).min(num_rings);
                let mut accumulator = CyclotomicRing::<F, D>::zero();
                for (ring_index, scalar) in (start..end).zip(scalars) {
                    self.reconstruct_ring::<D>(ring_index)?
                        .mul_accumulate_sparse_rhs_into(scalar, &mut accumulator);
                }
                Ok(accumulator)
            })
            .collect()
    }

    fn evaluate_and_fold<const D: usize>(
        &self,
        live_block_weights: &[F],
        position_weights: &[F],
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        Ok(crate::backend::poly_helpers::fused_evaluate_and_fold_base(
            self.fold_blocks::<D>(position_weights, num_positions_per_block)?,
            live_block_weights,
        ))
    }

    fn evaluate_and_fold_subfield<const D: usize>(
        &self,
        multipliers: &akita_types::SubfieldMultiplierOpeningPoint<F>,
        num_positions_per_block: usize,
    ) -> Result<(CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>), AkitaError> {
        let position_weights = multipliers.materialize_position_rings::<D>()?;
        let live_block_weights = multipliers.materialize_fold_rings::<D>()?;
        Ok(
            crate::backend::poly_helpers::fused_evaluate_and_fold_materialized(
                self.fold_blocks_ring(&position_weights, num_positions_per_block)?,
                &live_block_weights,
            ),
        )
    }

    fn decompose_fold<const D: usize>(
        &self,
        challenges: &[akita_challenges::SparseChallenge],
        num_positions_per_block: usize,
        num_digits: usize,
        log_basis: u32,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        self.validate_view::<D>()?;
        if self.num_digits != num_digits || self.log_basis != log_basis {
            return Err(AkitaError::InvalidInput(
                "prepared dense decomposition does not match the opening schedule".into(),
            ));
        }
        match &self.storage {
            PreparedDenseStorage::Packed(digits) => {
                let coefficients = packed_digit_decompose_fold_partitioned::<F, D>(
                    digits.view(),
                    self.num_ring_elems_at(D),
                    challenges,
                    num_positions_per_block,
                    num_digits,
                    log_basis,
                );
                let modulus = (-F::one()).to_canonical_u128() + 1;
                Ok(build_decompose_fold_witness::<F, D>(coefficients, modulus))
            }
            PreparedDenseStorage::Canonical(poly) => Ok(poly.decompose_fold::<D>(
                challenges,
                num_positions_per_block,
                num_digits,
                log_basis,
            )),
        }
    }
}

/// Borrowed kernel view over one prepared dense witness.
#[derive(Debug, Clone, Copy)]
pub struct PreparedDenseView<'a, F: FieldCore, const D: usize> {
    witness: &'a PreparedDenseWitness<F>,
}

/// Same-point batch view over prepared dense witnesses.
#[derive(Debug, Clone, Copy)]
pub struct PreparedDenseBatchView<'a, F: FieldCore, const D: usize> {
    witnesses: &'a [&'a PreparedDenseWitness<F>],
}

impl<F: FieldCore> RootPolyMeta<F> for PreparedDenseWitness<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: FieldCore, const D: usize> RootPolyShape<F, D> for PreparedDenseWitness<F> {
    fn num_ring_elems(&self) -> usize {
        self.num_ring_elems_at(D)
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: FieldCore, const D: usize> RootOpeningSource<F, D> for PreparedDenseWitness<F> {
    type OpeningView<'a>
        = PreparedDenseView<'a, F, D>
    where
        Self: 'a;

    type OpeningBatchView<'a>
        = PreparedDenseBatchView<'a, F, D>
    where
        Self: 'a;

    fn opening_view(&self) -> Result<Self::OpeningView<'_>, AkitaError> {
        self.validate_view::<D>()?;
        Ok(PreparedDenseView { witness: self })
    }

    fn opening_batch<'a>(polys: &'a [&'a Self]) -> Result<Self::OpeningBatchView<'a>, AkitaError> {
        for witness in polys {
            witness.validate_view::<D>()?;
        }
        Ok(PreparedDenseBatchView { witnesses: polys })
    }
}

impl<F, const D: usize> OpeningFoldKernel<PreparedDenseView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn evaluate_and_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: PreparedDenseView<'_, F, D>,
        plan: OpeningFoldPlan<'_, F>,
    ) -> Result<OpeningFoldOutput<F, D>, AkitaError> {
        let num_positions_per_block = plan.num_positions_per_block();
        if num_positions_per_block == 0 {
            return Err(AkitaError::InvalidInput(
                "num_positions_per_block must be positive".into(),
            ));
        }
        let num_live_blocks = source
            .witness
            .num_ring_elems_at(D)
            .div_ceil(num_positions_per_block);
        plan.validate::<D>(num_live_blocks)?;
        let (eval, folded) = match plan {
            OpeningFoldPlan::Base {
                live_block_weights,
                position_weights,
                num_positions_per_block,
            } => source.witness.evaluate_and_fold::<D>(
                live_block_weights,
                position_weights,
                num_positions_per_block,
            )?,
            OpeningFoldPlan::Subfield {
                multipliers,
                num_positions_per_block,
            } => source
                .witness
                .evaluate_and_fold_subfield(multipliers, num_positions_per_block)?,
        };
        Ok(OpeningFoldOutput { eval, folded })
    }

    fn decompose_fold(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: PreparedDenseView<'_, F, D>,
        plan: DecomposeFoldPlan<'_>,
    ) -> Result<DecomposeFoldWitness<F>, AkitaError> {
        source.witness.decompose_fold::<D>(
            plan.challenges,
            plan.num_positions_per_block,
            plan.num_digits,
            plan.log_basis,
        )
    }
}

impl<F, const D: usize> OpeningBatchKernel<PreparedDenseBatchView<'_, F, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn decompose_fold_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        _source: PreparedDenseBatchView<'_, F, D>,
        _plan: DecomposeFoldBatchPlan<'_>,
    ) -> Result<BatchDecomposeFoldOutcome<F, D>, AkitaError> {
        Ok(BatchDecomposeFoldOutcome::FallbackPerPoly)
    }
}

impl<F, E, const D: usize>
    SubringCoefficientPackingBatchKernel<PreparedDenseBatchView<'_, F, D>, F, E, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
    E: ExtField<F> + akita_types::FpExtEncoding<F>,
{
    fn coefficient_packing_partials_batch(
        &self,
        _prepared: Option<&Self::PreparedSetup>,
        source: PreparedDenseBatchView<'_, F, D>,
        plan: SubringCoefficientPackingPlan<'_, E>,
    ) -> Result<Vec<SubringCoefficientPackingPartials<F>>, AkitaError> {
        source
            .witnesses
            .iter()
            .map(|witness| {
                let num_rings = witness.num_ring_elems_at(D);
                if num_rings != plan.point.num_live_positions() {
                    return Err(AkitaError::InvalidSize {
                        expected: plan.point.num_live_positions(),
                        actual: num_rings,
                    });
                }
                let coordinates = partials_from_position_source::<F, E, _, D>(
                    plan,
                    RootPolyMeta::<F>::num_vars(*witness),
                    |position| witness.reconstruct_ring::<D>(position),
                    |_, coefficient, ring| ring.coefficients()[coefficient],
                )?;
                SubringCoefficientPackingPartials::new(
                    plan.point.geometry(),
                    plan.point.num_live_blocks(),
                    coordinates,
                )
            })
            .collect()
    }
}
