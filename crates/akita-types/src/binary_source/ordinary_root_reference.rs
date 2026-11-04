//! Test-only reference: the flat term-list construction of the ordinary rows.
//!
//! This is the compiler that preceded the structured families, kept term for
//! term. It pushes one `(address, weight)` pair per public coefficient weight;
//! the differential tests compare the structured families with it.

use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use jolt_field::{Field, Prime128Offset275};

use super::{KernelCache, OrdinaryNativeRootPlan};
use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::RootCoefficientRepacking;
use crate::{derive_public_matrix_prefix, AkitaSetupDescriptor};

impl OrdinaryNativeRootPlan {
    /// Flat reference for [`OrdinaryNativeRootPlan::linear_weights`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reference_linear_weights<E: Field>(
        &self,
        repacking: &RootCoefficientRepacking,
        setup: &AkitaSetupDescriptor,
        point: &[E],
        evaluation: E,
        payload: &[E],
        alpha: E,
        batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        self.validate_repacking(repacking)?;
        if point.len() != self.profile.group.num_vars()
            || batches.len() != self.linear_batch_count
            || payload.len() != self.compression.terminal_coefficients()
            || E::from_u128(BINARY_SETUP_P) != E::zero()
        {
            return Err(AkitaError::InvalidProof);
        }
        let a = self.profile.inner.matrix;
        let b = self.profile.outer.matrix;
        let first = *self
            .compression
            .maps()
            .first()
            .ok_or(AkitaError::InvalidProof)?;
        let second = *self
            .compression
            .maps()
            .get(1)
            .ok_or(AkitaError::InvalidProof)?;
        let required = [
            checked::product([a.output_rank(), a.input_width(), a.ring_dimension()]),
            checked::product([b.output_rank(), b.input_width(), b.ring_dimension()]),
            Some(self.compression.max_setup_field_elements()?),
        ]
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(AkitaError::InvalidProof)?
        .into_iter()
        .max()
        .ok_or(AkitaError::InvalidProof)?;
        if required > setup.num_field_elements
            || required > crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root exceeds the admitted setup prefix".into(),
            ));
        }
        let matrix = derive_public_matrix_prefix::<Prime128Offset275>(required, &setup.setup_seed);
        let matrix = matrix.as_field_slice();
        let source_coefficients = checked::product([
            self.segments
                .first()
                .ok_or(AkitaError::InvalidProof)?
                .elements,
            a.ring_dimension(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let t_coefficients = checked::product([
            self.segments
                .get(1)
                .ok_or(AkitaError::InvalidProof)?
                .elements,
            self.profile.outer.digits.num_digits,
            b.ring_dimension(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let term_capacity = [
            checked::product([source_coefficients, a.output_rank()]),
            checked::product([
                self.profile.blocks.live_blocks,
                a.output_rank(),
                a.ring_dimension(),
                self.profile.outer.digits.num_digits,
            ]),
            checked::product([t_coefficients, b.output_rank()]),
            Some(first.real_digit_count()),
            Some(first.padded_digit_count()),
            Some(second.real_digit_count()),
            Some(second.padded_digit_count()),
            Some(source_coefficients),
        ]
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .and_then(checked::sum)
        .ok_or(AkitaError::InvalidProof)?;
        let mut terms = Vec::new();
        terms
            .try_reserve_exact(term_capacity)
            .map_err(|_| AkitaError::InvalidProof)?;
        let mut row = 0usize;
        let mut claim = E::zero();
        let s_segment = self.first_segment;
        let t_segment = checked::sum([self.first_segment, 1]).ok_or(AkitaError::InvalidProof)?;
        let f_segment = checked::sum([self.first_segment, 2]).ok_or(AkitaError::InvalidProof)?;
        let h_segment = checked::sum([self.first_segment, 3]).ok_or(AkitaError::InvalidProof)?;
        let subcolumns = checked::exact_div(a.ring_dimension(), b.ring_dimension())
            .ok_or(AkitaError::InvalidProof)?;
        // Kernels depend on the matrix entry only, so blocks and slices that
        // reuse an entry share one kernel.
        let mut a_kernels = KernelCache::new(
            matrix,
            a.output_rank(),
            a.input_width(),
            a.ring_dimension(),
            alpha,
        )?;
        let mut b_kernels = KernelCache::new(
            matrix,
            b.output_rank(),
            b.input_width(),
            b.ring_dimension(),
            alpha,
        )?;
        for block in 0..self.profile.blocks.live_blocks {
            for a_row in 0..a.output_rank() {
                let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
                row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
                for position in 0..self.profile.blocks.positions_per_block {
                    let kernel = a_kernels.get(a_row, position)?;
                    let element =
                        checked::mul_add(block, self.profile.blocks.positions_per_block, position)
                            .ok_or(AkitaError::InvalidProof)?;
                    for (coefficient, &weight) in kernel.iter().enumerate() {
                        terms.push((
                            repacking.coefficient_address(s_segment, element, 0, coefficient)?,
                            batch * weight,
                        ));
                    }
                }
                let mut power = batch;
                for coefficient in 0..a.ring_dimension() {
                    let image_row = checked::mul_add(block, a.output_rank(), a_row)
                        .and_then(|image| {
                            checked::mul_add(image, subcolumns, coefficient / b.ring_dimension())
                        })
                        .ok_or(AkitaError::InvalidProof)?;
                    terms.extend(repacking.coefficient_weights(
                        t_segment,
                        image_row,
                        coefficient % b.ring_dimension(),
                        -power,
                    )?);
                    power *= alpha;
                }
            }
        }
        let geometry = self.profile.derive_slice_geometry()?;
        for block_range in geometry.block_ranges() {
            for b_row in 0..b.output_rank() {
                let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
                row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
                for (local_block, block) in block_range.clone().enumerate() {
                    for a_row in 0..a.output_rank() {
                        for subcolumn in 0..subcolumns {
                            let element = checked::mul_add(block, a.output_rank(), a_row)
                                .and_then(|image| checked::mul_add(image, subcolumns, subcolumn))
                                .ok_or(AkitaError::InvalidProof)?;
                            for digit in 0..self.profile.outer.digits.num_digits {
                                let column = checked::mul_add(local_block, a.output_rank(), a_row)
                                    .and_then(|image| {
                                        checked::mul_add(image, subcolumns, subcolumn)
                                    })
                                    .and_then(|image| {
                                        checked::mul_add(
                                            image,
                                            self.profile.outer.digits.num_digits,
                                            digit,
                                        )
                                    })
                                    .ok_or(AkitaError::InvalidProof)?;
                                let kernel = b_kernels.get(b_row, column)?;
                                for (coefficient, &weight) in kernel.iter().enumerate() {
                                    terms.push((
                                        repacking.coefficient_address(
                                            t_segment,
                                            element,
                                            digit,
                                            coefficient,
                                        )?,
                                        batch * weight,
                                    ));
                                }
                            }
                        }
                    }
                }
                // B rows are stacked in slice/row/coefficient order.
                let b_image_row = row
                    .checked_sub(
                        checked::product([self.profile.blocks.live_blocks, a.output_rank()])
                            .ok_or(AkitaError::InvalidProof)?,
                    )
                    .and_then(|index| index.checked_sub(1))
                    .ok_or(AkitaError::InvalidProof)?;
                let mut power = -batch;
                for coefficient in 0..b.ring_dimension() {
                    let image_coefficient =
                        checked::mul_add(b_image_row, b.ring_dimension(), coefficient)
                            .ok_or(AkitaError::InvalidProof)?;
                    scatter_negative_recomposition(
                        &mut terms,
                        repacking,
                        f_segment,
                        first.ring_dimension(),
                        first.input_coefficients(),
                        image_coefficient,
                        power,
                    )?;
                    power *= alpha;
                }
            }
        }
        for (map, source_segment, next_segment) in [
            (first, f_segment, Some(h_segment)),
            (second, h_segment, None),
        ] {
            let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
            row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
            let mut map_kernels =
                KernelCache::new(matrix, 1, map.input_width(), map.ring_dimension(), alpha)?;
            for column in 0..map.input_width() {
                let kernel = map_kernels.get(0, column)?;
                for (coefficient, &weight) in kernel.iter().enumerate() {
                    terms.push((
                        repacking.coefficient_address(source_segment, column, 0, coefficient)?,
                        batch * weight,
                    ));
                }
            }
            if let Some(next_segment) = next_segment {
                let mut power = -batch;
                for coefficient in 0..map.output_coefficients() {
                    scatter_negative_recomposition(
                        &mut terms,
                        repacking,
                        next_segment,
                        second.ring_dimension(),
                        second.input_coefficients(),
                        coefficient,
                        power,
                    )?;
                    power *= alpha;
                }
            } else {
                claim += batch
                    * payload
                        .iter()
                        .rev()
                        .fold(E::zero(), |sum, &coefficient| sum * alpha + coefficient);
            }
        }
        let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
        let equality = akita_algebra::offset_eq::OffsetEqWindow::new(point)?;
        let source_len = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
        for coefficient in 0..source_len {
            terms.push((
                repacking.coefficient_address(
                    s_segment,
                    coefficient / a.ring_dimension(),
                    0,
                    coefficient % a.ring_dimension(),
                )?,
                batch * equality.eval(coefficient),
            ));
        }
        claim += batch * evaluation;
        NativeRootWeights::new(repacking.domain_len(), terms, claim)
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_negative_recomposition<E: Field>(
    terms: &mut Vec<(usize, E)>,
    repacking: &RootCoefficientRepacking,
    segment: usize,
    degree: usize,
    image_coefficients: usize,
    coefficient: usize,
    scale: E,
) -> Result<(), AkitaError> {
    if coefficient >= image_coefficients {
        return Err(AkitaError::InvalidProof);
    }
    let mut power = scale;
    for bit in 0..128 {
        let compact = checked::mul_add(bit, image_coefficients, coefficient)
            .ok_or(AkitaError::InvalidProof)?;
        terms.push((
            repacking.coefficient_address(segment, compact / degree, 0, compact % degree)?,
            power,
        ));
        power += power;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root_repacking::RootRepackingSegment;
    use akita_params::sis::{
        sis_table_key_for_linf_bound, InnerCommitMatrixParams, OuterCommitMatrixParams,
        SisMatrixRole, SisTableDigest, DEFAULT_SIS_SECURITY_POLICY,
    };
    use akita_params::{
        BlockGeometry, CommitmentSliceCount, CommitmentSliceGeometry, GadgetDigits,
        GroupCommitPhaseParams, PolynomialGroupLayout, RelationRowGeometry, RoleParams,
        SisModulusProfileId,
    };
    use jolt_field::{Prime128Offset275 as F, Ring, Zero};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// The mixed end-to-end ordinary group, with `slices` B slices.
    fn profile(slices: CommitmentSliceCount) -> GroupCommitPhaseParams {
        let key = |role, degree| {
            sis_table_key_for_linf_bound(
                DEFAULT_SIS_SECURITY_POLICY,
                SisTableDigest::CURRENT,
                SisModulusProfileId::Q128Offset275,
                role,
                degree,
                8,
            )
            .unwrap()
        };
        let inner =
            InnerCommitMatrixParams::try_new_with_min_rank(key(SisMatrixRole::Inner, 128), 2)
                .unwrap();
        let geometry =
            CommitmentSliceGeometry::try_new(slices, 4, 1, inner.output_rank(), 43, 128, 64)
                .unwrap();
        let outer = OuterCommitMatrixParams::try_new_with_min_rank(
            key(SisMatrixRole::Outer, 64),
            geometry.physical_input_width(),
        )
        .unwrap();
        let profile = GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::singleton(10),
            blocks: BlockGeometry::new(8, 2, 4),
            outer_slice_count: slices,
            inner: RoleParams::new(GadgetDigits::new(3, 1), inner),
            outer: RoleParams::new(GadgetDigits::new(3, 43), outer),
        };
        profile.validate_frozen_precommit(128).unwrap();
        profile
    }

    fn field(rng: &mut StdRng) -> F {
        F::from_u128(rng.gen::<u128>() >> 1)
    }

    /// One and two B slices, with the ordinary segments first in the table and
    /// behind a 192-address prefix that misaligns every segment start against
    /// the 1024-address equality blocks.
    #[test]
    fn structured_ordinary_rows_match_the_flat_reference() {
        let setup = AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            setup_seed: [41u8; 32].into(),
        };
        let mut rng = StdRng::seed_from_u64(5);
        for slices in [CommitmentSliceCount::ONE, CommitmentSliceCount::TWO] {
            for prefix in [0, 3] {
                let profile = profile(slices);
                let first = usize::from(prefix > 0);
                let plan = OrdinaryNativeRootPlan::new(profile, first).unwrap();
                let mut segments = Vec::new();
                if prefix > 0 {
                    segments.push(RootRepackingSegment {
                        geometry: RelationRowGeometry::native(64).unwrap(),
                        elements: prefix,
                        digits: 1,
                    });
                }
                segments.extend_from_slice(plan.segments());
                let repacking = RootCoefficientRepacking::new(segments, 64).unwrap();
                let point = (0..10).map(|_| field(&mut rng)).collect::<Vec<_>>();
                let payload = (0..profile
                    .outer_compression_plan()
                    .unwrap()
                    .terminal_coefficients())
                    .map(|_| field(&mut rng))
                    .collect::<Vec<_>>();
                let batches = (0..plan.linear_batch_count())
                    .map(|_| field(&mut rng))
                    .collect::<Vec<_>>();
                let (alpha, evaluation) = (field(&mut rng), field(&mut rng));

                let structured = plan
                    .linear_weights(
                        &repacking, &setup, &point, evaluation, &payload, alpha, &batches,
                    )
                    .unwrap();
                let flat = plan
                    .reference_linear_weights(
                        &repacking, &setup, &point, evaluation, &payload, alpha, &batches,
                    )
                    .unwrap();
                assert_eq!(structured.claim(), flat.claim());
                assert!(structured.terms().is_empty());
                let mut dense = vec![F::zero(); repacking.domain_len()];
                for &(address, weight) in flat.terms() {
                    dense[address] += weight;
                }
                assert!(
                    structured.materialize().unwrap() == dense,
                    "{slices:?} {prefix}"
                );
                let num_vars = repacking.domain_len().trailing_zeros() as usize;
                for _ in 0..3 {
                    let r = (0..num_vars).map(|_| field(&mut rng)).collect::<Vec<_>>();
                    assert_eq!(
                        structured.evaluate(&r).unwrap(),
                        flat.evaluate(&r).unwrap(),
                        "{slices:?} {prefix}"
                    );
                }
            }
        }
    }
}
