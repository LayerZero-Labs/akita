//! Ordinary commitment preimages authenticated in a mixed native root.

use akita_algebra::ring::ResidueKernelPoint;
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::{CompressionChainPlan, GroupCommitPhaseParams, RelationRowGeometry};
use jolt_field::{CanonicalEncoding, Field, Prime128Offset275};

use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use crate::{derive_public_matrix_prefix, AkitaSetupDescriptor};

/// Four ordinary preimage families sharing the native root's outgoing table.
///
/// S is the original small coefficient table; T is the ordinary A image in
/// balanced base-eight digits; the remaining two families are the existing
/// commitment compression's negative-binary digits. Their `{-1,0}` predicates
/// are compression constraints, independent of the binary source agreement.
#[derive(Clone, Debug)]
pub struct OrdinaryNativeRootPlan {
    profile: GroupCommitPhaseParams,
    pub(super) first_segment: usize,
    segments: [RootRepackingSegment; 4],
    compression: CompressionChainPlan,
    linear_batch_count: usize,
}

impl OrdinaryNativeRootPlan {
    pub fn new(profile: GroupCommitPhaseParams, first_segment: usize) -> Result<Self, AkitaError> {
        profile.validate_frozen_precommit(128)?;
        if profile.group.num_polynomials() != 1
            || profile.inner.digits.num_digits != 1
            || profile.inner.digits.log_basis < 3
            || profile.outer.digits.log_basis != 3
            || profile.inner.matrix.sis_modulus_profile().modulus() != BINARY_SETUP_P
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root requires one ordinary small coefficient source and base-eight image digits".into(),
            ));
        }
        let a_degree = profile.inner.matrix.ring_dimension();
        let b_degree = profile.outer.matrix.ring_dimension();
        let subcolumns = checked::exact_div(a_degree, b_degree).ok_or(AkitaError::InvalidProof)?;
        let source_elements = checked::product([
            profile.blocks.live_blocks,
            profile.blocks.positions_per_block,
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let image_elements = checked::product([
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
            subcolumns,
        ])
        .ok_or(AkitaError::InvalidProof)?;
        let compression = profile.outer_compression_plan()?;
        let first = compression.maps().first().ok_or(AkitaError::InvalidProof)?;
        let second = compression.maps().get(1).ok_or(AkitaError::InvalidProof)?;
        if compression.maps().len() != 2 || compression.field_bits() != 128 {
            return Err(AkitaError::InvalidSetup(
                "mixed root compression geometry is unsupported".into(),
            ));
        }
        // The selected root grinding plan budgets a maximum native degree of
        // 648 for its shared ring-switch challenge.
        if [
            a_degree,
            b_degree,
            first.ring_dimension(),
            second.ring_dimension(),
        ]
        .into_iter()
        .any(|degree| degree > 648)
        {
            return Err(AkitaError::InvalidSetup(
                "mixed root native degree exceeds the root grinding budget".into(),
            ));
        }
        let segments = [
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(a_degree)?,
                elements: source_elements,
                digits: 1,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(b_degree)?,
                elements: image_elements,
                digits: profile.outer.digits.num_digits,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(first.ring_dimension())?,
                elements: first.input_width(),
                digits: 1,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::native(second.ring_dimension())?,
                elements: second.input_width(),
                digits: 1,
            },
        ];
        let linear_batch_count = checked::product([
            profile.blocks.live_blocks,
            profile.inner.matrix.output_rank(),
        ])
        .and_then(|a_rows| {
            profile
                .outer_slice_count
                .logical_output_rows(profile.outer.matrix.output_rank())
                .ok()
                .and_then(|b_rows| checked::sum([a_rows, b_rows, 3]))
        })
        .ok_or(AkitaError::InvalidProof)?;
        checked::sum([first_segment, segments.len()]).ok_or(AkitaError::InvalidProof)?;
        Ok(Self {
            profile,
            first_segment,
            segments,
            compression,
            linear_batch_count,
        })
    }

    pub fn segments(&self) -> &[RootRepackingSegment] {
        &self.segments
    }
    pub fn linear_batch_count(&self) -> usize {
        self.linear_batch_count
    }

    fn validate_repacking(&self, repacking: &RootCoefficientRepacking) -> Result<(), AkitaError> {
        let range = checked::range(self.first_segment, self.segments.len())
            .ok_or(AkitaError::InvalidProof)?;
        if repacking.segments().get(range) != Some(self.segments.as_slice()) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(())
    }

    /// Addresses of the existing compression digits requiring `w(w+1)=0`.
    pub fn negative_binary_addresses(
        &self,
        repacking: &RootCoefficientRepacking,
    ) -> Result<Vec<usize>, AkitaError> {
        self.validate_repacking(repacking)?;
        let mut addresses = Vec::new();
        for offset in [2, 3] {
            let segment =
                checked::sum([self.first_segment, offset]).ok_or(AkitaError::InvalidProof)?;
            let shape = self.segments.get(offset).ok_or(AkitaError::InvalidProof)?;
            let count = checked::product([
                shape.elements,
                shape.geometry.polynomial_modulus_dimension(),
            ])
            .ok_or(AkitaError::InvalidProof)?;
            addresses
                .try_reserve_exact(count)
                .map_err(|_| AkitaError::InvalidProof)?;
            for element in 0..shape.elements {
                for coefficient in 0..shape.geometry.polynomial_modulus_dimension() {
                    addresses.push(repacking.coefficient_address(
                        segment,
                        element,
                        0,
                        coefficient,
                    )?);
                }
            }
        }
        Ok(addresses)
    }

    /// Direct public A/B/compression and source-evaluation identities.
    ///
    /// Rows are A in block/row order, B in slice/row order, the two compression
    /// maps, then the scalar evaluation. The public matrix is the canonical
    /// ordinary setup prefix, reused with each role's own dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_weights<E: Field>(
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
        for block in 0..self.profile.blocks.live_blocks {
            for a_row in 0..a.output_rank() {
                let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
                row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
                for position in 0..self.profile.blocks.positions_per_block {
                    let kernel = ordinary_matrix_kernel(
                        matrix,
                        a_row,
                        a.input_width(),
                        position,
                        a.ring_dimension(),
                        alpha,
                    )?;
                    let element =
                        checked::mul_add(block, self.profile.blocks.positions_per_block, position)
                            .ok_or(AkitaError::InvalidProof)?;
                    for (coefficient, weight) in kernel.into_iter().enumerate() {
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
                                let kernel = ordinary_matrix_kernel(
                                    matrix,
                                    b_row,
                                    b.input_width(),
                                    column,
                                    b.ring_dimension(),
                                    alpha,
                                )?;
                                for (coefficient, weight) in kernel.into_iter().enumerate() {
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
            for column in 0..map.input_width() {
                let kernel = ordinary_matrix_kernel(
                    matrix,
                    0,
                    map.input_width(),
                    column,
                    map.ring_dimension(),
                    alpha,
                )?;
                for (coefficient, weight) in kernel.into_iter().enumerate() {
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

fn ordinary_matrix_kernel<E: Field>(
    matrix: &[Prime128Offset275],
    row: usize,
    width: usize,
    column: usize,
    degree: usize,
    alpha: E,
) -> Result<Vec<E>, AkitaError> {
    let start = checked::mul_add(row, width, column)
        .and_then(|entry| checked::product([entry, degree]))
        .ok_or(AkitaError::InvalidProof)?;
    let range = checked::range(start, degree).ok_or(AkitaError::InvalidProof)?;
    let coefficients = matrix
        .get(range)
        .ok_or(AkitaError::InvalidProof)?
        .iter()
        .map(|coefficient| {
            coefficient
                .to_u128_checked()
                .map(E::from_u128)
                .ok_or(AkitaError::InvalidProof)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut weights = vec![E::zero(); degree];
    ResidueKernelPoint::new(alpha, degree)?.field_kernel_into(&coefficients, &mut weights)?;
    Ok(weights)
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
