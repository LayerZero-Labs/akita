//! Test-only reference: the flat term-list construction of the ordinary rows.
//!
//! This is the compiler that preceded the structured families, kept term for
//! term. It pushes one `(address, weight)` pair per public coefficient weight;
//! the differential tests compare the structured families with it.

use akita_algebra::ring::ResidueKernelPoint;
use akita_error::{checked, AkitaError};
use jolt_field::Field;

use super::OrdinaryNativeRootPlan;
use crate::native_root_relation::NativeRootWeights;
use crate::root_repacking::RootCoefficientRepacking;
use crate::AkitaSetupDescriptor;

impl OrdinaryNativeRootPlan {
    /// Flat reference for [`OrdinaryNativeRootPlan::linear_weights`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reference_linear_weights<E: Field>(
        &self,
        repacking: &RootCoefficientRepacking,
        setup: &AkitaSetupDescriptor,
        point: &[E],
        evaluations: &[E],
        payload: &[E],
        alpha: E,
        batches: &[E],
    ) -> Result<NativeRootWeights<E>, AkitaError> {
        self.validate_repacking(repacking)?;
        if evaluations.len() != self.profile.group.num_polynomials()
            || point.len() != self.profile.group.num_vars()
            || batches.len() != self.linear_batch_count
            || payload.len() != self.compression.terminal_coefficients()
            || E::from_u128(self.profile.inner.matrix.sis_modulus_profile().modulus()) != E::zero()
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
        let matrix = self.setup_coefficients::<E>(required, setup)?;
        let matrix = matrix.as_slice();
        let source_coefficients = checked::product([
            self.segments
                .first()
                .ok_or(AkitaError::InvalidProof)?
                .elements,
            a.ring_dimension(),
            self.profile.inner.digits.num_digits,
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
                self.profile.group.num_polynomials(),
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
            checked::product([
                self.segments
                    .first()
                    .ok_or(AkitaError::InvalidProof)?
                    .elements,
                a.ring_dimension(),
                self.source_width.map_or(0, |width| {
                    width.bits() as usize + self.profile.inner.digits.num_digits
                }),
            ]),
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
        let owners = self.profile.group.num_polynomials();
        let blocks = self.profile.blocks.live_blocks;
        let positions = self.profile.blocks.positions_per_block;
        let digits = self.profile.inner.digits.num_digits;
        for owner in 0..owners {
            for block in 0..blocks {
                for a_row in 0..a.output_rank() {
                    let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
                    row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
                    let owner_block =
                        checked::mul_add(owner, blocks, block).ok_or(AkitaError::InvalidProof)?;
                    for position in 0..positions {
                        let element = checked::mul_add(owner_block, positions, position)
                            .ok_or(AkitaError::InvalidProof)?;
                        for digit in 0..digits {
                            let column = checked::mul_add(position, digits, digit)
                                .ok_or(AkitaError::InvalidProof)?;
                            let kernel = a_kernels.get(a_row, column)?;
                            for (coefficient, &weight) in kernel.iter().enumerate() {
                                terms.push((
                                    repacking.coefficient_address(
                                        s_segment,
                                        element,
                                        digit,
                                        coefficient,
                                    )?,
                                    batch * weight,
                                ));
                            }
                        }
                    }
                    let mut power = batch;
                    for coefficient in 0..a.ring_dimension() {
                        let image_row = checked::mul_add(owner_block, a.output_rank(), a_row)
                            .and_then(|image| {
                                checked::mul_add(
                                    image,
                                    subcolumns,
                                    coefficient / b.ring_dimension(),
                                )
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
        }
        let geometry = self.profile.derive_slice_geometry()?;
        for block_range in geometry.block_ranges() {
            for b_row in 0..b.output_rank() {
                let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
                row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
                for owner in 0..owners {
                    for (local_block, block) in block_range.clone().enumerate() {
                        for a_row in 0..a.output_rank() {
                            for subcolumn in 0..subcolumns {
                                let element = checked::mul_add(owner, blocks, block)
                                    .and_then(|block| {
                                        checked::mul_add(block, a.output_rank(), a_row)
                                    })
                                    .and_then(|image| {
                                        checked::mul_add(image, subcolumns, subcolumn)
                                    })
                                    .ok_or(AkitaError::InvalidProof)?;
                                for digit in 0..self.profile.outer.digits.num_digits {
                                    // B pads each owner's slice to max_blocks, not range.len().
                                    let column = checked::mul_add(
                                        owner,
                                        geometry.max_blocks_per_slice(),
                                        local_block,
                                    )
                                    .and_then(|block| {
                                        checked::mul_add(block, a.output_rank(), a_row)
                                    })
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
                }
                // B rows are stacked in slice/row/coefficient order.
                let b_image_row = row
                    .checked_sub(
                        checked::product([owners, blocks, a.output_rank()])
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
                        self.compression.field_bits(),
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
                        self.compression.field_bits(),
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
        let equality = akita_algebra::offset_eq::OffsetEqWindow::new(point)?;
        let source_len = checked::pow2(point.len()).ok_or(AkitaError::InvalidProof)?;
        let elements_per_owner =
            checked::product([blocks, positions]).ok_or(AkitaError::InvalidProof)?;
        for (owner, &evaluation) in evaluations.iter().enumerate() {
            let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
            row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
            for coefficient in 0..source_len {
                let element =
                    checked::mul_add(owner, elements_per_owner, coefficient / a.ring_dimension())
                        .ok_or(AkitaError::InvalidProof)?;
                terms.extend(repacking.coefficient_weights(
                    s_segment,
                    element,
                    coefficient % a.ring_dimension(),
                    batch * equality.eval(coefficient),
                )?);
            }
            claim += batch * evaluation;
        }
        if let Some(width) = self.source_width {
            let bit_segment =
                checked::sum([self.first_segment, 4]).ok_or(AkitaError::InvalidProof)?;
            let elements =
                checked::product([owners, elements_per_owner]).ok_or(AkitaError::InvalidProof)?;
            for element in 0..elements {
                let batch = *batches.get(row).ok_or(AkitaError::InvalidProof)?;
                row = checked::sum([row, 1]).ok_or(AkitaError::InvalidProof)?;
                let mut power = batch;
                for coefficient in 0..a.ring_dimension() {
                    // S + sum_j 2^j (-bit_j) = -2^(width-1).
                    terms.extend(repacking.coefficient_weights(
                        s_segment,
                        element,
                        coefficient,
                        power,
                    )?);
                    let mut bit_power = power;
                    for bit in 0..width.bits() as usize {
                        terms.push((
                            repacking.coefficient_address(
                                bit_segment,
                                element,
                                bit,
                                coefficient,
                            )?,
                            bit_power,
                        ));
                        bit_power += bit_power;
                    }
                    claim -= power * E::from_u128(width.offset());
                    power *= alpha;
                }
            }
        }
        if row != batches.len() {
            return Err(AkitaError::InvalidProof);
        }
        NativeRootWeights::new(repacking.domain_len(), terms, claim)
    }
}

/// Residue kernels of one public matrix role, each computed on first use.
///
/// Entry `(row, column)` is the degree-`degree` ring at flat offset
/// `(row * width + column) * degree` of the ordinary setup prefix.
struct KernelCache<'a, E: Field> {
    matrix: &'a [E],
    width: usize,
    degree: usize,
    point: ResidueKernelPoint<E>,
    kernels: Vec<Option<Vec<E>>>,
}

impl<'a, E: Field> KernelCache<'a, E> {
    fn new(
        matrix: &'a [E],
        rows: usize,
        width: usize,
        degree: usize,
        alpha: E,
    ) -> Result<Self, AkitaError> {
        let entries = checked::product([rows, width]).ok_or(AkitaError::InvalidProof)?;
        let mut kernels = Vec::new();
        kernels
            .try_reserve_exact(entries)
            .map_err(|_| AkitaError::InvalidProof)?;
        kernels.resize(entries, None);
        Ok(Self {
            matrix,
            width,
            degree,
            point: ResidueKernelPoint::new(alpha, degree)?,
            kernels,
        })
    }

    fn get(&mut self, row: usize, column: usize) -> Result<&[E], AkitaError> {
        if column >= self.width {
            return Err(AkitaError::InvalidProof);
        }
        let entry = checked::mul_add(row, self.width, column).ok_or(AkitaError::InvalidProof)?;
        let start = checked::product([entry, self.degree]).ok_or(AkitaError::InvalidProof)?;
        let range = checked::range(start, self.degree).ok_or(AkitaError::InvalidProof)?;
        let slot = self
            .kernels
            .get_mut(entry)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.is_none() {
            let coefficients = self
                .matrix
                .get(range)
                .ok_or(AkitaError::InvalidProof)?
                .to_vec();
            let mut weights = vec![E::zero(); self.degree];
            self.point.field_kernel_into(&coefficients, &mut weights)?;
            *slot = Some(weights);
        }
        slot.as_deref().ok_or(AkitaError::InvalidProof)
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
    bits: usize,
) -> Result<(), AkitaError> {
    if coefficient >= image_coefficients {
        return Err(AkitaError::InvalidProof);
    }
    let mut power = scale;
    for bit in 0..bits {
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
    use akita_params::ordinary_source::OrdinarySourceWidth;
    use akita_params::sis::{
        inner_coeff_linf_bounds, num_digits_inner, num_digits_open, sis_table_key_for_linf_bound,
        InnerCommitMatrixParams, OuterCommitMatrixParams, SisMatrixRole, SisTableDigest,
        SisTableKey, DEFAULT_SIS_SECURITY_POLICY,
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

    /// A signed `bits`-wide group with `slots` polynomials and multi-digit
    /// sources where `bits` needs them.
    fn signed_profile(
        bits: u32,
        slots: usize,
        slices: CommitmentSliceCount,
    ) -> GroupCommitPhaseParams {
        let width = OrdinarySourceWidth::new(bits).unwrap();
        let decomposition = width
            .decomposition(SisModulusProfileId::Q128Offset275)
            .unwrap();
        let inner_digits = num_digits_inner(decomposition, true);
        let outer_digits = num_digits_open(decomposition);
        let key = SisTableKey {
            policy: DEFAULT_SIS_SECURITY_POLICY,
            table_digest: SisTableDigest::CURRENT,
            modulus_profile: SisModulusProfileId::Q128Offset275,
            role: SisMatrixRole::Inner,
            ring_dimension: 64,
            coeff_linf_bound: inner_coeff_linf_bounds(SisModulusProfileId::Q128Offset275, 64)[0],
        };
        let inner = InnerCommitMatrixParams::try_new_with_min_rank(key, inner_digits).unwrap();
        let geometry = CommitmentSliceGeometry::try_new(
            slices,
            2,
            slots,
            inner.output_rank(),
            outer_digits,
            64,
            64,
        )
        .unwrap();
        let outer = OuterCommitMatrixParams::try_new_with_min_rank(
            SisTableKey {
                role: SisMatrixRole::Outer,
                coeff_linf_bound: 7,
                ..key
            },
            geometry.physical_input_width(),
        )
        .unwrap();
        let profile = GroupCommitPhaseParams {
            version: GroupCommitPhaseParams::VERSION,
            group: PolynomialGroupLayout::new(7, slots),
            blocks: BlockGeometry::new(2, 1, 2),
            outer_slice_count: slices,
            inner: RoleParams::new(GadgetDigits::new(3, inner_digits), inner),
            outer: RoleParams::new(GadgetDigits::new(3, outer_digits), outer),
        };
        width.validate_profile(&profile).unwrap();
        profile
    }

    fn field(rng: &mut StdRng) -> F {
        F::from_u128(rng.gen::<u128>() >> 1)
    }

    /// The single-polynomial mixed root with one and two B slices, and signed
    /// plans with several polynomials and multi-digit sources. Each sits
    /// first in the table and behind a 192-address prefix that misaligns
    /// every segment start against the 1024-address equality blocks.
    #[test]
    fn structured_ordinary_rows_match_the_flat_reference() {
        let setup = AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
            setup_seed: [41u8; 32].into(),
        };
        let mut rng = StdRng::seed_from_u64(5);
        let mut cases = Vec::new();
        for slices in [CommitmentSliceCount::ONE, CommitmentSliceCount::TWO] {
            cases.push((profile(slices), None));
            for (bits, slots) in [(2, 3), (9, 3), (33, 1)] {
                cases.push((
                    signed_profile(bits, slots, slices),
                    Some(OrdinarySourceWidth::new(bits).unwrap()),
                ));
            }
        }
        for (profile, width) in cases {
            for prefix in [0, 3] {
                let first = usize::from(prefix > 0);
                let plan = match width {
                    None => OrdinaryNativeRootPlan::new(profile, first),
                    Some(width) => OrdinaryNativeRootPlan::new_signed(profile, first, width),
                }
                .unwrap();
                let label = format!(
                    "{:?} {width:?} {} {prefix}",
                    profile.outer_slice_count,
                    profile.group.num_polynomials()
                );
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
                let point = (0..profile.group.num_vars())
                    .map(|_| field(&mut rng))
                    .collect::<Vec<_>>();
                let evaluations = (0..profile.group.num_polynomials())
                    .map(|_| field(&mut rng))
                    .collect::<Vec<_>>();
                let payload = (0..profile
                    .outer_compression_plan()
                    .unwrap()
                    .terminal_coefficients())
                    .map(|_| field(&mut rng))
                    .collect::<Vec<_>>();
                let batches = (0..plan.linear_batch_count())
                    .map(|_| field(&mut rng))
                    .collect::<Vec<_>>();
                let alpha = field(&mut rng);

                let structured = plan
                    .linear_weights(
                        &repacking,
                        &setup,
                        &point,
                        &evaluations,
                        &payload,
                        alpha,
                        &batches,
                    )
                    .unwrap();
                let flat = plan
                    .reference_linear_weights(
                        &repacking,
                        &setup,
                        &point,
                        &evaluations,
                        &payload,
                        alpha,
                        &batches,
                    )
                    .unwrap();
                assert_eq!(structured.claim(), flat.claim(), "{label}");
                assert_eq!(
                    structured.terms().is_empty(),
                    profile.inner.digits.num_digits == 1,
                    "{label}"
                );
                let mut dense = vec![F::zero(); repacking.domain_len()];
                for &(address, weight) in flat.terms() {
                    dense[address] += weight;
                }
                assert!(structured.materialize().unwrap() == dense, "{label}");
                let num_vars = repacking.domain_len().trailing_zeros() as usize;
                for _ in 0..3 {
                    let r = (0..num_vars).map(|_| field(&mut rng)).collect::<Vec<_>>();
                    assert_eq!(
                        structured.evaluate(&r).unwrap(),
                        flat.evaluate(&r).unwrap(),
                        "{label}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod p64_tests {
    use super::*;
    use crate::binary_source::binary_ordinary_profile_fixture;
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as F, Ring, Zero};
    type E = Ext2<F>;

    #[test]
    fn p64_mixed_ordinary_structured_matches_the_flat_reference() {
        for signed in [
            None,
            Some((2, 3)),
            Some((9, 3)),
            Some((18, 5)),
            Some((33, 1)),
        ] {
            let profile = binary_ordinary_profile_fixture(BinaryFieldProfile::P64, signed);
            if let Some((bits, _)) = signed {
                let width = akita_params::ordinary_source::OrdinarySourceWidth::new(bits).unwrap();
                let mut wrong_depth = profile;
                wrong_depth.outer.digits.num_digits = 43;
                assert!(width.validate_profile(&wrong_depth).is_err());
                let mut mixed_field = profile;
                mixed_field.outer =
                    binary_ordinary_profile_fixture(BinaryFieldProfile::P128, signed).outer;
                assert!(width.validate_profile(&mixed_field).is_err());
            }
            let plan = match signed {
                Some((bits, _)) => OrdinaryNativeRootPlan::new_signed(
                    profile,
                    0,
                    akita_params::ordinary_source::OrdinarySourceWidth::new(bits).unwrap(),
                ),
                None => OrdinaryNativeRootPlan::new(profile, 0),
            }
            .unwrap();
            let compression = profile.outer_compression_plan().unwrap();
            assert_eq!(compression.field_bits(), 64);
            assert_eq!(compression.terminal_coefficients(), 32);
            assert_eq!(
                plan.segments()[2].geometry.polynomial_modulus_dimension(),
                64
            );
            assert_eq!(
                plan.segments()[3].geometry.polynomial_modulus_dimension(),
                32
            );
            let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
            if signed.is_none() {
                assert_eq!(
                    plan.segments()
                        .iter()
                        .map(|shape| shape.elements
                            * shape.digits
                            * shape.geometry.polynomial_modulus_dimension())
                        .collect::<Vec<_>>(),
                    vec![1024, 11_264, 8192, 4096]
                );
                assert_eq!(plan.linear_batch_count(), 9);
                assert_eq!(
                    plan.negative_binary_addresses(&repacking).unwrap().len(),
                    12_288
                );
                assert_eq!(compression.unpacked_witness_bytes().unwrap(), 12_288);
            }
            assert_eq!(
                compression.field_bytes() * compression.terminal_coefficients(),
                256
            );
            let setup = AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
                setup_seed: [41; 32].into(),
            };
            let ext = |i: usize| E::from_base_fn(|c| F::from_u64((17 + i * 31 + c * 43) as u64));
            let point = (0..profile.group.num_vars()).map(ext).collect::<Vec<_>>();
            let payload = (0..compression.terminal_coefficients())
                .map(ext)
                .collect::<Vec<_>>();
            let evaluations = (0..profile.group.num_polynomials())
                .map(|slot| ext(300 + slot))
                .collect::<Vec<_>>();
            let batches = (0..plan.linear_batch_count()).map(ext).collect::<Vec<_>>();
            let structured = plan
                .linear_weights(
                    &repacking,
                    &setup,
                    &point,
                    &evaluations,
                    &payload,
                    ext(400),
                    &batches,
                )
                .unwrap();
            let flat = plan
                .reference_linear_weights(
                    &repacking,
                    &setup,
                    &point,
                    &evaluations,
                    &payload,
                    ext(400),
                    &batches,
                )
                .unwrap();
            assert_eq!(structured.claim(), flat.claim());
            let mut dense = vec![E::zero(); repacking.domain_len()];
            for &(address, weight) in flat.terms() {
                dense[address] += weight;
            }
            assert_eq!(structured.materialize().unwrap(), dense);
            let r = (0..repacking.domain_len().trailing_zeros())
                .map(|i| ext(500 + i as usize))
                .collect::<Vec<_>>();
            assert_eq!(structured.evaluate(&r).unwrap(), flat.evaluate(&r).unwrap());
        }
    }

    #[test]
    fn p64_second_reading_matches_independent_flat_rows() {
        for (bits, slots) in [(2, 3), (9, 3), (18, 5), (33, 1)] {
            let profile =
                binary_ordinary_profile_fixture(BinaryFieldProfile::P64, Some((bits, slots)));
            let plan = OrdinaryNativeRootPlan::new_signed(
                profile,
                0,
                akita_params::ordinary_source::OrdinarySourceWidth::new(bits).unwrap(),
            )
            .unwrap();
            let repacking = RootCoefficientRepacking::new(plan.segments().to_vec(), 64).unwrap();
            let setup = AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
                setup_seed: [41; 32].into(),
            };
            let ext = |i| E::from_base_fn(|c| F::from_u64(i + 43 * u64::try_from(c).unwrap()));
            let mut point = (0..profile.group.num_vars())
                .map(|i| ext(17 + u64::try_from(i).unwrap()))
                .collect::<Vec<_>>();
            let evaluations = (0..slots)
                .map(|i| ext(300 + u64::try_from(i).unwrap()))
                .collect::<Vec<_>>();
            let coefficients = (0..slots)
                .map(|i| ext(500 + u64::try_from(i).unwrap()))
                .collect::<Vec<_>>();
            let payload = vec![
                E::zero();
                profile
                    .outer_compression_plan()
                    .unwrap()
                    .terminal_coefficients()
            ];
            let mut batches = vec![E::zero(); plan.linear_batch_count()];
            let start = plan.linear_batch_count() - plan.segments()[0].elements - slots;
            batches[start..start + slots].copy_from_slice(&coefficients);
            for pass in 0..2 {
                point[0] += ext(1 + pass);
                let weights = plan
                    .evaluation_weights(&repacking, &point, &evaluations, &coefficients)
                    .unwrap();
                let reference = plan
                    .reference_linear_weights(
                        &repacking,
                        &setup,
                        &point,
                        &evaluations,
                        &payload,
                        ext(700),
                        &batches,
                    )
                    .unwrap();
                assert_eq!(weights.claim(), reference.claim());
                assert_eq!(
                    weights.materialize().unwrap(),
                    reference.materialize().unwrap()
                );
                let mut bad_point = point.clone();
                bad_point[0] += E::from_u64(1);
                let changed = plan
                    .evaluation_weights(&repacking, &bad_point, &evaluations, &coefficients)
                    .unwrap();
                assert_ne!(weights.materialize().unwrap(), changed.materialize().unwrap());
                let mut bad_values = evaluations.clone();
                bad_values[slots - 1] += E::from_u64(1);
                let changed = plan
                    .evaluation_weights(&repacking, &point, &bad_values, &coefficients)
                    .unwrap();
                assert_ne!(weights.claim(), changed.claim());
            }
        }
    }
}
