//! Shared table traversal for runtime-selected packed field kernels.

use super::{PackedField, PackedFpExt2, PackedFpExt4, PackedValue};
use crate::{ExtField, Fp32, Fp64, FpExt2, FpExt2Config, FpExt4};

pub(super) type CoefficientSlices<'a, const P: u32> = [&'a [Fp32<P>]; 4];
pub(super) type Fp64CoefficientSlices<'a, const P: u64> = [&'a [Fp64<P>]; 2];

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn fold_fp_ext4_fp32_packed<const P: u32, PF>(
    left: [&mut [Fp32<P>]; 4],
    right: CoefficientSlices<'_, P>,
    challenge: FpExt4<Fp32<P>>,
) where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let len = validate_fold_slices(&left, &right, PF::WIDTH);
    let left = left.map(|slice| slice.as_mut_ptr());
    let right = right.map(|slice| slice.as_ptr());
    let left_read = left.map(|pointer| pointer.cast_const());
    let challenge = PackedFpExt4::<Fp32<P>, PF>::broadcast(challenge);

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes complete packed chunks in every
        // coefficient slice at `row`.
        let even = unsafe { read_packed_fp_ext4::<P, PF>(left_read, row) };
        let odd = unsafe { read_packed_fp_ext4::<P, PF>(right, row) };
        let folded = even + (odd - even) * challenge;
        // SAFETY: the output aliases only the already-read left input chunk.
        unsafe { write_packed_fp_ext4(left, row, folded) };
    }
}

#[inline(always)]
pub(super) unsafe fn compute_product_round_packed<const P: u32, PF>(
    witness_0: CoefficientSlices<'_, P>,
    witness_1: CoefficientSlices<'_, P>,
    factor_0: CoefficientSlices<'_, P>,
    factor_1: CoefficientSlices<'_, P>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>)
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let slices = [&witness_0, &witness_1, &factor_0, &factor_1];
    let len = validate_product_round_slices(&slices, PF::WIDTH);
    let witness_0 = witness_0.map(|slice| slice.as_ptr());
    let witness_1 = witness_1.map(|slice| slice.as_ptr());
    let factor_0 = factor_0.map(|slice| slice.as_ptr());
    let factor_1 = factor_1.map(|slice| slice.as_ptr());
    let zero = FpExt4::<Fp32<P>>::zero();
    let mut constant = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero);
    let mut quadratic = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero);

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes a complete packed chunk in every
        // coefficient slice for this row.
        let witness_0 = unsafe { read_packed_fp_ext4::<P, PF>(witness_0, row) };
        let witness_1 = unsafe { read_packed_fp_ext4::<P, PF>(witness_1, row) };
        let factor_0 = unsafe { read_packed_fp_ext4::<P, PF>(factor_0, row) };
        let factor_1 = unsafe { read_packed_fp_ext4::<P, PF>(factor_1, row) };
        constant = constant + witness_0 * factor_0;
        quadratic = quadratic + (witness_1 - witness_0) * (factor_1 - factor_0);
    }

    sum_product_round_lanes(constant, quadratic)
}

/// Compute an equality-weighted batch of quadratic or quartic affine products.
#[inline(always)]
pub(super) unsafe fn compute_weighted_affine_product_round_packed<
    const P: u32,
    PF,
    const LANES: usize,
>(
    left: [CoefficientSlices<'_, P>; LANES],
    right: [CoefficientSlices<'_, P>; LANES],
    equality: CoefficientSlices<'_, P>,
    arity: usize,
    parent_weights: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let len = validate_weighted_affine_product_slices(
        &left,
        &right,
        &equality,
        arity,
        parent_weights.len(),
        PF::WIDTH,
    );
    let left = left.map(|lane| lane.map(|slice| slice.as_ptr()));
    let right = right.map(|lane| lane.map(|slice| slice.as_ptr()));
    let equality = equality.map(|slice| slice.as_ptr());
    let zero = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::zero());
    let mut sums = [zero; 5];

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes one complete packed chunk for every
        // lane and equality coefficient slice.
        let left: [PackedFpExt4<Fp32<P>, PF>; LANES] =
            std::array::from_fn(|lane| unsafe { read_packed_fp_ext4::<P, PF>(left[lane], row) });
        let right: [PackedFpExt4<Fp32<P>, PF>; LANES] =
            std::array::from_fn(|lane| unsafe { read_packed_fp_ext4::<P, PF>(right[lane], row) });
        let equality = unsafe { read_packed_fp_ext4::<P, PF>(equality, row) };

        for (parent, &parent_weight) in parent_weights.iter().enumerate() {
            let first_lane = parent * arity;
            let polynomial = match arity {
                2 => packed_quadratic_affine_product(
                    [left[first_lane], left[first_lane + 1]],
                    [right[first_lane], right[first_lane + 1]],
                    zero,
                ),
                4 => packed_quartic_affine_product(
                    [
                        left[first_lane],
                        left[first_lane + 1],
                        left[first_lane + 2],
                        left[first_lane + 3],
                    ],
                    [
                        right[first_lane],
                        right[first_lane + 1],
                        right[first_lane + 2],
                        right[first_lane + 3],
                    ],
                    zero,
                ),
                _ => unreachable!("validated affine-product arity"),
            };
            let scale = equality * PackedFpExt4::broadcast(parent_weight);
            for degree in 0..=arity {
                sums[degree] = sums[degree] + scale * polynomial[degree];
            }
        }
    }

    sums.map(sum_fp32_lanes)
}

/// Compute an equality-weighted polynomial-composition round.
#[inline(always)]
pub(super) unsafe fn compute_weighted_affine_polynomial_round_packed<const P: u32, PF>(
    left: CoefficientSlices<'_, P>,
    right: CoefficientSlices<'_, P>,
    equality: CoefficientSlices<'_, P>,
    polynomial_coefficients: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let len = validate_weighted_affine_polynomial_slices(
        &left,
        &right,
        &equality,
        polynomial_coefficients.len(),
        PF::WIDTH,
    );
    let left = left.map(|slice| slice.as_ptr());
    let right = right.map(|slice| slice.as_ptr());
    let equality = equality.map(|slice| slice.as_ptr());
    let zero_value = FpExt4::<Fp32<P>>::zero();
    let zero = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero_value);
    let coefficients: [PackedFpExt4<Fp32<P>, PF>; 5] = std::array::from_fn(|degree| {
        PackedFpExt4::broadcast(
            polynomial_coefficients
                .get(degree)
                .copied()
                .unwrap_or(zero_value),
        )
    });
    let mut sums = [zero; 5];

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes one complete packed chunk in every
        // value and equality coefficient slice.
        let left = unsafe { read_packed_fp_ext4::<P, PF>(left, row) };
        let right = unsafe { read_packed_fp_ext4::<P, PF>(right, row) };
        let equality = unsafe { read_packed_fp_ext4::<P, PF>(equality, row) };
        let composed = packed_polynomial_with_affine(coefficients, left, right - left);
        for degree in 0..polynomial_coefficients.len() {
            sums[degree] = sums[degree] + equality * composed[degree];
        }
    }

    sums.map(sum_fp32_lanes)
}

#[inline(always)]
fn packed_polynomial_with_affine<const P: u32, PF>(
    coefficients: [PackedFpExt4<Fp32<P>, PF>; 5],
    offset: PackedFpExt4<Fp32<P>, PF>,
    slope: PackedFpExt4<Fp32<P>, PF>,
) -> [PackedFpExt4<Fp32<P>, PF>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let [constant, linear, quadratic, cubic, quartic] = coefficients;
    let two_quadratic = quadratic + quadratic;
    let three_cubic = cubic + cubic + cubic;
    let four_quartic = (quartic + quartic) + (quartic + quartic);
    let six_quartic = four_quartic + quartic + quartic;
    let value =
        constant + offset * (linear + offset * (quadratic + offset * (cubic + offset * quartic)));
    let first_derivative =
        linear + offset * (two_quadratic + offset * (three_cubic + offset * four_quartic));
    let second_divided_derivative = quadratic + offset * (three_cubic + offset * six_quartic);
    let third_divided_derivative = cubic + offset * four_quartic;
    let slope_squared = slope * slope;
    [
        value,
        slope * first_derivative,
        slope_squared * second_divided_derivative,
        slope_squared * slope * third_divided_derivative,
        slope_squared * slope_squared * quartic,
    ]
}

/// Compute the explicit prefix of a compact class-indexed product round.
#[inline(always)]
pub(super) unsafe fn compute_compact_affine_product_round_packed<
    const P: u32,
    PF,
    const LANES: usize,
>(
    ordered_pair_indices: &[u16],
    folded_pair_rows: &[[FpExt4<Fp32<P>>; LANES]],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    arity: usize,
    parent_weights: &[FpExt4<Fp32<P>>],
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let quartet_count = ordered_pair_indices.len().div_ceil(2);
    validate_compact_affine_product_inputs(
        folded_pair_rows,
        first_equality,
        second_equality,
        quartet_count,
        arity,
        parent_weights.len(),
        PF::WIDTH,
    );
    let zero = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::zero());
    let mut sums = [zero; 5];

    for quartet in (0..quartet_count).step_by(PF::WIDTH) {
        let left: [PackedFpExt4<Fp32<P>, PF>; LANES] = std::array::from_fn(|child| {
            packed_class_values::<P, PF, LANES>(folded_pair_rows, child, |packed_lane| {
                usize::from(ordered_pair_indices[2 * (quartet + packed_lane)])
            })
        });
        let right: [PackedFpExt4<Fp32<P>, PF>; LANES] = std::array::from_fn(|child| {
            packed_class_values::<P, PF, LANES>(folded_pair_rows, child, |packed_lane| {
                ordered_pair_indices
                    .get(2 * (quartet + packed_lane) + 1)
                    .copied()
                    .map(usize::from)
                    .unwrap_or(0)
            })
        });
        let first = packed_extension_values::<P, PF>(|packed_lane| {
            first_equality[quartet % first_equality.len() + packed_lane]
        });
        let second = PackedFpExt4::broadcast(second_equality[quartet / first_equality.len()]);
        let equality = first * second;

        for (parent, &parent_weight) in parent_weights.iter().enumerate() {
            let first_lane = parent * arity;
            let polynomial = match arity {
                2 => packed_quadratic_affine_product(
                    [left[first_lane], left[first_lane + 1]],
                    [right[first_lane], right[first_lane + 1]],
                    zero,
                ),
                4 => packed_quartic_affine_product(
                    [
                        left[first_lane],
                        left[first_lane + 1],
                        left[first_lane + 2],
                        left[first_lane + 3],
                    ],
                    [
                        right[first_lane],
                        right[first_lane + 1],
                        right[first_lane + 2],
                        right[first_lane + 3],
                    ],
                    zero,
                ),
                _ => unreachable!("validated affine-product arity"),
            };
            let scale = equality * PackedFpExt4::broadcast(parent_weight);
            for degree in 0..=arity {
                sums[degree] = sums[degree] + scale * polynomial[degree];
            }
        }
    }
    sums.map(sum_fp32_lanes)
}

/// Compute an explicit class-coded polynomial round.
#[inline(always)]
pub(super) unsafe fn compute_class_coded_affine_polynomial_round_packed<const P: u32, PF>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    class_taylor_coefficients: &[[FpExt4<Fp32<P>>; 4]],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let pair_count = class_codes.len() / 2;
    validate_class_coded_affine_polynomial_inputs(
        class_codes,
        class_values.len(),
        class_taylor_coefficients.len(),
        first_equality,
        second_equality,
        degree,
        PF::WIDTH,
    );
    let zero = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::zero());
    let mut sums = [zero; 5];

    for pair in (0..pair_count).step_by(PF::WIDTH) {
        let left_classes = std::array::from_fn::<_, 16, _>(|lane| {
            if lane < PF::WIDTH && pair + lane < pair_count {
                usize::from(class_codes[2 * (pair + lane)])
            } else {
                0
            }
        });
        let right_classes = std::array::from_fn::<_, 16, _>(|lane| {
            if lane < PF::WIDTH && pair + lane < pair_count {
                usize::from(class_codes[2 * (pair + lane) + 1])
            } else {
                0
            }
        });
        let left = packed_extension_values::<P, PF>(|lane| class_values[left_classes[lane]]);
        let right = packed_extension_values::<P, PF>(|lane| class_values[right_classes[lane]]);
        let taylor: [PackedFpExt4<Fp32<P>, PF>; 4] = std::array::from_fn(|coefficient| {
            packed_extension_values::<P, PF>(|lane| {
                class_taylor_coefficients[left_classes[lane]][coefficient]
            })
        });
        let first = packed_extension_values::<P, PF>(|lane| {
            if pair + lane < pair_count {
                first_equality[(pair + lane) % first_equality.len()]
            } else {
                FpExt4::zero()
            }
        });
        let second = PackedFpExt4::broadcast(second_equality[pair / first_equality.len()]);
        let equality = first * second;
        let delta = right - left;
        let delta_squared = delta * delta;
        let coefficients = match degree {
            2 => [taylor[0], taylor[1] * delta, delta_squared, zero, zero],
            4 => {
                let delta_cubed = delta_squared * delta;
                [
                    taylor[0],
                    taylor[1] * delta,
                    taylor[2] * delta_squared,
                    taylor[3] * delta_cubed,
                    delta_squared * delta_squared,
                ]
            }
            _ => unreachable!("validated direct-range polynomial degree"),
        };
        for coefficient in 0..=degree {
            sums[coefficient] = sums[coefficient] + equality * coefficients[coefficient];
        }
    }
    sums.map(sum_fp32_lanes)
}

/// Fold adjacent value pairs and compute the next direct-range polynomial round.
#[inline(always)]
pub(super) unsafe fn fold_and_compute_sparse_affine_polynomial_round_packed<const P: u32, PF>(
    values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let pair_count = validate_sparse_affine_polynomial_shape(
        values.len(),
        folded_values,
        first_equality,
        second_equality,
        degree,
        PF::WIDTH,
    );
    unsafe {
        fold_and_compute_sparse_affine_polynomial_round_from_source_packed::<P, PF>(
            pair_count,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
            |index| values[index],
        )
    }
}

/// Fold class-coded value pairs and compute the next direct-range polynomial round.
#[inline(always)]
pub(super) unsafe fn fold_class_coded_and_compute_sparse_affine_polynomial_round_packed<
    const P: u32,
    PF,
>(
    class_codes: &[u16],
    class_values: &[FpExt4<Fp32<P>>],
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    assert!(
        class_codes
            .iter()
            .all(|&class| usize::from(class) < class_values.len()),
        "class code exceeds the prepared value table"
    );
    let pair_count = validate_sparse_affine_polynomial_shape(
        class_codes.len(),
        folded_values,
        first_equality,
        second_equality,
        degree,
        PF::WIDTH,
    );
    unsafe {
        fold_and_compute_sparse_affine_polynomial_round_from_source_packed::<P, PF>(
            pair_count,
            folded_values,
            first_equality,
            second_equality,
            challenge,
            degree,
            |index| class_values[usize::from(class_codes[index])],
        )
    }
}

/// Fold one binding-order Stage 2 coefficient coordinate and compute the next
/// norm and ordinary-relation round in the same packed traversal.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) unsafe fn fold_and_compute_stage2_coefficient_round_packed<const P: u32, PF>(
    witness: [&mut [Fp32<P>]; 4],
    live_lane_count: usize,
    old_coefficient_count: usize,
    next_alpha_factor: &[FpExt4<Fp32<P>>],
    relation_lane_weights: &[FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    include_norm_linear: bool,
) -> ([FpExt4<Fp32<P>>; 3], [FpExt4<Fp32<P>>; 3])
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    assert!(old_coefficient_count.is_power_of_two() && old_coefficient_count >= 4);
    assert!(witness
        .iter()
        .all(|coefficient| coefficient.len() == live_lane_count * old_coefficient_count));
    assert!(relation_lane_weights.len() >= live_lane_count);
    assert!(!first_equality.is_empty() && first_equality.len().is_power_of_two());
    assert!(!second_equality.is_empty());

    let next_coefficient_count = old_coefficient_count / 2;
    let next_pair_count = next_coefficient_count / 2;
    assert_eq!(next_alpha_factor.len(), next_coefficient_count);
    assert!(
        first_equality.len() * second_equality.len() >= live_lane_count * next_pair_count,
        "split equality table does not cover the live Stage 2 rows"
    );

    let witness = witness.map(|coefficient| coefficient.as_mut_ptr());
    let witness_read = witness.map(|pointer| pointer.cast_const());
    let zero_value = FpExt4::<Fp32<P>>::zero();
    let zero = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero_value);
    let one = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::one());
    let challenge = PackedFpExt4::<Fp32<P>, PF>::broadcast(challenge);
    let mut norm = [zero; 3];
    let mut relation = [zero; 3];

    let old_half = live_lane_count * next_coefficient_count;
    let next_half = live_lane_count * next_pair_count;
    let lanes_use_binding_order = live_lane_count == relation_lane_weights.len();
    for stored_pair in 0..next_pair_count {
        let logical_pair = reverse_power_of_two_index(stored_pair, next_pair_count);
        let pair_start = stored_pair * live_lane_count;
        let alpha_0 = PackedFpExt4::broadcast(next_alpha_factor[stored_pair]);
        let alpha_1 = PackedFpExt4::broadcast(next_alpha_factor[stored_pair + next_pair_count]);
        let alpha_delta = alpha_1 - alpha_0;

        for stored_lane in (0..live_lane_count).step_by(PF::WIDTH) {
            let active_lanes = (live_lane_count - stored_lane).min(PF::WIDTH);
            let row_0 = pair_start + stored_lane;
            let row_1 = row_0 + next_half;
            let load = |row| unsafe {
                if active_lanes == PF::WIDTH {
                    read_packed_fp_ext4::<P, PF>(witness_read, row)
                } else {
                    gather_packed_fp_ext4::<P, PF>(witness_read, active_lanes, |lane| row + lane)
                }
            };
            let source_00 = load(row_0);
            let source_01 = load(row_0 + old_half);
            let source_10 = load(row_1);
            let source_11 = load(row_1 + old_half);
            let folded_0 = source_00 + (source_01 - source_00) * challenge;
            let folded_1 = source_10 + (source_11 - source_10) * challenge;

            if active_lanes == PF::WIDTH {
                unsafe {
                    write_packed_fp_ext4(witness, row_0, folded_0);
                    write_packed_fp_ext4(witness, row_1, folded_1);
                }
            } else {
                for lane in 0..active_lanes {
                    unsafe {
                        write_scattered_fp_ext4_lane(witness, row_0 + lane, folded_0, lane);
                        write_scattered_fp_ext4_lane(witness, row_1 + lane, folded_1, lane);
                    }
                }
            }

            let first = packed_extension_values::<P, PF>(|lane| {
                if lane >= active_lanes {
                    return zero_value;
                }
                let lane = stored_lane + lane;
                let logical_lane = if lanes_use_binding_order {
                    reverse_power_of_two_index(lane, live_lane_count)
                } else {
                    lane
                };
                let address = logical_lane * next_pair_count + logical_pair;
                first_equality[address & (first_equality.len() - 1)]
            });
            let second = packed_extension_values::<P, PF>(|lane| {
                if lane >= active_lanes {
                    return zero_value;
                }
                let lane = stored_lane + lane;
                let logical_lane = if lanes_use_binding_order {
                    reverse_power_of_two_index(lane, live_lane_count)
                } else {
                    lane
                };
                let address = logical_lane * next_pair_count + logical_pair;
                second_equality[address / first_equality.len()]
            });
            let equality = first * second;
            let witness_delta = folded_1 - folded_0;
            norm[0] = norm[0] + equality * folded_0 * (folded_0 + one);
            if include_norm_linear {
                norm[1] = norm[1] + equality * witness_delta * (folded_0 + folded_0 + one);
            }
            norm[2] = norm[2] + equality * witness_delta * witness_delta;

            let lane_weight = packed_extension_values::<P, PF>(|lane| {
                if lane < active_lanes {
                    relation_lane_weights[stored_lane + lane]
                } else {
                    zero_value
                }
            });
            relation[0] = relation[0] + lane_weight * folded_0 * alpha_0;
            relation[1] =
                relation[1] + lane_weight * (folded_0 * alpha_delta + witness_delta * alpha_0);
            relation[2] = relation[2] + lane_weight * witness_delta * alpha_delta;
        }
    }

    (norm.map(sum_fp32_lanes), relation.map(sum_fp32_lanes))
}

#[inline(always)]
fn reverse_power_of_two_index(index: usize, len: usize) -> usize {
    debug_assert!(len.is_power_of_two());
    if len <= 1 {
        0
    } else {
        index.reverse_bits() >> (usize::BITS - len.trailing_zeros())
    }
}

#[inline(always)]
unsafe fn fold_and_compute_sparse_affine_polynomial_round_from_source_packed<const P: u32, PF>(
    pair_count: usize,
    folded_values: &mut [FpExt4<Fp32<P>>],
    first_equality: &[FpExt4<Fp32<P>>],
    second_equality: &[FpExt4<Fp32<P>>],
    challenge: FpExt4<Fp32<P>>,
    degree: usize,
    mut value_at: impl FnMut(usize) -> FpExt4<Fp32<P>>,
) -> [FpExt4<Fp32<P>>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let zero = PackedFpExt4::<Fp32<P>, PF>::broadcast(FpExt4::zero());
    let two = PackedFpExt4::broadcast(FpExt4::from_u64(2));
    let eighteen = PackedFpExt4::broadcast(FpExt4::from_u64(18));
    let twenty = PackedFpExt4::broadcast(FpExt4::from_u64(20));
    let seventy_two = PackedFpExt4::broadcast(FpExt4::from_u64(72));
    let one_hundred_eight = PackedFpExt4::broadcast(FpExt4::from_u64(108));
    let challenge = PackedFpExt4::broadcast(challenge);
    let mut sums = [zero; 5];

    for pair in (0..pair_count).step_by(PF::WIDTH) {
        let active_lanes = (pair_count - pair).min(PF::WIDTH);
        let mut packed_source = |offset| {
            packed_extension_values::<P, PF>(|lane| {
                if lane < active_lanes {
                    value_at(4 * (pair + lane) + offset)
                } else {
                    FpExt4::zero()
                }
            })
        };
        let first_even = packed_source(0);
        let first_odd = packed_source(1);
        let second_even = packed_source(2);
        let second_odd = packed_source(3);
        let first_folded = first_even + (first_odd - first_even) * challenge;
        let second_folded = second_even + (second_odd - second_even) * challenge;
        for lane in 0..active_lanes {
            folded_values[2 * (pair + lane)] = first_folded.extract(lane);
            folded_values[2 * (pair + lane) + 1] = second_folded.extract(lane);
        }

        let first = packed_extension_values::<P, PF>(|lane| {
            if lane < active_lanes {
                first_equality[(pair + lane) % first_equality.len()]
            } else {
                FpExt4::zero()
            }
        });
        let second = PackedFpExt4::broadcast(second_equality[pair / first_equality.len()]);
        let equality = first * second;
        let delta = second_folded - first_folded;
        let delta_squared = delta * delta;
        let coefficients = match degree {
            2 => [
                first_folded * (first_folded - two),
                (first_folded + first_folded - two) * delta,
                delta_squared,
                zero,
                zero,
            ],
            4 => {
                let four_left = first_folded + first_folded + first_folded + first_folded;
                let sixteen_left = four_left + four_left + four_left + four_left;
                let eighteen_left = sixteen_left + first_folded + first_folded;
                let sixty_four_left = sixteen_left + sixteen_left + sixteen_left + sixteen_left;
                let sixty_left = sixty_four_left - four_left;
                let left_squared = first_folded * first_folded;
                let first_quadratic = left_squared - first_folded - first_folded;
                let second_quadratic = left_squared - eighteen_left + seventy_two;
                let delta_cubed = delta_squared * delta;
                [
                    first_quadratic * second_quadratic,
                    (first_quadratic * (first_folded + first_folded - eighteen)
                        + second_quadratic * (first_folded + first_folded - two))
                        * delta,
                    (left_squared + left_squared + four_left * first_folded - sixty_left
                        + one_hundred_eight)
                        * delta_squared,
                    (four_left - twenty) * delta_cubed,
                    delta_squared * delta_squared,
                ]
            }
            _ => unreachable!("validated direct-range polynomial degree"),
        };
        for coefficient in 0..=degree {
            sums[coefficient] = sums[coefficient] + equality * coefficients[coefficient];
        }
    }
    sums.map(sum_fp32_lanes)
}

#[inline(always)]
fn packed_class_values<const P: u32, PF, const LANES: usize>(
    rows: &[[FpExt4<Fp32<P>>; LANES]],
    child: usize,
    mut row_index: impl FnMut(usize) -> usize,
) -> PackedFpExt4<Fp32<P>, PF>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let indices =
        std::array::from_fn::<_, 16, _>(|lane| if lane < PF::WIDTH { row_index(lane) } else { 0 });
    PackedFpExt4::new(std::array::from_fn(|coefficient| {
        PF::from_fn(|lane| rows[indices[lane]][child].base_coefficient(coefficient))
    }))
}

#[inline(always)]
fn packed_extension_values<const P: u32, PF>(
    mut value: impl FnMut(usize) -> FpExt4<Fp32<P>>,
) -> PackedFpExt4<Fp32<P>, PF>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let values = std::array::from_fn::<_, 16, _>(|lane| {
        if lane < PF::WIDTH {
            value(lane)
        } else {
            FpExt4::zero()
        }
    });
    PackedFpExt4::new(std::array::from_fn(|coefficient| {
        PF::from_fn(|lane| values[lane].base_coefficient(coefficient))
    }))
}

#[inline(always)]
fn packed_quadratic_affine_product<const P: u32, PF>(
    left: [PackedFpExt4<Fp32<P>, PF>; 2],
    right: [PackedFpExt4<Fp32<P>, PF>; 2],
    zero: PackedFpExt4<Fp32<P>, PF>,
) -> [PackedFpExt4<Fp32<P>, PF>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let first_slope = right[0] - left[0];
    let second_slope = right[1] - left[1];
    [
        left[0] * left[1],
        left[0] * second_slope + first_slope * left[1],
        first_slope * second_slope,
        zero,
        zero,
    ]
}

#[inline(always)]
fn packed_quartic_affine_product<const P: u32, PF>(
    left: [PackedFpExt4<Fp32<P>, PF>; 4],
    right: [PackedFpExt4<Fp32<P>, PF>; 4],
    zero: PackedFpExt4<Fp32<P>, PF>,
) -> [PackedFpExt4<Fp32<P>, PF>; 5]
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let first = packed_quadratic_affine_product([left[0], left[1]], [right[0], right[1]], zero);
    let second = packed_quadratic_affine_product([left[2], left[3]], [right[2], right[3]], zero);
    [
        first[0] * second[0],
        first[0] * second[1] + first[1] * second[0],
        first[0] * second[2] + first[1] * second[1] + first[2] * second[0],
        first[1] * second[2] + first[2] * second[1],
        first[2] * second[2],
    ]
}

#[inline(always)]
pub(super) unsafe fn fold_and_compute_product_round_packed<const P: u32, PF>(
    witness_left: [&mut [Fp32<P>]; 4],
    witness_right: CoefficientSlices<'_, P>,
    factor_left: [&mut [Fp32<P>]; 4],
    factor_right: CoefficientSlices<'_, P>,
    challenge: FpExt4<Fp32<P>>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>)
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let len = validate_fused_product_round_slices(
        &witness_left,
        &witness_right,
        &factor_left,
        &factor_right,
        PF::WIDTH,
    );
    let quarter = len / 2;
    let witness_left = witness_left.map(|slice| slice.as_mut_ptr());
    let witness_right = witness_right.map(|slice| slice.as_ptr());
    let factor_left = factor_left.map(|slice| slice.as_mut_ptr());
    let factor_right = factor_right.map(|slice| slice.as_ptr());
    let witness_left_read = witness_left.map(|pointer| pointer.cast_const());
    let factor_left_read = factor_left.map(|pointer| pointer.cast_const());
    let challenge = PackedFpExt4::<Fp32<P>, PF>::broadcast(challenge);
    let zero = FpExt4::<Fp32<P>>::zero();
    let mut constant = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero);
    let mut quadratic = PackedFpExt4::<Fp32<P>, PF>::broadcast(zero);

    for row in (0..quarter).step_by(PF::WIDTH) {
        // SAFETY: validation establishes complete chunks at `row` and
        // `row + quarter` in every coefficient slice.
        let witness_00 = unsafe { read_packed_fp_ext4::<P, PF>(witness_left_read, row) };
        let witness_01 = unsafe { read_packed_fp_ext4::<P, PF>(witness_right, row) };
        let witness_10 = unsafe { read_packed_fp_ext4::<P, PF>(witness_left_read, row + quarter) };
        let witness_11 = unsafe { read_packed_fp_ext4::<P, PF>(witness_right, row + quarter) };
        let factor_00 = unsafe { read_packed_fp_ext4::<P, PF>(factor_left_read, row) };
        let factor_01 = unsafe { read_packed_fp_ext4::<P, PF>(factor_right, row) };
        let factor_10 = unsafe { read_packed_fp_ext4::<P, PF>(factor_left_read, row + quarter) };
        let factor_11 = unsafe { read_packed_fp_ext4::<P, PF>(factor_right, row + quarter) };

        let witness_0 = witness_00 + (witness_01 - witness_00) * challenge;
        let witness_1 = witness_10 + (witness_11 - witness_10) * challenge;
        let factor_0 = factor_00 + (factor_01 - factor_00) * challenge;
        let factor_1 = factor_10 + (factor_11 - factor_10) * challenge;
        constant = constant + witness_0 * factor_0;
        quadratic = quadratic + (witness_1 - witness_0) * (factor_1 - factor_0);

        // SAFETY: the same validation covers both output chunks, and each
        // packed value is written only after all source values were loaded.
        unsafe {
            write_packed_fp_ext4(witness_left, row, witness_0);
            write_packed_fp_ext4(witness_left, row + quarter, witness_1);
            write_packed_fp_ext4(factor_left, row, factor_0);
            write_packed_fp_ext4(factor_left, row + quarter, factor_1);
        }
    }

    sum_product_round_lanes(constant, quadratic)
}

#[inline(always)]
pub(super) unsafe fn fold_fp_ext2_fp64_packed<const P: u64, C, PF>(
    left: [&mut [Fp64<P>]; 2],
    right: Fp64CoefficientSlices<'_, P>,
    challenge: FpExt2<Fp64<P>, C>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    let len = validate_fp64_fold_slices(&left, &right, PF::WIDTH);
    let left = left.map(|slice| slice.as_mut_ptr());
    let right = right.map(|slice| slice.as_ptr());
    let left_read = left.map(|pointer| pointer.cast_const());
    let challenge = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(challenge);

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes complete packed chunks in every
        // coefficient slice at `row`.
        let even = unsafe { read_packed_fp_ext2::<P, C, PF>(left_read, row) };
        let odd = unsafe { read_packed_fp_ext2::<P, C, PF>(right, row) };
        let folded = even + (odd - even) * challenge;
        // SAFETY: the output aliases only the already-read left input chunk.
        unsafe { write_packed_fp_ext2(left, row, folded) };
    }
}

#[inline(always)]
pub(super) unsafe fn compute_product_round_fp_ext2_fp64_packed<const P: u64, C, PF>(
    witness_0: Fp64CoefficientSlices<'_, P>,
    witness_1: Fp64CoefficientSlices<'_, P>,
    factor_0: Fp64CoefficientSlices<'_, P>,
    factor_1: Fp64CoefficientSlices<'_, P>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    let tables = [&witness_0, &witness_1, &factor_0, &factor_1];
    let len = validate_fp64_product_round_slices(&tables, PF::WIDTH);
    let witness_0 = witness_0.map(|slice| slice.as_ptr());
    let witness_1 = witness_1.map(|slice| slice.as_ptr());
    let factor_0 = factor_0.map(|slice| slice.as_ptr());
    let factor_1 = factor_1.map(|slice| slice.as_ptr());
    let zero = FpExt2::<Fp64<P>, C>::zero();
    let mut constant = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(zero);
    let mut quadratic = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(zero);

    for row in (0..len).step_by(PF::WIDTH) {
        // SAFETY: validation establishes a complete packed chunk in every
        // coefficient slice for this row.
        let witness_0 = unsafe { read_packed_fp_ext2::<P, C, PF>(witness_0, row) };
        let witness_1 = unsafe { read_packed_fp_ext2::<P, C, PF>(witness_1, row) };
        let factor_0 = unsafe { read_packed_fp_ext2::<P, C, PF>(factor_0, row) };
        let factor_1 = unsafe { read_packed_fp_ext2::<P, C, PF>(factor_1, row) };
        constant = constant + witness_0 * factor_0;
        quadratic = quadratic + (witness_1 - witness_0) * (factor_1 - factor_0);
    }

    sum_fp64_product_round_lanes(constant, quadratic)
}

#[inline(always)]
pub(super) unsafe fn fold_and_compute_product_round_fp_ext2_fp64_packed<const P: u64, C, PF>(
    witness_left: [&mut [Fp64<P>]; 2],
    witness_right: Fp64CoefficientSlices<'_, P>,
    factor_left: [&mut [Fp64<P>]; 2],
    factor_right: Fp64CoefficientSlices<'_, P>,
    challenge: FpExt2<Fp64<P>, C>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    let len = validate_fp64_fused_product_round_slices(
        &witness_left,
        &witness_right,
        &factor_left,
        &factor_right,
        PF::WIDTH,
    );
    let quarter = len / 2;
    let witness_left = witness_left.map(|slice| slice.as_mut_ptr());
    let witness_right = witness_right.map(|slice| slice.as_ptr());
    let factor_left = factor_left.map(|slice| slice.as_mut_ptr());
    let factor_right = factor_right.map(|slice| slice.as_ptr());
    let witness_left_read = witness_left.map(|pointer| pointer.cast_const());
    let factor_left_read = factor_left.map(|pointer| pointer.cast_const());
    let challenge = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(challenge);
    let zero = FpExt2::<Fp64<P>, C>::zero();
    let mut constant = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(zero);
    let mut quadratic = PackedFpExt2::<Fp64<P>, C, PF>::broadcast(zero);

    for row in (0..quarter).step_by(PF::WIDTH) {
        // SAFETY: validation establishes complete chunks at `row` and
        // `row + quarter` in every coefficient slice.
        let witness_00 = unsafe { read_packed_fp_ext2::<P, C, PF>(witness_left_read, row) };
        let witness_01 = unsafe { read_packed_fp_ext2::<P, C, PF>(witness_right, row) };
        let witness_10 =
            unsafe { read_packed_fp_ext2::<P, C, PF>(witness_left_read, row + quarter) };
        let witness_11 = unsafe { read_packed_fp_ext2::<P, C, PF>(witness_right, row + quarter) };
        let factor_00 = unsafe { read_packed_fp_ext2::<P, C, PF>(factor_left_read, row) };
        let factor_01 = unsafe { read_packed_fp_ext2::<P, C, PF>(factor_right, row) };
        let factor_10 = unsafe { read_packed_fp_ext2::<P, C, PF>(factor_left_read, row + quarter) };
        let factor_11 = unsafe { read_packed_fp_ext2::<P, C, PF>(factor_right, row + quarter) };

        let witness_0 = witness_00 + (witness_01 - witness_00) * challenge;
        let witness_1 = witness_10 + (witness_11 - witness_10) * challenge;
        let factor_0 = factor_00 + (factor_01 - factor_00) * challenge;
        let factor_1 = factor_10 + (factor_11 - factor_10) * challenge;
        constant = constant + witness_0 * factor_0;
        quadratic = quadratic + (witness_1 - witness_0) * (factor_1 - factor_0);

        // SAFETY: the same validation covers both output chunks, and each
        // packed value is written only after all source values were loaded.
        unsafe {
            write_packed_fp_ext2(witness_left, row, witness_0);
            write_packed_fp_ext2(witness_left, row + quarter, witness_1);
            write_packed_fp_ext2(factor_left, row, factor_0);
            write_packed_fp_ext2(factor_left, row + quarter, factor_1);
        }
    }

    sum_fp64_product_round_lanes(constant, quadratic)
}

#[inline(always)]
pub(super) unsafe fn read_packed_fp_ext2<const P: u64, C, PF>(
    coefficients: [*const Fp64<P>; 2],
    row: usize,
) -> PackedFpExt2<Fp64<P>, C, PF>
where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    PackedFpExt2::new(
        PF::from_fn(|lane| unsafe { *coefficients[0].add(row + lane) }),
        PF::from_fn(|lane| unsafe { *coefficients[1].add(row + lane) }),
    )
}

#[inline(always)]
pub(super) unsafe fn write_packed_fp_ext2<const P: u64, C, PF>(
    coefficients: [*mut Fp64<P>; 2],
    row: usize,
    value: PackedFpExt2<Fp64<P>, C, PF>,
) where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    for lane in 0..PF::WIDTH {
        // SAFETY: the caller validated the full output chunk starting at
        // `row` for both coefficient pointers.
        unsafe {
            coefficients[0]
                .add(row + lane)
                .write(value.c0.extract(lane));
            coefficients[1]
                .add(row + lane)
                .write(value.c1.extract(lane));
        }
    }
}

#[inline(always)]
pub(super) fn sum_fp64_product_round_lanes<const P: u64, C, PF>(
    constant: PackedFpExt2<Fp64<P>, C, PF>,
    quadratic: PackedFpExt2<Fp64<P>, C, PF>,
) -> (FpExt2<Fp64<P>, C>, FpExt2<Fp64<P>, C>)
where
    C: FpExt2Config<Fp64<P>> + 'static,
    PF: PackedField<Scalar = Fp64<P>>,
{
    let mut constant_sum = FpExt2::<Fp64<P>, C>::zero();
    let mut quadratic_sum = FpExt2::<Fp64<P>, C>::zero();
    for lane in 0..PF::WIDTH {
        constant_sum += constant.extract(lane);
        quadratic_sum += quadratic.extract(lane);
    }
    (constant_sum, quadratic_sum)
}

#[inline(always)]
pub(super) unsafe fn read_packed_fp_ext4<const P: u32, PF>(
    coefficients: [*const Fp32<P>; 4],
    row: usize,
) -> PackedFpExt4<Fp32<P>, PF>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    PackedFpExt4::new(std::array::from_fn(|coefficient| {
        PF::from_fn(|lane| {
            // SAFETY: the caller validated the full packed chunk starting at
            // `row` for every coefficient pointer.
            unsafe { *coefficients[coefficient].add(row + lane) }
        })
    }))
}

#[inline(always)]
unsafe fn gather_packed_fp_ext4<const P: u32, PF>(
    coefficients: [*const Fp32<P>; 4],
    active_lanes: usize,
    mut row_at: impl FnMut(usize) -> usize,
) -> PackedFpExt4<Fp32<P>, PF>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    PackedFpExt4::new(std::array::from_fn(|coefficient| {
        PF::from_fn(|lane| {
            if lane < active_lanes {
                unsafe { *coefficients[coefficient].add(row_at(lane)) }
            } else {
                Fp32::zero()
            }
        })
    }))
}

#[inline(always)]
unsafe fn write_scattered_fp_ext4_lane<const P: u32, PF>(
    coefficients: [*mut Fp32<P>; 4],
    row: usize,
    value: PackedFpExt4<Fp32<P>, PF>,
    packed_lane: usize,
) where
    PF: PackedField<Scalar = Fp32<P>>,
{
    for (coefficient, packed) in value.coeffs.into_iter().enumerate() {
        unsafe {
            coefficients[coefficient]
                .add(row)
                .write(packed.extract(packed_lane))
        };
    }
}

#[inline(always)]
pub(super) unsafe fn write_packed_fp_ext4<const P: u32, PF>(
    coefficients: [*mut Fp32<P>; 4],
    row: usize,
    value: PackedFpExt4<Fp32<P>, PF>,
) where
    PF: PackedField<Scalar = Fp32<P>>,
{
    for (coefficient, packed) in value.coeffs.into_iter().enumerate() {
        for lane in 0..PF::WIDTH {
            // SAFETY: the caller validated the full output chunk starting at
            // `row` for every coefficient pointer.
            unsafe {
                coefficients[coefficient]
                    .add(row + lane)
                    .write(packed.extract(lane))
            };
        }
    }
}

#[inline(always)]
pub(super) fn sum_product_round_lanes<const P: u32, PF>(
    constant: PackedFpExt4<Fp32<P>, PF>,
    quadratic: PackedFpExt4<Fp32<P>, PF>,
) -> (FpExt4<Fp32<P>>, FpExt4<Fp32<P>>)
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let mut constant_sum = FpExt4::<Fp32<P>>::zero();
    let mut quadratic_sum = FpExt4::<Fp32<P>>::zero();
    for lane in 0..PF::WIDTH {
        constant_sum += constant.extract(lane);
        quadratic_sum += quadratic.extract(lane);
    }
    (constant_sum, quadratic_sum)
}

#[inline(always)]
fn sum_fp32_lanes<const P: u32, PF>(packed: PackedFpExt4<Fp32<P>, PF>) -> FpExt4<Fp32<P>>
where
    PF: PackedField<Scalar = Fp32<P>>,
{
    let mut sum = FpExt4::<Fp32<P>>::zero();
    for lane in 0..PF::WIDTH {
        sum += packed.extract(lane);
    }
    sum
}

fn validate_product_round_slices<const P: u32>(
    tables: &[&CoefficientSlices<'_, P>; 4],
    width: usize,
) -> usize {
    let len = tables[0][0].len();
    assert!(
        tables
            .iter()
            .all(|table| table.iter().all(|slice| slice.len() == len)),
        "fp32 product round slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "fp32 product round slice length must be a multiple of the SIMD width"
    );
    len
}

fn validate_weighted_affine_product_slices<const P: u32, const LANES: usize>(
    left: &[CoefficientSlices<'_, P>; LANES],
    right: &[CoefficientSlices<'_, P>; LANES],
    equality: &CoefficientSlices<'_, P>,
    arity: usize,
    parent_count: usize,
    width: usize,
) -> usize {
    assert!(matches!(arity, 2 | 4), "product arity must be two or four");
    assert_eq!(LANES, arity * parent_count);
    let len = equality[0].len();
    assert!(
        equality.iter().all(|slice| slice.len() == len)
            && left
                .iter()
                .chain(right.iter())
                .all(|lane| lane.iter().all(|slice| slice.len() == len)),
        "weighted affine-product slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "weighted affine-product slice length must be a multiple of the SIMD width"
    );
    len
}

fn validate_weighted_affine_polynomial_slices<const P: u32>(
    left: &CoefficientSlices<'_, P>,
    right: &CoefficientSlices<'_, P>,
    equality: &CoefficientSlices<'_, P>,
    coefficient_count: usize,
    width: usize,
) -> usize {
    assert!(coefficient_count <= 5);
    let len = equality[0].len();
    assert!(
        left.iter()
            .chain(right.iter())
            .chain(equality.iter())
            .all(|slice| slice.len() == len),
        "weighted affine-polynomial slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "weighted affine-polynomial slice length must be a multiple of the SIMD width"
    );
    len
}

fn validate_compact_affine_product_inputs<T, const LANES: usize>(
    folded_rows: &[[T; LANES]],
    first_equality: &[T],
    second_equality: &[T],
    quartet_count: usize,
    arity: usize,
    parent_count: usize,
    width: usize,
) {
    assert!(
        !folded_rows.is_empty(),
        "folded class table cannot be empty"
    );
    assert!(matches!(arity, 2 | 4), "product arity must be two or four");
    assert_eq!(LANES, arity * parent_count);
    assert!(
        first_equality.len().is_power_of_two() && second_equality.len().is_power_of_two(),
        "split equality lengths must be powers of two"
    );
    assert!(
        quartet_count <= first_equality.len() * second_equality.len(),
        "compact product prefix exceeds its equality domain"
    );
    assert!(
        first_equality.len().is_multiple_of(width) && quartet_count.is_multiple_of(width),
        "compact product blocks must align to the SIMD width"
    );
}

fn validate_class_coded_affine_polynomial_inputs<T>(
    class_codes: &[u16],
    class_value_count: usize,
    taylor_row_count: usize,
    first_equality: &[T],
    second_equality: &[T],
    degree: usize,
    width: usize,
) {
    assert!(matches!(degree, 2 | 4));
    assert_eq!(class_value_count, taylor_row_count);
    assert!(class_codes.len().is_multiple_of(2));
    assert!(
        class_codes
            .iter()
            .all(|&class| usize::from(class) < class_value_count),
        "class code exceeds the prepared value table"
    );
    assert!(
        first_equality.len().is_power_of_two()
            && second_equality.len().is_power_of_two()
            && first_equality.len().is_multiple_of(width),
        "split equality tables must align to the SIMD width"
    );
    assert!(
        class_codes.len() / 2 <= first_equality.len() * second_equality.len(),
        "class-coded prefix exceeds its equality domain"
    );
}

fn validate_sparse_affine_polynomial_shape<T>(
    value_count: usize,
    folded_values: &[T],
    first_equality: &[T],
    second_equality: &[T],
    degree: usize,
    width: usize,
) -> usize {
    assert!(matches!(degree, 2 | 4));
    assert!(value_count.is_multiple_of(4));
    assert_eq!(folded_values.len(), value_count / 2);
    assert!(
        first_equality.len().is_power_of_two()
            && second_equality.len().is_power_of_two()
            && first_equality.len().is_multiple_of(width),
        "split equality tables must align to the SIMD width"
    );
    let pair_count = value_count / 4;
    assert!(
        pair_count <= first_equality.len() * second_equality.len(),
        "sparse value prefix exceeds its equality domain"
    );
    pair_count
}

fn validate_fused_product_round_slices<const P: u32>(
    witness_left: &[&mut [Fp32<P>]; 4],
    witness_right: &CoefficientSlices<'_, P>,
    factor_left: &[&mut [Fp32<P>]; 4],
    factor_right: &CoefficientSlices<'_, P>,
    width: usize,
) -> usize {
    let len = witness_left[0].len();
    assert!(
        witness_left.iter().all(|slice| slice.len() == len)
            && witness_right.iter().all(|slice| slice.len() == len)
            && factor_left.iter().all(|slice| slice.len() == len)
            && factor_right.iter().all(|slice| slice.len() == len),
        "fp32 fused product round slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(2 * width),
        "half the fp32 fused product round slice length must be a multiple of the SIMD width"
    );
    len
}

#[cfg(target_arch = "aarch64")]
fn validate_fold_slices<const P: u32>(
    left: &[&mut [Fp32<P>]; 4],
    right: &CoefficientSlices<'_, P>,
    width: usize,
) -> usize {
    let len = left[0].len();
    assert!(
        left.iter().all(|slice| slice.len() == len) && right.iter().all(|slice| slice.len() == len),
        "fp32 extension fold slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "fp32 extension fold slice length must be a multiple of the SIMD width"
    );
    len
}

fn validate_fp64_product_round_slices<const P: u64>(
    tables: &[&Fp64CoefficientSlices<'_, P>; 4],
    width: usize,
) -> usize {
    let len = tables[0][0].len();
    assert!(
        tables
            .iter()
            .all(|table| table.iter().all(|slice| slice.len() == len)),
        "fp64 product round slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "fp64 product round slice length must be a multiple of the SIMD width"
    );
    len
}

fn validate_fp64_fused_product_round_slices<const P: u64>(
    witness_left: &[&mut [Fp64<P>]; 2],
    witness_right: &Fp64CoefficientSlices<'_, P>,
    factor_left: &[&mut [Fp64<P>]; 2],
    factor_right: &Fp64CoefficientSlices<'_, P>,
    width: usize,
) -> usize {
    let len = witness_left[0].len();
    assert!(
        witness_left.iter().all(|slice| slice.len() == len)
            && witness_right.iter().all(|slice| slice.len() == len)
            && factor_left.iter().all(|slice| slice.len() == len)
            && factor_right.iter().all(|slice| slice.len() == len),
        "fp64 fused product round slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(2 * width),
        "half the fp64 fused product round slice length must be a multiple of the SIMD width"
    );
    len
}

fn validate_fp64_fold_slices<const P: u64>(
    left: &[&mut [Fp64<P>]; 2],
    right: &Fp64CoefficientSlices<'_, P>,
    width: usize,
) -> usize {
    let len = left[0].len();
    assert!(
        left.iter().all(|slice| slice.len() == len) && right.iter().all(|slice| slice.len() == len),
        "fp64 extension fold slices must have equal lengths"
    );
    assert!(
        len.is_multiple_of(width),
        "fp64 extension fold slice length must be a multiple of the SIMD width"
    );
    len
}
