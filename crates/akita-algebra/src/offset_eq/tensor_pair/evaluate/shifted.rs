//! Normalize a shared unit axis before the paired-address carry recurrence.
//!
//! Factor every stride as 2^k times an odd stride s. Its low address bits
//! stay fixed along the shared axis:
//! eq(r, offset + 2^k * s * i) = eq(r[..k], offset mod 2^k)
//!                              * eq(r[k..], floor(offset / 2^k) + s * i).
//! Normalize only collected stream batches, preserving multi-axis recurrences
//! and the shared evaluation of families with matching stream geometry.

use super::stream::{eval_tensor_seed_batch, EqPairSeed};
use super::{basis_eval_at_index, charge_work, checked_axis_offset};
use crate::Field;
use akita_error::AkitaError;

const MAX_SEEDS: usize = 65536;
const MAX_LOW_BITS: usize = 12;

pub(super) fn try_eval<F: Field, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
    left: &[F],
    right: &[F],
    left_stride: usize,
    right_stride: usize,
    len: usize,
    seeds: &[EqPairSeed<F>],
    work: &mut usize,
) -> Result<Option<F>, AkitaError> {
    let lb = left_stride.trailing_zeros() as usize;
    let rb = right_stride.trailing_zeros() as usize;
    if len < 128
        || seeds.len() > MAX_SEEDS
        || (lb == 0 && rb == 0)
        || lb > MAX_LOW_BITS
        || rb > MAX_LOW_BITS
        || lb > left.len()
        || rb > right.len()
    {
        return Ok(None);
    }
    // Validate before shifting: normalization must not hide address overflow.
    for seed in seeds {
        checked_axis_offset(seed.left_offset, left_stride, len - 1, "left")?;
        checked_axis_offset(seed.right_offset, right_stride, len - 1, "right")?;
    }
    let left_domain = 1usize << left.len();
    let right_domain = 1usize << right.len();
    let seeds = seeds
        .iter()
        .filter(|seed| seed.left_offset < left_domain && seed.right_offset < right_domain);
    let seed_count = seeds.clone().count();
    if seed_count == 0 {
        return Ok(Some(F::zero()));
    }
    // Each seed needs only its fixed low coordinates, not a full 2^k table.
    // Eligibility bounds this product by MAX_SEEDS * (2 * MAX_LOW_BITS + 1).
    charge_work(work, seed_count * (lb + rb + 1))?;
    let mut seeds = seeds.copied().collect::<Vec<_>>();
    for seed in &mut seeds {
        seed.weight *= basis_eval_at_index::<F, LEFT_MONOMIAL>(
            &left[..lb],
            seed.left_offset & ((1usize << lb) - 1),
        ) * basis_eval_at_index::<F, RIGHT_MONOMIAL>(
            &right[..rb],
            seed.right_offset & ((1usize << rb) - 1),
        );
        seed.left_offset >>= lb;
        seed.right_offset >>= rb;
    }
    Ok(Some(eval_tensor_seed_batch::<
        F,
        LEFT_MONOMIAL,
        RIGHT_MONOMIAL,
    >(
        &left[lb..],
        &right[rb..],
        left_stride >> lb,
        right_stride >> rb,
        len,
        &seeds,
        work,
    )?))
}

#[cfg(test)]
mod tests {
    use super::super::super::{EqPairTensorAxis, EqPairTensorFamily};
    use super::super::basis_eval_at_index;
    use super::super::stream::collect_tensor_family_seeds;
    use super::*;
    use jolt_field::{One, Prime128Offset6A23801F as F, Ring, Zero};
    use std::collections::BTreeMap;

    fn compare<const LM: bool, const RM: bool>(boolean: bool) {
        let point = |n, offset| {
            (0..n)
                .map(|i| {
                    F::from_u64(if boolean {
                        ((i + offset) % 2) as u64
                    } else {
                        (i + offset + 2) as u64
                    })
                })
                .collect::<Vec<_>>()
        };
        let left = point(11, 0);
        let right = point(15, 3);
        for (ls, rs) in [
            (1, 64),
            (4, 32),
            (32, 1),
            (1, 45),
            (4, 180),
            (45, 1),
            (12, 20),
        ] {
            for (lo, ro) in [(0, 0), (3, 19), (19, 253), (1900, 32700)] {
                let family = EqPairTensorFamily::new(
                    lo,
                    ro,
                    F::from_u64(7),
                    vec![
                        EqPairTensorAxis::dense(1, 3, vec![F::one(), F::from_u64(5), F::zero()]),
                        EqPairTensorAxis::unit(128, ls, rs),
                        EqPairTensorAxis::bit_product(
                            2,
                            1,
                            vec![[F::from_u64(3), F::from_u64(4)]; 2],
                        )
                        .unwrap(),
                    ],
                )
                .unwrap();
                let mut batches = BTreeMap::new();
                collect_tensor_family_seeds(
                    &family,
                    Some(1),
                    0,
                    lo,
                    ro,
                    family.scalar,
                    &mut batches,
                    &mut Vec::new(),
                    &mut 0,
                )
                .unwrap();
                let seeds = &batches[&(ls, rs, 128)];
                let actual =
                    eval_tensor_seed_batch::<F, LM, RM>(&left, &right, ls, rs, 128, seeds, &mut 0)
                        .unwrap();
                let mut expected = F::zero();
                for a in 0..3 {
                    for b in 0..128 {
                        for c in 0..4 {
                            let weight = family.axes[0].coordinate_weight(a).unwrap()
                                * family.axes[2].coordinate_weight(c).unwrap();
                            expected += family.scalar
                                * weight
                                * basis_eval_at_index::<F, LM>(&left, lo + a + ls * b + 2 * c)
                                * basis_eval_at_index::<F, RM>(&right, ro + 3 * a + rs * b + c);
                        }
                    }
                }
                assert_eq!(actual, expected, "strides={ls}/{rs}, offsets={lo}/{ro}");
            }
        }
    }

    #[test]
    fn shifted_axes_preserve_weighted_sums_and_clipping_in_each_basis() {
        for boolean in [false, true] {
            compare::<false, false>(boolean);
            compare::<false, true>(boolean);
            compare::<true, false>(boolean);
            compare::<true, true>(boolean);
        }
    }

    #[test]
    fn recurrence_capable_residuals_stay_compact() {
        use super::super::eval_unfactored_tensor_families;
        // At the Boolean origin, only the all-zero coordinate contributes.
        let point = [F::zero(); 20];
        for residual in [
            EqPairTensorAxis::unit(65536, 10, 10),
            EqPairTensorAxis::bit_product(10, 10, vec![[F::one(), F::from_u64(2)]; 16]).unwrap(),
        ] {
            let family = EqPairTensorFamily::new(
                0,
                0,
                F::one(),
                vec![EqPairTensorAxis::unit(65536, 6, 6), residual],
            )
            .unwrap();
            let mut work = 0;
            let actual = eval_unfactored_tensor_families::<F, false, false>(
                &point,
                &point,
                &vec![family; 2048],
                &mut work,
            )
            .unwrap();
            assert_eq!(actual, F::from_u64(2048));
            assert!(
                work < 1_000_000,
                "residual axes must not be expanded: {work}"
            );
        }
    }

    #[test]
    fn matching_shifted_streams_share_the_recurrence() {
        use super::super::eval_unfactored_tensor_families;
        let point = [F::from_u64(2); 20];
        let family =
            EqPairTensorFamily::new(0, 0, F::one(), vec![EqPairTensorAxis::unit(65536, 6, 10)])
                .unwrap();
        let single = eval_unfactored_tensor_families::<F, false, false>(
            &point,
            &point,
            std::slice::from_ref(&family),
            &mut 0,
        )
        .unwrap();
        let mut work = 0;
        let actual = eval_unfactored_tensor_families::<F, false, false>(
            &point,
            &point,
            &vec![family; 2048],
            &mut work,
        )
        .unwrap();
        assert_eq!(actual, single * F::from_u64(2048));
        assert!(work < 100_000, "matching streams must stay batched: {work}");
    }

    #[test]
    fn normalization_preserves_address_and_work_bounds() {
        let point = [F::one(); 20];
        for (left_offset, right_offset) in [(usize::MAX, 0), (0, usize::MAX)] {
            let family = EqPairTensorFamily::new(
                left_offset,
                right_offset,
                F::one(),
                vec![EqPairTensorAxis::unit(128, 6, 10)],
            )
            .unwrap();
            assert!(
                crate::offset_eq::eval_boolean_pair_tensor_families::<F, false, true>(
                    &point,
                    &point,
                    &[family],
                )
                .is_err()
            );
        }
        let seed = EqPairSeed {
            left_offset: 0,
            right_offset: 0,
            weight: F::one(),
        };
        let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS;
        assert!(eval_tensor_seed_batch::<F, false, false>(
            &point,
            &point,
            6,
            10,
            128,
            &[seed],
            &mut work,
        )
        .is_err());
    }

    #[test]
    fn ineligible_batches_leave_the_work_budget_untouched() {
        let point = [F::one(); 20];
        let seed = EqPairSeed {
            left_offset: 0,
            right_offset: 0,
            weight: F::one(),
        };
        for (ls, rs, len, count) in [
            (6, 10, 127, 1),
            (3, 5, 128, 1),
            (1 << 13, 10, 128, 1),
            (6, 1 << 13, 128, 1),
            (6, 10, 128, MAX_SEEDS + 1),
        ] {
            let mut work = 0;
            assert!(try_eval::<F, false, false>(
                &point,
                &point,
                ls,
                rs,
                len,
                &vec![seed; count],
                &mut work,
            )
            .unwrap()
            .is_none());
            assert_eq!(work, 0);
        }
        for (left, right) in [(&point[..0], &point[..]), (&point[..], &point[..0])] {
            let mut work = 0;
            assert!(
                try_eval::<F, false, false>(left, right, 6, 10, 128, &[seed], &mut work,)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(work, 0);
        }
    }

    #[test]
    fn large_shift_keeps_the_carry_recurrence_small() {
        use super::super::eval_unfactored_tensor_families;
        let left = [F::from_u64(2); 20];
        let right = [F::from_u64(3); 28];
        let family = EqPairTensorFamily::new(
            3,
            19,
            F::one(),
            vec![EqPairTensorAxis::unit(16384, 6, 4096)],
        )
        .unwrap();
        let mut work = 0;
        let actual =
            eval_unfactored_tensor_families::<F, false, false>(&left, &right, &[family], &mut work)
                .unwrap();
        let expected: F = (0..16384)
            .map(|i| {
                basis_eval_at_index::<F, false>(&left, 3 + 6 * i)
                    * basis_eval_at_index::<F, false>(&right, 19 + 4096 * i)
            })
            .sum();
        assert_eq!(actual, expected);
        assert!(
            work < 10_000,
            "low coordinates must leave the carry state: {work}"
        );
    }

    #[test]
    fn fully_clipped_batches_do_not_build_low_tables() {
        use super::super::eval_unfactored_tensor_families;
        let point = [F::from_u64(2); 20];
        let families = (128..32896)
            .map(|len| {
                EqPairTensorFamily::new(
                    1 << 20,
                    0,
                    F::one(),
                    vec![EqPairTensorAxis::unit(len, 12288, 4096)],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut work = 0;
        let actual = eval_unfactored_tensor_families::<F, false, false>(
            &point, &point, &families, &mut work,
        )
        .unwrap();
        assert_eq!(actual, F::zero());
        assert!(
            work < 100_000,
            "clipped seeds must not build low tables: {work}"
        );
    }

    #[test]
    fn tiny_live_batches_do_not_pay_for_exponential_tables() {
        use super::super::eval_unfactored_tensor_families;
        let point = [F::zero(); 20];
        let families = (128..32896)
            .map(|len| {
                EqPairTensorFamily::new(
                    0,
                    0,
                    F::one(),
                    vec![EqPairTensorAxis::unit(len, 12288, 4096)],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut work = 0;
        let actual = eval_unfactored_tensor_families::<F, false, false>(
            &point, &point, &families, &mut work,
        )
        .unwrap();
        assert_eq!(actual, F::from_u64(families.len() as u64));
        assert!(
            work < 4_000_000,
            "low-bit work must scale with live seeds: {work}"
        );
    }
}
