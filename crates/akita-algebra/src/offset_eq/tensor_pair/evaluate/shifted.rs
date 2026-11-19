//! Normalize a shared unit axis before the paired-address carry recurrence.
//!
//! Factor every stride as 2^k times an odd stride s. Its low address bits
//! stay fixed along the shared axis:
//! eq(r, offset + 2^k * s * i) = eq(r[..k], offset mod 2^k)
//!                              * eq(r[k..], floor(offset / 2^k) + s * i).
//! Eliminating these low coordinates avoids retaining a shifted copy of every
//! shared index bit in the carry state. Residual axes keep their exact weights.

use super::stream::{collect_tensor_family_seeds, eval_tensor_seed_batch};
use super::{charge_work, EqPairTensorFamily, EqPairTensorWeights};
use crate::Field;
use akita_error::AkitaError;
use std::collections::BTreeMap;

const MAX_RESIDUAL_TERMS: usize = 65536;
const MAX_LOW_BITS: usize = 12;

pub(super) fn try_eval<F: Field, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
    left: &[F],
    right: &[F],
    family: &EqPairTensorFamily<F>,
    work: &mut usize,
) -> Result<Option<F>, AkitaError> {
    let candidate = family
        .axes
        .iter()
        .enumerate()
        .filter(|(index, axis)| {
            if axis.len < 128
                || !matches!(axis.weights, EqPairTensorWeights::Unit)
                || axis.left_stride == 0
                || axis.right_stride == 0
            {
                return false;
            }
            let lb = axis.left_stride.trailing_zeros() as usize;
            let rb = axis.right_stride.trailing_zeros() as usize;
            if (lb == 0 && rb == 0 && family.axes.len() == 1)
                || lb > MAX_LOW_BITS
                || rb > MAX_LOW_BITS
                || lb > left.len()
                || rb > right.len()
            {
                return false;
            }
            family
                .axes
                .iter()
                .enumerate()
                .filter(|(i, _)| i != index)
                .try_fold(1usize, |n, (_, axis)| n.checked_mul(axis.len))
                .is_some_and(|n| n <= MAX_RESIDUAL_TERMS)
        })
        .max_by_key(|(_, axis)| axis.len);
    let Some((index, axis)) = candidate else {
        return Ok(None);
    };
    let lb = axis.left_stride.trailing_zeros() as usize;
    let rb = axis.right_stride.trailing_zeros() as usize;
    let mut batches = BTreeMap::new();
    let mut scalars = Vec::new();
    collect_tensor_family_seeds(
        family,
        Some(index),
        0,
        family.left_offset,
        family.right_offset,
        family.scalar,
        &mut batches,
        &mut scalars,
        work,
    )?;
    debug_assert!(scalars.is_empty());
    let Some(mut seeds) = batches.remove(&(axis.left_stride, axis.right_stride, axis.len)) else {
        return Ok(Some(F::zero()));
    };
    // These bounded tables are the low-coordinate MLE contraction, independent
    // of the length of the shared unit axis. No data survive this evaluation.
    let low_table = |point: &[F], monomial: bool| -> Result<Vec<F>, AkitaError> {
        let mut values = vec![F::one()];
        for &r in point {
            let n = values.len();
            values.resize(n * 2, F::zero());
            for i in 0..n {
                let one = values[i] * r;
                if !monomial {
                    values[i] -= one;
                }
                values[n + i] = one;
            }
        }
        Ok(values)
    };
    charge_work(work, (1usize << lb) + (1usize << rb) + seeds.len())?;
    let low_left = low_table(&left[..lb], LEFT_MONOMIAL)?;
    let low_right = low_table(&right[..rb], RIGHT_MONOMIAL)?;
    for seed in &mut seeds {
        seed.weight *= low_left[seed.left_offset & (low_left.len() - 1)]
            * low_right[seed.right_offset & (low_right.len() - 1)];
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
        axis.left_stride >> lb,
        axis.right_stride >> rb,
        axis.len,
        &seeds,
        work,
    )?))
}

#[cfg(test)]
mod tests {
    use super::super::super::EqPairTensorAxis;
    use super::super::basis_eval_at_index;
    use super::*;
    use jolt_field::{One, Prime128Offset6A23801F as F, Ring, Zero};

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
                let actual = try_eval::<F, LM, RM>(&left, &right, &family, &mut 0)
                    .unwrap()
                    .expect("eligible shifted axis");
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
}
