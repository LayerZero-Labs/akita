//! Remove independent aligned low coordinate axes before the general carry
//! recurrence. This is an exact contraction, including Boolean challenges.
use super::evaluate::eval_unfactored_tensor_families;
use super::{
    charge_work, checked_axis_offset, EqPairTensorAxis, EqPairTensorFamily, EqPairTensorWeights,
};
use akita_error::AkitaError;
use jolt_field::Field;
use std::collections::BTreeMap;
/// Evaluate paired tensor families, contracting independent aligned axes first.
///
/// # Errors
///
/// Returns an error for malformed geometry, address overflow, unsupported
/// equality arity, or work above [`crate::offset_eq::MAX_COMPACT_STRIDE_TERMS`]
/// across the whole call: peeled axes and every residual batch share one budget.
pub fn eval_boolean_pair_tensor_families<
    E: Field,
    const LEFT_MONOMIAL: bool,
    const RIGHT_MONOMIAL: bool,
>(
    left: &[E],
    right: &[E],
    families: &[EqPairTensorFamily<E>],
) -> Result<E, AkitaError> {
    let mut work = 0usize;
    if RIGHT_MONOMIAL || left.len() >= usize::BITS as usize || right.len() >= usize::BITS as usize {
        return eval_unfactored_tensor_families::<E, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
            left, right, families, &mut work,
        );
    }
    let mut batches = BTreeMap::<(usize, usize), Vec<EqPairTensorFamily<E>>>::new();
    let mut stage = |key, family| {
        let batch = batches.entry(key).or_default();
        batch.try_reserve(1).map_err(|_| {
            AkitaError::InvalidInput("paired tensor batch allocation failed".into())
        })?;
        batch.push(family);
        Ok::<_, AkitaError>(())
    };
    for source in families {
        source
            .axes
            .iter()
            .try_for_each(EqPairTensorAxis::validate_weights)?;
        if source.scalar.is_zero() {
            continue;
        }
        let mut axes = Vec::new();
        axes.try_reserve_exact(source.axes.len()).map_err(|_| {
            AkitaError::InvalidInput("paired tensor family allocation failed".into())
        })?;
        axes.extend_from_slice(&source.axes);
        let mut family = EqPairTensorFamily {
            left_offset: source.left_offset,
            right_offset: source.right_offset,
            scalar: source.scalar,
            axes,
        };
        // Zero-length axes and clipped addresses keep this family on the
        // unfactored path, which owns their selection semantics.
        let mut fits = source.axes.iter().all(|axis| axis.len != 0);
        if fits {
            let mut maximum = [source.left_offset, source.right_offset];
            for axis in &source.axes {
                let last = axis.len - 1;
                maximum = [
                    checked_axis_offset(maximum[0], axis.left_stride, last, "left")?,
                    checked_axis_offset(maximum[1], axis.right_stride, last, "right")?,
                ];
            }
            fits = maximum[0] < (1usize << left.len()) && maximum[1] < (1usize << right.len());
        }
        if !fits {
            stage((0, 0), family)?;
            continue;
        }
        let (mut lb, mut rb) = (0usize, 0usize);
        loop {
            let candidate = family
                .axes
                .iter()
                .enumerate()
                .find(|(index, a)| {
                    if a.len < 2
                        || !a.len.is_power_of_two()
                        || a.left_stride > 1
                        || a.right_stride > 1
                        || (a.left_stride == 0 && a.right_stride == 0)
                        || matches!(a.weights, EqPairTensorWeights::Dense(_))
                    {
                        return false;
                    }
                    let bits = a.len.trailing_zeros() as usize;
                    if (a.left_stride == 1
                        && (lb + bits > left.len() || family.left_offset & (a.len - 1) != 0))
                        || (a.right_stride == 1
                            && (rb + bits > right.len() || family.right_offset & (a.len - 1) != 0))
                    {
                        return false;
                    }
                    family.axes.iter().enumerate().all(|(j, b)| {
                        j == *index
                            || b.len <= 1
                            || (a.left_stride == 0 || b.left_stride.is_multiple_of(a.len))
                                && (a.right_stride == 0 || b.right_stride.is_multiple_of(a.len))
                    })
                })
                .map(|(i, _)| i);
            let Some(index) = candidate else {
                break;
            };
            let axis = family.axes.remove(index);
            let bits = axis.len.trailing_zeros() as usize;
            let mut factor = E::one();
            for bit in 0..bits {
                charge_work(&mut work, 2)?;
                let w = axis.bit_factors(bit).ok_or_else(|| {
                    AkitaError::Internal("peeled tensor axis bit factors are absent".into())
                })?;
                let l = if axis.left_stride == 0 {
                    [E::one(); 2]
                } else if LEFT_MONOMIAL {
                    [E::one(), left[lb + bit]]
                } else {
                    [E::one() - left[lb + bit], left[lb + bit]]
                };
                let r = if axis.right_stride == 0 {
                    [E::one(); 2]
                } else {
                    [E::one() - right[rb + bit], right[rb + bit]]
                };
                factor *= w[0] * l[0] * r[0] + w[1] * l[1] * r[1];
            }
            family.scalar *= factor;
            if axis.left_stride == 1 {
                family.left_offset >>= bits;
                for a in &mut family.axes {
                    a.left_stride = if a.len <= 1 { 0 } else { a.left_stride >> bits };
                }
                lb += bits;
            }
            if axis.right_stride == 1 {
                family.right_offset >>= bits;
                for a in &mut family.axes {
                    a.right_stride = if a.len <= 1 {
                        0
                    } else {
                        a.right_stride >> bits
                    };
                }
                rb += bits;
            }
        }
        stage((lb, rb), family)?;
    }
    let mut result = E::zero();
    for ((lb, rb), batch) in batches {
        result += eval_unfactored_tensor_families::<E, LEFT_MONOMIAL, false>(
            &left[lb..],
            &right[rb..],
            &batch,
            &mut work,
        )?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::super::EqPairTensorAxis;
    use super::*;
    use jolt_field::{One, Prime128Offset6A23801F as F, Ring, Zero};
    #[test]
    fn clipped_addresses_keep_boolean_selection_semantics() {
        let family = EqPairTensorFamily::new(
            252,
            0,
            F::from_u64(7),
            vec![EqPairTensorAxis::unit(8, 1, 1)],
        )
        .unwrap();
        let left: Vec<F> = (0..8).map(|bit| F::from_u64((254 >> bit) & 1)).collect();
        let right: Vec<F> = (0..3).map(|bit| F::from_u64((2 >> bit) & 1)).collect();
        assert_eq!(
            eval_boolean_pair_tensor_families::<_, false, false>(&left, &right, &[family]).unwrap(),
            F::from_u64(7)
        );
        let family = EqPairTensorFamily::new(256, 0, F::one(), vec![]).unwrap();
        assert_eq!(
            eval_boolean_pair_tensor_families::<_, false, false>(&left, &right, &[family]).unwrap(),
            F::zero()
        );
    }
    #[test]
    fn independent_axes_match_carry_recurrence_in_both_bases() {
        for base in [0, 64, 128, 192, 3] {
            let family = EqPairTensorFamily::new(
                0,
                base,
                F::from_u64(7),
                vec![
                    EqPairTensorAxis::unit(8, 1, 1),
                    EqPairTensorAxis::bit_product(0, 8, vec![[F::from_u64(2), F::from_u64(3)]; 2])
                        .unwrap(),
                    EqPairTensorAxis::unit(4, 8, 48),
                ],
            )
            .unwrap();
            for boolean in [false, true] {
                let l: Vec<F> = (0..8)
                    .map(|i| F::from_u64(if boolean { i % 2 } else { i + 2 }))
                    .collect();
                let r: Vec<F> = (0..10)
                    .map(|i| F::from_u64(if boolean { i % 2 } else { i + 11 }))
                    .collect();
                let families = [family.clone()];
                assert_eq!(
                    eval_boolean_pair_tensor_families::<_, false, false>(&l, &r, &families)
                        .unwrap(),
                    eval_unfactored_tensor_families::<_, false, false>(&l, &r, &families, &mut 0)
                        .unwrap()
                );
                assert_eq!(
                    eval_boolean_pair_tensor_families::<_, true, false>(&l, &r, &families).unwrap(),
                    eval_unfactored_tensor_families::<_, true, false>(&l, &r, &families, &mut 0)
                        .unwrap()
                );
            }
        }
    }
}
