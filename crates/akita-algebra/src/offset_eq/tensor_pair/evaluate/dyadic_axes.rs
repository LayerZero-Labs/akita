//! Lower long non-dyadic unit axes to logarithmically many dyadic intervals.
//! The carry recurrence can then consume their bits instead of enumerating
//! every coordinate as a residual seed.

use super::{
    charge_work, checked_axis_offset, EqPairTensorAxis, EqPairTensorFamily, EqPairTensorWeights,
};
use crate::Field;
use akita_error::AkitaError;
use std::borrow::Cow;

pub(super) fn split_residual_unit_axes<'a, F: Field>(
    families: &'a [EqPairTensorFamily<F>],
    work: &mut usize,
) -> Result<Cow<'a, [EqPairTensorFamily<F>]>, AkitaError> {
    let mut output: Option<Vec<EqPairTensorFamily<F>>> = None;
    for (index, family) in families.iter().enumerate() {
        charge_work(work, 1)?;
        charge_work(work, family.axes.len())?;
        let recurrence_axes = family
            .axes
            .iter()
            .filter(|axis| {
                axis.supports_bit_recurrence() && (axis.left_stride != 0 || axis.right_stride != 0)
            })
            .count();
        let has_factored = family.axes.iter().any(|axis| {
            axis.supports_bit_recurrence()
                && (axis.left_stride != 0 || axis.right_stride != 0)
                && matches!(axis.weights, EqPairTensorWeights::BitProduct(_))
        });
        let long_unit = |axis: &EqPairTensorAxis<F>| {
            axis.len >= 128
                && !axis.len.is_power_of_two()
                && matches!(axis.weights, EqPairTensorWeights::Unit)
                && (axis.left_stride != 0 || axis.right_stride != 0)
        };
        if !(recurrence_axes >= 2 || has_factored) || !family.axes.iter().any(long_unit) {
            if let Some(output) = &mut output {
                charge_work(work, family.axes.len() + 1)?;
                output.push(family.clone());
            }
            continue;
        }
        let _scope = akita_error::experiment_scope("search/dyadic_residual_axes");
        if output.is_none() {
            let prefix = families.get(..index).ok_or(AkitaError::InvalidProof)?;
            let cost = prefix.iter().try_fold(0usize, |n, f| {
                n.checked_add(f.axes.len() + 1)
                    .ok_or(AkitaError::InvalidProof)
            })?;
            charge_work(work, cost)?;
            output = Some(prefix.to_vec());
        }
        let mut parts = vec![family.clone()];
        for (axis_index, axis) in family.axes.iter().enumerate().filter(|(_, a)| long_unit(a)) {
            let mut segments = Vec::new();
            let mut start = 0usize;
            while start < axis.len {
                let remaining = axis.len - start;
                let bits = usize::BITS - 1 - remaining.leading_zeros();
                let len = 1usize << bits;
                segments.push((start, len));
                start = start.checked_add(len).ok_or(AkitaError::InvalidProof)?;
            }
            let count = parts
                .len()
                .checked_mul(segments.len())
                .ok_or(AkitaError::InvalidProof)?;
            let cost = count
                .checked_mul(family.axes.len() + 1)
                .ok_or(AkitaError::InvalidProof)?;
            charge_work(work, cost)?;
            let mut next = Vec::new();
            next.try_reserve_exact(count)
                .map_err(|_| AkitaError::InvalidInput("dyadic tensor allocation failed".into()))?;
            for part in parts {
                for &(start, len) in &segments {
                    let mut piece = part.clone();
                    piece.left_offset =
                        checked_axis_offset(part.left_offset, axis.left_stride, start, "left")?;
                    piece.right_offset =
                        checked_axis_offset(part.right_offset, axis.right_stride, start, "right")?;
                    piece
                        .axes
                        .get_mut(axis_index)
                        .ok_or(AkitaError::InvalidProof)?
                        .len = len;
                    next.push(piece);
                }
            }
            parts = next;
        }
        let output = output.as_mut().ok_or(AkitaError::InvalidProof)?;
        output
            .try_reserve(parts.len())
            .map_err(|_| AkitaError::InvalidInput("dyadic tensor allocation failed".into()))?;
        output.extend(parts);
    }
    Ok(output.map_or(Cow::Borrowed(families), Cow::Owned))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset_eq::{eq_eval_at_index, eval_boolean_pair_tensor_families};
    use jolt_field::{One, Prime128OffsetA7F7 as F, Ring, Zero};

    #[test]
    fn dyadic_residual_axis_matches_expanded_polynomial() {
        let left = (0..14).map(|i| F::from_u64(i + 3)).collect::<Vec<_>>();
        let right = (0..14).map(|i| F::from_u64(i + 17)).collect::<Vec<_>>();
        for (left_start, right_start) in [(3, 5), (16370, 16370)] {
            let family = EqPairTensorFamily::new(
                left_start,
                right_start,
                F::from_u64(7),
                vec![
                    EqPairTensorAxis::bit_product(1, 2, vec![[F::one(), F::from_u64(3)]; 3])
                        .unwrap(),
                    EqPairTensorAxis::unit(259, 17, 23),
                ],
            )
            .unwrap();
            let expected = (0usize..259)
                .flat_map(|row| (0usize..8).map(move |column| (row, column)))
                .fold(F::zero(), |sum, (row, column)| {
                    let weight = (0..column.count_ones()).fold(F::one(), |v, _| v * F::from_u64(3));
                    sum + F::from_u64(7)
                        * weight
                        * eq_eval_at_index(&left, left_start + column + 17 * row)
                        * eq_eval_at_index(&right, right_start + 2 * column + 23 * row)
                });
            let families = [family];
            let mut work = 0;
            let lowered = split_residual_unit_axes(&families, &mut work).unwrap();
            assert_eq!(lowered.len(), 3);
            assert!(lowered
                .iter()
                .all(|f| f.axes.iter().all(|a| a.len.is_power_of_two())));
            assert_eq!(
                eval_boolean_pair_tensor_families::<_, false, false>(&left, &right, &families)
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn dyadic_lowering_charges_expansion_before_allocation() {
        let family = EqPairTensorFamily::new(
            0,
            0,
            F::one(),
            vec![
                EqPairTensorAxis::bit_product(1, 2, vec![[F::one(); 2]; 3]).unwrap(),
                EqPairTensorAxis::unit(259, 17, 23),
            ],
        )
        .unwrap();
        let mut work = crate::offset_eq::MAX_COMPACT_STRIDE_TERMS - 3;
        assert!(split_residual_unit_axes(&[family], &mut work).is_err());
    }
}
