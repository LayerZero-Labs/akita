//! Seed collection and affine stream contractions for tensor families.

use super::{
    basis_eval_at_index, basis_pair_factors, charge_work, checked_axis_offset, merge_pair_states,
    unit_intervals, EqPairTensorFamily,
};
use crate::Field;
use akita_error::AkitaError;
use std::collections::BTreeMap;

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_tensor_family_seeds<F: Field>(
    family: &EqPairTensorFamily<F>,
    stream_axis: Option<usize>,
    axis_index: usize,
    left_offset: usize,
    right_offset: usize,
    weight: F,
    batches: &mut BTreeMap<(usize, usize, usize), Vec<EqPairSeed<F>>>,
    scalar_seeds: &mut Vec<EqPairSeed<F>>,
    work: &mut usize,
) -> Result<(), AkitaError> {
    if weight.is_zero() {
        return Ok(());
    }
    if axis_index == family.axes.len() {
        charge_work(work, 1)?;
        let seed = EqPairSeed {
            left_offset,
            right_offset,
            weight,
        };
        if let Some(stream_axis) = stream_axis {
            let axis = family.axes.get(stream_axis).ok_or_else(|| {
                AkitaError::Internal("selected tensor seed stream axis is absent".into())
            })?;
            batches
                .entry((axis.left_stride, axis.right_stride, axis.len))
                .or_default()
                .push(seed);
        } else {
            scalar_seeds.push(seed);
        }
        return Ok(());
    }
    if Some(axis_index) == stream_axis {
        return collect_tensor_family_seeds(
            family,
            stream_axis,
            axis_index + 1,
            left_offset,
            right_offset,
            weight,
            batches,
            scalar_seeds,
            work,
        );
    }

    let axis = family
        .axes
        .get(axis_index)
        .ok_or_else(|| AkitaError::Internal("tensor seed recursion axis is absent".into()))?;
    // Charge every visited coordinate, including zero-weight ones that never
    // reach a seed, before the loop so a long axis cannot bypass the bound.
    charge_work(work, axis.len)?;
    for coordinate in 0..axis.len {
        let axis_weight = axis.coordinate_weight(coordinate).ok_or_else(|| {
            AkitaError::InvalidInput("tensor seed axis coordinate weight is absent".into())
        })?;
        if axis_weight.is_zero() {
            continue;
        }
        let left = checked_axis_offset(left_offset, axis.left_stride, coordinate, "left")?;
        let right = checked_axis_offset(right_offset, axis.right_stride, coordinate, "right")?;
        let next_weight = if axis_weight == F::one() {
            weight
        } else if weight == F::one() {
            axis_weight
        } else {
            weight * axis_weight
        };
        collect_tensor_family_seeds(
            family,
            stream_axis,
            axis_index + 1,
            left,
            right,
            next_weight,
            batches,
            scalar_seeds,
            work,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct EqPairSeed<F> {
    pub(super) left_offset: usize,
    pub(super) right_offset: usize,
    pub(super) weight: F,
}

pub(super) fn eval_tensor_seed_batch<
    F: Field,
    const LEFT_MONOMIAL: bool,
    const RIGHT_MONOMIAL: bool,
>(
    left_challenges: &[F],
    right_challenges: &[F],
    left_stride: usize,
    right_stride: usize,
    len: usize,
    seeds: &[EqPairSeed<F>],
    work: &mut usize,
) -> Result<F, AkitaError> {
    if left_stride == 0 || right_stride == 0 {
        return Err(AkitaError::InvalidInput(
            "paired tensor stream strides must be non-zero".into(),
        ));
    }
    if len == 0 {
        return Ok(F::zero());
    }

    if left_stride == 1 && len >= 128 && seeds.len() >= 8 {
        let low_bits = right_stride.trailing_zeros();
        if low_bits as usize <= right_challenges.len() {
            let odd_stride = right_stride >> low_bits;
            macro_rules! cached_interval {
                ($states:literal) => {
                    return unit_intervals::eval_scaled_right::<
                        F,
                        LEFT_MONOMIAL,
                        RIGHT_MONOMIAL,
                        $states,
                    >(
                        left_challenges,
                        right_challenges,
                        right_stride,
                        len,
                        seeds,
                        work,
                    )
                };
            }
            match odd_stride {
                1 => cached_interval!(2),
                3 => cached_interval!(4),
                5 => cached_interval!(6),
                7 => cached_interval!(8),
                _ => {}
            }
        }
    }

    let left_domain = 1usize << left_challenges.len();
    let right_domain = 1usize << right_challenges.len();
    let block_bucket_count = usize::BITS as usize - len.leading_zeros() as usize;
    let mut blocks = vec![Vec::<((usize, usize), F)>::new(); block_bucket_count];
    for seed in seeds {
        checked_axis_offset(seed.left_offset, left_stride, len - 1, "left")?;
        checked_axis_offset(seed.right_offset, right_stride, len - 1, "right")?;
        if seed.left_offset >= left_domain || seed.right_offset >= right_domain {
            continue;
        }
        let live_len = len
            .min((left_domain - 1 - seed.left_offset) / left_stride + 1)
            .min((right_domain - 1 - seed.right_offset) / right_stride + 1);
        if live_len == 0 {
            continue;
        }
        let highest_bit = usize::BITS as usize - 1 - live_len.leading_zeros() as usize;
        let mut block_base = 0usize;
        for block_index_bits in (0..=highest_bit).rev() {
            let block_size = 1usize << block_index_bits;
            if live_len & block_size == 0 {
                continue;
            }
            let left_carry =
                checked_axis_offset(seed.left_offset, left_stride, block_base, "left")?;
            let right_carry =
                checked_axis_offset(seed.right_offset, right_stride, block_base, "right")?;
            blocks
                .get_mut(block_index_bits)
                .ok_or_else(|| AkitaError::Internal("tensor seed block bucket is absent".into()))?
                .push(((left_carry, right_carry), seed.weight));
            block_base = block_base.checked_add(block_size).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor block coverage overflow".into())
            })?;
        }
    }

    let mut acc = F::zero();
    for (block_index_bits, seed_states) in blocks.into_iter().enumerate() {
        let mut states = merge_pair_states(seed_states)?;
        if states.is_empty() {
            continue;
        }
        if block_index_bits > left_challenges.len() || block_index_bits > right_challenges.len() {
            return Err(AkitaError::InvalidInput(
                "paired tensor block exceeds equality arity".into(),
            ));
        }
        for bit in 0..block_index_bits {
            charge_work(
                work,
                states.len().checked_mul(2).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor work overflow".into())
                })?,
            )?;
            let left_challenge = *left_challenges.get(bit).ok_or_else(|| {
                AkitaError::Internal(
                    "tensor seed left challenge bit exceeds validated arity".into(),
                )
            })?;
            let right_challenge = *right_challenges.get(bit).ok_or_else(|| {
                AkitaError::Internal(
                    "tensor seed right challenge bit exceeds validated arity".into(),
                )
            })?;
            let bit_factors = basis_pair_factors::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
                Some(left_challenge),
                Some(right_challenge),
            );
            let next_capacity = states.len().checked_mul(2).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor state count overflow".into())
            })?;
            let mut next = Vec::new();
            next.try_reserve_exact(next_capacity).map_err(|_| {
                AkitaError::InvalidInput("paired tensor state allocation failed".into())
            })?;
            for ((left_carry, right_carry), state_weight) in states {
                for index_bit in 0..=1usize {
                    let left_sum = if index_bit == 0 {
                        left_carry
                    } else {
                        left_carry.checked_add(left_stride).ok_or_else(|| {
                            AkitaError::InvalidInput("paired tensor left carry overflow".into())
                        })?
                    };
                    let right_sum = if index_bit == 0 {
                        right_carry
                    } else {
                        right_carry.checked_add(right_stride).ok_or_else(|| {
                            AkitaError::InvalidInput("paired tensor right carry overflow".into())
                        })?
                    };
                    let factor =
                        bit_factors[(left_sum & 1) * 2 + (right_sum & 1)].ok_or_else(|| {
                            AkitaError::Internal("tensor seed binary factor is absent".into())
                        })?;
                    next.push(((left_sum >> 1, right_sum >> 1), state_weight * factor));
                }
            }
            states = merge_pair_states(next)?;
        }
        acc += finish_seed_states::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
            left_challenges,
            right_challenges,
            block_index_bits,
            states,
            work,
        )?;
    }
    Ok(acc)
}

fn finish_seed_states<F: Field, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
    left_challenges: &[F],
    right_challenges: &[F],
    mut bit: usize,
    mut states: Vec<((usize, usize), F)>,
    work: &mut usize,
) -> Result<F, AkitaError> {
    let max_bits = left_challenges.len().max(right_challenges.len());
    while states.len() > 1 && bit < max_bits {
        charge_work(work, states.len())?;
        let mut next = Vec::new();
        next.try_reserve_exact(states.len()).map_err(|_| {
            AkitaError::InvalidInput("paired tensor carry allocation failed".into())
        })?;
        let factors = basis_pair_factors::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
            left_challenges.get(bit).copied(),
            right_challenges.get(bit).copied(),
        );
        for ((left_carry, right_carry), state_weight) in states {
            if (bit >= left_challenges.len() && left_carry != 0)
                || (bit >= right_challenges.len() && right_carry != 0)
            {
                continue;
            }
            let Some(factor) = factors[(left_carry & 1) * 2 + (right_carry & 1)] else {
                continue;
            };
            next.push(((left_carry >> 1, right_carry >> 1), state_weight * factor));
        }
        states = merge_pair_states(next)?;
        bit += 1;
    }

    Ok(states
        .into_iter()
        .map(|((left_high, right_high), state_weight)| {
            let left_equality = left_challenges.get(bit..).map_or_else(
                || {
                    if left_high == 0 {
                        F::one()
                    } else {
                        F::zero()
                    }
                },
                |challenges| basis_eval_at_index::<F, LEFT_MONOMIAL>(challenges, left_high),
            );
            let right_equality = right_challenges.get(bit..).map_or_else(
                || {
                    if right_high == 0 {
                        F::one()
                    } else {
                        F::zero()
                    }
                },
                |challenges| basis_eval_at_index::<F, RIGHT_MONOMIAL>(challenges, right_high),
            );
            state_weight * left_equality * right_equality
        })
        .sum())
}
