//! Boolean paired-equality contractions over tensor-native address families.

use super::{charge_work, checked_axis_offset, EqPairTensorFamily, EqPairTensorWeights};
use crate::offset_eq::eq_eval_at_index;
use crate::{AkitaError, FieldCore};
use std::collections::BTreeMap;

/// Evaluate tensor-native paired Boolean-basis families.
///
/// Multiple power-of-two unit axes are contracted together by an exact binary
/// carry recurrence. Otherwise, a largest unit-weight axis is kept as an
/// affine stream and the remaining (normally tiny) tensor axes seed exact
/// stream contractions directly. No expanded term vector or rectangle
/// rediscovery is used.
///
/// # Errors
///
/// Returns an error for malformed geometry, address overflow, unsupported
/// equality arity, or recurrence work above
/// [`crate::offset_eq::MAX_COMPACT_STRIDE_TERMS`].
pub fn eval_boolean_pair_tensor_families<
    F: FieldCore,
    const LEFT_MONOMIAL: bool,
    const RIGHT_MONOMIAL: bool,
>(
    left_challenges: &[F],
    right_challenges: &[F],
    families: &[EqPairTensorFamily<F>],
) -> Result<F, AkitaError> {
    if left_challenges.len() >= usize::BITS as usize
        || right_challenges.len() >= usize::BITS as usize
    {
        return Err(AkitaError::InvalidSize {
            expected: usize::BITS as usize - 1,
            actual: left_challenges.len().max(right_challenges.len()),
        });
    }
    let mut batches = BTreeMap::<(usize, usize, usize), Vec<EqPairSeed<F>>>::new();
    let mut multi_axis_batches =
        BTreeMap::<Vec<(usize, usize, usize, usize)>, Vec<&EqPairTensorFamily<F>>>::new();
    let mut scalar_seeds = Vec::new();
    let mut work = 0usize;
    let mut acc = F::zero();
    for family in families {
        if family.scalar.is_zero() {
            continue;
        }
        let recurrence_axes = family
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| {
                matches!(axis.weights, EqPairTensorWeights::Unit)
                    && axis.len.is_power_of_two()
                    && (axis.left_stride != 0 || axis.right_stride != 0)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if recurrence_axes.len() >= 2 {
            let recurrence_geometry = recurrence_axes
                .iter()
                .map(|&axis_index| {
                    let axis = family
                        .axes
                        .get(axis_index)
                        .ok_or(AkitaError::InvalidProof)?;
                    Ok((axis_index, axis.len, axis.left_stride, axis.right_stride))
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            multi_axis_batches
                .entry(recurrence_geometry)
                .or_default()
                .push(family);
            continue;
        }
        let stream_axis = family
            .axes
            .iter()
            .enumerate()
            .filter(|(_, axis)| {
                matches!(axis.weights, EqPairTensorWeights::Unit)
                    && axis.left_stride != 0
                    && axis.right_stride != 0
            })
            .max_by_key(|(_, axis)| axis.len)
            .map(|(index, _)| index);
        collect_tensor_family_seeds(
            family,
            stream_axis,
            0,
            family.left_offset,
            family.right_offset,
            family.scalar,
            &mut batches,
            &mut scalar_seeds,
            &mut work,
        )?;
    }
    for (recurrence_geometry, families) in multi_axis_batches {
        acc += eval_multi_axis_unit_families::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
            left_challenges,
            right_challenges,
            &families,
            &recurrence_geometry,
            &mut work,
        )?;
    }
    acc += scalar_seeds.into_iter().fold(F::zero(), |sum, seed| {
        sum + seed.weight
            * basis_eval_checked::<F, LEFT_MONOMIAL>(left_challenges, seed.left_offset)
            * basis_eval_checked::<F, RIGHT_MONOMIAL>(right_challenges, seed.right_offset)
    });
    for ((left_stride, right_stride, len), seeds) in batches {
        acc += eval_tensor_seed_batch::<F, LEFT_MONOMIAL, RIGHT_MONOMIAL>(
            left_challenges,
            right_challenges,
            left_stride,
            right_stride,
            len,
            &seeds,
            &mut work,
        )?;
    }
    Ok(acc)
}

fn eval_multi_axis_unit_families<
    F: FieldCore,
    const LEFT_MONOMIAL: bool,
    const RIGHT_MONOMIAL: bool,
>(
    left_challenges: &[F],
    right_challenges: &[F],
    families: &[&EqPairTensorFamily<F>],
    recurrence_geometry: &[(usize, usize, usize, usize)],
    work: &mut usize,
) -> Result<F, AkitaError> {
    let recurrence_axes = recurrence_geometry
        .iter()
        .map(|&(axis_index, _, _, _)| axis_index)
        .collect::<Vec<_>>();
    let mut seeds = Vec::new();
    for family in families {
        collect_residual_seeds(
            family,
            &recurrence_axes,
            0,
            family.left_offset,
            family.right_offset,
            family.scalar,
            &mut seeds,
            work,
        )?;
    }
    let bit_count = left_challenges.len().max(right_challenges.len());
    let mut introductions = vec![Vec::<(usize, usize)>::new(); bit_count];
    for &(_, axis_len, left_stride, right_stride) in recurrence_geometry {
        for coordinate_bit in 0..axis_len.trailing_zeros() as usize {
            let coordinate = 1usize.checked_shl(coordinate_bit as u32).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor coordinate bit overflow".into())
            })?;
            let left = left_stride.checked_mul(coordinate).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor left stride overflow".into())
            })?;
            let right = right_stride.checked_mul(coordinate).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor right stride overflow".into())
            })?;
            let start_bit = [left, right]
                .into_iter()
                .filter(|value| *value != 0)
                .map(|value| value.trailing_zeros() as usize)
                .min()
                .ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor unit axis is constant".into())
                })?;
            if let Some(at_bit) = introductions.get_mut(start_bit) {
                at_bit.push((left >> start_bit, right >> start_bit));
            }
        }
    }

    let mut states = merge_pair_states(
        seeds
            .into_iter()
            .map(|seed| ((seed.left_offset, seed.right_offset), seed.weight))
            .collect(),
    );
    for bit in 0..bit_count {
        let choices = unit_axis_choices::<F>(
            introductions.get(bit).ok_or(AkitaError::InvalidProof)?,
            work,
        )?;
        charge_work(
            work,
            states.len().checked_mul(choices.len()).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor recurrence work overflow".into())
            })?,
        )?;
        let next_capacity = states
            .len()
            .checked_mul(choices.len())
            .ok_or_else(|| AkitaError::InvalidInput("paired tensor state count overflow".into()))?;
        let mut next = Vec::new();
        next.try_reserve_exact(next_capacity).map_err(|_| {
            AkitaError::InvalidInput("paired tensor recurrence state allocation failed".into())
        })?;
        for ((left_carry, right_carry), state_weight) in states {
            for &(left_add, right_add, multiplicity) in &choices {
                let left = left_carry.checked_add(left_add).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor left carry overflow".into())
                })?;
                let right = right_carry.checked_add(right_add).ok_or_else(|| {
                    AkitaError::InvalidInput("paired tensor right carry overflow".into())
                })?;
                let Some(left_factor) = basis_bit_factor::<F, LEFT_MONOMIAL>(
                    left_challenges.get(bit).copied(),
                    left & 1,
                ) else {
                    continue;
                };
                let Some(right_factor) = basis_bit_factor::<F, RIGHT_MONOMIAL>(
                    right_challenges.get(bit).copied(),
                    right & 1,
                ) else {
                    continue;
                };
                next.push((
                    (left >> 1, right >> 1),
                    state_weight * multiplicity * left_factor * right_factor,
                ));
            }
        }
        states = merge_pair_states(next);
    }
    Ok(states
        .into_iter()
        .find_map(|(key, weight)| (key == (0, 0)).then_some(weight))
        .unwrap_or_else(F::zero))
}

fn merge_pair_states<F: FieldCore>(
    mut states: Vec<((usize, usize), F)>,
) -> Vec<((usize, usize), F)> {
    states.sort_unstable_by_key(|(key, _)| *key);
    let mut merged: Vec<((usize, usize), F)> = Vec::with_capacity(states.len());
    for (key, weight) in states {
        if weight.is_zero() {
            continue;
        }
        if let Some((last_key, last_weight)) = merged.last_mut() {
            if *last_key == key {
                *last_weight += weight;
                continue;
            }
        }
        merged.push((key, weight));
    }
    merged.retain(|(_, weight)| !weight.is_zero());
    merged
}

#[allow(clippy::too_many_arguments)]
fn collect_residual_seeds<F: FieldCore>(
    family: &EqPairTensorFamily<F>,
    recurrence_axes: &[usize],
    axis_index: usize,
    left_offset: usize,
    right_offset: usize,
    weight: F,
    seeds: &mut Vec<EqPairSeed<F>>,
    work: &mut usize,
) -> Result<(), AkitaError> {
    if weight.is_zero() {
        return Ok(());
    }
    if axis_index == family.axes.len() {
        charge_work(work, 1)?;
        seeds.push(EqPairSeed {
            left_offset,
            right_offset,
            weight,
        });
        return Ok(());
    }
    if recurrence_axes.contains(&axis_index) {
        return collect_residual_seeds(
            family,
            recurrence_axes,
            axis_index + 1,
            left_offset,
            right_offset,
            weight,
            seeds,
            work,
        );
    }
    let axis = family
        .axes
        .get(axis_index)
        .ok_or(AkitaError::InvalidProof)?;
    for coordinate in 0..axis.len {
        let axis_weight = match &axis.weights {
            EqPairTensorWeights::Unit => F::one(),
            EqPairTensorWeights::Dense(weights) => {
                *weights.get(coordinate).ok_or(AkitaError::InvalidProof)?
            }
        };
        if axis_weight.is_zero() {
            continue;
        }
        let next_weight = if axis_weight == F::one() {
            weight
        } else if weight == F::one() {
            axis_weight
        } else {
            weight * axis_weight
        };
        collect_residual_seeds(
            family,
            recurrence_axes,
            axis_index + 1,
            checked_axis_offset(left_offset, axis.left_stride, coordinate, "left")?,
            checked_axis_offset(right_offset, axis.right_stride, coordinate, "right")?,
            next_weight,
            seeds,
            work,
        )?;
    }
    Ok(())
}

fn unit_axis_choices<F: FieldCore>(
    introductions: &[(usize, usize)],
    work: &mut usize,
) -> Result<Vec<(usize, usize, F)>, AkitaError> {
    let mut choices = BTreeMap::<(usize, usize), F>::from([((0, 0), F::one())]);
    for &(left, right) in introductions {
        charge_work(work, choices.len())?;
        let previous = choices
            .iter()
            .map(|(&key, &weight)| (key, weight))
            .collect::<Vec<_>>();
        for ((left_sum, right_sum), weight) in previous {
            let next_left = left_sum.checked_add(left).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor left choice overflow".into())
            })?;
            let next_right = right_sum.checked_add(right).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor right choice overflow".into())
            })?;
            *choices.entry((next_left, next_right)).or_insert(F::zero()) += weight;
        }
    }
    Ok(choices
        .into_iter()
        .filter_map(|((left, right), weight)| (!weight.is_zero()).then_some((left, right, weight)))
        .collect())
}

fn basis_bit_factor<F: FieldCore, const MONOMIAL: bool>(
    challenge: Option<F>,
    bit: usize,
) -> Option<F> {
    match (challenge, bit) {
        (Some(_), 0) if MONOMIAL => Some(F::one()),
        (Some(challenge), 0) => Some(F::one() - challenge),
        (Some(challenge), 1) => Some(challenge),
        (Some(_), _) => None,
        (None, 0) => Some(F::one()),
        (None, _) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_tensor_family_seeds<F: FieldCore>(
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
            let axis = family
                .axes
                .get(stream_axis)
                .ok_or(AkitaError::InvalidProof)?;
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
        .ok_or(AkitaError::InvalidProof)?;
    for coordinate in 0..axis.len {
        let axis_weight = match &axis.weights {
            EqPairTensorWeights::Unit => F::one(),
            EqPairTensorWeights::Dense(weights) => {
                *weights.get(coordinate).ok_or(AkitaError::InvalidProof)?
            }
        };
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
struct EqPairSeed<F> {
    left_offset: usize,
    right_offset: usize,
    weight: F,
}

fn eval_tensor_seed_batch<F: FieldCore, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
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
                .ok_or(AkitaError::InvalidProof)?
                .push(((left_carry, right_carry), seed.weight));
            block_base = block_base.checked_add(block_size).ok_or_else(|| {
                AkitaError::InvalidInput("paired tensor block coverage overflow".into())
            })?;
        }
    }

    let mut acc = F::zero();
    for (block_index_bits, seed_states) in blocks.into_iter().enumerate() {
        let mut states = merge_pair_states(seed_states);
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
            let left_challenge = *left_challenges.get(bit).ok_or(AkitaError::InvalidProof)?;
            let right_challenge = *right_challenges.get(bit).ok_or(AkitaError::InvalidProof)?;
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
                    let left_factor =
                        basis_bit_factor::<F, LEFT_MONOMIAL>(Some(left_challenge), left_sum & 1)
                            .ok_or(AkitaError::InvalidProof)?;
                    let right_factor =
                        basis_bit_factor::<F, RIGHT_MONOMIAL>(Some(right_challenge), right_sum & 1)
                            .ok_or(AkitaError::InvalidProof)?;
                    next.push((
                        (left_sum >> 1, right_sum >> 1),
                        state_weight * left_factor * right_factor,
                    ));
                }
            }
            states = merge_pair_states(next);
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

fn finish_seed_states<F: FieldCore, const LEFT_MONOMIAL: bool, const RIGHT_MONOMIAL: bool>(
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
        for ((left_carry, right_carry), state_weight) in states {
            let Some((left_high, left_factor)) =
                basis_carry_step::<F, LEFT_MONOMIAL>(left_challenges.get(bit).copied(), left_carry)
            else {
                continue;
            };
            let Some((right_high, right_factor)) = basis_carry_step::<F, RIGHT_MONOMIAL>(
                right_challenges.get(bit).copied(),
                right_carry,
            ) else {
                continue;
            };
            next.push((
                (left_high, right_high),
                state_weight * left_factor * right_factor,
            ));
        }
        states = merge_pair_states(next);
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

fn basis_carry_step<F: FieldCore, const MONOMIAL: bool>(
    challenge: Option<F>,
    carry: usize,
) -> Option<(usize, F)> {
    if let Some(challenge) = challenge {
        basis_bit_factor::<F, MONOMIAL>(Some(challenge), carry & 1)
            .map(|factor| (carry >> 1, factor))
    } else if carry == 0 {
        Some((0, F::one()))
    } else {
        None
    }
}

fn basis_eval_checked<F: FieldCore, const MONOMIAL: bool>(challenges: &[F], index: usize) -> F {
    if challenges.len() < usize::BITS as usize && index >= 1usize << challenges.len() {
        F::zero()
    } else {
        basis_eval_at_index::<F, MONOMIAL>(challenges, index)
    }
}

fn basis_eval_at_index<F: FieldCore, const MONOMIAL: bool>(challenges: &[F], index: usize) -> F {
    if MONOMIAL {
        challenges
            .iter()
            .enumerate()
            .filter(|(bit, _)| index & (1usize << bit) != 0)
            .fold(F::one(), |weight, (_, &challenge)| weight * challenge)
    } else {
        eq_eval_at_index(challenges, index)
    }
}
