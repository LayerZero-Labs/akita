use super::{eq_eval_at_index, MAX_COMPACT_STRIDE_TERMS};
use crate::{AkitaError, FieldCore};
use std::collections::BTreeMap;

/// Evaluate an exact contraction between two affine equality-address streams.
///
/// The interval is decomposed into aligned power-of-two blocks. Within each
/// block, a sparse recurrence tracks the pair of carries produced by adding
/// one shared index bit into the two affine addresses.
///
/// # Errors
///
/// Returns an error for zero strides, arithmetic overflow, unsupported
/// challenge arity, or recurrence work above [`MAX_COMPACT_STRIDE_TERMS`].
#[allow(clippy::too_many_arguments)]
pub fn eval_compact_pair_eq<F: FieldCore>(
    left_challenges: &[F],
    left_offset: usize,
    left_stride: usize,
    right_challenges: &[F],
    right_offset: usize,
    right_stride: usize,
    len: usize,
) -> Result<F, AkitaError> {
    if left_stride == 0 || right_stride == 0 {
        return Err(AkitaError::InvalidInput(
            "compact-pair strides must be non-zero".into(),
        ));
    }
    if left_challenges.len() >= usize::BITS as usize
        || right_challenges.len() >= usize::BITS as usize
    {
        return Err(AkitaError::InvalidSize {
            expected: usize::BITS as usize - 1,
            actual: left_challenges.len().max(right_challenges.len()),
        });
    }
    if len == 0 {
        return Ok(F::zero());
    }
    let last = len - 1;
    left_stride
        .checked_mul(last)
        .and_then(|delta| left_offset.checked_add(delta))
        .ok_or_else(|| AkitaError::InvalidInput("compact-pair left address overflow".into()))?;
    right_stride
        .checked_mul(last)
        .and_then(|delta| right_offset.checked_add(delta))
        .ok_or_else(|| AkitaError::InvalidInput("compact-pair right address overflow".into()))?;

    let left_domain = 1usize << left_challenges.len();
    let right_domain = 1usize << right_challenges.len();
    if left_offset >= left_domain || right_offset >= right_domain {
        return Ok(F::zero());
    }
    let left_live = (left_domain - 1 - left_offset) / left_stride + 1;
    let right_live = (right_domain - 1 - right_offset) / right_stride + 1;
    let live_len = len.min(left_live).min(right_live);
    if live_len == 0 {
        return Ok(F::zero());
    }
    if live_len > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: live_len,
        });
    }

    let highest_bit = usize::BITS as usize - 1 - live_len.leading_zeros() as usize;
    let mut block_base = 0usize;
    let mut work = 0usize;
    let mut acc = F::zero();
    for block_index_bits in (0..=highest_bit).rev() {
        let block_size = 1usize << block_index_bits;
        if live_len & block_size == 0 {
            continue;
        }
        acc += eval_compact_pair_pow2_block(
            left_challenges,
            left_offset,
            left_stride,
            right_challenges,
            right_offset,
            right_stride,
            block_base,
            block_index_bits,
            &mut work,
        )?;
        block_base = block_base.checked_add(block_size).ok_or_else(|| {
            AkitaError::InvalidInput("compact-pair block coverage overflow".into())
        })?;
    }
    Ok(acc)
}

#[allow(clippy::too_many_arguments)]
fn eval_compact_pair_pow2_block<F: FieldCore>(
    left_challenges: &[F],
    left_offset: usize,
    left_stride: usize,
    right_challenges: &[F],
    right_offset: usize,
    right_stride: usize,
    block_base: usize,
    block_index_bits: usize,
    work: &mut usize,
) -> Result<F, AkitaError> {
    if block_index_bits > left_challenges.len() || block_index_bits > right_challenges.len() {
        return Err(AkitaError::InvalidInput(
            "compact-pair block exceeds equality arity".into(),
        ));
    }
    let left_carry = left_stride
        .checked_mul(block_base)
        .and_then(|delta| left_offset.checked_add(delta))
        .ok_or_else(|| AkitaError::InvalidInput("compact-pair left address overflow".into()))?;
    let right_carry = right_stride
        .checked_mul(block_base)
        .and_then(|delta| right_offset.checked_add(delta))
        .ok_or_else(|| AkitaError::InvalidInput("compact-pair right address overflow".into()))?;
    let mut states = BTreeMap::from([((left_carry, right_carry), F::one())]);
    for bit in 0..block_index_bits {
        let left_challenge = *left_challenges.get(bit).ok_or(AkitaError::InvalidProof)?;
        let right_challenge = *right_challenges.get(bit).ok_or(AkitaError::InvalidProof)?;
        *work = work
            .checked_add(states.len().checked_mul(2).ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair recurrence work overflow".into())
            })?)
            .ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair recurrence work overflow".into())
            })?;
        if *work > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: *work,
            });
        }
        let mut next = BTreeMap::new();
        for ((left_carry, right_carry), state_weight) in states {
            for index_bit in 0..=1usize {
                let left_sum = if index_bit == 0 {
                    left_carry
                } else {
                    left_carry.checked_add(left_stride).ok_or_else(|| {
                        AkitaError::InvalidInput("compact-pair left carry overflow".into())
                    })?
                };
                let right_sum = if index_bit == 0 {
                    right_carry
                } else {
                    right_carry.checked_add(right_stride).ok_or_else(|| {
                        AkitaError::InvalidInput("compact-pair right carry overflow".into())
                    })?
                };
                let left_factor = if left_sum & 1 == 1 {
                    left_challenge
                } else {
                    F::one() - left_challenge
                };
                let right_factor = if right_sum & 1 == 1 {
                    right_challenge
                } else {
                    F::one() - right_challenge
                };
                *next
                    .entry((left_sum >> 1, right_sum >> 1))
                    .or_insert(F::zero()) += state_weight * left_factor * right_factor;
            }
        }
        states = next;
    }

    let left_high_challenges = left_challenges
        .get(block_index_bits..)
        .ok_or(AkitaError::InvalidProof)?;
    let right_high_challenges = right_challenges
        .get(block_index_bits..)
        .ok_or(AkitaError::InvalidProof)?;
    Ok(states
        .into_iter()
        .map(|((left_high, right_high), state_weight)| {
            state_weight
                * eq_eval_at_index(left_high_challenges, left_high)
                * eq_eval_at_index(right_high_challenges, right_high)
        })
        .sum())
}
