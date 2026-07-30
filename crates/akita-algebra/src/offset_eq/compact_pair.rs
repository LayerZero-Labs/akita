use super::{eq_eval_at_index, MAX_COMPACT_STRIDE_TERMS};
use crate::{AkitaError, FieldCore};
#[cfg(feature = "parallel")]
use akita_field::parallel::*;
use std::collections::{BTreeMap, HashMap};

/// One weighted pair of affine equality-address streams.
///
/// The represented contribution is
///
/// ```text
/// weight * sum_{i < len}
///     eq(left_challenges, left_offset + left_stride*i)
///   * eq(right_challenges, right_offset + right_stride*i).
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightedCompactPairTerm<F: FieldCore> {
    /// First address in the left equality domain.
    pub left_offset: usize,
    /// Positive left-address stride.
    pub left_stride: usize,
    /// First address in the right equality domain.
    pub right_offset: usize,
    /// Positive right-address stride.
    pub right_stride: usize,
    /// Exact number of occurrences before domain clipping.
    pub len: usize,
    /// Scalar applied to this affine-pair contraction.
    pub weight: F,
}

/// Evaluate a weighted union of exact affine equality-address contractions.
///
/// Every non-power-of-two stream is decomposed into aligned power-of-two
/// blocks. Blocks with equal lengths and stride pairs seed one shared sparse
/// carry recurrence. Equal carry states merge by adding their weights, so
/// fragmented affine streams share the expensive block-index traversal without
/// requiring their address union itself to be affine.
///
/// Boolean challenges require no inversions. Addresses outside either equality
/// domain contribute zero.
///
/// # Errors
///
/// Returns an error for zero strides, arithmetic overflow, unsupported
/// challenge arity, or aggregate seed/recurrence work above
/// [`MAX_COMPACT_STRIDE_TERMS`].
pub fn eval_weighted_compact_pair_eq<F: FieldCore>(
    left_challenges: &[F],
    right_challenges: &[F],
    terms: &[WeightedCompactPairTerm<F>],
) -> Result<F, AkitaError> {
    if left_challenges.len() >= usize::BITS as usize
        || right_challenges.len() >= usize::BITS as usize
    {
        return Err(AkitaError::InvalidSize {
            expected: usize::BITS as usize - 1,
            actual: left_challenges.len().max(right_challenges.len()),
        });
    }
    let terms = if rectangle_preprocessing_worthwhile(terms)? {
        coalesce_weighted_compact_pair_terms(terms)?
    } else {
        checked_weighted_compact_pair_terms(terms)?
    };

    let left_domain = 1usize << left_challenges.len();
    let right_domain = 1usize << right_challenges.len();
    let mut batches = BTreeMap::<(usize, usize, usize), HashMap<(usize, usize), F>>::new();
    let mut seed_work = 0usize;
    for term in &terms {
        if term.left_offset >= left_domain || term.right_offset >= right_domain {
            continue;
        }
        let left_live = (left_domain - 1 - term.left_offset) / term.left_stride + 1;
        let right_live = (right_domain - 1 - term.right_offset) / term.right_stride + 1;
        let live_len = term.len.min(left_live).min(right_live);
        if live_len == 0 {
            continue;
        }
        if live_len > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: live_len,
            });
        }

        let highest_bit = usize::BITS as usize - 1 - live_len.leading_zeros() as usize;
        let mut block_base = 0usize;
        for block_index_bits in (0..=highest_bit).rev() {
            let block_size = 1usize << block_index_bits;
            if live_len & block_size == 0 {
                continue;
            }
            let left_carry = term
                .left_stride
                .checked_mul(block_base)
                .and_then(|delta| term.left_offset.checked_add(delta))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("compact-pair left address overflow".into())
                })?;
            let right_carry = term
                .right_stride
                .checked_mul(block_base)
                .and_then(|delta| term.right_offset.checked_add(delta))
                .ok_or_else(|| {
                    AkitaError::InvalidInput("compact-pair right address overflow".into())
                })?;
            charge_compact_pair_work(&mut seed_work, 1)?;
            *batches
                .entry((term.left_stride, term.right_stride, block_index_bits))
                .or_default()
                .entry((left_carry, right_carry))
                .or_insert(F::zero()) += term.weight;
            block_base = block_base.checked_add(block_size).ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair block coverage overflow".into())
            })?;
        }
    }
    let mut work = seed_work;
    let mut acc = F::zero();
    for ((left_stride, right_stride, block_index_bits), mut states) in batches {
        states.retain(|_, weight| !weight.is_zero());
        if states.is_empty() {
            continue;
        }
        if block_index_bits > left_challenges.len() || block_index_bits > right_challenges.len() {
            return Err(AkitaError::InvalidInput(
                "compact-pair block exceeds equality arity".into(),
            ));
        }
        for bit in 0..block_index_bits {
            let transition_work = states.len().checked_mul(2).ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair recurrence work overflow".into())
            })?;
            charge_compact_pair_work(&mut work, transition_work)?;
            let mut next = HashMap::new();
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
                        left_challenges[bit]
                    } else {
                        F::one() - left_challenges[bit]
                    };
                    let right_factor = if right_sum & 1 == 1 {
                        right_challenges[bit]
                    } else {
                        F::one() - right_challenges[bit]
                    };
                    *next
                        .entry((left_sum >> 1, right_sum >> 1))
                        .or_insert(F::zero()) += state_weight * left_factor * right_factor;
                }
            }
            next.retain(|_, weight| !weight.is_zero());
            states = next;
        }
        acc += finish_compact_pair_states(
            left_challenges,
            right_challenges,
            block_index_bits,
            states,
            &mut work,
        )?;
    }
    Ok(acc)
}

pub(super) fn charge_compact_pair_work(
    work: &mut usize,
    additional: usize,
) -> Result<(), AkitaError> {
    *work = work
        .checked_add(additional)
        .ok_or_else(|| AkitaError::InvalidInput("compact-pair work overflow".into()))?;
    if *work > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: *work,
        });
    }
    Ok(())
}

fn finish_compact_pair_states<F: FieldCore>(
    left_challenges: &[F],
    right_challenges: &[F],
    mut bit: usize,
    mut states: HashMap<(usize, usize), F>,
    work: &mut usize,
) -> Result<F, AkitaError> {
    let max_bits = left_challenges.len().max(right_challenges.len());
    while states.len() > 1 && bit < max_bits {
        charge_compact_pair_work(work, states.len())?;
        let mut next = HashMap::new();
        for ((left_carry, right_carry), state_weight) in states {
            let (left_high, left_factor) = if let Some(&challenge) = left_challenges.get(bit) {
                (
                    left_carry >> 1,
                    if left_carry & 1 == 1 {
                        challenge
                    } else {
                        F::one() - challenge
                    },
                )
            } else if left_carry == 0 {
                (0, F::one())
            } else {
                continue;
            };
            let (right_high, right_factor) = if let Some(&challenge) = right_challenges.get(bit) {
                (
                    right_carry >> 1,
                    if right_carry & 1 == 1 {
                        challenge
                    } else {
                        F::one() - challenge
                    },
                )
            } else if right_carry == 0 {
                (0, F::one())
            } else {
                continue;
            };
            *next.entry((left_high, right_high)).or_insert(F::zero()) +=
                state_weight * left_factor * right_factor;
        }
        next.retain(|_, weight| !weight.is_zero());
        states = next;
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
                |challenges| eq_eval_at_index(challenges, left_high),
            );
            let right_equality = right_challenges.get(bit..).map_or_else(
                || {
                    if right_high == 0 {
                        F::one()
                    } else {
                        F::zero()
                    }
                },
                |challenges| eq_eval_at_index(challenges, right_high),
            );
            state_weight * left_equality * right_equality
        })
        .sum())
}

pub(super) fn coalesce_weighted_compact_pair_terms<F: FieldCore>(
    terms: &[WeightedCompactPairTerm<F>],
) -> Result<Vec<WeightedCompactPairTerm<F>>, AkitaError> {
    let mut coalesced = checked_weighted_compact_pair_terms(terms)?;
    loop {
        // Exhaust cheap rectangle fusion before building the heavier
        // predecessor map. A contiguous merge can expose another tensor axis,
        // so repeat only when that merge actually changes the term set.
        loop {
            let previous_len = coalesced.len();
            coalesced = fuse_interleaved_pair_rectangles(coalesced)?;
            if coalesced.len() == previous_len {
                break;
            }
        }
        let previous_len = coalesced.len();
        coalesced = coalesce_contiguous_pair_terms(coalesced)?;
        if coalesced.len() == previous_len {
            return Ok(coalesced);
        }
    }
}

fn checked_weighted_compact_pair_terms<F: FieldCore>(
    terms: &[WeightedCompactPairTerm<F>],
) -> Result<Vec<WeightedCompactPairTerm<F>>, AkitaError> {
    if terms.len() > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: terms.len(),
        });
    }
    let mut sorted = Vec::new();
    sorted
        .try_reserve_exact(terms.len())
        .map_err(|_| AkitaError::InvalidInput("compact-pair term allocation failed".into()))?;
    for term in terms {
        if term.left_stride == 0 || term.right_stride == 0 {
            return Err(AkitaError::InvalidInput(
                "compact-pair strides must be non-zero".into(),
            ));
        }
        if term.len == 0 {
            continue;
        }
        let last = term.len - 1;
        term.left_stride
            .checked_mul(last)
            .and_then(|delta| term.left_offset.checked_add(delta))
            .ok_or_else(|| AkitaError::InvalidInput("compact-pair left address overflow".into()))?;
        term.right_stride
            .checked_mul(last)
            .and_then(|delta| term.right_offset.checked_add(delta))
            .ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair right address overflow".into())
            })?;
        if !term.weight.is_zero() {
            sorted.push(*term);
        }
    }
    Ok(sorted)
}

/// Minimum optimistic compression required before rectangle preprocessing.
///
/// Sorting and hashing every fragmented stream regressed production shapes
/// where width-two axes could remove at most half the seeds. Require a strict
/// improvement over this ratio; the recurrence still merges equal carry
/// states without preprocessing.
const RECTANGLE_PREPROCESSING_MIN_COMPRESSION: usize = 2;

/// Decide whether sorting affine terms into tensor rectangles is worthwhile.
///
/// This is deliberately a rectangle policy, not a recurrence cost model.
/// Equal weighted shapes can fuse by at most the greatest common divisor of
/// their two outer strides. Contiguous-chain fusion remains an opportunistic
/// follow-up once a rectangle-rich workload crosses the production-measured
/// threshold.
pub(super) fn rectangle_preprocessing_worthwhile<F: FieldCore>(
    terms: &[WeightedCompactPairTerm<F>],
) -> Result<bool, AkitaError> {
    if terms.len() > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: terms.len(),
        });
    }
    let mut shape_counts = BTreeMap::<(usize, usize, usize), usize>::new();
    let mut live_terms = 0usize;
    for term in terms {
        if term.len == 0 || term.weight.is_zero() {
            continue;
        }
        live_terms = live_terms
            .checked_add(1)
            .ok_or_else(|| AkitaError::InvalidInput("compact-pair term count overflow".into()))?;
        *shape_counts
            .entry((term.left_stride, term.right_stride, term.len))
            .or_default() += 1;
    }
    let optimistic_shapes =
        shape_counts
            .into_iter()
            .try_fold(0usize, |total, (shape, count)| {
                let width = greatest_common_divisor(shape.0, shape.1).max(1);
                total.checked_add(count.div_ceil(width)).ok_or_else(|| {
                    AkitaError::InvalidInput("compact-pair normalized term count overflow".into())
                })
            })?;
    if optimistic_shapes
        .checked_mul(RECTANGLE_PREPROCESSING_MIN_COMPRESSION)
        .is_none_or(|optimistic| optimistic >= live_terms)
    {
        return Ok(false);
    }

    // Field weights can split one geometric shape into independent families.
    // Hash them only after the integer-only bound says a substantial collapse
    // is possible; fragmented shapes avoid this heavier preparation entirely.
    let mut weighted_counts = HashMap::<(usize, usize, usize, F), usize>::new();
    weighted_counts
        .try_reserve(terms.len())
        .map_err(|_| AkitaError::InvalidInput("compact-pair geometry allocation failed".into()))?;
    for term in terms {
        if term.len == 0 || term.weight.is_zero() {
            continue;
        }
        *weighted_counts
            .entry((term.left_stride, term.right_stride, term.len, term.weight))
            .or_default() += 1;
    }
    let optimistic_terms =
        weighted_counts
            .into_iter()
            .try_fold(0usize, |total, (shape, count)| {
                let width = greatest_common_divisor(shape.0, shape.1).max(1);
                total.checked_add(count.div_ceil(width)).ok_or_else(|| {
                    AkitaError::InvalidInput("compact-pair normalized term count overflow".into())
                })
            })?;
    Ok(optimistic_terms
        .checked_mul(RECTANGLE_PREPROCESSING_MIN_COMPRESSION)
        .is_some_and(|optimistic| optimistic < live_terms))
}

fn greatest_common_divisor(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        (lhs, rhs) = (rhs, lhs % rhs);
    }
    lhs
}

fn coalesce_contiguous_pair_terms<F: FieldCore>(
    mut terms: Vec<WeightedCompactPairTerm<F>>,
) -> Result<Vec<WeightedCompactPairTerm<F>>, AkitaError> {
    #[cfg(feature = "parallel")]
    terms.par_sort_unstable_by_key(|term| {
        (
            term.left_stride,
            term.right_stride,
            term.left_offset,
            term.right_offset,
            term.len,
        )
    });
    #[cfg(not(feature = "parallel"))]
    terms.sort_unstable_by_key(|term| {
        (
            term.left_stride,
            term.right_stride,
            term.left_offset,
            term.right_offset,
            term.len,
        )
    });
    // Include the field weight in the hash key so predecessor lookup stays
    // constant-time even when many projection lanes share one address.
    let mut pending =
        HashMap::<(usize, usize, usize, usize, F), Vec<WeightedCompactPairTerm<F>>>::new();
    let mut complete = Vec::new();
    complete
        .try_reserve_exact(terms.len())
        .map_err(|_| AkitaError::InvalidInput("compact-pair term allocation failed".into()))?;
    for term in terms {
        let start_key = (
            term.left_stride,
            term.right_stride,
            term.left_offset,
            term.right_offset,
            term.weight,
        );
        let predecessor = pending
            .get_mut(&start_key)
            .and_then(|candidates| candidates.pop());
        if pending
            .get(&start_key)
            .is_some_and(|candidates| candidates.is_empty())
        {
            pending.remove(&start_key);
        }
        let mut chain = predecessor.unwrap_or(term);
        if predecessor.is_some() {
            chain.len = chain.len.checked_add(term.len).ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair coalesced length overflow".into())
            })?;
        }
        let next_left = chain
            .left_stride
            .checked_mul(chain.len)
            .and_then(|delta| chain.left_offset.checked_add(delta));
        let next_right = chain
            .right_stride
            .checked_mul(chain.len)
            .and_then(|delta| chain.right_offset.checked_add(delta));
        if let (Some(next_left), Some(next_right)) = (next_left, next_right) {
            pending
                .entry((
                    chain.left_stride,
                    chain.right_stride,
                    next_left,
                    next_right,
                    chain.weight,
                ))
                .or_default()
                .push(chain);
        } else {
            complete.push(chain);
        }
    }
    complete.extend(pending.into_values().flatten());
    Ok(complete)
}

/// Fuse a rectangular family of interleaved affine streams.
///
/// If `width` equal-weight streams have outer strides
/// `(width * left_inner, width * right_inner)` and starts advancing by
/// `(left_inner, right_inner)`, their union is exactly one affine stream with
/// the inner strides. This recognizes tensor axes from address geometry alone:
/// no caller, role, or physical-subcolumn special case is involved.
fn fuse_interleaved_pair_rectangles<F: FieldCore>(
    mut terms: Vec<WeightedCompactPairTerm<F>>,
) -> Result<Vec<WeightedCompactPairTerm<F>>, AkitaError> {
    #[cfg(feature = "parallel")]
    terms.par_sort_unstable_by_key(|term| {
        (
            term.left_stride,
            term.right_stride,
            term.len,
            term.left_offset,
            term.right_offset,
        )
    });
    #[cfg(not(feature = "parallel"))]
    terms.sort_unstable_by_key(|term| {
        (
            term.left_stride,
            term.right_stride,
            term.len,
            term.left_offset,
            term.right_offset,
        )
    });
    let mut fused = Vec::new();
    fused
        .try_reserve_exact(terms.len())
        .map_err(|_| AkitaError::InvalidInput("compact-pair term allocation failed".into()))?;

    let mut index = 0usize;
    while index < terms.len() {
        let base = terms[index];
        let Some(next) = terms.get(index + 1).copied() else {
            fused.push(base);
            break;
        };
        let same_shape = next.left_stride == base.left_stride
            && next.right_stride == base.right_stride
            && next.len == base.len
            && next.weight == base.weight;
        let left_inner = next.left_offset.checked_sub(base.left_offset);
        let right_inner = next.right_offset.checked_sub(base.right_offset);
        let fusion = match (same_shape, left_inner, right_inner) {
            (true, Some(left_inner), Some(right_inner))
                if left_inner != 0
                    && right_inner != 0
                    && base.left_stride.is_multiple_of(left_inner)
                    && base.right_stride.is_multiple_of(right_inner)
                    && base.left_stride / left_inner == base.right_stride / right_inner =>
            {
                Some((left_inner, right_inner, base.left_stride / left_inner))
            }
            _ => None,
        };
        let Some((left_inner, right_inner, width)) = fusion else {
            fused.push(base);
            index += 1;
            continue;
        };
        if width < 2 || index.checked_add(width).is_none_or(|end| end > terms.len()) {
            fused.push(base);
            index += 1;
            continue;
        }
        let family = &terms[index..index + width];
        let complete_rectangle = family.iter().enumerate().all(|(lane, term)| {
            term.left_stride == base.left_stride
                && term.right_stride == base.right_stride
                && term.len == base.len
                && term.weight == base.weight
                && left_inner
                    .checked_mul(lane)
                    .and_then(|delta| base.left_offset.checked_add(delta))
                    == Some(term.left_offset)
                && right_inner
                    .checked_mul(lane)
                    .and_then(|delta| base.right_offset.checked_add(delta))
                    == Some(term.right_offset)
        });
        if !complete_rectangle {
            fused.push(base);
            index += 1;
            continue;
        }

        fused.push(WeightedCompactPairTerm {
            left_offset: base.left_offset,
            left_stride: left_inner,
            right_offset: base.right_offset,
            right_stride: right_inner,
            len: base.len.checked_mul(width).ok_or_else(|| {
                AkitaError::InvalidInput("compact-pair fused length overflow".into())
            })?,
            weight: base.weight,
        });
        index += width;
    }
    Ok(fused)
}
