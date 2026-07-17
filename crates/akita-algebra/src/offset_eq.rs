//! Offset-EQ helpers for structured inner products.
//!
//! The production evaluator is [`eval_affine_digit_interval`]. It contracts an
//! exact affine digit interval against factored outer weights while preserving
//! carries from arbitrary physical offsets. [`eq_eval_at_index`] is the scalar
//! equality primitive shared by the kernel and direct callers.

use crate::{AkitaError, FieldCore};
use std::collections::BTreeMap;

/// Verifier work cap for one compact-stride equality contraction.
pub const MAX_COMPACT_STRIDE_TERMS: usize = 1 << 28;

/// Coefficient algebra used by [`eval_affine_digit_interval`].
///
/// The equality and digit factors live in `F`; outer high/low factors may live
/// either in `F` itself or in a small coordinate algebra that is linear over
/// `F`. Keeping these operations abstract lets the trace evaluator preserve
/// its factored extension coordinates without introducing another address
/// kernel.
pub trait AffineWeight<F: FieldCore>: Clone {
    /// Additive identity carrying the same algebra metadata as `self`.
    fn zero_like(&self) -> Self;

    /// Add `factor * scale` to `self`.
    fn add_scaled(&mut self, factor: &Self, scale: F);

    /// Multiply two outer factors.
    fn multiply(&self, rhs: &Self) -> Self;
}

impl<F: FieldCore> AffineWeight<F> for F {
    fn zero_like(&self) -> Self {
        Self::zero()
    }

    fn add_scaled(&mut self, factor: &Self, scale: F) {
        *self += *factor * scale;
    }

    fn multiply(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}

/// Evaluate one exact digit-innermost affine interval with factored outer weights.
///
/// For `Q = low_weights.len()` and `i` in the exact global outer window
/// `[outer_start, outer_start + live_len)`, this computes
///
/// ```text
/// Σ_i Σ_d high[i / Q] · low[i % Q] · digit[d]
///            · eq(challenges, base_offset + outer_stride · (i - outer_start) + d).
/// ```
///
/// `Q` must be a power of two. The implementation splits the equality point at
/// `log2(Q)`, summarizes the low factor into at most `outer_stride + 1` carry
/// states, and reuses that summary for every complete high row. Unaligned first
/// and last rows are handled as exact low-factor subwindows, so distributed
/// chunks and a partial final tensor row do not enumerate the Cartesian
/// high-by-low domain. Boolean challenges require no inversion.
///
/// # Errors
///
/// Returns an error for malformed factors, an out-of-range outer window,
/// address overflow, insufficient equality arity, or work above
/// [`MAX_COMPACT_STRIDE_TERMS`]. The work bound is checked before allocating
/// carry summaries.
#[allow(clippy::too_many_arguments)]
pub fn eval_affine_digit_interval<F, A>(
    challenges: &[F],
    base_offset: usize,
    outer_start: usize,
    live_len: usize,
    outer_stride: usize,
    digit_weights: &[F],
    high_weights: &[A],
    low_weights: &[A],
) -> Result<A, AkitaError>
where
    F: FieldCore,
    A: AffineWeight<F>,
{
    eval_affine_digit_interval_impl(
        challenges,
        base_offset,
        outer_start,
        live_len,
        outer_stride,
        digit_weights,
        high_weights,
        low_weights,
        false,
    )
}

/// Verifier-only fast variant of [`eval_affine_digit_interval`].
///
/// Computes the **identical** sum, but contracts the fold-high rows with a
/// bounded high-equality table and a carry-bucketed inner product instead of the
/// base kernel's `rows * carry_count` on-demand `eq` evaluations. Concretely the
/// dominant high term drops from `O(rows * carry_count * high_bits)` to
/// `O(rows + carry_count^2)` table lookups, removing the digit/carry factor from
/// the fold-high dimension. The low carry-summary and every address are shared
/// with the base kernel, so results match [`eval_affine_digit_interval`]
/// bit-for-bit; it transparently falls back to the base row loop when the high
/// domain is smaller than the carry window or the row count is too small to
/// amortize the table setup.
///
/// This is intended for the verifier's structured relation-matrix evaluation.
/// It does not change the prover's witness layout or decomposition path.
#[allow(clippy::too_many_arguments)]
pub fn eval_affine_digit_interval_fast<F, A>(
    challenges: &[F],
    base_offset: usize,
    outer_start: usize,
    live_len: usize,
    outer_stride: usize,
    digit_weights: &[F],
    high_weights: &[A],
    low_weights: &[A],
) -> Result<A, AkitaError>
where
    F: FieldCore,
    A: AffineWeight<F>,
{
    eval_affine_digit_interval_impl(
        challenges,
        base_offset,
        outer_start,
        live_len,
        outer_stride,
        digit_weights,
        high_weights,
        low_weights,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn eval_affine_digit_interval_impl<F, A>(
    challenges: &[F],
    base_offset: usize,
    outer_start: usize,
    live_len: usize,
    outer_stride: usize,
    digit_weights: &[F],
    high_weights: &[A],
    low_weights: &[A],
    fast: bool,
) -> Result<A, AkitaError>
where
    F: FieldCore,
    A: AffineWeight<F>,
{
    let template = high_weights
        .first()
        .or_else(|| low_weights.first())
        .ok_or_else(|| AkitaError::InvalidInput("affine factors must be non-empty".into()))?;
    if live_len == 0 {
        return Ok(template.zero_like());
    }
    let low_len = low_weights.len();
    if !low_len.is_power_of_two() || digit_weights.is_empty() || outer_stride < digit_weights.len()
    {
        return Err(AkitaError::InvalidInput(
            "affine digit geometry requires power-of-two low length and a stride covering every digit"
                .into(),
        ));
    }
    let low_bits = low_len.trailing_zeros() as usize;
    if low_bits > challenges.len() {
        return Err(AkitaError::InvalidSize {
            expected: low_bits,
            actual: challenges.len(),
        });
    }
    let outer_end = outer_start
        .checked_add(live_len)
        .ok_or_else(|| AkitaError::InvalidInput("affine outer window overflow".into()))?;
    let outer_capacity = high_weights
        .len()
        .checked_mul(low_len)
        .ok_or_else(|| AkitaError::InvalidInput("affine outer capacity overflow".into()))?;
    if outer_end > outer_capacity {
        return Err(AkitaError::InvalidSize {
            expected: outer_capacity,
            actual: outer_end,
        });
    }
    let digit_count = digit_weights.len();
    let carry_count = outer_stride
        .checked_add(1)
        .ok_or_else(|| AkitaError::InvalidInput("affine carry count overflow".into()))?;
    let max_address = base_offset
        .checked_add(
            outer_stride
                .checked_mul(live_len - 1)
                .and_then(|delta| delta.checked_add(digit_count - 1))
                .ok_or_else(|| AkitaError::InvalidInput("affine address overflow".into()))?,
        )
        .ok_or_else(|| AkitaError::InvalidInput("affine address overflow".into()))?;
    if challenges.len() < usize::BITS as usize && max_address >= (1usize << challenges.len()) {
        return Err(AkitaError::InvalidSize {
            expected: challenges.len() + 1,
            actual: challenges.len(),
        });
    }

    let mut cursor = outer_start;
    let prefix_end = if cursor.is_multiple_of(low_len) {
        cursor
    } else {
        outer_end.min(
            cursor
                .checked_add(low_len - cursor % low_len)
                .ok_or_else(|| AkitaError::InvalidInput("affine row boundary overflow".into()))?,
        )
    };
    let suffix_start = outer_end - outer_end % low_len;
    let full_start = prefix_end;
    let full_end = suffix_start.max(full_start).min(outer_end);
    let prefix_span = prefix_end - cursor;
    cursor = prefix_end;
    let full_rows = full_end.saturating_sub(cursor) / low_len;
    cursor =
        cursor
            .checked_add(full_rows.checked_mul(low_len).ok_or_else(|| {
                AkitaError::InvalidInput("affine full-row coverage overflow".into())
            })?)
            .ok_or_else(|| AkitaError::InvalidInput("affine full-row coverage overflow".into()))?;
    let suffix_span = outer_end - cursor;
    let summarized_low = prefix_span
        .checked_add(suffix_span)
        .and_then(|span| span.checked_add(if full_rows == 0 { 0 } else { low_len }))
        .ok_or_else(|| AkitaError::InvalidInput("affine low work overflow".into()))?;
    let row_count = usize::from(prefix_span != 0)
        .checked_add(full_rows)
        .and_then(|rows| rows.checked_add(usize::from(suffix_span != 0)))
        .ok_or_else(|| AkitaError::InvalidInput("affine row work overflow".into()))?;
    let work = digit_count
        .checked_mul(summarized_low)
        .and_then(|low_work| {
            row_count
                .checked_mul(carry_count)
                .and_then(|high_work| low_work.checked_add(high_work))
        })
        .ok_or_else(|| AkitaError::InvalidInput("affine work overflow".into()))?;
    if work > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: work,
        });
    }

    let low_challenges = &challenges[..low_bits];
    let high_challenges = &challenges[low_bits..];
    let mut out = template.zero_like();
    if prefix_span != 0 {
        accumulate_affine_rows(
            fast,
            &mut out,
            low_challenges,
            high_challenges,
            base_offset,
            outer_start,
            outer_stride,
            digit_weights,
            high_weights,
            low_weights,
            outer_start / low_len,
            outer_start % low_len,
            outer_start % low_len + prefix_span,
            1,
        )?;
    }
    if full_rows != 0 {
        accumulate_affine_rows(
            fast,
            &mut out,
            low_challenges,
            high_challenges,
            base_offset,
            outer_start,
            outer_stride,
            digit_weights,
            high_weights,
            low_weights,
            full_start / low_len,
            0,
            low_len,
            full_rows,
        )?;
    }
    if suffix_span != 0 {
        accumulate_affine_rows(
            fast,
            &mut out,
            low_challenges,
            high_challenges,
            base_offset,
            outer_start,
            outer_stride,
            digit_weights,
            high_weights,
            low_weights,
            cursor / low_len,
            0,
            suffix_span,
            1,
        )?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_affine_rows<F, A>(
    fast: bool,
    out: &mut A,
    low_challenges: &[F],
    high_challenges: &[F],
    base_offset: usize,
    outer_start: usize,
    outer_stride: usize,
    digit_weights: &[F],
    high_weights: &[A],
    low_weights: &[A],
    first_high: usize,
    low_start: usize,
    low_end: usize,
    rows: usize,
) -> Result<(), AkitaError>
where
    F: FieldCore,
    A: AffineWeight<F>,
{
    let low_len = low_weights.len();
    let carry_count = outer_stride
        .checked_add(1)
        .ok_or_else(|| AkitaError::InvalidInput("affine carry count overflow".into()))?;
    let row_outer = first_high
        .checked_mul(low_len)
        .and_then(|base| base.checked_add(low_start))
        .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?;
    let local_outer = row_outer
        .checked_sub(outer_start)
        .ok_or_else(|| AkitaError::InvalidInput("affine row precedes outer window".into()))?;
    let first_address = base_offset
        .checked_add(
            outer_stride
                .checked_mul(local_outer)
                .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?,
        )
        .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?;
    let low_mask = low_len - 1;
    let address_low = first_address & low_mask;
    let template = high_weights
        .get(first_high)
        .ok_or_else(|| AkitaError::InvalidInput("affine high factor out of range".into()))?;
    // Precompute the low equality table once and share it across every
    // (low, digit) term instead of recomputing `eq(low_challenges, ·)` from
    // scratch per term. `low_len == 2^low_bits` is the affine low factor width
    // (a fold count), which is bounded by the interval work check above, but we
    // still cap the materialization to keep the allocation bounded and fall
    // back to the scalar primitive for pathologically wide low blocks.
    let eq_low_table: Option<Vec<F>> = if low_challenges.len() <= OFFSET_EQ_LOW_BITS_CAP {
        Some(crate::eq_poly::EqPolynomial::evals(low_challenges)?)
    } else {
        None
    };
    let mut summaries = vec![template.zero_like(); carry_count];
    for low in low_start..low_end {
        let low_factor = low_weights
            .get(low)
            .ok_or_else(|| AkitaError::InvalidInput("affine low factor out of range".into()))?;
        let low_delta = outer_stride
            .checked_mul(low - low_start)
            .ok_or_else(|| AkitaError::InvalidInput("affine low address overflow".into()))?;
        for (digit, &digit_weight) in digit_weights.iter().enumerate() {
            let shifted = address_low
                .checked_add(low_delta)
                .and_then(|value| value.checked_add(digit))
                .ok_or_else(|| AkitaError::InvalidInput("affine low address overflow".into()))?;
            let carry = shifted / low_len;
            let low_index = shifted & low_mask;
            let eq_low = match &eq_low_table {
                Some(table) => table.get(low_index).copied().ok_or_else(|| {
                    AkitaError::InvalidInput("affine low index out of range".into())
                })?,
                None => eq_eval_at_index(low_challenges, low_index),
            };
            summaries
                .get_mut(carry)
                .ok_or_else(|| AkitaError::InvalidInput("affine carry out of range".into()))?
                .add_scaled(low_factor, digit_weight * eq_low);
        }
    }

    // Verifier fast path: contract the high rows with a bounded high-equality
    // table and carry-bucketing (O(rows + carry_count^2)) instead of the
    // `rows * carry_count` on-demand `eq` loop below. Returns `false` when the
    // geometry is ineligible, in which case we fall through to the base loop.
    if fast
        && accumulate_high_rows_bucketed(
            out,
            high_challenges,
            first_address,
            outer_stride,
            low_challenges.len(),
            high_weights,
            first_high,
            rows,
            &summaries,
        )?
    {
        return Ok(());
    }

    for row in 0..rows {
        let high_index = first_high
            .checked_add(row)
            .ok_or_else(|| AkitaError::InvalidInput("affine high index overflow".into()))?;
        let high_factor = high_weights
            .get(high_index)
            .ok_or_else(|| AkitaError::InvalidInput("affine high factor out of range".into()))?;
        let row_address = first_address
            .checked_add(
                outer_stride
                    .checked_mul(low_len)
                    .and_then(|stride| stride.checked_mul(row))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("affine high address overflow".into())
                    })?,
            )
            .ok_or_else(|| AkitaError::InvalidInput("affine high address overflow".into()))?;
        let address_high = row_address >> low_challenges.len();
        for (carry, summary) in summaries.iter().enumerate() {
            let eq_high = eq_eval_at_index(
                high_challenges,
                address_high.checked_add(carry).ok_or_else(|| {
                    AkitaError::InvalidInput("affine high address overflow".into())
                })?,
            );
            if !eq_high.is_zero() {
                out.add_scaled(&high_factor.multiply(summary), eq_high);
            }
        }
    }
    Ok(())
}

/// Minimum full-row count before the bucketed high contraction is worth its
/// table setup. Below this the base row loop is cheaper.
const FAST_HIGH_ROWS_MIN: usize = 8;

/// Contract the fold-high rows against the shared low carry summaries using a
/// bounded high-equality table and carry-bucketing.
///
/// For each row `r` the base kernel adds
/// `high[first_high + r] * summaries[carry] * eq(high_challenges, h0 + stride*r + carry)`
/// over every `carry in 0..summaries.len()`, where `h0 = first_address >> low_bits`.
/// Because the equality point splits at `log2(next_pow2(carry_count))`, each row's
/// carry window straddles at most two high-table blocks, so the whole double loop
/// factors into: (1) one pass over rows that buckets `high[r] * eq_hi(block)` by
/// the row's low-window position, and (2) one `carry_count * window` combine.
/// Total `O(rows + carry_count^2)` field ops with `O(1)` table lookups, versus
/// the base `O(rows * carry_count * high_bits)`.
///
/// Returns `Ok(true)` when it handled the rows, or `Ok(false)` when the geometry
/// is ineligible (high domain smaller than the carry window, or too few rows to
/// amortize setup) and the caller should use the base loop. The result is
/// bit-identical to the base loop for every eligible input.
#[allow(clippy::too_many_arguments)]
fn accumulate_high_rows_bucketed<F, A>(
    out: &mut A,
    high_challenges: &[F],
    first_address: usize,
    outer_stride: usize,
    low_bits: usize,
    high_weights: &[A],
    first_high: usize,
    rows: usize,
    summaries: &[A],
) -> Result<bool, AkitaError>
where
    F: FieldCore,
    A: AffineWeight<F>,
{
    let carry_count = summaries.len();
    if rows < FAST_HIGH_ROWS_MIN || carry_count == 0 {
        return Ok(false);
    }
    let template = summaries
        .first()
        .ok_or_else(|| AkitaError::InvalidInput("affine summaries empty".into()))?;

    // Split the high equality point so the carry window fits inside the low part.
    let window = carry_count.next_power_of_two();
    let window_bits = window.trailing_zeros() as usize;
    if window_bits > high_challenges.len() {
        // High domain is smaller than the carry window; base loop is cheap here.
        return Ok(false);
    }
    let low_hi = &high_challenges[..window_bits];
    let high_hi = &high_challenges[window_bits..];
    let eq_low_hi = crate::eq_poly::EqPolynomial::evals(low_hi)?; // length == window
    let split_high = crate::eq_poly::SplitEqEvals::new(high_hi)?;
    let high_domain: usize = if high_hi.len() >= usize::BITS as usize {
        usize::MAX
    } else {
        1usize << high_hi.len()
    };
    let eval_high = |block: usize| -> Result<F, AkitaError> {
        if block < high_domain {
            split_high.eval_at(block)
        } else {
            Ok(F::zero())
        }
    };

    let h0 = first_address >> low_bits;
    let window_mask = window - 1;
    // Bucket each row by its low-window position, split into the "no carry into
    // the next high block" bucket (`bucket0`) and the "carries" bucket (`bucket1`).
    let mut bucket0 = vec![template.zero_like(); window];
    let mut bucket1 = vec![template.zero_like(); window];
    for row in 0..rows {
        let high_index = first_high
            .checked_add(row)
            .ok_or_else(|| AkitaError::InvalidInput("affine high index overflow".into()))?;
        let high_factor = high_weights
            .get(high_index)
            .ok_or_else(|| AkitaError::InvalidInput("affine high factor out of range".into()))?;
        let address_high = h0
            .checked_add(
                outer_stride
                    .checked_mul(row)
                    .ok_or_else(|| AkitaError::InvalidInput("affine high address overflow".into()))?,
            )
            .ok_or_else(|| AkitaError::InvalidInput("affine high address overflow".into()))?;
        let low_pos = address_high & window_mask;
        let block = address_high >> window_bits;
        let eq_block0 = eval_high(block)?;
        let eq_block1 = eval_high(
            block
                .checked_add(1)
                .ok_or_else(|| AkitaError::InvalidInput("affine high block overflow".into()))?,
        )?;
        bucket0[low_pos].add_scaled(high_factor, eq_block0);
        bucket1[low_pos].add_scaled(high_factor, eq_block1);
    }

    // Combine: out += Σ_carry (Σ_pos bucket[pos] * eq_low_hi[(pos+carry) mod window]) * summaries[carry].
    for (carry, summary) in summaries.iter().enumerate() {
        let mut phi = template.zero_like();
        for pos in 0..window {
            let shifted = pos + carry;
            if shifted < window {
                phi.add_scaled(&bucket0[pos], eq_low_hi[shifted]);
            } else {
                phi.add_scaled(&bucket1[pos], eq_low_hi[shifted - window]);
            }
        }
        out.add_scaled(&phi.multiply(summary), F::one());
    }
    Ok(true)
}

/// Evaluate an exact contraction between two affine equality-address streams.
///
/// This computes
///
/// ```text
/// sum_{i < len}
///     eq(left_challenges, left_offset + left_stride*i)
///   * eq(right_challenges, right_offset + right_stride*i).
/// ```
///
/// The interval is decomposed into aligned power-of-two blocks. Within each
/// block, a sparse recurrence tracks the pair of carries produced by adding
/// one shared index bit into the two affine addresses. Non-power-of-two
/// lengths are therefore exact and do not require a padded domain.
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
        states = next;
    }

    Ok(states
        .into_iter()
        .map(|((left_high, right_high), state_weight)| {
            state_weight
                * eq_eval_at_index(&left_challenges[block_index_bits..], left_high)
                * eq_eval_at_index(&right_challenges[block_index_bits..], right_high)
        })
        .sum())
}

/// Hard cap on the number of low bits materialized by [`OffsetEqWindow`].
///
/// A 16-bit low table holds at most `2^16 = 65_536` field elements
/// (about 1 MiB for 16-byte elements), which bounds the allocation regardless
/// of the full point width.
pub const OFFSET_EQ_LOW_BITS_CAP: usize = 16;

/// Hard cap on the number of high bits materialized by [`OffsetEqWindow`].
///
/// When the high remainder has at most this many bits, the high equality table
/// `eq_high[j] = eq(high_challenges, j)` is materialized so that each `eval`
/// costs two table lookups and a single multiply. The cap bounds the high table
/// at `2^16` field elements; wider high remainders fall back to on-demand
/// `O(high_bits)` evaluation.
pub const OFFSET_EQ_HIGH_BITS_CAP: usize = 16;

/// Bounded checked equality-window evaluator.
///
/// An `n`-coordinate equality point is split into a low block of at most
/// [`OFFSET_EQ_LOW_BITS_CAP`] bits and a high remainder. The low equality table
/// `eq_low[i] = eq(low_challenges, i)` is materialized once (at most
/// `2^low_bits` elements). When the high remainder is at most
/// [`OFFSET_EQ_HIGH_BITS_CAP`] bits, its equality table `eq_high` is materialized
/// as well, so each `eval` is two bounded lookups and one multiply — removing the
/// per-address `O(high_bits)` factor. Wider high remainders fall back to
/// on-demand high evaluation. Either way the low table (and, when present, the
/// high table) is shared across every address in a canonical interval.
///
/// This obeys the verifier no-panic contract: construction validates and caps
/// both table widths, the lookups are range-checked, and no unbounded
/// allocation is performed.
pub struct OffsetEqWindow<'a, F: FieldCore> {
    low_bits: usize,
    low_mask: usize,
    eq_low: Vec<F>,
    eq_high: Option<Vec<F>>,
    high_challenges: &'a [F],
}

impl<'a, F: FieldCore> OffsetEqWindow<'a, F> {
    /// Build a window over `challenges` using the default low-bit cap.
    ///
    /// # Errors
    ///
    /// Returns an error if the low equality table cannot be constructed.
    pub fn new(challenges: &'a [F]) -> Result<Self, AkitaError> {
        Self::with_low_bits(challenges, OFFSET_EQ_LOW_BITS_CAP)
    }

    /// Build a window over `challenges` choosing `min(len, cap, CAP)` low bits.
    ///
    /// # Errors
    ///
    /// Returns an error if the low equality table cannot be constructed.
    pub fn with_low_bits(challenges: &'a [F], low_bits_cap: usize) -> Result<Self, AkitaError> {
        let low_bits = challenges
            .len()
            .min(low_bits_cap)
            .min(OFFSET_EQ_LOW_BITS_CAP);
        let eq_low = crate::eq_poly::EqPolynomial::evals(&challenges[..low_bits])?;
        let low_mask = if low_bits == 0 {
            0
        } else {
            (1usize << low_bits) - 1
        };
        let high_challenges = &challenges[low_bits..];
        // Materialize the high table too when it stays within the bounded cap.
        // This makes every `eval` a pair of lookups instead of recomputing an
        // `O(high_bits)` equality product per address, which dominated the
        // verifier setup-weight builders.
        let eq_high = if high_challenges.len() <= OFFSET_EQ_HIGH_BITS_CAP {
            Some(crate::eq_poly::EqPolynomial::evals(high_challenges)?)
        } else {
            None
        };
        Ok(Self {
            low_bits,
            low_mask,
            eq_low,
            eq_high,
            high_challenges,
        })
    }

    /// Evaluate `eq(challenges, index)` for a little-endian hypercube index.
    ///
    /// Matches [`eq_eval_at_index`] exactly, including returning zero for
    /// out-of-domain indices.
    #[inline]
    pub fn eval(&self, index: usize) -> F {
        let low = index & self.low_mask;
        // `low < 2^low_bits == eq_low.len()` by construction; the fallback keeps
        // the accessor panic-free without masking a real bug.
        let eq_low = self.eq_low.get(low).copied().unwrap_or_else(F::zero);
        if eq_low.is_zero() {
            return F::zero();
        }
        let high = index >> self.low_bits;
        let eq_high = match &self.eq_high {
            // A high index beyond the materialized table is out of the equality
            // domain, so it contributes zero (matching `eq_eval_at_index`).
            Some(table) => table.get(high).copied().unwrap_or_else(F::zero),
            None => eq_eval_at_index(self.high_challenges, high),
        };
        eq_low * eq_high
    }
}

/// Evaluate `eq(r, index)` for a single hypercube index in little-endian order.
pub fn eq_eval_at_index<F: FieldCore>(x_challenges: &[F], index: usize) -> F {
    if x_challenges.len() < usize::BITS as usize && index >= (1usize << x_challenges.len()) {
        return F::zero();
    }

    x_challenges
        .iter()
        .enumerate()
        .fold(F::one(), |acc, (bit_idx, &r_t)| {
            let bit = if bit_idx < usize::BITS as usize {
                (index >> bit_idx) & 1
            } else {
                0
            };
            acc * if bit == 1 { r_t } else { F::one() - r_t }
        })
}

#[cfg(test)]
mod tests;
