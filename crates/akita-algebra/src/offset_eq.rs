//! Offset-EQ helpers for structured inner products.
//!
//! The production evaluator is [`eval_affine_digit_intervals`]. It contracts
//! exact affine digit interval against factored outer weights while preserving
//! carries from arbitrary physical offsets. [`eq_eval_at_index`] is the scalar
//! equality primitive shared by the kernel and direct callers.

use akita_error::{checked, AkitaError};

use crate::Field;
use jolt_field::solinas::parallel::*;

mod tensor_pair;
pub use tensor_pair::{
    eval_boolean_pair_tensor_families, materialize_eq_tensor_left, EqPairTensorAxis,
    EqPairTensorFamily, EqPairTensorWeights,
};

/// Verifier work cap for one compact-stride equality contraction.
pub const MAX_COMPACT_STRIDE_TERMS: usize = 1 << 28;

/// Coefficient algebra used by [`eval_affine_digit_intervals`].
///
/// The equality and digit factors live in `F`; outer high/low factors may live
/// either in `F` itself or in a small coordinate algebra that is linear over
/// `F`. Keeping these operations abstract lets the trace evaluator preserve
/// its factored extension coordinates without introducing another address
/// kernel.
pub trait AffineWeight<F: Field>: Clone + Send + Sync {
    /// Additive identity carrying the same algebra metadata as `self`.
    fn zero_like(&self) -> Self;

    /// Add `factor * scale` to `self`.
    fn add_scaled(&mut self, factor: &Self, scale: F);

    /// Add `factor` to `self` without a unit scalar multiplication.
    fn add(&mut self, factor: &Self);

    /// Add the embedded scalar `scale` to `self`.
    fn add_scalar(&mut self, scale: F) -> Result<(), AkitaError>;

    /// Multiply two outer factors.
    fn multiply(&self, rhs: &Self) -> Self;
}

/// Random-access outer weights consumed by [`eval_affine_digit_intervals`].
///
/// Implementations may expose an existing dense slice or compute a factored
/// weight only when the contraction reaches its row. The callback keeps dense
/// values borrowed while allowing computed values to remain stack-local.
#[allow(clippy::len_without_is_empty)]
pub trait AffineWeightSource<F: Field, A: AffineWeight<F>>: Sync {
    /// Number of available outer weights.
    fn len(&self) -> usize;

    /// Borrow or compute weight `index` for the duration of `consume`.
    fn with_weight<R>(&self, index: usize, consume: impl FnOnce(&A) -> R) -> Option<R>;
}

impl<F, A, H> AffineWeightSource<F, A> for H
where
    F: Field,
    A: AffineWeight<F>,
    H: AsRef<[A]> + Sync + ?Sized,
{
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn with_weight<R>(&self, index: usize, consume: impl FnOnce(&A) -> R) -> Option<R> {
        self.as_ref().get(index).map(consume)
    }
}

/// Cartesian product of two affine weight slices in outer-major order.
///
/// Weight `i` is `outer[i / inner.len()] * inner[i % inner.len()]`. This is the
/// native representation for block-by-row factors and avoids materializing the
/// full product before a single contraction.
pub struct AffineWeightProduct<'a, A> {
    outer: &'a [A],
    inner: &'a [A],
    len: usize,
}

impl<'a, A> AffineWeightProduct<'a, A> {
    /// Construct a checked outer-major Cartesian product.
    pub fn new(outer: &'a [A], inner: &'a [A]) -> Result<Self, AkitaError> {
        let len = outer
            .len()
            .checked_mul(inner.len())
            .ok_or_else(|| AkitaError::InvalidInput("affine weight product overflow".into()))?;
        Ok(Self { outer, inner, len })
    }
}

impl<F: Field, A: AffineWeight<F>> AffineWeightSource<F, A> for AffineWeightProduct<'_, A> {
    fn len(&self) -> usize {
        self.len
    }

    fn with_weight<R>(&self, index: usize, consume: impl FnOnce(&A) -> R) -> Option<R> {
        if self.inner.is_empty() {
            return None;
        }
        let outer = self.outer.get(index / self.inner.len())?;
        let inner = self.inner.get(index % self.inner.len())?;
        let product = outer.multiply(inner);
        Some(consume(&product))
    }
}

impl<F: Field> AffineWeight<F> for F {
    fn zero_like(&self) -> Self {
        Self::zero()
    }

    fn add_scaled(&mut self, factor: &Self, scale: F) {
        *self += *factor * scale;
    }

    fn add(&mut self, factor: &Self) {
        *self += *factor;
    }

    fn add_scalar(&mut self, scale: F) -> Result<(), AkitaError> {
        *self += scale;
        Ok(())
    }

    fn multiply(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}

/// Evaluate compatible digit-innermost affine intervals with factored outer
/// weights.
///
/// For `Q = max(low_weights.len(), 1)` and `i` in the exact global outer window
/// `[outer_start, outer_start + live_len)`, this computes
///
/// ```text
/// Σ_base base_scale[base] · Σ_i Σ_d high[i / Q] · low[i % Q] · digit[d]
///     · eq(challenges,
///          base + outer_stride · (i - outer_start) + digit_stride · d).
/// ```
/// An empty `base_scales` slice denotes unit scale for every base, avoiding
/// multiplication by one. Otherwise it must contain one scale per base offset.
/// An empty `low_weights` slice denotes the multiplicative identity at `Q = 1`.
/// This structural identity avoids allocating a singleton or multiplying by one.
///
/// `Q` must be a power of two. The implementation splits the equality point at
/// `log2(Q)`, summarizes the low factor into at most `outer_stride + 1` carry
/// states, reuses summaries for base offsets with the same low residue, and
/// seeds their high rows into shared carry buckets. Guarded geometric-prefix
/// and carry-bucketed contractions accelerate eligible layouts and
/// transparently fall back to the general row contraction.
/// Unaligned first and last rows are handled as exact low-factor subwindows, so
/// distributed chunks and a partial final tensor row do not enumerate the
/// Cartesian high-by-low domain. Boolean challenges require no inversion.
///
/// # Errors
///
/// Returns an error for malformed factors, an out-of-range outer window,
/// address overflow, insufficient equality arity, or work above
/// [`MAX_COMPACT_STRIDE_TERMS`]. The work bound is checked before allocating
/// carry summaries.
#[allow(clippy::too_many_arguments)]
pub fn eval_affine_digit_intervals<F, A, H>(
    challenges: &[F],
    base_offsets: &[usize],
    outer_start: usize,
    live_len: usize,
    outer_stride: usize,
    digit_stride: usize,
    digit_weights: &[F],
    high_weights: &H,
    low_weights: &[A],
    base_scales: &[F],
) -> Result<A, AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
    H: AffineWeightSource<F, A> + ?Sized,
{
    let template = high_weights
        .with_weight(0, |weight| weight.zero_like())
        .or_else(|| low_weights.first().map(AffineWeight::zero_like))
        .ok_or_else(|| AkitaError::InvalidInput("affine factors must be non-empty".into()))?;
    if live_len == 0 || base_offsets.is_empty() {
        return Ok(template.zero_like());
    }
    if !base_scales.is_empty() && base_scales.len() != base_offsets.len() {
        return Err(AkitaError::InvalidSize {
            expected: base_offsets.len(),
            actual: base_scales.len(),
        });
    }
    let low_len = low_weights.len().max(1);
    let digit_span = digit_weights
        .len()
        .checked_sub(1)
        .and_then(|count| count.checked_mul(digit_stride))
        .ok_or_else(|| AkitaError::InvalidInput("affine digit span overflow".into()))?;
    if !low_len.is_power_of_two()
        || digit_weights.is_empty()
        || digit_stride == 0
        || outer_stride <= digit_span
    {
        return Err(AkitaError::InvalidInput(
            "affine digit geometry requires power-of-two low length and an outer stride covering every strided digit"
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
    let address_span = outer_stride
        .checked_mul(live_len - 1)
        .and_then(|delta| delta.checked_add(digit_span))
        .ok_or_else(|| AkitaError::InvalidInput("affine address overflow".into()))?;
    for &base_offset in base_offsets {
        let max_address = base_offset
            .checked_add(address_span)
            .ok_or_else(|| AkitaError::InvalidInput("affine address overflow".into()))?;
        if challenges.len() < usize::BITS as usize && max_address >= (1usize << challenges.len()) {
            return Err(AkitaError::InvalidSize {
                expected: challenges.len() + 1,
                actual: challenges.len(),
            });
        }
    }

    if let Some(value) = try_eval_bit_aligned_digit_intervals(
        challenges,
        base_offsets,
        outer_start,
        live_len,
        outer_stride,
        digit_stride,
        digit_weights,
        high_weights,
        low_weights,
        base_scales,
    )? {
        return Ok(value);
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
    let work_per_family = digit_count
        .checked_mul(summarized_low)
        .and_then(|low_work| {
            row_count
                .checked_mul(carry_count)
                .and_then(|high_work| low_work.checked_add(high_work))
        })
        .ok_or_else(|| AkitaError::InvalidInput("affine work overflow".into()))?;
    let work = work_per_family
        .checked_mul(base_offsets.len())
        .ok_or_else(|| AkitaError::InvalidInput("affine family work overflow".into()))?;
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
            &mut out,
            low_challenges,
            high_challenges,
            base_offsets,
            outer_start,
            outer_stride,
            digit_stride,
            digit_weights,
            high_weights,
            low_weights,
            base_scales,
            outer_start / low_len,
            outer_start % low_len,
            outer_start % low_len + prefix_span,
            1,
        )?;
    }
    if full_rows != 0 {
        accumulate_affine_rows(
            &mut out,
            low_challenges,
            high_challenges,
            base_offsets,
            outer_start,
            outer_stride,
            digit_stride,
            digit_weights,
            high_weights,
            low_weights,
            base_scales,
            full_start / low_len,
            0,
            low_len,
            full_rows,
        )?;
    }
    if suffix_span != 0 {
        accumulate_affine_rows(
            &mut out,
            low_challenges,
            high_challenges,
            base_offsets,
            outer_start,
            outer_stride,
            digit_stride,
            digit_weights,
            high_weights,
            low_weights,
            base_scales,
            cursor / low_len,
            0,
            suffix_span,
            1,
        )?;
    }
    Ok(out)
}

/// Evaluate the carry-aware bit-aligned specialization of an affine interval.
///
/// When the physical outer stride is a power of two and there is no explicit
/// low factor, the address map separates exactly as
///
/// ```text
/// low  = (base_low + digit_stride * digit) mod stride
/// high = base_high + row
///      + floor((base_low + digit_stride * digit) / stride).
/// ```
///
/// Since the validated digit span is shorter than one outer stride, arbitrary
/// base residues produce at most two carry classes. Each class factors into one
/// digit contraction on the low point and one consecutive outer contraction on
/// the high point. This avoids constructing and combining a full carry matrix.
/// All other layouts return `None` and use the general affine evaluator.
#[allow(clippy::too_many_arguments)]
fn try_eval_bit_aligned_digit_intervals<F, A, H>(
    challenges: &[F],
    base_offsets: &[usize],
    outer_start: usize,
    live_len: usize,
    outer_stride: usize,
    digit_stride: usize,
    digit_weights: &[F],
    high_weights: &H,
    low_weights: &[A],
    base_scales: &[F],
) -> Result<Option<A>, AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
    H: AffineWeightSource<F, A> + ?Sized,
{
    if !low_weights.is_empty() || !outer_stride.is_power_of_two() {
        return Ok(None);
    }
    let low_bits = outer_stride.trailing_zeros() as usize;
    if low_bits > challenges.len() {
        return Ok(None);
    }
    let low_mask = outer_stride - 1;
    let work = base_offsets
        .len()
        .checked_mul(
            live_len
                .checked_mul(2)
                .and_then(|rows| rows.checked_add(digit_weights.len()))
                .ok_or_else(|| AkitaError::InvalidInput("affine aligned work overflow".into()))?,
        )
        .ok_or_else(|| AkitaError::InvalidInput("affine aligned work overflow".into()))?;
    if work > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: work,
        });
    }

    let (low_challenges, high_challenges) = challenges.split_at(low_bits);
    let high_window = OffsetEqWindow::new(high_challenges)?;
    let template = high_weights
        .with_weight(outer_start, |weight| weight.zero_like())
        .ok_or_else(|| AkitaError::InvalidInput("affine high factor out of range".into()))?;
    let outer_end = outer_start
        .checked_add(live_len)
        .ok_or_else(|| AkitaError::InvalidInput("affine outer window overflow".into()))?;
    let mut out = template.zero_like();
    for (base_index, &base_offset) in base_offsets.iter().enumerate() {
        let base_low = base_offset & low_mask;
        let mut digit_evaluations = [F::zero(), F::zero()];
        for (digit, &digit_weight) in digit_weights.iter().enumerate() {
            let shifted = base_low
                .checked_add(digit.checked_mul(digit_stride).ok_or_else(|| {
                    AkitaError::InvalidInput("affine digit offset overflow".into())
                })?)
                .ok_or_else(|| AkitaError::InvalidInput("affine digit offset overflow".into()))?;
            let carry = shifted / outer_stride;
            let low_index = shifted & low_mask;
            let evaluation = digit_evaluations.get_mut(carry).ok_or_else(|| {
                AkitaError::InvalidInput("affine aligned digit carry exceeds one".into())
            })?;
            *evaluation += digit_weight * eq_eval_at_index(low_challenges, low_index);
        }
        let base_high = base_offset >> low_bits;
        for (carry, mut digit_evaluation) in digit_evaluations.into_iter().enumerate() {
            if let Some(&scale) = base_scales.get(base_index) {
                digit_evaluation *= scale;
            }
            if digit_evaluation.is_zero() {
                continue;
            }
            let mut outer_evaluation = template.zero_like();
            for outer in outer_start..outer_end {
                let high_index = base_high
                    .checked_add(carry)
                    .and_then(|base| base.checked_add(outer - outer_start))
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("affine high address overflow".into())
                    })?;
                let eq_high = high_window.eval(high_index);
                if !eq_high.is_zero() {
                    high_weights
                        .with_weight(outer, |weight| outer_evaluation.add_scaled(weight, eq_high))
                        .ok_or_else(|| {
                            AkitaError::InvalidInput("affine high factor out of range".into())
                        })?;
                }
            }
            out.add_scaled(&outer_evaluation, digit_evaluation);
        }
    }
    Ok(Some(out))
}

#[derive(Clone, Copy)]
struct AffineAddress<F> {
    first: usize,
    scale: Option<F>,
}

#[allow(clippy::too_many_arguments)]
fn accumulate_affine_rows<F, A, H>(
    out: &mut A,
    low_challenges: &[F],
    high_challenges: &[F],
    base_offsets: &[usize],
    outer_start: usize,
    outer_stride: usize,
    digit_stride: usize,
    digit_weights: &[F],
    high_weights: &H,
    low_weights: &[A],
    base_scales: &[F],
    first_high: usize,
    low_start: usize,
    low_end: usize,
    rows: usize,
) -> Result<(), AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
    H: AffineWeightSource<F, A> + ?Sized,
{
    let low_len = low_weights.len().max(1);
    let row_outer = first_high
        .checked_mul(low_len)
        .and_then(|base| base.checked_add(low_start))
        .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?;
    let local_outer = row_outer
        .checked_sub(outer_start)
        .ok_or_else(|| AkitaError::InvalidInput("affine row precedes outer window".into()))?;
    let address_delta = outer_stride
        .checked_mul(local_outer)
        .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?;
    let template = high_weights
        .with_weight(first_high, |weight| weight.zero_like())
        .ok_or_else(|| AkitaError::InvalidInput("affine high factor out of range".into()))?;
    // Precompute the low equality table once and share it across every
    // (low, digit) term instead of recomputing `eq(low_challenges, ·)` from
    // scratch per term. `low_len == 2^low_bits` is the affine low factor width
    // (a fold count), which is bounded by the interval work check above, but we
    // still cap the materialization to keep the allocation bounded and fall
    // back to the scalar primitive for pathologically wide low blocks.
    let eq_low_table: Option<Vec<F>> = if low_weights.is_empty() {
        None
    } else if low_challenges.len() <= OFFSET_EQ_LOW_BITS_CAP {
        Some(crate::eq_poly::EqPolynomial::evals(low_challenges)?)
    } else {
        None
    };
    let mut high_equality = AmortizedOffsetEq::new(high_challenges);
    let mut accumulate_group = |address_low: usize,
                                addresses: &[AffineAddress<F>]|
     -> Result<(), AkitaError> {
        // Size the carry support from the addresses this low-residue group can
        // actually reach. `outer_stride + 1` is a valid global bound, but it
        // can be much too wide for a short low interval. In particular, the
        // identity-low coefficient-packing lane has one low position, so its
        // support is determined only by the digit span. Keeping the trailing
        // zero summaries would both waste the direct row/carry contraction and
        // make the bucketed kernel appear more expensive than its fallback.
        let low_steps = low_end
            .checked_sub(low_start)
            .and_then(|span| span.checked_sub(1))
            .ok_or_else(|| AkitaError::InvalidInput("affine low span is empty".into()))?;
        let digit_steps = digit_weights
            .len()
            .checked_sub(1)
            .ok_or_else(|| AkitaError::InvalidInput("affine digit span is empty".into()))?;
        let max_carry_numerator = address_low
            .checked_add(
                outer_stride
                    .checked_mul(low_steps)
                    .ok_or_else(|| AkitaError::InvalidInput("affine carry span overflow".into()))?,
            )
            .and_then(|value| {
                digit_stride
                    .checked_mul(digit_steps)
                    .and_then(|digit_span| value.checked_add(digit_span))
            })
            .ok_or_else(|| AkitaError::InvalidInput("affine carry span overflow".into()))?;
        let carry_count = max_carry_numerator
            .checked_div(low_len)
            .and_then(|carry| carry.checked_add(1))
            .ok_or_else(|| AkitaError::InvalidInput("affine carry count overflow".into()))?;
        let summaries = build_affine_low_summaries(
            &template,
            low_challenges,
            eq_low_table.as_deref(),
            address_low,
            outer_stride,
            digit_stride,
            low_len,
            low_start,
            low_end,
            digit_weights,
            low_weights,
            carry_count,
        )?;

        // Contract every compatible family into the same high buckets before
        // combining with the shared low summaries.
        if accumulate_high_rows_bucketed(
            out,
            high_challenges,
            addresses,
            outer_stride,
            low_challenges.len(),
            high_weights,
            first_high,
            rows,
            &summaries,
        )? {
            return Ok(());
        }

        high_equality.prepare_for_product(addresses.len(), rows, summaries.len())?;

        for &AffineAddress {
            first: first_address,
            scale: base_scale,
        } in addresses
        {
            for row in 0..rows {
                let high_index = first_high
                    .checked_add(row)
                    .ok_or_else(|| AkitaError::InvalidInput("affine high index overflow".into()))?;
                let row_address = first_address
                    .checked_add(
                        outer_stride
                            .checked_mul(low_len)
                            .and_then(|stride| stride.checked_mul(row))
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("affine high address overflow".into())
                            })?,
                    )
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("affine high address overflow".into())
                    })?;
                let address_high = row_address >> low_challenges.len();
                high_weights
                    .with_weight(high_index, |high_factor| {
                        for (carry, summary) in summaries.iter().enumerate() {
                            let eq_high = high_equality.eval(
                                address_high.checked_add(carry).ok_or_else(|| {
                                    AkitaError::InvalidInput("affine high address overflow".into())
                                })?,
                            );
                            if !eq_high.is_zero() {
                                let scale =
                                    base_scale.map_or(eq_high, |base_scale| eq_high * base_scale);
                                out.add_scaled(&high_factor.multiply(summary), scale);
                            }
                        }
                        Ok::<_, AkitaError>(())
                    })
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("affine high factor out of range".into())
                    })??;
            }
        }
        Ok(())
    };

    if base_offsets.len() == 1 {
        let first_address = base_offsets[0]
            .checked_add(address_delta)
            .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?;
        let addresses = [AffineAddress {
            first: first_address,
            scale: base_scales.first().copied(),
        }];
        return accumulate_group(first_address & (low_len - 1), &addresses);
    }

    let mut addresses = Vec::with_capacity(base_offsets.len());
    for (base_index, &base_offset) in base_offsets.iter().enumerate() {
        let first_address = base_offset
            .checked_add(address_delta)
            .ok_or_else(|| AkitaError::InvalidInput("affine row address overflow".into()))?;
        addresses.push(AffineAddress {
            first: first_address,
            scale: base_scales.get(base_index).copied(),
        });
    }
    addresses.sort_unstable_by_key(|address| address.first & (low_len - 1));
    let mut group_start = 0usize;
    while group_start < addresses.len() {
        let address_low = addresses[group_start].first & (low_len - 1);
        let group_len = addresses[group_start..]
            .partition_point(|address| address.first & (low_len - 1) == address_low);
        let group_end = group_start
            .checked_add(group_len)
            .ok_or_else(|| AkitaError::InvalidInput("affine address group overflow".into()))?;
        accumulate_group(address_low, &addresses[group_start..group_end])?;
        group_start = group_end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_affine_low_summaries<F, A>(
    template: &A,
    low_challenges: &[F],
    eq_low_table: Option<&[F]>,
    address_low: usize,
    outer_stride: usize,
    digit_stride: usize,
    low_len: usize,
    low_start: usize,
    low_end: usize,
    digit_weights: &[F],
    low_weights: &[A],
    carry_count: usize,
) -> Result<Vec<A>, AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
{
    if low_weights.is_empty() {
        if low_len != 1
            || !low_challenges.is_empty()
            || address_low != 0
            || low_start != 0
            || low_end != 1
        {
            return Err(AkitaError::InvalidInput(
                "affine identity low factor requires the unit low domain".into(),
            ));
        }
        let mut summaries = vec![template.zero_like(); carry_count];
        for (digit, &digit_weight) in digit_weights.iter().enumerate() {
            let carry = digit_stride
                .checked_mul(digit)
                .ok_or_else(|| AkitaError::InvalidInput("affine digit offset overflow".into()))?;
            summaries
                .get_mut(carry)
                .ok_or_else(|| AkitaError::InvalidInput("affine carry out of range".into()))?
                .add_scalar(digit_weight)?;
        }
        return Ok(summaries);
    }

    // Geometric digit windows share one prefix scan. Other weights use the
    // dense low summary, still once per distinct low-address residue.
    if digit_stride == 1 {
        if let Some(table) = eq_low_table {
            if let Some(summaries) = build_geometric_low_summaries(
                template,
                table,
                address_low,
                outer_stride,
                low_len,
                low_start,
                low_end,
                digit_weights,
                low_weights,
                carry_count,
            )? {
                return Ok(summaries);
            }
        }
    }
    let mut summaries = vec![template.zero_like(); carry_count];
    for low in low_start..low_end {
        let low_factor = low_weights
            .get(low)
            .ok_or_else(|| AkitaError::InvalidInput("affine low factor out of range".into()))?;
        let low_delta = outer_stride
            .checked_mul(low - low_start)
            .ok_or_else(|| AkitaError::InvalidInput("affine low address overflow".into()))?;
        for (digit, &digit_weight) in digit_weights.iter().enumerate() {
            let digit_offset = digit_stride
                .checked_mul(digit)
                .ok_or_else(|| AkitaError::InvalidInput("affine digit offset overflow".into()))?;
            let shifted = address_low
                .checked_add(low_delta)
                .and_then(|value| value.checked_add(digit_offset))
                .ok_or_else(|| AkitaError::InvalidInput("affine low address overflow".into()))?;
            let carry = shifted / low_len;
            let low_index = shifted & (low_len - 1);
            let eq_low = match eq_low_table {
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
    Ok(summaries)
}

/// Build the low carry summaries via a geometric prefix scan.
///
/// When the digit weights are geometric — `digit_weights[k] == digit_weights[0] * r^k`,
/// which holds for the gadget vector `g^k` used by the E/opening lane — and the
/// digit window fits inside one low block (`digit_count <= low_len`), the inner
/// digit sum for each `low` is a geometric-weighted contiguous window of the low
/// equality table. A single prefix `P[t] = Σ_{u<t} r^u eq_low[u]` then yields
/// each window (and its at-most-one block wrap) in `O(1)`, dropping the summary
/// cost from `O(low_len * digit_count)` to `O(low_len + digit_count)`.
///
/// Returns `None` (caller falls back to the dense loop) when the weights are not
/// geometric, the ratio or leading weight is zero, the window spans more than one
/// block, or the span is too short to amortize the prefix setup. The `Some`
/// result is bit-identical to the dense loop. The single field inversion may
/// use a nonzero gadget or projected-lane digit-sequence ratio; zero and
/// non-geometric sequences fall back. Equality-point coordinates are never
/// inverted.
#[allow(clippy::too_many_arguments)]
fn build_geometric_low_summaries<F, A>(
    template: &A,
    eq_low: &[F],
    address_low: usize,
    outer_stride: usize,
    low_len: usize,
    low_start: usize,
    low_end: usize,
    digit_weights: &[F],
    low_weights: &[A],
    carry_count: usize,
) -> Result<Option<Vec<A>>, AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
{
    let delta = digit_weights.len();
    // Require the window to fit one block, the table to match, and the span to be
    // long enough that the O(low_len) prefix setup beats the dense O(span*delta).
    if delta == 0
        || delta > low_len
        || eq_low.len() != low_len
        || low_end.saturating_sub(low_start).saturating_mul(delta) < low_len
    {
        return Ok(None);
    }
    let d0 = digit_weights[0];
    if d0.is_zero() {
        return Ok(None);
    }
    // Ratio r (= digit[1]/digit[0] for delta >= 2; unused for delta == 1).
    let r = if delta >= 2 {
        match d0.inverse() {
            Some(d0_inv) => digit_weights[1] * d0_inv,
            None => return Ok(None),
        }
    } else {
        F::one()
    };
    if r.is_zero() {
        return Ok(None);
    }
    // Forward powers r^0..r^{low_len}; confirm the weights really are geometric.
    let mut rpow = vec![F::one(); low_len + 1];
    for k in 1..=low_len {
        rpow[k] = rpow[k - 1] * r;
    }
    for (k, &weight) in digit_weights.iter().enumerate() {
        if weight != d0 * rpow[k] {
            return Ok(None);
        }
    }
    let r_inv = match r.inverse() {
        Some(inv) => inv,
        None => return Ok(None),
    };
    // Inverse powers r^{-0}..r^{-(low_len-1)} for the window-start anchor.
    let mut rinvpow = vec![F::one(); low_len];
    for s in 1..low_len {
        rinvpow[s] = rinvpow[s - 1] * r_inv;
    }
    // Prefix P[t] = Σ_{u<t} r^u eq_low[u].
    let mut prefix = vec![F::zero(); low_len + 1];
    for u in 0..low_len {
        prefix[u + 1] = prefix[u] + rpow[u] * eq_low[u];
    }

    let low_mask = low_len - 1;
    let mut summaries = vec![template.zero_like(); carry_count];
    for low in low_start..low_end {
        let low_factor = low_weights
            .get(low)
            .ok_or_else(|| AkitaError::InvalidInput("affine low factor out of range".into()))?;
        let start =
            address_low
                .checked_add(outer_stride.checked_mul(low - low_start).ok_or_else(|| {
                    AkitaError::InvalidInput("affine low address overflow".into())
                })?)
                .ok_or_else(|| AkitaError::InvalidInput("affine low address overflow".into()))?;
        let carry = start / low_len;
        let s = start & low_mask;
        let count1 = delta.min(low_len - s);
        // No-wrap window [s, s+count1): digits 0..count1 stay in block `carry`.
        // Σ_d digit[0] r^d eq_low[s+d] = digit[0] * r^{-s} * (P[s+count1] - P[s]).
        let seg = prefix[s + count1] - prefix[s];
        let val = d0 * rinvpow[s] * seg;
        let summary = summaries
            .get_mut(carry)
            .ok_or_else(|| AkitaError::InvalidInput("affine carry out of range".into()))?;
        summary.add_scaled(low_factor, val);
        if count1 < delta {
            // Wrap window: digits count1..delta land at the start of block carry+1.
            // Σ_{d>=count1} digit[0] r^d eq_low[d-count1] = digit[0] * r^{count1} * P[delta-count1].
            let seg = prefix[delta - count1];
            let val = d0 * rpow[count1] * seg;
            let carry = carry
                .checked_add(1)
                .ok_or_else(|| AkitaError::InvalidInput("affine carry overflow".into()))?;
            let summary = summaries
                .get_mut(carry)
                .ok_or_else(|| AkitaError::InvalidInput("affine carry out of range".into()))?;
            summary.add_scaled(low_factor, val);
        }
    }
    Ok(Some(summaries))
}

/// Minimum full-row count before the bucketed high contraction is worth its
/// table setup. Below this the base row loop is cheaper.
const FAST_HIGH_ROWS_MIN: usize = 8;

/// Minimum row work before compatible affine bases build high buckets in
/// parallel. Smaller contractions are commonly nested under an outer parallel
/// fold, so keeping them in one task avoids scheduler and bucket-allocation
/// overhead.
const PARALLEL_HIGH_ROWS_MIN: usize = 1 << 12;

/// Return the bounded bucket window when bucketing performs less charged work
/// than the direct row/carry contraction.
fn bucketed_high_rows_plan(
    total_rows: usize,
    carry_count: usize,
    high_challenge_count: usize,
) -> Result<Option<usize>, AkitaError> {
    if total_rows < FAST_HIGH_ROWS_MIN || carry_count == 0 {
        return Ok(None);
    }
    let window = carry_count
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidInput("affine bucket window overflow".into()))?;
    if window.trailing_zeros() as usize > OFFSET_EQ_LOW_BITS_CAP {
        return Ok(None);
    }
    let window_bits = window.trailing_zeros() as usize;
    let Some(split_bits) = high_challenge_count.checked_sub(window_bits) else {
        return Ok(None);
    };
    if split_bits > 2 * OFFSET_EQ_HIGH_BITS_CAP {
        return Ok(None);
    }
    let split_low_bits = split_bits / 2;
    let split_high_bits = split_bits - split_low_bits;
    let split_entries = 1usize
        .checked_shl(split_low_bits as u32)
        .and_then(|low| {
            1usize
                .checked_shl(split_high_bits as u32)
                .and_then(|high| low.checked_add(high))
        })
        .ok_or_else(|| AkitaError::InvalidInput("affine split table work overflow".into()))?;
    let fallback_work = total_rows
        .checked_mul(carry_count)
        .ok_or_else(|| AkitaError::InvalidInput("affine fallback work overflow".into()))?;
    let bucket_work = total_rows
        .checked_mul(2)
        .and_then(|rows| {
            carry_count
                .checked_mul(window)
                .and_then(|combine| rows.checked_add(combine))
        })
        .and_then(|work| {
            window
                .checked_mul(3)
                .and_then(|allocation| work.checked_add(allocation))
        })
        .and_then(|work| work.checked_add(split_entries))
        .ok_or_else(|| AkitaError::InvalidInput("affine bucket work overflow".into()))?;
    if bucket_work >= fallback_work {
        return Ok(None);
    }
    if bucket_work > MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: bucket_work,
        });
    }
    Ok(Some(window))
}

/// Contract the high rows against the shared low carry summaries using a
/// bounded high-equality table and carry-bucketing.
///
/// For each compatible family and row `r` the base kernel adds
/// `high[first_high + r] * summaries[carry] * eq(high_challenges, h0 + stride*r + carry)`
/// over every `carry in 0..summaries.len()`, where
/// `h0 = first_address >> low_bits`.
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
fn accumulate_high_rows_bucketed<F, A, H>(
    out: &mut A,
    high_challenges: &[F],
    addresses: &[AffineAddress<F>],
    outer_stride: usize,
    low_bits: usize,
    high_weights: &H,
    first_high: usize,
    rows: usize,
    summaries: &[A],
) -> Result<bool, AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
    H: AffineWeightSource<F, A> + ?Sized,
{
    let carry_count = summaries.len();
    let total_rows = rows
        .checked_mul(addresses.len())
        .ok_or_else(|| AkitaError::InvalidInput("affine high row count overflow".into()))?;
    let Some(window) = bucketed_high_rows_plan(total_rows, carry_count, high_challenges.len())?
    else {
        return Ok(false);
    };
    let template = summaries
        .first()
        .ok_or_else(|| AkitaError::InvalidInput("affine summaries empty".into()))?;

    // Split the high equality point so the carry window fits inside the low part.
    let window_bits = window.trailing_zeros() as usize;
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

    let window_mask = window - 1;
    // Bucket each row by its low-window position, split into the "no carry into
    // the next high block" bucket (`bucket0`) and the "carries" bucket (`bucket1`).
    // Large compatible-base families use one private bucket pair per base and
    // reduce them afterward. Small or single-base contractions stay in one
    // task, which is important when this kernel is already under a parallel
    // outer fold.
    let task_count = if total_rows >= PARALLEL_HIGH_ROWS_MIN {
        addresses.len()
    } else {
        1
    };
    let addresses_per_task = addresses.len().div_ceil(task_count);
    let (bucket0, bucket1) = cfg_fold_reduce!(
        0..task_count,
        || Ok((
            vec![template.zero_like(); window],
            vec![template.zero_like(); window]
        )),
        |acc: Result<(Vec<A>, Vec<A>), AkitaError>, task| {
            let (mut bucket0, mut bucket1) = acc?;
            let start = task
                .checked_mul(addresses_per_task)
                .ok_or_else(|| AkitaError::InvalidInput("affine task range overflow".into()))?;
            let end = start
                .checked_add(addresses_per_task)
                .map(|end| end.min(addresses.len()))
                .ok_or_else(|| AkitaError::InvalidInput("affine task range overflow".into()))?;
            let addresses = addresses
                .get(start..end)
                .ok_or_else(|| AkitaError::InvalidInput("affine task range invalid".into()))?;
            for &AffineAddress {
                first: first_address,
                scale: base_scale,
            } in addresses
            {
                let h0 = first_address >> low_bits;
                for row in 0..rows {
                    let high_index = first_high.checked_add(row).ok_or_else(|| {
                        AkitaError::InvalidInput("affine high index overflow".into())
                    })?;
                    let address_high = h0
                        .checked_add(outer_stride.checked_mul(row).ok_or_else(|| {
                            AkitaError::InvalidInput("affine high address overflow".into())
                        })?)
                        .ok_or_else(|| {
                            AkitaError::InvalidInput("affine high address overflow".into())
                        })?;
                    let low_pos = address_high & window_mask;
                    let block = address_high >> window_bits;
                    let eq_block0 = eval_high(block)?;
                    let eq_block1 = eval_high(block.checked_add(1).ok_or_else(|| {
                        AkitaError::InvalidInput("affine high block overflow".into())
                    })?)?;
                    let eq_block0 =
                        base_scale.map_or(eq_block0, |base_scale| eq_block0 * base_scale);
                    let eq_block1 =
                        base_scale.map_or(eq_block1, |base_scale| eq_block1 * base_scale);
                    high_weights
                        .with_weight(high_index, |high_factor| {
                            bucket0[low_pos].add_scaled(high_factor, eq_block0);
                            bucket1[low_pos].add_scaled(high_factor, eq_block1);
                        })
                        .ok_or_else(|| {
                            AkitaError::InvalidInput("affine high factor out of range".into())
                        })?;
                }
            }
            Ok((bucket0, bucket1))
        },
        |lhs: Result<(Vec<A>, Vec<A>), AkitaError>, rhs: Result<(Vec<A>, Vec<A>), AkitaError>| {
            let (mut lhs0, mut lhs1) = lhs?;
            let (rhs0, rhs1) = rhs?;
            for (slot, value) in lhs0.iter_mut().zip(rhs0) {
                slot.add(&value);
            }
            for (slot, value) in lhs1.iter_mut().zip(rhs1) {
                slot.add(&value);
            }
            Ok((lhs0, lhs1))
        }
    )?;

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
        out.add(&phi.multiply(summary));
    }
    Ok(true)
}

/// Hard cap on the number of low bits materialized by [`OffsetEqWindow`].
///
/// A 16-bit low table holds at most `2^16 = 65_536` field elements. When the
/// high side can also be materialized, construction balances the split to
/// minimize the sum of both table sizes.
pub const OFFSET_EQ_LOW_BITS_CAP: usize = 16;

/// Hard cap on the number of high bits materialized by [`OffsetEqWindow`].
///
/// When the high remainder has at most this many bits, the high equality table
/// `eq_high[j] = eq(high_challenges, j)` is materialized so that each `eval`
/// costs two table lookups and a single multiply. The cap bounds the high table
/// at `2^16` field elements; wider high remainders fall back to on-demand
/// `O(high_bits)` evaluation.
pub const OFFSET_EQ_HIGH_BITS_CAP: usize = 16;

struct AmortizedOffsetEq<'a, F: Field> {
    challenges: &'a [F],
    window: Option<OffsetEqWindow<F>>,
}

impl<'a, F: Field> AmortizedOffsetEq<'a, F> {
    fn new(challenges: &'a [F]) -> Self {
        Self {
            challenges,
            window: None,
        }
    }

    fn prepare_for_product(
        &mut self,
        families: usize,
        rows: usize,
        carries: usize,
    ) -> Result<(), AkitaError> {
        if self.window.is_none() {
            let evaluations = checked::product([families, rows, carries])
                .ok_or_else(|| AkitaError::InvalidInput("affine high work overflow".into()))?;
            self.window = OffsetEqWindow::new_if_amortized(self.challenges, evaluations)?;
        }
        Ok(())
    }

    #[inline]
    fn eval(&self, index: usize) -> F {
        self.window.as_ref().map_or_else(
            || eq_eval_at_index(self.challenges, index),
            |window| window.eval(index),
        )
    }
}

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
pub struct OffsetEqWindow<F: Field> {
    low_bits: usize,
    low_mask: usize,
    eq_low: Vec<F>,
    eq_high: Option<Vec<F>>,
    high_challenges: Vec<F>,
}

impl<F: Field> OffsetEqWindow<F> {
    fn split_low_bits(challenge_count: usize, low_bits_cap: usize) -> usize {
        let low_cap = low_bits_cap.min(OFFSET_EQ_LOW_BITS_CAP);
        if challenge_count <= low_cap + OFFSET_EQ_HIGH_BITS_CAP {
            challenge_count
                .div_ceil(2)
                .max(challenge_count.saturating_sub(OFFSET_EQ_HIGH_BITS_CAP))
                .min(low_cap)
        } else {
            challenge_count.min(low_cap)
        }
    }

    /// Construct a window only when the requested evaluations match or exceed
    /// the number of materialized table entries.
    fn new_if_amortized(
        challenges: &[F],
        evaluation_count: usize,
    ) -> Result<Option<Self>, AkitaError> {
        if challenges.is_empty() {
            return Ok(None);
        }
        let low_bits = Self::split_low_bits(challenges.len(), OFFSET_EQ_LOW_BITS_CAP);
        let high_bits = challenges.len() - low_bits;
        let low_len = checked::pow2(low_bits)
            .ok_or_else(|| AkitaError::InvalidInput("eq table dimension overflow".into()))?;
        let high_len = if high_bits <= OFFSET_EQ_HIGH_BITS_CAP {
            checked::pow2(high_bits)
                .ok_or_else(|| AkitaError::InvalidInput("eq table dimension overflow".into()))?
        } else {
            0
        };
        let materialized_values = checked::sum([low_len, high_len])
            .ok_or_else(|| AkitaError::InvalidInput("eq table dimension overflow".into()))?;
        if evaluation_count < materialized_values {
            return Ok(None);
        }
        Self::new(challenges).map(Some)
    }

    /// Build a window over `challenges` using the default low-bit cap.
    ///
    /// # Errors
    ///
    /// Returns an error if the low equality table cannot be constructed.
    pub fn new(challenges: &[F]) -> Result<Self, AkitaError> {
        Self::with_low_bits(challenges, OFFSET_EQ_LOW_BITS_CAP)
    }

    /// Build a window over `challenges` with at most `min(cap, CAP)` low bits.
    /// When both sides fit their caps, the split is balanced to minimize total
    /// materialization. Wider high remainders stay on demand.
    ///
    /// # Errors
    ///
    /// Returns an error if the low equality table cannot be constructed.
    pub fn with_low_bits(challenges: &[F], low_bits_cap: usize) -> Result<Self, AkitaError> {
        let low_bits = Self::split_low_bits(challenges.len(), low_bits_cap);
        let eq_low = crate::eq_poly::EqPolynomial::evals(&challenges[..low_bits])?;
        let low_mask = if low_bits == 0 {
            0
        } else {
            (1usize << low_bits) - 1
        };
        let high_challenges = challenges[low_bits..].to_vec();
        // Materialize the high table too when it stays within the bounded cap.
        // This makes every `eval` a pair of lookups instead of recomputing an
        // `O(high_bits)` equality product per address, which dominated the
        // verifier setup-weight builders.
        let eq_high = if high_challenges.len() <= OFFSET_EQ_HIGH_BITS_CAP {
            Some(crate::eq_poly::EqPolynomial::evals(&high_challenges)?)
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

    /// Number of Boolean variables in the represented equality domain.
    #[must_use]
    pub fn variable_count(&self) -> usize {
        self.low_bits + self.high_challenges.len()
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
            None => eq_eval_at_index(&self.high_challenges, high),
        };
        eq_low * eq_high
    }

    /// Fill a contiguous physical-index interval.
    ///
    /// The interval is checked once; individual entries then reuse the same
    /// bounded equality tables without semantic address reconstruction.
    pub fn fill_interval(&self, start: usize, output: &mut [F]) -> Result<(), AkitaError> {
        start
            .checked_add(output.len())
            .ok_or_else(|| AkitaError::InvalidInput("equality interval overflow".into()))?;
        const PARALLEL_THRESHOLD: usize = 1 << 14;
        if let Some(eq_high) = &self.eq_high {
            if output.len() >= PARALLEL_THRESHOLD {
                cfg_chunks_mut!(output, PARALLEL_THRESHOLD)
                    .enumerate()
                    .try_for_each(|(chunk_index, chunk)| {
                        let chunk_start = chunk_index
                            .checked_mul(PARALLEL_THRESHOLD)
                            .and_then(|offset| start.checked_add(offset))
                            .ok_or_else(|| {
                                AkitaError::InvalidInput("equality interval overflow".into())
                            })?;
                        self.fill_bounded_high_interval(chunk_start, chunk, eq_high)
                    })?;
            } else {
                self.fill_bounded_high_interval(start, output, eq_high)?;
            }
            return Ok(());
        }
        if output.len() >= PARALLEL_THRESHOLD {
            cfg_iter_mut!(output)
                .enumerate()
                .for_each(|(offset, value)| *value = self.eval(start + offset));
        } else {
            output
                .iter_mut()
                .enumerate()
                .for_each(|(offset, value)| *value = self.eval(start + offset));
        }
        Ok(())
    }

    fn fill_bounded_high_interval(
        &self,
        mut start: usize,
        mut output: &mut [F],
        eq_high: &[F],
    ) -> Result<(), AkitaError> {
        while !output.is_empty() {
            let low = start & self.low_mask;
            let available_low = self
                .eq_low
                .len()
                .checked_sub(low)
                .ok_or(AkitaError::InvalidProof)?;
            let take = available_low.min(output.len());
            let low_end = low.checked_add(take).ok_or(AkitaError::InvalidProof)?;
            let low_values = self
                .eq_low
                .get(low..low_end)
                .ok_or(AkitaError::InvalidProof)?;
            let (destination, tail) = output
                .split_at_mut_checked(take)
                .ok_or(AkitaError::InvalidProof)?;
            let high = start >> self.low_bits;
            let Some(scale) = eq_high.get(high).copied() else {
                destination.fill(F::zero());
                tail.fill(F::zero());
                return Ok(());
            };
            if scale.is_zero() {
                destination.fill(F::zero());
            } else if scale == F::one() {
                destination.copy_from_slice(low_values);
            } else {
                destination
                    .iter_mut()
                    .zip(low_values)
                    .for_each(|(value, &low_value)| *value = low_value * scale);
            }
            start = start
                .checked_add(take)
                .ok_or_else(|| AkitaError::InvalidInput("equality interval overflow".into()))?;
            output = tail;
        }
        Ok(())
    }
}

/// Evaluate `eq(r, index)` for a single hypercube index in little-endian order.
pub fn eq_eval_at_index<F: Field>(x_challenges: &[F], index: usize) -> F {
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

#[cfg(test)]
mod amortization_tests {
    use super::*;
    use jolt_field::{Fp64, Ring};

    type F = Fp64<4294967197>;

    #[test]
    fn materialization_requires_table_reuse() {
        let challenges = vec![F::from_u64(3); 8];
        assert!(OffsetEqWindow::new_if_amortized(&challenges, 31)
            .unwrap()
            .is_none());

        let window = OffsetEqWindow::new_if_amortized(&challenges, 32)
            .unwrap()
            .unwrap();
        assert_eq!(window.eq_low.len(), 16);
        assert_eq!(window.eq_high.as_ref().unwrap().len(), 16);
        assert!(OffsetEqWindow::<F>::new_if_amortized(&[], usize::MAX)
            .unwrap()
            .is_none());
    }
}
