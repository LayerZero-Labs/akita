//! Shared exact contraction for adjacent digit windows.
//!
//! One prepared row covers `stride = digits.len() * max(low.len(), 1)`
//! consecutive addresses, with lane `l` scaled by `low[l]`. Preparation folds
//! the digits and low factor into no-carry and carry tables over the smallest
//! power-of-two block holding one row, so each later row costs one table lookup
//! and at most one new high equality evaluation. Callers share one prepared
//! window across base offsets, claims and witness units on the same point.

use akita_error::{checked, AkitaError};

use super::{AffineWeight, AffineWeightSource, OffsetEqWindow, MAX_COMPACT_STRIDE_TERMS};
use crate::{eq_poly::EqPolynomial, Field};

/// Widest prepared row; wider rows keep the general affine contraction.
pub const MAX_PREPARED_STRIDE: usize = 512;

/// Interleaved geometric lane periods recognized by the prepared recurrence.
const GEOMETRIC_PERIODS: [usize; 5] = [1, 2, 4, 8, 16];

enum Tables<F, A> {
    /// Unit low factor: digit-only tables in the scalar field.
    Unit { no_carry: Vec<F>, carry: Vec<F> },
    /// Low factor folded into each residue's table entry.
    Folded { no_carry: Vec<A>, carry: Vec<A> },
}

/// Prepared adjacent-digit window shared by every row on one equality point.
///
/// For `Q = max(low.len(), 1)`, [`Self::accumulate`] adds
///
/// ```text
/// Σ_{row in [first, first + rows)} high[row]
///     · Σ_{lane < Q} Σ_d low[lane] · digit[d]
///         · eq(challenges, base + (row - first) · Q · D + lane · D + d)
/// ```
///
/// with `D = digits.len()`. This is
/// [`eval_affine_digit_intervals`](super::eval_affine_digit_intervals) with
/// `outer_stride = D`, `digit_stride = 1` and a row-aligned outer window, which
/// dispatches here when preparation is cheaper than its general contraction.
pub struct PreparedDigits<F: Field, A> {
    stride: usize,
    bits: u32,
    point_bits: usize,
    low_mask: usize,
    tables: Tables<F, A>,
    high: OffsetEqWindow<F>,
}

fn invalid() -> AkitaError {
    AkitaError::InvalidInput("invalid prepared digit contraction".into())
}

impl<F: Field, A: AffineWeight<F>> PreparedDigits<F, A> {
    /// Conservative preparation work, before any row visits.
    ///
    /// Counts the full residue/digit contraction and the materialized high
    /// equality window, even when geometric digits make preparation cheaper.
    /// Returns `None` for an unrepresentable shape or insufficient point arity.
    /// This estimate does not replace the eligibility checks in [`Self::new`].
    #[must_use]
    pub fn preparation_work(
        challenge_count: usize,
        digit_count: usize,
        low_count: usize,
    ) -> Option<usize> {
        let stride = checked::product([digit_count, low_count.max(1)])?;
        let bits = checked::ceil_log2(stride)?;
        let width = checked::pow2(bits)?;
        let high_bits = challenge_count.checked_sub(bits)?;
        checked::sum([
            checked::product([width, stride])?,
            OffsetEqWindow::<F>::materialized_len(high_bits).ok()?,
        ])
    }

    /// Prepare rows of `digits` and `low` lanes on `challenges`.
    ///
    /// Returns `None` when a row is wider than the prepared bound, the point is
    /// shorter than one row block, or the point has at least `usize::BITS`
    /// coordinates; callers then use the general affine contraction.
    ///
    /// # Errors
    ///
    /// Returns an error for empty digits, a non-power-of-two low factor, or a
    /// row width overflow.
    pub fn new(challenges: &[F], digits: &[F], low: &[A]) -> Result<Option<Self>, AkitaError> {
        if digits.is_empty() || (!low.is_empty() && !low.len().is_power_of_two()) {
            return Err(invalid());
        }
        let lanes = low.len().max(1);
        let stride = digits.len().checked_mul(lanes).ok_or_else(invalid)?;
        if stride > MAX_PREPARED_STRIDE || challenges.len() >= usize::BITS as usize {
            return Ok(None);
        }
        let width = stride.checked_next_power_of_two().ok_or_else(invalid)?;
        let bits = width.trailing_zeros();
        if bits as usize > challenges.len() {
            return Ok(None);
        }
        let equality = EqPolynomial::evals(&challenges[..bits as usize])?;
        let mut tables = match low.first() {
            None => Tables::Unit {
                no_carry: vec![F::zero(); width],
                carry: vec![F::zero(); width],
            },
            Some(template) => Tables::Folded {
                no_carry: vec![template.zero_like(); width],
                carry: vec![template.zero_like(); width],
            },
        };
        // Lane sums stay in the scalar field; the low factor is folded once per
        // residue and lane.
        let mut fold =
            |residue: usize, lane: usize, no_carry_sum: F, carry_sum: F| match &mut tables {
                Tables::Unit { no_carry, carry } => {
                    no_carry[residue] += no_carry_sum;
                    carry[residue] += carry_sum;
                }
                Tables::Folded { no_carry, carry } => {
                    no_carry[residue].add_scaled(&low[lane], no_carry_sum);
                    carry[residue].add_scaled(&low[lane], carry_sum);
                }
            };
        let period = GEOMETRIC_PERIODS.into_iter().find(|&p| {
            if digits[0] != F::one() || !digits.len().is_multiple_of(p) {
                return false;
            }
            let beta = digits.get(p).copied().unwrap_or_else(F::zero);
            (p..digits.len()).all(|i| digits[i - p] * beta == digits[i])
        });
        if let Some(period) = period {
            // Each interleaved lane is geometric. H(t) contracts one lane:
            // H(t)=eq_low(t)+beta H(t+P)-beta^D eq_low(t+D*P).
            // Point coordinates are never inverted, including Boolean ones.
            let count = digits.len() / period;
            let beta = digits.get(period).copied().unwrap_or_else(F::zero);
            let prefix = (count - 1) * period;
            let mut h = vec![F::zero(); width + prefix];
            let tail = digits[prefix] * beta;
            for i in (0..h.len()).rev() {
                let mut value = if i + period < h.len() {
                    beta * h[i + period]
                } else {
                    F::zero()
                };
                if i >= prefix {
                    value += equality[i - prefix];
                }
                if i + period < width {
                    value -= tail * equality[i + period];
                }
                h[i] = value;
            }
            let at = |offset: isize| -> F {
                let i = offset + prefix as isize;
                if i < 0 {
                    F::zero()
                } else {
                    h.get(i as usize).copied().unwrap_or_else(F::zero)
                }
            };
            for residue in 0..width {
                for lane in 0..lanes {
                    let mut no_carry_sum = F::zero();
                    let mut carry_sum = F::zero();
                    for (j, &digit_lane) in digits[..period].iter().enumerate() {
                        let offset = (residue + lane * digits.len() + j) as isize;
                        no_carry_sum += digit_lane * at(offset);
                        carry_sum += digit_lane * at(offset - width as isize);
                    }
                    fold(residue, lane, no_carry_sum, carry_sum);
                }
            }
        } else {
            for residue in 0..width {
                let split = stride.min(width - residue);
                for lane in 0..lanes {
                    let mut no_carry_sum = F::zero();
                    let mut carry_sum = F::zero();
                    for (d, &digit) in digits.iter().enumerate() {
                        let position = lane * digits.len() + d;
                        if position < split {
                            no_carry_sum += digit * equality[residue + position];
                        } else {
                            carry_sum += digit * equality[residue + position - width];
                        }
                    }
                    fold(residue, lane, no_carry_sum, carry_sum);
                }
            }
        }
        Ok(Some(Self {
            stride,
            bits,
            point_bits: challenges.len(),
            low_mask: width - 1,
            tables,
            high: OffsetEqWindow::new(&challenges[bits as usize..])?,
        }))
    }

    /// Add the prepared rows `[first, first + rows)` of `weights`, starting at
    /// address `base`, to `out`.
    ///
    /// # Errors
    ///
    /// Returns an error when the row window exceeds `weights`, the last
    /// address leaves the equality domain, or `rows` exceeds
    /// [`MAX_COMPACT_STRIDE_TERMS`]. Prepared work is bounded independently of
    /// the general evaluator's carry-state estimate: preparation uses at most
    /// `MAX_PREPARED_STRIDE²` residue/digit terms and two 16-bit equality
    /// tables, followed by one bounded equality update per row.
    pub fn accumulate<H: AffineWeightSource<F, A> + ?Sized>(
        &self,
        out: &mut A,
        base: usize,
        first: usize,
        rows: usize,
        weights: &H,
    ) -> Result<(), AkitaError> {
        if rows > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: rows,
            });
        }
        let end = first
            .checked_add(rows)
            .filter(|&end| end <= weights.len())
            .ok_or_else(invalid)?;
        if rows == 0 {
            return Ok(());
        }
        let span = rows
            .checked_mul(self.stride)
            .and_then(|span| span.checked_sub(1))
            .ok_or_else(invalid)?;
        let maximum = base.checked_add(span).ok_or_else(invalid)?;
        if maximum >= (1usize << self.point_bits) {
            return Err(invalid());
        }
        // Contract repeated low residues after the row walk. Linearity moves
        // each low-window multiplication outside the long row loop.
        let width = self.low_mask + 1;
        if rows >= 4 * width {
            let template = weights
                .with_weight(first, AffineWeight::zero_like)
                .ok_or_else(invalid)?;
            let mut left_buckets = vec![template.zero_like(); width];
            let mut right_buckets = vec![template.zero_like(); width];
            self.walk(base, first, end, |row, residue, left, right| {
                weights.with_weight(row, |weight| {
                    left_buckets[residue].add_scaled(weight, left);
                    right_buckets[residue].add_scaled(weight, right);
                })
            })?;
            match &self.tables {
                Tables::Unit { no_carry, carry } => {
                    for i in 0..width {
                        out.add_scaled(&left_buckets[i], no_carry[i]);
                        out.add_scaled(&right_buckets[i], carry[i]);
                    }
                }
                Tables::Folded { no_carry, carry } => {
                    for i in 0..width {
                        out.add(&left_buckets[i].multiply(&no_carry[i]));
                        out.add(&right_buckets[i].multiply(&carry[i]));
                    }
                }
            }
            return Ok(());
        }
        match &self.tables {
            Tables::Unit { no_carry, carry } => {
                self.walk(base, first, end, |row, residue, left, right| {
                    let kernel = no_carry[residue] * left + carry[residue] * right;
                    weights.with_weight(row, |weight| out.add_scaled(weight, kernel))
                })
            }
            Tables::Folded { no_carry, carry } => {
                self.walk(base, first, end, |row, residue, left, right| {
                    let mut kernel = no_carry[residue].zero_like();
                    kernel.add_scaled(&no_carry[residue], left);
                    kernel.add_scaled(&carry[residue], right);
                    weights.with_weight(row, |weight| out.add(&weight.multiply(&kernel)))
                })
            }
        }
    }

    /// Visit rows `[first, end)` with their low residue and the high equality
    /// values of the residue's block and the next one. The checked maximum
    /// address bounds every increment and the final window.
    fn walk(
        &self,
        base: usize,
        first: usize,
        end: usize,
        mut visit: impl FnMut(usize, usize, F, F) -> Option<()>,
    ) -> Result<(), AkitaError> {
        let mut address = base;
        let mut cached_block = address >> self.bits;
        let mut left = self.high.eval(cached_block);
        let mut right = self.high.eval(cached_block + 1);
        for row in first..end {
            let block = address >> self.bits;
            if block != cached_block {
                // Construction makes stride no greater than the block width,
                // so a changed block is exactly the next cached block.
                cached_block = block;
                left = right;
                right = self.high.eval(block + 1);
            }
            visit(row, address & self.low_mask, left, right).ok_or_else(invalid)?;
            if row + 1 < end {
                address += self.stride;
            }
        }
        Ok(())
    }
}

/// Contract an eligible affine interval through a prepared window when its
/// work is below the general contraction's `general_work`. Prepared work counts
/// the row tables, the high equality window built by [`PreparedDigits::new`],
/// and one table lookup per row and base.
///
/// Eligible intervals have adjacent digits (`digit_stride = 1`), one digit
/// block per outer index (`outer_stride = digits.len()`) and a row-aligned
/// outer window. The caller has already validated the geometry, address bound
/// and work cap, so this never changes which inputs are accepted.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_eval_prepared_digit_intervals<F, A, H>(
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
    template: &A,
    general_work: usize,
) -> Result<Option<A>, AkitaError>
where
    F: Field,
    A: AffineWeight<F>,
    H: AffineWeightSource<F, A> + ?Sized,
{
    let lanes = low_weights.len().max(1);
    if digit_stride != 1
        || outer_stride != digit_weights.len()
        || !outer_start.is_multiple_of(lanes)
        || !live_len.is_multiple_of(lanes)
    {
        return Ok(None);
    }
    let rows = live_len / lanes;
    let prepared_work =
        PreparedDigits::<F, A>::preparation_work(challenges.len(), outer_stride, low_weights.len())
            .and_then(|preparation| {
                checked::sum([preparation, checked::product([rows, base_offsets.len()])?])
            });
    if prepared_work.is_none_or(|work| work >= general_work) {
        return Ok(None);
    }
    let Some(prepared) = PreparedDigits::new(challenges, digit_weights, low_weights)? else {
        return Ok(None);
    };
    let first = outer_start / lanes;
    let mut out = template.zero_like();
    for (index, &base) in base_offsets.iter().enumerate() {
        let mut rows_sum = template.zero_like();
        prepared.accumulate(&mut rows_sum, base, first, rows, high_weights)?;
        match base_scales.get(index) {
            Some(&scale) => out.add_scaled(&rows_sum, scale),
            None => out.add(&rows_sum),
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset_eq::{eq_eval_at_index, eval_affine_digit_intervals};
    use jolt_field::{One, Prime128Offset6A23801F as F, Ring, Zero};

    fn prepare(point: &[F], digits: &[F], low: &[F]) -> PreparedDigits<F, F> {
        PreparedDigits::new(point, digits, low).unwrap().unwrap()
    }

    fn rows(
        prepared: &PreparedDigits<F, F>,
        base: usize,
        first: usize,
        rows: usize,
        high: &[F],
    ) -> F {
        let mut out = F::zero();
        prepared
            .accumulate(&mut out, base, first, rows, high)
            .unwrap();
        out
    }

    #[test]
    fn prepared_windows_match_general_and_literal_oracles() {
        for point_kind in 0..3 {
            let point: Vec<F> = (0..15)
                .map(|i| match point_kind {
                    0 => F::from_u64(11 + i),
                    1 => F::zero(),
                    _ => F::one(),
                })
                .collect();
            for digit_count in [1, 3, 7, 43, 86] {
                let digits: Vec<F> = (0..digit_count)
                    .scan(F::one(), |power, _| {
                        let v = *power;
                        *power *= F::from_u64(8);
                        Some(v)
                    })
                    .collect();
                for lanes in [1, 2, 4] {
                    let low: Vec<F> = if lanes == 1 {
                        Vec::new()
                    } else {
                        (0..lanes).map(|i| F::from_u64(19 + i as u64)).collect()
                    };
                    let high: Vec<F> = (0..29).map(|i| F::from_u64(101 + i)).collect();
                    let prepared = prepare(&point, &digits, &low);
                    for base in [0, 1, 17, 63, 128, 253] {
                        let first = 3;
                        let row_count = 7;
                        let got = rows(&prepared, base, first, row_count, &high);
                        let old = eval_affine_digit_intervals(
                            &point,
                            &[base],
                            first * lanes,
                            row_count * lanes,
                            digit_count,
                            1,
                            &digits,
                            &high,
                            &low,
                            &[],
                        )
                        .unwrap();
                        let mut literal = F::zero();
                        for row in 0..row_count {
                            for lane in 0..lanes {
                                for (d, &digit) in digits.iter().enumerate() {
                                    let a = base + (row * lanes + lane) * digit_count + d;
                                    literal += high[first + row]
                                        * low.get(lane).copied().unwrap_or_else(F::one)
                                        * digit
                                        * eq_eval_at_index(&point, a);
                                }
                            }
                        }
                        assert_eq!(
                            got, literal,
                            "literal base={base}, digits={digit_count}, lanes={lanes}"
                        );
                        assert_eq!(
                            got, old,
                            "general base={base}, digits={digit_count}, lanes={lanes}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_windows_validate_bounds_and_specialization_limits() {
        let point = vec![F::one(); 10];
        let weights = vec![F::one(); 20];
        assert!(PreparedDigits::<F, F>::new(&point, &[], &[]).is_err());
        assert!(PreparedDigits::new(&point, &[F::one()], &[F::one(); 3]).is_err());
        assert!(PreparedDigits::<F, F>::new(&point, &[F::one(); 513], &[])
            .unwrap()
            .is_none());
        assert!(PreparedDigits::<F, F>::new(&[], &[F::one(); 3], &[])
            .unwrap()
            .is_none());
        let p = prepare(&point, &[F::one(); 43], &[F::one(); 2]);
        let mut out = F::zero();
        assert!(matches!(
            p.accumulate(&mut out, 0, 0, MAX_COMPACT_STRIDE_TERMS + 1, &weights),
            Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                ..
            })
        ));
        assert!(p.accumulate(&mut out, usize::MAX, 0, 1, &weights).is_err());
        assert!(p.accumulate(&mut out, 1020, 0, 1, &weights).is_err());
        assert!(p.accumulate(&mut out, 0, 19, 2, &weights).is_err());
        assert!(p.accumulate(&mut out, 0, usize::MAX, 1, &weights).is_err());
        assert_eq!(rows(&p, 0, 20, 0, &weights), F::zero());
    }

    #[test]
    fn interleaved_geometric_lanes_match_the_weighted_sum() {
        for period in [1usize, 2, 4] {
            for beta in [F::zero(), F::from_u64(8)] {
                for kind in 0..3 {
                    let point: Vec<F> = (0..14)
                        .map(|i| {
                            F::from_u64(match kind {
                                0 => i + 7,
                                1 => 0,
                                _ => 1,
                            })
                        })
                        .collect();
                    let mut digits = Vec::new();
                    let mut power = F::one();
                    for _ in 0..43 {
                        for lane in 0..period {
                            digits.push(
                                power * F::from_u64(if lane == 0 { 1 } else { lane as u64 * 11 }),
                            );
                        }
                        power *= beta;
                    }
                    let low = [F::from_u64(3), F::from_u64(5)];
                    let weights: Vec<F> = (0..19).map(|i| F::from_u64(i + 13)).collect();
                    let prepared = prepare(&point, &digits, &low);
                    for base in [0, 17, 253, 511] {
                        let mut expected = F::zero();
                        for row in 0..7 {
                            for (lane, &lane_weight) in low.iter().enumerate() {
                                for (d, &digit) in digits.iter().enumerate() {
                                    expected += weights[row + 3]
                                        * lane_weight
                                        * digit
                                        * eq_eval_at_index(
                                            &point,
                                            base + (row * 2 + lane) * digits.len() + d,
                                        );
                                }
                            }
                        }
                        assert_eq!(rows(&prepared, base, 3, 7, &weights), expected);
                    }
                }
            }
        }
    }
    #[test]
    fn bucketed_prepared_rows_match_dense_for_unit_and_folded_windows() {
        for boolean in [false, true] {
            let point = (0..16)
                .map(|i| {
                    if boolean {
                        F::from_u64((37 >> i) & 1)
                    } else {
                        F::from_u64(i + 7)
                    }
                })
                .collect::<Vec<_>>();
            let digits = (0..5)
                .scan(F::one(), |v, _| {
                    let out = *v;
                    *v *= F::from_u64(3);
                    Some(out)
                })
                .collect::<Vec<_>>();
            let weights = (0..128).map(|i| F::from_u64(i + 13)).collect::<Vec<_>>();
            for low in [Vec::new(), vec![F::from_u64(11), F::from_u64(19)]] {
                let prepared = prepare(&point, &digits, &low);
                for base in [0, 7, 31] {
                    let first = 7;
                    let count = 79;
                    let mut expected = F::zero();
                    for row in 0..count {
                        for lane in 0..low.len().max(1) {
                            for (d, &digit) in digits.iter().enumerate() {
                                let address = base
                                    + row * low.len().max(1) * digits.len()
                                    + lane * digits.len()
                                    + d;
                                expected += weights[first + row]
                                    * low.get(lane).copied().unwrap_or_else(F::one)
                                    * digit
                                    * eq_eval_at_index(&point, address);
                            }
                        }
                    }
                    assert_eq!(rows(&prepared, base, first, count, &weights), expected);
                }
            }
        }
    }
}
