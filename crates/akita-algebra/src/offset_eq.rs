//! Offset-EQ helpers for structured inner products.
//!
//! This module provides two related evaluators over the offset-shifted
//! equality polynomial:
//!
//! 1. [`eval_offset_eq_interval`], a sparse/pruned partial multilinear binding
//!    of a single materialized factor over the contiguous global interval
//!    `[offset, offset + len)`. It places the values in global index
//!    coordinates and runs the standard little-endian fold, pruning every
//!    parent whose whole subtree lies outside the live interval.
//! 2. A `2`-adic peel for shapes `x = u + 2^m q` via
//!    [`summarize_pow2_block_carries`], which strips the aligned inner `2^m`
//!    block into the two carry buckets `[A0, A1]`, leaving a small coarse outer
//!    sum over `q` for the caller to combine with the high `eq` factor.
//!
//! Binary addition `offset + z` produces carries that propagate across bit
//! positions; the peel captures that carry state (0 or 1) as the two buckets
//! `[A0, A1]` of [`summarize_pow2_block_carries`].

use akita_error::AkitaError;

use crate::FieldCore;

/// Sparse/pruned partial multilinear evaluation of a single materialized
/// factor over the contiguous global interval `[offset, offset + factor.len())`.
///
/// Computes:
///
/// ```text
/// scale · Σ_{z=0}^{factor.len()-1}  eq(x_challenges, offset + z) · factor[z]
/// ```
///
/// where indices `offset + z ≥ 2^n` (with `n = x_challenges.len()`) fall
/// outside the equality domain and contribute zero.
///
/// This places the values in **global** index coordinates and runs the
/// standard little-endian multilinear binding fold, pruning every parent node
/// whose whole subtree is outside the live interval. Each live parent costs
/// exactly one field multiplication, so the
/// total is `Σ_k (⌊hi/2^{k+1}⌋ − ⌊lo/2^{k+1}⌋ + 1)` multiplications plus one
/// final `scale` product.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidInput`] if `offset + factor.len()` overflows
/// `usize`.
pub fn eval_offset_eq_interval<F: FieldCore>(
    x_challenges: &[F],
    offset: usize,
    scale: F,
    factor: &[F],
) -> Result<F, AkitaError> {
    let n = x_challenges.len();
    if factor.is_empty() {
        return Ok(F::zero());
    }

    // Indices at or beyond `2^n` are outside the equality domain (weight 0).
    let in_domain = n < usize::BITS as usize;
    if in_domain && offset >= (1usize << n) {
        return Ok(F::zero());
    }

    let last = offset
        .checked_add(factor.len() - 1)
        .ok_or_else(|| AkitaError::InvalidInput("offset-eq interval overflow".to_string()))?;

    let mut lo = offset;
    let mut hi = if in_domain {
        core::cmp::min(last, (1usize << n) - 1)
    } else {
        last
    };

    // Active values in global coordinates: `a[i - lo] = factor[i - offset]`.
    let mut a: Vec<F> = factor[..=(hi - lo)].to_vec();

    for &r in x_challenges.iter() {
        let new_lo = lo >> 1;
        let new_hi = hi >> 1;
        let mut next = Vec::with_capacity(new_hi - new_lo + 1);
        for p in new_lo..=new_hi {
            let left = 2 * p;
            let right = left + 1;
            let has_left = left >= lo && left <= hi;
            let has_right = right >= lo && right <= hi;
            let val = if has_left && has_right {
                let x0 = a[left - lo];
                let x1 = a[right - lo];
                x0 + r * (x1 - x0)
            } else if has_left {
                let x0 = a[left - lo];
                x0 - r * x0
            } else {
                let x1 = a[right - lo];
                r * x1
            };
            next.push(val);
        }
        a = next;
        lo = new_lo;
        hi = new_hi;
    }

    debug_assert_eq!(a.len(), 1);
    Ok(scale * a[0])
}

/// Build `table[k] = eq(high_challenges, offset_high + k)` for `k ∈ [0, hi_len]`.
pub fn high_eq_window<F: FieldCore>(
    high_challenges: &[F],
    offset_high: usize,
    hi_len: usize,
) -> Vec<F> {
    (0..=hi_len)
        .map(|k| eq_eval_at_index(high_challenges, offset_high + k))
        .collect()
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

/// Summarize one power-of-two inner block `values[u]` into the two carry cases
/// induced by adding `offset_low + u`, where `offset_low < values.len()`.
///
/// `eq_low` must be the equality table on the low `log2(values.len())` bits.
///
/// # Errors
///
/// Returns an error if `values` is not power-of-two sized, if `eq_low` has the
/// wrong length, or if `offset_low` does not lie inside the peeled block.
pub fn summarize_pow2_block_carries<F: FieldCore>(
    eq_low: &[F],
    offset_low: usize,
    values: &[F],
) -> Result<[F; 2], AkitaError> {
    if !values.len().is_power_of_two() {
        return Err(AkitaError::InvalidInput(
            "peeled inner block length must be a power of two".to_string(),
        ));
    }
    if eq_low.len() != values.len() {
        return Err(AkitaError::InvalidSize {
            expected: values.len(),
            actual: eq_low.len(),
        });
    }
    if offset_low >= values.len() {
        return Err(AkitaError::InvalidInput(
            "low offset must lie inside the peeled block".to_string(),
        ));
    }

    let inner_bits = values.len().trailing_zeros() as usize;
    let inner_mask = values.len() - 1;
    let mut out = [F::zero(), F::zero()];

    for (u, &value) in values.iter().enumerate() {
        let sum = offset_low + u;
        let carry = sum >> inner_bits;
        debug_assert!(
            carry < 2,
            "sum of two peeled indices must carry at most one bit"
        );
        let low_idx = sum & inner_mask;
        out[carry] += value * eq_low[low_idx];
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq_poly::EqPolynomial;
    use crate::RandomSampling;
    use jolt_field::Fp64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;

    fn reference_offset_eq_tensor(
        x_challenges: &[F],
        offset: usize,
        scale: F,
        factors: &[&[F]],
    ) -> F {
        let dims: Vec<usize> = factors
            .iter()
            .map(|f| {
                if f.is_empty() {
                    1
                } else {
                    f.len().next_power_of_two()
                }
            })
            .collect();
        let total: usize = dims.iter().product();
        let eq_table = EqPolynomial::evals(x_challenges).unwrap();
        let mut acc = F::zero();
        for z in 0..total {
            let mut idx = z;
            let mut prod = scale;
            for (j, &f) in factors.iter().enumerate() {
                let local = idx % dims[j];
                idx /= dims[j];
                prod *= if f.is_empty() {
                    if local == 0 {
                        F::one()
                    } else {
                        F::zero()
                    }
                } else if local < f.len() {
                    f[local]
                } else {
                    F::zero()
                };
            }
            let global = offset + z;
            if global < eq_table.len() {
                acc += eq_table[global] * prod;
            }
        }
        acc
    }

    fn random_vec(rng: &mut StdRng, len: usize) -> Vec<F> {
        (0..len).map(|_| F::random(rng)).collect()
    }

    fn reference_pow2_peeled_blocks(x_challenges: &[F], offset: usize, blocks: &[Vec<F>]) -> F {
        let inner_len = blocks.first().map_or(1, Vec::len);
        let eq_table = EqPolynomial::evals(x_challenges).unwrap();
        let mut acc = F::zero();

        for (q, block) in blocks.iter().enumerate() {
            assert_eq!(block.len(), inner_len);
            for (u, &value) in block.iter().enumerate() {
                let idx = offset + u + inner_len * q;
                if idx < eq_table.len() {
                    acc += value * eq_table[idx];
                }
            }
        }

        acc
    }

    #[test]
    fn interval_matches_reference_offset_zero() {
        let mut rng = StdRng::seed_from_u64(0xB1);
        let factor = random_vec(&mut rng, 21);
        let r = random_vec(&mut rng, 5);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_interval(&r, 0, scale, &factor).unwrap();
        let expected = reference_offset_eq_tensor(&r, 0, scale, &[&factor]);
        assert_eq!(got, expected);
    }

    #[test]
    fn interval_matches_reference_carry_offset() {
        let mut rng = StdRng::seed_from_u64(0xB2);
        let factor = random_vec(&mut rng, 21);
        let r = random_vec(&mut rng, 5);
        let scale = F::random(&mut rng);

        // Interval [11, 31] inside domain 2^5 = 32, carry-heavy offset.
        let got = eval_offset_eq_interval(&r, 11, scale, &factor).unwrap();
        let expected = reference_offset_eq_tensor(&r, 11, scale, &[&factor]);
        assert_eq!(got, expected);
    }

    #[test]
    fn interval_matches_reference_sweep() {
        let mut rng = StdRng::seed_from_u64(0xB3);
        for n in 3..12usize {
            let domain = 1usize << n;
            for &len in &[1usize, 3, 8, 21, 100, 300] {
                let factor = random_vec(&mut rng, len);
                let r = random_vec(&mut rng, n);
                let scale = F::random(&mut rng);
                // Offsets: zero, carry-heavy flush-to-top, a mid value, plus an
                // offset that pushes the interval tail past the domain (clamp).
                let max_offset = domain.saturating_sub(len);
                let mut offsets = vec![0usize];
                if max_offset > 0 {
                    offsets.push(max_offset);
                    offsets.push(max_offset / 2);
                }
                offsets.push(domain); // fully outside the domain -> zero
                for &offset in &offsets {
                    let got = eval_offset_eq_interval(&r, offset, scale, &factor).unwrap();
                    let expected = reference_offset_eq_tensor(&r, offset, scale, &[&factor]);
                    assert_eq!(got, expected, "n={n} len={len} offset={offset}");
                }
            }
        }
    }

    #[test]
    fn interval_matches_reference_with_partial_clamp() {
        let mut rng = StdRng::seed_from_u64(0xB4);
        // len 300 padded to 512 = 2^9 fits in n = 9 bits; offset pushes the tail
        // of the interval past 2^9 so the high indices are clamped/dropped.
        let n = 9usize;
        let factor = random_vec(&mut rng, 300);
        let r = random_vec(&mut rng, n);
        let scale = F::random(&mut rng);
        let offset = 300; // 300 + 300 = 600 > 512, so indices >= 512 drop out
        let got = eval_offset_eq_interval(&r, offset, scale, &factor).unwrap();
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[&factor]);
        assert_eq!(got, expected);
    }

    #[test]
    fn interval_offset_outside_domain_is_zero() {
        let mut rng = StdRng::seed_from_u64(0xB5);
        let factor = random_vec(&mut rng, 4);
        let r = random_vec(&mut rng, 3);
        let got = eval_offset_eq_interval(&r, 1usize << r.len(), F::one(), &factor).unwrap();
        assert_eq!(got, F::zero());
    }

    /// Combine per-block carry buckets `[A0, A1]` with the high `eq` factor,
    /// the way `compute_r_contribution` does: `A0` lands on `offset_high + q`
    /// and the carried `A1` on `offset_high + q + 1`.
    fn combine_pow2_carry_terms(
        x_challenges: &[F],
        offset: usize,
        peeled_bits: usize,
        carry_terms: &[[F; 2]],
    ) -> F {
        let offset_high = offset >> peeled_bits;
        let high = &x_challenges[peeled_bits..];
        let mut out = F::zero();
        for (q, terms) in carry_terms.iter().enumerate() {
            out += terms[0] * eq_eval_at_index(high, offset_high + q);
            out += terms[1] * eq_eval_at_index(high, offset_high + q + 1);
        }
        out
    }

    #[test]
    fn summarize_pow2_block_carries_matches_reference_ragged() {
        let mut rng = StdRng::seed_from_u64(0xAC);
        let peeled_bits = 3usize;
        let inner_len = 1usize << peeled_bits;
        let outer_len = 5usize;
        let r = random_vec(&mut rng, 7);
        let offset = 0b101101usize;
        let eq_low = EqPolynomial::evals(&r[..peeled_bits]).unwrap();
        let offset_low = offset & (inner_len - 1);

        let blocks: Vec<Vec<F>> = (0..outer_len)
            .map(|_| random_vec(&mut rng, inner_len))
            .collect();
        let carry_terms: Vec<[F; 2]> = blocks
            .iter()
            .map(|block| summarize_pow2_block_carries(&eq_low, offset_low, block))
            .collect::<Result<_, _>>()
            .unwrap();

        let got = combine_pow2_carry_terms(&r, offset, peeled_bits, &carry_terms);
        let expected = reference_pow2_peeled_blocks(&r, offset, &blocks);
        assert_eq!(got, expected);
    }

    #[test]
    fn summarize_pow2_block_carries_matches_reference_high_overflow() {
        let mut rng = StdRng::seed_from_u64(0xAD);
        let peeled_bits = 2usize;
        let inner_len = 1usize << peeled_bits;
        let outer_len = 6usize;
        let r = random_vec(&mut rng, 5);
        let offset = 27usize;
        let eq_low = EqPolynomial::evals(&r[..peeled_bits]).unwrap();
        let offset_low = offset & (inner_len - 1);

        let blocks: Vec<Vec<F>> = (0..outer_len)
            .map(|_| random_vec(&mut rng, inner_len))
            .collect();
        let carry_terms: Vec<[F; 2]> = blocks
            .iter()
            .map(|block| summarize_pow2_block_carries(&eq_low, offset_low, block))
            .collect::<Result<_, _>>()
            .unwrap();

        let got = combine_pow2_carry_terms(&r, offset, peeled_bits, &carry_terms);
        let expected = reference_pow2_peeled_blocks(&r, offset, &blocks);
        assert_eq!(got, expected);
    }
}
