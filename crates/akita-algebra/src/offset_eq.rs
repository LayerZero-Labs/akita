//! Offset-EQ helpers for structured inner products.
//!
//! This module provides two related evaluators:
//!
//! 1. `eval_offset_eq_tensor`, a binary carry-DP for rank-1 tensors whose
//!    exposed factor boundaries align with power-of-two lower strides.
//! 2. A `2`-adic peel for shapes `x = u + 2^m q`, which strips the aligned
//!    inner block exactly and leaves only a small coarse outer sum over `q`.
//!
//! Binary addition `offset + z` produces carries that propagate across bit
//! positions. The tensor path tracks that carry state (0 or 1) via 2×2
//! transition matrices, then folds each tensor factor into a summary matrix in
//! a DP analogous to `multilinear_eval_ref`.
//!
//! When `offset = 0`, the tensor path has no carries and collapses to a product
//! of small MLE evaluations (the aligned fast path).

use crate::FieldCore;

/// Sparse carry transition for one bit position.
///
/// Each carry_in maps to exactly one carry_out with a weight derived from
/// `eq_bit(r[t], x_t)` where `x_t = (offset_bit + local_bit + carry_in) mod 2`.
#[derive(Clone, Copy)]
struct CarryTransition<F> {
    weight: [F; 2],
    target: [u8; 2],
}

/// Dense 2×2 carry summary matrix.
///
/// Entry `[carry_in][carry_out]` accumulates the weighted contribution
/// from all local-bit assignments folded so far.
#[derive(Clone, Copy)]
struct CarryMatrix<F>([[F; 2]; 2]);

impl<F: FieldCore> CarryMatrix<F> {
    fn identity() -> Self {
        Self([[F::one(), F::zero()], [F::zero(), F::one()]])
    }

    fn zero() -> Self {
        Self([[F::zero(); 2]; 2])
    }

    fn scaled(s: F) -> Self {
        Self([[s, F::zero()], [F::zero(), s]])
    }

    /// `self = self + rhs`
    fn add_assign(&mut self, rhs: &Self) {
        self.0[0][0] += rhs.0[0][0];
        self.0[0][1] += rhs.0[0][1];
        self.0[1][0] += rhs.0[1][0];
        self.0[1][1] += rhs.0[1][1];
    }

    /// Right-multiply by a sparse carry transition (4 muls, 0–2 adds).
    fn mul_transition(&self, tr: &CarryTransition<F>) -> Self {
        let t0 = tr.target[0] as usize;
        let t1 = tr.target[1] as usize;
        let w0 = tr.weight[0];
        let w1 = tr.weight[1];

        let mut out = Self::zero();
        out.0[0][t0] += self.0[0][0] * w0;
        out.0[0][t1] += self.0[0][1] * w1;
        out.0[1][t0] += self.0[1][0] * w0;
        out.0[1][t1] += self.0[1][1] * w1;
        out
    }

    /// Dense 2×2 matrix multiply: `self * rhs`.
    fn mul_matrix(&self, rhs: &Self) -> Self {
        let a = &self.0;
        let b = &rhs.0;
        Self([
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ])
    }
}

/// Build the two sparse transitions (for local_bit = 0 and 1) at one bit
/// position given `r[t]` and whether the offset bit is set.
fn carry_transition_pair<F: FieldCore>(r_t: F, offset_bit: bool) -> [CarryTransition<F>; 2] {
    let one_minus_r = F::one() - r_t;

    // eq_bit(r, x) = r if x=1, (1-r) if x=0
    let eq_bit = [one_minus_r, r_t];
    let o = offset_bit as u8;

    // For each (local_bit, carry_in): sum = o + b + c, x = sum%2, c' = sum/2
    let mut result = [CarryTransition {
        weight: [F::zero(); 2],
        target: [0; 2],
    }; 2];

    for b in 0u8..2 {
        for c_in in 0u8..2 {
            let sum = o + b + c_in;
            let x_bit = sum & 1;
            let c_out = sum >> 1;
            result[b as usize].weight[c_in as usize] = eq_bit[x_bit as usize];
            result[b as usize].target[c_in as usize] = c_out;
        }
    }

    result
}

/// Fold one tensor factor into its carry summary matrix.
///
/// `factor` is internally padded to `next_power_of_two()` with zeros.
/// `base_bit` is the starting global bit position for this factor.
fn factor_summary<F: FieldCore>(
    factor: &[F],
    x_challenges: &[F],
    offset: usize,
    base_bit: usize,
) -> CarryMatrix<F> {
    let len = factor.len();
    if len == 0 {
        return CarryMatrix::identity();
    }
    let m = len.next_power_of_two().trailing_zeros() as usize;
    let padded_len = 1usize << m;

    let mut acc: Vec<CarryMatrix<F>> = (0..padded_len)
        .map(|u| {
            let val = if u < len { factor[u] } else { F::zero() };
            CarryMatrix::scaled(val)
        })
        .collect();

    for s in 0..m {
        let bit_pos = base_bit + s;
        let [tr0, tr1] = if bit_pos < x_challenges.len() {
            let o_bit = (offset >> bit_pos) & 1 != 0;
            carry_transition_pair(x_challenges[bit_pos], o_bit)
        } else {
            carry_transition_pair(F::zero(), false)
        };

        let half = acc.len() / 2;
        let mut next = Vec::with_capacity(half);
        for q in 0..half {
            let mut m0 = acc[2 * q].mul_transition(&tr0);
            let m1 = acc[2 * q + 1].mul_transition(&tr1);
            m0.add_assign(&m1);
            next.push(m0);
        }
        acc = next;
    }

    debug_assert_eq!(acc.len(), 1);
    acc.into_iter().next().unwrap()
}

/// Evaluate `Σ_z eq(r, offset + z) · scale · Π_j factors[j][idx_j(z)]`.
///
/// `factors` are ordered least-significant to most-significant, matching
/// Akita's little-endian `EqPolynomial` convention. Each factor is
/// internally padded to `next_power_of_two()`; padding zeros cancel out.
///
/// Fast path: when `offset = 0`, uses simple products of small MLE
/// evaluations (no carry matrices).
pub fn eval_offset_eq_tensor<F: FieldCore>(
    x_challenges: &[F],
    offset: usize,
    scale: F,
    factors: &[&[F]],
) -> F {
    if factors.is_empty() {
        if offset == 0 {
            return scale * EqPolynomial::zero_selector(x_challenges);
        }
        let n = x_challenges.len();
        if offset < (1usize << n) {
            let mut prod = scale;
            for (t, &r_t) in x_challenges.iter().enumerate() {
                let x_bit = (offset >> t) & 1;
                prod *= if x_bit == 1 { r_t } else { F::one() - r_t };
            }
            return prod;
        }
        return F::zero();
    }

    if offset == 0 {
        return eval_offset_eq_tensor_aligned(x_challenges, scale, factors);
    }

    eval_offset_eq_tensor_carry(x_challenges, offset, scale, factors)
}

/// Evaluate `eq(r, index)` for a single hypercube index in little-endian order.
fn eq_eval_at_index<F: FieldCore>(x_challenges: &[F], index: usize) -> F {
    if x_challenges.len() < usize::BITS as usize && index >= (1usize << x_challenges.len()) {
        return F::zero();
    }

    x_challenges
        .iter()
        .enumerate()
        .fold(F::one(), |acc, (bit_idx, &r_t)| {
            let bit = (index >> bit_idx) & 1;
            acc * if bit == 1 { r_t } else { F::one() - r_t }
        })
}

/// Summarize one power-of-two inner block `values[u]` into the two carry cases
/// induced by adding `offset_low + u`, where `offset_low < values.len()`.
///
/// `eq_low` must be the equality table on the low `log2(values.len())` bits.
///
/// # Panics
///
/// Panics if `values` is not power-of-two sized, if `eq_low` has the wrong
/// length, or if `offset_low` does not lie inside the peeled block.
pub fn summarize_pow2_block_carries<F: FieldCore>(
    eq_low: &[F],
    offset_low: usize,
    values: &[F],
) -> [F; 2] {
    assert!(
        values.len().is_power_of_two(),
        "peeled inner block length must be a power of two"
    );
    assert_eq!(
        eq_low.len(),
        values.len(),
        "low eq table must match peeled inner block length"
    );
    assert!(
        offset_low < values.len(),
        "low offset must lie inside the peeled block"
    );

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

    out
}

/// Evaluate a coarse outer sum after peeling an inner `2^m` block.
///
/// `carry_terms[q][c]` is the low-bit summary for outer index `q` and carry
/// `c`, so the result is:
///
/// ```text
/// Σ_q carry_terms[q][0] * eq_high(offset_high + q)
///   + carry_terms[q][1] * eq_high(offset_high + q + 1)
/// ```
pub fn eval_offset_eq_peeled_carry_terms<F: FieldCore>(
    x_challenges: &[F],
    offset: usize,
    peeled_bits: usize,
    carry_terms: &[[F; 2]],
) -> F {
    let offset_high = offset >> peeled_bits;
    let high_challenges = &x_challenges[peeled_bits..];

    carry_terms
        .iter()
        .enumerate()
        .fold(F::zero(), |acc, (q, terms)| {
            let acc = if terms[0].is_zero() {
                acc
            } else {
                acc + terms[0] * eq_eval_at_index(high_challenges, offset_high + q)
            };
            if terms[1].is_zero() {
                acc
            } else {
                acc + terms[1] * eq_eval_at_index(high_challenges, offset_high + q + 1)
            }
        })
}

/// Aligned fast path: offset = 0, no carry ever generated.
///
/// Degenerates to `scale · Π_j MLE(f_j, r_j) · Π_{t >= m} (1 - r[t])`.
fn eval_offset_eq_tensor_aligned<F: FieldCore>(
    x_challenges: &[F],
    scale: F,
    factors: &[&[F]],
) -> F {
    let mut result = scale;
    let mut bit_cursor = 0usize;

    for &factor in factors {
        if factor.is_empty() {
            continue;
        }
        let m = factor.len().next_power_of_two().trailing_zeros() as usize;
        let r_slice = &x_challenges[bit_cursor..bit_cursor + m];
        result *= mle_small(factor, r_slice);
        bit_cursor += m;
    }

    for &r_t in &x_challenges[bit_cursor..] {
        result *= F::one() - r_t;
    }

    result
}

/// General carry-DP path for arbitrary offsets.
fn eval_offset_eq_tensor_carry<F: FieldCore>(
    x_challenges: &[F],
    offset: usize,
    scale: F,
    factors: &[&[F]],
) -> F {
    let mut composed = CarryMatrix::identity();
    let mut bit_cursor = 0usize;

    for &factor in factors {
        if factor.is_empty() {
            continue;
        }
        let summary = factor_summary(factor, x_challenges, offset, bit_cursor);
        composed = composed.mul_matrix(&summary);
        bit_cursor += factor.len().next_power_of_two().trailing_zeros() as usize;
    }

    for (t, &r_t) in x_challenges.iter().enumerate().skip(bit_cursor) {
        let o_bit = (offset >> t) & 1 != 0;
        let [tr0, _] = carry_transition_pair(r_t, o_bit);
        composed = composed.mul_transition(&tr0);
    }

    scale * composed.0[0][0]
}

/// Evaluate a small multilinear polynomial at a point.
///
/// Like `multilinear_eval` but works on non-power-of-two lengths by
/// padding with zeros.
fn mle_small<F: FieldCore>(evals: &[F], point: &[F]) -> F {
    if point.is_empty() {
        return if evals.is_empty() {
            F::zero()
        } else {
            evals[0]
        };
    }

    let m = point.len();
    let padded_len = 1usize << m;
    debug_assert!(evals.len() <= padded_len);

    let mut buf: Vec<F> = Vec::with_capacity(padded_len);
    buf.extend_from_slice(evals);
    buf.resize(padded_len, F::zero());

    for &r in point.iter() {
        let half = buf.len() / 2;
        for i in 0..half {
            buf[i] = buf[2 * i] + r * (buf[2 * i + 1] - buf[2 * i]);
        }
        buf.truncate(half);
    }

    buf[0]
}

use super::eq_poly::EqPolynomial;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RandomSampling;
    use akita_field::Fp64;
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
        let eq_table = EqPolynomial::evals(x_challenges);
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
        let eq_table = EqPolynomial::evals(x_challenges);
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
    fn single_factor_offset_zero_pow2() {
        let mut rng = StdRng::seed_from_u64(0xA1);
        let factor = random_vec(&mut rng, 8);
        let r = random_vec(&mut rng, 8);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, 0, scale, &[&factor]);
        let expected = reference_offset_eq_tensor(&r, 0, scale, &[&factor]);
        assert_eq!(got, expected);
    }

    #[test]
    fn single_factor_offset_zero_non_pow2() {
        let mut rng = StdRng::seed_from_u64(0xA2);
        let factor = random_vec(&mut rng, 5);
        let r = random_vec(&mut rng, 6);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, 0, scale, &[&factor]);
        let expected = reference_offset_eq_tensor(&r, 0, scale, &[&factor]);
        assert_eq!(got, expected);
    }

    #[test]
    fn three_factors_offset_zero() {
        let mut rng = StdRng::seed_from_u64(0xA3);
        let f0 = random_vec(&mut rng, 4);
        let f1 = random_vec(&mut rng, 2);
        let f2 = random_vec(&mut rng, 8);
        let r = random_vec(&mut rng, 10);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, 0, scale, &[&f0, &f1, &f2]);
        let expected = reference_offset_eq_tensor(&r, 0, scale, &[&f0, &f1, &f2]);
        assert_eq!(got, expected);
    }

    #[test]
    fn three_factors_aligned_offset() {
        let mut rng = StdRng::seed_from_u64(0xA4);
        let f0 = random_vec(&mut rng, 4);
        let f1 = random_vec(&mut rng, 2);
        let f2 = random_vec(&mut rng, 4);
        // total factor bits = 2 + 1 + 2 = 5, so offset must be < 2^10 - 2^5
        let offset = 32; // = 2^5, aligned to total factor width
        let r = random_vec(&mut rng, 10);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, offset, scale, &[&f0, &f1, &f2]);
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[&f0, &f1, &f2]);
        assert_eq!(got, expected);
    }

    #[test]
    fn three_factors_carry_heavy_offset() {
        let mut rng = StdRng::seed_from_u64(0xA5);
        let f0 = random_vec(&mut rng, 4);
        let f1 = random_vec(&mut rng, 2);
        let f2 = random_vec(&mut rng, 4);
        let offset = 0b10101; // many low bits set
        let r = random_vec(&mut rng, 10);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, offset, scale, &[&f0, &f1, &f2]);
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[&f0, &f1, &f2]);
        assert_eq!(got, expected);
    }

    #[test]
    fn mixed_non_pow2_factors() {
        let mut rng = StdRng::seed_from_u64(0xA6);
        let f0 = random_vec(&mut rng, 3);
        let f1 = random_vec(&mut rng, 5);
        let f2 = random_vec(&mut rng, 6);
        let offset = 7;
        let r = random_vec(&mut rng, 12);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, offset, scale, &[&f0, &f1, &f2]);
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[&f0, &f1, &f2]);
        assert_eq!(got, expected);
    }

    #[test]
    fn degenerate_length_one_factor() {
        let mut rng = StdRng::seed_from_u64(0xA7);
        let f0 = random_vec(&mut rng, 1);
        let f1 = random_vec(&mut rng, 4);
        let offset = 3;
        let r = random_vec(&mut rng, 6);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, offset, scale, &[&f0, &f1]);
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[&f0, &f1]);
        assert_eq!(got, expected);
    }

    #[test]
    fn large_n_small_factors() {
        let mut rng = StdRng::seed_from_u64(0xA8);
        let f0 = random_vec(&mut rng, 4);
        let f1 = random_vec(&mut rng, 2);
        // total factor bits = 3, but 20 challenge bits
        let offset = 137;
        let r = random_vec(&mut rng, 20);
        let scale = F::random(&mut rng);

        let got = eval_offset_eq_tensor(&r, offset, scale, &[&f0, &f1]);
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[&f0, &f1]);
        assert_eq!(got, expected);
    }

    #[test]
    fn offset_zero_matches_aligned_fast_path() {
        let mut rng = StdRng::seed_from_u64(0xA9);
        let f0 = random_vec(&mut rng, 4);
        let f1 = random_vec(&mut rng, 8);
        let r = random_vec(&mut rng, 10);
        let scale = F::random(&mut rng);

        let via_fast = eval_offset_eq_tensor(&r, 0, scale, &[&f0, &f1]);
        let via_carry = eval_offset_eq_tensor_carry(&r, 0, scale, &[&f0, &f1]);
        assert_eq!(via_fast, via_carry);
    }

    #[test]
    fn no_factors_offset_zero() {
        let mut rng = StdRng::seed_from_u64(0xAA);
        let r = random_vec(&mut rng, 4);
        let scale = F::random(&mut rng);
        let got = eval_offset_eq_tensor(&r, 0, scale, &[]);
        let expected = scale * EqPolynomial::zero_selector(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn no_factors_nonzero_offset() {
        let mut rng = StdRng::seed_from_u64(0xAB);
        let r = random_vec(&mut rng, 4);
        let scale = F::random(&mut rng);
        let offset = 5;
        let got = eval_offset_eq_tensor(&r, offset, scale, &[]);
        let expected = reference_offset_eq_tensor(&r, offset, scale, &[]);
        assert_eq!(got, expected);
    }

    #[test]
    fn pow2_peel_matches_reference_with_ragged_outer_len() {
        let mut rng = StdRng::seed_from_u64(0xAC);
        let peeled_bits = 3usize;
        let inner_len = 1usize << peeled_bits;
        let outer_len = 5usize;
        let r = random_vec(&mut rng, 7);
        let offset = 0b101101usize;
        let eq_low = EqPolynomial::evals(&r[..peeled_bits]);
        let offset_low = offset & (inner_len - 1);

        let blocks: Vec<Vec<F>> = (0..outer_len)
            .map(|_| random_vec(&mut rng, inner_len))
            .collect();
        let carry_terms: Vec<[F; 2]> = blocks
            .iter()
            .map(|block| summarize_pow2_block_carries(&eq_low, offset_low, block))
            .collect();

        let got = eval_offset_eq_peeled_carry_terms(&r, offset, peeled_bits, &carry_terms);
        let expected = reference_pow2_peeled_blocks(&r, offset, &blocks);
        assert_eq!(got, expected);
    }

    #[test]
    fn pow2_peel_matches_reference_with_high_overflow() {
        let mut rng = StdRng::seed_from_u64(0xAD);
        let peeled_bits = 2usize;
        let inner_len = 1usize << peeled_bits;
        let outer_len = 6usize;
        let r = random_vec(&mut rng, 5);
        let offset = 27usize;
        let eq_low = EqPolynomial::evals(&r[..peeled_bits]);
        let offset_low = offset & (inner_len - 1);

        let blocks: Vec<Vec<F>> = (0..outer_len)
            .map(|_| random_vec(&mut rng, inner_len))
            .collect();
        let carry_terms: Vec<[F; 2]> = blocks
            .iter()
            .map(|block| summarize_pow2_block_carries(&eq_low, offset_low, block))
            .collect();

        let got = eval_offset_eq_peeled_carry_terms(&r, offset, peeled_bits, &carry_terms);
        let expected = reference_pow2_peeled_blocks(&r, offset, &blocks);
        assert_eq!(got, expected);
    }
}
