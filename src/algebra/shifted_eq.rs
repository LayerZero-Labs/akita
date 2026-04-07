//! Shifted equality evaluators for contiguous slices of the eq table.
//!
//! These helpers evaluate sums of the form
//!
//! `sum_y eq(r_addr, offset + y) * weights(y)`
//!
//! without materializing the full `eq(r_addr, ·)` table.
//!
//! All bit and index conventions are little-endian, matching
//! [`crate::algebra::eq_poly::EqPolynomial`]:
//! bit `k` of an integer index corresponds to `r_addr[k]`.

#![allow(dead_code)]

use crate::FieldCore;

/// Weight descriptions supported by [`shifted_eq_dp`].
pub(crate) enum ShiftedEqWeights<'a, E: FieldCore> {
    /// Product weights in little-endian bit order.
    ///
    /// Entry `bit_weights[k] = (w_0, w_1)` means the local contribution is
    /// `w_0` when `y_k = 0` and `w_1` when `y_k = 1`.
    Tensor(&'a [(E, E)]),
    /// Equality weights `weights(y) = eq(point, y)` in little-endian order.
    EqPoint(&'a [E]),
    /// Dense weights `weights[y]` in little-endian index order.
    Dense(&'a [E]),
}

/// Evaluate a shifted equality slice against one of the supported weight forms.
pub(crate) fn shifted_eq_dp<E: FieldCore>(
    address_point: &[E],
    offset: usize,
    weights: ShiftedEqWeights<'_, E>,
) -> E {
    match weights {
        ShiftedEqWeights::Tensor(bit_weights) => {
            shifted_eq_tensor_sum(address_point, offset, bit_weights)
        }
        ShiftedEqWeights::EqPoint(point) => shifted_eq_eq_point(address_point, offset, point),
        ShiftedEqWeights::Dense(weights) => shifted_eq_dense_sum(address_point, offset, weights),
    }
}

/// Evaluate
/// `sum_{y in {0,1}^m} eq(address_point, offset + y) * tensor_weight(y)`
/// in `O(address_bits)` time using the 2-state carry DP from the Jolt appendix.
pub(crate) fn shifted_eq_tensor_sum<E: FieldCore>(
    address_point: &[E],
    offset: usize,
    bit_weights: &[(E, E)],
) -> E {
    shifted_eq_tensor_sum_with(address_point, offset, bit_weights.len(), |bit_idx| {
        bit_weights[bit_idx]
    })
}

/// Evaluate
/// `sum_{y in {0,1}^m} eq(address_point, offset + y) * eq(point, y)`
/// in `O(address_bits)` time.
pub(crate) fn shifted_eq_eq_point<E: FieldCore>(
    address_point: &[E],
    offset: usize,
    point: &[E],
) -> E {
    shifted_eq_tensor_sum_with(address_point, offset, point.len(), |bit_idx| {
        let r_y = point[bit_idx];
        (E::one() - r_y, r_y)
    })
}

/// Evaluate
/// `sum_{0 <= y < weights.len()} eq(address_point, offset + y) * weights[y]`.
///
/// This uses a linear-time carry DP over the dense weight table. The table is
/// zero-padded to the next power of two, then folded in place across the local
/// `y` bits, after which the remaining high address bits are processed with the
/// same 2-state carry machine.
pub(crate) fn shifted_eq_dense_sum<E: FieldCore>(
    address_point: &[E],
    offset: usize,
    weights: &[E],
) -> E {
    if weights.is_empty() {
        return E::zero();
    }

    assert_dense_segment_in_range(address_point.len(), offset, weights.len());

    let padded_len = weights.len().next_power_of_two();
    let local_bits = padded_len.trailing_zeros() as usize;

    let mut no_carry = vec![E::zero(); padded_len];
    no_carry[..weights.len()].copy_from_slice(weights);
    let mut carry = vec![E::zero(); padded_len];
    let mut current_len = padded_len;

    for (bit_idx, &r_bit) in address_point.iter().take(local_bits).enumerate() {
        let k0 = E::one() - r_bit;
        let k1 = r_bit;
        let offset_bit = bit_is_set(offset, bit_idx);
        let next_len = current_len / 2;

        for j in 0..next_len {
            let a0 = no_carry[2 * j];
            let a1 = no_carry[2 * j + 1];
            let b0 = carry[2 * j];
            let b1 = carry[2 * j + 1];

            if offset_bit {
                no_carry[j] = a0 * k1;
                carry[j] = a1 * k0 + b0 * k0 + b1 * k1;
            } else {
                no_carry[j] = a0 * k0 + a1 * k1 + b0 * k1;
                carry[j] = b1 * k0;
            }
        }

        current_len = next_len;
    }

    debug_assert_eq!(current_len, 1);

    let mut no_carry_acc = no_carry[0];
    let mut carry_acc = carry[0];

    for (bit_idx, &r_bit) in address_point.iter().enumerate().skip(local_bits) {
        let k0 = E::one() - r_bit;
        let k1 = r_bit;
        let next_no_carry;
        let next_carry;

        if bit_is_set(offset, bit_idx) {
            next_no_carry = no_carry_acc * k1;
            next_carry = carry_acc * k0;
        } else {
            next_no_carry = no_carry_acc * k0 + carry_acc * k1;
            next_carry = E::zero();
        }

        no_carry_acc = next_no_carry;
        carry_acc = next_carry;
    }

    debug_assert!(
        carry_acc.is_zero(),
        "shifted-eq dense segment overflowed address width"
    );
    no_carry_acc
}

fn shifted_eq_tensor_sum_with<E, W>(
    address_point: &[E],
    offset: usize,
    num_local_bits: usize,
    mut weight_for_bit: W,
) -> E
where
    E: FieldCore,
    W: FnMut(usize) -> (E, E),
{
    assert_tensor_segment_in_range(address_point.len(), offset, num_local_bits);

    let mut no_carry = E::one();
    let mut carry = E::zero();

    for (bit_idx, &r_bit) in address_point.iter().enumerate() {
        let k0 = E::one() - r_bit;
        let k1 = r_bit;
        let next_no_carry;
        let next_carry;

        if bit_idx < num_local_bits {
            let (w0, w1) = weight_for_bit(bit_idx);
            if bit_is_set(offset, bit_idx) {
                next_no_carry = no_carry * k1 * w0;
                next_carry = no_carry * k0 * w1 + carry * k0 * w0 + carry * k1 * w1;
            } else {
                next_no_carry = no_carry * k0 * w0 + no_carry * k1 * w1 + carry * k1 * w0;
                next_carry = carry * k0 * w1;
            }
        } else if bit_is_set(offset, bit_idx) {
            next_no_carry = no_carry * k1;
            next_carry = carry * k0;
        } else {
            next_no_carry = no_carry * k0 + carry * k1;
            next_carry = E::zero();
        }

        no_carry = next_no_carry;
        carry = next_carry;
    }

    debug_assert!(
        carry.is_zero(),
        "shifted-eq tensor segment overflowed address width"
    );
    no_carry
}

fn bit_is_set(value: usize, bit_idx: usize) -> bool {
    if bit_idx >= usize::BITS as usize {
        false
    } else {
        ((value >> bit_idx) & 1) == 1
    }
}

fn assert_tensor_segment_in_range(address_bits: usize, offset: usize, num_local_bits: usize) {
    assert!(
        num_local_bits <= address_bits,
        "shifted-eq tensor segment uses more local bits than the address space"
    );

    if address_bits >= usize::BITS as usize {
        assert!(
            num_local_bits < usize::BITS as usize,
            "shifted-eq tensor segment width exceeds usize-backed offsets"
        );
        return;
    }

    let domain_len = 1usize << address_bits;
    let segment_len = 1usize
        .checked_shl(num_local_bits as u32)
        .expect("shifted-eq tensor segment width exceeds usize-backed offsets");
    assert!(
        offset
            .checked_add(segment_len)
            .is_some_and(|end| end <= domain_len),
        "shifted-eq tensor segment exceeds the address domain"
    );
}

fn assert_dense_segment_in_range(address_bits: usize, offset: usize, segment_len: usize) {
    assert!(
        segment_len > 0,
        "dense shifted-eq segment must be non-empty"
    );

    if address_bits >= usize::BITS as usize {
        return;
    }

    let domain_len = 1usize << address_bits;
    assert!(
        offset
            .checked_add(segment_len)
            .is_some_and(|end| end <= domain_len),
        "shifted-eq dense segment exceeds the address domain"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::eq_poly::EqPolynomial;
    use crate::algebra::Prime128Offset275;
    use crate::{FieldSampling, FromSmallInt};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    type F = Prime128Offset275;

    fn direct_dense_sum(address_point: &[F], offset: usize, weights: &[F]) -> F {
        let eq_table = EqPolynomial::evals(address_point);
        weights
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (idx, &weight)| {
                acc + eq_table[offset + idx] * weight
            })
    }

    fn dense_weights_from_tensor(bit_weights: &[(F, F)]) -> Vec<F> {
        let len = 1usize << bit_weights.len();
        (0..len)
            .map(|idx| {
                bit_weights
                    .iter()
                    .enumerate()
                    .fold(F::one(), |acc, (bit_idx, &(w0, w1))| {
                        acc * if bit_is_set(idx, bit_idx) { w1 } else { w0 }
                    })
            })
            .collect()
    }

    #[test]
    fn tensor_dp_matches_direct_dense_sum() {
        let mut rng = StdRng::seed_from_u64(0x51F7_0001);
        for address_bits in 1..8 {
            let address_point: Vec<F> = (0..address_bits).map(|_| F::sample(&mut rng)).collect();
            for local_bits in 0..=address_bits {
                let bit_weights: Vec<(F, F)> = (0..local_bits)
                    .map(|_| (F::sample(&mut rng), F::sample(&mut rng)))
                    .collect();
                let dense_weights = dense_weights_from_tensor(&bit_weights);
                let segment_len = dense_weights.len();
                let max_offset = (1usize << address_bits) - segment_len;
                let offset = rng.gen_range(0..=max_offset);

                let expected = direct_dense_sum(&address_point, offset, &dense_weights);
                let actual = shifted_eq_tensor_sum(&address_point, offset, &bit_weights);
                assert_eq!(
                    actual, expected,
                    "address_bits={address_bits} local_bits={local_bits} offset={offset}"
                );
            }
        }
    }

    #[test]
    fn eq_point_dp_matches_direct_dense_sum() {
        let mut rng = StdRng::seed_from_u64(0x51F7_0002);
        for address_bits in 1..8 {
            let address_point: Vec<F> = (0..address_bits).map(|_| F::sample(&mut rng)).collect();
            for local_bits in 0..=address_bits {
                let point: Vec<F> = (0..local_bits).map(|_| F::sample(&mut rng)).collect();
                let dense_weights = EqPolynomial::evals(&point);
                let segment_len = dense_weights.len();
                let max_offset = (1usize << address_bits) - segment_len;
                let offset = rng.gen_range(0..=max_offset);

                let expected = direct_dense_sum(&address_point, offset, &dense_weights);
                let actual = shifted_eq_eq_point(&address_point, offset, &point);
                assert_eq!(
                    actual, expected,
                    "address_bits={address_bits} local_bits={local_bits} offset={offset}"
                );
            }
        }
    }

    #[test]
    fn dense_dp_matches_direct_sum_for_arbitrary_lengths() {
        let mut rng = StdRng::seed_from_u64(0x51F7_0003);
        for address_bits in 1..8 {
            let address_point: Vec<F> = (0..address_bits).map(|_| F::sample(&mut rng)).collect();
            let domain_len = 1usize << address_bits;
            for len in 1..=domain_len {
                let max_offset = domain_len - len;
                let offset = rng.gen_range(0..=max_offset);
                let weights: Vec<F> = (0..len).map(|_| F::sample(&mut rng)).collect();

                let expected = direct_dense_sum(&address_point, offset, &weights);
                let actual = shifted_eq_dense_sum(&address_point, offset, &weights);
                assert_eq!(
                    actual, expected,
                    "address_bits={address_bits} len={len} offset={offset}"
                );
            }
        }
    }

    #[test]
    fn dispatcher_routes_all_weight_forms() {
        let address_point = vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        let tensor_weights = vec![
            (F::from_u64(11), F::from_u64(13)),
            (F::from_u64(17), F::from_u64(19)),
        ];
        let dense_weights = vec![F::from_u64(23), F::from_u64(29), F::from_u64(31)];
        let eq_point = vec![F::from_u64(37), F::from_u64(41)];

        assert_eq!(
            shifted_eq_dp(&address_point, 1, ShiftedEqWeights::Tensor(&tensor_weights)),
            shifted_eq_tensor_sum(&address_point, 1, &tensor_weights)
        );
        assert_eq!(
            shifted_eq_dp(&address_point, 2, ShiftedEqWeights::Dense(&dense_weights)),
            shifted_eq_dense_sum(&address_point, 2, &dense_weights)
        );
        assert_eq!(
            shifted_eq_dp(&address_point, 0, ShiftedEqWeights::EqPoint(&eq_point)),
            shifted_eq_eq_point(&address_point, 0, &eq_point)
        );
    }

    #[test]
    fn empty_dense_segment_returns_zero() {
        let address_point = vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        assert_eq!(shifted_eq_dense_sum(&address_point, 0, &[]), F::zero());
    }
}
