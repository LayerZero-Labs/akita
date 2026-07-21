use std::array::from_fn;

use crate::{CanonicalField, CyclotomicRing, FieldCore};

use super::{CrtNttParamSet, CyclotomicCrtNtt};

/// CRT parameters with an i32 prefix and one i16 exactness tail prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedCrtNttParamSet<const K: usize, const D: usize> {
    /// Existing 30-bit CRT profile.
    pub wide: CrtNttParamSet<i32, K, D>,
    /// One 14-bit tail prime, materialized only for schedules that require it.
    pub tail: CrtNttParamSet<i16, 1, D>,
    tail_residue_weight: i64,
    tail_digit_weights: [i64; K],
}

impl<const K: usize, const D: usize> MixedCrtNttParamSet<K, D> {
    /// Extend an existing i32 profile by one coprime i16 prime.
    #[must_use]
    pub fn new(wide: CrtNttParamSet<i32, K, D>, tail: CrtNttParamSet<i16, 1, D>) -> Self {
        let tail_modulus = i64::from(tail.primes[0].p);
        let tail_gamma: [i64; K] = from_fn(|i| {
            mod_inverse_i64(
                i64::from(wide.primes[i].p).rem_euclid(tail_modulus),
                tail_modulus,
            )
        });
        // The final mixed-radix digit is affine in the tail residue and the
        // already reconstructed wide digits. Precompute that linear form so
        // reconstruction needs one reduction, not K dependent reductions.
        let mut tail_residue_weight = 1;
        let mut tail_digit_weights = [0; K];
        for i in (0..K).rev() {
            tail_residue_weight = (tail_residue_weight * tail_gamma[i]) % tail_modulus;
            tail_digit_weights[i] = (-tail_residue_weight).rem_euclid(tail_modulus);
        }
        Self {
            wide,
            tail,
            tail_residue_weight,
            tail_digit_weights,
        }
    }
}

/// NTT representation whose last CRT residue uses a 14-bit prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixedCrtNtt<const K: usize, const D: usize> {
    /// Existing i32 residues.
    pub wide: CyclotomicCrtNtt<i32, K, D>,
    /// Additional i16 residue.
    pub tail: CyclotomicCrtNtt<i16, 1, D>,
}

impl<const K: usize, const D: usize> MixedCrtNtt<K, D> {
    /// Additive identity in both residue families.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            wide: CyclotomicCrtNtt::zero(),
            tail: CyclotomicCrtNtt::zero(),
        }
    }

    /// Convert one coefficient-form ring into the mixed NTT representation.
    #[must_use]
    pub fn from_ring<F: FieldCore + CanonicalField>(
        ring: &CyclotomicRing<F, D>,
        params: &MixedCrtNttParamSet<K, D>,
    ) -> Self {
        Self {
            wide: CyclotomicCrtNtt::from_ring_with_params(ring, &params.wide),
            tail: CyclotomicCrtNtt::from_ring_with_params(ring, &params.tail),
        }
    }

    /// Convert a signed i16 coefficient vector into both residue families.
    #[must_use]
    pub fn from_i16(coefficients: &[i16; D], params: &MixedCrtNttParamSet<K, D>) -> Self {
        let wide_coefficients = coefficients.map(i32::from);
        Self {
            wide: CyclotomicCrtNtt::from_centered_i32_with_params(&wide_coefficients, &params.wide),
            tail: CyclotomicCrtNtt::from_centered_i32_with_params(&wide_coefficients, &params.tail),
        }
    }

    /// Accumulate one pointwise product in both residue families.
    pub fn add_assign_pointwise_mul(
        &mut self,
        lhs: &Self,
        rhs: &Self,
        params: &MixedCrtNttParamSet<K, D>,
    ) {
        self.wide
            .add_assign_pointwise_mul_with_params(&lhs.wide, &rhs.wide, &params.wide);
        self.tail
            .add_assign_pointwise_mul_with_params(&lhs.tail, &rhs.tail, &params.tail);
    }

    /// Invert both NTT families and reconstruct directly in the target field.
    #[must_use]
    pub fn to_ring<F: FieldCore + CanonicalField>(
        &self,
        params: &MixedCrtNttParamSet<K, D>,
    ) -> CyclotomicRing<F, D> {
        let wide = self.wide.centered_coefficients_with_params(&params.wide);
        let tail = self.tail.centered_coefficients_with_params(&params.tail);
        let tail_modulus = i64::from(params.tail.primes[0].p);

        let coefficients = from_fn(|d| {
            let mut digits = [0i64; K];
            if K != 0 {
                digits[0] = i64::from(wide[0][d]);
            }
            for i in 1..K {
                let modulus = i64::from(params.wide.primes[i].p);
                let mut value = i64::from(wide[i][d]);
                for (j, digit) in digits[..i].iter().enumerate() {
                    value = (value - digit).rem_euclid(modulus);
                    value = (value * i64::from(params.wide.garner.gamma[i][j])) % modulus;
                }
                if value > modulus / 2 {
                    value -= modulus;
                }
                digits[i] = value;
            }

            let tail_digit = i64::from(tail[0][d]) * params.tail_residue_weight
                + digits
                    .iter()
                    .zip(params.tail_digit_weights)
                    .map(|(digit, weight)| digit * weight)
                    .sum::<i64>();
            let mut tail_digit = tail_digit.rem_euclid(tail_modulus);
            if tail_digit > tail_modulus / 2 {
                tail_digit -= tail_modulus;
            }

            let mut result = F::zero();
            let mut product = F::one();
            for (digit, prime) in digits.iter().zip(params.wide.primes.iter()) {
                result += F::from_i64(*digit) * product;
                product *= F::from_i64(i64::from(prime.p));
            }
            result + F::from_i64(tail_digit) * product
        });
        CyclotomicRing::from_coefficients(coefficients)
    }
}

fn mod_inverse_i64(a: i64, modulus: i64) -> i64 {
    let (mut old_r, mut r) = (a, modulus);
    let (mut old_s, mut s) = (1i64, 0i64);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }
    debug_assert_eq!(old_r.abs(), 1);
    old_s.rem_euclid(modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::tables::{I16_TAIL_PRIME, Q64_NUM_PRIMES, Q64_PRIMES};
    use akita_field::{Fp64, Prime64Offset59};

    #[test]
    fn mixed_i16_matvec_handles_signed_boundaries() {
        const D: usize = 64;
        type F = Fp64<18446744073709551557>;
        let params = MixedCrtNttParamSet::new(
            CrtNttParamSet::<i32, Q64_NUM_PRIMES, D>::new(Q64_PRIMES),
            CrtNttParamSet::<i16, 1, D>::new([I16_TAIL_PRIME]),
        );
        let lhs_coefficients = from_fn(|i| F::from_i64((i as i64 % 17) - 8));
        let lhs = CyclotomicRing::<F, D>::from_coefficients(lhs_coefficients);
        let digits = from_fn(|i| match i % 4 {
            0 => -1024,
            1 => -1,
            2 => 0,
            _ => 1023,
        });
        let rhs_coefficients = digits.map(|digit| F::from_i64(i64::from(digit)));
        let rhs = CyclotomicRing::<F, D>::from_coefficients(rhs_coefficients);

        let lhs_ntt = MixedCrtNtt::from_ring(&lhs, &params);
        let rhs_ntt = MixedCrtNtt::from_i16(&digits, &params);
        let mut product = MixedCrtNtt::zero();
        product.add_assign_pointwise_mul(&lhs_ntt, &rhs_ntt, &params);

        assert_eq!(product.to_ring::<Prime64Offset59>(&params), lhs * rhs);
    }
}
