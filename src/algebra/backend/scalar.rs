//! Default scalar backend wiring existing CRT+NTT kernels.

use super::traits::{CrtReconstruct, NttPrimeOps, NttTransform};
use crate::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use crate::algebra::ntt::crt::QData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime};
use crate::algebra::ring::CrtNttConvertibleField;

/// Default scalar backend implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarBackend;

impl<const D: usize> NttPrimeOps<D> for ScalarBackend {
    #[inline]
    fn from_canonical(prime: NttPrime, value: i16) -> MontCoeff {
        prime.from_canonical(value)
    }

    #[inline]
    fn to_canonical(prime: NttPrime, value: MontCoeff) -> i16 {
        prime.to_canonical(value)
    }

    #[inline]
    fn reduce(prime: NttPrime, value: MontCoeff) -> MontCoeff {
        prime.reduce(value)
    }

    #[inline]
    fn pointwise_mul(
        prime: NttPrime,
        out: &mut [MontCoeff; D],
        lhs: &[MontCoeff; D],
        rhs: &[MontCoeff; D],
    ) {
        prime.pointwise_mul(out, lhs, rhs);
    }
}

impl<const D: usize> NttTransform<D> for ScalarBackend {
    #[inline]
    fn forward_ntt(limb: &mut [MontCoeff; D], prime: NttPrime, twiddles: &NttTwiddles<D>) {
        forward_ntt(limb, prime, twiddles);
    }

    #[inline]
    fn inverse_ntt(limb: &mut [MontCoeff; D], prime: NttPrime, twiddles: &NttTwiddles<D>) {
        inverse_ntt(limb, prime, twiddles);
    }
}

impl<const K: usize, const D: usize, const L: usize> CrtReconstruct<K, D, L> for ScalarBackend {
    fn reconstruct<F: CrtNttConvertibleField>(
        primes: &[NttPrime; K],
        canonical_limbs: &[[i16; D]; K],
        qdata: &QData<K, L>,
    ) -> [F; D] {
        let q = qdata.q_u128().expect("q must fit in u128");
        let prime_moduli: [u128; K] = std::array::from_fn(|k| {
            u128::try_from(primes[k].p).expect("CRT prime modulus must be positive")
        });
        let big_p = prime_moduli.iter().fold(1u128, |acc, p| {
            acc.checked_mul(*p)
                .expect("product of CRT primes must fit u128")
        });
        let crt_m: [u128; K] = std::array::from_fn(|k| big_p / prime_moduli[k]);
        let crt_inv: [u16; K] = std::array::from_fn(|k| {
            let inv = mod_inverse(
                (crt_m[k] % prime_moduli[k]) as i128,
                prime_moduli[k] as i128,
            );
            u16::try_from(inv).expect("CRT inverse must fit u16 for small-prime backend")
        });

        let mut coeffs = [F::zero(); D];
        for (d, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc: u128 = 0;
            for k in 0..K {
                let ck_i16 = canonical_limbs[k][d];
                debug_assert!(ck_i16 >= 0 && ck_i16 < primes[k].p);
                let ck = u16::try_from(ck_i16).expect("canonical residue must fit u16");

                // Multiply by tiny residues/inverses (<= 15 bits) in fixed-time
                // loops, then accumulate modulo P.
                let term = mul_mod_by_small_u16(crt_m[k], ck, big_p);
                let term = mul_mod_by_small_u16(term, crt_inv[k], big_p);
                acc = add_mod_u128(acc, term, big_p);
            }

            // Final projection into [0, q).
            *coeff = F::from_q_residue_u128(acc % q);
        }

        coeffs
    }
}

fn mod_inverse(a: i128, modulus: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (modulus, ((a % modulus) + modulus) % modulus);

    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }

    assert_eq!(r, 1, "CRT inverse does not exist");
    (t % modulus + modulus) % modulus
}

#[inline]
fn add_mod_u128(a: u128, b: u128, modulus: u128) -> u128 {
    debug_assert!(a < modulus);
    debug_assert!(b < modulus);

    let (sum_lo, carry) = a.overflowing_add(b);
    let hi = carry as u128;
    let (sub_lo, borrow) = sum_lo.overflowing_sub(modulus);
    let sum_ge_modulus = (!borrow) as u128;
    let should_sub = hi | sum_ge_modulus;
    let mask = should_sub.wrapping_neg();
    (sum_lo & !mask) | (sub_lo & mask)
}

#[inline]
fn mul_mod_by_small_u16(a: u128, b: u16, modulus: u128) -> u128 {
    debug_assert!(a < modulus);
    let mut acc = 0u128;
    let mut cur = a;
    for i in 0..16 {
        let candidate = add_mod_u128(acc, cur, modulus);
        let bit = ((b >> i) & 1) as u128;
        let mask = bit.wrapping_neg();
        acc = (acc & !mask) | (candidate & mask);
        cur = add_mod_u128(cur, cur, modulus);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::{add_mod_u128, mul_mod_by_small_u16};

    #[test]
    fn add_mod_matches_native_when_sum_fits_u128() {
        let modulus = (1u128 << 100) - 159;
        for i in 0..4096u128 {
            let a = (i * 104_729 + 17) % modulus;
            let b = (i * 130_363 + 31) % modulus;
            let expected = (a + b) % modulus;
            assert_eq!(add_mod_u128(a, b, modulus), expected);
        }
    }

    #[test]
    fn mul_mod_small_matches_native_when_product_fits_u128() {
        let modulus = (1u128 << 100) - 159;
        for i in 0..4096u128 {
            let a = (i * 786_433 + 19) % modulus;
            let b = ((i * 97 + 7) & 0xFFFF) as u16;
            let expected = (a * (b as u128)) % modulus;
            assert_eq!(mul_mod_by_small_u16(a, b, modulus), expected);
        }
    }
}
