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
        let q_i128 = i128::try_from(q).expect("q must fit in i128");
        let prime_moduli: [i128; K] = std::array::from_fn(|k| primes[k].p as i128);
        let big_p = prime_moduli.iter().fold(1i128, |acc, p| {
            acc.checked_mul(*p)
                .expect("product of CRT primes must fit i128")
        });
        let crt_m: [i128; K] = std::array::from_fn(|k| big_p / prime_moduli[k]);
        let crt_inv: [i128; K] =
            std::array::from_fn(|k| mod_inverse(crt_m[k] % prime_moduli[k], prime_moduli[k]));

        let mut coeffs = [F::zero(); D];
        for (d, coeff) in coeffs.iter_mut().enumerate() {
            let mut acc: i128 = 0;
            for k in 0..K {
                let ck = canonical_limbs[k][d] as i128;
                acc = (acc + ck * crt_m[k] * crt_inv[k]) % big_p;
            }
            let lifted = (acc % big_p + big_p) % big_p;
            let residue = ((lifted % q_i128) + q_i128) % q_i128;
            *coeff = F::from_q_residue_u128(residue as u128);
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
