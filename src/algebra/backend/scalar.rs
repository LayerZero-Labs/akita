//! Default scalar backend: delegates to NTT kernels and uses Garner's
//! algorithm for CRT reconstruction.

use super::traits::{CrtReconstruct, NttPrimeOps, NttTransform};
use crate::algebra::ntt::butterfly::{forward_ntt, inverse_ntt, NttTwiddles};
use crate::algebra::ntt::crt::GarnerData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
use crate::algebra::ring::CrtNttConvertibleField;

/// Default scalar backend implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarBackend;

impl<W: PrimeWidth, const D: usize> NttPrimeOps<W, D> for ScalarBackend {
    #[inline]
    fn from_canonical(prime: NttPrime<W>, value: W) -> MontCoeff<W> {
        prime.from_canonical(value)
    }

    #[inline]
    fn to_canonical(prime: NttPrime<W>, value: MontCoeff<W>) -> W {
        prime.to_canonical(value)
    }

    #[inline]
    fn reduce_range(prime: NttPrime<W>, value: MontCoeff<W>) -> MontCoeff<W> {
        prime.reduce_range(value)
    }

    #[inline]
    fn pointwise_mul(
        prime: NttPrime<W>,
        out: &mut [MontCoeff<W>; D],
        lhs: &[MontCoeff<W>; D],
        rhs: &[MontCoeff<W>; D],
    ) {
        prime.pointwise_mul(out, lhs, rhs);
    }
}

impl<W: PrimeWidth, const D: usize> NttTransform<W, D> for ScalarBackend {
    #[inline]
    fn forward_ntt(limb: &mut [MontCoeff<W>; D], prime: NttPrime<W>, twiddles: &NttTwiddles<W, D>) {
        forward_ntt(limb, prime, twiddles);
    }

    #[inline]
    fn inverse_ntt(limb: &mut [MontCoeff<W>; D], prime: NttPrime<W>, twiddles: &NttTwiddles<W, D>) {
        inverse_ntt(limb, prime, twiddles);
    }
}

impl<W: PrimeWidth, const K: usize, const D: usize> CrtReconstruct<W, K, D> for ScalarBackend {
    fn reconstruct<F: CrtNttConvertibleField>(
        primes: &[NttPrime<W>; K],
        canonical: &[[W; D]; K],
        garner: &GarnerData<W, K>,
    ) -> [F; D] {
        let mut coeffs = [F::zero(); D];
        for (d, coeff) in coeffs.iter_mut().enumerate() {
            // Garner mixed-radix decomposition (all arithmetic in i64, mod p_i).
            let mut v = [0i64; K];
            v[0] = canonical[0][d].to_i64();
            for i in 1..K {
                let pi = primes[i].p.to_i64();
                let mut temp = canonical[i][d].to_i64();
                #[allow(clippy::needless_range_loop)]
                for j in 0..i {
                    temp -= v[j];
                    temp = ((temp % pi) + pi) % pi;
                    temp = (temp * garner.gamma[i][j].to_i64()) % pi;
                }
                // Center the mixed-radix digit to keep the final reconstruction
                // in a small signed range when inputs are centered.
                if temp > pi / 2 {
                    temp -= pi;
                }
                v[i] = temp;
            }

            // Horner accumulation in the target field F.
            let mut result = F::from_i64(v[0]);
            let mut partial_prod = F::from_i64(primes[0].p.to_i64());
            for i in 1..K {
                result += F::from_i64(v[i]) * partial_prod;
                if i + 1 < K {
                    partial_prod = partial_prod * F::from_i64(primes[i].p.to_i64());
                }
            }
            *coeff = result;
        }
        coeffs
    }
}
