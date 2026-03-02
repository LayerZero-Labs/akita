//! CRT+NTT-domain representation of cyclotomic ring elements.

use crate::algebra::backend::{CrtReconstruct, NttPrimeOps, NttTransform, ScalarBackend};
use crate::algebra::ntt::butterfly::NttTwiddles;
use crate::algebra::ntt::crt::GarnerData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
use crate::{CanonicalField, FieldCore};

use super::cyclotomic::CyclotomicRing;

/// CRT+NTT-domain representation of a cyclotomic ring element.
///
/// Stores `K` arrays of `D` [`MontCoeff<W>`] values, one per CRT prime.
/// Multiplication is pointwise per prime — O(K*D) vs O(D^2) for coefficient form.
///
/// Generic over:
/// - `W: PrimeWidth` — integer width (`i16` or `i32`)
/// - `K` — number of CRT primes
/// - `D` — polynomial degree
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclotomicCrtNtt<W: PrimeWidth, const K: usize, const D: usize> {
    pub(crate) limbs: [[MontCoeff<W>; D]; K],
}

/// Field types that can convert to/from the CRT+NTT representation.
///
/// Blanket-implemented for all `FieldCore + CanonicalField` types.
pub trait CrtNttConvertibleField: FieldCore + CanonicalField {}

impl<F: FieldCore + CanonicalField> CrtNttConvertibleField for F {}

/// Bundled CRT+NTT parameters for a fixed width/prime-count/degree tuple.
///
/// Keeps primes/twiddles/Garner constants consistent and avoids passing them
/// independently at every call site.
pub struct CrtNttParamSet<W: PrimeWidth, const K: usize, const D: usize> {
    /// CRT primes with Montgomery constants.
    pub primes: [NttPrime<W>; K],
    /// Per-prime twiddle tables for forward/inverse NTT.
    pub twiddles: [NttTwiddles<W, D>; K],
    /// Garner reconstruction constants for CRT lift-back.
    pub garner: GarnerData<W, K>,
}

impl<W: PrimeWidth, const K: usize, const D: usize> CrtNttParamSet<W, K, D> {
    /// Build a full parameter set from CRT primes.
    ///
    /// Computes per-prime twiddles and Garner reconstruction constants.
    pub fn new(primes: [NttPrime<W>; K]) -> Self {
        let twiddles = std::array::from_fn(|k| NttTwiddles::compute(primes[k]));
        let garner = GarnerData::compute(&primes);
        Self {
            primes,
            twiddles,
            garner,
        }
    }
}

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    /// The additive identity (all zeros in every CRT limb).
    pub fn zero() -> Self {
        Self {
            limbs: [[MontCoeff::from_raw(W::default()); D]; K],
        }
    }

    /// Convert a coefficient-form ring element into CRT+NTT domain
    /// using the default scalar backend.
    pub fn from_ring<F: CrtNttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        primes: &[NttPrime<W>; K],
        twiddles: &[NttTwiddles<W, D>; K],
    ) -> Self {
        Self::from_ring_with_backend::<F, ScalarBackend>(ring, primes, twiddles)
    }

    /// Convert a coefficient-form ring element into CRT+NTT domain
    /// using a bundled parameter set and the scalar backend.
    pub fn from_ring_with_params<F: CrtNttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        Self::from_ring(ring, &params.primes, &params.twiddles)
    }

    /// Convert a coefficient-form ring element into CRT+NTT domain
    /// through an explicit backend implementation.
    pub fn from_ring_with_backend<
        F: CrtNttConvertibleField,
        B: NttPrimeOps<W, D> + NttTransform<W, D>,
    >(
        ring: &CyclotomicRing<F, D>,
        primes: &[NttPrime<W>; K],
        twiddles: &[NttTwiddles<W, D>; K],
    ) -> Self {
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let centered_coeffs: [i128; D] = std::array::from_fn(|i| {
            let canonical = ring.coeffs[i].to_canonical_u128();
            if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            }
        });

        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs.iter_mut().zip(primes.iter()).zip(twiddles.iter()) {
            // Interpret coefficients in centered form (-q/2, q/2] before reducing
            // into the CRT primes. This makes the reduction map consistent with
            // negacyclic subtraction (which naturally produces negative values).
            let p = prime.p.to_i64() as i128;
            let half_p = p / 2;
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                let mut r = *centered % p;
                if r < 0 {
                    r += p;
                }
                // Center residues into [-p/2, p/2) for stable signed arithmetic.
                if r >= half_p {
                    r -= p;
                }
                *dst = B::from_canonical(*prime, W::from_i64(r as i64));
            }
            B::forward_ntt(limb, *prime, tw);
        }
        Self { limbs }
    }

    /// Convert from CRT+NTT domain back to coefficient form
    /// using the default scalar backend.
    pub fn to_ring<F: CrtNttConvertibleField>(
        &self,
        primes: &[NttPrime<W>; K],
        twiddles: &[NttTwiddles<W, D>; K],
        garner: &GarnerData<W, K>,
    ) -> CyclotomicRing<F, D> {
        self.to_ring_with_backend::<F, ScalarBackend>(primes, twiddles, garner)
    }

    /// Convert from CRT+NTT domain back to coefficient form
    /// using a bundled parameter set and the scalar backend.
    pub fn to_ring_with_params<F: CrtNttConvertibleField>(
        &self,
        params: &CrtNttParamSet<W, K, D>,
    ) -> CyclotomicRing<F, D> {
        self.to_ring(&params.primes, &params.twiddles, &params.garner)
    }

    /// Convert from CRT+NTT domain back to coefficient form
    /// through an explicit backend implementation.
    pub fn to_ring_with_backend<
        F: CrtNttConvertibleField,
        B: NttPrimeOps<W, D> + NttTransform<W, D> + CrtReconstruct<W, K, D>,
    >(
        &self,
        primes: &[NttPrime<W>; K],
        twiddles: &[NttTwiddles<W, D>; K],
        garner: &GarnerData<W, K>,
    ) -> CyclotomicRing<F, D> {
        let mut canonical = [[W::default(); D]; K];
        for (k, ((can, prime), tw)) in canonical
            .iter_mut()
            .zip(primes.iter())
            .zip(twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            B::inverse_ntt(&mut limb, *prime, tw);
            for (dst, src) in can.iter_mut().zip(limb.iter()) {
                let canon = B::to_canonical(*prime, *src);
                *dst = prime.center(canon);
            }
        }

        let coeffs = B::reconstruct::<F>(primes, &canonical, garner);

        CyclotomicRing::from_coefficients(coeffs)
    }

    /// Add another CRT+NTT element and reduce each coefficient with the matching
    /// prime to maintain valid Montgomery ranges using the scalar backend.
    pub fn add_reduced(&self, rhs: &Self, primes: &[NttPrime<W>; K]) -> Self {
        self.add_reduced_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Add another CRT+NTT element and reduce using a bundled parameter set.
    pub fn add_reduced_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.add_reduced(rhs, &params.primes)
    }

    /// Add another CRT+NTT element and reduce each coefficient with the matching
    /// prime through an explicit backend implementation.
    pub fn add_reduced_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let sum = MontCoeff::from_raw(a.raw().wrapping_add(b.raw()));
                *a = B::reduce_range(prime, sum);
            }
        }
        out
    }

    /// Subtract another CRT+NTT element and reduce using the scalar backend.
    pub fn sub_reduced(&self, rhs: &Self, primes: &[NttPrime<W>; K]) -> Self {
        self.sub_reduced_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Subtract another CRT+NTT element and reduce using a bundled parameter set.
    pub fn sub_reduced_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.sub_reduced(rhs, &params.primes)
    }

    /// Subtract another CRT+NTT element and reduce through an explicit backend.
    pub fn sub_reduced_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let diff = MontCoeff::from_raw(a.raw().wrapping_sub(b.raw()));
                *a = B::reduce_range(prime, diff);
            }
        }
        out
    }

    /// Negate each CRT+NTT coefficient and reduce using the scalar backend.
    pub fn neg_reduced(&self, primes: &[NttPrime<W>; K]) -> Self {
        self.neg_reduced_with_backend::<ScalarBackend>(primes)
    }

    /// Negate each CRT+NTT coefficient and reduce using a bundled parameter set.
    pub fn neg_reduced_with_params(&self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.neg_reduced(&params.primes)
    }

    /// Negate each CRT+NTT coefficient and reduce through an explicit backend.
    pub fn neg_reduced_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, limb) in out.limbs.iter_mut().enumerate() {
            let prime = primes[k];
            for a in limb.iter_mut() {
                let neg = MontCoeff::from_raw(a.raw().wrapping_neg());
                *a = B::reduce_range(prime, neg);
            }
        }
        out
    }

    /// Pointwise multiplication in CRT+NTT domain using the scalar backend.
    pub fn pointwise_mul(&self, rhs: &Self, primes: &[NttPrime<W>; K]) -> Self {
        self.pointwise_mul_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Pointwise multiplication in CRT+NTT domain using a bundled parameter set.
    pub fn pointwise_mul_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.pointwise_mul(rhs, &params.primes)
    }

    /// Pointwise multiplication in CRT+NTT domain through an explicit backend.
    pub fn pointwise_mul_with_backend<B: NttPrimeOps<W, D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime<W>; K],
    ) -> Self {
        let mut out = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, ((o, a), b)) in out
            .iter_mut()
            .zip(self.limbs.iter())
            .zip(rhs.limbs.iter())
            .enumerate()
        {
            let prime = primes[k];
            B::pointwise_mul(prime, o, a, b);
            // Keep coefficients in a bounded range for subsequent inverse NTT.
            for c in o.iter_mut() {
                *c = B::reduce_range(prime, *c);
            }
        }
        Self { limbs: out }
    }

    /// Apply `sigma_{-1}` directly in NTT domain (`slot[j] -> slot[D-1-j]`).
    ///
    /// This is a pure index permutation per CRT limb and does not negate values.
    pub fn conjugation_automorphism_ntt(&self) -> Self {
        let limbs = std::array::from_fn(|k| {
            std::array::from_fn(|j| self.limbs[k][D.saturating_sub(1) - j])
        });
        Self { limbs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::fields::Fp64;
    use crate::algebra::ntt::tables::{Q32_NUM_PRIMES, Q32_PRIMES};
    use crate::FromSmallInt;

    type F = Fp64<4294967197>;
    const D: usize = 64;

    #[test]
    fn conjugation_automorphism_ntt_matches_coefficient_sigma_m1() {
        let params = CrtNttParamSet::<i16, Q32_NUM_PRIMES, D>::new(Q32_PRIMES);
        let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
            F::from_i64((i as i64 % 17) - 8)
        }));
        let ntt = CyclotomicCrtNtt::<i16, Q32_NUM_PRIMES, D>::from_ring_with_params(&ring, &params);
        let got = ntt
            .conjugation_automorphism_ntt()
            .to_ring_with_params::<F>(&params);
        assert_eq!(got, ring.sigma_m1());
    }

    #[test]
    fn conjugation_automorphism_ntt_is_involution() {
        let params = CrtNttParamSet::<i16, Q32_NUM_PRIMES, D>::new(Q32_PRIMES);
        let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|i| {
            F::from_i64((i as i64 % 11) - 5)
        }));
        let ntt = CyclotomicCrtNtt::<i16, Q32_NUM_PRIMES, D>::from_ring_with_params(&ring, &params);
        let roundtrip = ntt
            .conjugation_automorphism_ntt()
            .conjugation_automorphism_ntt();
        assert_eq!(roundtrip, ntt);
    }
}
