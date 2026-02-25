//! CRT+NTT-domain representation of cyclotomic ring elements.

use crate::algebra::backend::{CrtReconstruct, NttPrimeOps, NttTransform, ScalarBackend};
use crate::algebra::fields::{Fp128, Fp32, Fp64, SolinasFp128, SolinasParams};
use crate::algebra::ntt::butterfly::NttTwiddles;
use crate::algebra::ntt::crt::QData;
use crate::algebra::ntt::prime::{MontCoeff, NttPrime};
use crate::{CanonicalField, FieldCore};

use super::cyclotomic::CyclotomicRing;

/// CRT+NTT-domain representation of a cyclotomic ring element.
///
/// Stores `K` arrays of `D` [`MontCoeff`] values, one per CRT prime.
/// Multiplication is pointwise per prime - O(K*D) vs O(D^2) for coefficient form.
///
/// Use [`CyclotomicCrtNtt::from_ring`] and [`CyclotomicCrtNtt::to_ring`] to convert
/// between coefficient and CRT+NTT domains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclotomicCrtNtt<const K: usize, const D: usize> {
    pub(crate) limbs: [[MontCoeff; D]; K],
}

/// Field types that can safely convert to and from the CRT+NTT representation.
pub trait CrtNttConvertibleField: FieldCore + CanonicalField {
    /// Reduce this field element modulo a small prime `p`.
    fn mod_small_prime(self, p: i16) -> i16;

    /// Reconstruct from a residue in `[0, q)` after CRT combination.
    fn from_q_residue_u128(x: u128) -> Self;
}

impl<const MODULUS: u32> CrtNttConvertibleField for Fp32<MODULUS> {
    #[inline]
    fn mod_small_prime(self, p: i16) -> i16 {
        (self.to_canonical_u32() % (p as u32)) as i16
    }

    #[inline]
    fn from_q_residue_u128(x: u128) -> Self {
        let narrowed = u32::try_from(x).expect("CRT residue does not fit in u32");
        Self::from_u64(narrowed as u64)
    }
}

impl<const MODULUS: u64> CrtNttConvertibleField for Fp64<MODULUS> {
    #[inline]
    fn mod_small_prime(self, p: i16) -> i16 {
        (self.to_canonical_u64() % (p as u64)) as i16
    }

    #[inline]
    fn from_q_residue_u128(x: u128) -> Self {
        let narrowed = u64::try_from(x).expect("CRT residue does not fit in u64");
        Self::from_u64(narrowed)
    }
}

impl<const MODULUS: u128> CrtNttConvertibleField for Fp128<MODULUS> {
    #[inline]
    fn mod_small_prime(self, p: i16) -> i16 {
        (self.to_canonical_u128() % (p as u128)) as i16
    }

    #[inline]
    fn from_q_residue_u128(x: u128) -> Self {
        Self::from_canonical_u128_reduced(x)
    }
}

impl<M: SolinasParams> CrtNttConvertibleField for SolinasFp128<M> {
    #[inline]
    fn mod_small_prime(self, p: i16) -> i16 {
        (self.to_canonical_u128() % (p as u128)) as i16
    }

    #[inline]
    fn from_q_residue_u128(x: u128) -> Self {
        Self::from_canonical_u128_reduced(x)
    }
}

impl<const K: usize, const D: usize> CyclotomicCrtNtt<K, D> {
    /// The additive identity (all zeros in every CRT limb).
    pub fn zero() -> Self {
        Self {
            limbs: [[MontCoeff::from_raw(0); D]; K],
        }
    }

    /// Convert a coefficient-form ring element into CRT+NTT domain
    /// using the default scalar backend.
    pub fn from_ring<F: CrtNttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
    ) -> Self {
        Self::from_ring_with_backend::<F, ScalarBackend>(ring, primes, twiddles)
    }

    /// Convert a coefficient-form ring element into CRT+NTT domain
    /// through an explicit backend implementation.
    pub fn from_ring_with_backend<
        F: CrtNttConvertibleField,
        B: NttPrimeOps<D> + NttTransform<D>,
    >(
        ring: &CyclotomicRing<F, D>,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(0); D]; K];
        for ((limb, prime), tw) in limbs.iter_mut().zip(primes.iter()).zip(twiddles.iter()) {
            // Reduce each coefficient mod p and convert to Montgomery form.
            for (dst, src) in limb.iter_mut().zip(ring.coeffs.iter()) {
                let val = src.mod_small_prime(prime.p);
                *dst = B::from_canonical(*prime, val);
            }
            B::forward_ntt(limb, *prime, tw);
        }
        Self { limbs }
    }

    /// Convert from CRT+NTT domain back to coefficient form
    /// using the default scalar backend.
    pub fn to_ring<F: CrtNttConvertibleField, const L: usize>(
        &self,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
        qdata: &QData<K, L>,
    ) -> CyclotomicRing<F, D> {
        self.to_ring_with_backend::<F, ScalarBackend, L>(primes, twiddles, qdata)
    }

    /// Convert from CRT+NTT domain back to coefficient form
    /// through an explicit backend implementation.
    pub fn to_ring_with_backend<
        F: CrtNttConvertibleField,
        B: NttPrimeOps<D> + NttTransform<D> + CrtReconstruct<K, D, L>,
        const L: usize,
    >(
        &self,
        primes: &[NttPrime; K],
        twiddles: &[NttTwiddles<D>; K],
        qdata: &QData<K, L>,
    ) -> CyclotomicRing<F, D> {
        let mut canonical = [[0i16; D]; K];
        for (k, ((can, prime), tw)) in canonical
            .iter_mut()
            .zip(primes.iter())
            .zip(twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            B::inverse_ntt(&mut limb, *prime, tw);
            for (dst, src) in can.iter_mut().zip(limb.iter()) {
                *dst = B::to_canonical(*prime, *src);
            }
        }

        let coeffs = B::reconstruct::<F>(primes, &canonical, qdata);

        CyclotomicRing::from_coefficients(coeffs)
    }

    /// Add another CRT+NTT element and reduce each coefficient with the matching
    /// prime to maintain valid Montgomery ranges using the scalar backend.
    pub fn add_reduced(&self, rhs: &Self, primes: &[NttPrime; K]) -> Self {
        self.add_reduced_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Add another CRT+NTT element and reduce each coefficient with the matching
    /// prime through an explicit backend implementation.
    pub fn add_reduced_with_backend<B: NttPrimeOps<D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let sum = MontCoeff::from_raw(a.raw().wrapping_add(b.raw()));
                *a = B::reduce(prime, sum);
            }
        }
        out
    }

    /// Subtract another CRT+NTT element and reduce each coefficient with the
    /// matching prime to maintain valid Montgomery ranges using the scalar backend.
    pub fn sub_reduced(&self, rhs: &Self, primes: &[NttPrime; K]) -> Self {
        self.sub_reduced_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Subtract another CRT+NTT element and reduce each coefficient with the
    /// matching prime through an explicit backend implementation.
    pub fn sub_reduced_with_backend<B: NttPrimeOps<D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime; K],
    ) -> Self {
        let mut out = self.clone();
        for (k, (limb, rhs_limb)) in out.limbs.iter_mut().zip(rhs.limbs.iter()).enumerate() {
            let prime = primes[k];
            for (a, b) in limb.iter_mut().zip(rhs_limb.iter()) {
                let diff = MontCoeff::from_raw(a.raw().wrapping_sub(b.raw()));
                *a = B::reduce(prime, diff);
            }
        }
        out
    }

    /// Negate each CRT+NTT coefficient and reduce with the matching prime
    /// using the scalar backend.
    pub fn neg_reduced(&self, primes: &[NttPrime; K]) -> Self {
        self.neg_reduced_with_backend::<ScalarBackend>(primes)
    }

    /// Negate each CRT+NTT coefficient and reduce with the matching prime
    /// through an explicit backend implementation.
    pub fn neg_reduced_with_backend<B: NttPrimeOps<D>>(&self, primes: &[NttPrime; K]) -> Self {
        let mut out = self.clone();
        for (k, limb) in out.limbs.iter_mut().enumerate() {
            let prime = primes[k];
            for a in limb.iter_mut() {
                let neg = MontCoeff::from_raw(a.raw().wrapping_neg());
                *a = B::reduce(prime, neg);
            }
        }
        out
    }

    /// Pointwise multiplication in CRT+NTT domain.
    ///
    /// Each limb is multiplied pointwise using the corresponding prime's
    /// Montgomery multiplication.
    pub fn pointwise_mul(&self, rhs: &Self, primes: &[NttPrime; K]) -> Self {
        self.pointwise_mul_with_backend::<ScalarBackend>(rhs, primes)
    }

    /// Pointwise multiplication in CRT+NTT domain through an explicit backend implementation.
    pub fn pointwise_mul_with_backend<B: NttPrimeOps<D>>(
        &self,
        rhs: &Self,
        primes: &[NttPrime; K],
    ) -> Self {
        let mut out = [[MontCoeff::from_raw(0); D]; K];
        for (k, ((o, a), b)) in out
            .iter_mut()
            .zip(self.limbs.iter())
            .zip(rhs.limbs.iter())
            .enumerate()
        {
            B::pointwise_mul(primes[k], o, a, b);
        }
        Self { limbs: out }
    }
}
