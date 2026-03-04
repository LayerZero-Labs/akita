//! CRT+NTT-domain representation of cyclotomic ring elements.

use std::array::from_fn;

use crate::algebra::backend::{CrtReconstruct, NttPrimeOps, NttTransform, ScalarBackend};
use crate::algebra::ntt::butterfly::{
    forward_ntt, forward_ntt_cyclic, inverse_ntt_cyclic, NttTwiddles,
};
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrtNttParamSet<W: PrimeWidth, const K: usize, const D: usize> {
    /// CRT primes with Montgomery constants.
    pub primes: [NttPrime<W>; K],
    /// Per-prime twiddle tables for forward/inverse NTT.
    pub twiddles: [NttTwiddles<W, D>; K],
    /// Garner reconstruction constants for CRT lift-back.
    pub garner: GarnerData<W, K>,
}

/// Precomputed Montgomery forms for small balanced digit values.
///
/// Covers the full `{-8, ..., 7}` range (16 entries per CRT prime),
/// which is sufficient for any `log_basis <= 4`. Storing the Montgomery
/// representation eliminates one `from_canonical` (a Montgomery multiply)
/// per coefficient in the `from_i8` hot path.
#[derive(Debug, Clone)]
pub struct DigitMontLut<W: PrimeWidth, const K: usize> {
    vals: [[MontCoeff<W>; 16]; K],
}

const DIGIT_LUT_HALF_B: i16 = 8;

impl<W: PrimeWidth, const K: usize> DigitMontLut<W, K> {
    /// Build the lookup table from CRT primes.
    ///
    /// Covers digit values in `{-8, ..., 7}` (balanced representation for
    /// `log_basis <= 4`).
    pub fn new<const D: usize>(params: &CrtNttParamSet<W, K, D>) -> Self {
        let mut vals = [[MontCoeff::from_raw(W::default()); 16]; K];
        for (k, prime) in params.primes.iter().enumerate() {
            for v_idx in 0..16u8 {
                let v = v_idx as i64 - DIGIT_LUT_HALF_B as i64;
                vals[k][v_idx as usize] = prime.from_canonical(W::from_i64(v));
            }
        }
        Self { vals }
    }

    /// Look up the Montgomery form of a balanced digit for CRT prime `k`.
    #[inline(always)]
    pub fn get(&self, k: usize, digit: i8) -> MontCoeff<W> {
        unsafe {
            *self
                .vals
                .get_unchecked(k)
                .get_unchecked((digit as i16 + DIGIT_LUT_HALF_B) as usize)
        }
    }
}

impl<W: PrimeWidth, const K: usize, const D: usize> CrtNttParamSet<W, K, D> {
    /// Build a full parameter set from CRT primes.
    ///
    /// Computes per-prime twiddles and Garner reconstruction constants.
    pub fn new(primes: [NttPrime<W>; K]) -> Self {
        let twiddles = from_fn(|k| NttTwiddles::compute(primes[k]));
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
        let centered_coeffs: [i128; D] = from_fn(|i| {
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
            let p = prime.p.to_i64();
            let p_u64 = p as u64;
            let r64 = ((1u128 << 64) % p_u64 as u128) as i64;
            let half_p = p / 2;
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                let c = *centered;
                let lo = (c as u64 % p_u64) as i64;
                let hi = ((c >> 64) as i64).rem_euclid(p);
                let mut r = (lo + hi * r64) % p;
                if r >= half_p {
                    r -= p;
                }
                *dst = B::from_canonical(*prime, W::from_i64(r));
            }
            B::forward_ntt(limb, *prime, tw);
        }
        Self { limbs }
    }

    /// Convert small integer coefficients (e.g. gadget digits) into
    /// negacyclic CRT+NTT domain, bypassing Fp128 centering entirely.
    pub fn from_i8_with_params(digits: &[i8; D], params: &CrtNttParamSet<W, K, D>) -> Self {
        Self::from_i8_negacyclic_backend::<ScalarBackend>(digits, params)
    }

    /// Like [`Self::from_i8_with_params`] but uses a precomputed
    /// [`DigitMontLut`] to replace per-coefficient `from_canonical`
    /// (Montgomery multiply) with a table lookup.
    #[inline]
    pub fn from_i8_with_lut(
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, (limb, tw)) in limbs.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &d) in limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, d);
            }
            forward_ntt(limb, params.primes[k], tw);
        }
        Self { limbs }
    }

    /// Like [`Self::from_i8_cyclic`] but uses a precomputed [`DigitMontLut`].
    #[inline]
    pub fn from_i8_cyclic_with_lut(
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, (limb, tw)) in limbs.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &d) in limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, d);
            }
            forward_ntt_cyclic(limb, params.primes[k], tw);
        }
        Self { limbs }
    }

    fn from_i8_negacyclic_backend<B: NttPrimeOps<W, D> + NttTransform<W, D>>(
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            for (dst, &d) in limb.iter_mut().zip(digits.iter()) {
                *dst = B::from_canonical(*prime, W::from_i64(d as i64));
            }
            B::forward_ntt(limb, *prime, tw);
        }
        Self { limbs }
    }

    /// Convert small integer coefficients into cyclic CRT+NTT domain,
    /// bypassing Fp128 centering entirely.
    pub fn from_i8_cyclic(digits: &[i8; D], params: &CrtNttParamSet<W, K, D>) -> Self {
        Self::from_i8_cyclic_backend::<ScalarBackend>(digits, params)
    }

    fn from_i8_cyclic_backend<B: NttPrimeOps<W, D>>(
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            for (dst, &d) in limb.iter_mut().zip(digits.iter()) {
                *dst = B::from_canonical(*prime, W::from_i64(d as i64));
            }
            forward_ntt_cyclic(limb, *prime, tw);
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

    /// Convert a coefficient-form ring element into CRT+**cyclic** NTT domain.
    ///
    /// Evaluates at D-th roots of unity (X^D - 1) instead of X^D + 1.
    /// Used together with `to_ring_cyclic` to compute unreduced polynomial products.
    pub fn from_ring_cyclic<F: CrtNttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        Self::from_ring_cyclic_with_backend::<F, ScalarBackend>(ring, params)
    }

    /// Convert a coefficient-form ring element into CRT+**cyclic** NTT domain
    /// through an explicit backend.
    pub fn from_ring_cyclic_with_backend<F: CrtNttConvertibleField, B: NttPrimeOps<W, D>>(
        ring: &CyclotomicRing<F, D>,
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let centered_coeffs: [i128; D] = from_fn(|i| {
            let canonical = ring.coeffs[i].to_canonical_u128();
            if canonical > half_q {
                -((q - canonical) as i128)
            } else {
                canonical as i128
            }
        });

        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            let p = prime.p.to_i64();
            let p_u64 = p as u64;
            let r64 = ((1u128 << 64) % p_u64 as u128) as i64;
            let half_p = p / 2;
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                let c = *centered;
                let lo = (c as u64 % p_u64) as i64;
                let hi = ((c >> 64) as i64).rem_euclid(p);
                let mut r = (lo + hi * r64) % p;
                if r >= half_p {
                    r -= p;
                }
                *dst = B::from_canonical(*prime, W::from_i64(r));
            }
            forward_ntt_cyclic(limb, *prime, tw);
        }
        Self { limbs }
    }

    /// Convert from CRT+**cyclic** NTT domain back to coefficient form.
    ///
    /// Inverse of `from_ring_cyclic`: applies inverse cyclic NTT then CRT reconstruction.
    pub fn to_ring_cyclic<F: CrtNttConvertibleField>(
        &self,
        params: &CrtNttParamSet<W, K, D>,
    ) -> CyclotomicRing<F, D> {
        self.to_ring_cyclic_with_backend::<F, ScalarBackend>(params)
    }

    /// Convert from CRT+**cyclic** NTT domain back to coefficient form
    /// through an explicit backend.
    pub fn to_ring_cyclic_with_backend<
        F: CrtNttConvertibleField,
        B: NttPrimeOps<W, D> + CrtReconstruct<W, K, D>,
    >(
        &self,
        params: &CrtNttParamSet<W, K, D>,
    ) -> CyclotomicRing<F, D> {
        let mut canonical = [[W::default(); D]; K];
        for (k, ((can, prime), tw)) in canonical
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            inverse_ntt_cyclic(&mut limb, *prime, tw);
            for (dst, src) in can.iter_mut().zip(limb.iter()) {
                let canon = B::to_canonical(*prime, *src);
                *dst = prime.center(canon);
            }
        }
        let coeffs = B::reconstruct::<F>(&params.primes, &canonical, &params.garner);
        CyclotomicRing::from_coefficients(coeffs)
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
