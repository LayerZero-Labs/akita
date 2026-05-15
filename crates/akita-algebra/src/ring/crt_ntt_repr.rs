//! CRT+NTT-domain representation of cyclotomic ring elements.

use std::array::from_fn;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use std::mem::size_of;

use crate::backend::{CrtReconstruct, NttPrimeOps, NttTransform, ScalarBackend};
use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt_cyclic, NttTwiddles};
use crate::ntt::crt::GarnerData;
#[cfg(target_arch = "aarch64")]
use crate::ntt::neon;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
#[cfg(target_arch = "x86_64")]
use crate::ntt::x86;
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
    /// Per-prime NTT-domain Montgomery limbs.
    pub limbs: [[MontCoeff<W>; D]; K],
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
/// Covers the full `{-32, ..., 31}` range (64 entries per CRT prime),
/// which is sufficient for any `log_basis <= 6`. Storing the Montgomery
/// representation eliminates one `from_canonical` (a Montgomery multiply)
/// per coefficient in the `from_i8` hot path.
#[derive(Debug, Clone)]
pub struct DigitMontLut<W: PrimeWidth, const K: usize> {
    vals: [[MontCoeff<W>; 64]; K],
}

/// Precomputed Montgomery forms for centered integer coefficients in
/// `[-max_abs, max_abs]`.
#[derive(Debug, Clone)]
pub struct CenteredMontLut<W: PrimeWidth, const K: usize> {
    vals: [Vec<MontCoeff<W>>; K],
    offset: i32,
}

const DIGIT_LUT_HALF_B: i16 = 32;

impl<W: PrimeWidth, const K: usize> DigitMontLut<W, K> {
    /// Build the lookup table from CRT primes.
    ///
    /// Covers digit values in `{-32, ..., 31}` (balanced representation for
    /// `log_basis <= 6`).
    pub fn new<const D: usize>(params: &CrtNttParamSet<W, K, D>) -> Self {
        let mut vals = [[MontCoeff::from_raw(W::default()); 64]; K];
        for (k, prime) in params.primes.iter().enumerate() {
            for v_idx in 0..64u8 {
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

impl<W: PrimeWidth, const K: usize> CenteredMontLut<W, K> {
    /// Build a lookup table for all centered coefficients in `[-max_abs, max_abs]`.
    pub fn new<const D: usize>(params: &CrtNttParamSet<W, K, D>, max_abs: i32) -> Self {
        let vals = from_fn(|k| {
            let prime = params.primes[k];
            (-max_abs..=max_abs)
                .map(|v| prime.from_canonical(W::from_i64(v as i64)))
                .collect()
        });
        Self {
            vals,
            offset: max_abs,
        }
    }

    /// Look up the Montgomery form of a centered coefficient for CRT prime `k`.
    #[inline(always)]
    pub fn get(&self, k: usize, coeff: i32) -> MontCoeff<W> {
        unsafe {
            *self
                .vals
                .get_unchecked(k)
                .get_unchecked((coeff + self.offset) as usize)
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

    #[inline(always)]
    fn add_assign_pointwise_mul_limb(
        acc_limb: &mut [MontCoeff<W>; D],
        lhs_limb: &[MontCoeff<W>; D],
        rhs_limb: &[MontCoeff<W>; D],
        prime: NttPrime<W>,
    ) {
        let mut idx = 0usize;
        while idx + 4 <= D {
            for lane in 0..4 {
                let i = idx + lane;
                let prod = prime.mul(lhs_limb[i], rhs_limb[i]);
                acc_limb[i] = prime.add_reduce(acc_limb[i], prod);
            }
            idx += 4;
        }

        while idx < D {
            let prod = prime.mul(lhs_limb[idx], rhs_limb[idx]);
            acc_limb[idx] = prime.add_reduce(acc_limb[idx], prod);
            idx += 1;
        }
    }

    #[inline(always)]
    fn try_add_assign_pointwise_mul_limb_simd(
        acc_limb: &mut [MontCoeff<W>; D],
        lhs_limb: &[MontCoeff<W>; D],
        rhs_limb: &[MontCoeff<W>; D],
        prime: NttPrime<W>,
    ) -> bool {
        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    neon::pointwise_mul_acc_i32(
                        acc_limb.as_mut_ptr() as *mut i32,
                        lhs_limb.as_ptr() as *const i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    );
                } else {
                    neon::pointwise_mul_acc_i16(
                        acc_limb.as_mut_ptr() as *mut i16,
                        lhs_limb.as_ptr() as *const i16,
                        rhs_limb.as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                        prime.pinv.to_i64() as i16,
                    );
                }
            }
            return true;
        }

        #[cfg(target_arch = "x86_64")]
        if x86::use_x86_ntt() {
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    x86::pointwise_mul_acc_i32(
                        acc_limb.as_mut_ptr() as *mut i32,
                        lhs_limb.as_ptr() as *const i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    );
                } else {
                    x86::pointwise_mul_acc_i16(
                        acc_limb.as_mut_ptr() as *mut i16,
                        lhs_limb.as_ptr() as *const i16,
                        rhs_limb.as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                        prime.pinv.to_i64() as i16,
                    );
                }
            }
            return true;
        }

        false
    }

    #[inline(always)]
    fn try_add_assign_reduced_limb_simd(
        acc_limb: &mut [MontCoeff<W>; D],
        rhs_limb: &[MontCoeff<W>; D],
        prime: NttPrime<W>,
    ) -> bool {
        #[cfg(all(target_arch = "aarch64", feature = "parallel"))]
        if neon::use_neon_ntt() {
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    neon::add_reduce_i32(
                        acc_limb.as_mut_ptr() as *mut i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                    );
                } else {
                    neon::add_reduce_i16(
                        acc_limb.as_mut_ptr() as *mut i16,
                        rhs_limb.as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                    );
                }
            }
            return true;
        }

        #[cfg(all(target_arch = "x86_64", feature = "parallel"))]
        if x86::use_x86_ntt() {
            unsafe {
                if size_of::<W>() == size_of::<i32>() {
                    x86::add_reduce_i32(
                        acc_limb.as_mut_ptr() as *mut i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                    );
                } else {
                    x86::add_reduce_i16(
                        acc_limb.as_mut_ptr() as *mut i16,
                        rhs_limb.as_ptr() as *const i16,
                        D,
                        prime.p.to_i64() as i16,
                    );
                }
            }
            return true;
        }

        let _ = (acc_limb, rhs_limb, prime);
        false
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

    /// Convert a field scalar (constant polynomial) into CRT+NTT domain.
    ///
    /// A constant polynomial evaluates to the same value at every NTT point,
    /// so this broadcasts the reduced scalar to all `D` positions in each CRT
    /// limb — skipping the full forward NTT entirely.
    pub fn from_scalar_with_params<F: CrtNttConvertibleField>(
        scalar: &F,
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let q = (-F::one()).to_canonical_u128() + 1;
        let half_q = q / 2;
        let canonical = scalar.to_canonical_u128();
        let centered: i128 = if canonical > half_q {
            -((q - canonical) as i128)
        } else {
            canonical as i128
        };

        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (limb, prime) in limbs.iter_mut().zip(params.primes.iter()) {
            let p = prime.p.to_i64();
            let p_u64 = p as u64;
            let r64 = ((1u128 << 64) % p_u64 as u128) as i64;
            let half_p = p / 2;
            // `centered as u64` reinterprets via 2's complement. For negative
            // `centered`, the result is a large u64 whose `% p_u64` still
            // yields the correct unsigned residue because
            // `(-x) as u64 == 2^64 - x` and `(2^64 - x) mod p == (-x) mod p`
            // when `p` divides into 2^64 evenly in residue terms.
            let lo = (centered as u64 % p_u64) as i64;
            let hi = ((centered >> 64) as i64).rem_euclid(p);
            let mut r = (lo + hi * r64) % p;
            if r >= half_p {
                r -= p;
            }
            let mont_val = prime.from_canonical(W::from_i64(r));
            limb.fill(mont_val);
        }
        Self { limbs }
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

    /// Convert centered i32 coefficients into negacyclic CRT+NTT domain.
    pub fn from_centered_i32_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        Self::from_centered_i32_negacyclic_backend::<ScalarBackend>(coeffs, params)
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

    /// Accumulate `lhs * rhs(digits)` into `self` while reusing caller-owned
    /// scratch storage for the digit CRT+NTT conversion.
    #[inline]
    pub fn add_assign_pointwise_mul_i8_with_lut_scratch(
        &mut self,
        lhs: &Self,
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        scratch: &mut [[MontCoeff<W>; D]; K],
    ) {
        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw);

            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let lhs_limb = &lhs.limbs[k];
            if Self::try_add_assign_pointwise_mul_limb_simd(acc_limb, lhs_limb, scratch_limb, prime)
            {
                continue;
            }
            Self::add_assign_pointwise_mul_limb(acc_limb, lhs_limb, scratch_limb, prime);
        }
    }

    /// Accumulate `lhs0 * rhs(digits)` and `lhs1 * rhs(digits)` into
    /// `(acc0, acc1)` while sharing the digit CRT+NTT conversion scratch.
    #[inline]
    pub fn add_assign_pointwise_mul_i8_pair_with_lut_scratch(
        accs: [&mut Self; 2],
        lhs: [&Self; 2],
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        scratch: &mut [[MontCoeff<W>; D]; K],
    ) {
        let [acc0, acc1] = accs;
        let [lhs0, lhs1] = lhs;

        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw);

            let prime = params.primes[k];
            let acc0_limb = &mut acc0.limbs[k];
            let acc1_limb = &mut acc1.limbs[k];
            let lhs0_limb = &lhs0.limbs[k];
            let lhs1_limb = &lhs1.limbs[k];
            if Self::try_add_assign_pointwise_mul_limb_simd(
                acc0_limb,
                lhs0_limb,
                scratch_limb,
                prime,
            ) {
                let used_simd = Self::try_add_assign_pointwise_mul_limb_simd(
                    acc1_limb,
                    lhs1_limb,
                    scratch_limb,
                    prime,
                );
                debug_assert!(used_simd);
                continue;
            }
            for (((acc0_coeff, lhs0_coeff), acc1_coeff), (lhs1_coeff, rhs_coeff)) in acc0_limb
                .iter_mut()
                .zip(lhs0_limb.iter())
                .zip(acc1_limb.iter_mut())
                .zip(lhs1_limb.iter().zip(scratch_limb.iter()))
            {
                let prod0 = prime.mul(*lhs0_coeff, *rhs_coeff);
                *acc0_coeff = prime.add_reduce(*acc0_coeff, prod0);

                let prod1 = prime.mul(*lhs1_coeff, *rhs_coeff);
                *acc1_coeff = prime.add_reduce(*acc1_coeff, prod1);
            }
        }
    }

    /// Accumulate `lhs0 * rhs(digits)`, `lhs1 * rhs(digits)`, and
    /// `lhs2 * rhs(digits)` into `(acc0, acc1, acc2)` while sharing the digit
    /// CRT+NTT conversion scratch.
    #[inline]
    pub fn add_assign_pointwise_mul_i8_triple_with_lut_scratch(
        accs: [&mut Self; 3],
        lhs: [&Self; 3],
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        scratch: &mut [[MontCoeff<W>; D]; K],
    ) {
        let [acc0, acc1, acc2] = accs;
        let [lhs0, lhs1, lhs2] = lhs;

        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw);

            let prime = params.primes[k];
            let acc0_limb = &mut acc0.limbs[k];
            let acc1_limb = &mut acc1.limbs[k];
            let acc2_limb = &mut acc2.limbs[k];
            let lhs0_limb = &lhs0.limbs[k];
            let lhs1_limb = &lhs1.limbs[k];
            let lhs2_limb = &lhs2.limbs[k];
            if Self::try_add_assign_pointwise_mul_limb_simd(
                acc0_limb,
                lhs0_limb,
                scratch_limb,
                prime,
            ) {
                let used_simd = Self::try_add_assign_pointwise_mul_limb_simd(
                    acc1_limb,
                    lhs1_limb,
                    scratch_limb,
                    prime,
                ) && Self::try_add_assign_pointwise_mul_limb_simd(
                    acc2_limb,
                    lhs2_limb,
                    scratch_limb,
                    prime,
                );
                debug_assert!(used_simd);
                continue;
            }
            for idx in 0..D {
                let rhs_coeff = scratch_limb[idx];

                let prod0 = prime.mul(lhs0_limb[idx], rhs_coeff);
                acc0_limb[idx] = prime.add_reduce(acc0_limb[idx], prod0);

                let prod1 = prime.mul(lhs1_limb[idx], rhs_coeff);
                acc1_limb[idx] = prime.add_reduce(acc1_limb[idx], prod1);

                let prod2 = prime.mul(lhs2_limb[idx], rhs_coeff);
                acc2_limb[idx] = prime.add_reduce(acc2_limb[idx], prod2);
            }
        }
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

    /// Convert centered i32 coefficients into cyclic CRT+NTT domain.
    pub fn from_centered_i32_cyclic_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        Self::from_centered_i32_cyclic_backend::<ScalarBackend>(coeffs, params)
    }

    /// Convert centered i32 coefficients into both negacyclic and cyclic
    /// CRT+NTT domains while sharing the coefficient preparation step.
    pub fn from_centered_i32_pair_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> (Self, Self) {
        Self::from_centered_i32_pair_backend::<ScalarBackend>(coeffs, params, None)
    }

    /// Like [`Self::from_centered_i32_pair_with_params`] but uses a precomputed
    /// [`CenteredMontLut`] for the coefficient-to-Montgomery conversion.
    pub fn from_centered_i32_pair_with_lut(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &CenteredMontLut<W, K>,
    ) -> (Self, Self) {
        Self::from_centered_i32_pair_backend::<ScalarBackend>(coeffs, params, Some(lut))
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

    fn from_centered_i32_negacyclic_backend<B: NttPrimeOps<W, D> + NttTransform<W, D>>(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            let p = prime.p.to_i64();
            let half_p = p / 2;
            for (dst, &coeff) in limb.iter_mut().zip(coeffs.iter()) {
                let mut r = (coeff as i64).rem_euclid(p);
                if r >= half_p {
                    r -= p;
                }
                *dst = B::from_canonical(*prime, W::from_i64(r));
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

    fn from_centered_i32_cyclic_backend<B: NttPrimeOps<W, D>>(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            let p = prime.p.to_i64();
            let half_p = p / 2;
            for (dst, &coeff) in limb.iter_mut().zip(coeffs.iter()) {
                let mut r = (coeff as i64).rem_euclid(p);
                if r >= half_p {
                    r -= p;
                }
                *dst = B::from_canonical(*prime, W::from_i64(r));
            }
            forward_ntt_cyclic(limb, *prime, tw);
        }
        Self { limbs }
    }

    fn from_centered_i32_pair_backend<B: NttPrimeOps<W, D>>(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: Option<&CenteredMontLut<W, K>>,
    ) -> (Self, Self) {
        let mut neg_limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        let mut cyc_limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, (((neg_limb, cyc_limb), prime), tw)) in neg_limbs
            .iter_mut()
            .zip(cyc_limbs.iter_mut())
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            if let Some(lut) = lut {
                for (dst, &coeff) in neg_limb.iter_mut().zip(coeffs.iter()) {
                    *dst = lut.get(k, coeff);
                }
            } else {
                let p = prime.p.to_i64();
                let half_p = p / 2;
                for (dst, &coeff) in neg_limb.iter_mut().zip(coeffs.iter()) {
                    let mut r = (coeff as i64).rem_euclid(p);
                    if r >= half_p {
                        r -= p;
                    }
                    *dst = B::from_canonical(*prime, W::from_i64(r));
                }
            }
            *cyc_limb = *neg_limb;
            forward_ntt(neg_limb, *prime, tw);
            forward_ntt_cyclic(cyc_limb, *prime, tw);
        }
        (Self { limbs: neg_limbs }, Self { limbs: cyc_limbs })
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
        let mut out = self.clone();
        out.add_assign_reduced_with_params(rhs, params);
        out
    }

    /// Add another CRT+NTT element into `self` and reduce each coefficient.
    ///
    /// Uses the architecture-specific SIMD add-reduce kernel when available.
    pub fn add_assign_reduced_with_params(&mut self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) {
        for k in 0..K {
            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let rhs_limb = &rhs.limbs[k];
            if Self::try_add_assign_reduced_limb_simd(acc_limb, rhs_limb, prime) {
                continue;
            }
            for (a, b) in acc_limb.iter_mut().zip(rhs_limb.iter()) {
                *a = prime.add_reduce(*a, *b);
            }
        }
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
                *a = B::add_reduce(prime, *a, *b);
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
                *a = B::sub_reduce(prime, *a, *b);
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
                *a = B::neg_reduce(prime, *a);
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
    #[inline(always)]
    pub fn pointwise_mul_with_params(&self, rhs: &Self, params: &CrtNttParamSet<W, K, D>) -> Self {
        self.pointwise_mul(rhs, &params.primes)
    }

    /// Accumulate `lhs * rhs` into `self` in CRT+NTT domain.
    ///
    /// On supported architectures, this uses a fused SIMD
    /// pointwise-multiply-accumulate kernel when available; otherwise it falls
    /// back to the scalar loop.
    #[inline(always)]
    pub fn add_assign_pointwise_mul_with_params(
        &mut self,
        lhs: &Self,
        rhs: &Self,
        params: &CrtNttParamSet<W, K, D>,
    ) {
        for k in 0..K {
            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let lhs_limb = &lhs.limbs[k];
            let rhs_limb = &rhs.limbs[k];
            if Self::try_add_assign_pointwise_mul_limb_simd(acc_limb, lhs_limb, rhs_limb, prime) {
                continue;
            }
            Self::add_assign_pointwise_mul_limb(acc_limb, lhs_limb, rhs_limb, prime);
        }
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
