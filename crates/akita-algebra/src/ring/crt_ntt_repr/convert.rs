use std::array::from_fn;

use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt, inverse_ntt_cyclic};
use crate::ntt::prime::{MontCoeff, PrimeWidth};
use crate::ring::cyclotomic::CyclotomicRing;

use super::lut::{CenteredPrimeReducer, CenteredPrimeWideReducer};
use super::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
};

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    pub(super) fn centered_coefficients_with_params(
        &self,
        params: &CrtNttParamSet<W, K, D>,
    ) -> [[W; D]; K] {
        let mut canonical = [[W::default(); D]; K];
        for (k, ((coefficients, prime), twiddles)) in canonical
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            inverse_ntt(&mut limb, *prime, twiddles, params.kernel_plan);
            for (dst, src) in coefficients.iter_mut().zip(limb.iter()) {
                *dst = prime.center(prime.to_canonical(*src));
            }
        }
        canonical
    }

    /// Convert a coefficient-form ring element into CRT+NTT domain.
    pub fn from_ring<F: CrtNttConvertibleField>(
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
            let reducer = CenteredPrimeWideReducer::new(*prime);
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                *dst = prime.from_canonical(reducer.reduce_i128(*centered));
            }
            forward_ntt(limb, *prime, tw, params.kernel_plan);
        }
        Self { limbs }
    }

    /// Convert a coefficient-form ring element into both negacyclic and cyclic
    /// CRT+NTT domains, sharing coefficient centering and CRT reduction.
    pub fn from_ring_pair_with_params<F: CrtNttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        params: &CrtNttParamSet<W, K, D>,
    ) -> (Self, Self) {
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

        let mut neg_limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        let mut cyc_limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (((neg_limb, cyc_limb), prime), tw) in neg_limbs
            .iter_mut()
            .zip(cyc_limbs.iter_mut())
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            let reducer = CenteredPrimeWideReducer::new(*prime);
            for (dst, centered) in neg_limb.iter_mut().zip(centered_coeffs.iter()) {
                *dst = prime.from_canonical(reducer.reduce_i128(*centered));
            }
            *cyc_limb = *neg_limb;
            forward_ntt(neg_limb, *prime, tw, params.kernel_plan);
            forward_ntt_cyclic(cyc_limb, *prime, tw, params.kernel_plan);
        }
        (Self { limbs: neg_limbs }, Self { limbs: cyc_limbs })
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
            let reducer = CenteredPrimeWideReducer::new(*prime);
            let mont_val = prime.from_canonical(reducer.reduce_i128(centered));
            limb.fill(mont_val);
        }
        Self { limbs }
    }

    /// Convert small integer coefficients (e.g. gadget digits) into
    /// negacyclic CRT+NTT domain, bypassing Fp128 centering entirely.
    pub fn from_i8_with_params(digits: &[i8; D], params: &CrtNttParamSet<W, K, D>) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            for (dst, &digit) in limb.iter_mut().zip(digits.iter()) {
                *dst = prime.from_canonical(W::from_i64(i64::from(digit)));
            }
            forward_ntt(limb, *prime, tw, params.kernel_plan);
        }
        Self { limbs }
    }

    /// Convert centered i32 coefficients into negacyclic CRT+NTT domain.
    pub fn from_centered_i32_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            let reducer = CenteredPrimeReducer::new(*prime);
            for (dst, &coefficient) in limb.iter_mut().zip(coeffs.iter()) {
                *dst = prime.from_canonical(reducer.reduce_i64(i64::from(coefficient)));
            }
            forward_ntt(limb, *prime, tw, params.kernel_plan);
        }
        Self { limbs }
    }

    /// Convert centered i32 coefficients into negacyclic CRT+NTT form using a
    /// precomputed Montgomery lookup table.
    pub fn from_centered_i32_with_lut(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &CenteredMontLut<W, K>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, ((limb, prime), tw)) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let reducer = CenteredPrimeReducer::new(*prime);
            for (dst, &coefficient) in limb.iter_mut().zip(coeffs) {
                *dst = lut.get(k, coefficient).unwrap_or_else(|| {
                    prime.from_canonical(reducer.reduce_i64(i64::from(coefficient)))
                });
            }
            forward_ntt(limb, *prime, tw, params.kernel_plan);
        }
        Self { limbs }
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
            forward_ntt(limb, params.primes[k], tw, params.kernel_plan);
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
            forward_ntt_cyclic(limb, params.primes[k], tw, params.kernel_plan);
        }
        Self { limbs }
    }

    /// Convert centered i32 coefficients into cyclic CRT+NTT domain.
    pub fn from_centered_i32_cyclic_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            let reducer = CenteredPrimeReducer::new(*prime);
            for (dst, &coefficient) in limb.iter_mut().zip(coeffs.iter()) {
                *dst = prime.from_canonical(reducer.reduce_i64(i64::from(coefficient)));
            }
            forward_ntt_cyclic(limb, *prime, tw, params.kernel_plan);
        }
        Self { limbs }
    }

    /// Convert centered i32 coefficients into both negacyclic and cyclic
    /// CRT+NTT domains while sharing the coefficient preparation step.
    pub fn from_centered_i32_pair_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> (Self, Self) {
        Self::from_centered_i32_pair(coeffs, params, None, false)
    }

    /// Like [`Self::from_centered_i32_pair_with_params`] but uses a precomputed
    /// [`CenteredMontLut`] for the coefficient-to-Montgomery conversion.
    pub fn from_centered_i32_pair_with_lut(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &CenteredMontLut<W, K>,
    ) -> (Self, Self) {
        Self::from_centered_i32_pair(coeffs, params, Some(lut), false)
    }

    /// Like [`Self::from_centered_i32_pair_with_lut`] for caller-validated
    /// centered coefficients.
    ///
    /// # Safety
    ///
    /// Every entry in `coeffs` must be within the range covered by `lut`.
    #[inline]
    pub unsafe fn from_centered_i32_pair_with_lut_unchecked(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &CenteredMontLut<W, K>,
    ) -> (Self, Self) {
        Self::from_centered_i32_pair(coeffs, params, Some(lut), true)
    }

    /// Convert small integer coefficients into cyclic CRT+NTT domain,
    /// bypassing Fp128 centering entirely.
    pub fn from_i8_cyclic(digits: &[i8; D], params: &CrtNttParamSet<W, K, D>) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            for (dst, &digit) in limb.iter_mut().zip(digits.iter()) {
                *dst = prime.from_canonical(W::from_i64(i64::from(digit)));
            }
            forward_ntt_cyclic(limb, *prime, tw, params.kernel_plan);
        }
        Self { limbs }
    }

    fn from_centered_i32_pair(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: Option<&CenteredMontLut<W, K>>,
        unchecked_lut: bool,
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
            let reducer = CenteredPrimeReducer::new(*prime);
            if let Some(lut) = lut {
                for (dst, &coeff) in neg_limb.iter_mut().zip(coeffs.iter()) {
                    *dst = if unchecked_lut {
                        unsafe { lut.get_unchecked(k, coeff) }
                    } else {
                        lut.get(k, coeff).unwrap_or_else(|| {
                            prime.from_canonical(reducer.reduce_i64(i64::from(coeff)))
                        })
                    };
                }
            } else {
                for (dst, &coeff) in neg_limb.iter_mut().zip(coeffs.iter()) {
                    *dst = prime.from_canonical(reducer.reduce_i64(i64::from(coeff)));
                }
            }
            *cyc_limb = *neg_limb;
            forward_ntt(neg_limb, *prime, tw, params.kernel_plan);
            forward_ntt_cyclic(cyc_limb, *prime, tw, params.kernel_plan);
        }
        (Self { limbs: neg_limbs }, Self { limbs: cyc_limbs })
    }

    /// Convert from CRT+NTT domain back to coefficient form.
    pub fn to_ring<F: CrtNttConvertibleField>(
        &self,
        params: &CrtNttParamSet<W, K, D>,
    ) -> CyclotomicRing<F, D> {
        let mut canonical = [[W::default(); D]; K];
        for (k, ((coefficients, prime), twiddles)) in canonical
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            inverse_ntt(&mut limb, *prime, twiddles, params.kernel_plan);
            for (dst, src) in coefficients.iter_mut().zip(limb.iter()) {
                *dst = prime.center(prime.to_canonical(*src));
            }
        }

        let coefficients = params.reconstruct::<F>(&canonical);
        CyclotomicRing::from_coefficients(coefficients)
    }

    /// Convert a coefficient-form ring element into CRT+**cyclic** NTT domain
    /// using a bundled parameter set.
    pub fn from_ring_cyclic<F: CrtNttConvertibleField>(
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
            let reducer = CenteredPrimeWideReducer::new(*prime);
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                *dst = prime.from_canonical(reducer.reduce_i128(*centered));
            }
            forward_ntt_cyclic(limb, *prime, tw, params.kernel_plan);
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
        let mut canonical = [[W::default(); D]; K];
        for (k, ((can, prime), tw)) in canonical
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
            .enumerate()
        {
            let mut limb = self.limbs[k];
            inverse_ntt_cyclic(&mut limb, *prime, tw, params.kernel_plan);
            for (dst, src) in can.iter_mut().zip(limb.iter()) {
                let canon = prime.to_canonical(*src);
                *dst = prime.center(canon);
            }
        }
        let coeffs = params.reconstruct::<F>(&canonical);
        CyclotomicRing::from_coefficients(coeffs)
    }
}
