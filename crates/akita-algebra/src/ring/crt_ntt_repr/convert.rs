#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::avx;
use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt, inverse_ntt_cyclic};
use crate::ntt::field_limbs::{field_residues, FieldModulusLimbs, FIELD_CHUNK};
#[cfg(target_arch = "aarch64")]
use crate::ntt::neon;
use crate::ntt::prime::{MontCoeff, PrimeWidth};
use crate::ring::cyclotomic::CyclotomicRing;

use super::lut::CenteredPrimeReducer;
use super::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
};

enum CenteredI16NttStrategy<W: PrimeWidth, const K: usize> {
    Lut(CenteredMontLut<W, K>),
    #[cfg(target_arch = "aarch64")]
    NeonI16,
    #[cfg(target_arch = "aarch64")]
    NeonI32,
}

/// Prepared conversion policy for repeated centered-i16 NTT inputs.
pub(super) struct CenteredI16NttConverter<'a, W: PrimeWidth, const K: usize, const D: usize> {
    params: &'a CrtNttParamSet<W, K, D>,
    strategy: CenteredI16NttStrategy<W, K>,
}

impl<'a, W: PrimeWidth, const K: usize, const D: usize> CenteredI16NttConverter<'a, W, K, D> {
    pub(super) fn new(params: &'a CrtNttParamSet<W, K, D>, rhs: &[[i16; D]]) -> Self {
        #[cfg(target_arch = "aarch64")]
        if params.kernel_plan.uses_neon() {
            if size_of::<W>() == size_of::<i16>() {
                return Self {
                    params,
                    strategy: CenteredI16NttStrategy::NeonI16,
                };
            }
            if size_of::<W>() == size_of::<i32>() {
                return Self {
                    params,
                    strategy: CenteredI16NttStrategy::NeonI32,
                };
            }
        }

        let rhs_abs_bound = rhs
            .iter()
            .flatten()
            .map(|&digit| i32::from(digit).unsigned_abs())
            .max()
            .unwrap_or(0) as i32;
        Self {
            params,
            strategy: CenteredI16NttStrategy::Lut(CenteredMontLut::new(params, rhs_abs_bound)),
        }
    }

    pub(super) fn transform(&self, coefficients: &[i16; D]) -> CyclotomicCrtNtt<W, K, D> {
        match &self.strategy {
            CenteredI16NttStrategy::Lut(lut) => {
                let centered = coefficients.map(i32::from);
                CyclotomicCrtNtt::from_centered_i32_with_lut(&centered, self.params, lut)
            }
            #[cfg(target_arch = "aarch64")]
            CenteredI16NttStrategy::NeonI16 => self.transform_neon_i16(coefficients),
            #[cfg(target_arch = "aarch64")]
            CenteredI16NttStrategy::NeonI32 => self.transform_neon_i32(coefficients),
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn transform_neon_i16(&self, coefficients: &[i16; D]) -> CyclotomicCrtNtt<W, K, D> {
        debug_assert_eq!(size_of::<W>(), size_of::<i16>());
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), twiddles) in limbs
            .iter_mut()
            .zip(self.params.primes.iter())
            .zip(self.params.twiddles.iter())
        {
            // SAFETY: `new` selects this strategy only for the sealed i16
            // residue width. The source and destination arrays are disjoint.
            unsafe {
                neon::centered_i16_to_mont_i16(
                    limb.as_mut_ptr().cast::<i16>(),
                    coefficients.as_ptr(),
                    D,
                    prime.p.to_i64() as i16,
                    prime.pinv.to_i64() as i16,
                    prime.montsq.to_i64() as i16,
                );
            }
            forward_ntt(limb, *prime, twiddles, self.params.kernel_plan);
        }
        CyclotomicCrtNtt { limbs }
    }

    #[cfg(target_arch = "aarch64")]
    fn transform_neon_i32(&self, coefficients: &[i16; D]) -> CyclotomicCrtNtt<W, K, D> {
        debug_assert_eq!(size_of::<W>(), size_of::<i32>());
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for ((limb, prime), twiddles) in limbs
            .iter_mut()
            .zip(self.params.primes.iter())
            .zip(self.params.twiddles.iter())
        {
            // SAFETY: `new` selects this strategy only for the sealed i32
            // residue width. The source and destination arrays are disjoint.
            unsafe {
                neon::centered_i16_to_mont_i32(
                    limb.as_mut_ptr().cast::<i32>(),
                    coefficients.as_ptr(),
                    D,
                    prime.p.to_i64() as i32,
                    prime.pinv.to_i64() as i32,
                    prime.montsq.to_i64() as i32,
                );
            }
            forward_ntt(limb, *prime, twiddles, self.params.kernel_plan);
        }
        CyclotomicCrtNtt { limbs }
    }
}

impl<W: PrimeWidth, const K: usize, const D: usize> CrtNttParamSet<W, K, D> {
    /// Write the Montgomery residues of the centered coefficients of `ring`
    /// modulo every CRT prime into `out`, in `(-p, p)` and coefficient order.
    pub(super) fn field_residues<F: CrtNttConvertibleField>(
        &self,
        ring: &CyclotomicRing<F, D>,
        out: &mut [[MontCoeff<W>; D]; K],
    ) {
        let modulus = FieldModulusLimbs::new(
            (-F::one())
                .to_u128_checked()
                .expect("Akita field element must fit in u128")
                + 1,
        );
        match modulus.len {
            1 => self.field_residues_with::<F, 1>(ring, &modulus, out),
            2 => self.field_residues_with::<F, 2>(ring, &modulus, out),
            3 => self.field_residues_with::<F, 3>(ring, &modulus, out),
            4 => self.field_residues_with::<F, 4>(ring, &modulus, out),
            _ => self.field_residues_with::<F, 5>(ring, &modulus, out),
        }
    }

    /// [`Self::field_residues`] with `L` limbs, one chunk of coefficients at a
    /// time.
    fn field_residues_with<F: CrtNttConvertibleField, const L: usize>(
        &self,
        ring: &CyclotomicRing<F, D>,
        modulus: &FieldModulusLimbs,
        out: &mut [[MontCoeff<W>; D]; K],
    ) {
        let mut buffer = [0u128; FIELD_CHUNK];
        for (start, coefficients) in (0..D)
            .step_by(FIELD_CHUNK)
            .zip(ring.coeffs.chunks(FIELD_CHUNK))
        {
            let canonical = &mut buffer[..coefficients.len()];
            for (dst, coefficient) in canonical.iter_mut().zip(coefficients) {
                *dst = coefficient
                    .to_u128_checked()
                    .expect("Akita field element must fit in u128");
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if self.kernel_plan.uses_x86_transform()
                && size_of::<W>() == size_of::<i32>()
                && canonical.len() % 8 == 0
            {
                // SAFETY: PrimeWidth is sealed to i16 and i32, so the width
                // check identifies W as i32 and MontCoeff is transparent. The
                // prepared plan proves AVX2, the chunk length is a multiple of
                // eight and ends within D, L covers the modulus, and every
                // coefficient is canonical.
                unsafe {
                    avx::field_residues_i32::<L, K, D>(
                        &mut *(out as *mut _ as *mut [[MontCoeff<i32>; D]; K]),
                        start,
                        canonical,
                        modulus,
                        &self.field_scales,
                    );
                }
                continue;
            }
            #[cfg(target_arch = "aarch64")]
            if self.kernel_plan.uses_neon()
                && size_of::<W>() == size_of::<i32>()
                && canonical.len() % 4 == 0
            {
                // SAFETY: PrimeWidth is sealed to i16 and i32, so the width
                // check identifies W as i32 and MontCoeff is transparent. The
                // chunk length is a multiple of four and ends within D, L
                // covers the modulus, and every coefficient is canonical.
                unsafe {
                    neon::field_residues_i32::<L, K, D>(
                        &mut *(out as *mut _ as *mut [[MontCoeff<i32>; D]; K]),
                        start,
                        canonical,
                        modulus,
                        &self.field_scales,
                    );
                }
                continue;
            }
            field_residues::<W, L, K, D>(out, start, canonical, modulus, &self.field_scales);
        }
    }
}

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    pub(crate) fn centered_coefficients_with_params(
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

    pub(crate) fn centered_cyclic_coefficients_with_params(
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
            inverse_ntt_cyclic(&mut limb, *prime, twiddles, params.kernel_plan);
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
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        params.field_residues(ring, &mut limbs);
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
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
        let mut neg_limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        params.field_residues(ring, &mut neg_limbs);
        let mut cyc_limbs = neg_limbs;
        for (((neg_limb, cyc_limb), prime), tw) in neg_limbs
            .iter_mut()
            .zip(cyc_limbs.iter_mut())
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
            forward_ntt(neg_limb, *prime, tw, params.kernel_plan);
            forward_ntt_cyclic(cyc_limb, *prime, tw, params.kernel_plan);
        }
        (Self { limbs: neg_limbs }, Self { limbs: cyc_limbs })
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
        for (k, limb) in limbs.iter_mut().enumerate() {
            lut.fill_negacyclic_limb(k, digits, params, limb);
        }
        Self { limbs }
    }

    /// Convert small integer digits into cyclic CRT+NTT domain using a
    /// precomputed [`DigitMontLut`].
    #[inline]
    pub fn from_i8_cyclic_with_lut(
        digits: &[i8; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
    ) -> Self {
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        for (k, (limb, tw)) in limbs.iter_mut().zip(params.twiddles.iter()).enumerate() {
            lut.fill_limb(k, digits, params, limb);
            forward_ntt_cyclic(limb, params.primes[k], tw, params.kernel_plan);
        }
        Self { limbs }
    }

    /// Convert centered i32 coefficients into both negacyclic and cyclic
    /// CRT+NTT domains while sharing the coefficient preparation step.
    pub fn from_centered_i32_pair_with_params(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
    ) -> (Self, Self) {
        // SAFETY: without a LUT every coefficient takes the checked reduction path.
        unsafe { Self::from_centered_i32_pair(coeffs, params, None) }
    }

    /// Like [`Self::from_centered_i32_pair_with_params`] but uses a precomputed
    /// [`CenteredMontLut`] for caller-validated centered coefficients.
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
        // SAFETY: forwarded from this function's contract.
        unsafe { Self::from_centered_i32_pair(coeffs, params, Some(lut)) }
    }

    /// # Safety
    ///
    /// When `lut` is `Some`, every entry in `coeffs` must be within its range.
    unsafe fn from_centered_i32_pair(
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
            let reducer = CenteredPrimeReducer::new(*prime);
            if let Some(lut) = lut {
                for (dst, &coeff) in neg_limb.iter_mut().zip(coeffs.iter()) {
                    // SAFETY: the caller guarantees `coeff` is covered by `lut`.
                    *dst = unsafe { lut.get_unchecked(k, coeff) };
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
        let mut limbs = [[MontCoeff::from_raw(W::default()); D]; K];
        params.field_residues(ring, &mut limbs);
        for ((limb, prime), tw) in limbs
            .iter_mut()
            .zip(params.primes.iter())
            .zip(params.twiddles.iter())
        {
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
