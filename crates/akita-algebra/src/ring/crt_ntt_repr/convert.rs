use std::array::from_fn;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;

use crate::backend::{CrtReconstruct, NttPrimeOps, NttTransform, ScalarBackend};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::avx::{self, AvxNttMode};
use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt_cyclic, NttTwiddles};
use crate::ntt::crt::GarnerData;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
use crate::ring::cyclotomic::CyclotomicRing;

use super::lut::{CenteredPrimeReducer, CenteredPrimeWideReducer};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::NTT_BATCH_LANES;
use super::{
    CenteredMontLut, CrtNttConvertibleField, CrtNttParamSet, CyclotomicCrtNtt, DigitMontLut,
};

impl<W: PrimeWidth, const K: usize, const D: usize> CyclotomicCrtNtt<W, K, D> {
    /// Convert a coefficient-form ring element into CRT+NTT domain
    /// using explicit prime and twiddle tables.
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

    /// Convert a coefficient-form ring element into both negacyclic and cyclic
    /// CRT+NTT domains, sharing coefficient centering and CRT reduction.
    pub fn from_ring_pair_with_params<F: CrtNttConvertibleField>(
        ring: &CyclotomicRing<F, D>,
        params: &CrtNttParamSet<W, K, D>,
    ) -> (Self, Self) {
        Self::from_ring_pair_with_backend::<F, ScalarBackend>(ring, params)
    }

    fn from_ring_pair_with_backend<
        F: CrtNttConvertibleField,
        B: NttPrimeOps<W, D> + NttTransform<W, D>,
    >(
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
                *dst = B::from_canonical(*prime, reducer.reduce_i128(*centered));
            }
            *cyc_limb = *neg_limb;
            B::forward_ntt(neg_limb, *prime, tw);
            forward_ntt_cyclic(cyc_limb, *prime, tw);
        }
        (Self { limbs: neg_limbs }, Self { limbs: cyc_limbs })
    }

    /// Apply a forward NTT to up to [`NTT_BATCH_LANES`] CRT+NTT elements whose
    /// limbs are already filled in coefficient form.
    ///
    /// When the group is exactly [`NTT_BATCH_LANES`] `i32` rows and AVX-512 is
    /// active, this uses the batched-row kernel (transforming lane = row); the
    /// per-element fallback is bit-identical. `chunk` must be one contiguous run
    /// of elements so the batched kernel can stride by `K*D` across rows.
    fn transform_chunk(chunk: &mut [Self], params: &CrtNttParamSet<W, K, D>, cyclic: bool) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if chunk.len() == NTT_BATCH_LANES
            && size_of::<W>() == size_of::<i32>()
            && avx::avx_ntt_mode() == Some(AvxNttMode::Avx512)
        {
            let row_stride = K * D;
            let chunk_base = chunk.as_mut_ptr().cast::<MontCoeff<W>>();
            for k in 0..K {
                let base = chunk_base.wrapping_add(k * D) as *mut i32;
                // SAFETY: `W == i32`, `Self` is transparent over its limbs, and
                // `base` derives from the whole mutable chunk. Row strides stay
                // within that allocation; AVX-512 is proven by the mode check.
                unsafe {
                    let prime = *(&params.primes[k] as *const NttPrime<W> as *const NttPrime<i32>);
                    let tw = &*(&params.twiddles[k] as *const NttTwiddles<W, D>
                        as *const NttTwiddles<i32, D>);
                    if cyclic {
                        avx::batch::batched_forward_ntt_cyclic_16rows::<D>(
                            base, row_stride, prime, tw,
                        );
                    } else {
                        avx::batch::batched_forward_ntt_16rows::<D>(base, row_stride, prime, tw);
                    }
                }
            }
            return;
        }

        for el in chunk.iter_mut() {
            for ((limb, prime), tw) in el
                .limbs
                .iter_mut()
                .zip(params.primes.iter())
                .zip(params.twiddles.iter())
            {
                if cyclic {
                    forward_ntt_cyclic(limb, *prime, tw);
                } else {
                    forward_ntt(limb, *prime, tw);
                }
            }
        }
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
            let reducer = CenteredPrimeWideReducer::new(*prime);
            // Interpret coefficients in centered form (-q/2, q/2] before reducing
            // into the CRT primes. This makes the reduction map consistent with
            // negacyclic subtraction (which naturally produces negative values).
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                *dst = B::from_canonical(*prime, reducer.reduce_i128(*centered));
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

    /// Batched form of [`Self::from_i8_with_lut`]: build the CRT+NTT element for
    /// each digit plane in `digits` into the matching slot of `out`.
    ///
    /// Fills every slot's limbs from the LUT, then applies one batched forward
    /// negacyclic NTT across the whole group (bit-identical to mapping
    /// [`Self::from_i8_with_lut`]). `out.len()` must equal `digits.len()`; the
    /// AVX-512 batched-row kernel engages when the group is exactly 16 `i32`
    /// rows, otherwise each row transforms per-element.
    pub fn batch_from_i8_with_lut_into(
        digits: &[[i8; D]],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        out: &mut [Self],
    ) {
        assert_eq!(
            digits.len(),
            out.len(),
            "digit and output batches must have the same length"
        );
        for (slot, digit) in out.iter_mut().zip(digits.iter()) {
            for (k, limb) in slot.limbs.iter_mut().enumerate() {
                for (dst, &d) in limb.iter_mut().zip(digit.iter()) {
                    *dst = lut.get(k, d);
                }
            }
        }
        Self::transform_chunk(out, params, false);
    }

    /// Like [`Self::batch_from_i8_with_lut_into`], but accepts borrowed digit
    /// planes so callers can pack sparse nonzero rows without copying them.
    pub fn batch_from_i8_refs_with_lut_into(
        digits: &[&[i8; D]],
        params: &CrtNttParamSet<W, K, D>,
        lut: &DigitMontLut<W, K>,
        out: &mut [Self],
    ) {
        assert_eq!(
            digits.len(),
            out.len(),
            "digit and output batches must have the same length"
        );
        for (slot, digit) in out.iter_mut().zip(digits.iter()) {
            for (k, limb) in slot.limbs.iter_mut().enumerate() {
                for (dst, &d) in limb.iter_mut().zip(digit.iter()) {
                    *dst = lut.get(k, d);
                }
            }
        }
        Self::transform_chunk(out, params, false);
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
        Self::from_centered_i32_pair_backend::<ScalarBackend>(coeffs, params, None, false)
    }

    /// Like [`Self::from_centered_i32_pair_with_params`] but uses a precomputed
    /// [`CenteredMontLut`] for the coefficient-to-Montgomery conversion.
    pub fn from_centered_i32_pair_with_lut(
        coeffs: &[i32; D],
        params: &CrtNttParamSet<W, K, D>,
        lut: &CenteredMontLut<W, K>,
    ) -> (Self, Self) {
        Self::from_centered_i32_pair_backend::<ScalarBackend>(coeffs, params, Some(lut), false)
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
        Self::from_centered_i32_pair_backend::<ScalarBackend>(coeffs, params, Some(lut), true)
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
            let reducer = CenteredPrimeReducer::new(*prime);
            for (dst, &coeff) in limb.iter_mut().zip(coeffs.iter()) {
                *dst = B::from_canonical(*prime, reducer.reduce_i64(i64::from(coeff)));
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
            let reducer = CenteredPrimeReducer::new(*prime);
            for (dst, &coeff) in limb.iter_mut().zip(coeffs.iter()) {
                *dst = B::from_canonical(*prime, reducer.reduce_i64(i64::from(coeff)));
            }
            forward_ntt_cyclic(limb, *prime, tw);
        }
        Self { limbs }
    }

    fn from_centered_i32_pair_backend<B: NttPrimeOps<W, D>>(
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
                            B::from_canonical(*prime, reducer.reduce_i64(i64::from(coeff)))
                        })
                    };
                }
            } else {
                for (dst, &coeff) in neg_limb.iter_mut().zip(coeffs.iter()) {
                    *dst = B::from_canonical(*prime, reducer.reduce_i64(i64::from(coeff)));
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

    /// Convert a coefficient-form ring element into CRT+**cyclic** NTT domain
    /// using the scalar backend.
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
            let reducer = CenteredPrimeWideReducer::new(*prime);
            for (dst, centered) in limb.iter_mut().zip(centered_coeffs.iter()) {
                *dst = B::from_canonical(*prime, reducer.reduce_i128(*centered));
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
}
