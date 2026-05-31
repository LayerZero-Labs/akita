//! CRT+NTT-domain representation of cyclotomic ring elements.

use std::array::from_fn;
use std::marker::PhantomData;
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
use std::mem::size_of;

use crate::backend::{CrtReconstruct, NttPrimeOps, NttTransform, ScalarBackend};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::ntt::avx::{self, AvxNttMode};
use crate::ntt::butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt_cyclic, NttTwiddles};
use crate::ntt::crt::GarnerData;
#[cfg(target_arch = "aarch64")]
use crate::ntt::neon;
use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};
use crate::{CanonicalField, FieldCore};

use super::cyclotomic::CyclotomicRing;

/// Polynomial rows processed per AVX-512 batched-row NTT call.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const NTT_BATCH_LANES: usize = avx::batch::BATCH_LANES;

/// CRT+NTT-domain representation of a cyclotomic ring element.
///
/// Stores `K` arrays of `D` [`MontCoeff<W>`] values, one per CRT prime.
/// Multiplication is pointwise per prime.
#[repr(transparent)]
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

/// Number of balanced-digit slots covered by [`DigitMontLut`].
///
/// Balanced base-`2^log_basis` decomposition uses `log_basis` in `1..=6`, so
/// every digit lands in `[-32, 31]`. One fixed 64-entry table therefore covers
/// all bases, which is what lets the lookup drop the const-generic `L` (and the
/// per-`log_basis` monomorphization it forced) without widening to the full
/// signed-byte range.
const DIGIT_LUT_LEN: usize = 64;
const DIGIT_LUT_OFFSET: i16 = (DIGIT_LUT_LEN / 2) as i16;

/// Precomputed Montgomery forms for balanced digit values in `[-32, 31]`.
///
/// Storing the Montgomery representation eliminates one `from_canonical`
/// Montgomery multiply per coefficient in the hot digit path. The table is
/// built once per mat-vec and is independent of the decomposition `log_basis`.
#[derive(Debug, Clone)]
pub struct DigitMontLut<W: PrimeWidth, const K: usize> {
    vals: [[MontCoeff<W>; DIGIT_LUT_LEN]; K],
    len: usize,
    offset: i16,
}

/// Precomputed Montgomery forms for centered integer coefficients in
/// `[-max_abs, max_abs]`.
#[derive(Debug, Clone)]
pub struct CenteredMontLut<W: PrimeWidth, const K: usize> {
    vals: [Vec<MontCoeff<W>>; K],
    offset: i32,
}

#[derive(Debug, Clone, Copy)]
struct CenteredPrimeReducer<W: PrimeWidth> {
    p: i64,
    _width: PhantomData<W>,
}

impl<W: PrimeWidth> CenteredPrimeReducer<W> {
    #[inline(always)]
    fn new(prime: NttPrime<W>) -> Self {
        let p = prime.p.to_i64();
        Self {
            p,
            _width: PhantomData,
        }
    }

    #[inline(always)]
    fn reduce_i64(self, value: i64) -> W {
        let mut r = value.rem_euclid(self.p);
        if r > self.p / 2 {
            r -= self.p;
        }
        W::from_i64(r)
    }
}

#[derive(Debug, Clone, Copy)]
struct CenteredPrimeWideReducer<W: PrimeWidth> {
    narrow: CenteredPrimeReducer<W>,
    p_u64: u64,
    r64: i64,
}

impl<W: PrimeWidth> CenteredPrimeWideReducer<W> {
    #[inline(always)]
    fn new(prime: NttPrime<W>) -> Self {
        let narrow = CenteredPrimeReducer::new(prime);
        let p_u64 = narrow.p as u64;
        let r64 = ((1u128 << 64) % p_u64 as u128) as i64;
        Self { narrow, p_u64, r64 }
    }

    #[inline(always)]
    fn reduce_i128(self, value: i128) -> W {
        // Split the signed value into a low 64-bit limb and a sign-extended high
        // word, then reduce `hi * 2^64 + lo` modulo the small CRT prime.
        let lo = (value as u64 % self.p_u64) as i64;
        let hi = ((value >> 64) as i64).rem_euclid(self.narrow.p);
        let r = (lo + hi * self.r64) % self.narrow.p;
        self.narrow.reduce_i64(r)
    }
}

#[cfg(test)]
#[inline(always)]
fn centered_prime_residue_i64<W: PrimeWidth>(prime: NttPrime<W>, value: i64) -> W {
    CenteredPrimeReducer::new(prime).reduce_i64(value)
}

#[cfg(test)]
#[inline(always)]
fn centered_prime_residue_i128<W: PrimeWidth>(prime: NttPrime<W>, value: i128) -> W {
    CenteredPrimeWideReducer::new(prime).reduce_i128(value)
}

impl<W: PrimeWidth, const K: usize> DigitMontLut<W, K> {
    /// Build the lookup table from CRT primes, covering balanced digits in
    /// `[-32, 31]`.
    pub fn new<const D: usize>(params: &CrtNttParamSet<W, K, D>) -> Self {
        Self::new_with_digit_bound(params, DIGIT_LUT_OFFSET as u64)
    }

    /// Build a lookup table for the active balanced range `[-bound, bound)`.
    ///
    /// This keeps the fixed non-monomorphized LUT type while avoiding needless
    /// Montgomery conversions for common small bases (`log_basis` 2, 3, or 4).
    pub fn new_with_digit_bound<const D: usize>(
        params: &CrtNttParamSet<W, K, D>,
        digit_abs_bound: u64,
    ) -> Self {
        debug_assert!(digit_abs_bound.is_power_of_two());
        debug_assert!((1..=DIGIT_LUT_OFFSET as u64).contains(&digit_abs_bound));
        let digit_abs_bound = digit_abs_bound
            .max(1)
            .next_power_of_two()
            .min(DIGIT_LUT_OFFSET as u64);
        let len = (digit_abs_bound as usize) * 2;
        let offset = digit_abs_bound as i16;
        let mut vals = [[MontCoeff::from_raw(W::default()); DIGIT_LUT_LEN]; K];
        for (k, limb) in vals.iter_mut().enumerate() {
            let prime = params.primes[k];
            for (idx, dst) in limb.iter_mut().enumerate().take(len) {
                let v = idx as i64 - i64::from(offset);
                *dst = prime.from_canonical(W::from_i64(v));
            }
        }
        Self { vals, len, offset }
    }

    /// Look up the Montgomery form of a balanced digit for CRT prime `k`.
    ///
    /// Contract: `digit` is in this LUT's active balanced range and `k < K`.
    /// The i8-NTT kernel boundary upholds this with validated `log_basis` and
    /// per-block digit range checks. Because the active table length is a power
    /// of two, masking keeps the lookup in-bounds and branch-free without a
    /// per-coefficient bounds check. The debug assertion surfaces any contract
    /// violation in debug builds.
    #[inline(always)]
    pub fn get(&self, k: usize, digit: i8) -> MontCoeff<W> {
        let idx = (i16::from(digit) + self.offset) as usize;
        debug_assert!(
            idx < self.len,
            "digit LUT lookup outside active balanced range"
        );
        self.vals[k][idx & (self.len - 1)]
    }
}

impl<W: PrimeWidth, const K: usize> CenteredMontLut<W, K> {
    /// Build a lookup table for all centered coefficients in `[-max_abs, max_abs]`.
    pub fn new<const D: usize>(params: &CrtNttParamSet<W, K, D>, max_abs: i32) -> Self {
        let max_abs = max_abs.max(0);
        let vals = from_fn(|k| {
            let prime = params.primes[k];
            let reducer = CenteredPrimeReducer::new(prime);
            (-max_abs..=max_abs)
                .map(|v| prime.from_canonical(reducer.reduce_i64(i64::from(v))))
                .collect()
        });
        Self {
            vals,
            offset: max_abs,
        }
    }

    /// Look up the Montgomery form of a centered coefficient for CRT prime `k`.
    #[inline(always)]
    pub fn get(&self, k: usize, coeff: i32) -> Option<MontCoeff<W>> {
        let idx = coeff.checked_add(self.offset)?;
        self.vals.get(k)?.get(usize::try_from(idx).ok()?).copied()
    }

    /// Look up the Montgomery form of a caller-validated centered coefficient.
    ///
    /// # Safety
    ///
    /// `k` must be less than `K`, and `coeff` must be within this LUT's
    /// covered centered range.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, k: usize, coeff: i32) -> MontCoeff<W> {
        let idx = coeff + self.offset;
        debug_assert!(idx >= 0);
        debug_assert!((idx as usize) < self.vals[k].len());
        unsafe { *self.vals.get_unchecked(k).get_unchecked(idx as usize) }
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
                let sum = MontCoeff::from_raw(acc_limb[i].raw().wrapping_add(prod.raw()));
                acc_limb[i] = prime.reduce_range(sum);
            }
            idx += 4;
        }

        while idx < D {
            let prod = prime.mul(lhs_limb[idx], rhs_limb[idx]);
            let sum = MontCoeff::from_raw(acc_limb[idx].raw().wrapping_add(prod.raw()));
            acc_limb[idx] = prime.reduce_range(sum);
            idx += 1;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline(always)]
    fn x86_pointwise_mode() -> Option<AvxNttMode> {
        let mode = avx::avx_ntt_mode()?;
        if size_of::<W>() == size_of::<i16>() {
            return avx::use_avx2_transform_ntt().then_some(AvxNttMode::Avx2);
        }
        (size_of::<W>() == size_of::<i32>()).then_some(mode)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline(always)]
    unsafe fn add_assign_pointwise_mul_limb_x86(
        acc_limb: &mut [MontCoeff<W>; D],
        lhs_limb: &[MontCoeff<W>; D],
        rhs_limb: &[MontCoeff<W>; D],
        prime: NttPrime<W>,
        mode: AvxNttMode,
    ) {
        // SAFETY: caller checked x86 SIMD dispatch. `MontCoeff<W>` is
        // transparent over the sealed `i16`/`i32` widths and the arrays are
        // valid for `D`.
        unsafe {
            if size_of::<W>() == size_of::<i16>() {
                avx::pointwise_mul_acc_i16(
                    acc_limb.as_mut_ptr() as *mut i16,
                    lhs_limb.as_ptr() as *const i16,
                    rhs_limb.as_ptr() as *const i16,
                    D,
                    prime.p.to_i64() as i16,
                    prime.pinv.to_i64() as i16,
                );
            } else if size_of::<W>() == size_of::<i32>() {
                match mode {
                    AvxNttMode::Avx2 => avx::pointwise_mul_acc_i32(
                        acc_limb.as_mut_ptr() as *mut i32,
                        lhs_limb.as_ptr() as *const i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    ),
                    AvxNttMode::Avx512 => avx::pointwise_mul_acc_i32_avx512(
                        acc_limb.as_mut_ptr() as *mut i32,
                        lhs_limb.as_ptr() as *const i32,
                        rhs_limb.as_ptr() as *const i32,
                        D,
                        prime.p.to_i64() as i32,
                        prime.pinv.to_i64() as i32,
                    ),
                }
            }
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
        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw);
            }

            for (k, rhs_limb) in scratch.iter().enumerate() {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::pointwise_mul_acc_i32(
                            self.limbs[k].as_mut_ptr() as *mut i32,
                            lhs.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                    } else {
                        neon::pointwise_mul_acc_i16(
                            self.limbs[k].as_mut_ptr() as *mut i16,
                            lhs.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
        for (k, (scratch_limb, tw)) in scratch.iter_mut().zip(params.twiddles.iter()).enumerate() {
            for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                *dst = lut.get(k, digit);
            }
            forward_ntt(scratch_limb, params.primes[k], tw);

            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let lhs_limb = &lhs.limbs[k];
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(mode) = x86_mode {
                // SAFETY: guarded by x86 runtime dispatch.
                unsafe {
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc_limb,
                        lhs_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                }
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

        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw);
            }

            for (k, rhs_limb) in scratch.iter().enumerate() {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::pointwise_mul_acc_i32(
                            acc0.limbs[k].as_mut_ptr() as *mut i32,
                            lhs0.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                        neon::pointwise_mul_acc_i32(
                            acc1.limbs[k].as_mut_ptr() as *mut i32,
                            lhs1.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                    } else {
                        neon::pointwise_mul_acc_i16(
                            acc0.limbs[k].as_mut_ptr() as *mut i16,
                            lhs0.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                        neon::pointwise_mul_acc_i16(
                            acc1.limbs[k].as_mut_ptr() as *mut i16,
                            lhs1.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
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
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(mode) = x86_mode {
                // SAFETY: guarded by x86 runtime dispatch.
                unsafe {
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc0_limb,
                        lhs0_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc1_limb,
                        lhs1_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                }
                continue;
            }
            for (((acc0_coeff, lhs0_coeff), acc1_coeff), (lhs1_coeff, rhs_coeff)) in acc0_limb
                .iter_mut()
                .zip(lhs0_limb.iter())
                .zip(acc1_limb.iter_mut())
                .zip(lhs1_limb.iter().zip(scratch_limb.iter()))
            {
                let prod0 = prime.mul(*lhs0_coeff, *rhs_coeff);
                let sum0 = MontCoeff::from_raw(acc0_coeff.raw().wrapping_add(prod0.raw()));
                *acc0_coeff = prime.reduce_range(sum0);

                let prod1 = prime.mul(*lhs1_coeff, *rhs_coeff);
                let sum1 = MontCoeff::from_raw(acc1_coeff.raw().wrapping_add(prod1.raw()));
                *acc1_coeff = prime.reduce_range(sum1);
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

        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for (k, (scratch_limb, tw)) in
                scratch.iter_mut().zip(params.twiddles.iter()).enumerate()
            {
                for (dst, &digit) in scratch_limb.iter_mut().zip(digits.iter()) {
                    *dst = lut.get(k, digit);
                }
                forward_ntt(scratch_limb, params.primes[k], tw);
            }

            for (k, rhs_limb) in scratch.iter().enumerate() {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::pointwise_mul_acc_i32(
                            acc0.limbs[k].as_mut_ptr() as *mut i32,
                            lhs0.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                        neon::pointwise_mul_acc_i32(
                            acc1.limbs[k].as_mut_ptr() as *mut i32,
                            lhs1.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                        neon::pointwise_mul_acc_i32(
                            acc2.limbs[k].as_mut_ptr() as *mut i32,
                            lhs2.limbs[k].as_ptr() as *const i32,
                            rhs_limb.as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                    } else {
                        neon::pointwise_mul_acc_i16(
                            acc0.limbs[k].as_mut_ptr() as *mut i16,
                            lhs0.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                        neon::pointwise_mul_acc_i16(
                            acc1.limbs[k].as_mut_ptr() as *mut i16,
                            lhs1.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                        neon::pointwise_mul_acc_i16(
                            acc2.limbs[k].as_mut_ptr() as *mut i16,
                            lhs2.limbs[k].as_ptr() as *const i16,
                            rhs_limb.as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
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
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(mode) = x86_mode {
                // SAFETY: guarded by x86 runtime dispatch.
                unsafe {
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc0_limb,
                        lhs0_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc1_limb,
                        lhs1_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc2_limb,
                        lhs2_limb,
                        scratch_limb,
                        prime,
                        mode,
                    );
                }
                continue;
            }
            for idx in 0..D {
                let rhs_coeff = scratch_limb[idx];

                let prod0 = prime.mul(lhs0_limb[idx], rhs_coeff);
                let sum0 = MontCoeff::from_raw(acc0_limb[idx].raw().wrapping_add(prod0.raw()));
                acc0_limb[idx] = prime.reduce_range(sum0);

                let prod1 = prime.mul(lhs1_limb[idx], rhs_coeff);
                let sum1 = MontCoeff::from_raw(acc1_limb[idx].raw().wrapping_add(prod1.raw()));
                acc1_limb[idx] = prime.reduce_range(sum1);

                let prod2 = prime.mul(lhs2_limb[idx], rhs_coeff);
                let sum2 = MontCoeff::from_raw(acc2_limb[idx].raw().wrapping_add(prod2.raw()));
                acc2_limb[idx] = prime.reduce_range(sum2);
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
    /// On AArch64, this uses the fused NEON pointwise-multiply-accumulate kernel
    /// when available; otherwise it falls back to the scalar loop.
    #[inline(always)]
    pub fn add_assign_pointwise_mul_with_params(
        &mut self,
        lhs: &Self,
        rhs: &Self,
        params: &CrtNttParamSet<W, K, D>,
    ) {
        #[cfg(target_arch = "aarch64")]
        if neon::use_neon_ntt() {
            for k in 0..K {
                let prime = params.primes[k];
                unsafe {
                    if size_of::<W>() == size_of::<i32>() {
                        neon::pointwise_mul_acc_i32(
                            self.limbs[k].as_mut_ptr() as *mut i32,
                            lhs.limbs[k].as_ptr() as *const i32,
                            rhs.limbs[k].as_ptr() as *const i32,
                            D,
                            prime.p.to_i64() as i32,
                            prime.pinv.to_i64() as i32,
                        );
                    } else {
                        neon::pointwise_mul_acc_i16(
                            self.limbs[k].as_mut_ptr() as *mut i16,
                            lhs.limbs[k].as_ptr() as *const i16,
                            rhs.limbs[k].as_ptr() as *const i16,
                            D,
                            prime.p.to_i64() as i16,
                            prime.pinv.to_i64() as i16,
                        );
                    }
                }
            }
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let x86_mode = Self::x86_pointwise_mode();
        for k in 0..K {
            let prime = params.primes[k];
            let acc_limb = &mut self.limbs[k];
            let lhs_limb = &lhs.limbs[k];
            let rhs_limb = &rhs.limbs[k];
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if let Some(mode) = x86_mode {
                // SAFETY: guarded by x86 runtime dispatch.
                unsafe {
                    Self::add_assign_pointwise_mul_limb_x86(
                        acc_limb, lhs_limb, rhs_limb, prime, mode,
                    );
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::tables::{Q16_NUM_PRIMES, Q16_PRIMES, Q32_PRIMES};

    #[test]
    fn centered_prime_residue_keeps_positive_half_boundary() {
        let prime16 = Q16_PRIMES[0];
        let half16 = i64::from(prime16.p) / 2;
        assert_eq!(centered_prime_residue_i64(prime16, half16), half16 as i16);
        assert_eq!(
            centered_prime_residue_i64(prime16, half16 + 1),
            (half16 + 1 - i64::from(prime16.p)) as i16
        );

        let prime32 = Q32_PRIMES[0];
        let half32 = i64::from(prime32.p) / 2;
        assert_eq!(centered_prime_residue_i64(prime32, half32), half32 as i32);
        assert_eq!(
            centered_prime_residue_i64(prime32, half32 + 1),
            (half32 + 1 - i64::from(prime32.p)) as i32
        );
        assert_eq!(
            centered_prime_residue_i128(prime32, i128::from(half32)),
            half32 as i32
        );
        assert_eq!(
            centered_prime_residue_i128(prime32, i128::from(half32 + 1)),
            (half32 + 1 - i64::from(prime32.p)) as i32
        );
    }

    #[test]
    fn centered_mont_lut_matches_centered_residue_boundary() {
        const D: usize = 64;
        let params = CrtNttParamSet::<i16, Q16_NUM_PRIMES, D>::new(Q16_PRIMES);
        let prime = params.primes[0];
        let half = i32::from(prime.p) / 2;
        let lut = CenteredMontLut::<i16, Q16_NUM_PRIMES>::new(&params, half + 1);

        let boundary = centered_prime_residue_i64(prime, i64::from(half));
        let past_boundary = centered_prime_residue_i64(prime, i64::from(half + 1));
        assert_eq!(boundary, half as i16);
        assert_eq!(past_boundary, (half + 1 - i32::from(prime.p)) as i16);
        assert_eq!(lut.get(0, half), Some(prime.from_canonical(boundary)));
        assert_eq!(
            lut.get(0, half + 1),
            Some(prime.from_canonical(past_boundary))
        );
    }
}
