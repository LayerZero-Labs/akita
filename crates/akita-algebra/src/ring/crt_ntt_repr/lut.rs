use std::array::from_fn;
use std::marker::PhantomData;

use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};

use super::CrtNttParamSet;

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
pub(super) struct CenteredPrimeReducer<W: PrimeWidth> {
    p: i64,
    _width: PhantomData<W>,
}

impl<W: PrimeWidth> CenteredPrimeReducer<W> {
    #[inline(always)]
    pub(super) fn new(prime: NttPrime<W>) -> Self {
        let p = prime.p.to_i64();
        Self {
            p,
            _width: PhantomData,
        }
    }

    #[inline(always)]
    pub(super) fn reduce_i64(self, value: i64) -> W {
        let mut r = value.rem_euclid(self.p);
        if r > self.p / 2 {
            r -= self.p;
        }
        W::from_i64(r)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct CenteredPrimeWideReducer<W: PrimeWidth> {
    narrow: CenteredPrimeReducer<W>,
    p_u64: u64,
    r64: i64,
}

impl<W: PrimeWidth> CenteredPrimeWideReducer<W> {
    #[inline(always)]
    pub(super) fn new(prime: NttPrime<W>) -> Self {
        let narrow = CenteredPrimeReducer::new(prime);
        let p_u64 = narrow.p as u64;
        let r64 = ((1u128 << 64) % p_u64 as u128) as i64;
        Self { narrow, p_u64, r64 }
    }

    #[inline(always)]
    pub(super) fn reduce_i128(self, value: i128) -> W {
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
pub(super) fn centered_prime_residue_i64<W: PrimeWidth>(prime: NttPrime<W>, value: i64) -> W {
    CenteredPrimeReducer::new(prime).reduce_i64(value)
}

#[cfg(test)]
#[inline(always)]
pub(super) fn centered_prime_residue_i128<W: PrimeWidth>(prime: NttPrime<W>, value: i128) -> W {
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
