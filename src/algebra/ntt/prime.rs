//! NTT prime arithmetic kernels generic over coefficient width.
//!
//! Per-prime scalar operations:
//! - Montgomery multiplication ([`NttPrime::mul`])
//! - Branchless conditional add/sub and range reduction
//!
//! Coefficients in Montgomery domain are wrapped in [`MontCoeff`] to prevent
//! accidental mixing with canonical values.
//!
//! The [`PrimeWidth`] trait abstracts over `i16` (R = 2^16, for primes < 2^14)
//! and `i32` (R = 2^32, for primes < 2^30). All NTT types are generic over
//! `W: PrimeWidth`; monomorphization produces optimal code for each width.

use std::fmt;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
}

/// Integer width abstraction for NTT prime arithmetic.
///
/// Sealed with exactly two implementations: `i16` and `i32`.
pub trait PrimeWidth:
    sealed::Sealed + Copy + Clone + Eq + Default + fmt::Debug + Send + Sync + 'static
{
    /// Double-width type for intermediate Montgomery products.
    type Wide: Copy + Clone;

    /// log2(R) for Montgomery reduction: 16 for `i16`, 32 for `i32`.
    const R_LOG: u32;

    /// Widening multiply: `a * b` as `Wide`.
    fn wide_mul(a: Self, b: Self) -> Self::Wide;

    /// Truncate wide value to narrow (low half, i.e. mod R).
    fn truncate(w: Self::Wide) -> Self;

    /// Arithmetic right shift of wide value by `R_LOG` bits.
    fn wide_shift(w: Self::Wide) -> Self;

    /// Wide subtraction (wrapping).
    fn wide_sub(a: Self::Wide, b: Self::Wide) -> Self::Wide;

    /// Wrapping addition.
    fn wrapping_add(self, rhs: Self) -> Self;
    /// Wrapping subtraction.
    fn wrapping_sub(self, rhs: Self) -> Self;
    /// Wrapping multiplication.
    fn wrapping_mul(self, rhs: Self) -> Self;
    /// Wrapping negation.
    fn wrapping_neg(self) -> Self;

    /// Arithmetic right shift by `BITS - 1`: all-1s if negative, all-0s otherwise.
    fn sign_mask(self) -> Self;

    /// Bitwise AND.
    fn bitand(self, rhs: Self) -> Self;

    /// Convert from `i64` (truncating).
    fn from_i64(v: i64) -> Self;
    /// Convert to `i64` (sign-extending).
    fn to_i64(self) -> i64;
}

impl PrimeWidth for i16 {
    type Wide = i32;
    const R_LOG: u32 = 16;

    #[inline]
    fn wide_mul(a: Self, b: Self) -> i32 {
        (a as i32) * (b as i32)
    }
    #[inline]
    fn truncate(w: i32) -> Self {
        w as i16
    }
    #[inline]
    fn wide_shift(w: i32) -> Self {
        (w >> 16) as i16
    }
    #[inline]
    fn wide_sub(a: i32, b: i32) -> i32 {
        a.wrapping_sub(b)
    }
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        i16::wrapping_add(self, rhs)
    }
    #[inline]
    fn wrapping_sub(self, rhs: Self) -> Self {
        i16::wrapping_sub(self, rhs)
    }
    #[inline]
    fn wrapping_mul(self, rhs: Self) -> Self {
        i16::wrapping_mul(self, rhs)
    }
    #[inline]
    fn wrapping_neg(self) -> Self {
        i16::wrapping_neg(self)
    }
    #[inline]
    fn sign_mask(self) -> Self {
        self >> 15
    }
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as i16
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

impl PrimeWidth for i32 {
    type Wide = i64;
    const R_LOG: u32 = 32;

    #[inline]
    fn wide_mul(a: Self, b: Self) -> i64 {
        (a as i64) * (b as i64)
    }
    #[inline]
    fn truncate(w: i64) -> Self {
        w as i32
    }
    #[inline]
    fn wide_shift(w: i64) -> Self {
        (w >> 32) as i32
    }
    #[inline]
    fn wide_sub(a: i64, b: i64) -> i64 {
        a.wrapping_sub(b)
    }
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        i32::wrapping_add(self, rhs)
    }
    #[inline]
    fn wrapping_sub(self, rhs: Self) -> Self {
        i32::wrapping_sub(self, rhs)
    }
    #[inline]
    fn wrapping_mul(self, rhs: Self) -> Self {
        i32::wrapping_mul(self, rhs)
    }
    #[inline]
    fn wrapping_neg(self) -> Self {
        i32::wrapping_neg(self)
    }
    #[inline]
    fn sign_mask(self) -> Self {
        self >> 31
    }
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        self & rhs
    }
    #[inline]
    fn from_i64(v: i64) -> Self {
        v as i32
    }
    #[inline]
    fn to_i64(self) -> i64 {
        self as i64
    }
}

/// A coefficient in Montgomery domain for an NTT prime.
///
/// Wraps a `W` representing `a * R mod p` (where `R = 2^{W::R_LOG}`).
/// Use [`NttPrime::from_canonical`] to enter and [`NttPrime::to_canonical`]
/// to leave Montgomery domain.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct MontCoeff<W: PrimeWidth>(W);

impl<W: PrimeWidth> MontCoeff<W> {
    /// Wrap a raw Montgomery-domain value.
    #[inline]
    pub fn from_raw(val: W) -> Self {
        Self(val)
    }

    /// Extract the raw value (still in Montgomery domain).
    #[inline]
    pub fn raw(self) -> W {
        self.0
    }
}

impl<W: PrimeWidth> fmt::Debug for MontCoeff<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mont({:?})", self.0)
    }
}

/// Per-prime constants for NTT arithmetic.
///
/// Generic over `W: PrimeWidth` — use `i16` for primes below 2^14 (R = 2^16),
/// or `i32` for primes below 2^30 (R = 2^32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttPrime<W: PrimeWidth> {
    /// Prime modulus.
    pub p: W,
    /// `p^{-1} mod R` (centered signed). Used in Montgomery reduction.
    pub pinv: W,
    /// `R mod p` (centered signed). Montgomery form of 1.
    pub mont: W,
    /// `R^2 mod p` (centered signed). Used for canonical → Montgomery conversion.
    pub montsq: W,
}

impl<W: PrimeWidth> NttPrime<W> {
    /// Derive all Montgomery constants from a raw prime value.
    pub fn compute(p: W) -> Self {
        let p_i64 = p.to_i64();
        debug_assert!(p_i64 > 1 && p_i64 % 2 == 1, "NTT prime must be odd and > 1");

        // pinv via Newton's method: x_{n+1} = x_n * (2 - p * x_n).
        // 5 iterations gives correctness mod 2^32 (sufficient for both i16 and i32).
        let mut pinv: i64 = 1;
        for _ in 0..5 {
            pinv = pinv.wrapping_mul(2i64.wrapping_sub(p_i64.wrapping_mul(pinv)));
        }
        let pinv = W::from_i64(pinv);

        let half = p_i64 / 2;
        let center = |x: i64| -> W { W::from_i64(if x > half { x - p_i64 } else { x }) };

        let r_mod_p = ((1i128 << W::R_LOG) % (p_i64 as i128)) as i64;
        let mont = center(r_mod_p);

        let rsq_mod_p = ((1i128 << (2 * W::R_LOG)) % (p_i64 as i128)) as i64;
        let montsq = center(rsq_mod_p);

        Self {
            p,
            pinv,
            mont,
            montsq,
        }
    }

    /// Montgomery product: `a * b * R^{-1} mod p`.
    #[inline]
    pub fn mul(self, a: MontCoeff<W>, b: MontCoeff<W>) -> MontCoeff<W> {
        MontCoeff(self.mont_mul_raw(a.0, b.0))
    }

    /// Raw Montgomery multiply on bare `W` values.
    #[inline]
    pub(crate) fn mont_mul_raw(self, a: W, b: W) -> W {
        let c = W::wide_mul(a, b);
        let t = W::truncate(c).wrapping_mul(self.pinv);
        let tp = W::wide_mul(t, self.p);
        W::wide_shift(W::wide_sub(c, tp))
    }

    /// Conditionally subtract `p` if `a >= p` (branchless).
    #[inline]
    pub fn csubp(self, a: MontCoeff<W>) -> MontCoeff<W> {
        let diff = a.0.wrapping_sub(self.p);
        let mask = diff.sign_mask();
        MontCoeff(diff.wrapping_add(mask.bitand(self.p)))
    }

    /// Conditionally add `p` if `a < 0` (branchless).
    #[inline]
    pub fn caddp(self, a: MontCoeff<W>) -> MontCoeff<W> {
        let mask = a.0.sign_mask();
        MontCoeff(a.0.wrapping_add(mask.bitand(self.p)))
    }

    /// Range-reduce from `(-2p, 2p)` to `(-p, p)`.
    #[inline]
    pub fn reduce_range(self, a: MontCoeff<W>) -> MontCoeff<W> {
        self.caddp(self.csubp(a))
    }

    /// Fully normalize a Montgomery coefficient to `[0, p)`.
    #[inline]
    pub fn normalize(self, a: MontCoeff<W>) -> MontCoeff<W> {
        self.csubp(self.caddp(a))
    }

    /// Convert a canonical value into Montgomery domain: `a ↦ a * R mod p`.
    #[inline]
    pub fn from_canonical(self, a: W) -> MontCoeff<W> {
        MontCoeff(self.mont_mul_raw(a, self.montsq))
    }

    /// Convert from Montgomery domain to canonical `[0, p)`.
    #[inline]
    pub fn to_canonical(self, a: MontCoeff<W>) -> W {
        let raw = MontCoeff(self.mont_mul_raw(a.0, W::from_i64(1)));
        self.normalize(raw).0
    }

    /// Center a canonical value from approximately `(-p, p)` into `[-p/2, p/2)`.
    #[inline]
    pub fn center(self, a: W) -> W {
        let mask_neg = a.sign_mask();
        let canonical = a.wrapping_add(mask_neg.bitand(self.p));
        let half = W::from_i64(self.p.to_i64() / 2);
        let needs_sub = half.wrapping_sub(canonical).sign_mask();
        canonical.wrapping_add(needs_sub.bitand(self.p.wrapping_neg()))
    }

    /// Pointwise Montgomery multiplication of two coefficient slices.
    ///
    /// # Panics
    ///
    /// Panics if slices have different lengths.
    #[inline]
    pub fn pointwise_mul(
        self,
        out: &mut [MontCoeff<W>],
        lhs: &[MontCoeff<W>],
        rhs: &[MontCoeff<W>],
    ) {
        assert_eq!(out.len(), lhs.len());
        assert_eq!(lhs.len(), rhs.len());
        for ((o, a), b) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
            *o = self.mul(*a, *b);
        }
    }

    /// In-place Montgomery scaling by a constant.
    #[inline]
    pub fn scale_in_place(self, coeffs: &mut [MontCoeff<W>], scalar: MontCoeff<W>) {
        for c in coeffs {
            *c = self.mul(*c, scalar);
        }
    }

    /// In-place range reduction on a coefficient slice.
    #[inline]
    pub fn reduce_range_in_place(self, coeffs: &mut [MontCoeff<W>]) {
        for c in coeffs {
            *c = self.reduce_range(*c);
        }
    }

    /// In-place centering of canonical values to `[-p/2, p/2)`.
    #[inline]
    pub fn center_slice(self, coeffs: &mut [W]) {
        for c in coeffs {
            *c = self.center(*c);
        }
    }
}
