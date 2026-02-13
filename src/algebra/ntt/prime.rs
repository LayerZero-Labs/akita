//! Small-prime arithmetic kernels for NTT-friendly parameter sets.
//!
//! Per-prime scalar operations:
//! - Montgomery multiplication ([`NttPrime::mul`])
//! - Barrett-style reduction ([`NttPrime::reduce`])
//! - Branchless conditional add/sub
//!
//! Coefficients in Montgomery domain are wrapped in [`MontCoeff`] to prevent
//! accidental mixing with canonical `i16` values.

use std::fmt;

/// A coefficient in Montgomery domain for an NTT prime.
///
/// Wraps an `i16` representing `a * R mod p` (where `R = 2^16`).
/// Use [`NttPrime::from_canonical`] to enter and [`NttPrime::to_canonical`]
/// to leave Montgomery domain.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct MontCoeff(i16);

impl MontCoeff {
    /// Wrap a raw Montgomery-domain value.
    ///
    /// The caller must ensure `val` is a valid Montgomery-domain representative
    /// for the intended prime.
    #[inline]
    pub const fn from_raw(val: i16) -> Self {
        Self(val)
    }

    /// Extract the raw `i16` (still in Montgomery domain).
    #[inline]
    pub const fn raw(self) -> i16 {
        self.0
    }
}

impl fmt::Debug for MontCoeff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mont({})", self.0)
    }
}

/// Per-prime constants for NTT arithmetic over a small prime `p < 2^14`.
///
/// All scalar operations use Montgomery representation internally. Constants
/// are stored as raw `i16` in centered signed form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttPrime {
    /// Prime modulus.
    pub p: i16,
    /// `p^{-1} mod 2^{16}` (centered signed). Used in Montgomery reduction.
    pub pinv: i16,
    /// `round(2^{27} / p)` for Barrett-style range reduction.
    pub v: i16,
    /// `R mod p` where `R = 2^{16}` (centered signed). Montgomery form of 1.
    pub mont: i16,
    /// `R^2 mod p` (centered signed). Used for canonical → Montgomery conversion.
    pub montsq: i16,
    /// NTT injection scale factor (centered signed).
    pub s: i16,
    /// Inverse-NTT post-scale factor (centered signed).
    pub f: i16,
    /// CRT reconstruction factor (centered signed).
    pub t: i16,
}

impl NttPrime {
    /// Montgomery product: `a * b * R^{-1} mod p` where `R = 2^{16}`.
    #[inline]
    pub fn mul(self, a: MontCoeff, b: MontCoeff) -> MontCoeff {
        MontCoeff(self.mont_mul_raw(a.0, b.0))
    }

    /// Raw Montgomery multiply on bare `i16` values.
    ///
    /// Computes `(a*b - ((a*b mod R) * p_inv mod R) * p) / R`.
    #[inline]
    pub(crate) fn mont_mul_raw(self, a: i16, b: i16) -> i16 {
        let c = (a as i32) * (b as i32);
        let t = (c as i16).wrapping_mul(self.pinv) as i32;
        ((c - t * (self.p as i32)) >> 16) as i16
    }

    /// Barrett-style reduction: tighten a Montgomery coefficient's range
    /// toward `[0, p)` without full normalization.
    #[inline]
    pub fn reduce(self, a: MontCoeff) -> MontCoeff {
        let val = a.0;
        let mut t = (((self.v as i32) * (val as i32) + (1 << 26)) >> 27) as i16;
        t = t.wrapping_mul(self.p);
        MontCoeff(val.wrapping_sub(t))
    }

    /// Conditionally subtract `p` if `a >= p` (branchless).
    #[inline]
    pub fn csubp(self, a: MontCoeff) -> MontCoeff {
        let diff = a.0.wrapping_sub(self.p);
        let mask = diff >> 15;
        MontCoeff(diff.wrapping_add(mask & self.p))
    }

    /// Conditionally add `p` if `a < 0` (branchless).
    #[inline]
    pub fn caddp(self, a: MontCoeff) -> MontCoeff {
        let mask = a.0 >> 15;
        MontCoeff(a.0.wrapping_add(mask & self.p))
    }

    /// Fully normalize a Montgomery coefficient to `[0, p)`.
    #[inline]
    pub fn normalize(self, a: MontCoeff) -> MontCoeff {
        self.csubp(self.caddp(self.reduce(a)))
    }

    /// Convert a canonical `i16` into Montgomery domain: `a ↦ a * R mod p`.
    #[inline]
    pub fn from_canonical(self, a: i16) -> MontCoeff {
        MontCoeff(self.mont_mul_raw(a, self.montsq))
    }

    /// Convert from Montgomery domain to canonical `[0, p)`.
    #[inline]
    pub fn to_canonical(self, a: MontCoeff) -> i16 {
        let raw = MontCoeff(self.mont_mul_raw(a.0, 1));
        self.normalize(raw).0
    }

    /// Center a canonical value from approximately `(-p, p)` into `[-p/2, p/2)`.
    ///
    /// First conditionally adds `p` (if negative), then subtracts `p` (if > p/2).
    /// Both steps are branchless.
    #[inline]
    pub fn center(self, a: i16) -> i16 {
        let mask_neg = a >> 15;
        let canonical = a.wrapping_add(mask_neg & self.p);
        let half = self.p / 2;
        let needs_sub = half.wrapping_sub(canonical) >> 15;
        canonical.wrapping_add(needs_sub & self.p.wrapping_neg())
    }

    /// Pointwise Montgomery multiplication of two coefficient slices.
    ///
    /// # Panics
    ///
    /// Panics if slices have different lengths.
    #[inline]
    pub fn pointwise_mul(self, out: &mut [MontCoeff], lhs: &[MontCoeff], rhs: &[MontCoeff]) {
        assert_eq!(out.len(), lhs.len());
        assert_eq!(lhs.len(), rhs.len());
        for ((o, a), b) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
            *o = self.mul(*a, *b);
        }
    }

    /// In-place Montgomery scaling by a constant.
    #[inline]
    pub fn scale_in_place(self, coeffs: &mut [MontCoeff], scalar: MontCoeff) {
        for c in coeffs {
            *c = self.mul(*c, scalar);
        }
    }

    /// In-place Barrett reduction on a coefficient slice.
    #[inline]
    pub fn reduce_in_place(self, coeffs: &mut [MontCoeff]) {
        for c in coeffs {
            *c = self.reduce(*c);
        }
    }

    /// In-place centering of canonical values to `[-p/2, p/2)`.
    #[inline]
    pub fn center_slice(self, coeffs: &mut [i16]) {
        for c in coeffs {
            *c = self.center(*c);
        }
    }
}
