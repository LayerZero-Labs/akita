//! Small-prime arithmetic kernels for NTT-friendly parameter sets.
//!
//! Per-prime scalar operations:
//! - Montgomery-like multiplication (`fpmul`)
//! - Barrett-like reduction (`fpred`)
//! - Conditional add/sub by `p`

/// Per-prime constants for NTT arithmetic over a small prime `p < 2^14`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NttPrime {
    /// Prime modulus.
    pub p: i16,
    /// `p^-1 mod 2^16` in centered signed representation.
    pub pinv: i16,
    /// `round(2^27 / p)` for Barrett-style reduction.
    pub v: i16,
    /// `2^16 mod p` in centered signed representation.
    pub mont: i16,
    /// `2^32 mod p` in centered signed representation.
    pub montsq: i16,
    /// Scaling constant for injection into NTT representation.
    pub s: i16,
    /// Inverse-NTT post-scale helper constant.
    pub f: i16,
    /// CRT reconstruction helper constant.
    pub t: i16,
}

impl NttPrime {
    /// Labrador-style Montgomery-like product:
    /// `((a*b - ((a*b)*pinv mod 2^16)*p) >> 16)`.
    #[inline]
    pub fn fpmul(self, a: i16, b: i16) -> i16 {
        let c = (a as i32) * (b as i32);
        let t = (c as i16).wrapping_mul(self.pinv) as i32;
        ((c - t * (self.p as i32)) >> 16) as i16
    }

    /// Labrador-style Barrett-like reducer used for keeping values near `[0, p)`.
    #[inline]
    pub fn fpred(self, a: i16) -> i16 {
        let mut t = (((self.v as i32) * (a as i32) + (1 << 26)) >> 27) as i16;
        t = t.wrapping_mul(self.p);
        a.wrapping_sub(t)
    }

    /// Conditionally subtract `p` if `a >= p` (branchless).
    #[inline]
    pub fn csubq(self, a: i16) -> i16 {
        let diff = a.wrapping_sub(self.p);
        // Arithmetic right shift: -1 (all-ones) if diff < 0, 0 otherwise.
        let mask = diff >> 15;
        diff.wrapping_add(mask & self.p)
    }

    /// Conditionally add `p` if `a < 0` (branchless).
    #[inline]
    pub fn caddq(self, a: i16) -> i16 {
        let mask = a >> 15;
        a.wrapping_add(mask & self.p)
    }

    /// Center a canonical residue into `[-p/2, p/2)` (branchless).
    #[inline]
    pub fn center(self, a: i16) -> i16 {
        let half = self.p / 2;
        // Arithmetic right shift: -1 if a > half (i.e. half - a < 0), 0 otherwise.
        let needs_sub = half.wrapping_sub(a) >> 15;
        a.wrapping_add(needs_sub & self.p.wrapping_neg())
    }

    /// Normalize to canonical range `[0, p)`.
    #[inline]
    pub fn normalize(self, a: i16) -> i16 {
        self.csubq(self.caddq(self.fpred(a)))
    }
}

/// Scalar reference pointwise multiplication in the NTT domain.
///
/// # Panics
///
/// Panics if `out`, `lhs`, and `rhs` do not have identical lengths.
#[inline]
pub fn pointwise_mul(out: &mut [i16], lhs: &[i16], rhs: &[i16], prime: NttPrime) {
    assert_eq!(out.len(), lhs.len());
    assert_eq!(lhs.len(), rhs.len());
    for ((o, a), b) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
        *o = prime.fpmul(*a, *b);
    }
}

/// In-place `fpmul` on a slice by a scalar.
#[inline]
pub fn scale_montgomery_in_place(coeffs: &mut [i16], s: i16, prime: NttPrime) {
    for c in coeffs {
        *c = prime.fpmul(*c, s);
    }
}

/// In-place `fpred` on a slice.
#[inline]
pub fn reduce_in_place(coeffs: &mut [i16], prime: NttPrime) {
    for c in coeffs {
        *c = prime.fpred(*c);
    }
}

/// In-place centered normalization on a slice.
#[inline]
pub fn center_in_place(coeffs: &mut [i16], prime: NttPrime) {
    for c in coeffs {
        *c = prime.center(prime.caddq(*c));
    }
}
