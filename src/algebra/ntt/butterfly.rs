//! NTT butterfly transforms for negacyclic rings `Z_p[X]/(X^D + 1)`.
//!
//! Implements a negacyclic NTT via the standard **twist + cyclic NTT** method.
//!
//! Let `psi` be a primitive `2D`-th root of unity (`psi^D = -1 mod p`) and
//! `omega = psi^2`, a primitive `D`-th root of unity. For polynomials modulo
//! `X^D + 1`, we:
//! - pre-twist coefficients by `psi^i`
//! - run a cyclic size-`D` NTT using `omega`
//! - inverse-cyclic NTT using `omega^{-1}`
//! - post-untwist by `psi^{-i}`

use super::prime::{MontCoeff, NttPrime, PrimeWidth};

/// Precomputed twiddle factors for a specific prime and degree `D`.
///
/// `D` must be a power of two.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NttTwiddles<W: PrimeWidth, const D: usize> {
    /// Stage roots for iterative forward cyclic NTT in Montgomery form.
    pub(crate) fwd_wlen: [MontCoeff<W>; D],
    /// Stage roots for iterative inverse cyclic NTT in Montgomery form.
    pub(crate) inv_wlen: [MontCoeff<W>; D],
    /// Number of active stages in the twiddle arrays (`log2(D)`).
    pub(crate) num_stages: usize,
    /// Twist factors `psi^i` for negacyclic embedding, in Montgomery form.
    pub(crate) psi_pows: [MontCoeff<W>; D],
    /// Untwist factors `psi^{-i}`, in Montgomery form.
    pub(crate) psi_inv_pows: [MontCoeff<W>; D],
    /// `D^{-1} mod p` in Montgomery form, used for inverse NTT final scaling.
    pub(crate) d_inv: MontCoeff<W>,
    /// Fused `D^{-1} * psi^{-i}` for each index, in Montgomery form.
    pub(crate) d_inv_psi_inv: [MontCoeff<W>; D],
}

impl<W: PrimeWidth, const D: usize> NttTwiddles<W, D> {
    /// Compute twiddle factors for the given prime.
    ///
    /// Finds a primitive `2D`-th root `psi` and derives `omega = psi^2`.
    /// Fills cyclic forward/inverse twiddles for `omega` and twist/untwist
    /// tables for `psi`. All values are stored in Montgomery form.
    ///
    /// # Panics
    ///
    /// Panics if `D` is not a power of two, or if `2D` does not divide `p - 1`.
    pub fn compute(prime: NttPrime<W>) -> Self {
        assert!(D.is_power_of_two(), "D must be a power of two");
        let p = prime.p.to_i64();
        assert!(
            (p - 1) % (2 * D as i64) == 0,
            "2D must divide p - 1 for NTT roots to exist"
        );

        let psi = find_primitive_root_2d(p, D);
        let omega = (psi * psi) % p;
        let omega_inv = pow_mod(omega, p - 2, p);

        let psi_inv = pow_mod(psi, p - 2, p);
        let mut psi_pows = [MontCoeff::from_raw(W::default()); D];
        let mut psi_inv_pows = [MontCoeff::from_raw(W::default()); D];
        let mut cur = 1i64;
        let mut cur_inv = 1i64;
        for i in 0..D {
            psi_pows[i] = prime.from_canonical(W::from_i64(cur));
            psi_inv_pows[i] = prime.from_canonical(W::from_i64(cur_inv));
            cur = (cur * psi) % p;
            cur_inv = (cur_inv * psi_inv) % p;
        }

        let mut fwd_wlen = [MontCoeff::from_raw(W::default()); D];
        let mut inv_wlen = [MontCoeff::from_raw(W::default()); D];
        let mut len = 1usize;
        let mut stage = 0usize;
        while len < D {
            let exp = (D / (2 * len)) as i64;
            fwd_wlen[stage] = prime.from_canonical(W::from_i64(pow_mod(omega, exp, p)));
            inv_wlen[stage] = prime.from_canonical(W::from_i64(pow_mod(omega_inv, exp, p)));
            len *= 2;
            stage += 1;
        }

        let d_inv_canonical = pow_mod(D as i64, p - 2, p);
        let d_inv = prime.from_canonical(W::from_i64(d_inv_canonical));

        let mut d_inv_psi_inv = [MontCoeff::from_raw(W::default()); D];
        for i in 0..D {
            d_inv_psi_inv[i] = prime.mul(d_inv, psi_inv_pows[i]);
        }

        Self {
            fwd_wlen,
            inv_wlen,
            num_stages: stage,
            psi_pows,
            psi_inv_pows,
            d_inv,
            d_inv_psi_inv,
        }
    }
}

/// Forward negacyclic NTT (twist + cyclic Gentleman-Sande DIF).
///
/// Transforms `D` coefficients in-place from coefficient form to NTT
/// evaluation form. Both outputs of each butterfly are range-reduced
/// to prevent overflow.
pub fn forward_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    for (ai, psi) in a.iter_mut().zip(tw.psi_pows.iter()) {
        *ai = prime.mul(*ai, *psi);
    }

    let one = prime.from_canonical(W::from_i64(1));

    let mut len = D / 2;
    let mut stage = tw.num_stages;
    while len > 0 {
        stage -= 1;
        let wlen = tw.fwd_wlen[stage];
        let mut start = 0usize;
        while start < D {
            let mut w = one;
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len];
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                w = prime.mul(w, wlen);
            }
            start += 2 * len;
        }
        len /= 2;
    }

    // Keep exported NTT-domain coefficients in the same reduced range expected
    // by add/sub reduced operations and equality checks.
    prime.reduce_range_in_place(a);
}

/// Inverse negacyclic NTT (cyclic Cooley-Tukey DIT + untwist).
///
/// Transforms `D` evaluations in-place back to coefficient form.
/// Includes the final `D^{-1}` scaling.
pub fn inverse_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let one = prime.from_canonical(W::from_i64(1));

    let mut len = 1usize;
    let mut stage = 0usize;
    while len < D {
        let wlen = tw.inv_wlen[stage];
        let mut start = 0usize;
        while start < D {
            let mut w = one;
            for j in 0..len {
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], w);
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                w = prime.mul(w, wlen);
            }
            start += 2 * len;
        }
        len *= 2;
        stage += 1;
    }

    for (ai, fused) in a.iter_mut().zip(tw.d_inv_psi_inv.iter()) {
        *ai = prime.mul(*ai, *fused);
    }
}

/// Forward cyclic NTT (Gentleman-Sande DIF, **no** negacyclic twist).
///
/// Evaluates a polynomial at the D-th roots of *unity* (roots of X^D - 1)
/// rather than X^D + 1. Used with `inverse_ntt_cyclic` to compute unreduced
/// polynomial products via CRT over (X^D - 1)(X^D + 1).
pub fn forward_ntt_cyclic<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let one = prime.from_canonical(W::from_i64(1));
    let mut len = D / 2;
    let mut stage = tw.num_stages;
    while len > 0 {
        stage -= 1;
        let wlen = tw.fwd_wlen[stage];
        let mut start = 0usize;
        while start < D {
            let mut w = one;
            for j in 0..len {
                let u = a[start + j];
                let v = a[start + j + len];
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.mul(MontCoeff::from_raw(diff), w);
                w = prime.mul(w, wlen);
            }
            start += 2 * len;
        }
        len /= 2;
    }
    prime.reduce_range_in_place(a);
}

/// Inverse cyclic NTT (Cooley-Tukey DIT, **no** negacyclic untwist).
///
/// Recovers coefficients of a polynomial from evaluations at D-th roots of unity.
/// Includes the `D^{-1}` scaling factor.
pub fn inverse_ntt_cyclic<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let one = prime.from_canonical(W::from_i64(1));
    let mut len = 1usize;
    let mut stage = 0usize;
    while len < D {
        let wlen = tw.inv_wlen[stage];
        let mut start = 0usize;
        while start < D {
            let mut w = one;
            for j in 0..len {
                let u = a[start + j];
                let v = prime.mul(a[start + j + len], w);
                let sum = u.raw().wrapping_add(v.raw());
                let diff = u.raw().wrapping_sub(v.raw());
                a[start + j] = prime.reduce_range(MontCoeff::from_raw(sum));
                a[start + j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                w = prime.mul(w, wlen);
            }
            start += 2 * len;
        }
        len *= 2;
        stage += 1;
    }

    for c in a.iter_mut() {
        *c = prime.mul(*c, tw.d_inv);
    }
}

/// Find a primitive `2D`-th root of unity mod `p`.
fn find_primitive_root_2d(p: i64, d: usize) -> i64 {
    let half = (p - 1) / 2;
    let exp = (p - 1) / (2 * d as i64);
    for a in 2..p {
        if pow_mod(a, half, p) == p - 1 {
            let psi = pow_mod(a, exp, p);
            debug_assert_eq!(pow_mod(psi, d as i64, p), p - 1, "psi^D != -1");
            return psi;
        }
    }
    panic!("no primitive root found for p={p}");
}

/// Modular exponentiation: `base^exp mod modulus`.
fn pow_mod(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }
    result
}
