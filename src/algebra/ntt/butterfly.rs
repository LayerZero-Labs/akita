//! NTT butterfly transforms for negacyclic rings `Z_p[X]/(X^D + 1)`.
//!
//! Uses a merged negacyclic Cooley-Tukey / Gentleman-Sande butterfly where
//! the twist factors for `X^D + 1` are folded directly into the twiddles.
//! Twiddle factors are powers of `psi`, a primitive `2D`-th root of unity
//! (`psi^D = -1 mod p`), rather than a `D`-th root.

use super::prime::{MontCoeff, NttPrime, PrimeWidth};

/// Precomputed twiddle factors for a specific prime and degree `D`.
///
/// `D` must be a power of two.
pub struct NttTwiddles<W: PrimeWidth, const D: usize> {
    /// Twiddle factors in Montgomery form, indexed by bit-reversed position.
    pub(crate) zetas: [MontCoeff<W>; D],
    /// `D^{-1} mod p` in Montgomery form, used for inverse NTT final scaling.
    pub(crate) d_inv: MontCoeff<W>,
}

impl<W: PrimeWidth, const D: usize> NttTwiddles<W, D> {
    /// Compute twiddle factors for the given prime.
    ///
    /// Finds a primitive `2D`-th root of unity mod `p`, then fills the
    /// twiddle table in bit-reversed order. All values are stored in
    /// Montgomery form.
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

        let n = D.trailing_zeros();
        let psi = find_primitive_root_2d(p, D);

        let mut zetas = [MontCoeff::from_raw(W::default()); D];
        for (i, z) in zetas.iter_mut().enumerate() {
            let brv_i = bit_reverse(i, n);
            let power = pow_mod(psi, brv_i as i64, p);
            *z = prime.from_canonical(W::from_i64(power));
        }

        let d_inv_canonical = pow_mod(D as i64, p - 2, p);
        let d_inv = prime.from_canonical(W::from_i64(d_inv_canonical));

        Self { zetas, d_inv }
    }
}

/// Forward negacyclic NTT (Cooley-Tukey, decimation-in-time).
///
/// Transforms `D` coefficients in-place from coefficient form to NTT
/// evaluation form. Both outputs of each butterfly are range-reduced
/// to prevent overflow.
pub fn forward_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let mut k = 1usize;
    let mut len = D / 2;
    while len >= 1 {
        let mut start = 0;
        while start < D {
            let zeta = tw.zetas[k];
            k += 1;
            for j in start..(start + len) {
                let t = prime.mul(a[j + len], zeta);
                let diff = a[j].raw().wrapping_sub(t.raw());
                let sum = a[j].raw().wrapping_add(t.raw());
                a[j + len] = prime.reduce_range(MontCoeff::from_raw(diff));
                a[j] = prime.reduce_range(MontCoeff::from_raw(sum));
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

/// Inverse negacyclic NTT (Gentleman-Sande, decimation-in-frequency).
///
/// Transforms `D` evaluations in-place back to coefficient form.
/// Includes the final `D^{-1}` scaling.
pub fn inverse_ntt<W: PrimeWidth, const D: usize>(
    a: &mut [MontCoeff<W>; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
) {
    let mut k = D - 1;
    let mut len = 1;
    while len <= D / 2 {
        let mut start = 0;
        while start < D {
            let zeta = tw.zetas[k];
            k = k.wrapping_sub(1);
            for j in start..(start + len) {
                let t = a[j];
                let sum = t.raw().wrapping_add(a[j + len].raw());
                let diff = t.raw().wrapping_sub(a[j + len].raw());
                a[j] = prime.reduce_range(MontCoeff::from_raw(sum));
                let neg_zeta = MontCoeff::from_raw(zeta.raw().wrapping_neg());
                a[j + len] = prime.mul(MontCoeff::from_raw(diff), neg_zeta);
            }
            start += 2 * len;
        }
        len *= 2;
    }

    for c in a.iter_mut() {
        *c = prime.mul(*c, tw.d_inv);
    }
}

fn bit_reverse(x: usize, n: u32) -> usize {
    x.reverse_bits() >> (usize::BITS - n)
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
