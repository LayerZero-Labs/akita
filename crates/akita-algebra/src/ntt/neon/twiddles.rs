//! Barrett-form twiddle tables for the NEON transforms.
//!
//! A constant `w` is stored as its centered plain residue (`|w| <= p/2`, not
//! in Montgomery form) together with the quotient
//! `w' = round(w * 2^(R_LOG - 1) / p)`. The NEON kernels multiply by such a
//! constant with `sqrdmulh`, `mul` and `mls`; see `barrett_mul_4x_i32`. A
//! plain multiplier preserves the Montgomery scaling of the other operand.

use crate::ntt::prime::{MontCoeff, NttPrime, PrimeWidth};

/// One constant multiplier with its Barrett quotient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct BarrettConstant<W: PrimeWidth> {
    pub(crate) value: W,
    pub(crate) quotient: W,
}

/// `D` constant multipliers with their Barrett quotients, stored as two
/// parallel arrays so four consecutive entries load as one vector each.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct BarrettTable<W: PrimeWidth, const D: usize> {
    pub(crate) values: [W; D],
    pub(crate) quotients: [W; D],
}

/// Barrett-form tables consumed by the NEON transforms.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct BarrettTwiddles<W: PrimeWidth, const D: usize> {
    /// Forward stage twiddles, packed like `NttTwiddles::fwd_twiddles`.
    pub(crate) fwd: BarrettTable<W, D>,
    /// Inverse stage twiddles, packed like `NttTwiddles::inv_twiddles`.
    pub(crate) inv: BarrettTable<W, D>,
    /// First negacyclic forward stage with the `psi^i` twist folded in.
    /// Entry `j < D/2` is `psi^j`, and entry `D/2 + j` is `psi^j * w_j`,
    /// where `w_j` is the stage-`D/2` forward twiddle.
    pub(crate) twist: BarrettTable<W, D>,
    /// `twist` scaled by `R`, for plain-integer inputs such as signed
    /// digits. The first stage then also enters Montgomery form.
    pub(crate) twist_digits: BarrettTable<W, D>,
    /// Last negacyclic inverse stage scaling: entry `i` is `D^{-1} psi^{-i}`.
    pub(crate) untwist: BarrettTable<W, D>,
    /// `psi^{D/2}`, a square root of `-1`.
    pub(crate) quarter_root: BarrettConstant<W>,
    /// `D^{-1}`, the last cyclic inverse stage scaling.
    pub(crate) d_inv: BarrettConstant<W>,
}

impl<W: PrimeWidth, const D: usize> BarrettTwiddles<W, D> {
    /// Derive the Barrett tables from the Montgomery-form tables.
    pub(crate) fn compute(
        prime: NttPrime<W>,
        fwd_twiddles: &[MontCoeff<W>; D],
        inv_twiddles: &[MontCoeff<W>; D],
        psi_pows: &[MontCoeff<W>; D],
        d_inv_psi_inv: &[MontCoeff<W>; D],
        d_inv: MontCoeff<W>,
    ) -> Self {
        let p = i128::from(prime.p.to_i64());
        let plain = |value: MontCoeff<W>| i128::from(prime.to_canonical(value).to_i64());
        let constant = |value: i128| {
            let value = value.rem_euclid(p);
            let value = if value > p / 2 { value - p } else { value };
            // round(value * 2^(R_LOG - 1) / p) = floor((2 * value * 2^(R_LOG - 1) + p) / 2p).
            let quotient = ((value << W::R_LOG) + p).div_euclid(2 * p);
            BarrettConstant {
                value: W::from_i64(value as i64),
                quotient: W::from_i64(quotient as i64),
            }
        };
        let table = |entry: &dyn Fn(usize) -> i128| {
            let entries: [BarrettConstant<W>; D] = std::array::from_fn(|i| constant(entry(i)));
            BarrettTable {
                values: entries.map(|entry| entry.value),
                quotients: entries.map(|entry| entry.quotient),
            }
        };

        let half = D / 2;
        let r = (1i128 << W::R_LOG) % p;
        let twist_entry = |i: usize| {
            if i < half {
                plain(psi_pows[i])
            } else if half == 0 {
                1
            } else {
                let j = i - half;
                plain(psi_pows[j]) * plain(fwd_twiddles[half - 1 + j]) % p
            }
        };

        Self {
            fwd: table(&|i| plain(fwd_twiddles[i])),
            inv: table(&|i| plain(inv_twiddles[i])),
            twist: table(&twist_entry),
            twist_digits: table(&|i| twist_entry(i) * r % p),
            untwist: table(&|i| plain(d_inv_psi_inv[i])),
            quarter_root: constant(if D >= 2 { plain(psi_pows[half]) } else { 1 }),
            d_inv: constant(plain(d_inv)),
        }
    }
}
