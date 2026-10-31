//! Sums of two-factor products at the structured setup-scan sites (feature `setup-sop-structured`, default off).
//!
//! `lb2_alt!` selects, at each call site, between the original tokens (`off`, compiled without the feature, token for
//! token as before) and the summed path (`on`). The call sites keep their line numbers: the selector opens on the first
//! line and closes on the last line of the original statement, and every new item is appended after the last original
//! line of its file. Without the feature the expansion is exactly the original tokens, so the guest image is unchanged.
//!
//! [`PairSum`] buffers up to [`PAIR_CHUNK`] factor pairs on the stack and hands each full chunk to
//! [`Field::sum_of_products`]. On a field-inline guest that call keeps the running sum in the field register file and
//! reads out one value per chunk, where the plain `sum += a * b` loop reads out every product and adds it in software.
//! The result is the same field element on every path. A chunk holding a single pair takes the plain multiply, so a
//! one-product sum costs what it did before.

/// Feature-off: the original tokens.
#[cfg(not(feature = "setup-sop-structured"))]
macro_rules! lb2_alt {
    (off { $($off:tt)* } on { $($on:tt)* }) => { $($off)* };
}
/// Feature-on: the sums-of-products path.
#[cfg(feature = "setup-sop-structured")]
macro_rules! lb2_alt {
    (off { $($off:tt)* } on { $($on:tt)* }) => { $($on)* };
}
pub(super) use lb2_alt;

#[cfg(any(feature = "setup-sop-structured", test))]
use core::mem::MaybeUninit;
#[cfg(any(feature = "setup-sop-structured", test))]
use jolt_field::Field;

/// Pairs per [`Field::sum_of_products`] call.
#[cfg(any(feature = "setup-sop-structured", test))]
pub(super) const PAIR_CHUNK: usize = 8;

/// `Σ a_i · b_i` over the pushed pairs.
#[cfg(any(feature = "setup-sop-structured", test))]
pub(super) struct PairSum<E: Field> {
    buf: [MaybeUninit<[E; 2]>; PAIR_CHUNK],
    len: usize,
    sum: Option<E>,
}

#[cfg(any(feature = "setup-sop-structured", test))]
impl<E: Field> PairSum<E> {
    #[inline(always)]
    pub(super) fn new() -> Self {
        Self {
            buf: [const { MaybeUninit::uninit() }; PAIR_CHUNK],
            len: 0,
            sum: None,
        }
    }

    #[inline(always)]
    pub(super) fn push(&mut self, a: E, b: E) {
        if self.len == PAIR_CHUNK {
            self.flush();
        }
        self.buf[self.len].write([a, b]);
        self.len += 1;
    }

    #[inline(always)]
    fn flush(&mut self) {
        // SAFETY: `push` wrote the first `len` entries of `buf`, and
        // `MaybeUninit<[E; 2]>` has the layout of `[E; 2]`.
        let pairs: &[[E; 2]] =
            unsafe { core::slice::from_raw_parts(self.buf.as_ptr().cast::<[E; 2]>(), self.len) };
        let chunk = match pairs {
            [] => return,
            [[a, b]] => *a * *b,
            _ => E::sum_of_products(pairs),
        };
        self.len = 0;
        self.sum = Some(match self.sum {
            Some(sum) => sum + chunk,
            None => chunk,
        });
    }

    /// The sum of every pushed product (zero when none was pushed).
    #[inline(always)]
    pub(super) fn finish(mut self) -> E {
        self.flush();
        self.sum.unwrap_or_else(E::zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Prime128OffsetA7F7, Ring, Zero};

    type F = Prime128OffsetA7F7;

    /// Deterministic full-width elements (a 64-bit LCG, two draws per element).
    fn elements(seed: u64, n: usize) -> Vec<F> {
        let mut state = seed;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        (0..n)
            .map(|_| {
                F::from_u64(next()) * F::from_u64(1 << 63) * F::from_u64(2) + F::from_u64(next())
            })
            .collect()
    }

    fn plain(a: &[F], b: &[F]) -> F {
        a.iter().zip(b).fold(F::zero(), |sum, (&x, &y)| sum + x * y)
    }

    #[test]
    fn pair_sum_matches_plain_fold_for_every_length() {
        for len in 0..=3 * PAIR_CHUNK + 1 {
            for seed in 0..4u64 {
                let a = elements(2 * seed + 1, len);
                let b = elements(2 * seed + 2 + 100 * len as u64, len);
                let mut sum = PairSum::new();
                for (&x, &y) in a.iter().zip(&b) {
                    sum.push(x, y);
                }
                assert_eq!(sum.finish(), plain(&a, &b), "length {len}");
            }
        }
    }

    #[test]
    fn pair_sum_matches_sum_of_products() {
        let a = elements(7, 2 * PAIR_CHUNK + 3);
        let b = elements(11, 2 * PAIR_CHUNK + 3);
        let terms: Vec<[F; 2]> = a.iter().zip(&b).map(|(&x, &y)| [x, y]).collect();
        let mut sum = PairSum::new();
        for &[x, y] in &terms {
            sum.push(x, y);
        }
        assert_eq!(sum.finish(), F::sum_of_products(&terms));
    }

    #[test]
    fn pair_sum_of_nothing_is_zero() {
        assert_eq!(PairSum::<F>::new().finish(), F::zero());
    }
}
