//! Utilities for the equality polynomial `eq(x, y) = Πᵢ (xᵢ yᵢ + (1 − xᵢ)(1 − yᵢ))`.
//!
//! The equality polynomial evaluates to 1 when `x = y` (over the boolean hypercube)
//! and 0 otherwise. Its multilinear extension (MLE) is used throughout sumcheck
//! protocols.
//!
//! Adapted from Jolt's `EqPolynomial` implementation.
//!
//! ## Bit / index order: Little-endian
//!
//! The evaluation tables produced by this module use **little-endian** bit order:
//! entry `b` (as an integer index) corresponds to the boolean vector where
//! bit `k` of `b` equals `x[k]`. In other words, `r[0]` corresponds to the
//! **least-significant bit** (bit 0) and `r[n-1]` to the MSB.

use crate::FieldCore;
use std::marker::PhantomData;

/// Utilities for the equality polynomial `eq(x, y) = Πᵢ (xᵢ yᵢ + (1 − xᵢ)(1 − yᵢ))`.
pub struct EqPolynomial<E: FieldCore>(PhantomData<E>);

impl<E: FieldCore> EqPolynomial<E> {
    /// Compute the MLE of the equality polynomial at two points:
    /// `eq(x, y) = Πᵢ (xᵢ yᵢ + (1 − xᵢ)(1 − yᵢ))`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != y.len()`.
    pub fn mle(x: &[E], y: &[E]) -> E {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(&x_i, &y_i)| x_i * y_i + (E::one() - x_i) * (E::one() - y_i))
            .fold(E::one(), |acc, v| acc * v)
    }

    /// Compute the zero selector: `eq(r, 0) = Πᵢ (1 − rᵢ)`.
    pub fn zero_selector(r: &[E]) -> E {
        r.iter().fold(E::one(), |acc, &r_i| acc * (E::one() - r_i))
    }

    /// Compute the full evaluation table `{ eq(r, x) : x ∈ {0,1}^n }`.
    ///
    /// Uses **little-endian** bit order: entry `b` has bit `k` of `b`
    /// corresponding to `r[k]`.
    ///
    /// For a scaled table, use [`Self::evals_with_scaling`].
    pub fn evals(r: &[E]) -> Vec<E> {
        Self::evals_with_scaling(r, None)
    }

    /// Compute the full evaluation table with optional scaling:
    /// `scaling_factor · eq(r, x)` for all `x ∈ {0,1}^n`.
    ///
    /// Uses the same **little-endian** index order as [`Self::evals`].
    /// If `scaling_factor` is `None`, defaults to 1 (no scaling).
    pub fn evals_with_scaling(r: &[E], scaling_factor: Option<E>) -> Vec<E> {
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 16;
            if r.len() > PARALLEL_THRESHOLD {
                return Self::evals_parallel(r, scaling_factor);
            }
        }
        Self::evals_serial(r, scaling_factor)
    }

    /// Serial (single-threaded) version of [`Self::evals_with_scaling`].
    ///
    /// Uses **little-endian** index order.
    pub fn evals_serial(r: &[E], scaling_factor: Option<E>) -> Vec<E> {
        let size = 1usize << r.len();
        let mut evals = vec![E::zero(); size];
        evals[0] = scaling_factor.unwrap_or(E::one());
        let mut len = 1usize;
        for &t in r.iter().rev() {
            let one_minus_t = E::one() - t;
            for j in (0..len).rev() {
                evals[2 * j + 1] = evals[j] * t;
                evals[2 * j] = evals[j] * one_minus_t;
            }
            len *= 2;
        }
        evals
    }

    /// Compute eq evaluations and cache intermediate tables.
    ///
    /// Returns `result` where `result[j]` contains evaluations for the prefix
    /// `r[..j]`: `result[j][x] = eq(r[..j], x)` for `x ∈ {0,1}^j`.
    ///
    /// So `result[0] = [1]`, `result[1]` has 2 entries, ..., and `result[n]`
    /// equals [`Self::evals(r)`].
    pub fn evals_cached(r: &[E]) -> Vec<Vec<E>> {
        Self::evals_cached_with_scaling(r, None)
    }

    /// Like [`Self::evals_cached`], but with optional scaling.
    pub fn evals_cached_with_scaling(r: &[E], scaling_factor: Option<E>) -> Vec<Vec<E>> {
        let mut result: Vec<Vec<E>> = (0..r.len() + 1).map(|i| vec![E::zero(); 1 << i]).collect();
        result[0][0] = scaling_factor.unwrap_or(E::one());
        for j in 0..r.len() {
            let idx = r.len() - 1 - j;
            let t = r[idx];
            let one_minus_t = E::one() - t;
            let prev_len = 1 << j;
            for i in (0..prev_len).rev() {
                result[j + 1][2 * i + 1] = result[j][i] * t;
                result[j + 1][2 * i] = result[j][i] * one_minus_t;
            }
        }
        result
    }

    /// Parallel version of [`Self::evals_with_scaling`].
    ///
    /// Uses rayon to compute the largest layers of the DP tree in parallel.
    /// Uses the same **little-endian** index order as [`Self::evals`].
    #[cfg(feature = "parallel")]
    pub fn evals_parallel(r: &[E], scaling_factor: Option<E>) -> Vec<E> {
        use rayon::prelude::*;

        let final_size = 1usize << r.len();
        let mut evals = vec![E::zero(); final_size];
        evals[0] = scaling_factor.unwrap_or(E::one());
        let mut size = 1;

        // Forward iteration (r[0] first) produces little-endian ordering.
        for &r_i in r.iter() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_i;
                    *x -= *y;
                });

            size *= 2;
        }

        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RandomSampling;
    use akita_field::Fp64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;

    #[test]
    fn evals_matches_mle_pointwise() {
        let mut rng = StdRng::seed_from_u64(0xEE);
        for n in 1..8 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let table = EqPolynomial::evals(&r);
            assert_eq!(table.len(), 1 << n);
            for (idx, &val) in table.iter().enumerate() {
                let bits: Vec<F> = (0..n)
                    .map(|k| {
                        if (idx >> k) & 1 == 1 {
                            F::one()
                        } else {
                            F::zero()
                        }
                    })
                    .collect();
                let expected = EqPolynomial::mle(&r, &bits);
                assert_eq!(val, expected, "n={n} idx={idx}");
            }
        }
    }

    #[test]
    fn evals_with_scaling_scales_uniformly() {
        let mut rng = StdRng::seed_from_u64(0xAB);
        let r: Vec<F> = (0..5).map(|_| F::random(&mut rng)).collect();
        let scale = F::from_u64(7);
        let unscaled = EqPolynomial::evals(&r);
        let scaled = EqPolynomial::evals_with_scaling(&r, Some(scale));
        for (u, s) in unscaled.iter().zip(scaled.iter()) {
            assert_eq!(*s, *u * scale);
        }
    }

    #[test]
    fn evals_cached_last_matches_evals() {
        let mut rng = StdRng::seed_from_u64(0xCD);
        for n in 1..8 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let table = EqPolynomial::evals(&r);
            let cached = EqPolynomial::evals_cached(&r);
            assert_eq!(cached.len(), n + 1);
            assert_eq!(cached[0], vec![F::one()]);
            assert_eq!(*cached.last().unwrap(), table);
        }
    }

    #[test]
    fn zero_selector_matches_mle_at_origin() {
        let mut rng = StdRng::seed_from_u64(0x00);
        for n in 1..8 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let zeros = vec![F::zero(); n];
            let expected = EqPolynomial::mle(&r, &zeros);
            let actual = EqPolynomial::zero_selector(&r);
            assert_eq!(actual, expected, "n={n}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn evals_parallel_matches_serial() {
        let mut rng = StdRng::seed_from_u64(0xFF);
        for n in 1..20 {
            let r: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let serial = EqPolynomial::evals_serial(&r, None);
            let parallel = EqPolynomial::evals_parallel(&r, None);
            assert_eq!(serial, parallel, "n={n}");
        }
    }
}
