//! Gruen/Dao-Thaler split equality polynomial for efficient sumcheck.
//!
//! Factors `eq(τ, x)` into a running scalar, a linear factor for the current
//! variable, and precomputed split tables for the remaining variables. This
//! avoids maintaining and folding a full-size eq table during sumcheck.
//!
//! For details, see <https://eprint.iacr.org/2024/1210.pdf>.
//!
//! Adapted from Jolt's `GruenSplitEqPolynomial`.
//!
//! ## Variable Layout (forward binding, little-endian)
//!
//! ```text
//! τ = [τ_current, τ_first_half, τ_second_half]
//!       1 var       m vars         (n-1-m) vars
//! ```
//!
//! where `m = (n-1) / 2` and `n = τ.len()`.
//!
//! After binding `τ_current`, the next variable comes from `τ_first_half`,
//! then from `τ_second_half`. Suffix-cached eq tables for each half enable
//! O(1) pops per round instead of an O(2^n) fold.

use super::eq_poly::EqPolynomial;
use super::UniPoly;
use crate::{CanonicalField, FieldCore};

/// Split equality polynomial with Gruen scalar accumulation.
///
/// Instead of storing and folding a full eq table each round, this struct
/// maintains:
/// - `current_scalar`: accumulated `eq(τ_bound, r_bound)` from already-bound
///   variables
/// - `E_first` / `E_second`: suffix-cached eq tables for two halves of the
///   remaining (unbound, non-current) variables
///
/// The eq contribution for a pair index `j` in the inner sum is:
/// ```text
/// eq_remaining(j) = E_first[j_low] · E_second[j_high]
/// ```
/// and the full round polynomial is `l(X) · q(X)` where `l(X)` is the linear
/// eq factor for the current variable.
#[allow(non_snake_case)]
pub struct GruenSplitEq<E: FieldCore> {
    tau: Vec<E>,
    current_round: usize,
    current_scalar: E,
    /// Suffix-cached eq tables for the first half of remaining variables.
    /// `E_first[k]` = `eq(τ[split-k..split], ·)` with `2^k` entries.
    /// Invariant: never empty; `E_first[0] = [1]`.
    E_first: Vec<Vec<E>>,
    /// Suffix-cached eq tables for the second half of remaining variables.
    /// `E_second[k]` = `eq(τ[n-k..n], ·)` with `2^k` entries.
    /// Invariant: never empty; `E_second[0] = [1]`.
    E_second: Vec<Vec<E>>,
}

#[allow(non_snake_case)]
impl<E: FieldCore + CanonicalField> GruenSplitEq<E> {
    /// Create a new split-eq from the full challenge vector `τ`.
    ///
    /// Precomputes suffix-cached eq tables for two halves of `τ[1..n]`.
    ///
    /// # Panics
    ///
    /// Panics if `tau` is empty.
    pub fn new(tau: &[E]) -> Self {
        let n = tau.len();
        assert!(n >= 1);
        let m = (n - 1) / 2;
        let split = 1 + m;
        let first_half = &tau[1..split];
        let second_half = &tau[split..n];
        let E_first = EqPolynomial::evals_cached(first_half);
        let E_second = EqPolynomial::evals_cached(second_half);
        Self {
            tau: tau.to_vec(),
            current_round: 0,
            current_scalar: E::one(),
            E_first,
            E_second,
        }
    }

    /// The accumulated scalar `Π_{k < current_round} eq(τ[k], r[k])`.
    pub fn current_scalar(&self) -> E {
        self.current_scalar
    }

    /// The τ value for the variable about to be bound.
    pub fn current_tau(&self) -> E {
        self.tau[self.current_round]
    }

    /// Return the current top-level split-eq tables `(E_first, E_second)`.
    ///
    /// For a pair index `j` in the inner sum, the eq factor for the
    /// remaining (non-current) variables is:
    /// ```text
    /// eq_remaining(j) = E_first[j & (E_first.len()-1)]
    ///                  · E_second[j >> E_first.len().trailing_zeros()]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either `E_first` or `E_second` is empty (invariant violation).
    pub fn remaining_eq_tables(&self) -> (&[E], &[E]) {
        (
            self.E_first.last().expect("E_first is never empty"),
            self.E_second.last().expect("E_second is never empty"),
        )
    }

    /// Bind the current variable to challenge `r`, advancing to the next round.
    ///
    /// Updates `current_scalar` with `eq(τ[current_round], r)` and pops the
    /// appropriate split table level.
    pub fn bind(&mut self, r: E) {
        let tau_k = self.tau[self.current_round];
        self.current_scalar =
            self.current_scalar * (tau_k * r + (E::one() - tau_k) * (E::one() - r));
        self.current_round += 1;
        if self.E_first.len() > 1 {
            self.E_first.pop();
        } else if self.E_second.len() > 1 {
            self.E_second.pop();
        }
    }

    /// Compute the round polynomial `s(X) = l(X) · q(X)` from the inner
    /// polynomial `q` (given as evaluations at integer points `0, 1, ..., d`).
    ///
    /// `l(X) = current_scalar · eq(τ_current, X)` is the linear eq factor
    /// for the current variable. The result has degree `d + 1`.
    pub fn gruen_mul(&self, q_poly: &UniPoly<E>) -> UniPoly<E> {
        let tau_k = self.current_tau();
        let scalar = self.current_scalar();
        let l_0 = scalar * (E::one() - tau_k);
        let l_1 = scalar * tau_k;
        let slope = l_1 - l_0;
        let mut coeffs = vec![E::zero(); q_poly.coeffs.len() + 1];
        for (i, &c) in q_poly.coeffs.iter().enumerate() {
            coeffs[i] = coeffs[i] + c * l_0;
            coeffs[i + 1] = coeffs[i + 1] + c * slope;
        }
        UniPoly::from_coeffs(coeffs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Fp64;
    use crate::protocol::sumcheck::fold_evals;
    use crate::FieldSampling;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Fp64<4294967197>;

    #[test]
    fn gruen_eq_matches_full_eq_table() {
        let mut rng = StdRng::seed_from_u64(0xBB);
        for n in 1..10 {
            let tau: Vec<F> = (0..n).map(|_| F::sample(&mut rng)).collect();
            let mut full_eq = EqPolynomial::evals(&tau);
            let mut split_eq = GruenSplitEq::new(&tau);

            for _round in 0..n {
                let half = full_eq.len() / 2;
                let (e_first, e_second) = split_eq.remaining_eq_tables();
                let num_first = e_first.len();

                for j in 0..half {
                    let j_low = j & (num_first - 1);
                    let j_high = j >> num_first.trailing_zeros();
                    let eq_rem = e_first[j_low] * e_second[j_high];

                    let tau_k = split_eq.current_tau();
                    let scalar = split_eq.current_scalar();
                    let eq_0 = scalar * (F::one() - tau_k) * eq_rem;
                    let eq_1 = scalar * tau_k * eq_rem;

                    assert_eq!(eq_0, full_eq[2 * j], "n={n} round={_round} j={j} eq_0");
                    assert_eq!(eq_1, full_eq[2 * j + 1], "n={n} round={_round} j={j} eq_1");
                }

                let r = F::sample(&mut rng);
                full_eq = fold_evals(&full_eq, r);
                split_eq.bind(r);
            }
        }
    }

    #[test]
    fn gruen_mul_matches_direct_product() {
        let mut rng = StdRng::seed_from_u64(0xCC);
        let tau: Vec<F> = (0..5).map(|_| F::sample(&mut rng)).collect();
        let split_eq = GruenSplitEq::new(&tau);

        let q = UniPoly::from_coeffs(vec![F::from_u64(3), F::from_u64(7), F::from_u64(2)]);
        let s = split_eq.gruen_mul(&q);

        let tau_k = split_eq.current_tau();
        let scalar = split_eq.current_scalar();
        for t in 0..10u64 {
            let x = F::from_u64(t);
            let l_x = scalar * (tau_k * x + (F::one() - tau_k) * (F::one() - x));
            let q_x = q.evaluate(&x);
            assert_eq!(s.evaluate(&x), l_x * q_x, "t={t}");
        }
    }
}
