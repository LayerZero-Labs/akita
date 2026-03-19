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
use crate::{FieldCore, FromSmallInt};

/// Split equality polynomial with Gruen scalar accumulation.
///
/// Instead of storing and folding a full eq table each round, this struct
/// maintains:
/// - `current_scalar`: accumulated leading scalar times `eq(τ_bound, r_bound)`
///   from already-bound variables
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
impl<E: FieldCore> GruenSplitEq<E> {
    /// Create a new split-eq from the full challenge vector `τ`.
    ///
    /// Precomputes suffix-cached eq tables for two halves of `τ[1..n]`.
    ///
    /// # Panics
    ///
    /// Panics if `tau` is empty.
    pub fn new(tau: &[E]) -> Self {
        Self::with_initial_scalar(tau, E::one())
    }

    /// Create a new split-eq whose running scalar starts at `initial_scalar`.
    ///
    /// This is useful when a round-independent batching scalar should be folded
    /// into the split-eq factor once up front rather than re-applied to every
    /// round polynomial after `gruen_mul()`.
    ///
    /// # Panics
    ///
    /// Panics if `tau` is empty.
    pub fn with_initial_scalar(tau: &[E], initial_scalar: E) -> Self {
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
            current_scalar: initial_scalar,
            E_first,
            E_second,
        }
    }

    /// The accumulated scalar `c * Π_{k < current_round} eq(τ[k], r[k])`,
    /// where `c` is the constructor-supplied leading scalar.
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
    /// Multiplies `current_scalar` by `eq(τ[current_round], r)` and pops the
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

    #[inline]
    fn linear_factor_evals(&self) -> (E, E) {
        let l_at_1 = self.current_scalar * self.current_tau();
        let l_at_0 = self.current_scalar - l_at_1;
        (l_at_0, l_at_1)
    }

    /// Returns whether the current Gruen linear factor lets us recover the
    /// omitted linear coefficient of the inner polynomial from `s(0) + s(1)`.
    pub fn can_recover_linear_q_term_from_claim(&self) -> bool {
        let (_, l_at_1) = self.linear_factor_evals();
        l_at_1.inv().is_some()
    }

    /// Compute the round polynomial `s(X) = l(X) · q(X)` from the inner
    /// polynomial `q` (given as evaluations at integer points `0, 1, ..., d`).
    ///
    /// `l(X) = current_scalar · eq(τ_current, X)` is the linear eq factor
    /// for the current variable, including any constructor-supplied leading
    /// scalar. The result has degree `d + 1`.
    pub fn gruen_mul(&self, q_poly: &UniPoly<E>) -> UniPoly<E> {
        let (l_at_0, l_at_1) = self.linear_factor_evals();
        let slope = l_at_1 - l_at_0;
        let mut coeffs = vec![E::zero(); q_poly.coeffs.len() + 1];
        for (i, &c) in q_poly.coeffs.iter().enumerate() {
            coeffs[i] += c * l_at_0;
            coeffs[i + 1] += c * slope;
        }
        UniPoly::from_coeffs(coeffs)
    }

    /// Recover a missing linear coefficient of `q(X)` from `s(0) + s(1)` and
    /// return the full round polynomial `s(X) = l(X) · q(X)`.
    ///
    /// The input is `[q_0, q_2, q_3, ..., q_d]`, i.e. all coefficients except
    /// the linear term. Returns `None` when `l(1) = 0`, in which case that
    /// missing coefficient is not recoverable from the claim alone.
    pub fn try_gruen_poly_from_coeffs_except_linear(
        &self,
        q_coeffs_except_linear: &[E],
        s_0_plus_s_1: E,
    ) -> Option<UniPoly<E>> {
        if q_coeffs_except_linear.is_empty() {
            return Some(UniPoly::from_coeffs(vec![E::zero()]));
        }

        let (l_at_0, l_at_1) = self.linear_factor_evals();
        if l_at_0.is_zero() && l_at_1.is_zero() {
            return Some(UniPoly::from_coeffs(vec![E::zero()]));
        }

        let l_at_1_inv = l_at_1.inv()?;
        let q_at_0 = q_coeffs_except_linear[0];
        let q_at_1 = (s_0_plus_s_1 - l_at_0 * q_at_0) * l_at_1_inv;
        let sum_except_linear = q_coeffs_except_linear
            .iter()
            .copied()
            .fold(E::zero(), |acc, coeff| acc + coeff);
        let q_linear = q_at_1 - sum_except_linear;

        let mut q_coeffs = Vec::with_capacity(q_coeffs_except_linear.len() + 1);
        q_coeffs.push(q_at_0);
        q_coeffs.push(q_linear);
        q_coeffs.extend_from_slice(&q_coeffs_except_linear[1..]);
        Some(self.gruen_mul(&UniPoly::from_coeffs(q_coeffs)))
    }
}

impl<E: FieldCore + FromSmallInt> GruenSplitEq<E> {
    /// Recover the middle coefficient of a quadratic inner polynomial
    /// `q(X) = c + dX + eX^2` from `s(0) + s(1)` and return
    /// `s(X) = l(X) · q(X)`.
    ///
    /// Returns `None` when `l(1) = 0`, in which case `q(1)` is not recoverable
    /// from the claim alone.
    pub fn try_gruen_poly_deg_3(
        &self,
        q_constant: E,
        q_quadratic_coeff: E,
        s_0_plus_s_1: E,
    ) -> Option<UniPoly<E>> {
        let (l_at_0, l_at_1) = self.linear_factor_evals();
        if l_at_0.is_zero() && l_at_1.is_zero() {
            return Some(UniPoly::from_coeffs(vec![E::zero()]));
        }

        let l_at_1_inv = l_at_1.inv()?;
        let slope = l_at_1 - l_at_0;
        let l_at_2 = l_at_1 + slope;
        let l_at_3 = l_at_2 + slope;

        let q_at_0 = q_constant;
        let s_at_0 = l_at_0 * q_at_0;
        let s_at_1 = s_0_plus_s_1 - s_at_0;
        let q_at_1 = s_at_1 * l_at_1_inv;

        let twice_q_quadratic = q_quadratic_coeff + q_quadratic_coeff;
        let q_at_2 = q_at_1 + q_at_1 - q_at_0 + twice_q_quadratic;
        let q_at_3 = q_at_2 + q_at_1 - q_at_0 + twice_q_quadratic + twice_q_quadratic;

        Some(UniPoly::from_evals(&[
            s_at_0,
            s_at_1,
            l_at_2 * q_at_2,
            l_at_3 * q_at_3,
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128M8M4M1M0;
    use crate::protocol::sumcheck::fold_evals_in_place;
    use crate::{FieldSampling, FromSmallInt};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Prime128M8M4M1M0;

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
                fold_evals_in_place(&mut full_eq, r);
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

    #[test]
    fn recover_round_poly_from_coeffs_except_linear() {
        let mut rng = StdRng::seed_from_u64(0xCD);
        let mut tau: Vec<F> = (0..5).map(|_| F::sample(&mut rng)).collect();
        if tau[0].is_zero() {
            tau[0] = F::one();
        }
        let split_eq = GruenSplitEq::new(&tau);

        let q = UniPoly::from_coeffs(vec![
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(2),
        ]);
        let s = split_eq.gruen_mul(&q);
        let q_except_linear = vec![q.coeffs[0], q.coeffs[2], q.coeffs[3]];
        let previous_claim = s.evaluate(&F::zero()) + s.evaluate(&F::one());

        let recovered = split_eq
            .try_gruen_poly_from_coeffs_except_linear(&q_except_linear, previous_claim)
            .expect("tau_0 is nonzero, so q(1) is recoverable");

        assert_eq!(recovered, s);
    }

    #[test]
    fn recover_quadratic_round_poly_from_claim() {
        let mut rng = StdRng::seed_from_u64(0xCE);
        let mut tau: Vec<F> = (0..4).map(|_| F::sample(&mut rng)).collect();
        if tau[0].is_zero() {
            tau[0] = F::one();
        }
        let split_eq = GruenSplitEq::new(&tau);

        let q = UniPoly::from_coeffs(vec![F::from_u64(5), F::from_u64(9), F::from_u64(4)]);
        let s = split_eq.gruen_mul(&q);
        let previous_claim = s.evaluate(&F::zero()) + s.evaluate(&F::one());

        let recovered = split_eq
            .try_gruen_poly_deg_3(q.coeffs[0], q.coeffs[2], previous_claim)
            .expect("tau_0 is nonzero, so q(1) is recoverable");

        assert_eq!(recovered, s);
    }
}
