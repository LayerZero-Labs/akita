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
use jolt_poly::UnivariatePoly;

use crate::{Field, Ring};
use akita_error::AkitaError;

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
pub struct GruenSplitEq<E: Field> {
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

/// Recovery for one round, borrowing its split equality state so it cannot advance.
/// Private fields certify an invertible `l(1)`; consuming methods accept arbitrary
/// coefficients and claims, whose consistency remains the caller's responsibility.
pub struct PreparedLinearQRecovery<'a, E: Field> {
    at_zero: E,
    slope: E,
    at_one_inverse: E,
    round: std::marker::PhantomData<&'a GruenSplitEq<E>>,
}

#[allow(non_snake_case)]
impl<E: Field> GruenSplitEq<E> {
    /// Create a new split-eq from the full challenge vector `τ`.
    ///
    /// Precomputes suffix-cached eq tables for two halves of `τ[1..n]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `tau` is empty or if a cached equality table would
    /// exceed the verifier sequence bound.
    pub fn new(tau: &[E]) -> Result<Self, AkitaError> {
        Self::with_initial_scalar(tau, E::one())
    }

    /// Create a new split-eq whose running scalar starts at `initial_scalar`.
    ///
    /// This is useful when a round-independent batching scalar should be folded
    /// into the split-eq factor once up front rather than re-applied to every
    /// round polynomial after `gruen_mul()`.
    ///
    /// # Errors
    ///
    /// Returns an error if `tau` is empty or if a cached equality table would
    /// exceed the verifier sequence bound.
    pub fn with_initial_scalar(tau: &[E], initial_scalar: E) -> Result<Self, AkitaError> {
        let n = tau.len();
        if n == 0 {
            return Err(AkitaError::InvalidSize {
                expected: 1,
                actual: 0,
            });
        }
        let m = (n - 1) / 2;
        let split = 1 + m;
        let first_half = &tau[1..split];
        let second_half = &tau[split..n];
        let E_first = EqPolynomial::evals_cached(first_half)?;
        let E_second = EqPolynomial::evals_cached(second_half)?;
        Ok(Self {
            tau: tau.to_vec(),
            current_round: 0,
            current_scalar: initial_scalar,
            E_first,
            E_second,
        })
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

    /// Return the tables [`remaining_eq_tables`](Self::remaining_eq_tables)
    /// will return after `binds` more calls to [`bind`](Self::bind), or `None`
    /// when fewer than `binds` variables follow the current one.
    ///
    /// The tables do not depend on the challenges, so a prover can weight data
    /// by a later round's equality factor before that round's challenges exist.
    pub fn remaining_eq_tables_after(&self, binds: usize) -> Option<(&[E], &[E])> {
        let first_levels = self.E_first.len() - 1;
        let second_levels = self.E_second.len() - 1;
        if binds <= first_levels {
            return Some((
                self.E_first[first_levels - binds].as_slice(),
                self.E_second[second_levels].as_slice(),
            ));
        }
        let second_level = second_levels.checked_sub(binds - first_levels)?;
        Some((
            self.E_first[0].as_slice(),
            self.E_second[second_level].as_slice(),
        ))
    }

    /// Bind the current variable to challenge `r`, advancing to the next round.
    ///
    /// Multiplies `current_scalar` by `eq(τ[current_round], r)` and pops the
    /// appropriate split table level.
    pub fn bind(&mut self, r: E) {
        let tau_k = self.tau[self.current_round];
        self.current_scalar *= tau_k * r + (E::one() - tau_k) * (E::one() - r);
        self.current_round += 1;
        if self.E_first.len() > 1 {
            self.E_first.pop();
        } else if self.E_second.len() > 1 {
            self.E_second.pop();
        }
    }

    /// Linear eq-factor evaluations `(l(0), l(1))` for the current round.
    #[inline]
    pub fn linear_factor_evals(&self) -> (E, E) {
        let l_at_1 = self.current_scalar * self.current_tau();
        let l_at_0 = self.current_scalar - l_at_1;
        (l_at_0, l_at_1)
    }

    /// Prepare recovery once; returns `None` for `l(1) = 0` or after the last round.
    pub fn prepare_linear_q_recovery(&self) -> Option<PreparedLinearQRecovery<'_, E>> {
        let tau = *self.tau.get(self.current_round)?;
        let at_one = self.current_scalar * tau;
        let at_zero = self.current_scalar - at_one;
        Some(PreparedLinearQRecovery {
            at_zero,
            slope: at_one - at_zero,
            at_one_inverse: at_one.inverse()?,
            round: std::marker::PhantomData,
        })
    }

    /// Returns whether the current Gruen linear factor lets us recover the
    /// omitted linear coefficient of the inner polynomial from `s(0) + s(1)`.
    /// Returns `false` after the last round because there is no current round to recover.
    pub fn can_recover_linear_q_term_from_claim(&self) -> bool {
        self.prepare_linear_q_recovery().is_some()
    }

    /// Compute the round polynomial `s(X) = l(X) · q(X)` from the inner
    /// polynomial `q` (given as evaluations at integer points `0, 1, ..., d`).
    ///
    /// `l(X) = current_scalar · eq(τ_current, X)` is the linear eq factor
    /// for the current variable, including any constructor-supplied leading
    /// scalar. The result has degree `d + 1`.
    pub fn gruen_mul(&self, q_poly: &UnivariatePoly<E>) -> UnivariatePoly<E> {
        let (at_zero, at_one) = self.linear_factor_evals();
        mul_linear(q_poly.coefficients(), at_zero, at_one - at_zero)
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
    ) -> Option<UnivariatePoly<E>> {
        if q_coeffs_except_linear.is_empty() {
            return Some(UnivariatePoly::new(vec![E::zero()]));
        }

        let (l_at_0, l_at_1) = self.linear_factor_evals();
        if l_at_0.is_zero() && l_at_1.is_zero() {
            return Some(UnivariatePoly::new(vec![E::zero()]));
        }

        Some(
            self.prepare_linear_q_recovery()?
                .gruen_poly_from_coeffs_except_linear(q_coeffs_except_linear, s_0_plus_s_1),
        )
    }
}

impl<E: Field + Ring> GruenSplitEq<E> {
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
    ) -> Option<UnivariatePoly<E>> {
        let (l_at_0, l_at_1) = self.linear_factor_evals();
        if l_at_0.is_zero() && l_at_1.is_zero() {
            return Some(UnivariatePoly::new(vec![E::zero()]));
        }

        Some(self.prepare_linear_q_recovery()?.gruen_poly_deg_3(
            q_constant,
            q_quadratic_coeff,
            s_0_plus_s_1,
        ))
    }
}

impl<E: Field> PreparedLinearQRecovery<'_, E> {
    fn linear_term(&self, constant: E, sum_except_linear: E, claim: E) -> E {
        (claim - self.at_zero * constant) * self.at_one_inverse - sum_except_linear
    }

    /// Apply recovery to `[q_0, q_2, ..., q_d]`; an empty slice gives the zero polynomial.
    fn gruen_poly_from_coeffs_except_linear(
        self,
        coefficients: &[E],
        claim: E,
    ) -> UnivariatePoly<E> {
        let Some((&constant, rest)) = coefficients.split_first() else {
            return UnivariatePoly::new(vec![E::zero()]);
        };
        let sum = coefficients
            .iter()
            .copied()
            .fold(E::zero(), |sum, c| sum + c);
        let mut full = Vec::with_capacity(coefficients.len() + 1);
        full.push(constant);
        full.push(self.linear_term(constant, sum, claim));
        full.extend_from_slice(rest);
        mul_linear(&full, self.at_zero, self.slope)
    }

    /// Apply recovery to a quadratic inner polynomial without allocating its coefficients.
    pub fn gruen_poly_deg_3(self, constant: E, quadratic: E, claim: E) -> UnivariatePoly<E> {
        let linear = self.linear_term(constant, constant + quadratic, claim);
        let mut polynomial = mul_linear(&[constant, linear, quadratic], self.at_zero, self.slope);
        polynomial.trim_trailing_zeros();
        polynomial
    }
}

fn mul_linear<E: Field>(q_coeffs: &[E], at_zero: E, slope: E) -> UnivariatePoly<E> {
    let mut coeffs = vec![E::zero(); q_coeffs.len() + 1];
    for (i, &c) in q_coeffs.iter().enumerate() {
        coeffs[i] += c * at_zero;
        coeffs[i + 1] += c * slope;
    }
    UnivariatePoly::new(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::fold_evals_in_place;
    use crate::Field;
    use jolt_field::{One, Prime128Offset275, Zero};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    type F = Prime128Offset275;

    #[test]
    fn prepared_recovery_matches_existing_paths_at_every_round() {
        let mut saw_zero_at_one = false;
        let mut saw_zero_factor = false;
        for tau in [
            vec![F::from_u64(3); 5],
            vec![F::zero(), F::one(), F::from_u64(7), F::one(), F::zero()],
        ] {
            for scalar in [F::one(), F::zero()] {
                let mut split = GruenSplitEq::with_initial_scalar(&tau, scalar).unwrap();
                for round in 0..tau.len() {
                    let q =
                        UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(9), F::from_u64(4)]);
                    let mut expected = split.gruen_mul(&q);
                    let claim = expected.evaluate(F::zero()) + expected.evaluate(F::one());
                    expected.trim_trailing_zeros();
                    let prepared = split.prepare_linear_q_recovery();
                    assert_eq!(
                        prepared.is_some(),
                        split.can_recover_linear_q_term_from_claim()
                    );
                    let checked = split.try_gruen_poly_deg_3(F::from_u64(5), F::from_u64(4), claim);
                    if let Some(prepared) = prepared {
                        let actual =
                            prepared.gruen_poly_deg_3(F::from_u64(5), F::from_u64(4), claim);
                        assert_eq!(Some(actual.clone()), checked);
                        assert_eq!(actual, expected);
                        // General recovery retains its existing coefficient representation,
                        // including untrimmed zero polynomials and arbitrary claims.
                        for coefficients in [
                            vec![],
                            vec![F::zero()],
                            vec![F::from_u64(5)],
                            vec![F::from_u64(5), F::from_u64(4), F::from_u64(6)],
                        ] {
                            for arbitrary_claim in [F::zero(), claim, F::from_u64(17)] {
                                let cubic =
                                    split.prepare_linear_q_recovery().unwrap().gruen_poly_deg_3(
                                        F::from_u64(5),
                                        F::from_u64(4),
                                        arbitrary_claim,
                                    );
                                assert_eq!(
                                    cubic.evaluate(F::zero()) + cubic.evaluate(F::one()),
                                    arbitrary_claim
                                );
                                assert_eq!(
                                    Some(cubic),
                                    split.try_gruen_poly_deg_3(
                                        F::from_u64(5),
                                        F::from_u64(4),
                                        arbitrary_claim,
                                    )
                                );
                                let actual = split
                                    .prepare_linear_q_recovery()
                                    .unwrap()
                                    .gruen_poly_from_coeffs_except_linear(
                                        &coefficients,
                                        arbitrary_claim,
                                    );
                                assert_eq!(
                                    Some(actual),
                                    split.try_gruen_poly_from_coeffs_except_linear(
                                        &coefficients,
                                        arbitrary_claim,
                                    )
                                );
                            }
                        }
                    } else {
                        let (at_zero, at_one) = split.linear_factor_evals();
                        assert!(at_one.is_zero());
                        saw_zero_at_one = true;
                        if at_zero.is_zero() {
                            saw_zero_factor = true;
                            assert_eq!(checked, Some(UnivariatePoly::new(vec![F::zero()])));
                            assert_eq!(
                                split.try_gruen_poly_from_coeffs_except_linear(
                                    &[F::from_u64(5), F::from_u64(4)],
                                    claim,
                                ),
                                checked
                            );
                            assert_eq!(expected, checked.unwrap());
                        } else {
                            assert!(checked.is_none());
                            assert!(split
                                .try_gruen_poly_from_coeffs_except_linear(
                                    &[F::from_u64(5), F::from_u64(4)],
                                    claim,
                                )
                                .is_none());
                        }
                    }
                    split.bind(F::from_u64(round as u64 + 11));
                }
                assert!(split.prepare_linear_q_recovery().is_none());
                assert!(!split.can_recover_linear_q_term_from_claim());
            }
        }
        assert!(saw_zero_at_one);
        assert!(saw_zero_factor);
    }

    #[test]
    fn gruen_eq_matches_full_eq_table() {
        let mut rng = StdRng::seed_from_u64(0xBB);
        for n in 1..10 {
            let tau: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let mut full_eq = EqPolynomial::evals(&tau).unwrap();
            let mut split_eq = GruenSplitEq::new(&tau).unwrap();

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

                let r = F::random(&mut rng);
                fold_evals_in_place(&mut full_eq, r);
                split_eq.bind(r);
            }
        }
    }

    #[test]
    fn remaining_eq_tables_after_matches_bound_tables() {
        let mut rng = StdRng::seed_from_u64(0xBC);
        for n in 1..10 {
            let tau: Vec<F> = (0..n).map(|_| F::random(&mut rng)).collect();
            let fresh = GruenSplitEq::new(&tau).unwrap();
            let mut bound = GruenSplitEq::new(&tau).unwrap();
            for binds in 0..n {
                assert_eq!(
                    fresh.remaining_eq_tables_after(binds),
                    Some(bound.remaining_eq_tables()),
                    "n={n} binds={binds}"
                );
                bound.bind(F::random(&mut rng));
            }
            assert_eq!(fresh.remaining_eq_tables_after(n), None, "n={n}");
        }
    }

    #[test]
    fn gruen_mul_matches_direct_product() {
        let mut rng = StdRng::seed_from_u64(0xCC);
        let tau: Vec<F> = (0..5).map(|_| F::random(&mut rng)).collect();
        let split_eq = GruenSplitEq::new(&tau).unwrap();

        let q = UnivariatePoly::new(vec![F::from_u64(3), F::from_u64(7), F::from_u64(2)]);
        let s = split_eq.gruen_mul(&q);

        let tau_k = split_eq.current_tau();
        let scalar = split_eq.current_scalar();
        for t in 0..10u64 {
            let x = F::from_u64(t);
            let l_x = scalar * (tau_k * x + (F::one() - tau_k) * (F::one() - x));
            let q_x = q.evaluate(x);
            assert_eq!(s.evaluate(x), l_x * q_x, "t={t}");
        }
    }

    #[test]
    fn recover_round_poly_from_coeffs_except_linear() {
        let mut rng = StdRng::seed_from_u64(0xCD);
        let mut tau: Vec<F> = (0..5).map(|_| F::random(&mut rng)).collect();
        if tau[0].is_zero() {
            tau[0] = F::one();
        }
        let split_eq = GruenSplitEq::new(&tau).unwrap();

        let q = UnivariatePoly::new(vec![
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(11),
            F::from_u64(2),
        ]);
        let s = split_eq.gruen_mul(&q);
        let q_except_linear = vec![
            q.coefficients()[0],
            q.coefficients()[2],
            q.coefficients()[3],
        ];
        let previous_claim = s.evaluate(F::zero()) + s.evaluate(F::one());

        let recovered = split_eq
            .try_gruen_poly_from_coeffs_except_linear(&q_except_linear, previous_claim)
            .expect("tau_0 is nonzero, so q(1) is recoverable");

        assert_eq!(recovered, s);
    }

    #[test]
    fn recover_quadratic_round_poly_from_claim() {
        let mut rng = StdRng::seed_from_u64(0xCE);
        let mut tau: Vec<F> = (0..4).map(|_| F::random(&mut rng)).collect();
        if tau[0].is_zero() {
            tau[0] = F::one();
        }
        let split_eq = GruenSplitEq::new(&tau).unwrap();

        let q = UnivariatePoly::new(vec![F::from_u64(5), F::from_u64(9), F::from_u64(4)]);
        let s = split_eq.gruen_mul(&q);
        let previous_claim = s.evaluate(F::zero()) + s.evaluate(F::one());

        let recovered = split_eq
            .try_gruen_poly_deg_3(q.coefficients()[0], q.coefficients()[2], previous_claim)
            .expect("tau_0 is nonzero, so q(1) is recoverable");

        assert_eq!(recovered, s);
    }
}
