//! Sumcheck prover/verifier trait interfaces.
//!
//! The standard `SumcheckInstance{Prover,Verifier}` pair drives the generic
//! sumcheck loop. The `EqFactored*` variants are for sumchecks whose round
//! polynomial factors as `s(X) = l(X) * q(X)`, where `l` is a linear eq
//! factor; the prover sends `q` with its linear term omitted.

use super::types::EqFactoredUniPoly;
use crate::FieldCore;
use akita_algebra::split_eq::GruenSplitEq;
use akita_algebra::uni_poly::UniPoly;
use akita_field::HachiError;

/// Prover-side sumcheck instance interface.
///
/// This trait encapsulates the protocol-specific logic required to compute each
/// per-round univariate polynomial `g_j(X)` and to update (fold) internal state
/// after receiving the verifier challenge `r_j`.
///
/// Hachi §4.3 will implement concrete instances for `H_0` and `H_α`.
pub trait SumcheckInstanceProver<E: FieldCore>: Send + Sync {
    /// Number of rounds (i.e. number of variables bound by sumcheck).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree for any round univariate polynomial.
    fn degree_bound(&self) -> usize;

    /// The initial claimed sum that this sumcheck instance is proving.
    fn input_claim(&self) -> E;

    /// Compute the prover message `g_round(X)` given the previous running claim.
    ///
    /// In standard sumcheck, `previous_claim` is the expected value of the
    /// remaining sum after binding previous challenges, and must satisfy:
    ///
    /// `g_round(0) + g_round(1) == previous_claim`.
    fn compute_round_univariate(&mut self, round: usize, previous_claim: E) -> UniPoly<E>;

    /// Ingest the verifier challenge `r_round` to fold/bind the current variable.
    fn ingest_challenge(&mut self, round: usize, r_round: E);

    /// Optional end-of-protocol hook after the last challenge has been ingested.
    fn finalize(&mut self) {}
}

/// Verifier-side sumcheck instance interface.
///
/// Implementations provide the initial claim and the oracle evaluation at the
/// challenge point, enabling the verifier to perform the final consistency check.
pub trait SumcheckInstanceVerifier<E: FieldCore>: Send + Sync {
    /// Number of rounds (i.e. number of variables bound by sumcheck).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree for any round univariate polynomial.
    fn degree_bound(&self) -> usize;

    /// The initial claimed sum that this sumcheck instance is proving.
    fn input_claim(&self) -> E;

    /// Compute the expected final evaluation `f(r_0, ..., r_{n-1})` at the
    /// challenge point derived during the protocol.
    ///
    /// # Errors
    ///
    /// May return an error if internal evaluations fail (e.g., malformed
    /// evaluation tables from untrusted proof data).
    fn expected_output_claim(&self, challenges: &[E]) -> Result<E, HachiError>;
}

/// Prover-side interface for eq-factored sumchecks of the form `s(X) = l(X) * q(X)`.
///
/// Here `l(X)` is the current linear eq factor for the active round. The
/// prover sends the inner polynomial `q(X)` with its linear term omitted, and
/// the verifier advances a scaled claim directly from `l(0)`, `l(1)`, `q(0)`,
/// the higher-degree contribution of `q(1)`, and the sampled challenge.
pub trait EqFactoredSumcheckInstanceProver<E: FieldCore>: Send + Sync {
    /// Number of rounds (i.e. number of variables bound by sumcheck).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree of the inner polynomial `q(X)` in each round.
    fn degree_bound(&self) -> usize;

    /// The initial unscaled sum claim proved by the instance.
    fn input_claim(&self) -> E;

    /// Linear eq-factor evaluations `(l(0), l(1))` for the current round.
    fn current_linear_factor_evals(&self) -> (E, E);

    /// Compute the eq-factored round message.
    fn compute_round_eq_factored(&mut self, round: usize) -> EqFactoredUniPoly<E>;

    /// Ingest the verifier challenge `r_round` to fold/bind the current variable.
    fn ingest_challenge(&mut self, round: usize, r_round: E);

    /// Optional end-of-protocol hook after the last challenge has been ingested.
    fn finalize(&mut self) {}
}

/// Mutable verifier round state for an eq-factored sumcheck proof.
pub trait EqFactoredSumcheckRoundState<E: FieldCore>: Send {
    /// Linear eq-factor evaluations `(l(0), l(1))` for the current round.
    fn current_linear_factor_evals(&self) -> (E, E);

    /// Ingest the verifier challenge `r_round` to bind the current variable.
    fn ingest_challenge(&mut self, round: usize, r_round: E);
}

impl<E: FieldCore> EqFactoredSumcheckRoundState<E> for GruenSplitEq<E> {
    fn current_linear_factor_evals(&self) -> (E, E) {
        self.linear_factor_evals()
    }

    fn ingest_challenge(&mut self, _round: usize, r_round: E) {
        self.bind(r_round);
    }
}

/// Verifier-side interface for eq-factored sumchecks.
///
/// The verifier itself is immutable. Any per-round mutable state needed to
/// track the evolving eq factor lives in [`Self::RoundState`], which is created
/// fresh for each proof verification.
pub trait EqFactoredSumcheckInstanceVerifier<E: FieldCore>: Send + Sync {
    /// Mutable per-proof round state used by the verifier driver.
    type RoundState: EqFactoredSumcheckRoundState<E>;

    /// Number of rounds (i.e. number of variables bound by sumcheck).
    fn num_rounds(&self) -> usize;

    /// Maximum allowed degree of the inner polynomial `q(X)` in each round.
    fn degree_bound(&self) -> usize;

    /// The initial unscaled sum claim proved by the instance.
    fn input_claim(&self) -> E;

    /// Construct the fresh mutable round state used by the verifier driver.
    fn start_round_state(&self) -> Self::RoundState;

    /// Compute the expected final oracle evaluation `f(r_0, ..., r_{n-1})`.
    ///
    /// # Errors
    ///
    /// May return an error if the verifier cannot evaluate the final folded
    /// instance at the sampled challenge point.
    fn expected_output_claim(
        &self,
        round_state: &Self::RoundState,
        challenges: &[E],
    ) -> Result<E, HachiError>;
}
