//! Sumcheck protocol: traits, proof driver, and concrete instances.
//!
//! Types (`UniPoly`, `CompressedUniPoly`, `SumcheckProof`) live in the
//! [`types`] submodule. Polynomial evaluation utilities (`multilinear_eval`,
//! `fold_evals_in_place`, `range_check_eval`) live in [`crate::algebra::poly`].
//!
//! ## Temporary duplication notice (Jolt integration)
//!
//! Jolt already has a mature, streaming-aware sumcheck implementation. Long-term, we
//! expect to extract the common sumcheck machinery into a dedicated crate and depend
//! on it from both Hachi and Jolt. Until that exists, this module intentionally
//! duplicates the essential sumcheck data types and transcript-driving logic as a
//! pragmatic workaround.

pub mod batched_sumcheck;
pub mod eq_poly;
pub mod hachi_sumcheck;
pub mod norm_sumcheck;
pub mod relation_sumcheck;
pub mod split_eq;
pub mod types;

use crate::error::HachiError;
use crate::protocol::transcript::labels;
use crate::protocol::transcript::Transcript;
use crate::{CanonicalField, FieldCore};

pub use crate::algebra::poly::{fold_evals_in_place, multilinear_eval, range_check_eval};
pub use types::{CompressedUniPoly, SumcheckProof, UniPoly};

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

/// Produce a sumcheck proof for a single instance, driving the Fiat-Shamir transcript.
///
/// This method:
/// - does **not** absorb the initial claim into the transcript (callers should do so),
/// - appends each round message under `labels::ABSORB_SUMCHECK_ROUND`,
/// - samples one challenge per round via `sample_challenge`,
/// - updates the running claim using the per-round hint (`g(0)+g(1)`).
///
/// It returns the proof, the derived point `r`, and the final claimed value at `r`.
///
/// # Errors
///
/// Returns an error if any per-round polynomial exceeds the instance's degree bound.
#[tracing::instrument(skip_all, name = "prove_sumcheck")]
pub fn prove_sumcheck<F, T, E, S, Inst>(
    instance: &mut Inst,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    Inst: SumcheckInstanceProver<E>,
{
    let mut claim = instance.input_claim();
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let num_rounds = instance.num_rounds();
    let degree_bound = instance.degree_bound();

    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let g = instance.compute_round_univariate(round, claim);
        debug_assert!(
            g.evaluate(&E::zero()) + g.evaluate(&E::one()) == claim,
            "sumcheck round univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(HachiError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        claim = compressed.eval_from_hint(&claim, &r_i);

        instance.ingest_challenge(round, r_i);
        round_polys.push(compressed);
    }

    instance.finalize();

    let proof = SumcheckProof { round_polys };
    Ok((proof, r, claim))
}

/// Verify a single-instance sumcheck proof.
///
/// This function:
/// - absorbs the initial claim into the transcript,
/// - delegates round-by-round verification to [`SumcheckProof::verify`],
/// - performs the final oracle check: `final_claim == verifier.expected_output_claim(r)`.
///
/// Returns the challenge point `r` on success.
///
/// # Errors
///
/// Returns [`HachiError::InvalidProof`] if the final sumcheck claim does not
/// match the oracle evaluation, or propagates any error from the per-round
/// verification (e.g. degree-bound violation, round-count mismatch).
#[tracing::instrument(skip_all, name = "verify_sumcheck")]
pub fn verify_sumcheck<F, T, E, S, V>(
    proof: &SumcheckProof<E>,
    verifier: &V,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<Vec<E>, HachiError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore,
    S: FnMut(&mut T) -> E,
    V: SumcheckInstanceVerifier<E>,
{
    let claim = verifier.input_claim();
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);
    let (final_claim, challenges) = proof.verify::<F, T, S>(
        claim,
        verifier.num_rounds(),
        verifier.degree_bound(),
        transcript,
        sample_challenge,
    )?;

    let expected = verifier.expected_output_claim(&challenges)?;
    if final_claim != expected {
        return Err(HachiError::InvalidProof);
    }

    Ok(challenges)
}
