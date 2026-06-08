//! Clear (non-ZK) sumcheck proof sinks.
//!
//! These functions own the Fiat-Shamir loop for a single sumcheck instance:
//! absorb the input claim, append each round message, sample the round
//! challenge, and fold prover state. Protocol drivers delegate here so stage
//! provers and the descriptor engine share one transcript driver.
//!
//! ZK committed-round sinks are out of the stage-2 pilot scope; they remain on
//! the driver extension traits until FLOW-owned assembly migrates.

use crate::drivers::advance_eq_factored_claim;
use crate::traits::{EqFactoredSumcheckInstanceProver, SumcheckInstanceProver};
use crate::types::{EqFactoredSumcheckProof, SumcheckProof};
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::labels;
use akita_transcript::Transcript;

/// Prove a single regular (compressed univariate) sumcheck instance.
///
/// Returns the proof, the sampled challenge vector, and the final claim at the
/// derived point.
///
/// # Errors
///
/// Returns an error if any round polynomial exceeds the instance degree bound.
#[tracing::instrument(skip_all, name = "prove_clear_regular_sumcheck")]
#[inline(never)]
pub fn prove_clear_regular<E, F, T, P, S>(
    prover: &mut P,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    E: FieldCore + AkitaSerialize,
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    P: SumcheckInstanceProver<E>,
    S: FnMut(&mut T) -> E,
{
    let num_rounds = prover.num_rounds();
    let mut claim = prover.input_claim();
    tracing::debug!(
        is_zero = claim.is_zero(),
        num_rounds,
        "prove_clear_regular input_claim"
    );
    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &claim);

    let degree_bound = prover.degree_bound();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut r = Vec::with_capacity(num_rounds);

    for round in 0..num_rounds {
        let _round_span = tracing::info_span!(
            "sumcheck_round",
            round,
            table_len = 1usize << (num_rounds - round)
        )
        .entered();
        let g = {
            let _s = tracing::info_span!("sumcheck_round_univariate").entered();
            prover.compute_round_univariate(round, claim)
        };
        let round_sum = g.evaluate(&E::zero()) + g.evaluate(&E::one());
        debug_assert!(
            round_sum == claim,
            "sumcheck round {round} univariate does not match previous claim hint"
        );

        let compressed = g.compress();
        if compressed.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "sumcheck round poly degree {} exceeds bound {}",
                compressed.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let r_i = sample_challenge(transcript);
        r.push(r_i);

        claim = compressed.eval_from_hint(&claim, &r_i);
        {
            let _s = tracing::info_span!("sumcheck_round_fold").entered();
            prover.ingest_challenge(round, r_i);
        }
        round_polys.push(compressed);
    }

    prover.finalize();
    Ok((SumcheckProof { round_polys }, r, claim))
}

/// Prove a single eq-factored sumcheck instance.
///
/// Each round sends the inner polynomial `q(X)` with its linear coefficient
/// omitted. The driver maintains the verifier-equivalent scaled-claim update.
///
/// # Errors
///
/// Returns an error if any round polynomial exceeds the instance degree bound.
#[tracing::instrument(skip_all, name = "prove_clear_eq_factored_sumcheck")]
#[inline(never)]
pub fn prove_clear_eq_factored<E, F, T, P, S>(
    prover: &mut P,
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<(EqFactoredSumcheckProof<E>, Vec<E>, E), AkitaError>
where
    E: FieldCore + AkitaSerialize,
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    P: EqFactoredSumcheckInstanceProver<E>,
    S: FnMut(&mut T) -> E,
{
    let num_rounds = prover.num_rounds();
    let degree_bound = prover.degree_bound();
    let mut scaled_claim = prover.input_claim();
    let mut claim_scale = E::one();
    let mut round_polys = Vec::with_capacity(num_rounds);
    let mut challenges = Vec::with_capacity(num_rounds);

    transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &scaled_claim);

    for round in 0..num_rounds {
        let poly = prover.compute_round_eq_factored(round);
        if poly.degree() > degree_bound {
            return Err(AkitaError::InvalidInput(format!(
                "eq-factored sumcheck round poly degree {} exceeds bound {}",
                poly.degree(),
                degree_bound
            )));
        }

        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &poly);
        let r_i = sample_challenge(transcript);
        let (l_at_0, l_at_1) = prover.current_linear_factor_evals();
        (scaled_claim, claim_scale) =
            advance_eq_factored_claim(scaled_claim, claim_scale, l_at_0, l_at_1, &poly, r_i);
        challenges.push(r_i);
        prover.ingest_challenge(round, r_i);
        round_polys.push(poly);
    }

    prover.finalize();
    Ok((
        EqFactoredSumcheckProof { round_polys },
        challenges,
        scaled_claim,
    ))
}
