//! Verifier-side transcript driver for the non-zk extension-opening reduction.
//!
//! The EOR sumcheck rounds are public-transcript checks. Their final oracle is
//! the y-ring trace opening that is now enforced by the fused stage-2 trace
//! term, so this helper returns the derived `(final_claim, rho)` instead of
//! attempting to evaluate the final oracle from on-wire y-rings.

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SumcheckProof;
use akita_transcript::labels::ABSORB_SUMCHECK_CLAIM;
use akita_transcript::Transcript;
use akita_types::EXTENSION_OPENING_REDUCTION_DEGREE;

/// Verify the non-zk EOR sumcheck rounds and return the final running claim
/// together with the sampled sumcheck point.
pub(crate) fn verify_extension_opening_reduction_sumcheck<F, T, E, S>(
    input_claim: E,
    num_rounds: usize,
    proof: &SumcheckProof<E>,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(E, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    transcript.append_serde(ABSORB_SUMCHECK_CLAIM, &input_claim);
    proof.verify::<F, T, _>(
        input_claim,
        num_rounds,
        EXTENSION_OPENING_REDUCTION_DEGREE,
        transcript,
        sample_challenge,
    )
}
