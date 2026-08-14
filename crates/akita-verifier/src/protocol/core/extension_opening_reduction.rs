//! Verifier-side transcript driver for the non-zk extension-opening reduction.
//!
//! The EOR sumcheck rounds are public-transcript checks. Their final claim is
//! enforced through fused stage-2 `trace_eval_target` and per-claim scales, so
//! this helper returns all derived final claims and the shared `rho` instead of
//! reading standalone on-wire opening handles.

use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::SharedChallengeSumcheckProof;
use akita_transcript::Transcript;
use akita_types::EXTENSION_OPENING_REDUCTION_DEGREE;

/// Verify the non-zk EOR sumcheck rounds and return every final running claim
/// together with the shared sampled sumcheck point.
pub(crate) fn verify_extension_opening_reduction_sumcheck<F, T, E, S>(
    input_claims: &[E],
    num_rounds: usize,
    proof: &SharedChallengeSumcheckProof<E>,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(Vec<E>, Vec<E>), AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    proof.verify::<F, T, _>(
        input_claims,
        num_rounds,
        EXTENSION_OPENING_REDUCTION_DEGREE,
        transcript,
        sample_challenge,
    )
}
