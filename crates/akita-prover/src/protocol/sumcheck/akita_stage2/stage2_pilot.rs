//! Stage-2 clear sumcheck pilot: prove through the shared clear sink.
//!
//! Production call sites in `flow/` still invoke [`AkitaStage2Prover`] via
//! [`SumcheckInstanceProverExt::prove`], which already delegates to
//! [`prove_clear_regular`]. This module is the explicit stage-2 entry point
//! for the descriptor-engine pilot and will replace direct driver calls when
//! FLOW wiring migrates.

use super::AkitaStage2Prover;
use akita_field::unreduced::{HasOptimizedFold, HasUnreducedOps};
use akita_field::{AkitaError, CanonicalField, FieldCore, FromPrimitiveInt};
use akita_protocol::{matches_stage2_intermediate_descriptor, stage2_descriptor, LevelRole};
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{
    prove_clear_regular, InstanceProverAdapter, ResolvedSumcheckProver, SumcheckProof,
};
use akita_transcript::Transcript;

/// Prove stage 2 through the clear proof sink on the optimized prover.
#[allow(dead_code)]
///
/// Byte-identical to [`SumcheckInstanceProverExt::prove`] for the same prover
/// state and transcript sampling.
///
/// # Errors
///
/// Returns an error if any round polynomial exceeds the instance degree bound.
pub(crate) fn prove_clear<E, F, T, S>(
    prover: &mut AkitaStage2Prover<E>,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold + AkitaSerialize,
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    S: FnMut(&mut T) -> E,
{
    prove_clear_regular(prover, transcript, sample_challenge)
}

/// Prove stage 2 through [`ResolvedSumcheckProver`] with the optimized kernel.
#[allow(dead_code)]
///
/// This exercises the registry selection path used by the descriptor-engine
/// pilot: when the intermediate stage-2 descriptor matches, the hand-tuned
/// [`AkitaStage2Prover`] is wrapped in [`InstanceProverAdapter`] and driven by
/// the same clear sink.
///
/// # Errors
///
/// Returns an error when `num_rounds` does not describe an intermediate
/// stage-2 instance, or when proving fails.
pub(crate) fn prove_clear_via_registry<E, F, T, S>(
    prover: AkitaStage2Prover<E>,
    num_rounds: usize,
    transcript: &mut T,
    sample_challenge: S,
) -> Result<(SumcheckProof<E>, Vec<E>, E), AkitaError>
where
    E: FieldCore + FromPrimitiveInt + HasUnreducedOps + HasOptimizedFold + AkitaSerialize + 'static,
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    S: FnMut(&mut T) -> E,
{
    let descriptor = stage2_descriptor(num_rounds, LevelRole::Intermediate);
    if !matches_stage2_intermediate_descriptor(&descriptor) {
        return Err(AkitaError::InvalidInput(
            "stage-2 registry pilot requires the intermediate fused descriptor".to_string(),
        ));
    }

    let mut resolved =
        ResolvedSumcheckProver::Optimized(Box::new(InstanceProverAdapter::new(prover)));
    prove_clear_regular(&mut resolved, transcript, sample_challenge)
}
