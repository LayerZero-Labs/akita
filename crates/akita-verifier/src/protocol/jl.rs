//! Verifier replay for schedule-inert iterated JL projection reductions.

use akita_algebra::jl::{
    eval_block_tensor_mle, eval_power_of_two_block_projection_reduction_factor,
};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::{
    absorb_ext_field, absorb_jl_clear_image, absorb_jl_retry_and_sample_seed, labels,
    sample_ext_challenge, sample_jl_image_point, Transcript,
};
use akita_types::{
    JlProjectionChainPlan, JlProjectionProof, JlSourceClaim, JL_PROJECTION_REDUCTION_DEGREE,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Verify a clear image and every block projection reduction in reverse order.
///
/// The caller must already have absorbed the outgoing recursive witness. The
/// resulting source claim is intentionally deferred: production Stage 2 must
/// still bind it to aggregate semantic `Z` or literal `Ehat`/`That`.
pub fn verify_jl_projection_chain<F, E, T>(
    transcript: &mut T,
    plan: &JlProjectionChainPlan,
    proof: &JlProjectionProof<E>,
) -> Result<JlSourceClaim<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    if proof.clear_image.len() != plan.final_image_len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.final_image_len(),
            actual: proof.clear_image.len(),
        });
    }
    if proof.reverse_layers.len() != plan.layers().len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.layers().len(),
            actual: proof.reverse_layers.len(),
        });
    }
    let retry = plan.selected_retry(proof.retry_index)?;
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, proof.retry_index);
    let matrices = plan.derive_matrices(&master_seed, retry)?;
    let energy = akita_algebra::jl::squared_l2_i128(&proof.clear_image)?;
    if energy > plan.final_energy_bound() {
        return Err(AkitaError::InvalidProof);
    }

    absorb_jl_clear_image::<F, T>(transcript, &proof.clear_image);
    let final_layer = plan
        .layers()
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let mut output_point =
        sample_jl_image_point::<F, E, T>(transcript, final_layer.output_num_vars()?);
    let clear_field = proof
        .clear_image
        .iter()
        .copied()
        .map(E::from_i128)
        .collect::<Vec<_>>();
    let mut output_claim = eval_block_tensor_mle(
        &clear_field,
        final_layer.blocks(),
        final_layer.matrix_context(retry).shape()?.rows(),
        &output_point,
    )?;

    for ((layer_proof, layer), matrix) in proof
        .reverse_layers
        .iter()
        .zip(plan.layers().iter().rev())
        .zip(matrices.iter().rev())
    {
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, &output_claim);
        let (final_claim, challenges) = layer_proof.sumcheck.verify::<F, T, _>(
            output_claim,
            layer.reduction_num_vars()?,
            JL_PROJECTION_REDUCTION_DEGREE,
            transcript,
            |tr| {
                Ok(sample_ext_challenge::<F, E, T>(
                    tr,
                    labels::CHALLENGE_SUMCHECK_ROUND,
                ))
            },
        )?;
        absorb_ext_field::<F, E, T>(
            transcript,
            labels::ABSORB_JL_REDUCTION_TERMINAL,
            &layer_proof.input_evaluation,
        );
        let factor = eval_power_of_two_block_projection_reduction_factor(
            matrix,
            layer.blocks(),
            &output_point,
            &challenges,
        )?;
        if final_claim != layer_proof.input_evaluation * factor {
            return Err(AkitaError::InvalidProof);
        }
        output_point = challenges;
        output_claim = layer_proof.input_evaluation;
    }

    Ok(JlSourceClaim {
        point: output_point,
        evaluation: output_claim,
    })
}
