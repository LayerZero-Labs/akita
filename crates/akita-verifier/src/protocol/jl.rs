//! Verifier replay for schedule-inert iterated JL projection reductions.

use akita_algebra::jl::{
    eval_block_tensor_mle, eval_power_of_two_block_projection_reduction_factor,
    validate_centered_i128, TernaryProjectionMatrix,
};
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_transcript::{
    absorb_ext_field, absorb_jl_clear_image, absorb_jl_retry_and_sample_seed, labels,
    sample_ext_challenge, sample_jl_image_point, Transcript,
};
use akita_types::{
    JlProjectionBatchPlan, JlProjectionBatchProof, JlProjectionChainPlan, JlProjectionProof,
    JlSourceClaim, JL_PROJECTION_REDUCTION_DEGREE,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

struct PreparedChain<'a, E: Field> {
    plan: &'a JlProjectionChainPlan,
    proof: &'a JlProjectionProof<E>,
    matrices: Vec<TernaryProjectionMatrix>,
}

/// Opaque verifier batch whose retries/seeds and matrices are fixed.
pub struct PreparedJlProjectionVerification<'a, E: Field> {
    chains: Vec<PreparedChain<'a, E>>,
}

/// Opaque verifier batch whose complete checked image array is absorbed.
pub struct JlVerifierReductionBatch<'a, E: Field> {
    chains: Vec<PreparedChain<'a, E>>,
}

/// Fix the whole-forest candidate and derive every shared envelope.
///
/// The outgoing witness must already be bound. No image is absorbed and no
/// reduction challenge is sampled in this phase.
pub fn prepare_jl_projection_verification<'a, F, E, T>(
    transcript: &mut T,
    plan: &'a JlProjectionBatchPlan,
    proof: &'a JlProjectionBatchProof<E>,
) -> Result<PreparedJlProjectionVerification<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    if proof.chains.len() != plan.chains().len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.chains().len(),
            actual: proof.chains.len(),
        });
    }
    let retry = plan
        .selected_retry(proof.retry_index)
        .map_err(|_| AkitaError::InvalidProof)?;
    for (chain_plan, chain_proof) in plan.chains().iter().zip(&proof.chains) {
        if chain_proof.clear_image.len() != chain_plan.final_image_len() {
            return Err(AkitaError::InvalidSize {
                expected: chain_plan.final_image_len(),
                actual: chain_proof.clear_image.len(),
            });
        }
        if chain_proof.reverse_layers.len() != chain_plan.layers().len() {
            return Err(AkitaError::InvalidSize {
                expected: chain_plan.layers().len(),
                actual: chain_proof.reverse_layers.len(),
            });
        }
    }
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, proof.retry_index);
    let envelopes = plan.derive_envelopes(&master_seed, retry)?;
    let mut chains = Vec::new();
    chains
        .try_reserve_exact(plan.chains().len())
        .map_err(|_| AkitaError::InvalidInput("JL verifier-batch allocation failed".into()))?;
    for (chain_index, (chain_plan, chain_proof)) in
        plan.chains().iter().zip(&proof.chains).enumerate()
    {
        let matrices = plan.member_matrices(chain_index, &envelopes)?;
        chains.push(PreparedChain {
            plan: chain_plan,
            proof: chain_proof,
            matrices,
        });
    }
    Ok(PreparedJlProjectionVerification { chains })
}

/// Check and absorb every clear image before any reduction challenge.
pub fn absorb_jl_projection_verification_images<'a, F, E, T>(
    transcript: &mut T,
    prepared: PreparedJlProjectionVerification<'a, E>,
) -> Result<JlVerifierReductionBatch<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    for chain in &prepared.chains {
        for &coordinate in &chain.proof.clear_image {
            validate_centered_i128::<F>(coordinate).map_err(|_| AkitaError::InvalidProof)?;
        }
        let energy = akita_algebra::jl::squared_l2_i128(&chain.proof.clear_image)?;
        if energy > chain.plan.final_energy_bound() {
            return Err(AkitaError::InvalidProof);
        }
    }
    for chain in &prepared.chains {
        absorb_jl_clear_image::<F, T>(transcript, &chain.proof.clear_image)
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    Ok(JlVerifierReductionBatch {
        chains: prepared.chains,
    })
}

/// Replay every reverse chain after the full image array has been absorbed.
///
/// Each returned claim evaluates exactly the corresponding plan-chain source.
/// In particular, an `EtTail` claim addresses the already-joined table; the
/// surrounding protocol must still link that table to authenticated E/T data.
pub fn verify_jl_projection_reduction_batch<F, E, T>(
    transcript: &mut T,
    ready: JlVerifierReductionBatch<'_, E>,
) -> Result<Vec<JlSourceClaim<E>>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let mut claims = Vec::new();
    claims
        .try_reserve_exact(ready.chains.len())
        .map_err(|_| AkitaError::InvalidInput("JL source-claim allocation failed".into()))?;
    for chain in ready.chains {
        claims.push(verify_prepared_chain::<F, E, T>(transcript, chain)?);
    }
    Ok(claims)
}

fn verify_prepared_chain<F, E, T>(
    transcript: &mut T,
    chain: PreparedChain<'_, E>,
) -> Result<JlSourceClaim<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let final_layer = chain
        .plan
        .layers()
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let mut output_point =
        sample_jl_image_point::<F, E, T>(transcript, final_layer.output_num_vars()?);
    let clear_field = try_embed_i128::<E>(&chain.proof.clear_image)?;
    let mut output_claim = eval_block_tensor_mle(
        &clear_field,
        final_layer.blocks(),
        final_layer.matrix_member().shape()?.rows(),
        &output_point,
    )?;

    for ((layer_proof, layer), matrix) in chain
        .proof
        .reverse_layers
        .iter()
        .zip(chain.plan.layers().iter().rev())
        .zip(chain.matrices.iter().rev())
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

fn try_embed_i128<E: Field>(values: &[i128]) -> Result<Vec<E>, AkitaError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| AkitaError::InvalidInput("JL field-vector allocation failed".into()))?;
    output.extend(values.iter().copied().map(E::from_i128));
    Ok(output)
}
