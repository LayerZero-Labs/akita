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
    JlAlignedEtProjectionPlan, JlAlignedEtProjectionProof, JlProjectionBatchPlan,
    JlProjectionBatchProof, JlProjectionChainPlan, JlProjectionProof, JlSourceClaim,
    JL_PROJECTION_REDUCTION_DEGREE,
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

struct PreparedMatrixChain<'a> {
    plan: &'a JlProjectionChainPlan,
    matrices: Vec<TernaryProjectionMatrix>,
}

/// Opaque aligned verifier graph whose candidate and matrices are fixed.
pub struct PreparedJlAlignedEtVerification<'a, E: Field> {
    proof: &'a JlAlignedEtProjectionProof<E>,
    z: PreparedMatrixChain<'a>,
    e: PreparedMatrixChain<'a>,
    t: PreparedMatrixChain<'a>,
    et_tail: PreparedMatrixChain<'a>,
}

/// Opaque aligned verifier graph whose two public images are absorbed.
pub struct JlAlignedEtVerifierReduction<'a, E: Field> {
    proof: &'a JlAlignedEtProjectionProof<E>,
    z: PreparedMatrixChain<'a>,
    e: PreparedMatrixChain<'a>,
    t: PreparedMatrixChain<'a>,
    et_tail: PreparedMatrixChain<'a>,
}

/// Deferred source claims that the surrounding protocol must authenticate.
pub struct JlAlignedEtSourceClaims<E: Field> {
    /// Aggregate semantic-Z source evaluation.
    pub z: JlSourceClaim<E>,
    /// Literal-E source evaluation.
    pub e: JlSourceClaim<E>,
    /// Literal-T source evaluation.
    pub t: JlSourceClaim<E>,
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
    plan.validate_verifier_workspace(std::mem::size_of::<E>())?;
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

/// Fix the whole-forest candidate and derive the aligned graph envelopes.
pub fn prepare_jl_aligned_et_verification<'a, F, E, T>(
    transcript: &mut T,
    plan: &'a JlAlignedEtProjectionPlan,
    proof: &'a JlAlignedEtProjectionProof<E>,
) -> Result<PreparedJlAlignedEtVerification<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    plan.batch()
        .validate_verifier_workspace(std::mem::size_of::<E>())?;
    validate_chain_proof(plan.z()?, &proof.z)?;
    validate_chain_proof(plan.et_tail()?, &proof.et_tail)?;
    if proof.e_stem_reverse_layers.len() != plan.e_stem()?.layers().len()
        || proof.t_stem_reverse_layers.len() != plan.t_stem()?.layers().len()
    {
        return Err(AkitaError::InvalidProof);
    }
    let batch = plan.batch();
    let retry = batch
        .selected_retry(proof.retry_index)
        .map_err(|_| AkitaError::InvalidProof)?;
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, proof.retry_index);
    let envelopes = batch.derive_envelopes(&master_seed, retry)?;
    Ok(PreparedJlAlignedEtVerification {
        proof,
        z: prepared_matrix_chain(batch, 0, plan.z()?, &envelopes)?,
        e: prepared_matrix_chain(batch, 1, plan.e_stem()?, &envelopes)?,
        t: prepared_matrix_chain(batch, 2, plan.t_stem()?, &envelopes)?,
        et_tail: prepared_matrix_chain(batch, 3, plan.et_tail()?, &envelopes)?,
    })
}

fn validate_chain_proof<E: Field>(
    plan: &JlProjectionChainPlan,
    proof: &JlProjectionProof<E>,
) -> Result<(), AkitaError> {
    if proof.clear_image.len() != plan.final_image_len()
        || proof.reverse_layers.len() != plan.layers().len()
    {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

fn prepared_matrix_chain<'a>(
    batch: &JlProjectionBatchPlan,
    index: usize,
    plan: &'a JlProjectionChainPlan,
    envelopes: &[TernaryProjectionMatrix],
) -> Result<PreparedMatrixChain<'a>, AkitaError> {
    Ok(PreparedMatrixChain {
        plan,
        matrices: batch.member_matrices(index, envelopes)?,
    })
}

/// Check and absorb both logical-certificate images before graph reduction.
pub fn absorb_jl_aligned_et_verification_images<'a, F, E, T>(
    transcript: &mut T,
    prepared: PreparedJlAlignedEtVerification<'a, E>,
) -> Result<JlAlignedEtVerifierReduction<'a, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    for (chain, image) in [
        (&prepared.z, prepared.proof.z.clear_image.as_slice()),
        (
            &prepared.et_tail,
            prepared.proof.et_tail.clear_image.as_slice(),
        ),
    ] {
        validate_clear_image::<F>(chain.plan, image)?;
    }
    for image in [
        prepared.proof.z.clear_image.as_slice(),
        prepared.proof.et_tail.clear_image.as_slice(),
    ] {
        absorb_jl_clear_image::<F, T>(transcript, image).map_err(|_| AkitaError::InvalidProof)?;
    }
    Ok(JlAlignedEtVerifierReduction {
        proof: prepared.proof,
        z: prepared.z,
        e: prepared.e,
        t: prepared.t,
        et_tail: prepared.et_tail,
    })
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
        validate_clear_image::<F>(chain.plan, &chain.proof.clear_image)?;
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
        claims.push(verify_chain_from_clear::<F, E, T>(
            transcript,
            chain.plan,
            &chain.matrices,
            chain.proof,
        )?);
    }
    Ok(claims)
}

fn verify_chain_from_clear<F, E, T>(
    transcript: &mut T,
    plan: &JlProjectionChainPlan,
    matrices: &[TernaryProjectionMatrix],
    proof: &JlProjectionProof<E>,
) -> Result<JlSourceClaim<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let final_layer = plan
        .layers()
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let output_point = sample_jl_image_point::<F, E, T>(transcript, final_layer.output_num_vars()?);
    let clear_field = try_embed_i128::<E>(&proof.clear_image)?;
    let output_claim = eval_block_tensor_mle(
        &clear_field,
        final_layer.blocks(),
        final_layer.matrix_member().shape()?.rows(),
        &output_point,
    )?;
    drop(clear_field);
    verify_chain_at::<F, E, T>(
        transcript,
        plan,
        matrices,
        &proof.reverse_layers,
        output_point,
        output_claim,
    )
}

fn verify_chain_at<F, E, T>(
    transcript: &mut T,
    plan: &JlProjectionChainPlan,
    matrices: &[TernaryProjectionMatrix],
    reverse_layers: &[akita_types::JlLayerReductionProof<E>],
    mut output_point: Vec<E>,
    mut output_claim: E,
) -> Result<JlSourceClaim<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    if reverse_layers.len() != plan.layers().len() || matrices.len() != plan.layers().len() {
        return Err(AkitaError::InvalidProof);
    }
    for ((layer_proof, layer), matrix) in reverse_layers
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

/// Verify the Z chain, ET tail, selector join, and private E/T stems.
pub fn verify_jl_aligned_et_reduction<F, E, T>(
    transcript: &mut T,
    ready: JlAlignedEtVerifierReduction<'_, E>,
) -> Result<JlAlignedEtSourceClaims<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let z = verify_chain_from_clear::<F, E, T>(
        transcript,
        ready.z.plan,
        &ready.z.matrices,
        &ready.proof.z,
    )?;
    let joined = verify_chain_from_clear::<F, E, T>(
        transcript,
        ready.et_tail.plan,
        &ready.et_tail.matrices,
        &ready.proof.et_tail,
    )?;
    let (selector, inner_point) = joined.point.split_last().ok_or(AkitaError::InvalidProof)?;
    absorb_ext_field::<F, E, T>(
        transcript,
        labels::ABSORB_JL_ET_STEM_EVALUATION,
        &ready.proof.e_stem_image_evaluation,
    );
    absorb_ext_field::<F, E, T>(
        transcript,
        labels::ABSORB_JL_ET_STEM_EVALUATION,
        &ready.proof.t_stem_image_evaluation,
    );
    let expected_join = ready.proof.e_stem_image_evaluation * (E::one() - *selector)
        + ready.proof.t_stem_image_evaluation * *selector;
    if expected_join != joined.evaluation {
        return Err(AkitaError::InvalidProof);
    }
    let e = verify_chain_at::<F, E, T>(
        transcript,
        ready.e.plan,
        &ready.e.matrices,
        &ready.proof.e_stem_reverse_layers,
        try_copy_field(inner_point)?,
        ready.proof.e_stem_image_evaluation,
    )?;
    let t = verify_chain_at::<F, E, T>(
        transcript,
        ready.t.plan,
        &ready.t.matrices,
        &ready.proof.t_stem_reverse_layers,
        try_copy_field(inner_point)?,
        ready.proof.t_stem_image_evaluation,
    )?;
    Ok(JlAlignedEtSourceClaims { z, e, t })
}

fn validate_clear_image<F: Field + CanonicalEncoding>(
    plan: &JlProjectionChainPlan,
    image: &[i128],
) -> Result<(), AkitaError> {
    for &coordinate in image {
        validate_centered_i128::<F>(coordinate).map_err(|_| AkitaError::InvalidProof)?;
    }
    let bound = plan.final_energy_bound().ok_or(AkitaError::InvalidProof)?;
    let energy = akita_algebra::jl::squared_l2_i128(image)?;
    if energy > bound {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

fn try_copy_field<E: Field>(values: &[E]) -> Result<Vec<E>, AkitaError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| AkitaError::InvalidInput("JL point allocation failed".into()))?;
    output.extend_from_slice(values);
    Ok(output)
}

fn try_embed_i128<E: Field>(values: &[i128]) -> Result<Vec<E>, AkitaError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| AkitaError::InvalidInput("JL field-vector allocation failed".into()))?;
    output.extend(values.iter().copied().map(E::from_i128));
    Ok(output)
}
