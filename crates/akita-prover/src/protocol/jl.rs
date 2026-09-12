//! Schedule-inert prover for iterated JL block projection reductions.

use akita_algebra::jl::{
    build_block_projection_weight_table, eval_block_tensor_mle,
    eval_power_of_two_block_projection_reduction_factor, validate_centered_i128,
    TernaryProjectionMatrix,
};
use akita_algebra::UniPoly;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_sumcheck::{SumcheckInstanceProver, SumcheckInstanceProverExt};
use akita_transcript::{
    absorb_ext_field, absorb_jl_clear_image, absorb_jl_retry_and_sample_seed, labels,
    sample_ext_challenge, sample_jl_image_point, Transcript,
};
use akita_types::{
    JlAlignedEtProjectionPlan, JlAlignedEtProjectionProof, JlBlockLayerPlan, JlLayerReductionProof,
    JlProjectionBatchPlan, JlProjectionBatchProof, JlProjectionChainPlan, JlProjectionProof,
    JlSourceClaim, JL_PROJECTION_REDUCTION_DEGREE,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// One schedule-owned source in batch-plan order.
///
/// An `EtTail` source is the already-formed selector join. This foundation
/// does not prove the private E/T stems or link them to that joined source;
/// callers must authenticate the returned deferred source claim later.
pub struct JlProjectionProverInput<'a> {
    /// Exact centered source coordinates.
    pub source: &'a [i128],
}

/// Authenticated source tables for an aligned `Z,E,T -> ET-tail` graph.
pub struct JlAlignedEtProverInput<'a> {
    /// Aggregate semantic Z.
    pub z: &'a [i128],
    /// Literal E table.
    pub e: &'a [i128],
    /// Literal T table.
    pub t: &'a [i128],
}

struct PreparedChain<'a> {
    plan: &'a JlProjectionChainPlan,
    matrices: Vec<TernaryProjectionMatrix>,
    values: Vec<Vec<i128>>,
}

/// Opaque batch whose retries/seeds and forward projections are fixed.
pub struct PreparedJlProjectionBatch<'a> {
    batch: &'a JlProjectionBatchPlan,
    encoded_retry: Option<u32>,
    chains: Vec<PreparedChain<'a>>,
}

/// Opaque batch whose complete clear-image array has been absorbed.
pub struct JlProverReductionBatch<'a> {
    batch: &'a JlProjectionBatchPlan,
    encoded_retry: Option<u32>,
    chains: Vec<PreparedChain<'a>>,
}

/// Opaque aligned graph after all envelopes and forward values are fixed.
pub struct PreparedJlAlignedEtProjection<'a> {
    batch: &'a JlProjectionBatchPlan,
    encoded_retry: Option<u32>,
    z: PreparedChain<'a>,
    e: PreparedChain<'a>,
    t: PreparedChain<'a>,
    et_tail: PreparedChain<'a>,
}

/// Opaque aligned graph after both public clear images are absorbed.
pub struct JlAlignedEtReductionBatch<'a> {
    batch: &'a JlProjectionBatchPlan,
    encoded_retry: Option<u32>,
    z: PreparedChain<'a>,
    e: PreparedChain<'a>,
    t: PreparedChain<'a>,
    et_tail: PreparedChain<'a>,
}

/// Fix the whole-forest candidate, derive every envelope, and project all chains.
///
/// The outgoing witness must already be bound. No image is absorbed and no
/// reduction challenge is sampled in this phase.
pub fn prepare_jl_projection_batch<'a, F, T>(
    transcript: &mut T,
    plan: &'a JlProjectionBatchPlan,
    retry_index: u32,
    inputs: &[JlProjectionProverInput<'_>],
) -> Result<PreparedJlProjectionBatch<'a>, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
{
    plan.validate_projection_workspace(std::mem::size_of::<F>())?;
    if inputs.len() != plan.chains().len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.chains().len(),
            actual: inputs.len(),
        });
    }
    let encoded_retry = plan.encoded_retry(retry_index)?;
    let selected_retry = plan.selected_retry(encoded_retry)?;
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, encoded_retry);
    let envelopes = plan.derive_envelopes(&master_seed, selected_retry)?;
    let mut chains = Vec::new();
    chains
        .try_reserve_exact(inputs.len())
        .map_err(|_| AkitaError::InvalidInput("JL certificate-batch allocation failed".into()))?;
    for (chain_index, (input, chain_plan)) in inputs.iter().zip(plan.chains()).enumerate() {
        chains.push(prepare_chain::<F>(
            plan,
            chain_index,
            chain_plan,
            &envelopes,
            try_copy_i128(input.source)?,
        )?);
    }
    Ok(PreparedJlProjectionBatch {
        batch: plan,
        encoded_retry,
        chains,
    })
}

fn prepare_chain<'a, F: Field + CanonicalEncoding>(
    batch: &JlProjectionBatchPlan,
    chain_index: usize,
    plan: &'a JlProjectionChainPlan,
    envelopes: &[TernaryProjectionMatrix],
    source: Vec<i128>,
) -> Result<PreparedChain<'a>, AkitaError> {
    if source.len() != plan.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.source_len(),
            actual: source.len(),
        });
    }
    let matrices = batch.member_matrices(chain_index, envelopes)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(plan.layers().len() + 1)
        .map_err(|_| AkitaError::InvalidInput("JL projection chain allocation failed".into()))?;
    values.push(source);
    for matrix in &matrices {
        let previous = values
            .last()
            .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
        values.push(matrix.project_centered_i128_blocks::<F>(previous)?);
    }
    Ok(PreparedChain {
        plan,
        matrices,
        values,
    })
}

/// Derive one whole-forest candidate and compute the aligned E/T projection graph.
pub fn prepare_jl_aligned_et_projection<'a, F, T>(
    transcript: &mut T,
    plan: &'a JlAlignedEtProjectionPlan,
    retry_index: u32,
    input: &JlAlignedEtProverInput<'_>,
) -> Result<PreparedJlAlignedEtProjection<'a>, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
{
    let batch = plan.batch();
    batch.validate_projection_workspace(std::mem::size_of::<F>())?;
    let encoded_retry = batch.encoded_retry(retry_index)?;
    let selected_retry = batch.selected_retry(encoded_retry)?;
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, encoded_retry);
    let envelopes = batch.derive_envelopes(&master_seed, selected_retry)?;
    let z = prepare_chain::<F>(batch, 0, plan.z()?, &envelopes, try_copy_i128(input.z)?)?;
    let e = prepare_chain::<F>(
        batch,
        1,
        plan.e_stem()?,
        &envelopes,
        try_copy_i128(input.e)?,
    )?;
    let t = prepare_chain::<F>(
        batch,
        2,
        plan.t_stem()?,
        &envelopes,
        try_copy_i128(input.t)?,
    )?;
    let e_image = e
        .values
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL E stem is empty".into()))?;
    let t_image = t
        .values
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL T stem is empty".into()))?;
    let joined = plan.join_stem_images(e_image, t_image)?;
    let et_tail = prepare_chain::<F>(batch, 3, plan.et_tail()?, &envelopes, joined)?;
    Ok(PreparedJlAlignedEtProjection {
        batch,
        encoded_retry,
        z,
        e,
        t,
        et_tail,
    })
}

/// Check and absorb Z and final ET images before any graph reduction.
pub fn absorb_jl_aligned_et_images<'a, F, T>(
    transcript: &mut T,
    prepared: PreparedJlAlignedEtProjection<'a>,
) -> Result<JlAlignedEtReductionBatch<'a>, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
{
    for chain in [&prepared.z, &prepared.et_tail] {
        let image = chain
            .values
            .last()
            .ok_or_else(|| AkitaError::InvalidInput("JL clear-image chain is empty".into()))?;
        check_clear_image::<F>(chain.plan, image)?;
    }
    for chain in [&prepared.z, &prepared.et_tail] {
        let image = chain
            .values
            .last()
            .ok_or_else(|| AkitaError::InvalidInput("JL clear-image chain is empty".into()))?;
        absorb_jl_clear_image::<F, T>(transcript, image)
            .map_err(|message| AkitaError::InvalidInput(message.into()))?;
    }
    Ok(JlAlignedEtReductionBatch {
        batch: prepared.batch,
        encoded_retry: prepared.encoded_retry,
        z: prepared.z,
        e: prepared.e,
        t: prepared.t,
        et_tail: prepared.et_tail,
    })
}

/// Check and absorb every clear image before any reduction challenge.
pub fn absorb_jl_projection_batch_images<'a, F, T>(
    transcript: &mut T,
    prepared: PreparedJlProjectionBatch<'a>,
) -> Result<JlProverReductionBatch<'a>, AkitaError>
where
    F: Field + CanonicalEncoding,
    T: Transcript<F>,
{
    for chain in &prepared.chains {
        let clear_image = chain
            .values
            .last()
            .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
        check_clear_image::<F>(chain.plan, clear_image)?;
    }
    for chain in &prepared.chains {
        let clear_image = chain
            .values
            .last()
            .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
        absorb_jl_clear_image::<F, T>(transcript, clear_image)
            .map_err(|message| AkitaError::InvalidInput(message.into()))?;
    }
    Ok(JlProverReductionBatch {
        batch: prepared.batch,
        encoded_retry: prepared.encoded_retry,
        chains: prepared.chains,
    })
}

/// Prove every reverse chain after the full image array has been absorbed.
///
/// Returned chain proofs establish only projection consistency. Their source
/// evaluations remain deferred obligations for the surrounding protocol.
pub fn prove_jl_projection_reduction_batch<F, E, T>(
    transcript: &mut T,
    ready: JlProverReductionBatch<'_>,
) -> Result<JlProjectionBatchProof<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    ready
        .batch
        .validate_prover_workspace(std::mem::size_of::<E>())?;
    let mut proofs = Vec::new();
    proofs
        .try_reserve_exact(ready.chains.len())
        .map_err(|_| AkitaError::InvalidInput("JL proof-batch allocation failed".into()))?;
    for chain in ready.chains {
        let (proof, _) = prove_prepared_chain_with_claim::<F, E, T>(transcript, chain)?;
        proofs.push(proof);
    }
    Ok(JlProjectionBatchProof {
        retry_index: ready.encoded_retry,
        chains: proofs,
    })
}

fn prove_prepared_chain_with_claim<F, E, T>(
    transcript: &mut T,
    chain: PreparedChain<'_>,
) -> Result<(JlProjectionProof<E>, JlSourceClaim<E>), AkitaError>
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
    let output_point = sample_jl_image_point::<F, E, T>(transcript, final_layer.output_num_vars()?);
    let clear_image = chain
        .values
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let clear_field = try_embed_i128::<E>(clear_image)?;
    let output_claim = eval_block_tensor_mle(
        &clear_field,
        final_layer.blocks(),
        final_layer.matrix_member().shape()?.rows(),
        &output_point,
    )?;
    drop(clear_field);

    let clear_image = try_copy_i128(clear_image)?;
    let (reverse_layers, source_claim) =
        prove_prepared_chain_at::<F, E, T>(transcript, chain, output_point, output_claim)?;
    Ok((
        JlProjectionProof {
            clear_image,
            reverse_layers,
        },
        source_claim,
    ))
}

fn prove_prepared_chain_at<F, E, T>(
    transcript: &mut T,
    chain: PreparedChain<'_>,
    mut output_point: Vec<E>,
    mut output_claim: E,
) -> Result<(Vec<JlLayerReductionProof<E>>, JlSourceClaim<E>), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    let expected_values = chain
        .plan
        .layers()
        .len()
        .checked_add(1)
        .ok_or_else(|| AkitaError::InvalidInput("JL projection value count overflow".into()))?;
    if chain.matrices.len() != chain.plan.layers().len() || chain.values.len() != expected_values {
        return Err(AkitaError::InvalidInput(
            "JL prepared chain has inconsistent layer storage".into(),
        ));
    }
    let mut reverse_layers = Vec::new();
    reverse_layers
        .try_reserve_exact(chain.plan.layers().len())
        .map_err(|_| AkitaError::InvalidInput("JL reduction proof allocation failed".into()))?;
    for ((layer, matrix), input_values) in chain
        .plan
        .layers()
        .iter()
        .zip(&chain.matrices)
        .zip(&chain.values)
        .rev()
    {
        let layer = *layer;
        let input_table = try_embed_i128::<E>(input_values)?;
        let weights = build_block_projection_weight_table(matrix, layer.blocks(), &output_point)?;
        let mut prover =
            BlockProjectionReductionProver::new(layer, input_table, weights, output_claim)?;
        let (sumcheck, challenges, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
            Ok(sample_ext_challenge::<F, E, T>(
                tr,
                labels::CHALLENGE_SUMCHECK_ROUND,
            ))
        })?;
        let input_evaluation = prover.terminal_input_evaluation()?;
        drop(prover);
        let factor = eval_power_of_two_block_projection_reduction_factor(
            matrix,
            layer.blocks(),
            &output_point,
            &challenges,
        )?;
        if final_claim != input_evaluation * factor {
            return Err(AkitaError::InvalidInput(
                "JL prover reduction terminal is inconsistent".into(),
            ));
        }
        absorb_ext_field::<F, E, T>(
            transcript,
            labels::ABSORB_JL_REDUCTION_TERMINAL,
            &input_evaluation,
        );
        reverse_layers.push(JlLayerReductionProof {
            sumcheck,
            input_evaluation,
        });
        output_point = challenges;
        output_claim = input_evaluation;
    }

    Ok((
        reverse_layers,
        JlSourceClaim {
            point: output_point,
            evaluation: output_claim,
        },
    ))
}

/// Prove the Z chain, ET tail, selector join, then private E and T stems.
pub fn prove_jl_aligned_et_reduction<F, E, T>(
    transcript: &mut T,
    ready: JlAlignedEtReductionBatch<'_>,
) -> Result<JlAlignedEtProjectionProof<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    ready
        .batch
        .validate_prover_workspace(std::mem::size_of::<E>())?;
    let (z, _) = prove_prepared_chain_with_claim::<F, E, T>(transcript, ready.z)?;
    let (et_tail, joined_claim) =
        prove_prepared_chain_with_claim::<F, E, T>(transcript, ready.et_tail)?;
    let (selector, inner_point) = joined_claim.point.split_last().ok_or_else(|| {
        AkitaError::InvalidInput("JL aligned tail source point has no selector".into())
    })?;
    let expected_inner_vars = ready
        .e
        .plan
        .layers()
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL E stem is empty".into()))?
        .output_num_vars()?;
    if inner_point.len() != expected_inner_vars {
        return Err(AkitaError::InvalidPointDimension {
            expected: expected_inner_vars,
            actual: inner_point.len(),
        });
    }
    let e_stem_image_evaluation = eval_prepared_image::<E>(&ready.e, inner_point)?;
    let t_stem_image_evaluation = eval_prepared_image::<E>(&ready.t, inner_point)?;
    absorb_ext_field::<F, E, T>(
        transcript,
        labels::ABSORB_JL_ET_STEM_EVALUATION,
        &e_stem_image_evaluation,
    );
    absorb_ext_field::<F, E, T>(
        transcript,
        labels::ABSORB_JL_ET_STEM_EVALUATION,
        &t_stem_image_evaluation,
    );
    let joined_evaluation =
        e_stem_image_evaluation * (E::one() - *selector) + t_stem_image_evaluation * *selector;
    if joined_evaluation != joined_claim.evaluation {
        return Err(AkitaError::InvalidInput(
            "JL aligned selector join is inconsistent".into(),
        ));
    }
    let (e_stem_reverse_layers, _) = prove_prepared_chain_at::<F, E, T>(
        transcript,
        ready.e,
        inner_point.to_vec(),
        e_stem_image_evaluation,
    )?;
    let (t_stem_reverse_layers, _) = prove_prepared_chain_at::<F, E, T>(
        transcript,
        ready.t,
        inner_point.to_vec(),
        t_stem_image_evaluation,
    )?;
    Ok(JlAlignedEtProjectionProof {
        retry_index: ready.encoded_retry,
        z,
        et_tail,
        e_stem_image_evaluation,
        t_stem_image_evaluation,
        e_stem_reverse_layers,
        t_stem_reverse_layers,
    })
}

fn eval_prepared_image<E: Field>(chain: &PreparedChain<'_>, point: &[E]) -> Result<E, AkitaError> {
    let final_layer = chain
        .plan
        .layers()
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let image = chain
        .values
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let field_image = try_embed_i128::<E>(image)?;
    eval_block_tensor_mle(
        &field_image,
        final_layer.blocks(),
        final_layer.matrix_member().shape()?.rows(),
        point,
    )
}

fn check_clear_image<F: Field + CanonicalEncoding>(
    plan: &JlProjectionChainPlan,
    clear_image: &[i128],
) -> Result<(), AkitaError> {
    for &coordinate in clear_image {
        validate_centered_i128::<F>(coordinate)?;
    }
    let bound = plan.final_energy_bound().ok_or_else(|| {
        AkitaError::InvalidInput("JL private stem has no clear-image energy predicate".into())
    })?;
    let energy = akita_algebra::jl::squared_l2_i128(clear_image)?;
    if energy > bound {
        return Err(AkitaError::InvalidInput(format!(
            "JL clear-image energy {energy} exceeds public bound {bound}",
        )));
    }
    Ok(())
}

fn try_copy_i128(values: &[i128]) -> Result<Vec<i128>, AkitaError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| AkitaError::InvalidInput("JL integer-vector allocation failed".into()))?;
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

#[derive(Clone, Debug)]
struct BlockProjectionReductionProver<E: Field> {
    num_rounds: usize,
    input_claim: E,
    input_table: Vec<E>,
    weight_table: Vec<E>,
}

impl<E: Field> BlockProjectionReductionProver<E> {
    fn new(
        layer: JlBlockLayerPlan,
        input_table: Vec<E>,
        weight_table: Vec<E>,
        input_claim: E,
    ) -> Result<Self, AkitaError> {
        if input_table.len() != weight_table.len() || !input_table.len().is_power_of_two() {
            return Err(AkitaError::InvalidInput(
                "JL reduction tables have inconsistent geometry".into(),
            ));
        }
        let num_rounds = layer.reduction_num_vars()?;
        let expected = 1usize
            .checked_shl(u32::try_from(num_rounds).map_err(|_| {
                AkitaError::InvalidInput("JL reduction dimension exceeds u32".into())
            })?)
            .ok_or_else(|| AkitaError::InvalidInput("JL reduction table length overflow".into()))?;
        if input_table.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: input_table.len(),
            });
        }
        Ok(Self {
            num_rounds,
            input_claim,
            input_table,
            weight_table,
        })
    }

    fn terminal_input_evaluation(&self) -> Result<E, AkitaError> {
        if self.input_table.len() != 1 {
            return Err(AkitaError::InvalidInput(
                "JL sumcheck did not fold its input table to one value".into(),
            ));
        }
        self.input_table.first().copied().ok_or_else(|| {
            AkitaError::InvalidInput("JL sumcheck terminal input table is empty".into())
        })
    }
}

impl<E: Field> SumcheckInstanceProver<E> for BlockProjectionReductionProver<E> {
    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        JL_PROJECTION_REDUCTION_DEGREE
    }

    fn input_claim(&self) -> E {
        self.input_claim
    }

    fn compute_round_univariate(&mut self, _round: usize, _previous_claim: E) -> UniPoly<E> {
        let (constant, linear, quadratic) =
            accumulate_product_round(&self.input_table, &self.weight_table);
        UniPoly::from_coeffs(vec![constant, linear, quadratic])
    }

    fn ingest_challenge(&mut self, _round: usize, challenge: E) {
        fold_table(&mut self.input_table, challenge);
        fold_table(&mut self.weight_table, challenge);
    }
}

fn accumulate_product_round<E: Field>(lhs: &[E], rhs: &[E]) -> (E, E, E) {
    let mut constant = E::zero();
    let mut linear = E::zero();
    let mut quadratic = E::zero();
    for (left, right) in lhs.chunks_exact(2).zip(rhs.chunks_exact(2)) {
        let left_delta = left[1] - left[0];
        let right_delta = right[1] - right[0];
        constant += left[0] * right[0];
        linear += left[0] * right_delta + left_delta * right[0];
        quadratic += left_delta * right_delta;
    }
    (constant, linear, quadratic)
}

fn fold_table<E: Field>(table: &mut Vec<E>, challenge: E) {
    let half = table.len() / 2;
    for pair in 0..half {
        let left = table[2 * pair];
        let right = table[2 * pair + 1];
        table[pair] = left + challenge * (right - left);
    }
    table.truncate(half);
}
