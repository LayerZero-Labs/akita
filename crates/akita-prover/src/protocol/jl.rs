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
    JlBlockLayerPlan, JlLayerReductionProof, JlProjectionBatchPlan, JlProjectionBatchProof,
    JlProjectionChainPlan, JlProjectionProof, JL_PROJECTION_REDUCTION_DEGREE,
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

struct PreparedChain<'a> {
    plan: &'a JlProjectionChainPlan,
    matrices: Vec<TernaryProjectionMatrix>,
    values: Vec<Vec<i128>>,
}

/// Opaque batch whose retries/seeds and forward projections are fixed.
pub struct PreparedJlProjectionBatch<'a> {
    encoded_retry: Option<u32>,
    chains: Vec<PreparedChain<'a>>,
}

/// Opaque batch whose complete clear-image array has been absorbed.
pub struct JlProverReductionBatch<'a> {
    encoded_retry: Option<u32>,
    chains: Vec<PreparedChain<'a>>,
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
    if inputs.len() != plan.chains().len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.chains().len(),
            actual: inputs.len(),
        });
    }
    let encoded_retry = plan.retry_is_encoded().then_some(retry_index);
    let selected_retry = plan.selected_retry(encoded_retry)?;
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, encoded_retry);
    let envelopes = plan.derive_envelopes(&master_seed, selected_retry)?;
    let mut chains = Vec::new();
    chains
        .try_reserve_exact(inputs.len())
        .map_err(|_| AkitaError::InvalidInput("JL certificate-batch allocation failed".into()))?;
    for (chain_index, (input, chain_plan)) in inputs.iter().zip(plan.chains()).enumerate() {
        if input.source.len() != chain_plan.source_len() {
            return Err(AkitaError::InvalidSize {
                expected: chain_plan.source_len(),
                actual: input.source.len(),
            });
        }
        let matrices = plan.member_matrices(chain_index, &envelopes)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(chain_plan.layers().len() + 1)
            .map_err(|_| {
                AkitaError::InvalidInput("JL projection chain allocation failed".into())
            })?;
        values.push(try_copy_i128(input.source)?);
        for matrix in &matrices {
            let previous = values
                .last()
                .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
            values.push(matrix.project_centered_i128_blocks::<F>(previous)?);
        }
        chains.push(PreparedChain {
            plan: chain_plan,
            matrices,
            values,
        });
    }
    Ok(PreparedJlProjectionBatch {
        encoded_retry,
        chains,
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
    let mut proofs = Vec::new();
    proofs
        .try_reserve_exact(ready.chains.len())
        .map_err(|_| AkitaError::InvalidInput("JL proof-batch allocation failed".into()))?;
    for chain in ready.chains {
        proofs.push(prove_prepared_chain::<F, E, T>(transcript, chain)?);
    }
    Ok(JlProjectionBatchProof {
        retry_index: ready.encoded_retry,
        chains: proofs,
    })
}

fn prove_prepared_chain<F, E, T>(
    transcript: &mut T,
    chain: PreparedChain<'_>,
) -> Result<JlProjectionProof<E>, AkitaError>
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
    let clear_image = chain
        .values
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let clear_field = try_embed_i128::<E>(clear_image)?;
    let mut output_claim = eval_block_tensor_mle(
        &clear_field,
        final_layer.blocks(),
        final_layer.matrix_member().shape()?.rows(),
        &output_point,
    )?;

    let mut reverse_layers = Vec::new();
    reverse_layers
        .try_reserve_exact(chain.plan.layers().len())
        .map_err(|_| AkitaError::InvalidInput("JL reduction proof allocation failed".into()))?;
    for layer_index in (0..chain.plan.layers().len()).rev() {
        let layer = chain.plan.layers()[layer_index];
        let matrix = &chain.matrices[layer_index];
        let input = try_embed_i128::<E>(&chain.values[layer_index])?;
        let input_table = input.clone();
        let weights = build_block_projection_weight_table(matrix, layer.blocks(), &output_point)?;
        let mut prover =
            BlockProjectionReductionProver::new(layer, input_table, weights, output_claim)?;
        let (sumcheck, challenges, final_claim) = prover.prove::<F, T, _>(transcript, |tr| {
            Ok(sample_ext_challenge::<F, E, T>(
                tr,
                labels::CHALLENGE_SUMCHECK_ROUND,
            ))
        })?;
        let input_evaluation = eval_block_tensor_mle(
            &input,
            layer.blocks(),
            layer.matrix_member().shape()?.cols(),
            &challenges,
        )?;
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

    Ok(JlProjectionProof {
        clear_image: try_copy_i128(clear_image)?,
        reverse_layers,
    })
}

fn check_clear_image<F: Field + CanonicalEncoding>(
    plan: &JlProjectionChainPlan,
    clear_image: &[i128],
) -> Result<(), AkitaError> {
    for &coordinate in clear_image {
        validate_centered_i128::<F>(coordinate)?;
    }
    let energy = akita_algebra::jl::squared_l2_i128(clear_image)?;
    if energy > plan.final_energy_bound() {
        return Err(AkitaError::InvalidInput(format!(
            "JL clear-image energy {energy} exceeds public bound {}",
            plan.final_energy_bound()
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
