//! Schedule-inert prover for iterated JL block projection reductions.

use akita_algebra::jl::{
    build_block_projection_weight_table, eval_block_tensor_mle,
    eval_power_of_two_block_projection_reduction_factor,
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
    JlBlockLayerPlan, JlLayerReductionProof, JlProjectionChainPlan, JlProjectionProof,
    JL_PROJECTION_REDUCTION_DEGREE,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

/// Build the clear image and prove every projection layer in reverse order.
///
/// The caller must have already bound the outgoing recursive witness. This
/// function then performs the schedule-fixed retry/seed, image, evaluation,
/// reduction, and terminal-claim order. It does not cut over any production
/// schedule or authenticate the returned source claim.
pub fn prove_jl_projection_chain<F, E, T>(
    transcript: &mut T,
    plan: &JlProjectionChainPlan,
    retry_index: u32,
    source: &[i128],
) -> Result<JlProjectionProof<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + AkitaSerialize,
    T: Transcript<F>,
{
    if source.len() != plan.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: plan.source_len(),
            actual: source.len(),
        });
    }
    let encoded_retry = (plan.max_retries() > 1).then_some(retry_index);
    let selected_retry = plan.selected_retry(encoded_retry)?;
    let master_seed = absorb_jl_retry_and_sample_seed::<F, T>(transcript, encoded_retry);
    let matrices = plan.derive_matrices(&master_seed, selected_retry)?;

    let mut values = Vec::new();
    values
        .try_reserve_exact(plan.layers().len() + 1)
        .map_err(|_| AkitaError::InvalidInput("JL projection chain allocation failed".into()))?;
    values.push(source.to_vec());
    for matrix in &matrices {
        let previous = values
            .last()
            .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
        values.push(matrix.project_i128_blocks(previous)?);
    }
    let clear_image = values
        .last()
        .cloned()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let energy = akita_algebra::jl::squared_l2_i128(&clear_image)?;
    if energy > plan.final_energy_bound() {
        return Err(AkitaError::InvalidInput(format!(
            "JL clear-image energy {energy} exceeds public bound {}",
            plan.final_energy_bound()
        )));
    }

    absorb_jl_clear_image::<F, T>(transcript, &clear_image);
    let final_layer = plan
        .layers()
        .last()
        .ok_or_else(|| AkitaError::InvalidInput("JL projection chain is empty".into()))?;
    let mut output_point =
        sample_jl_image_point::<F, E, T>(transcript, final_layer.output_num_vars()?);
    let clear_field = clear_image
        .iter()
        .copied()
        .map(E::from_i128)
        .collect::<Vec<_>>();
    let mut output_claim = eval_block_tensor_mle(
        &clear_field,
        final_layer.blocks(),
        final_layer.matrix_context(selected_retry).shape()?.rows(),
        &output_point,
    )?;

    let mut reverse_layers = Vec::new();
    reverse_layers
        .try_reserve_exact(plan.layers().len())
        .map_err(|_| AkitaError::InvalidInput("JL reduction proof allocation failed".into()))?;
    for layer_index in (0..plan.layers().len()).rev() {
        let layer = plan.layers()[layer_index];
        let matrix = &matrices[layer_index];
        let input = values[layer_index]
            .iter()
            .copied()
            .map(E::from_i128)
            .collect::<Vec<_>>();
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
            layer.matrix_context(selected_retry).shape()?.cols(),
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
        retry_index: encoded_retry,
        clear_image,
        reverse_layers,
    })
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
