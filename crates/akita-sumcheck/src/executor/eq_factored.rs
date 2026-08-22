use super::plan::CheckedEqFactoredBatch;
use super::shared::{
    absorb_claims_and_sample_coefficients, combine_eq_polynomials, constant_eq_polynomial,
    finish_and_check_terminals, gather_group_values, group_weighted_claims, invalid,
    previous_local_challenge, validate_eq_message, validate_execution_inputs, EqTerminalScaling,
};
use super::{
    CheckedLocalRound, CheckedRoundContext, CheckedRoundRequest, GroupRoundMessage,
    SumcheckRoundExecutor,
};
use crate::{advance_eq_factored_claim, EqFactoredSumcheckProof};
use akita_error::AkitaError;
use akita_field::{CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};

/// Result of a master-lifted eq-factored executor batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EqFactoredBatchExecution<E: FieldCore> {
    /// Existing eq-factored proof container, with one combined message per round.
    pub proof: EqFactoredSumcheckProof<E>,
    /// Shared master challenge point.
    pub master_point: Vec<E>,
    /// Front-loaded coefficient for each logical member.
    pub batching_coefficients: Vec<E>,
    /// Raw terminal claims restored to logical member order.
    pub terminal_claims: Vec<E>,
}

#[derive(Clone, Copy)]
struct EqSharedState<E: FieldCore> {
    claim_scale: E,
    equality_scalar: E,
}

/// Drive a checked master-lifted eq-factored batch through group executors.
///
/// A group with suffix offset `k` represents
/// `eq(tau[..k], x[..k]) * F(x[k..])`. During its virtual prefix, the protocol
/// emits the group's constant weighted input claim without starting the source
/// executor. Once the suffix starts, the executor returns the ordinary local
/// `q`: the lifted relation and master factor contain the same prefix scalar.
pub fn prove_eq_factored_executor_batch<F, T, E, S>(
    plan: &CheckedEqFactoredBatch<E>,
    input_claims: &[E],
    executors: &mut [Box<dyn SumcheckRoundExecutor<E>>],
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<EqFactoredBatchExecution<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    let geometry = &plan.geometry;
    validate_execution_inputs(geometry, input_claims, executors.len())?;
    let coefficients =
        absorb_claims_and_sample_coefficients(input_claims, transcript, &mut sample_challenge)?;
    let group_coefficients = gather_group_values(geometry, &coefficients)?;
    let group_input_claims = group_weighted_claims(geometry, input_claims, &coefficients)?;
    let mut group_claims = Vec::new();
    group_claims
        .try_reserve_exact(group_input_claims.len())
        .map_err(|_| invalid("eq-factored group claim allocation failed"))?;
    group_claims.extend_from_slice(&group_input_claims);
    let mut shared = EqSharedState {
        claim_scale: E::one(),
        equality_scalar: E::one(),
    };

    let mut round_polys = Vec::new();
    round_polys
        .try_reserve_exact(geometry.master_rounds)
        .map_err(|_| invalid("eq-factored proof allocation failed"))?;
    let mut master_point = Vec::new();
    master_point
        .try_reserve_exact(geometry.master_rounds)
        .map_err(|_| invalid("eq-factored challenge allocation failed"))?;

    for master_round in 0..geometry.master_rounds {
        let tau = plan.equality_coordinate(master_round)?;
        let factor_at_one = shared.equality_scalar * tau;
        let factor_at_zero = shared.equality_scalar - factor_at_one;

        for (group_index, group) in geometry.groups.iter().enumerate() {
            let Some(local_round) = group.local_round(master_round) else {
                continue;
            };
            let previous_challenge = previous_local_challenge(&master_point, local_round)?;
            let master_lift_prefix = plan.group_prefix_scalar(&master_point, group_index)?;
            let executor = executors
                .get_mut(group_index)
                .ok_or_else(|| invalid("sumcheck executor is missing"))?;
            let scaled_claim = *group_claims
                .get(group_index)
                .ok_or_else(|| invalid("eq-factored group claim is missing"))?;
            let batching_coefficients = group_coefficients
                .get(group_index)
                .ok_or_else(|| invalid("eq-factored group coefficients are missing"))?;
            executor.start_round(CheckedRoundRequest {
                round: CheckedLocalRound {
                    master_round,
                    local_round,
                    local_rounds: group.num_rounds(),
                },
                previous_challenge,
                context: CheckedRoundContext::EqFactored {
                    scaled_claim,
                    claim_scale: shared.claim_scale,
                    factor_at_zero,
                    factor_at_one,
                    master_lift_prefix,
                    batching_coefficients,
                },
            })?;
        }

        let mut group_messages = Vec::new();
        group_messages
            .try_reserve_exact(geometry.groups.len())
            .map_err(|_| invalid("eq-factored round message allocation failed"))?;
        for (group_index, group) in geometry.groups.iter().enumerate() {
            let polynomial = if group.local_round(master_round).is_some() {
                let executor = executors
                    .get_mut(group_index)
                    .ok_or_else(|| invalid("sumcheck executor is missing"))?;
                let GroupRoundMessage::EqFactored(polynomial) = executor.finish_round()? else {
                    return Err(invalid("eq-factored executor returned a standard message"));
                };
                polynomial
            } else {
                let input_claim = group_input_claims
                    .get(group_index)
                    .copied()
                    .ok_or_else(|| invalid("eq-factored group input claim is missing"))?;
                constant_eq_polynomial(input_claim, group.degree_bound())?
            };
            validate_eq_message(&polynomial, group.degree_bound())?;
            group_messages.push(polynomial);
        }

        let combined = combine_eq_polynomials(&group_messages)?;
        let degree_bound = geometry
            .groups
            .iter()
            .map(|group| group.degree_bound())
            .max()
            .ok_or_else(|| invalid("sumcheck batch must contain at least one group"))?;
        validate_eq_message(&combined, degree_bound)?;
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &combined);
        let challenge = sample_challenge(transcript);
        master_point.push(challenge);

        let next_claim_scale = shared.claim_scale * factor_at_one;
        for ((claim, polynomial), _) in group_claims
            .iter_mut()
            .zip(&group_messages)
            .zip(&geometry.groups)
        {
            let (next_claim, computed_scale) = advance_eq_factored_claim(
                *claim,
                shared.claim_scale,
                factor_at_zero,
                factor_at_one,
                polynomial,
                challenge,
            );
            if computed_scale != next_claim_scale {
                return Err(invalid("eq-factored claim scale diverged between groups"));
            }
            *claim = next_claim;
        }
        shared.claim_scale = next_claim_scale;
        shared.equality_scalar *= tau * challenge + (E::one() - tau) * (E::one() - challenge);
        round_polys.push(combined);
    }

    let mut group_prefix_scalars = Vec::new();
    group_prefix_scalars
        .try_reserve_exact(geometry.groups.len())
        .map_err(|_| invalid("eq-factored prefix scalar allocation failed"))?;
    for group_index in 0..geometry.groups.len() {
        group_prefix_scalars.push(plan.group_prefix_scalar(&master_point, group_index)?);
    }
    let terminal_claims = finish_and_check_terminals(
        geometry,
        executors,
        &coefficients,
        &group_claims,
        master_point.last().copied(),
        Some(EqTerminalScaling {
            claim_scale: shared.claim_scale,
            group_prefix_scalars: &group_prefix_scalars,
        }),
    )?;
    let proof = EqFactoredSumcheckProof { round_polys };
    plan.validate_proof(&proof)?;
    Ok(EqFactoredBatchExecution {
        proof,
        master_point,
        batching_coefficients: coefficients,
        terminal_claims,
    })
}
