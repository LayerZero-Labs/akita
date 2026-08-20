use super::plan::CheckedStandardBatch;
use super::shared::{
    absorb_claims_and_sample_coefficients, combine_standard_polynomials,
    finish_and_check_terminals, gather_group_values, group_weighted_claims, invalid, mul_pow_2,
    previous_local_challenge, validate_execution_inputs, validate_standard_executor_message,
};
use super::{
    CheckedLocalRound, CheckedRoundContext, CheckedRoundRequest, GroupRoundMessage,
    SumcheckRoundExecutor,
};
use crate::{SumcheckProof, UniPoly};
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};

/// Result of the standard executor shell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StandardBatchExecution<E: FieldCore> {
    /// Existing standard proof container, with one combined message per master round.
    pub proof: SumcheckProof<E>,
    /// Shared master challenge point.
    pub master_point: Vec<E>,
    /// Front-loaded coefficient for each logical member.
    pub batching_coefficients: Vec<E>,
    /// Raw terminal claims restored to logical member order.
    pub terminal_claims: Vec<E>,
}

/// Drive a checked standard front-loaded batch through group executors.
pub fn prove_standard_executor_batch<F, T, E, S>(
    plan: &CheckedStandardBatch,
    input_claims: &[E],
    executors: &mut [Box<dyn SumcheckRoundExecutor<E>>],
    transcript: &mut T,
    mut sample_challenge: S,
) -> Result<StandardBatchExecution<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + HalvingField + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    let geometry = &plan.0;
    validate_execution_inputs(geometry, input_claims, executors.len())?;
    let coefficients =
        absorb_claims_and_sample_coefficients(input_claims, transcript, &mut sample_challenge)?;
    let group_coefficients = gather_group_values(geometry, &coefficients)?;
    let mut group_claims = group_weighted_claims(geometry, input_claims, &coefficients)?;
    for (claim, group) in group_claims.iter_mut().zip(&geometry.groups) {
        *claim = mul_pow_2(*claim, group.suffix_offset());
    }

    let mut round_polys = Vec::new();
    round_polys
        .try_reserve_exact(geometry.master_rounds)
        .map_err(|_| invalid("standard proof allocation failed"))?;
    let mut master_point = Vec::new();
    master_point
        .try_reserve_exact(geometry.master_rounds)
        .map_err(|_| invalid("standard challenge allocation failed"))?;

    for master_round in 0..geometry.master_rounds {
        for (group_index, group) in geometry.groups.iter().enumerate() {
            let Some(local_round) = group.local_round(master_round) else {
                continue;
            };
            let previous_challenge = previous_local_challenge(&master_point, local_round)?;
            let executor = executors
                .get_mut(group_index)
                .ok_or_else(|| invalid("sumcheck executor is missing"))?;
            let previous_claim = *group_claims
                .get(group_index)
                .ok_or_else(|| invalid("standard group claim is missing"))?;
            let batching_coefficients = group_coefficients
                .get(group_index)
                .ok_or_else(|| invalid("standard group coefficients are missing"))?;
            executor.start_round(CheckedRoundRequest {
                round: CheckedLocalRound {
                    master_round,
                    local_round,
                    local_rounds: group.num_rounds(),
                },
                previous_challenge,
                context: CheckedRoundContext::Standard {
                    previous_claim,
                    batching_coefficients,
                },
            })?;
        }

        let mut messages = Vec::new();
        messages
            .try_reserve_exact(geometry.groups.len())
            .map_err(|_| invalid("standard round message allocation failed"))?;
        for (group_index, group) in geometry.groups.iter().enumerate() {
            let group_claim = *group_claims
                .get(group_index)
                .ok_or_else(|| invalid("standard group claim is missing"))?;
            if group.local_round(master_round).is_some() {
                let executor = executors
                    .get_mut(group_index)
                    .ok_or_else(|| invalid("sumcheck executor is missing"))?;
                let GroupRoundMessage::Standard(poly) = executor.finish_round()? else {
                    return Err(invalid("standard executor returned an eq-factored message"));
                };
                validate_standard_executor_message(&poly, group.degree_bound())?;
                if poly.evaluate(&E::zero()) + poly.evaluate(&E::one()) != group_claim {
                    return Err(invalid(
                        "standard executor message violates claim recurrence",
                    ));
                }
                messages.push(poly);
            } else {
                messages.push(UniPoly::from_coeffs(vec![group_claim.half()]));
            }
        }

        let batched_poly = combine_standard_polynomials(&messages)?;
        let combined_claim = group_claims
            .iter()
            .copied()
            .fold(E::zero(), |acc, claim| acc + claim);
        if batched_poly.evaluate(&E::zero()) + batched_poly.evaluate(&E::one()) != combined_claim {
            return Err(invalid(
                "combined standard message violates claim recurrence",
            ));
        }
        let compressed = batched_poly.compress();
        transcript.append_serde(labels::ABSORB_SUMCHECK_ROUND, &compressed);
        let challenge = sample_challenge(transcript);
        master_point.push(challenge);
        for (claim, poly) in group_claims.iter_mut().zip(&messages) {
            *claim = poly.evaluate(&challenge);
        }
        round_polys.push(compressed);
    }

    let terminal_claims = finish_and_check_terminals(
        geometry,
        executors,
        &coefficients,
        &group_claims,
        master_point.last().copied(),
        None,
    )?;
    let proof = SumcheckProof { round_polys };
    plan.validate_proof(&proof)?;
    Ok(StandardBatchExecution {
        proof,
        master_point,
        batching_coefficients: coefficients,
        terminal_claims,
    })
}
