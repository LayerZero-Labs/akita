use super::plan::CheckedBatchGeometry;
use super::{GroupTerminalClaims, SumcheckRoundExecutor};
use crate::{CompressedUniPoly, EqFactoredUniPoly, UniPoly};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_serialization::AkitaSerialize;
use akita_transcript::{labels, Transcript};

pub(super) fn validate_execution_inputs<E: FieldCore>(
    geometry: &CheckedBatchGeometry,
    input_claims: &[E],
    executor_count: usize,
) -> Result<(), AkitaError> {
    if input_claims.len() != geometry.members.len() {
        return Err(AkitaError::InvalidSize {
            expected: geometry.members.len(),
            actual: input_claims.len(),
        });
    }
    if executor_count != geometry.groups.len() {
        return Err(AkitaError::InvalidSize {
            expected: geometry.groups.len(),
            actual: executor_count,
        });
    }
    Ok(())
}

pub(super) fn absorb_claims_and_sample_coefficients<F, T, E, S>(
    input_claims: &[E],
    transcript: &mut T,
    sample_challenge: &mut S,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
    E: FieldCore + AkitaSerialize,
    S: FnMut(&mut T) -> E,
{
    for claim in input_claims {
        transcript.append_serde(labels::ABSORB_SUMCHECK_CLAIM, claim);
    }
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(input_claims.len())
        .map_err(|_| invalid("batching coefficient allocation failed"))?;
    for _ in input_claims {
        coefficients.push(sample_challenge(transcript));
    }
    Ok(coefficients)
}

pub(super) fn gather_group_values<E: Copy>(
    geometry: &CheckedBatchGeometry,
    values: &[E],
) -> Result<Vec<Vec<E>>, AkitaError> {
    let mut grouped = Vec::new();
    grouped
        .try_reserve_exact(geometry.groups.len())
        .map_err(|_| invalid("group value allocation failed"))?;
    for group in &geometry.groups {
        let mut group_values = Vec::new();
        group_values
            .try_reserve_exact(group.member_indices().len())
            .map_err(|_| invalid("group value allocation failed"))?;
        for &member_index in group.member_indices() {
            group_values.push(
                *values
                    .get(member_index)
                    .ok_or_else(|| invalid("sumcheck group member index is out of range"))?,
            );
        }
        grouped.push(group_values);
    }
    Ok(grouped)
}

pub(super) fn group_weighted_claims<E: FieldCore>(
    geometry: &CheckedBatchGeometry,
    claims: &[E],
    coefficients: &[E],
) -> Result<Vec<E>, AkitaError> {
    let mut out = Vec::new();
    out.try_reserve_exact(geometry.groups.len())
        .map_err(|_| invalid("group claim allocation failed"))?;
    for group in &geometry.groups {
        let mut claim = E::zero();
        for &member_index in group.member_indices() {
            let member_claim = claims
                .get(member_index)
                .ok_or_else(|| invalid("sumcheck group member index is out of range"))?;
            let coefficient = coefficients
                .get(member_index)
                .ok_or_else(|| invalid("sumcheck batching coefficient is missing"))?;
            claim += *member_claim * *coefficient;
        }
        out.push(claim);
    }
    Ok(out)
}

pub(super) fn previous_local_challenge<E: Copy>(
    master_point: &[E],
    local_round: usize,
) -> Result<Option<E>, AkitaError> {
    if local_round == 0 {
        return Ok(None);
    }
    master_point
        .last()
        .copied()
        .map(Some)
        .ok_or_else(|| invalid("previous local challenge is missing"))
}

pub(super) fn finish_and_check_terminals<E: FieldCore>(
    geometry: &CheckedBatchGeometry,
    executors: &mut [Box<dyn SumcheckRoundExecutor<E>>],
    coefficients: &[E],
    final_group_claims: &[E],
    final_active_challenge: Option<E>,
    eq_claim_scale: Option<E>,
) -> Result<Vec<E>, AkitaError> {
    let mut terminal_claims = Vec::new();
    terminal_claims
        .try_reserve_exact(geometry.members.len())
        .map_err(|_| invalid("terminal claim allocation failed"))?;
    terminal_claims.resize(geometry.members.len(), E::zero());
    for (group_index, group) in geometry.groups.iter().enumerate() {
        let final_challenge = (group.num_rounds() > 0)
            .then_some(final_active_challenge)
            .flatten();
        if group.num_rounds() > 0 && final_challenge.is_none() {
            return Err(invalid("final active challenge is missing"));
        }
        let executor = executors
            .get_mut(group_index)
            .ok_or_else(|| invalid("sumcheck executor is missing"))?;
        let GroupTerminalClaims { claims } = executor.finish_binding(final_challenge)?;
        if claims.len() != group.member_indices().len() {
            return Err(AkitaError::InvalidSize {
                expected: group.member_indices().len(),
                actual: claims.len(),
            });
        }
        let mut weighted_terminal = E::zero();
        for (&member_index, terminal) in group.member_indices().iter().zip(&claims) {
            let coefficient = coefficients
                .get(member_index)
                .ok_or_else(|| invalid("sumcheck batching coefficient is missing"))?;
            weighted_terminal += *terminal * *coefficient;
            let output = terminal_claims
                .get_mut(member_index)
                .ok_or_else(|| invalid("terminal claim member index is out of range"))?;
            *output = *terminal;
        }
        let expected = eq_claim_scale.map_or(weighted_terminal, |scale| weighted_terminal * scale);
        let final_claim = final_group_claims
            .get(group_index)
            .ok_or_else(|| invalid("final group claim is missing"))?;
        if *final_claim != expected {
            return Err(invalid(
                "executor terminal claims violate the final recurrence",
            ));
        }
    }
    Ok(terminal_claims)
}

pub(super) fn combine_standard_polynomials<E: FieldCore>(
    polys: &[UniPoly<E>],
) -> Result<UniPoly<E>, AkitaError> {
    let max_len = polys
        .iter()
        .map(|poly| poly.coeffs.len())
        .max()
        .unwrap_or(0);
    if max_len == 0 {
        return Err(invalid("standard round messages cannot all be empty"));
    }
    let mut coeffs = Vec::new();
    coeffs
        .try_reserve_exact(max_len)
        .map_err(|_| invalid("combined round polynomial allocation failed"))?;
    coeffs.resize(max_len, E::zero());
    for poly in polys {
        for (index, coefficient) in poly.coeffs.iter().enumerate() {
            let output = coeffs
                .get_mut(index)
                .ok_or_else(|| invalid("combined round coefficient is out of range"))?;
            *output += *coefficient;
        }
    }
    Ok(UniPoly::from_coeffs(coeffs))
}

pub(super) fn combine_eq_polynomials<E: FieldCore>(
    polys: &[EqFactoredUniPoly<E>],
) -> Result<EqFactoredUniPoly<E>, AkitaError> {
    let max_len = polys
        .iter()
        .map(|poly| poly.coeffs_except_linear_term.len())
        .max()
        .unwrap_or(0);
    if max_len == 0 {
        return Err(invalid("eq-factored round messages cannot all be empty"));
    }
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(max_len)
        .map_err(|_| invalid("combined eq-factored polynomial allocation failed"))?;
    coefficients.resize(max_len, E::zero());
    for poly in polys {
        for (index, coefficient) in poly.coeffs_except_linear_term.iter().enumerate() {
            let output = coefficients
                .get_mut(index)
                .ok_or_else(|| invalid("combined eq-factored coefficient is out of range"))?;
            *output += *coefficient;
        }
    }
    Ok(EqFactoredUniPoly {
        coeffs_except_linear_term: coefficients,
    })
}

pub(super) fn validate_standard_message<E: FieldCore>(
    poly: &CompressedUniPoly<E>,
    degree_bound: usize,
) -> Result<(), AkitaError> {
    if poly.coeffs_except_linear_term.is_empty() {
        return Err(invalid("standard round message cannot be empty"));
    }
    if poly.degree() > degree_bound {
        return Err(invalid("standard round message exceeds its degree bound"));
    }
    Ok(())
}

pub(super) fn validate_standard_executor_message<E: FieldCore>(
    poly: &UniPoly<E>,
    degree_bound: usize,
) -> Result<(), AkitaError> {
    if poly.coeffs.is_empty() {
        return Err(invalid("standard round message cannot be empty"));
    }
    if poly.degree() > degree_bound {
        return Err(invalid("standard round message exceeds its degree bound"));
    }
    Ok(())
}

pub(super) fn validate_eq_message<E: FieldCore>(
    poly: &EqFactoredUniPoly<E>,
    degree_bound: usize,
) -> Result<(), AkitaError> {
    let expected = EqFactoredUniPoly::<E>::stored_coeff_count_for_degree(degree_bound);
    let actual = poly.coeffs_except_linear_term.len();
    if actual != expected {
        return Err(AkitaError::InvalidSize { expected, actual });
    }
    if poly.degree() > degree_bound {
        return Err(invalid(
            "eq-factored round message exceeds its degree bound",
        ));
    }
    Ok(())
}

pub(super) fn mul_pow_2<E: FieldCore>(mut value: E, exponent: usize) -> E {
    for _ in 0..exponent {
        value = value + value;
    }
    value
}

pub(super) fn check_addressable_count(count: usize, name: &str) -> Result<(), AkitaError> {
    if count > isize::MAX as usize {
        return Err(invalid(&format!("{name} overflows addressable allocation")));
    }
    Ok(())
}

pub(super) fn invalid(message: &str) -> AkitaError {
    AkitaError::InvalidInput(message.to_owned())
}
