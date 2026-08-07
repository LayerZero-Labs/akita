use akita_field::AkitaError;
use akita_types::{
    terminal_response_bytes, CommittedGroupParams, OpeningClaimsLayout, PolynomialGroupLayout,
    TerminalResponseShape,
};

use crate::schedule_params::CandidateTerminalResponse;

/// Price a terminal response, returning `None` when the final fold is multi-chunk
/// or its fixed inner matrix cannot admit the response.
pub(crate) fn try_direct_cost(
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<Option<(CandidateTerminalResponse, usize)>, AkitaError> {
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Ok(None);
    }
    match direct_cost(
        input_witness_len,
        terminal_lp,
        field_bits,
        key,
        terminal_fold_level,
        opening_layout,
    ) {
        Ok(candidate) => Ok(Some(candidate)),
        Err(AkitaError::InvalidSetup(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn direct_cost(
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
) -> Result<(CandidateTerminalResponse, usize), AkitaError> {
    let num_polynomials = if terminal_fold_level == 0 {
        key.num_polynomials()
    } else {
        1
    };
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Err(AkitaError::InvalidSetup(
            "terminal-direct witness does not support a multi-chunk last fold level".to_string(),
        ));
    }
    if opening_layout.is_some() || num_polynomials != 1 || terminal_lp.has_precommitted_groups() {
        return Err(AkitaError::InvalidSetup(
            "terminal direct response must be a scalar flat fold".to_string(),
        ));
    }
    let (terminal_params, admission_cap) =
        akita_types::TerminalCommittedGroupParams::try_from_expanded_group(terminal_lp.clone())?;
    let witness_shape = TerminalResponseShape::derive(&terminal_params, admission_cap)?;
    let terminal_bytes = terminal_response_bytes(field_bits, &witness_shape);
    let direct = CandidateTerminalResponse {
        params: terminal_params,
        sparse_challenge_config: terminal_lp.fold_challenge_config,
        input_witness_len,
        estimated_direct_payload_bytes: 0,
        response_shape: witness_shape,
        estimated_payload_bytes: terminal_bytes,
    };
    Ok((direct, terminal_bytes))
}
