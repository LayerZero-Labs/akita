use super::*;

/// Like [`terminal_direct_suffix_cost`], but returns `None` when the fold at
/// `terminal_fold_level` is multi-chunk. The suffix DP uses this to skip the
/// fold-then-direct branch without aborting fold-then-fold exploration.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_terminal_direct_suffix_cost(
    policy: &PlannerPolicy,
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
) -> Result<Option<(CandidateTerminalResponse, usize)>, AkitaError> {
    if terminal_lp.witness_chunk.num_chunks > 1 {
        return Ok(None);
    }
    let result = terminal_direct_suffix_cost(
        policy,
        input_witness_len,
        terminal_lp,
        field_bits,
        key,
        terminal_fold_level,
        opening_layout,
        source_moment,
    );
    match result {
        Ok(candidate) => Ok(Some(candidate)),
        // Candidate construction is an optimization search. A geometry whose
        // fixed inner matrix cannot admit the directly checked terminal response is
        // infeasible, not a fatal planner error.
        Err(AkitaError::InvalidSetup(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn terminal_direct_suffix_cost(
    policy: &PlannerPolicy,
    input_witness_len: usize,
    terminal_lp: &CommittedGroupParams,
    field_bits: u32,
    key: PolynomialGroupLayout,
    terminal_fold_level: usize,
    opening_layout: Option<&OpeningClaimsLayout>,
    source_moment: Option<crate::response_model::SourceMomentEstimate>,
) -> Result<(CandidateTerminalResponse, usize), AkitaError> {
    // Scalar same-point root fold: polynomial count at the root, 1 recursively.
    let num_polynomials = if terminal_fold_level == 0 {
        key.num_polynomials()
    } else {
        1
    };
    // The terminal-direct (cleartext) witness is single-chunk by construction:
    // the prover emits the global folded response and one shared `r̂` tail, so
    // chunking the cleartext tail is unsupported. The last fold level must be
    // single-chunk (only the leading activated levels are chunked). Reject here
    // to match `resolve.rs` and avoid a cryptic prover-side layout mismatch.
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
    let (mut terminal_params, certified_admission_cap) =
        akita_types::TerminalCommittedGroupParams::try_from_expanded_group(terminal_lp.clone())?;
    let mut sparse_challenge_config = terminal_lp.fold_challenge_config;
    if let Some(l2_challenge) =
        akita_challenges::selective_l2_challenge_config(terminal_params.d_a())
    {
        let fold_basis = 1usize
            .checked_shl(terminal_lp.log_basis_open)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal L2 basis overflow".into()))?;
        let response_l2_sq_cap = source_moment
            .and_then(|moment| moment.response_l2_sq_cap(l2_challenge.challenge_l2_sq_max()));
        if let Some(l2_matrix) = akita_schedules::planner_support::selective_l2_inner_matrix(
            policy,
            akita_schedules::planner_support::SelectiveL2CandidateGeometry {
                fold_level: terminal_fold_level,
                num_claims: 1,
                num_chunks: 1,
                inner_width: terminal_params.inner_width(),
                ring_dimension: terminal_params.d_a(),
                fold_basis,
                fold_digit_count: terminal_lp.num_digits_fold,
                fold_challenge_config: &l2_challenge,
                response_l2_sq_cap,
                norm_proof_shape: Some(akita_types::PhysicalL2NormProofShape::Direct {
                    physical_response_len: terminal_params
                        .inner_width()
                        .checked_mul(terminal_params.d_a())
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("terminal L2 response length overflow".into())
                        })?,
                }),
            },
        )? {
            if l2_matrix.output_rank() < terminal_params.inner_commit_matrix.output_rank() {
                terminal_params.inner_commit_matrix = l2_matrix;
                sparse_challenge_config = l2_challenge;
            }
        }
    }
    let num_fold_coeffs = terminal_params
        .inner_width()
        .checked_mul(terminal_params.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("terminal response length overflow".into()))?;
    let modeled_admission_cap = source_moment.and_then(|moment| {
        moment.response_linf_cap(
            sparse_challenge_config.challenge_l2_sq_max(),
            terminal_params.num_live_blocks,
            1,
            num_fold_coeffs,
            terminal_params.d_a(),
        )
    });
    let admission_cap = modeled_admission_cap
        .map(|cap| cap.min(certified_admission_cap))
        .unwrap_or(certified_admission_cap);
    let witness_shape = TerminalResponseShape::derive(&terminal_params, admission_cap)?;
    let estimated_terminal_bytes = terminal_response_planner_bytes(
        field_bits,
        &witness_shape,
        terminal_params.response_l2_sq_cap(),
    );
    let direct = CandidateTerminalResponse {
        params: terminal_params,
        sparse_challenge_config,
        input_witness_len,
        estimated_direct_payload_bytes: 0,
        response_shape: witness_shape,
        estimated_payload_bytes: estimated_terminal_bytes,
    };
    Ok((direct, estimated_terminal_bytes))
}
