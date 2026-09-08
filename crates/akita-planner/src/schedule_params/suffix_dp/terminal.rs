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
    route_guide: Option<CandidateInnerRoute>,
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
        route_guide,
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
    route_guide: Option<CandidateInnerRoute>,
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
    if !input_witness_len.is_multiple_of(terminal_lp.d_a()) {
        return Err(AkitaError::InvalidSetup(
            "terminal-direct input length must be divisible by its A-ring dimension".to_string(),
        ));
    }
    if opening_layout.is_some() || num_polynomials != 1 || terminal_lp.has_preceding_groups() {
        return Err(AkitaError::InvalidSetup(
            "terminal direct response must be a scalar flat fold".to_string(),
        ));
    }
    let (mut terminal_params, certified_linf_cap) =
        akita_types::TerminalFoldParams::try_from_expanded_group(terminal_lp.clone())?;
    let mut sparse_challenge_config = terminal_lp.fold_challenge_config();
    if route_guide != Some(CandidateInnerRoute::Linf) {
        if let Some(l2_challenge) =
            akita_challenges::selective_l2_challenge_config(terminal_params.d_a())
        {
            let fold_basis = 1usize
                .checked_shl(terminal_lp.open().digits.log_basis)
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
                    fold_digit_count: terminal_lp.num_digits_fold(),
                    fold_challenge_config: &l2_challenge,
                    response_l2_sq_cap,
                    norm_proof_shape: Some(akita_types::PhysicalL2NormProofShape::Direct {
                        physical_response_len: terminal_params
                            .inner_width()
                            .checked_mul(terminal_params.d_a())
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup(
                                    "terminal L2 response length overflow".into(),
                                )
                            })?,
                    }),
                },
            )? {
                if route_guide == Some(CandidateInnerRoute::L2)
                    || l2_matrix.output_rank() < terminal_params.inner.matrix.output_rank()
                {
                    terminal_params.inner.matrix = l2_matrix;
                    sparse_challenge_config = l2_challenge;
                }
            }
        }
    }
    let num_fold_coeffs = terminal_params
        .inner_width()
        .checked_mul(terminal_params.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("terminal response length overflow".into()))?;
    let modeled_encoding_scale = source_moment.and_then(|moment| {
        moment.response_linf_cap(
            sparse_challenge_config.challenge_l2_sq_max(),
            terminal_params.blocks.live_blocks,
            1,
            num_fold_coeffs,
            terminal_params.d_a(),
        )
    });
    // For an L2 terminal this scale chooses only the Golomb parameters and
    // byte budget. It is not emitted or enforced as a coefficient cap.
    let encoding_scale = modeled_encoding_scale
        .map(|cap| {
            if terminal_params.response_l2_sq_cap().is_some() {
                cap
            } else {
                cap.min(certified_linf_cap)
            }
        })
        .unwrap_or(certified_linf_cap);
    let witness_shape = TerminalResponseShape::derive(&terminal_params, encoding_scale)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_rejects_an_input_not_divisible_by_its_a_ring() {
        let policy = akita_config::policy_of::<akita_config::proof_optimized::fp128::Dense>();
        let challenge = akita_challenges::SparseChallengeConfig::production_for_ring_dim(256)
            .expect("D256 challenge");
        let params = CommittedGroupParams::params_only(
            akita_types::SisModulusProfileId::Q128OffsetA7F7,
            256,
            2,
            2,
            2,
            2,
            challenge,
        );
        let error = terminal_direct_suffix_cost(
            &policy,
            257,
            &params,
            128,
            PolynomialGroupLayout::new(8, 1),
            1,
            None,
            None,
            None,
        )
        .expect_err("nondivisible terminal input");
        assert!(
            matches!(error, AkitaError::InvalidSetup(message) if message.contains("divisible"))
        );
    }

    #[test]
    fn terminal_route_guide_preserves_either_feasible_route() {
        use crate::schedule_params::PlannerOpeningCandidate;
        use akita_config::{
            policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
        };
        use akita_types::InnerCommitSecurityRoute;

        type Recursive = RecursiveCommitmentConfig<OneHot>;
        let policy = policy_of::<Recursive>();
        let source_moment = crate::response_model::SourceMomentEstimate::new(1_000_000);
        let request = RecursiveCandidateRequest {
            policy: &policy,
            payload_mode: akita_types::CommitmentPayloadMode::Compressed,
            opening: PlannerOpeningCandidate::evaluation_trace(
                Recursive::ring_challenge_config(64).expect("challenge config"),
            ),
            dimensions: CommitmentRingDims::uniform(64),
            current_witness_len: 948_672,
            source: crate::InnerBasisSource::BalancedDigits { log_basis: 4 },
            log_basis_inner: 4,
            log_basis_open: 4,
            fold_level: 3,
            source_moment,
            relation_traversal_order: RelationTraversalOrder::Canonical,
            guide: None,
        };
        let candidate = derive_terminal_candidates(request)
            .expect("terminal candidates")
            .into_iter()
            .find(|candidate| {
                terminal_direct_suffix_cost(
                    &policy,
                    request.current_witness_len,
                    candidate,
                    policy.decomposition.field_bits(),
                    PolynomialGroupLayout::singleton(1),
                    request.fold_level,
                    None,
                    source_moment,
                    None,
                )
                .is_ok_and(|(terminal, _)| {
                    matches!(
                        terminal.params.inner.matrix.security_route(),
                        InnerCommitSecurityRoute::L2 { .. }
                    )
                })
            })
            .expect("fixture where unguided terminal pricing prefers L2");

        let (guided, _) = terminal_direct_suffix_cost(
            &policy,
            request.current_witness_len,
            &candidate,
            policy.decomposition.field_bits(),
            PolynomialGroupLayout::singleton(1),
            request.fold_level,
            None,
            source_moment,
            Some(CandidateInnerRoute::Linf),
        )
        .expect("the frozen Linf terminal remains feasible");
        assert!(matches!(
            guided.params.inner.matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        ));
        assert_eq!(
            guided.sparse_challenge_config,
            candidate.fold_challenge_config()
        );

        let non_improving_l2 = derive_terminal_candidates(request)
            .expect("terminal candidates")
            .into_iter()
            .find_map(|candidate| {
                let (unguided, _) = terminal_direct_suffix_cost(
                    &policy,
                    request.current_witness_len,
                    &candidate,
                    policy.decomposition.field_bits(),
                    PolynomialGroupLayout::singleton(1),
                    request.fold_level,
                    None,
                    source_moment,
                    None,
                )
                .ok()?;
                let (guided, _) = terminal_direct_suffix_cost(
                    &policy,
                    request.current_witness_len,
                    &candidate,
                    policy.decomposition.field_bits(),
                    PolynomialGroupLayout::singleton(1),
                    request.fold_level,
                    None,
                    source_moment,
                    Some(CandidateInnerRoute::L2),
                )
                .ok()?;
                (matches!(
                    unguided.params.inner.matrix.security_route(),
                    InnerCommitSecurityRoute::Linf(_)
                ) && matches!(
                    guided.params.inner.matrix.security_route(),
                    InnerCommitSecurityRoute::L2 { .. }
                ))
                .then_some(guided)
            })
            .expect("fixture where a feasible L2 route is not the greedy winner");
        assert!(matches!(
            non_improving_l2.params.inner.matrix.security_route(),
            InnerCommitSecurityRoute::L2 { .. }
        ));
    }
}
