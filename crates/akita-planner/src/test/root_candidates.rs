use super::*;

/// Enumerate every root split and slice for the single-group oracle fixture.
/// Candidate materialization stays canonical, while this reference domain is
/// independent of production split bounds and local slice pruning.
#[allow(clippy::too_many_arguments)]
pub(crate) fn exhaustive_root_candidates_for_reference(
    key: &AkitaScheduleLookupKey,
    final_honest_fold_policy: HonestFoldPolicySpec,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
    policy: &PlannerPolicy,
    dimensions: CommitmentRingDims,
    opening: PlannerOpeningCandidate,
    precommitted_openings: &[PlannerOpeningCandidate],
    candidate_log_basis_inner: u32,
    candidate_log_basis_open: u32,
) -> Result<Vec<(CommittedGroupParams, usize)>, AkitaError> {
    key.validate(policy.decomposition.field_bits())?;
    dimensions.validate_role_projection()?;
    opening.validate_for(0, policy.claim_ext_degree, dimensions)?;
    if !key.precommitteds.is_empty()
        || !precommitted_honest_fold_policies.is_empty()
        || !precommitted_openings.is_empty()
    {
        return Err(AkitaError::InvalidSetup(
            "the exhaustive root reference supports one uncommitted group".into(),
        ));
    }
    let alpha = dimensions.d_a().trailing_zeros() as usize;
    let reduced_vars = key.final_group.num_vars().saturating_sub(alpha);
    if reduced_vars == 0 {
        return Ok(Vec::new());
    }
    let candidate_ctx = MultiGroupRootCandidateCtx {
        policy,
        dimensions,
        opening,
        final_honest_fold_policy,
        final_num_vars: key.final_group.num_vars(),
        main_num_polys: key.final_group.num_polynomials(),
        source: crate::schedule_params::root_inner_basis_source(
            final_honest_fold_policy,
            policy.decomposition.log_commit_bound,
        ),
    };
    let opening_batch = key.opening_layout()?;
    let min_split = usize::from(reduced_vars >= 3);
    let max_split = (reduced_vars - 1).min(usize::BITS as usize - 1);
    let mut candidates = Vec::new();
    for block_index_bits in (min_split..=max_split).rev() {
        let position_index_bits = reduced_vars - block_index_bits;
        let num_live_blocks = 1usize << block_index_bits;
        for outer_slice_count in akita_types::CommitmentSliceCount::ALL {
            if outer_slice_count
                .validate_for_commitment(
                    0,
                    akita_types::CommitmentPayloadMode::Compressed,
                    num_live_blocks,
                )
                .is_err()
            {
                continue;
            }
            let Some(mut params) = root_final_group_level_params_candidate(
                &candidate_ctx,
                RootFinalGroupCandidateInput {
                    log_basis_inner: candidate_log_basis_inner,
                    log_basis_open: candidate_log_basis_open,
                    position_index_bits,
                    block_index_bits,
                    outer_slice_count,
                    precommitted_groups: &[],
                    precommitted_d_width: 0,
                },
            )?
            else {
                continue;
            };
            params.witness_chunk = crate::policy::witness_chunk_at_level(policy, 0);
            let Some(output_witness_len) = root_batch_next_w_len(
                policy.decomposition.field_bits(),
                policy.claim_ext_degree,
                &params,
                &opening_batch,
            )?
            else {
                continue;
            };
            candidates.push((params, output_witness_len));
        }
    }
    Ok(candidates)
}
