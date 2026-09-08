use super::*;

/// Inputs shared by every split considered for one recursive level.
pub(super) struct RecursiveLevelSearch {
    pub(super) num_chunks: usize,
    pub(super) num_ring_elems: usize,
    pub(super) reduced_vars: usize,
    pub(super) current_witness_len: usize,
    pub(super) opening_layout: OpeningClaimsLayout,
    pub(super) setup_prefixes: Vec<Option<akita_types::GroupOpenPhaseParams>>,
}

pub(super) fn prepare_recursive_level_search(
    request: &RecursiveCandidateRequest<'_>,
    setup_prefix: RecursiveSetupPrefix<'_>,
) -> Result<Option<RecursiveLevelSearch>, AkitaError> {
    let RecursiveCandidateRequest {
        policy,
        opening,
        dimensions,
        current_witness_len,
        log_basis_open,
        fold_level,
        ..
    } = *request;
    let num_chunks = policy.chunks_at_level(fold_level);
    dimensions.validate_role_projection()?;
    opening.validate_for(fold_level, policy.claim_ext_degree, dimensions)?;
    let d_a = dimensions.d_a();
    if current_witness_len == 0 {
        return Ok(None);
    }
    // The previous fold owns a compact field-coefficient buffer. It need not
    // end on the next A-ring boundary; commitment alignment pads only the
    // transient ring view. Plan from the live coefficient count, rounding up
    // solely to determine the next fold's block geometry.
    let num_ring_elems = current_witness_len.div_ceil(d_a);
    let reduced_vars = num_ring_elems
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("recursive witness capacity overflow".to_string()))?
        .max(1)
        .trailing_zeros() as usize;

    if reduced_vars <= 2 || reduced_vars >= 53 {
        return Err(AkitaError::InvalidSetup(format!(
            "recursive fold candidate reduced_vars={reduced_vars} is outside \
             the optimizable range [3, 52]"
        )));
    }

    let incoming_setup_prefix = match &setup_prefix {
        RecursiveSetupPrefix::None => None,
        RecursiveSetupPrefix::Search { natural_len, .. } => Some(*natural_len),
    };
    let opening_layout = suffix_opening_layout(current_witness_len, incoming_setup_prefix)?;
    let setup_prefixes = match setup_prefix {
        RecursiveSetupPrefix::Search { cache, natural_len } => {
            let n_prefix = padded_setup_prefix_len(natural_len);
            let groups = derive_setup_prefix_groups(
                cache,
                SetupPrefixSearchRequest {
                    policy,
                    opening,
                    log_basis_open,
                    n_prefix,
                    num_chunks,
                    inner_ring_dimension: d_a,
                    outer_ring_dimension: dimensions.d_b(),
                    guide: request.guide.and_then(|guide| guide.setup_prefix),
                },
            )?;
            if groups.is_empty() {
                return Ok(None);
            }
            groups
                .into_iter()
                .map(|group| Some(akita_types::scheduled_setup_prefix(natural_len, group)))
                .collect()
        }
        RecursiveSetupPrefix::None => vec![None],
    };
    Ok(Some(RecursiveLevelSearch {
        num_chunks,
        num_ring_elems,
        reduced_vars,
        current_witness_len,
        opening_layout,
        setup_prefixes,
    }))
}

pub(super) fn attach_recursive_setup_prefix(
    setup_prefix: Option<&akita_types::GroupOpenPhaseParams>,
    extension_degree: usize,
    mut candidate_params: CommittedGroupParams,
) -> Result<CommittedGroupParams, AkitaError> {
    candidate_params.set_setup_prefix(setup_prefix.cloned())?;
    if let Some(prefix) = &candidate_params.setup_prefix() {
        let prefix_d_width =
            prefix.d_segment_width(extension_degree, candidate_params.role_dims().d_d())?;
        let total_d_width = candidate_params
            .open()
            .matrix
            .input_width()
            .checked_add(prefix_d_width)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("setup-prefix shared D width overflow".to_string())
            })?;
        candidate_params.open_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
            candidate_params.open().matrix.sis_table_key(),
            total_d_width,
        )?;
    }
    Ok(candidate_params)
}

pub(super) fn finalize_recursive_level_candidate(
    policy: &PlannerPolicy,
    search: &RecursiveLevelSearch,
    candidate_params: CommittedGroupParams,
) -> Result<Option<(LayoutCandidateScore, CommittedGroupParams, usize)>, AkitaError> {
    let Some(next_witness_len) = planned_next_witness_len(
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        &candidate_params,
        1,
        search.num_chunks,
    )?
    else {
        return Ok(None);
    };
    let score = layout_candidate_score(
        next_witness_len,
        candidate_params.blocks().live_blocks,
        search.num_chunks,
    )?;
    Ok(Some((score, candidate_params, next_witness_len)))
}
