use super::*;

pub(super) fn attach_source_moments(
    ctx: &SuffixCtx<'_>,
    state: SuffixState,
    is_root_level: bool,
    current_opening_layout: &OpeningClaimsLayout,
    candidates: Vec<candidates::RawFoldCandidate>,
) -> Result<Vec<PlannedFoldCandidate>, AkitaError> {
    let policy = ctx.policy;
    let incoming_setup_prefix = state.topology.incoming_setup_prefix();
    let mut candidates_with_source = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let candidates::RawFoldCandidate {
            params,
            next_witness_len,
            opening_reduction_bytes,
        } = candidate;
        let next_source_moment = if policy.selective_l2_response_model_enabled() {
            let source_groups = if is_root_level {
                crate::response_model::root_group_source_moments(
                    &params,
                    current_opening_layout,
                    ctx.root_honest_fold_policy.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "root batch is missing its response source policy".into(),
                        )
                    })?,
                    ctx.precommitted_honest_fold_policies,
                    policy.decomposition,
                )?
            } else if let Some(natural_prefix_len) = incoming_setup_prefix {
                let prefix_params = params.group_params(current_opening_layout, 0)?;
                let prefix_moment = crate::response_model::uniform_field_source_moment(
                    natural_prefix_len,
                    policy.decomposition.field_bits(),
                    prefix_params.log_basis_inner(),
                    prefix_params.num_digits_inner(),
                )?;
                vec![
                    prefix_moment,
                    state.source_moment.ok_or_else(|| {
                        AkitaError::InvalidSetup("recursive response source is missing".into())
                    })?,
                ]
            } else {
                vec![state.source_moment.ok_or_else(|| {
                    AkitaError::InvalidSetup("recursive response source is missing".into())
                })?]
            };
            Some(crate::response_model::next_source_moment(
                &params,
                current_opening_layout,
                &source_groups,
                policy.decomposition.field_bits(),
                policy.claim_ext_degree,
            )?)
        } else {
            None
        };
        candidates_with_source.push(PlannedFoldCandidate {
            params,
            next_witness_len,
            opening_reduction_bytes,
            next_source_moment,
        });
    }
    Ok(candidates_with_source)
}
