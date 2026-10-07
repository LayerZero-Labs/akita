use super::*;

pub(super) fn attach_source_moments(
    ctx: &SuffixCtx<'_>,
    state: SuffixState<'_>,
    is_root_level: bool,
    candidates: Vec<candidates::RawFoldCandidate>,
) -> Result<Vec<PlannedFoldCandidate>, AkitaError> {
    let policy = ctx.policy;
    let incoming_setup_prefix = state.topology.incoming_setup_prefix();
    let mut candidates_with_source = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let candidates::RawFoldCandidate {
            params,
            next_witness_len,
        } = candidate;
        let opening_layout = params.opening_layout_for_final_group(params.group())?;
        let current_opening_layout = &opening_layout;
        let opening_reduction_bytes = if params
            .opening_method()
            .requires_extension_opening_reduction(policy.claim_ext_degree)
        {
            akita_params::extension_opening_reduction_level_bytes(
                policy.challenge_field_bits()?,
                policy.claim_ext_degree,
                opening_layout.aggregate_polynomial_group_layout()?,
            )?
        } else {
            0
        };
        let chunk_shape = if params.witness_chunk.num_chunks > 1 {
            Some(
                akita_params::WitnessLayout::new(
                    &params,
                    &opening_layout,
                    &akita_params::RelationWitnessGeometry::for_level(
                        &params,
                        &opening_layout,
                        policy.claim_ext_degree,
                    )?,
                    params.witness_chunk.num_chunks,
                    akita_params::RelationQuotientPlan::for_field_bits(
                        &params,
                        policy.decomposition.field_bits(),
                    )?,
                )?
                .chunk_shape()?,
            )
        } else {
            None
        };
        let next_source_moment = if policy.selective_l2_response_model_enabled() {
            let source_groups = if is_root_level {
                crate::response_model::root_group_source_moments(
                    &params,
                    current_opening_layout,
                    ctx.root_source_contract.ok_or_else(|| {
                        AkitaError::InvalidSetup("root batch is missing its source contract".into())
                    })?,
                    ctx.precommitted_source_contracts,
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
            chunk_shape,
            params,
            next_witness_len,
            opening_reduction_bytes,
            next_source_moment,
        });
    }
    Ok(candidates_with_source)
}
