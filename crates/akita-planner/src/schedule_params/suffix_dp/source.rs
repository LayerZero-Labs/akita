use super::*;

pub(super) fn attach_source_moments(
    ctx: &SuffixCtx<'_>,
    state: SuffixState<'_>,
    is_root_level: bool,
    candidates: Vec<candidates::RawFoldCandidate>,
) -> Result<Vec<PlannedFoldCandidate>, AkitaError> {
    let policy = ctx.policy;
    let incoming_setup_prefix = state.topology.incoming_setup_prefix();
    // Each candidate's moment is a function of the candidate and the state;
    // they are computed in parallel and kept in candidate order. A fold that
    // cannot be priced yields `None` and is dropped.
    let attach = |candidate: candidates::RawFoldCandidate|
     -> Result<Option<PlannedFoldCandidate>, AkitaError> {
        let candidates::RawFoldCandidate {
            params,
            next_witness_len,
        } = candidate;
        let opening_layout = params.opening_layout_for_final_group(params.group())?;
        // A fold that cannot be priced, such as one whose challenge message
        // exceeds the message limit, can never be planned.
        let fold_cost = match akita_params::FoldChallengeCost::new(&params, &opening_layout) {
            Ok(cost) => cost,
            Err(AkitaError::InvalidSetup(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
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
        Ok(Some(PlannedFoldCandidate {
            chunk_shape,
            params,
            fold_cost,
            next_witness_len,
            opening_reduction_bytes,
            next_source_moment,
        }))
    };
    let attached = crate::schedule_params::map_in_order(candidates, attach)
        .into_iter()
        .collect::<Result<Vec<_>, AkitaError>>()?;
    Ok(attached.into_iter().flatten().collect())
}
