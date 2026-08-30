use super::*;

pub(super) fn terminal(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    opening_reduction_bytes: usize,
    params: &CommittedGroupParams,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    let Some((mut terminal, terminal_bytes)) = suffix_dp::try_terminal_direct_suffix_cost(
        ctx.policy,
        state.input_witness_len,
        params,
        ctx.policy.decomposition.field_bits(),
        ctx.key,
        state.level,
        None,
        state.source_moment,
    )?
    else {
        return Ok(None);
    };
    let payload_bytes = opening_reduction_bytes
        .checked_add(terminal_bytes)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned traversal terminal proof size overflow".into())
        })?;
    terminal.estimated_direct_payload_bytes = opening_reduction_bytes;
    Ok(Some(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_types::active_setup_field_len(
                params,
                &suffix_opening_layout(state.input_witness_len, None)?,
            )?,
        ),
        cost: PackedProofCost::new(payload_bytes, 0)?,
        setup_field_elements: terminal_setup_field_elements(&terminal.params)?,
        folds: CandidateFoldChain::default(),
        terminal: Arc::new(terminal),
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepend_fold(
    policy: &PlannerPolicy,
    level: usize,
    field_bits: u32,
    challenge_field_bits: u32,
    input_witness_len: usize,
    output_witness_len: usize,
    opening_reduction_bytes: usize,
    params: &CommittedGroupParams,
    child: &ScheduleCandidate,
) -> Result<ScheduleCandidate, AkitaError> {
    let opening_layout = suffix_opening_layout(input_witness_len, None)?;
    let binding = if child.folds.is_empty() {
        akita_types::NextWitnessBindingPolicy::TerminalInnerState
    } else {
        akita_types::NextWitnessBindingPolicy::OuterPayload
    };
    let direct_bytes = level_proof_bytes(
        field_bits,
        challenge_field_bits,
        params,
        child.first_fold_params(),
        output_witness_len,
        Some(binding),
    )?
    .checked_add(opening_reduction_bytes)
    .ok_or_else(|| {
        AkitaError::InvalidSetup("unpruned traversal fold proof size overflow".into())
    })?;
    let successor = child.folds.first().map_or_else(
        || akita_types::GrindingPlanSuccessor::Terminal(&child.terminal.params),
        |fold| akita_types::GrindingPlanSuccessor::Recursive(fold.params.as_ref()),
    );
    let edge_nonce_bits = akita_types::transcript_grinding_nonce_bits_for_planner_edge(
        params,
        output_witness_len,
        &opening_layout,
        successor,
        field_bits,
        policy.claim_ext_degree,
        u32::try_from(level)
            .map_err(|_| AkitaError::InvalidSetup("unpruned fold level exceeds u32".into()))?,
    )?;
    Ok(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_types::active_setup_field_len(params, &opening_layout)?,
        ),
        cost: child.cost.checked_prepend(direct_bytes, edge_nonce_bits)?,
        setup_field_elements: level_setup_field_elements(params)?.max(child.setup_field_elements),
        folds: child.folds.prepend(CandidateFoldStep {
            params: Arc::new(params.clone()),
            input_witness_len,
            output_witness_len,
            estimated_direct_payload_bytes: direct_bytes,
            estimated_stage3_payload_bytes: 0,
        }),
        terminal: Arc::clone(&child.terminal),
    })
}

pub(super) fn prepend_root(
    policy: &PlannerPolicy,
    schedule_key: &akita_types::AkitaScheduleLookupKey,
    input_witness_len: usize,
    root_params: &CommittedGroupParams,
    output_witness_len: usize,
    suffix: &ScheduleCandidate,
) -> Result<ScheduleCandidate, AkitaError> {
    let opening_layout = schedule_key.opening_layout()?;
    let first_direct_setup_field_len =
        std::num::NonZeroUsize::new(active_setup_field_len(root_params, &opening_layout)?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("unpruned root setup field length must be nonzero".into())
            })?;
    let child_is_terminal = suffix.folds.is_empty();
    let successor = suffix.folds.first().map_or_else(
        || akita_types::GrindingPlanSuccessor::Terminal(&suffix.terminal.params),
        |fold| akita_types::GrindingPlanSuccessor::Recursive(fold.params.as_ref()),
    );
    let root_bytes = level_proof_bytes(
        policy.decomposition.field_bits(),
        policy.challenge_field_bits()?,
        root_params,
        suffix.first_fold_params(),
        output_witness_len,
        Some(if child_is_terminal {
            akita_types::NextWitnessBindingPolicy::TerminalInnerState
        } else {
            akita_types::NextWitnessBindingPolicy::OuterPayload
        }),
    )?;
    let root_nonce_bits = akita_types::transcript_grinding_nonce_bits_for_planner_edge(
        root_params,
        output_witness_len,
        &opening_layout,
        successor,
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        0,
    )?;
    let candidate = ScheduleCandidate {
        first_direct_setup_field_len: Some(first_direct_setup_field_len),
        cost: suffix.cost.checked_prepend(root_bytes, root_nonce_bits)?,
        setup_field_elements: level_setup_field_elements(root_params)?
            .max(suffix.setup_field_elements),
        folds: suffix.folds.prepend(CandidateFoldStep {
            params: Arc::new(root_params.clone()),
            input_witness_len,
            output_witness_len,
            estimated_direct_payload_bytes: root_bytes,
            estimated_stage3_payload_bytes: 0,
        }),
        terminal: Arc::clone(&suffix.terminal),
    };
    let canonical_nonce_bits = akita_schedules::planner_support::candidate_grinding_nonce_bits(
        policy,
        &opening_layout,
        &candidate.folds.to_vec(),
        candidate.terminal.as_ref(),
    )?;
    if candidate.cost.nonce_bits() != canonical_nonce_bits {
        return Err(AkitaError::InvalidSetup(
            "edge-wise oracle grinding cost disagrees with the canonical complete schedule".into(),
        ));
    }
    Ok(candidate)
}
