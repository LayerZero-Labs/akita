use super::*;

fn reference_setup_field_elements(params: &CommittedGroupParams) -> Result<usize, AkitaError> {
    let mut elements = akita_types::SetupMatrixCapacity::minimum().num_field_elements;
    akita_types::accumulate_matrix_field_elements_for_level(params, &mut elements)?;
    Ok(elements)
}

fn reference_terminal_setup_field_elements(
    params: &akita_types::TerminalFoldParams,
) -> Result<usize, AkitaError> {
    let mut elements = akita_types::SetupMatrixCapacity::minimum().num_field_elements;
    akita_types::accumulate_terminal_matrix_field_elements(params, &mut elements)?;
    Ok(elements)
}

fn reference_payload_bytes(
    field_bits: u32,
    profile: akita_types::SisModulusProfileId,
    geometry: akita_types::CommitmentPayloadGeometry,
) -> Result<usize, AkitaError> {
    if profile.field_bits() != field_bits {
        return Err(AkitaError::InvalidSetup(
            "reference payload profile disagrees with field width".into(),
        ));
    }
    Ok(akita_types::proof_ring_vec_bytes(
        geometry.transmitted_rows()?,
        geometry.transcript_ring_dimension(),
        akita_types::field_bytes(field_bits),
    ))
}

fn reference_level_proof_bytes(
    base_field_bits: u32,
    challenge_field_bits: u32,
    params: &CommittedGroupParams,
    successor: Option<&CommittedGroupParams>,
    output_witness_len: usize,
) -> Result<usize, AkitaError> {
    let challenge_bytes = akita_types::field_bytes(challenge_field_bits);
    let rounds = akita_types::sumcheck_rounds(params.d_a(), output_witness_len);
    let stage2_bytes = rounds
        .checked_mul(3)
        .and_then(|count| count.checked_mul(challenge_bytes))
        .ok_or_else(|| AkitaError::InvalidSetup("reference Stage-2 size overflow".into()))?;
    let range_plan = akita_types::DigitRangePlan::new(1usize << params.open().digits.log_basis)?;
    let (range_stages, norm) =
        range_plan.proof_shapes_for_route(rounds, params.inner().matrix.security_route())?;
    let mut stage1_bytes = challenge_bytes;
    for stage in range_stages {
        stage1_bytes = stage1_bytes
            .checked_add(
                stage
                    .sumcheck_proof
                    .0
                    .checked_mul(stage.sumcheck_proof.1)
                    .and_then(|count| count.checked_mul(challenge_bytes))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("reference Stage-1 proof size overflow".into())
                    })?,
            )
            .and_then(|bytes| {
                stage
                    .child_claims
                    .checked_mul(challenge_bytes)
                    .and_then(|claims| bytes.checked_add(claims))
            })
            .ok_or_else(|| AkitaError::InvalidSetup("reference Stage-1 size overflow".into()))?;
    }
    if let Some(norm) = norm {
        let norm_scalars = norm
            .subclaims
            .checked_add(norm.virtual_evaluations)
            .and_then(|count| count.checked_add(norm.sumcheck.into_iter().sum::<usize>()))
            .ok_or_else(|| AkitaError::InvalidSetup("reference norm size overflow".into()))?;
        stage1_bytes = stage1_bytes
            .checked_add(16)
            .and_then(|bytes| {
                norm_scalars
                    .checked_mul(challenge_bytes)
                    .and_then(|norm_bytes| bytes.checked_add(norm_bytes))
            })
            .ok_or_else(|| AkitaError::InvalidSetup("reference norm byte size overflow".into()))?;
    }
    let opening_bytes = reference_payload_bytes(
        base_field_bits,
        params.open().matrix.sis_modulus_profile(),
        params.opening_payload_geometry()?,
    )?;
    let successor_bytes = successor
        .map(|successor| {
            reference_payload_bytes(
                base_field_bits,
                successor.outer().matrix.sis_modulus_profile(),
                successor.outer_payload_geometry()?,
            )
        })
        .transpose()?
        .unwrap_or_default();
    opening_bytes
        .checked_add(stage1_bytes)
        .and_then(|bytes| bytes.checked_add(stage2_bytes))
        .and_then(|bytes| bytes.checked_add(successor_bytes))
        .and_then(|bytes| bytes.checked_add(challenge_bytes))
        .ok_or_else(|| AkitaError::InvalidSetup("reference level proof size overflow".into()))
}

pub(super) fn terminal(
    ctx: &UnprunedCtx<'_>,
    state: UnprunedState,
    opening_reduction_bytes: usize,
    params: &CommittedGroupParams,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    if params.witness_chunk.num_chunks > 1
        || !state.input_witness_len.is_multiple_of(params.d_a())
        || params.has_preceding_groups()
    {
        return Ok(None);
    }
    let (mut terminal_params, certified_linf_cap) =
        match akita_types::TerminalFoldParams::try_from_expanded_group(params.clone()) {
            Ok(result) => result,
            Err(AkitaError::InvalidSetup(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
    let mut sparse_challenge_config = params.fold_challenge_config();
    if let Some(l2_challenge) =
        akita_challenges::selective_l2_challenge_config(terminal_params.d_a())
    {
        let fold_basis = 1usize
            .checked_shl(params.open().digits.log_basis)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("reference terminal L2 basis overflow".into())
            })?;
        let response_l2_sq_cap = state
            .source_moment
            .and_then(|moment| moment.response_l2_sq_cap(l2_challenge.challenge_l2_sq_max()));
        let physical_response_len = terminal_params
            .inner_width()
            .checked_mul(terminal_params.d_a())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("reference terminal L2 response length overflow".into())
            })?;
        if let Some(l2_matrix) = akita_schedules::planner_support::selective_l2_inner_matrix(
            ctx.policy,
            akita_schedules::planner_support::SelectiveL2CandidateGeometry {
                fold_level: state.level,
                num_claims: 1,
                num_chunks: 1,
                inner_width: terminal_params.inner_width(),
                ring_dimension: terminal_params.d_a(),
                fold_basis,
                fold_digit_count: params.num_digits_fold(),
                fold_challenge_config: &l2_challenge,
                response_l2_sq_cap,
                norm_proof_shape: Some(akita_types::PhysicalL2NormProofShape::Direct {
                    physical_response_len,
                }),
            },
        )? {
            if l2_matrix.output_rank() < terminal_params.inner.matrix.output_rank() {
                terminal_params.inner.matrix = l2_matrix;
                sparse_challenge_config = l2_challenge;
            }
        }
    }
    let num_fold_coeffs = terminal_params
        .inner_width()
        .checked_mul(terminal_params.d_a())
        .ok_or_else(|| {
            AkitaError::InvalidSetup("reference terminal response length overflow".into())
        })?;
    let modeled_encoding_scale = state.source_moment.and_then(|moment| {
        moment.response_linf_cap(
            sparse_challenge_config.challenge_l2_sq_max(),
            terminal_params.blocks.live_blocks,
            1,
            num_fold_coeffs,
            terminal_params.d_a(),
        )
    });
    let encoding_scale = modeled_encoding_scale
        .map(|cap| {
            if terminal_params.response_l2_sq_cap().is_some() {
                cap
            } else {
                cap.min(certified_linf_cap)
            }
        })
        .unwrap_or(certified_linf_cap);
    let response_shape =
        akita_types::TerminalResponseShape::derive(&terminal_params, encoding_scale)?;
    let terminal_bytes = akita_types::terminal_response_planner_bytes(
        ctx.policy.decomposition.field_bits(),
        &response_shape,
        terminal_params.response_l2_sq_cap(),
    );
    let payload_bytes = opening_reduction_bytes
        .checked_add(terminal_bytes)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned traversal terminal proof size overflow".into())
        })?;
    Ok(Some(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_types::active_setup_field_len(
                params,
                &suffix_opening_layout(state.input_witness_len, None)?,
            )?,
        ),
        first_direct_output_witness_len: 0,
        cost: PackedProofCost::new(payload_bytes, 0)?,
        setup_field_elements: reference_terminal_setup_field_elements(&terminal_params)?,
        folds: CandidateFoldChain::default(),
        terminal: Arc::new(CandidateTerminalResponse {
            params: terminal_params,
            sparse_challenge_config,
            input_witness_len: state.input_witness_len,
            estimated_direct_payload_bytes: opening_reduction_bytes,
            response_shape,
            estimated_payload_bytes: terminal_bytes,
        }),
    }))
}

pub(super) fn prepend_fold(
    policy: &PlannerPolicy,
    level: usize,
    input_witness_len: usize,
    output_witness_len: usize,
    opening_reduction_bytes: usize,
    params: &CommittedGroupParams,
    child: &ScheduleCandidate,
) -> Result<ScheduleCandidate, AkitaError> {
    let opening_layout = suffix_opening_layout(input_witness_len, None)?;
    let direct_bytes = reference_level_proof_bytes(
        policy.decomposition.field_bits(),
        policy.challenge_field_bits()?,
        params,
        child.first_fold_params(),
        output_witness_len,
    )?
    .checked_add(opening_reduction_bytes)
    .ok_or_else(|| {
        AkitaError::InvalidSetup("unpruned traversal fold proof size overflow".into())
    })?;
    let successor = child.folds.first().map_or_else(
        || akita_types::FoldSuccessor::Terminal(&child.terminal.params),
        |fold| akita_types::FoldSuccessor::Recursive(fold.params.as_ref()),
    );
    let relation_geometry = params.relation_address_geometry(
        &opening_layout,
        policy.claim_ext_degree,
        successor.ring_dimension(),
        output_witness_len,
    )?;
    let edge_nonce_bits = akita_types::transcript_grinding_nonce_bits_for_planner_edge(
        params,
        relation_geometry,
        &opening_layout,
        successor,
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        u32::try_from(level)
            .map_err(|_| AkitaError::InvalidSetup("unpruned fold level exceeds u32".into()))?,
    )?;
    Ok(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_types::active_setup_field_len(params, &opening_layout)?,
        ),
        first_direct_output_witness_len: output_witness_len,
        cost: child.cost.checked_prepend(direct_bytes, edge_nonce_bits)?,
        setup_field_elements: reference_setup_field_elements(params)?
            .max(child.setup_field_elements),
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
    let successor = suffix.folds.first().map_or_else(
        || akita_types::FoldSuccessor::Terminal(&suffix.terminal.params),
        |fold| akita_types::FoldSuccessor::Recursive(fold.params.as_ref()),
    );
    let root_bytes = reference_level_proof_bytes(
        policy.decomposition.field_bits(),
        policy.challenge_field_bits()?,
        root_params,
        suffix.first_fold_params(),
        output_witness_len,
    )?;
    let relation_geometry = root_params.relation_address_geometry(
        &opening_layout,
        policy.claim_ext_degree,
        successor.ring_dimension(),
        output_witness_len,
    )?;
    let root_nonce_bits = akita_types::transcript_grinding_nonce_bits_for_planner_edge(
        root_params,
        relation_geometry,
        &opening_layout,
        successor,
        policy.decomposition.field_bits(),
        policy.claim_ext_degree,
        0,
    )?;
    let candidate = ScheduleCandidate {
        first_direct_setup_field_len: Some(first_direct_setup_field_len),
        first_direct_output_witness_len: output_witness_len,
        cost: suffix.cost.checked_prepend(root_bytes, root_nonce_bits)?,
        setup_field_elements: reference_setup_field_elements(root_params)?
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
