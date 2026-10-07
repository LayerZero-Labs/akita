use super::*;

fn reference_setup_field_elements(params: &CommittedGroupParams) -> Result<usize, AkitaError> {
    let mut elements = akita_params::SetupMatrixCapacity::minimum().num_field_elements;
    akita_params::accumulate_matrix_field_elements_for_level(params, &mut elements)?;
    Ok(elements)
}

fn reference_terminal_setup_field_elements(
    params: &akita_params::TerminalFoldParams,
) -> Result<usize, AkitaError> {
    let mut elements = akita_params::SetupMatrixCapacity::minimum().num_field_elements;
    akita_params::accumulate_terminal_matrix_field_elements(params, &mut elements)?;
    Ok(elements)
}

fn reference_payload_bytes(
    field_bits: u32,
    profile: akita_params::SisModulusProfileId,
    geometry: akita_params::CommitmentPayloadGeometry,
) -> Result<usize, AkitaError> {
    if profile.field_bits() != field_bits {
        return Err(AkitaError::InvalidSetup(
            "reference payload profile disagrees with field width".into(),
        ));
    }
    geometry
        .transmitted_coefficients()
        .checked_mul(akita_params::field_bytes(field_bits))
        .ok_or_else(|| AkitaError::InvalidSetup("reference payload bytes overflow".into()))
}

fn reference_level_proof_bytes(
    base_field_bits: u32,
    challenge_field_bits: u32,
    params: &CommittedGroupParams,
    successor: Option<&CommittedGroupParams>,
    output_witness_len: usize,
) -> Result<usize, AkitaError> {
    let challenge_bytes = akita_params::field_bytes(challenge_field_bits);
    let rounds = akita_params::sumcheck_rounds(params.d_a(), output_witness_len);
    let stage2_bytes = rounds
        .checked_mul(3)
        .and_then(|count| count.checked_mul(challenge_bytes))
        .ok_or_else(|| AkitaError::InvalidSetup("reference Stage-2 size overflow".into()))?;
    let range_plan = akita_params::DigitRangePlan::new(1usize << params.open().digits.log_basis)?;
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
    params: &CommittedGroupParams,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    let opening_reduction_bytes = akita_params::extension_opening_reduction_level_bytes(
        ctx.policy.challenge_field_bits()?,
        ctx.policy.claim_ext_degree,
        params.group(),
    )?;
    let input_witness_len = state
        .input_chunks
        .align(
            akita_params::FoldSuccessor::Recursive(params).source_block_len()?,
            1,
        )?
        .0;
    if params.witness_chunk.num_chunks > 1
        || !input_witness_len.is_multiple_of(params.d_a())
        || params.has_preceding_groups()
    {
        return Ok(None);
    }
    let (mut terminal_params, certified_linf_cap) =
        match akita_params::TerminalFoldParams::try_from_expanded_group(params.clone()) {
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
                norm_proof_shape: Some(akita_params::PhysicalL2NormProofShape::Direct {
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
        akita_params::TerminalResponseShape::derive(&terminal_params, encoding_scale)?;
    let terminal_bytes = akita_params::terminal_response_planner_bytes(
        ctx.policy.decomposition.field_bits(),
        &response_shape,
        terminal_params.response_l2_sq_cap(),
    )?;
    let payload_bytes = opening_reduction_bytes
        .checked_add(terminal_bytes)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("unpruned traversal terminal proof size overflow".into())
        })?;
    Ok(Some(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(
            akita_params::active_setup_field_len(
                params,
                &suffix_opening_layout(input_witness_len, None)?,
            )?,
        ),
        first_direct_output_witness_len: 0,
        cost: ProofCost::new(payload_bytes, 0, 0, 0)?,
        setup_field_elements: reference_terminal_setup_field_elements(&terminal_params)?,
        folds: CandidateFoldChain::default(),
        terminal: Arc::new(CandidateTerminalResponse {
            params: terminal_params,
            sparse_challenge_config,
            input_witness_len,
            estimated_direct_payload_bytes: opening_reduction_bytes,
            response_shape,
            estimated_payload_bytes: terminal_bytes,
        }),
    }))
}

pub(in crate::schedule_params) fn prepend_fold(
    policy: &PlannerPolicy,
    level: usize,
    input_witness_len: usize,
    params: &CommittedGroupParams,
    child: &ScheduleCandidate,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    let output_witness_len = child
        .folds
        .first()
        .map_or(child.terminal.input_witness_len, |fold| {
            fold.input_witness_len
        });
    let mut aligned_params = params.clone();
    aligned_params.successor_block_len = (aligned_params.witness_chunk.num_chunks > 1).then_some(
        child
            .folds
            .first()
            .map_or(
                akita_params::FoldSuccessor::Terminal(&child.terminal.params),
                |fold| akita_params::FoldSuccessor::Recursive(&fold.params),
            )
            .source_block_len()?,
    );
    let params = &aligned_params;
    let opening_layout = suffix_opening_layout(input_witness_len, None)?;
    let opening_reduction_bytes = if matches!(
        params.opening_method(),
        akita_params::OpeningMethod::EvaluationTrace
    ) {
        akita_params::extension_opening_reduction_level_bytes(
            policy.challenge_field_bits()?,
            policy.claim_ext_degree,
            params.group(),
        )?
    } else {
        0
    };
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
        || akita_params::FoldSuccessor::Terminal(&child.terminal.params),
        |fold| akita_params::FoldSuccessor::Recursive(fold.params.as_ref()),
    );
    let relation_geometry = params.relation_address_geometry(
        &opening_layout,
        policy.claim_ext_degree,
        successor.ring_dimension(),
        output_witness_len,
    )?;
    let edge_grinding_cost = akita_params::transcript_grinding_cost_for_planner_edge(
        params,
        relation_geometry,
        &opening_layout,
        successor,
        policy.transcript_grinding_order()?,
        policy.claim_ext_degree,
        u32::try_from(level)
            .map_err(|_| AkitaError::InvalidSetup("unpruned fold level exceeds u32".into()))?,
    )?;
    let natural_setup_field_len = akita_params::active_setup_field_len(params, &opening_layout)?;
    let scan_work = crate::schedule_params::direct_setup_scan_work_elements(
        natural_setup_field_len,
        relation_geometry.relation_coefficient_block_len(),
    )?;
    let work = output_witness_len
        .checked_add(scan_work)
        .ok_or_else(|| AkitaError::InvalidSetup("unpruned fold work overflow".into()))?;
    let cost = child.cost.checked_prepend(
        direct_bytes,
        edge_grinding_cost.nonce_max_bytes,
        edge_grinding_cost.total_nonce_bits,
        edge_grinding_cost.expanded_query_count,
        work,
    )?;
    if !cost.fits_query_limit() {
        return Ok(None);
    }
    Ok(Some(ScheduleCandidate {
        first_direct_setup_field_len: std::num::NonZeroUsize::new(natural_setup_field_len),
        first_direct_output_witness_len: output_witness_len,
        cost,
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
    }))
}

pub(in crate::schedule_params) fn prepend_root(
    policy: &PlannerPolicy,
    schedule_key: &akita_params::ScheduleLookupKey,
    input_witness_len: usize,
    root_params: &CommittedGroupParams,
    suffix: &ScheduleCandidate,
) -> Result<Option<ScheduleCandidate>, AkitaError> {
    let output_witness_len = suffix
        .folds
        .first()
        .map_or(suffix.terminal.input_witness_len, |fold| {
            fold.input_witness_len
        });
    let mut aligned_root = root_params.clone();
    aligned_root.successor_block_len = (aligned_root.witness_chunk.num_chunks > 1).then_some(
        suffix
            .folds
            .first()
            .map_or(
                akita_params::FoldSuccessor::Terminal(&suffix.terminal.params),
                |fold| akita_params::FoldSuccessor::Recursive(&fold.params),
            )
            .source_block_len()?,
    );
    let root_params = &aligned_root;
    let opening_layout = schedule_key.opening_layout()?;
    let first_direct_setup_field_len =
        std::num::NonZeroUsize::new(active_setup_field_len(root_params, &opening_layout)?)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("unpruned root setup field length must be nonzero".into())
            })?;
    let successor = suffix.folds.first().map_or_else(
        || akita_params::FoldSuccessor::Terminal(&suffix.terminal.params),
        |fold| akita_params::FoldSuccessor::Recursive(fold.params.as_ref()),
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
    let root_grinding_cost = akita_params::transcript_grinding_cost_for_planner_edge(
        root_params,
        relation_geometry,
        &opening_layout,
        successor,
        policy.transcript_grinding_order()?,
        policy.claim_ext_degree,
        0,
    )?;
    let scan_work = crate::schedule_params::direct_setup_scan_work_elements(
        first_direct_setup_field_len.get(),
        relation_geometry.relation_coefficient_block_len(),
    )?;
    let work = output_witness_len
        .checked_add(scan_work)
        .ok_or_else(|| AkitaError::InvalidSetup("unpruned root work overflow".into()))?;
    let cost = suffix.cost.checked_prepend(
        root_bytes,
        root_grinding_cost.nonce_max_bytes,
        root_grinding_cost.total_nonce_bits,
        root_grinding_cost.expanded_query_count,
        work,
    )?;
    if !cost.fits_query_limit() {
        return Ok(None);
    }
    let candidate = ScheduleCandidate {
        first_direct_setup_field_len: Some(first_direct_setup_field_len),
        first_direct_output_witness_len: output_witness_len,
        cost,
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
    let mut folds = candidate
        .folds
        .to_vec()
        .into_iter()
        .map(|fold| akita_params::FoldParams {
            params: (*fold.params).clone(),
            input_witness_len: fold.input_witness_len,
            output_witness_len: fold.output_witness_len,
        });
    let schedule = akita_params::FoldSchedule {
        root: folds.next().ok_or_else(|| {
            AkitaError::UnsupportedSchedule("oracle candidate has no root fold".into())
        })?,
        recursive_folds: folds.collect(),
        terminal: akita_params::TerminalFoldParams {
            fold_challenge_config: candidate.terminal.sparse_challenge_config,
            response_shape: candidate.terminal.response_shape.clone(),
            input_witness_len: candidate.terminal.input_witness_len,
            ..candidate.terminal.params.clone()
        },
    };
    let plan = akita_params::derive_transcript_grinding_plan_from_public_shape(
        &schedule,
        &opening_layout,
        policy.transcript_grinding_order()?,
        policy.claim_ext_degree,
    )?;
    let edge_wise_cost = candidate.cost.grinding_cost();
    if edge_wise_cost.total_nonce_bits != plan.total_nonce_bits()
        || edge_wise_cost.nonce_max_bytes != plan.nonce_max_bytes()
        || edge_wise_cost.expanded_query_count != plan.expanded_query_count()
    {
        return Err(AkitaError::InvalidSetup(
            "edge-wise oracle grinding cost disagrees with the canonical complete schedule".into(),
        ));
    }
    Ok(Some(candidate))
}
