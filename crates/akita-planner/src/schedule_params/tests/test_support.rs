use super::*;

/// One recursive fold of an independently planned test suffix.
#[derive(Clone, Debug)]
pub struct PlannedSuffixFold {
    /// Committed-group params for this fold level (already priced at
    /// `policy.ring_dimension`).
    pub params: CommittedGroupParams,
    /// Field-element witness length entering this fold.
    pub input_witness_len: usize,
    /// Field-element witness length produced for the next level.
    pub output_witness_len: usize,
}

/// Terminal (cleartext) response of an independently planned test suffix.
#[derive(Clone, Debug)]
pub struct PlannedSuffixTerminal {
    /// Terminal committed-group params.
    pub params: akita_types::TerminalCommittedGroupParams,
    /// Short ring challenge family for the terminal fold.
    pub sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    /// Field-element witness length entering the terminal fold.
    pub input_witness_len: usize,
    /// Cleartext response wire shape.
    pub response_shape: TerminalResponseShape,
}

/// Optimal recursive suffix planned from an intermediate witness for synthetic tests.
#[derive(Clone, Debug)]
pub struct PlannedSuffix {
    /// Recursive fold levels, starting at `start_level`.
    pub folds: Vec<PlannedSuffixFold>,
    /// Terminal fold.
    pub terminal: PlannedSuffixTerminal,
    /// Header-stripped direct-mode proof bytes of the suffix (folds + terminal).
    pub total_bytes: usize,
}

/// Plan the proof-size-optimal recursive suffix that folds a witness of
/// `start_witness_len` field elements down to a cleartext terminal.
///
/// This test-support helper lets synthetic schedule builders splice an optimal
/// fixed-D suffix onto a custom predecessor boundary without exposing the API
/// from normal planner builds.
///
/// # Errors
///
/// Returns [`AkitaError::UnsupportedSchedule`] if no terminating suffix exists
/// for the requested state, or propagates SIS-sizing / overflow failures.
pub fn plan_optimal_suffix(
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
    num_vars: usize,
    start_level: usize,
    start_witness_len: usize,
    start_lb: u32,
) -> Result<PlannedSuffix, AkitaError> {
    validate_policy(policy)?;
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning is not supported by plan_optimal_suffix".to_string(),
        ));
    }
    let ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &ring_challenge_cfg,
        ring_challenge_config: &ring_challenge_config,
        fold_challenge_shape_at_level: &fold_challenge_shape_at_level,
        num_vars,
        key: PolynomialGroupLayout::singleton(num_vars),
        setup_field_budget: None,
        root_lookup_key: None,
        level_zero_is_root: false,
    };
    let mut memo = ScheduleMemo::new();
    let result = derive_optimal_suffix_schedule(
        &ctx,
        &mut memo,
        SuffixState {
            level: start_level,
            current_witness_len: start_witness_len,
            current_lb: start_lb,
            incoming_setup_prefix: None,
        },
        0,
    )?;
    let best = result
        .best_by_payload_per_lb
        .values()
        .min_by_key(|suffix| suffix.total_bytes)
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no terminating suffix for witness_len={start_witness_len} at level {start_level}"
            ))
        })?;
    Ok(PlannedSuffix {
        folds: best
            .folds
            .iter()
            .map(|fold| PlannedSuffixFold {
                params: fold.params.clone(),
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            })
            .collect(),
        terminal: PlannedSuffixTerminal {
            params: best.terminal.params.clone(),
            sparse_challenge_config: best.terminal.sparse_challenge_config,
            input_witness_len: best.terminal.input_witness_len,
            response_shape: best.terminal.response_shape.clone(),
        },
        total_bytes: best.total_bytes,
    })
}

/// Inputs for synthetic setup-prefix commitment planning.
pub struct SetupPrefixPlanRequest<'a> {
    pub policy: &'a PlannerPolicy,
    pub ring_challenge: &'a SparseChallengeConfig,
    pub fold_shape: TensorChallengeShape,
    pub log_basis_outer: u32,
    pub log_basis_open: u32,
    pub prefix_field_elements: usize,
    pub num_chunks: usize,
    pub outer_ring_dimension: usize,
}

/// Plan one synthetic setup-prefix commitment used by test and profile fixtures.
///
/// # Errors
///
/// Returns an error for malformed policy or dimensions, or when no audited
/// secure setup-prefix geometry exists.
pub fn plan_setup_prefix_commitment(
    request: SetupPrefixPlanRequest<'_>,
) -> Result<PrecommittedLevelParams, AkitaError> {
    validate_policy(request.policy)?;
    candidate::derive_setup_prefix_group(
        request.policy,
        request.ring_challenge,
        request.fold_shape,
        request.log_basis_outer,
        request.log_basis_open,
        request.prefix_field_elements,
        request.num_chunks,
        request.outer_ring_dimension,
    )?
    .ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "no setup-prefix commitment at A{}/B{} for n_prefix={}",
            request.policy.ring_dimension,
            request.outer_ring_dimension,
            request.prefix_field_elements
        ))
    })
}
