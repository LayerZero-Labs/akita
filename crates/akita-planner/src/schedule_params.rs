//! FoldSchedule planner that finds the global minimum proof size. Recursive
//! grouped scheduling additionally minimizes the first direct setup footprint
//! before proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key)` for offline table generation.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{
    extension_opening_reduction_level_bytes, intermediate_w_ring_element_count_for_chunks,
    level_proof_bytes, padded_setup_prefix_len, AkitaScheduleInputs, CommitmentRingDims,
    CommittedGroupParams, CommittedGroupProfile, DecompositionParams, FoldSchedule,
    FoldScheduleEstimate, OpeningClaimsLayout, PlannedFoldSchedule, PolynomialGroupLayout,
    PrecommittedLevelParams, RecursiveFoldParams, RecursiveFoldStep, RootFinalChallenge,
    RootFinalGroupParams, RootFoldParams, RootFoldStep, RootPrecommittedGroupParams,
    TerminalFoldParams, TerminalFoldStep, TerminalResponseShape, WitnessLayout, WitnessPartition,
};

use crate::PlannerPolicy;

mod candidate;
mod suffix_dp;

pub use candidate::suffix_opening_layout;
pub(crate) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
    scalar_root_fold_level_params_candidate,
};
pub(crate) use suffix_dp::{
    derive_optimal_suffix_schedule, MixedFrontierMode, ScheduleMemo, SuffixCtx, SuffixState,
};

const MIXED_SEARCH_FOLD_LEVELS: usize = 2;
const MIXED_SEARCH_SUFFIX_RING_DIMENSION: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct CandidateFoldStep {
    pub(crate) params: CommittedGroupParams,
    pub(crate) input_witness_len: usize,
    pub(crate) output_witness_len: usize,
    pub(crate) estimated_direct_payload_bytes: usize,
    pub(crate) estimated_stage3_payload_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateTerminalResponse {
    pub(crate) params: akita_types::TerminalCommittedGroupParams,
    pub(crate) sparse_challenge_config: akita_challenges::SparseChallengeConfig,
    pub(crate) input_witness_len: usize,
    pub(crate) estimated_direct_payload_bytes: usize,
    pub(crate) response_shape: TerminalResponseShape,
    pub(crate) estimated_payload_bytes: usize,
}

/// Explicit A/B/D dimensions admitted by mixed-D planner search.
///
/// `PlannerPolicy::ring_dimension` remains the setup generation dimension and
/// the implicit singleton domain used by [`find_schedule`]. This separate
/// offline-only value makes mixed-D search opt-in without changing runtime
/// policy or existing catalog identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingDimensionSearchDomain {
    candidates: Vec<CommitmentRingDims>,
}

impl RingDimensionSearchDomain {
    /// Construct and canonicalize a non-empty dimension domain.
    ///
    /// Every tuple must satisfy the A-carrier invariant, and every role
    /// dimension must divide `setup_generation_dimension`.
    pub fn new(
        setup_generation_dimension: usize,
        candidates: impl IntoIterator<Item = CommitmentRingDims>,
    ) -> Result<Self, AkitaError> {
        if setup_generation_dimension == 0 || !setup_generation_dimension.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "setup generation dimension must be a nonzero power of two".into(),
            ));
        }
        let mut candidates = candidates.into_iter().collect::<Vec<_>>();
        candidates.sort_by_key(|dims| (dims.d_a(), dims.d_b(), dims.d_d()));
        candidates.dedup();
        if candidates.is_empty() {
            return Err(AkitaError::InvalidSetup(
                "ring-dimension search domain must be nonempty".into(),
            ));
        }
        for dims in &candidates {
            dims.validate_a_carrier()?;
            for d in [dims.d_a(), dims.d_b(), dims.d_d()] {
                if !setup_generation_dimension.is_multiple_of(d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "candidate dimension D{d} does not divide setup generation dimension \
                         D{setup_generation_dimension}"
                    )));
                }
            }
        }
        Ok(Self { candidates })
    }

    fn uniform(d: usize) -> Self {
        Self {
            candidates: vec![CommitmentRingDims::uniform(d)],
        }
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }
}

/// Plan the commitment geometry for one setup prefix consumed by a recursive
/// fold.
///
/// The prefix source and A matrix use `policy.ring_dimension`; B may use an
/// independently selected divisor. This is the same candidate derivation used
/// by recursive schedule planning, exposed for exact synthetic schedules that
/// establish a mixed-D boundary before the production planner can search it.
///
/// # Errors
///
/// Returns an error for malformed policy/dimensions or when no audited secure
/// setup-prefix geometry exists.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "test-support")]
pub fn plan_setup_prefix_commitment(
    policy: &PlannerPolicy,
    ring_challenge_cfg: &SparseChallengeConfig,
    requested_fold_shape: TensorChallengeShape,
    log_basis_outer: u32,
    log_basis_open: u32,
    n_prefix: usize,
    num_chunks: usize,
    outer_ring_dimension: usize,
) -> Result<PrecommittedLevelParams, AkitaError> {
    validate_policy(policy)?;
    candidate::derive_setup_prefix_group(
        policy,
        ring_challenge_cfg,
        requested_fold_shape,
        log_basis_outer,
        log_basis_open,
        n_prefix,
        num_chunks,
        outer_ring_dimension,
    )?
    .ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "no setup-prefix commitment at A{}/B{outer_ring_dimension} for n_prefix={n_prefix}",
            policy.ring_dimension
        ))
    })
}

#[derive(Clone, Debug)]
pub(crate) struct CandidateScheduleChoice {
    pub(crate) first_direct_setup_field_len: Option<usize>,
    pub(crate) total_bytes: usize,
    pub(crate) setup_envelope_ring_elements: usize,
    pub(crate) folds: Vec<CandidateFoldStep>,
    pub(crate) terminal: CandidateTerminalResponse,
}

pub(crate) fn level_setup_envelope_at_generation(
    params: &CommittedGroupParams,
    setup_generation_dimension: usize,
) -> Result<usize, AkitaError> {
    if setup_generation_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup generation dimension must be nonzero".into(),
        ));
    }
    let mut field_elements = 1;
    akita_types::accumulate_matrix_field_elements_for_level(params, &mut field_elements)?;
    Ok(field_elements.div_ceil(setup_generation_dimension))
}

pub(crate) fn terminal_setup_envelope_at_generation(
    params: &akita_types::TerminalCommittedGroupParams,
    setup_generation_dimension: usize,
) -> Result<usize, AkitaError> {
    if setup_generation_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup generation dimension must be nonzero".into(),
        ));
    }
    let mut field_elements = 1;
    akita_types::accumulate_terminal_matrix_field_elements(params, &mut field_elements)?;
    Ok(field_elements.div_ceil(setup_generation_dimension))
}

/// Exact Stage-3 payload induced when `successor` consumes the setup prefix
/// produced by the current fold. Absence of a successor prefix is direct mode.
pub(crate) fn stage3_payload_bytes_for_successor(
    policy: &PlannerPolicy,
    successor: Option<&CommittedGroupParams>,
) -> Result<usize, AkitaError> {
    let Some(prefix) = successor.and_then(|params| params.setup_prefix.as_ref()) else {
        return Ok(usize::default());
    };
    let n_prefix = prefix.n_prefix()?;
    if prefix.d_setup == 0 || !n_prefix.is_multiple_of(prefix.d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix field length does not align with its ring dimension".to_string(),
        ));
    }
    let challenge_field_bits = policy
        .decomposition
        .field_bits()
        .checked_mul(policy.chal_ext_degree as u32)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("challenge field bit width overflow".to_string())
        })?;
    Ok(akita_types::proof_size::stage3_setup_product_bytes(
        challenge_field_bits,
        prefix.d_setup,
        n_prefix / prefix.d_setup,
    ))
}

pub(crate) fn materialize_candidate_schedule(
    cached_total: usize,
    cached_setup_envelope: usize,
    setup_generation_dimension: usize,
    first_direct_setup_field_len: Option<usize>,
    mut folds: Vec<CandidateFoldStep>,
    terminal_response: CandidateTerminalResponse,
) -> Result<PlannedFoldSchedule, AkitaError> {
    if folds.is_empty() {
        return Err(AkitaError::UnsupportedSchedule(
            "a fold schedule requires root and terminal folds".to_string(),
        ));
    }
    let root = folds.remove(0);
    let mut estimate = FoldScheduleEstimate {
        estimated_root_direct_payload_bytes: root.estimated_direct_payload_bytes,
        estimated_root_stage3_payload_bytes: root.estimated_stage3_payload_bytes,
        estimated_recursive_direct_payload_bytes: folds
            .iter()
            .map(|fold| fold.estimated_direct_payload_bytes)
            .collect(),
        estimated_recursive_stage3_payload_bytes: folds
            .iter()
            .map(|fold| fold.estimated_stage3_payload_bytes)
            .collect(),
        estimated_terminal_direct_payload_bytes: terminal_response
            .estimated_direct_payload_bytes
            .checked_add(terminal_response.estimated_payload_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal estimate overflow".to_string()))?,
        estimated_terminal_response_payload_bytes: terminal_response.estimated_payload_bytes,
        estimated_setup_envelope_ring_elements: cached_setup_envelope,
        first_direct_setup_field_len,
        selected_offload_edges: 0,
    };
    let recomputed = estimate.estimated_proof_payload_bytes()?;
    if recomputed != cached_total {
        return Err(AkitaError::InvalidSetup(format!(
            "cached planner cost {cached_total} disagrees with materialized estimate {recomputed}"
        )));
    }
    let schedule = FoldSchedule {
        root: RootFoldStep {
            params: RootFoldParams {
                final_group: RootFinalGroupParams {
                    challenge: match root.params.fold_challenge_shape {
                        TensorChallengeShape::Flat => RootFinalChallenge::Flat,
                        TensorChallengeShape::Tensor { fold_low_len } => {
                            RootFinalChallenge::Tensor { fold_low_len }
                        }
                    },
                    commitment: root.params.clone(),
                },
                precommitted_groups: root
                    .params
                    .precommitted_groups
                    .iter()
                    .cloned()
                    .map(|commitment| RootPrecommittedGroupParams {
                        descriptor: commitment.layout,
                        commitment,
                    })
                    .collect(),
                open_commit_matrix: root.params.open_commit_matrix,
                sparse_challenge_config: root.params.fold_challenge_config,
                witness_partition: witness_partition(root.params.witness_chunk.num_chunks),
            },
            input_witness_len: root.input_witness_len,
            output_witness_len: root.output_witness_len,
        },
        recursive_folds: folds
            .into_iter()
            .map(|fold| RecursiveFoldStep {
                params: RecursiveFoldParams {
                    open_commit_matrix: fold.params.open_commit_matrix,
                    sparse_challenge_config: fold.params.fold_challenge_config,
                    incoming_setup_prefix: fold.params.setup_prefix.clone(),
                    witness_partition: witness_partition(fold.params.witness_chunk.num_chunks),
                    witness: fold.params,
                },
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
            })
            .collect(),
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                sparse_challenge_config: terminal_response.sparse_challenge_config,
                witness: terminal_response.params,
                response_shape: terminal_response.response_shape,
            },
            input_witness_len: terminal_response.input_witness_len,
        },
    };
    schedule.validate_structure()?;
    if setup_generation_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup generation dimension must be nonzero".into(),
        ));
    }
    let recomputed_envelope = akita_types::setup_matrix_field_elements_for_schedule(&schedule)?
        .div_ceil(setup_generation_dimension);
    if recomputed_envelope != cached_setup_envelope {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup envelope {cached_setup_envelope} disagrees with materialized envelope {recomputed_envelope}"
        )));
    }
    estimate.selected_offload_edges = schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.incoming_setup_prefix.is_some())
        .count();
    Ok(PlannedFoldSchedule { schedule, estimate })
}

fn candidate_schedule_descriptor_bytes(
    choice: &CandidateScheduleChoice,
    setup_generation_dimension: usize,
) -> Result<Vec<u8>, AkitaError> {
    Ok(materialize_candidate_schedule(
        choice.total_bytes,
        choice.setup_envelope_ring_elements,
        setup_generation_dimension,
        choice.first_direct_setup_field_len,
        choice.folds.clone(),
        choice.terminal.clone(),
    )?
    .schedule
    .canonical_descriptor_bytes())
}

fn witness_partition(num_chunks: usize) -> WitnessPartition {
    if num_chunks == 1 {
        WitnessPartition::Single
    } else {
        WitnessPartition::Distributed { num_chunks }
    }
}

/// Validate the complete planner policy at a verifier-reachable entry point.
///
/// Layout-only rules live on [`akita_types::ChunkedWitnessCfg::validate`]; the recursion-depth
/// bound (which needs the planner-private [`MAX_RECURSION_DEPTH`]) is enforced
/// here so `akita-types` stays free of planner internals.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for an invalid [`akita_types::ChunkedWitnessCfg`], or
/// `num_activated_levels` beyond the planner recursion cap. Verifier-reachable: never panics.
pub(crate) fn validate_policy(policy: &PlannerPolicy) -> Result<(), AkitaError> {
    let expected_selection_policy = if policy.recursive_setup_planning {
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope
    } else {
        crate::SelectionPolicyId::MinEstimatedProofPayload
    };
    if policy.selection_policy != expected_selection_policy {
        return Err(AkitaError::InvalidSetup(
            "planner selection policy disagrees with recursive setup capability".to_string(),
        ));
    }
    if policy.max_setup_envelope_field_elements == 0 {
        return Err(AkitaError::InvalidSetup(
            "maximum setup envelope must be positive".to_string(),
        ));
    }
    if policy.min_offloaded_witness_contraction == 0 {
        return Err(AkitaError::InvalidSetup(
            "minimum offloaded witness contraction must be positive".to_string(),
        ));
    }
    let mc = policy.witness_chunk;
    mc.validate()?;
    if mc.num_activated_levels > MAX_RECURSION_DEPTH {
        return Err(AkitaError::InvalidSetup(format!(
            "num_activated_levels={} exceeds the planner recursion cap {MAX_RECURSION_DEPTH}",
            mc.num_activated_levels
        )));
    }
    Ok(())
}

/// Stage-1 sparse-challenge closure shared by the planner entry points.
pub(crate) type RingChallengeConfigFn<'a> =
    &'a dyn Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>;

pub(crate) type LayoutCandidateScore = (usize, usize, usize, usize);

/// Resolve the tensor low length independently from the num_positions_per_block split.
/// A tensor-enabled policy selects the shape family; the planner enumerates
/// every power-of-two low length through the Boolean block-index domain size and chooses
/// the minimum exact `Q + ceil(F/Q)` verifier work.
pub(crate) fn optimize_fold_challenge_shape(
    requested: TensorChallengeShape,
    num_live_blocks: usize,
) -> Result<TensorChallengeShape, AkitaError> {
    if num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "fold-shape optimization requires a positive num_live_blocks".to_string(),
        ));
    }
    if matches!(requested, TensorChallengeShape::Flat) {
        return Ok(TensorChallengeShape::Flat);
    }

    let capacity = num_live_blocks.checked_next_power_of_two().ok_or_else(|| {
        AkitaError::InvalidSetup("tensor low-length capacity overflow".to_string())
    })?;
    let mut best = None;
    let mut low_len = 1usize;
    loop {
        let high_len = num_live_blocks.div_ceil(low_len);
        let work = high_len
            .checked_add(low_len)
            .ok_or_else(|| AkitaError::InvalidSetup("tensor verifier-work overflow".to_string()))?;
        if best.is_none_or(|(best_work, best_low)| (work, low_len) < (best_work, best_low)) {
            best = Some((work, low_len));
        }
        if low_len == capacity {
            break;
        }
        low_len = low_len.checked_mul(2).ok_or_else(|| {
            AkitaError::InvalidSetup("tensor low-length enumeration overflow".to_string())
        })?;
    }
    let (_, fold_low_len) = best.ok_or_else(|| {
        AkitaError::InvalidSetup("tensor low-length enumeration was empty".to_string())
    })?;
    Ok(TensorChallengeShape::Tensor { fold_low_len })
}

/// Combine exact physical width, challenge-factor work, chunk evaluator work,
/// and load imbalance when comparing `M` candidates. All terms count ring or
/// scalar work units; exact physical width remains an explicit tie-breaker.
pub(crate) fn layout_candidate_score(
    physical_width: usize,
    num_live_blocks: usize,
    num_chunks: usize,
    fold_shape: TensorChallengeShape,
) -> Result<LayoutCandidateScore, AkitaError> {
    let challenge_work = match fold_shape {
        TensorChallengeShape::Flat => num_live_blocks,
        TensorChallengeShape::Tensor { fold_low_len } => fold_low_len
            .checked_add(num_live_blocks.div_ceil(fold_low_len))
            .ok_or_else(|| AkitaError::InvalidSetup("challenge-work overflow".to_string()))?,
    };
    let chunk_ranges = WitnessLayout::resolve_chunk_block_ranges(num_live_blocks, num_chunks)?;
    let min_load = chunk_ranges
        .iter()
        .map(|range| range.len())
        .min()
        .ok_or_else(|| AkitaError::InvalidSetup("balanced chunk geometry is empty".to_string()))?;
    let max_load = chunk_ranges
        .iter()
        .map(|range| range.len())
        .max()
        .ok_or_else(|| AkitaError::InvalidSetup("balanced chunk geometry is empty".to_string()))?;
    let chunk_work = num_live_blocks;
    let imbalance = max_load - min_load;
    let combined = physical_width
        .checked_add(challenge_work)
        .and_then(|cost| cost.checked_add(chunk_work))
        .and_then(|cost| cost.checked_add(imbalance))
        .ok_or_else(|| AkitaError::InvalidSetup("layout candidate score overflow".to_string()))?;
    Ok((combined, physical_width, chunk_work, imbalance))
}

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
pub(crate) const MAX_RECURSION_DEPTH: usize = 12;

/// Find the optimal schedule for a root schedule lookup key under `policy`.
///
/// Runs an exhaustive DP that minimizes proof size. The result is a pure,
/// deterministic function of `(policy, key)` (plus the `ring_challenge_config` /
/// `fold_challenge_shape_at_level` closures, which presets derive from the same hooks the
/// generated tables were emitted from), so the prover and verifier
/// regenerate identical schedules on a table miss.
///
/// # Errors
///
/// Returns an error if vector counts are invalid or if the witness length
/// overflows. The function never panics on malformed input — it is
/// verifier-reachable and audited under the no-panic contract.
pub fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let dimensions = RingDimensionSearchDomain::uniform(policy.ring_dimension);
    find_schedule_inner(
        key,
        policy,
        &dimensions,
        ScheduleSelectionObjective::ProofPayload,
        MixedFrontierMode::Pareto,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

/// Find a schedule over an explicit mixed A/B/D dimension domain.
///
/// This offline-only entry point minimizes physical setup field elements
/// first and exact modeled proof payload second. Existing callers should keep
/// using [`find_schedule`], whose singleton domain and payload-only comparator
/// preserve the current scalar-D behavior.
///
/// Recursive setup offloading remains on the existing D64-only planner path;
/// this first mixed-D cut supports direct scalar schedules. Mixed A/B/D
/// candidates are searched only at fold levels 0 and 1, dimensions are
/// component-wise non-increasing, and level 2 onward reuses the existing
/// uniform-D64 split search.
pub fn find_schedule_with_ring_dimension_domain(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    dimensions: &RingDimensionSearchDomain,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "mixed-D search does not yet support recursive setup planning".into(),
        ));
    }
    if policy.witness_chunk.uses_multi_chunk() {
        return Err(AkitaError::InvalidSetup(
            "mixed-D search does not yet support direct multi-chunk planning".into(),
        ));
    }
    let validated = RingDimensionSearchDomain::new(
        policy.ring_dimension,
        dimensions.candidates().iter().copied(),
    )?;
    let suffix_dimensions = CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION);
    if !validated.candidates().contains(&suffix_dimensions) {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed-D search requires the D{MIXED_SEARCH_SUFFIX_RING_DIMENSION} uniform candidate \
             used from fold level {MIXED_SEARCH_FOLD_LEVELS} onward"
        )));
    }
    if validated.candidates().iter().any(|dims| {
        dims.d_a() < MIXED_SEARCH_SUFFIX_RING_DIMENSION
            || dims.d_b() < MIXED_SEARCH_SUFFIX_RING_DIMENSION
            || dims.d_d() < MIXED_SEARCH_SUFFIX_RING_DIMENSION
    }) {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed-D candidates must be component-wise at least \
             D{MIXED_SEARCH_SUFFIX_RING_DIMENSION} so the schedule can return monotonically to \
             uniform D{MIXED_SEARCH_SUFFIX_RING_DIMENSION}"
        )));
    }
    find_schedule_inner(
        key,
        policy,
        &validated,
        ScheduleSelectionObjective::SetupThenProofPayload,
        MixedFrontierMode::Pareto,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScheduleSelectionObjective {
    ProofPayload,
    SetupThenProofPayload,
}

fn componentwise_dimensions_at_most(
    dimensions: CommitmentRingDims,
    ceiling: CommitmentRingDims,
) -> bool {
    dimensions.d_a() <= ceiling.d_a()
        && dimensions.d_b() <= ceiling.d_b()
        && dimensions.d_d() <= ceiling.d_d()
}

/// One recursive fold of an independently planned suffix
/// ([`plan_optimal_suffix`]).
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

/// Terminal (cleartext) response of an independently planned suffix.
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

/// Optimal recursive suffix planned from an intermediate witness.
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
/// `start_witness_len` field elements (produced by some predecessor fold at
/// `start_level - 1`) down to a cleartext terminal, at `policy.ring_dimension`.
///
/// This is the exact suffix DP [`find_schedule`] runs after choosing a root,
/// exposed so callers can splice an optimal suffix onto a differently sized
/// predecessor — e.g. a mixed ring-dimension-per-level schedule whose root
/// folds at a larger ring dimension than the suffix. `start_lb` is the
/// predecessor level's `log_basis` (fold `log_basis` is non-decreasing), and
/// `num_vars` is the opening arity (used for the singleton opening layout the
/// suffix prices against).
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
    let dimensions = [CommitmentRingDims::uniform(policy.ring_dimension)];
    let ctx = SuffixCtx {
        policy,
        dimension_candidates: &dimensions,
        objective: ScheduleSelectionObjective::ProofPayload,
        default_ring_challenge_cfg: &ring_challenge_cfg,
        ring_challenge_config: &ring_challenge_config,
        fold_challenge_shape_at_level: &fold_challenge_shape_at_level,
        num_vars,
        key: PolynomialGroupLayout::singleton(num_vars),
        setup_envelope_budget: None,
        root_lookup_key: None,
        mixed_frontier_mode: MixedFrontierMode::Pareto,
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
            dimension_ceiling: None,
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

fn find_schedule_inner(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    dimensions: &RingDimensionSearchDomain,
    objective: ScheduleSelectionObjective,
    mixed_frontier_mode: MixedFrontierMode,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape = &fold_challenge_shape_at_level;

    key.validate()?;
    validate_policy(policy)?;
    let default_ring_challenge_cfg = match objective {
        ScheduleSelectionObjective::ProofPayload => ring_challenge_config(policy.ring_dimension)?,
        ScheduleSelectionObjective::SetupThenProofPayload => dimensions
            .candidates()
            .iter()
            .find_map(|dims| ring_challenge_config(dims.d_a()).ok())
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule(
                    "no ring-dimension candidate has fold-challenge support".into(),
                )
            })?,
    };
    let suffix_ctx = SuffixCtx {
        policy,
        dimension_candidates: dimensions.candidates(),
        objective,
        default_ring_challenge_cfg: &default_ring_challenge_cfg,
        ring_challenge_config,
        fold_challenge_shape_at_level: fold_shape,
        num_vars: key.num_vars(),
        key,
        setup_envelope_budget: None,
        root_lookup_key: None,
        mixed_frontier_mode,
    };

    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let field_bits = policy.decomposition.field_bits();
    let mut best: Option<CandidateScheduleChoice> = None;
    let fold_challenge_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len: witness_len,
    });
    let mut memo = ScheduleMemo::new();

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);
    let root_eor_bytes = extension_opening_reduction_level_bytes(
        policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
        policy.claim_ext_degree,
        0,
        key,
        witness_len,
        policy.ring_dimension,
    )
    .ok();

    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let mut root_candidates = Vec::new();
        for candidate_dimensions in dimensions.candidates() {
            let alpha = (candidate_dimensions.d_a() as u32).trailing_zeros() as usize;
            let reduced_vars = key.num_vars().saturating_sub(alpha);
            if reduced_vars == 0 {
                continue;
            }
            let min_block_index_bits: usize = if reduced_vars >= 3 { 1 } else { 0 };
            let max_block_index_bits: usize = (reduced_vars - 1).min(usize::BITS as usize - 1);
            let Ok(ring_challenge_cfg) = ring_challenge_config(candidate_dimensions.d_a()) else {
                continue;
            };
            for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
                let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                    policy,
                    &ring_challenge_cfg,
                    *candidate_dimensions,
                    key.num_vars(),
                    key.num_polynomials(),
                    candidate_log_basis,
                    block_index_bits,
                    fold_challenge_shape,
                )?
                else {
                    continue;
                };

                let output_witness_len = intermediate_w_ring_element_count_for_chunks(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials(),
                    root_num_chunks,
                )?
                .checked_mul(candidate_dimensions.d_a())
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("root next witness length overflow".into())
                })?;
                let initial_witness_len_bits = witness_len
                    .checked_mul(field_bits as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root witness bit length overflow".into())
                    })?;
                if output_witness_len
                    .checked_mul(candidate_log_basis as usize)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("root next witness bit length overflow".into())
                    })?
                    >= initial_witness_len_bits
                {
                    continue;
                }
                root_candidates.push((candidate_params, output_witness_len));
            }
        }
        for (candidate_params, output_witness_len) in root_candidates {
            let candidate_dimensions = candidate_params.role_dims();
            let suffix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                SuffixState {
                    level: 1,
                    current_witness_len: output_witness_len,
                    current_lb: candidate_log_basis,
                    incoming_setup_prefix: None,
                    dimension_ceiling: matches!(
                        objective,
                        ScheduleSelectionObjective::SetupThenProofPayload
                    )
                    .then_some(candidate_dimensions),
                },
                0,
            )?;
            if suffix.is_empty() {
                continue;
            }
            let Some(eor_bytes) = root_eor_bytes else {
                continue;
            };

            let suffix_candidates = match objective {
                ScheduleSelectionObjective::ProofPayload => {
                    suffix.best_by_payload_per_lb.values().collect::<Vec<_>>()
                }
                ScheduleSelectionObjective::SetupThenProofPayload => {
                    suffix.mixed_frontier.iter().collect::<Vec<_>>()
                }
            };
            // A supported root must recurse into at least one suffix fold.
            for suffix_fold in suffix_candidates {
                let next_witness_binding = if suffix_fold.folds.is_empty() {
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                } else {
                    akita_types::NextWitnessBindingPolicy::OuterCommitment
                };
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    suffix_fold.first_fold_params.as_ref(),
                    output_witness_len,
                    Some(next_witness_binding),
                )? + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                let root_envelope =
                    level_setup_envelope_at_generation(&candidate_params, policy.ring_dimension)?;
                let setup_envelope = root_envelope.max(suffix_fold.setup_envelope_ring_elements);
                let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
                folds.push(CandidateFoldStep {
                    params: candidate_params.clone(),
                    input_witness_len: witness_len,
                    output_witness_len,
                    estimated_direct_payload_bytes: root_proof_size,
                    estimated_stage3_payload_bytes: 0,
                });
                folds.extend(suffix_fold.folds.iter().cloned());
                let candidate = CandidateScheduleChoice {
                    first_direct_setup_field_len: None,
                    total_bytes: total,
                    setup_envelope_ring_elements: setup_envelope,
                    folds,
                    terminal: suffix_fold.terminal.clone(),
                };
                let replace = match &best {
                    None => true,
                    Some(current) => match objective {
                        ScheduleSelectionObjective::ProofPayload => {
                            candidate.total_bytes < current.total_bytes
                        }
                        ScheduleSelectionObjective::SetupThenProofPayload => {
                            let candidate_cost = (
                                candidate.setup_envelope_ring_elements,
                                candidate.total_bytes,
                            );
                            let current_cost =
                                (current.setup_envelope_ring_elements, current.total_bytes);
                            candidate_cost < current_cost
                                || (candidate_cost == current_cost
                                    && candidate_schedule_descriptor_bytes(
                                        &candidate,
                                        policy.ring_dimension,
                                    )? < candidate_schedule_descriptor_bytes(
                                        current,
                                        policy.ring_dimension,
                                    )?)
                        }
                    },
                };
                if replace {
                    best = Some(candidate);
                }
            }
        }
    }

    let Some(best) = best else {
        return Err(AkitaError::UnsupportedSchedule(format!(
            "no schedule with at least two folds for num_vars={}, num_polynomials={}",
            key.num_vars(),
            key.num_polynomials()
        )));
    };
    materialize_candidate_schedule(
        best.total_bytes,
        best.setup_envelope_ring_elements,
        policy.ring_dimension,
        best.first_direct_setup_field_len,
        best.folds,
        best.terminal,
    )
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    #[test]
    fn tensor_low_length_is_selected_independently() {
        assert_eq!(
            optimize_fold_challenge_shape(TensorChallengeShape::Tensor { fold_low_len: 1 }, 13,)
                .unwrap(),
            TensorChallengeShape::Tensor { fold_low_len: 4 },
        );
    }

    #[test]
    fn balanced_chunk_geometry_prices_exact_work_and_residual_imbalance() {
        let flat = TensorChallengeShape::Flat;
        assert_eq!(
            layout_candidate_score(100, 13, 3, flat).unwrap(),
            (127, 100, 13, 1)
        );
        assert_eq!(
            layout_candidate_score(100, 12, 3, flat).unwrap(),
            (124, 100, 12, 0)
        );
    }

    #[test]
    fn ring_dimension_domain_is_canonical_and_rejects_invalid_carriers() {
        let domain = RingDimensionSearchDomain::new(
            256,
            [
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64,
                },
                CommitmentRingDims::uniform(64),
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            domain.candidates(),
            &[
                CommitmentRingDims::uniform(64),
                CommitmentRingDims {
                    inner: 128,
                    outer: 64,
                    opening: 64
                },
            ]
        );
        assert!(RingDimensionSearchDomain::new(
            256,
            [CommitmentRingDims {
                inner: 64,
                outer: 128,
                opening: 64
            }]
        )
        .is_err());
        assert!(RingDimensionSearchDomain::new(128, [CommitmentRingDims::uniform(256)]).is_err());
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_domain_search_beats_or_ties_uniform_d64() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

        let policy = policy_of::<D256OneHot>();
        let dimensions = [
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
        ];
        let domain = RingDimensionSearchDomain::new(policy.ring_dimension, dimensions).unwrap();
        let key = PolynomialGroupLayout::singleton(16);
        let selected = find_schedule_with_ring_dimension_domain(
            key,
            &policy,
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let selected_score = (
            selected.estimate.estimated_setup_envelope_ring_elements,
            selected.estimate.estimated_proof_payload_bytes().unwrap(),
        );

        let uniform =
            RingDimensionSearchDomain::new(policy.ring_dimension, [dimensions[0]]).unwrap();
        let candidate = find_schedule_with_ring_dimension_domain(
            key,
            &policy,
            &uniform,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        assert!(
            selected_score
                <= (
                    candidate.estimate.estimated_setup_envelope_ring_elements,
                    candidate.estimate.estimated_proof_payload_bytes().unwrap(),
                )
        );

        let schedule = &selected.schedule;
        assert!(domain
            .candidates()
            .contains(&schedule.root.params.final_group.commitment.role_dims()));
        let mut previous = schedule.root.params.final_group.commitment.role_dims();
        for (index, fold) in schedule.recursive_folds.iter().enumerate() {
            let current = fold.params.witness.role_dims();
            assert!(componentwise_dimensions_at_most(current, previous));
            if index + 1 >= MIXED_SEARCH_FOLD_LEVELS {
                assert_eq!(
                    current,
                    CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION)
                );
            }
            previous = current;
        }
        assert_eq!(
            schedule.terminal.params.witness.d_a(),
            MIXED_SEARCH_SUFFIX_RING_DIMENSION
        );
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_frontier_matches_exhaustive_oracle_and_is_canonical() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

        let policy = policy_of::<D256OneHot>();
        let d64 = CommitmentRingDims::uniform(64);
        let a128 = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        };
        let reversed_with_duplicate =
            RingDimensionSearchDomain::new(policy.ring_dimension, [a128, d64, a128]).unwrap();
        let canonical = RingDimensionSearchDomain::new(policy.ring_dimension, [d64, a128]).unwrap();
        let key = PolynomialGroupLayout::singleton(16);

        let selected = find_schedule_with_ring_dimension_domain(
            key,
            &policy,
            &reversed_with_duplicate,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let exhaustive = find_schedule_inner(
            key,
            &policy,
            &canonical,
            ScheduleSelectionObjective::SetupThenProofPayload,
            MixedFrontierMode::Exhaustive,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let repeated = find_schedule_with_ring_dimension_domain(
            key,
            &policy,
            &canonical,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();

        assert_eq!(
            (
                selected.estimate.estimated_setup_envelope_ring_elements,
                selected.estimate.estimated_proof_payload_bytes().unwrap(),
            ),
            (
                exhaustive.estimate.estimated_setup_envelope_ring_elements,
                exhaustive.estimate.estimated_proof_payload_bytes().unwrap(),
            )
        );
        let selected_descriptor = selected.schedule.canonical_descriptor_bytes();
        assert_eq!(
            selected_descriptor,
            exhaustive.schedule.canonical_descriptor_bytes()
        );
        assert_eq!(
            selected_descriptor,
            repeated.schedule.canonical_descriptor_bytes()
        );
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_search_parallel_generation_is_descriptor_deterministic() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

        let handles = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let policy = policy_of::<D256OneHot>();
                    let domain = RingDimensionSearchDomain::new(
                        policy.ring_dimension,
                        [
                            CommitmentRingDims {
                                inner: 128,
                                outer: 64,
                                opening: 64,
                            },
                            CommitmentRingDims::uniform(64),
                        ],
                    )
                    .expect("mixed dimension domain");
                    find_schedule_with_ring_dimension_domain(
                        PolynomialGroupLayout::singleton(16),
                        &policy,
                        &domain,
                        D256OneHot::ring_challenge_config,
                        D256OneHot::fold_challenge_shape_at_level,
                    )
                    .expect("parallel mixed planner run")
                    .schedule
                    .canonical_descriptor_bytes()
                })
            })
            .collect::<Vec<_>>();
        let descriptors = handles
            .into_iter()
            .map(|handle| handle.join().expect("planner thread"))
            .collect::<Vec<_>>();
        assert!(descriptors.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_nv36_benchmark_policy_selects_minimum_setup_schedule() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

        let policy = policy_of::<D256OneHot>();
        let d64 = CommitmentRingDims::uniform(64);
        let d128_mixed = CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        };
        let d128 = CommitmentRingDims::uniform(128);
        let d256_mixed = CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 128,
        };
        let domain = RingDimensionSearchDomain::new(
            policy.ring_dimension,
            [d64, d128_mixed, d128, d256_mixed],
        )
        .expect("benchmark dimension domain");
        let selected = find_schedule_with_ring_dimension_domain(
            PolynomialGroupLayout::singleton(36),
            &policy,
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .expect("nv36 mixed planner");
        let rank_one_capped_domain =
            RingDimensionSearchDomain::new(policy.ring_dimension, [d64, d128_mixed, d128])
                .expect("rank-one-capped comparison domain");
        let rank_one_capped = find_schedule_with_ring_dimension_domain(
            PolynomialGroupLayout::singleton(36),
            &policy,
            &rank_one_capped_domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .expect("rank-one-capped nv36 planner");
        let selected_root = &selected.schedule.root.params.final_group.commitment;
        let rank_one_capped_root = &rank_one_capped.schedule.root.params.final_group.commitment;

        assert_eq!(selected_root.role_dims(), d256_mixed);
        assert_eq!(
            selected.schedule.recursive_folds[0]
                .params
                .witness
                .role_dims(),
            CommitmentRingDims::uniform(64)
        );
        assert_eq!(
            selected.estimate.estimated_setup_envelope_ring_elements,
            262_144
        );
        assert_eq!(
            selected.estimate.estimated_proof_payload_bytes().unwrap(),
            99_368
        );
        assert_eq!(rank_one_capped_root.inner_commit_matrix.output_rank(), 3);
        assert_eq!(selected_root.inner_commit_matrix.output_rank(), 2);
        assert_eq!(rank_one_capped_root.outer_commit_matrix.output_rank(), 1);
        assert_eq!(selected_root.outer_commit_matrix.output_rank(), 1);
        assert!(
            selected_root.outer_commit_matrix.input_width()
                < rank_one_capped_root.outer_commit_matrix.input_width(),
            "the lower D256 A rank must reduce B width despite both B matrices having rank one"
        );
        assert!(
            selected.estimate.estimated_setup_envelope_ring_elements
                < rank_one_capped
                    .estimate
                    .estimated_setup_envelope_ring_elements,
            "the rank-two D256 candidate must beat the rank-one-capped search on setup"
        );
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_search_requires_a_monotonic_d64_suffix_domain() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

        let policy = policy_of::<D256OneHot>();
        let missing_d64 = RingDimensionSearchDomain::new(
            policy.ring_dimension,
            [CommitmentRingDims::uniform(128)],
        )
        .unwrap();
        let error = find_schedule_with_ring_dimension_domain(
            PolynomialGroupLayout::singleton(16),
            &policy,
            &missing_d64,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires the D64 uniform candidate"));

        let below_d64 = RingDimensionSearchDomain::new(
            policy.ring_dimension,
            [
                CommitmentRingDims::uniform(64),
                CommitmentRingDims {
                    inner: 128,
                    outer: 32,
                    opening: 64,
                },
            ],
        )
        .unwrap();
        let error = find_schedule_with_ring_dimension_domain(
            PolynomialGroupLayout::singleton(16),
            &policy,
            &below_d64,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error.to_string().contains("component-wise at least D64"));
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_search_rejects_direct_multi_chunk_policy() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

        let mut policy = policy_of::<D256OneHot>();
        policy.witness_chunk = akita_types::ChunkedWitnessCfg::d64_production();
        let domain = RingDimensionSearchDomain::new(
            policy.ring_dimension,
            [CommitmentRingDims::uniform(64)],
        )
        .unwrap();
        let error = find_schedule_with_ring_dimension_domain(
            PolynomialGroupLayout::singleton(16),
            &policy,
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not yet support direct multi-chunk planning"));
    }
}
