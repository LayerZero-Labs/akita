//! FoldSchedule planner that finds the global minimum proof size. Recursive
//! grouped scheduling additionally minimizes the first direct setup footprint
//! before proof size.
//!
//! Public entry: [`find_schedule`]. The search is `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, BalancedSignedDigitFoldPolicy,
    FoldWitnessNorms, HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams, SisTableKey,
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
#[cfg(all(test, feature = "catalog-gen"))]
mod exhaustive_oracle;
mod mixed_search;
mod setup_score;
mod suffix_dp;
#[cfg(feature = "test-support")]
pub(crate) mod test_support;

pub use candidate::suffix_opening_layout;
pub(crate) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
    scalar_root_fold_level_params_candidate,
};
pub(crate) use setup_score::{
    level_setup_field_elements, terminal_setup_field_elements, MixedScore,
};
pub(crate) use suffix_dp::{derive_optimal_suffix_schedule, ScheduleMemo, SuffixCtx, SuffixState};

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
/// The planner policy's ring dimension defines only the implicit uniform
/// domain used by [`find_schedule`]. Mixed-dimension search supplies this
/// explicit set of schedule-owned A/B/D tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingDimensionSearchDomain {
    candidates: Vec<CommitmentRingDims>,
}

impl RingDimensionSearchDomain {
    /// Construct and canonicalize a non-empty dimension domain.
    ///
    /// Every tuple must satisfy the schedule-local A-carrier invariant.
    pub fn new(
        candidates: impl IntoIterator<Item = CommitmentRingDims>,
    ) -> Result<Self, AkitaError> {
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
        }
        Ok(Self { candidates })
    }

    /// Construct the explicit singleton domain used by a uniform policy.
    pub fn uniform(ring_dimension: usize) -> Result<Self, AkitaError> {
        Self::new([CommitmentRingDims::uniform(ring_dimension)])
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }

    fn is_uniform_policy_domain(&self, policy: &PlannerPolicy) -> bool {
        self.candidates.as_slice() == [CommitmentRingDims::uniform(policy.uniform_ring_dimension)]
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduleCandidate {
    pub(crate) first_direct_setup_field_len: Option<usize>,
    pub(crate) total_bytes: usize,
    pub(crate) setup_field_elements: usize,
    pub(crate) folds: Vec<CandidateFoldStep>,
    pub(crate) terminal: CandidateTerminalResponse,
}

impl ScheduleCandidate {
    pub(crate) fn first_fold_params(&self) -> Option<&CommittedGroupParams> {
        self.folds.first().map(|fold| &fold.params)
    }

    pub(crate) fn first_direct_setup_field_len_or_max(&self) -> usize {
        self.first_direct_setup_field_len.unwrap_or(usize::MAX)
    }
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
    if prefix.d_setup() == 0 || !n_prefix.is_multiple_of(prefix.d_setup()) {
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
        prefix.d_setup(),
        n_prefix / prefix.d_setup(),
    ))
}

pub(crate) fn materialize_candidate_schedule(
    cached_total: usize,
    cached_num_setup_field_elements: usize,
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
        estimated_num_setup_field_elements: cached_num_setup_field_elements,
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
    let recomputed_setup_field_elements =
        akita_types::setup_matrix_field_elements_for_schedule(&schedule)?;
    if recomputed_setup_field_elements != cached_num_setup_field_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup field count {cached_num_setup_field_elements} disagrees with materialized \
             count {recomputed_setup_field_elements}"
        )));
    }
    estimate.selected_offload_edges = schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.incoming_setup_prefix.is_some())
        .count();
    Ok(PlannedFoldSchedule { schedule, estimate })
}

fn candidate_schedule_descriptor_bytes(choice: &ScheduleCandidate) -> Result<Vec<u8>, AkitaError> {
    Ok(materialize_candidate_schedule(
        choice.total_bytes,
        choice.setup_field_elements,
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
    if policy.max_num_setup_field_elements == 0 {
        return Err(AkitaError::InvalidSetup(
            "maximum setup field capacity must be positive".to_string(),
        ));
    }
    for (label, dimension) in [
        ("uniform", policy.uniform_ring_dimension),
        (
            "setup-prefix inner",
            policy.setup_prefix_inner_ring_dimension,
        ),
    ] {
        if !dimension.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(format!(
                "planner {label} ring dimension must be a nonzero power of two"
            )));
        }
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

/// Find the optimal schedule for a root schedule lookup key and dimension domain.
///
/// A singleton domain preserves the uniform proof-payload objective. An
/// explicit mixed domain selects exact physical setup fields first and exact
/// proof payload second.
///
/// The result is a pure,
/// deterministic function of `(policy, key, dimensions)` (plus the `ring_challenge_config` /
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
    honest_fold_policy: HonestFoldPolicySpec,
    dimensions: &RingDimensionSearchDomain,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate()?;
    validate_policy(policy)?;
    if dimensions.is_uniform_policy_domain(policy) {
        return find_schedule_inner(
            key,
            policy,
            honest_fold_policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }
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
    let suffix_dimensions = CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION);
    if !dimensions.candidates().contains(&suffix_dimensions) {
        return Err(AkitaError::InvalidSetup(format!(
            "mixed-D search requires the D{MIXED_SEARCH_SUFFIX_RING_DIMENSION} uniform candidate \
             used from fold level {MIXED_SEARCH_FOLD_LEVELS} onward"
        )));
    }
    if dimensions.candidates().iter().any(|dims| {
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
    mixed_search::find_schedule(
        key,
        policy,
        honest_fold_policy,
        dimensions,
        ring_challenge_config,
        fold_challenge_shape_at_level,
    )
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
    /// `policy.uniform_ring_dimension`).
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
/// `start_level - 1`) down to a cleartext terminal, at `policy.uniform_ring_dimension`.
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
    let ring_challenge_cfg = ring_challenge_config(policy.uniform_ring_dimension)?;
    let ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &ring_challenge_cfg,
        ring_challenge_config: &ring_challenge_config,
        fold_challenge_shape_at_level: &fold_challenge_shape_at_level,
        num_vars,
        key: PolynomialGroupLayout::singleton(num_vars),
        setup_field_budget: None,
        root_lookup_key: None,
        root_honest_fold_policy: None,
        precommitted_honest_fold_policies: &[],
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

fn find_schedule_inner(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape = &fold_challenge_shape_at_level;

    let default_ring_challenge_cfg = ring_challenge_config(policy.uniform_ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &default_ring_challenge_cfg,
        ring_challenge_config,
        fold_challenge_shape_at_level: fold_shape,
        num_vars: key.num_vars(),
        key,
        setup_field_budget: None,
        root_lookup_key: None,
        root_honest_fold_policy: None,
        precommitted_honest_fold_policies: &[],
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
    let mut best: Option<ScheduleCandidate> = None;
    let fold_challenge_shape = fold_shape(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len: witness_len,
    });
    let mut memo = ScheduleMemo::new();

    // Chunk count of the witness committed at the root fold (absolute level 0).
    let root_num_chunks = policy.chunks_at_level(0);
    let (min_log_basis, max_log_basis) = policy.log_basis_search_range_at_level(0);
    let candidate_dimensions = CommitmentRingDims::uniform(policy.uniform_ring_dimension);
    for candidate_log_basis in min_log_basis..=max_log_basis {
        let mut root_candidates = Vec::new();
        {
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
                    candidate_dimensions,
                    key.num_vars(),
                    key.num_polynomials(),
                    candidate_log_basis,
                    block_index_bits,
                    fold_challenge_shape,
                    honest_fold_policy,
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
            // Root projection is governed by the candidate's committed A dimension,
            // not the uniform planner default.
            let Ok(eor_bytes) = extension_opening_reduction_level_bytes(
                policy.decomposition.field_bits() * policy.chal_ext_degree as u32,
                policy.claim_ext_degree,
                0,
                key,
                witness_len,
                candidate_dimensions.d_a(),
            ) else {
                continue;
            };
            let suffix = derive_optimal_suffix_schedule(
                &suffix_ctx,
                &mut memo,
                SuffixState {
                    level: 1,
                    current_witness_len: output_witness_len,
                    current_lb: candidate_log_basis,
                    incoming_setup_prefix: None,
                },
                0,
            )?;
            if suffix.is_empty() {
                continue;
            }
            // A supported root must recurse into at least one suffix fold.
            for suffix_fold in suffix.best_by_payload_per_lb.values() {
                let next_witness_binding = if suffix_fold.folds.is_empty() {
                    akita_types::NextWitnessBindingPolicy::TerminalInnerState
                } else {
                    akita_types::NextWitnessBindingPolicy::OuterCommitment
                };
                let root_proof_size = level_proof_bytes(
                    field_bits,
                    field_bits * policy.chal_ext_degree as u32,
                    &candidate_params,
                    suffix_fold.first_fold_params(),
                    output_witness_len,
                    Some(next_witness_binding),
                )? + eor_bytes;
                let total = root_proof_size + suffix_fold.total_bytes;
                let root_envelope = level_setup_field_elements(&candidate_params)?;
                let setup_envelope = root_envelope.max(suffix_fold.setup_field_elements);
                let mut folds = Vec::with_capacity(1 + suffix_fold.folds.len());
                folds.push(CandidateFoldStep {
                    params: candidate_params.clone(),
                    input_witness_len: witness_len,
                    output_witness_len,
                    estimated_direct_payload_bytes: root_proof_size,
                    estimated_stage3_payload_bytes: 0,
                });
                folds.extend(suffix_fold.folds.iter().cloned());
                let candidate = ScheduleCandidate {
                    first_direct_setup_field_len: None,
                    total_bytes: total,
                    setup_field_elements: setup_envelope,
                    folds,
                    terminal: suffix_fold.terminal.clone(),
                };
                let replace = match &best {
                    None => true,
                    Some(current) => candidate.total_bytes < current.total_bytes,
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
        best.setup_field_elements,
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
        let domain = RingDimensionSearchDomain::new([
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
        ])
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
        assert!(RingDimensionSearchDomain::new([CommitmentRingDims {
            inner: 64,
            outer: 128,
            opening: 64
        }])
        .is_err());
        assert!(RingDimensionSearchDomain::new([CommitmentRingDims::uniform(256)]).is_ok());
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
        let domain = RingDimensionSearchDomain::new(dimensions).unwrap();
        let key = PolynomialGroupLayout::singleton(16);
        let selected = find_schedule(
            key,
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let selected_score = (
            selected.estimate.estimated_num_setup_field_elements,
            selected.estimate.estimated_proof_payload_bytes().unwrap(),
        );

        let uniform = RingDimensionSearchDomain::new([dimensions[0]]).unwrap();
        let candidate = find_schedule(
            key,
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &uniform,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        assert!(
            selected_score
                <= (
                    candidate.estimate.estimated_num_setup_field_elements,
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
        let reversed_with_duplicate = RingDimensionSearchDomain::new([a128, d64, a128]).unwrap();
        let canonical = RingDimensionSearchDomain::new([d64, a128]).unwrap();
        let key = PolynomialGroupLayout::singleton(16);

        let selected = find_schedule(
            key,
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &reversed_with_duplicate,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let exhaustive = exhaustive_oracle::find_schedule(
            key,
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &canonical,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let repeated = find_schedule(
            key,
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &canonical,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();

        assert_eq!(
            (
                selected.estimate.estimated_num_setup_field_elements,
                selected.estimate.estimated_proof_payload_bytes().unwrap(),
            ),
            (
                exhaustive.estimate.estimated_num_setup_field_elements,
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
                    let domain = RingDimensionSearchDomain::new([
                        CommitmentRingDims {
                            inner: 128,
                            outer: 64,
                            opening: 64,
                        },
                        CommitmentRingDims::uniform(64),
                    ])
                    .expect("mixed dimension domain");
                    find_schedule(
                        PolynomialGroupLayout::singleton(16),
                        &policy,
                        D256OneHot::root_honest_fold_policy(),
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
    fn mixed_root_prices_eor_at_candidate_a_dimension() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};
        let mut policy = policy_of::<D256OneHot>();
        // D256 enables root projection at this width while the D64 candidate does not.
        policy.claim_ext_degree = 64;
        let candidate_dimensions = CommitmentRingDims::uniform(64);
        let domain =
            RingDimensionSearchDomain::new([candidate_dimensions]).expect("mixed dimension domain");
        let key = PolynomialGroupLayout::singleton(16);
        let selected = find_schedule(
            key,
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .expect("mixed planner boundary schedule");
        let schedule = &selected.schedule;
        let root_params = &schedule.root.params.final_group.commitment;
        assert_eq!(root_params.role_dims(), candidate_dimensions);

        let challenge_field_bits =
            policy.decomposition.field_bits() * policy.chal_ext_degree as u32;
        let candidate_eor_bytes = extension_opening_reduction_level_bytes(
            challenge_field_bits,
            policy.claim_ext_degree,
            0,
            key,
            schedule.root.input_witness_len,
            candidate_dimensions.d_a(),
        )
        .expect("candidate EOR bytes");
        let uniform_default_eor_bytes = extension_opening_reduction_level_bytes(
            challenge_field_bits,
            policy.claim_ext_degree,
            0,
            key,
            schedule.root.input_witness_len,
            policy.uniform_ring_dimension,
        )
        .expect("uniform-default EOR bytes");
        assert_eq!(candidate_eor_bytes, 0);
        assert!(uniform_default_eor_bytes > 0);

        let next_params = schedule
            .recursive_folds
            .first()
            .map(|step| &step.params.witness);
        let next_binding = if next_params.is_some() {
            akita_types::NextWitnessBindingPolicy::OuterCommitment
        } else {
            akita_types::NextWitnessBindingPolicy::TerminalInnerState
        };
        let root_without_eor = level_proof_bytes(
            policy.decomposition.field_bits(),
            challenge_field_bits,
            root_params,
            next_params,
            schedule.root.output_witness_len,
            Some(next_binding),
        )
        .expect("root bytes without EOR");
        assert_eq!(
            selected.estimate.estimated_root_direct_payload_bytes,
            root_without_eor + candidate_eor_bytes,
        );
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
        let domain = RingDimensionSearchDomain::new([d64, d128_mixed, d128, d256_mixed])
            .expect("benchmark dimension domain");
        let selected = find_schedule(
            PolynomialGroupLayout::singleton(36),
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .expect("nv36 mixed planner");
        let rank_one_capped_domain = RingDimensionSearchDomain::new([d64, d128_mixed, d128])
            .expect("rank-one-capped comparison domain");
        let mut comparison_policy = policy;
        comparison_policy.max_num_setup_field_elements = usize::MAX;
        let rank_one_capped = find_schedule(
            PolynomialGroupLayout::singleton(36),
            &comparison_policy,
            D256OneHot::root_honest_fold_policy(),
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
            selected.estimate.estimated_num_setup_field_elements,
            45_088_768
        );
        assert_eq!(
            selected.estimate.estimated_proof_payload_bytes().unwrap(),
            98_728
        );
        assert_eq!(rank_one_capped_root.inner_commit_matrix.output_rank(), 3);
        assert_eq!(selected_root.inner_commit_matrix.output_rank(), 1);
        assert_eq!(rank_one_capped_root.outer_commit_matrix.output_rank(), 1);
        assert_eq!(selected_root.outer_commit_matrix.output_rank(), 1);
        assert!(
            selected_root.outer_commit_matrix.input_width()
                < rank_one_capped_root.outer_commit_matrix.input_width(),
            "the lower D256 A rank must reduce B width despite both B matrices having rank one"
        );
        assert!(
            selected.estimate.estimated_num_setup_field_elements
                < rank_one_capped.estimate.estimated_num_setup_field_elements,
            "the rank-two D256 candidate must beat the rank-one-capped search on setup"
        );
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_search_requires_a_monotonic_d64_suffix_domain() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};
        let policy = policy_of::<D256OneHot>();
        let missing_d64 =
            RingDimensionSearchDomain::new([CommitmentRingDims::uniform(128)]).unwrap();
        let error = find_schedule(
            PolynomialGroupLayout::singleton(16),
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &missing_d64,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("requires the D64 uniform candidate"));

        let below_d64 = RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 32,
                opening: 64,
            },
        ])
        .unwrap();
        let error = find_schedule(
            PolynomialGroupLayout::singleton(16),
            &policy,
            D256OneHot::root_honest_fold_policy(),
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
        let domain = RingDimensionSearchDomain::new([CommitmentRingDims::uniform(64)]).unwrap();
        let error = find_schedule(
            PolynomialGroupLayout::singleton(16),
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("does not yet support direct multi-chunk planning"));
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_search_validates_key_and_policy_at_entry() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};
        let policy = policy_of::<D256OneHot>();
        let domain = RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims::uniform(policy.uniform_ring_dimension),
        ])
        .unwrap();

        let error = find_schedule(
            PolynomialGroupLayout::new(16, 0),
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("opening group layouts must be nonempty"));

        let mut invalid_policy = policy;
        invalid_policy.max_num_setup_field_elements = 0;
        let error = find_schedule(
            PolynomialGroupLayout::singleton(16),
            &invalid_policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("maximum setup field capacity must be positive"));
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn mixed_search_applies_setup_budget_in_physical_fields() {
        use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};
        let mut policy = policy_of::<D256OneHot>();
        let domain = RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
        ])
        .unwrap();
        let selected = find_schedule(
            PolynomialGroupLayout::singleton(16),
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let exact_fields =
            akita_types::setup_matrix_field_elements_for_schedule(&selected.schedule).unwrap();
        policy.max_num_setup_field_elements = exact_fields - 1;

        let error = find_schedule(
            PolynomialGroupLayout::singleton(16),
            &policy,
            D256OneHot::root_honest_fold_policy(),
            &domain,
            D256OneHot::ring_challenge_config,
            D256OneHot::fold_challenge_shape_at_level,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no mixed-D schedule"));
    }

    #[cfg(feature = "catalog-gen")]
    #[test]
    fn preserved_recursive_proof_size_is_documented() {
        use akita_config::{
            policy_of, proof_optimized::fp128::D64OneHot, CommitmentConfig,
            RecursiveCommitmentConfig,
        };
        use akita_types::{AkitaScheduleLookupKey, CommittedGroupProfile};

        type Recursive = RecursiveCommitmentConfig<D64OneHot>;
        let precommit_layout = PolynomialGroupLayout::singleton(16);
        let precommit_policy = policy_of::<D64OneHot>();
        let precommit_domain =
            RingDimensionSearchDomain::uniform(precommit_policy.uniform_ring_dimension).unwrap();
        let precommit = find_schedule(
            precommit_layout,
            &precommit_policy,
            D64OneHot::root_honest_fold_policy(),
            &precommit_domain,
            D64OneHot::ring_challenge_config,
            D64OneHot::fold_challenge_shape_at_level,
        )
        .unwrap();
        let descriptor = CommittedGroupProfile::from_params(
            precommit_layout,
            &precommit.schedule.root.params.final_group.commitment,
        );
        let key = AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![descriptor, descriptor],
        };
        let precommitted_honest_fold_policies = vec![
            D64OneHot::root_honest_fold_policy(),
            D64OneHot::root_honest_fold_policy(),
        ];
        let planned = crate::find_group_batch_schedule(
            &key,
            Recursive::root_honest_fold_policy(),
            &precommitted_honest_fold_policies,
            &policy_of::<Recursive>(),
            Recursive::ring_challenge_config,
            Recursive::fold_challenge_shape_at_level,
        )
        .unwrap();

        assert_eq!(
            planned.estimate.estimated_proof_payload_bytes().unwrap(),
            102_732
        );
    }
}
