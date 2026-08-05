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
#[cfg(all(test, feature = "catalog-gen"))]
use akita_types::extension_opening_reduction_level_bytes;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, BalancedSignedDigitFoldPolicy,
    FoldWitnessNorms, HonestFoldPolicy, HonestFoldPolicySpec, HonestFoldSizingQuery,
    InnerCommitMatrixParams, OpenCommitMatrixParams, OuterCommitMatrixParams,
};
use akita_types::{
    level_proof_bytes, padded_setup_prefix_len, try_extension_opening_reduction_level_bytes,
    AkitaScheduleInputs, CommitmentRingDims, CommittedGroupParams, CommittedGroupProfile,
    DecompositionParams, FoldSchedule, FoldScheduleEstimate, PlannedFoldSchedule,
    PolynomialGroupLayout, PrecommittedLevelParams, RecursiveFoldParams, RecursiveFoldStep,
    RootFinalChallenge, RootFinalGroupParams, RootFoldParams, RootFoldStep,
    RootPrecommittedGroupParams, TerminalFoldParams, TerminalFoldStep, TerminalResponseShape,
    WitnessLayout, WitnessPartition,
};

use crate::PlannerPolicy;

mod candidate;
mod mixed_search;
mod objective;
mod setup_score;
mod suffix_dp;
#[cfg(feature = "test-support")]
pub(crate) mod test_support;
#[cfg(test)]
mod unpruned_search;

pub use akita_types::suffix_opening_layout;
pub(crate) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
    planned_next_witness_len, scalar_root_fold_level_params_candidate,
};
pub(crate) use objective::select_complete_candidate;
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
/// The planner policy's uniform ring dimension defines only the implicit
/// singleton domain used by [`find_schedule`]. Mixed-dimension search supplies
/// this explicit set of schedule-owned A/B/D tuples.
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
            dims.validate_role_projection()?;
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

    fn validate_for_policy(&self, policy: &PlannerPolicy) -> Result<(), AkitaError> {
        if self.candidates.as_slice() != policy.ring_dimension_candidates {
            if policy.ring_dimension_candidates.len() == 1 {
                return Ok(());
            }
            return Err(AkitaError::InvalidSetup(
                "ring-dimension search domain disagrees with the catalog-bound policy domain"
                    .to_string(),
            ));
        }
        Ok(())
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

    pub(crate) fn direct_frontier_score(&self) -> (usize, usize) {
        (self.total_bytes, self.setup_field_elements)
    }

    pub(crate) fn recursive_setup_frontier_score(&self) -> (usize, usize, usize) {
        (
            self.first_direct_setup_field_len_or_max(),
            self.total_bytes,
            self.setup_field_elements,
        )
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
    let challenge_field_bits = policy.challenge_field_bits()?;
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

pub(crate) fn candidate_schedule_descriptor_bytes(
    choice: &ScheduleCandidate,
) -> Result<Vec<u8>, AkitaError> {
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
    policy.challenge_field_bits()?;
    let expected_selection_policy = if policy.recursive_setup_planning {
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload
    } else if policy.ring_dimension_candidates.len() > 1 {
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
    } else {
        match policy.selection_policy {
            crate::SelectionPolicyId::MinEstimatedProofPayload
            | crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
                policy.selection_policy
            }
            crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => {
                crate::SelectionPolicyId::MinEstimatedProofPayload
            }
        }
    };
    if policy.selection_policy != expected_selection_policy {
        return Err(AkitaError::InvalidSetup(
            "planner selection policy disagrees with recursive setup capability".to_string(),
        ));
    }
    if policy.setup_field_budget == Some(0) {
        return Err(AkitaError::InvalidSetup(
            "explicit setup field budget must be positive".to_string(),
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

/// Offline canonical standalone precommit descriptor for one group.
///
/// Runtime config code consumes generated [`CommittedGroupProfile`] rows
/// instead of running this search. The returned profile freezes only group-local
/// A/B commitment geometry; grouped-root schedule expansion later derives D/open
/// metadata when the final group is known.
pub fn derive_standalone_precommit_profile(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<CommittedGroupProfile, AkitaError> {
    key.validate()?;
    let mut direct_policy = policy.direct_only();
    direct_policy.basis_range = (direct_policy.basis_range.0, direct_policy.basis_range.0);
    validate_policy(&direct_policy)?;

    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("precommit witness too large".into()))?;
    let requested_fold_shape = fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars: key.num_vars(),
        level: 0,
        input_witness_len: witness_len,
    });
    let field_bits = direct_policy.decomposition.field_bits();
    let (min_log_basis, max_log_basis) = direct_policy.log_basis_search_range_at_level(0);
    let mut best: Option<(usize, CommittedGroupParams)> = None;

    for candidate_log_basis in min_log_basis..=max_log_basis {
        for dimensions in direct_policy.ring_dimension_candidates.iter().copied() {
            let Ok(ring_challenge_cfg) = ring_challenge_config(dimensions.d_a()) else {
                continue;
            };
            let alpha = (dimensions.d_a() as u32).trailing_zeros() as usize;
            let reduced_vars = key.num_vars().saturating_sub(alpha);
            if reduced_vars == 0 {
                continue;
            }
            let min_block_index_bits = if reduced_vars >= 3 { 1 } else { 0 };
            let max_block_index_bits = (reduced_vars - 1).min(usize::BITS as usize - 1);
            for block_index_bits in (min_block_index_bits..=max_block_index_bits).rev() {
                let Some(candidate_params) = scalar_root_fold_level_params_candidate(
                    &direct_policy,
                    &ring_challenge_cfg,
                    dimensions,
                    key.num_vars(),
                    key.num_polynomials(),
                    candidate_log_basis,
                    block_index_bits,
                    requested_fold_shape,
                    honest_fold_policy,
                )?
                else {
                    continue;
                };
                let Some(next_witness_len) = planned_next_witness_len(
                    field_bits,
                    &candidate_params,
                    key.num_polynomials(),
                    direct_policy.chunks_at_level(0),
                )?
                else {
                    continue;
                };
                match &best {
                    Some((best_len, _)) if *best_len <= next_witness_len => {}
                    _ => best = Some((next_witness_len, candidate_params)),
                }
            }
        }
    }

    best.map(|(_, params)| CommittedGroupProfile::from_params(key, &params))
        .ok_or_else(|| {
            AkitaError::InvalidSetup(format!(
                "no standalone precommit profile found for layout {key:?} under this policy"
            ))
        })
}

// Suffix-DP depth cap. Schedules in our working parameter range never need
// more than this many recursive fold levels; deeper search only blows up
// memo state without changing emitted tables.
pub(crate) const MAX_RECURSION_DEPTH: usize = 12;

/// Find the optimal scalar schedule using the dimension domain bound in
/// `policy`.
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
pub(crate) fn find_schedule_singular(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate()?;
    validate_policy(policy)?;
    let dimensions =
        RingDimensionSearchDomain::new(policy.ring_dimension_candidates.iter().copied())?;
    dimensions.validate_for_policy(policy)?;
    if dimensions.is_uniform_policy_domain(policy) {
        if policy.selection_policy
            == crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
        {
            return Err(AkitaError::InvalidSetup(
                "setup-field mixed selection requires an explicit mixed dimension domain"
                    .to_string(),
            ));
        }
        return find_schedule_inner(
            key,
            policy,
            honest_fold_policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        );
    }
    if policy.selection_policy
        != crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
    {
        return Err(AkitaError::InvalidSetup(
            "mixed-D search requires MinSetupMatrixFieldElementsThenProofPayload".into(),
        ));
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
        &dimensions,
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

/// Boundary state for an independently planned recursive suffix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuffixPlanStart {
    /// Absolute fold level of the first suffix fold.
    pub level: usize,
    /// Field-element witness length entering the suffix.
    pub witness_len: usize,
    /// Predecessor fold's `log_basis` lower bound.
    pub log_basis: u32,
    /// Monotone compression phase after the retained predecessor prefix.
    pub payload_phase: akita_types::CommitmentPayloadPhase,
}

/// Plan the proof-size-optimal recursive suffix that folds a witness of
/// `start_witness_len` field elements (produced by some predecessor fold at
/// `start_level - 1`) down to a cleartext terminal, at `policy.uniform_ring_dimension`.
///
/// This is the exact suffix DP [`find_schedule`] runs after choosing a root,
/// exposed so callers can splice an optimal suffix onto a differently sized
/// predecessor — e.g. a mixed ring-dimension-per-level schedule whose root
/// folds at a larger ring dimension than the suffix. `start.log_basis` is the
/// predecessor level's `log_basis` (fold `log_basis` is non-decreasing), and
/// `start.payload_phase` is the monotone compression phase after the retained
/// predecessor prefix. A raw predecessor must use
/// [`akita_types::CommitmentPayloadPhase::RawSuffix`] so the independent suffix
/// cannot resume compression. `num_vars` is the opening arity (used for the
/// singleton opening layout the suffix prices against).
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
    start: SuffixPlanStart,
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
        setup_field_budget: policy.setup_field_budget,
        root_lookup_key: None,
        root_honest_fold_policy: None,
        precommitted_honest_fold_policies: &[],
        level_zero_is_root: false,
    };
    let mut memo = ScheduleMemo::new();
    let result = derive_optimal_suffix_schedule(
        &ctx,
        &mut memo,
        SuffixState {
            level: start.level,
            current_witness_len: start.witness_len,
            current_lb: start.log_basis,
            incoming_setup_prefix: None,
            payload_phase: start.payload_phase,
        },
        0,
    )?;
    let best = result
        .best_by_payload_per_lb
        .values()
        .min_by_key(|suffix| suffix.direct_frontier_score())
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(format!(
                "no terminating suffix for witness_len={} at level {}",
                start.witness_len, start.level
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
        setup_field_budget: policy.setup_field_budget,
        root_lookup_key: None,
        root_honest_fold_policy: Some(honest_fold_policy),
        precommitted_honest_fold_policies: &[],
        level_zero_is_root: true,
    };

    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let witness_len = 1usize
        .checked_shl(key.num_vars() as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("witness too large".into()))?;

    let mut memo = ScheduleMemo::new();
    let suffix = derive_optimal_suffix_schedule(
        &suffix_ctx,
        &mut memo,
        SuffixState {
            level: 0,
            current_witness_len: witness_len,
            current_lb: 0,
            incoming_setup_prefix: None,
            payload_phase: akita_types::CommitmentPayloadPhase::CompressedPrefix,
        },
        0,
    )?;
    let best = match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => {
            select_complete_candidate(policy, suffix.best_by_payload_per_lb.values())?
        }
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
            return Err(AkitaError::UnsupportedSchedule(
                "mixed ring-dimension selection is not supported for singular schedules"
                    .to_string(),
            ));
        }
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload => {
            select_complete_candidate(policy, suffix.best_by_first_direct_setup_per_lb.values())?
        }
    };

    let Some(best) = best.cloned() else {
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

mod tests;
