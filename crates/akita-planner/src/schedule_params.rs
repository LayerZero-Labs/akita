//! FoldSchedule planner that finds the global minimum proof size. Recursive
//! grouped scheduling additionally minimizes the first direct setup footprint
//! before proof size.
//!
//! Scalar schedule search behind the public grouped gate. The search is
//! `Cfg`-free: every
//! per-preset input is carried by the plain-value [`PlannerPolicy`] plus
//! the `ring_challenge_config` / `fold_challenge_shape_at_level` closures,
//! exactly the shape generated catalog emission consumes. This keeps the DP a
//! pure function of `(policy, key, dimension domain)` for offline table generation.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
#[cfg(test)]
use akita_types::extension_opening_reduction_level_bytes;
use akita_types::sis::{
    decomposed_s_block_ring_count, decomposed_t_ring_count, decomposed_w_ring_count,
    fold_witness_digit_plan, num_digits_inner, num_digits_open, num_digits_setup_prefix_commit,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, FoldChallengeNorms,
    FoldWitnessLinfCapConfig, FoldWitnessNorms, InnerCommitMatrixParams, OpenCommitMatrixParams,
    OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{
    level_proof_bytes, padded_setup_prefix_len, try_extension_opening_reduction_level_bytes,
    AkitaScheduleInputs, CommitmentRingDims, CommittedGroupParams, DecompositionParams,
    FoldSchedule, FoldScheduleEstimate, OpeningClaimsLayout, PlannedFoldSchedule,
    PolynomialGroupLayout, PrecommittedGroupDescriptor, PrecommittedLevelParams,
    RecursiveFoldParams, RecursiveFoldStep, RootFinalChallenge, RootFinalGroupParams,
    RootFoldParams, RootFoldStep, RootPrecommittedGroupParams, RootSource, TerminalFoldParams,
    TerminalFoldStep, TerminalResponseShape, WitnessLayout, WitnessPartition,
};

use crate::PlannerPolicy;

mod candidate;
mod mixed_search;
mod setup_score;
mod suffix_dp;
#[cfg(feature = "test-support")]
#[path = "schedule_params/tests/test_support.rs"]
pub(crate) mod test_support;
#[cfg(all(test, feature = "catalog-gen"))]
#[path = "schedule_params/tests/unpruned_search.rs"]
mod unpruned_search;

pub use candidate::suffix_opening_layout;
pub(crate) use candidate::{
    derive_candidate_level_params, derive_candidate_level_params_all_splits,
    planned_next_witness_len, scalar_root_fold_level_params_candidate,
};
pub(crate) use mixed_search::find_schedule_mixed_ring;
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
/// `PlannerPolicy::ring_dimension` remains the setup generation dimension and
/// the implicit singleton domain used by scalar schedule search. This separate
/// offline-only value makes mixed-D search opt-in without changing runtime
/// policy or existing catalog identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingDimensionSearchDomain {
    setup_generation_dimension: usize,
    candidates: Vec<CommitmentRingDims>,
}

impl RingDimensionSearchDomain {
    /// Construct and canonicalize a non-empty dimension domain.
    ///
    /// Every tuple must satisfy the native role-projection invariant, and every role
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
            dims.validate_role_projection()?;
            for d in [dims.d_a(), dims.d_b(), dims.d_d()] {
                if !setup_generation_dimension.is_multiple_of(d) {
                    return Err(AkitaError::InvalidSetup(format!(
                        "candidate dimension D{d} does not divide setup generation dimension \
                         D{setup_generation_dimension}"
                    )));
                }
            }
        }
        Ok(Self {
            setup_generation_dimension,
            candidates,
        })
    }

    /// Construct the explicit singleton domain used by a uniform policy.
    pub fn uniform(setup_generation_dimension: usize) -> Result<Self, AkitaError> {
        Self::new(
            setup_generation_dimension,
            [CommitmentRingDims::uniform(setup_generation_dimension)],
        )
    }

    pub(crate) fn for_policy(policy: &PlannerPolicy) -> Result<Self, AkitaError> {
        Self::new(
            policy.ring_dimension,
            policy.ring_dimension_candidates.iter().copied(),
        )
    }

    /// Canonically ordered admitted A/B/D tuples.
    pub fn candidates(&self) -> &[CommitmentRingDims] {
        &self.candidates
    }

    /// Setup generation dimension against which this domain was validated.
    pub fn setup_generation_dimension(&self) -> usize {
        self.setup_generation_dimension
    }

    fn validate_for_policy(&self, policy: &PlannerPolicy) -> Result<(), AkitaError> {
        if self.setup_generation_dimension != policy.ring_dimension {
            return Err(AkitaError::InvalidSetup(format!(
                "ring-dimension domain uses setup generation D{}, but policy uses D{}",
                self.setup_generation_dimension, policy.ring_dimension
            )));
        }
        if self.candidates.as_slice() != policy.ring_dimension_candidates {
            return Err(AkitaError::InvalidSetup(
                "ring-dimension search domain disagrees with the catalog-bound policy domain"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn is_uniform_policy_domain(&self, policy: &PlannerPolicy) -> bool {
        self.candidates.as_slice() == [CommitmentRingDims::uniform(policy.ring_dimension)]
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
    if prefix.d_setup == 0 || !n_prefix.is_multiple_of(prefix.d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix field length does not align with its ring dimension".to_string(),
        ));
    }
    let challenge_field_bits = policy.challenge_field_bits()?;
    Ok(akita_types::proof_size::stage3_setup_product_bytes(
        challenge_field_bits,
        prefix.d_setup,
        n_prefix / prefix.d_setup,
    ))
}

pub(crate) fn materialize_candidate_schedule(
    cached_total: usize,
    cached_setup_field_elements: usize,
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
    if setup_generation_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup generation dimension must be nonzero".into(),
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
        estimated_setup_envelope_ring_elements: cached_setup_field_elements
            .div_ceil(setup_generation_dimension),
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
                    source: RootSource::from_commitment(&root.params),
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
                open_commit_matrix: root.params.open_commit_matrix.clone(),
                sparse_challenge_config: root.params.fold_challenge_config,
                witness_partition: if root.params.witness_chunk.num_chunks == 1 {
                    WitnessPartition::Single
                } else {
                    WitnessPartition::Distributed {
                        num_chunks: root.params.witness_chunk.num_chunks,
                    }
                },
            },
            input_witness_len: root.input_witness_len,
            output_witness_len: root.output_witness_len,
        },
        recursive_folds: folds
            .into_iter()
            .map(|fold| RecursiveFoldStep {
                params: RecursiveFoldParams {
                    open_commit_matrix: fold.params.open_commit_matrix.clone(),
                    sparse_challenge_config: fold.params.fold_challenge_config,
                    incoming_setup_prefix: fold.params.setup_prefix.clone(),
                    witness_partition: if fold.params.witness_chunk.num_chunks == 1 {
                        WitnessPartition::Single
                    } else {
                        WitnessPartition::Distributed {
                            num_chunks: fold.params.witness_chunk.num_chunks,
                        }
                    },
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
    if recomputed_setup_field_elements != cached_setup_field_elements {
        return Err(AkitaError::InvalidSetup(format!(
            "cached setup field count {cached_setup_field_elements} disagrees with materialized \
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
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope
    } else if policy.ring_dimension_candidates.len() > 1 {
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
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

fn componentwise_dimensions_at_most(
    dimensions: CommitmentRingDims,
    ceiling: CommitmentRingDims,
) -> bool {
    dimensions.d_a() <= ceiling.d_a()
        && dimensions.d_b() <= ceiling.d_b()
        && dimensions.d_d() <= ceiling.d_d()
}

pub(crate) fn find_schedule_singular(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
    fold_challenge_shape_at_level: impl Fn(AkitaScheduleInputs) -> TensorChallengeShape,
) -> Result<PlannedFoldSchedule, AkitaError> {
    key.validate()?;
    validate_policy(policy)?;
    if policy.recursive_setup_planning {
        return Err(AkitaError::InvalidSetup(
            "recursive setup planning requires the grouped-batch scheduler".to_string(),
        ));
    }
    let ring_challenge_config: RingChallengeConfigFn<'_> = &ring_challenge_config;
    let fold_shape = &fold_challenge_shape_at_level;

    let default_ring_challenge_cfg = ring_challenge_config(policy.ring_dimension)?;
    let suffix_ctx = SuffixCtx {
        policy,
        default_ring_challenge_cfg: &default_ring_challenge_cfg,
        ring_challenge_config,
        fold_challenge_shape_at_level: fold_shape,
        num_vars: key.num_vars(),
        key,
        setup_field_budget: None,
        root_lookup_key: None,
        level_zero_is_root: true,
    };
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
        },
        0,
    )?;
    let best = match policy.selection_policy {
        crate::SelectionPolicyId::MinEstimatedProofPayload => suffix
            .best_by_payload_per_lb
            .values()
            .min_by_key(|candidate| candidate.total_bytes),
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope => suffix
            .best_by_first_direct_setup_per_lb
            .values()
            .min_by_key(|candidate| {
                (
                    candidate.first_direct_setup_field_len_or_max(),
                    candidate.total_bytes,
                )
            }),
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload => {
            return Err(AkitaError::UnsupportedSchedule(
                "mixed ring-dimension selection is not supported for singular schedules"
                    .to_string(),
            ));
        }
    }
    .cloned();

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
        policy.ring_dimension,
        best.first_direct_setup_field_len,
        best.folds,
        best.terminal,
    )
}

#[cfg(test)]
mod tests;
