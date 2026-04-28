use super::config::{
    compute_num_digits_fold, compute_num_digits_full_field, num_digits_for_bound,
    optimal_m_r_split_with_params, CommitmentConfig, DecompositionParams,
};
use super::generated::{
    table_entry, GeneratedDirectWitnessShape, GeneratedFoldStep, GeneratedScheduleKey,
    GeneratedScheduleTable, GeneratedStep,
};
use super::schedule_planner::{
    cached_dp_best_basis, cached_dp_suffix_bytes, dp_suffix_plan, PlannerConfig, PlannerState,
};
use crate::error::HachiError;
use crate::planner::digit_math::compute_num_digits_fold_with_claims;
use crate::primitives::serialization::{Compress, HachiSerialize};
use crate::protocol::params::{AjtaiKeyParams, LevelParams};
#[cfg(test)]
use crate::protocol::proof::LevelProofShape;
use crate::protocol::proof::{
    DirectWitnessShape, FlatRingVec, HachiLevelProof, HachiStage1Proof, HachiStage1StageProof,
    HachiStage2Proof,
};
use crate::protocol::ring_switch::w_ring_element_count_with_batch_summary;
use crate::protocol::sumcheck::hachi_stage1_tree::stage1_tree_stage_shapes;
use crate::protocol::sumcheck::{
    CompressedUniPoly, EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof,
};
use crate::FieldCore;
use std::fmt::Write;

/// Public inputs that deterministically select one level's active Hachi params.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HachiScheduleInputs {
    /// Root polynomial variable count.
    pub max_num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub current_w_len: usize,
}

/// Aggregate root-batching context relevant to runtime schedule selection.
///
/// The current batched root path depends on aggregate counts rather than the
/// exact partition:
/// - `num_claims`: total flattened root claims/proofs `y_ell`
/// - `num_commitment_groups`: number of committed root groups
/// - `num_points`: number of distinct opening points
/// - `num_claims` controls concatenated root witness width and batch-effective
///   `B/D` security sizing
/// - `num_points` controls only the public y rows and serialized `y_ring`
///   objects carried by the root proof
///
/// Future schedule-table lookup should key on this summary unless later tests
/// demonstrate that additional batch-shape detail affects runtime behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HachiRootBatchSummary {
    /// Total number of flattened root claims.
    pub num_claims: usize,
    /// Number of committed root groups.
    pub num_commitment_groups: usize,
    /// Number of distinct opening points.
    pub num_points: usize,
}

impl HachiRootBatchSummary {
    /// Singleton root-opening context.
    pub const fn singleton() -> Self {
        Self {
            num_claims: 1,
            num_commitment_groups: 1,
            num_points: 1,
        }
    }

    /// Build a validated batch summary from aggregate counts.
    ///
    /// # Errors
    ///
    /// Returns an error if any count is zero or if groups/points exceed the
    /// total claim count.
    pub fn new(
        num_claims: usize,
        num_commitment_groups: usize,
        num_points: usize,
    ) -> Result<Self, HachiError> {
        if num_claims == 0 {
            return Err(HachiError::InvalidInput(
                "root batching requires at least one claim".to_string(),
            ));
        }
        if num_commitment_groups == 0 {
            return Err(HachiError::InvalidInput(
                "root batching requires at least one commitment group".to_string(),
            ));
        }
        if num_points == 0 {
            return Err(HachiError::InvalidInput(
                "root batching requires at least one opening point".to_string(),
            ));
        }
        if num_commitment_groups > num_claims {
            return Err(HachiError::InvalidInput(format!(
                "root batching has {num_commitment_groups} commitment groups but only {num_claims} claims"
            )));
        }
        if num_points > num_claims {
            return Err(HachiError::InvalidInput(format!(
                "root batching has {num_points} opening points but only {num_claims} claims"
            )));
        }
        Ok(Self {
            num_claims,
            num_commitment_groups,
            num_points,
        })
    }

    /// Derive a batch summary from claim-group sizes and opening-point count.
    ///
    /// # Errors
    ///
    /// Returns an error if the claim-group list is empty, contains an empty
    /// group, overflows the total claim count, or does not admit the requested
    /// number of opening points.
    pub fn from_claim_group_sizes(
        claim_group_sizes: &[usize],
        num_points: usize,
    ) -> Result<Self, HachiError> {
        if claim_group_sizes.is_empty() {
            return Err(HachiError::InvalidInput(
                "root batching requires at least one commitment group".to_string(),
            ));
        }
        if let Some(group_idx) = claim_group_sizes.iter().position(|&size| size == 0) {
            return Err(HachiError::InvalidInput(format!(
                "root batching group {group_idx} must be nonempty"
            )));
        }
        let num_claims = claim_group_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                HachiError::InvalidInput("root batching total claim count overflow".to_string())
            })
        })?;
        Self::new(num_claims, claim_group_sizes.len(), num_points)
    }
}

/// Planner-facing root envelope.
///
/// This keeps the current homogeneous API unchanged while giving planner/cache
/// lookups a stable place to hang future mixed-family root metadata.
/// Recursive suffix planning currently depends on the actual recursive state,
/// but root lookup/reporting still benefits from carrying the normalized root
/// coefficient bounds alongside aggregate batch counts.
///
/// Full same-proof mixed-family batching would still need broader protocol
/// changes outside the planner:
/// - batched input and hint types that retain per-claim family metadata
/// - root layout derivation driven by that normalized family envelope
/// - root witness/relation construction that can aggregate mixed-family claims
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HachiBatchPlanningEnvelope {
    /// Aggregate root opening-batch summary for the concrete invocation.
    pub batch: HachiRootBatchSummary,
    /// Normalized bound for root commitment coefficients.
    pub root_log_commit_bound: u32,
    /// Normalized bound for root opening coefficients.
    pub root_log_open_bound: u32,
}

impl HachiBatchPlanningEnvelope {
    /// Build a normalized planner envelope from explicit root bounds.
    pub const fn new(
        batch: HachiRootBatchSummary,
        root_log_commit_bound: u32,
        root_log_open_bound: u32,
    ) -> Self {
        Self {
            batch,
            root_log_commit_bound,
            root_log_open_bound,
        }
    }

    /// Current homogeneous root envelope derived from `Cfg`.
    pub fn homogeneous<Cfg: CommitmentConfig>(batch: HachiRootBatchSummary) -> Self {
        let decomp = Cfg::decomposition();
        Self {
            batch,
            root_log_commit_bound: decomp.log_commit_bound,
            root_log_open_bound: decomp.log_open_bound.unwrap_or(decomp.log_commit_bound),
        }
    }

    /// Singleton homogeneous root envelope derived from `Cfg`.
    pub fn singleton<Cfg: CommitmentConfig>() -> Self {
        Self::homogeneous::<Cfg>(HachiRootBatchSummary::singleton())
    }
}

/// Public runtime key that selects a concrete root schedule context.
///
/// This is intentionally narrower than a full schedule table entry: it records
/// only the public inputs that pick a root plan, not the resulting plan data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HachiScheduleLookupKey {
    /// Setup/public schedule bucket.
    pub max_num_vars: usize,
    /// Actual root polynomial arity.
    pub num_vars: usize,
    /// Number of claims the root commitment layout was sized for at commit
    /// time. This can exceed `batch.num_claims`.
    pub layout_num_claims: usize,
    /// Aggregate opening-batch summary for the concrete invocation.
    pub batch: HachiRootBatchSummary,
}

impl HachiScheduleLookupKey {
    /// Singleton root-opening context.
    pub const fn singleton(max_num_vars: usize, num_vars: usize, layout_num_claims: usize) -> Self {
        Self {
            max_num_vars,
            num_vars,
            layout_num_claims,
            batch: HachiRootBatchSummary::singleton(),
        }
    }

    /// General root-opening context.
    pub const fn with_batch(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: HachiRootBatchSummary,
    ) -> Self {
        Self {
            max_num_vars,
            num_vars,
            layout_num_claims,
            batch,
        }
    }
}

pub(crate) const fn generated_schedule_lookup_key(
    key: HachiScheduleLookupKey,
) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        max_num_vars: key.max_num_vars,
        num_vars: key.num_vars,
        layout_num_claims: key.layout_num_claims,
        batch_num_claims: key.batch.num_claims,
        batch_num_commitment_groups: key.batch.num_commitment_groups,
        batch_num_points: key.batch.num_points,
    }
}

/// Canonical runtime context for the root Hachi level.
///
/// This captures the currently split root decisions in one place:
/// - `root_layout` is the per-polynomial commitment layout chosen at commit
///   time, parameterized by the setup's supported batch capacity.
/// - `level_layout` is the actual root-level runtime layout after scaling for
///   the concrete opening batch represented by `batch`.
/// - `next_inputs` / `next_level_params` reflect the real recursive handoff
///   taken by the runtime from the current root basis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiRootRuntimePlan {
    /// Setup/public schedule bucket that selects the root family policy.
    pub max_num_vars: usize,
    /// Actual root polynomial arity.
    pub num_vars: usize,
    /// Number of claims the root commitment layout was sized for.
    pub layout_num_claims: usize,
    /// Aggregate opening-batch summary for this root invocation.
    pub batch: HachiRootBatchSummary,
    /// Normalized planner envelope carried into recursive miss-path planning.
    pub planning_envelope: HachiBatchPlanningEnvelope,
    /// Public inputs for the root level.
    pub inputs: HachiScheduleInputs,
    /// Actual runtime root-level layout after scaling for `batch`.
    pub level_lp: LevelParams,
    /// Active root params under the root layout's log_basis.
    pub root_lp: LevelParams,
    /// Public inputs for the first recursive level after the root fold.
    pub next_inputs: HachiScheduleInputs,
    /// Active params for the first recursive level, respecting the current
    /// root basis when selecting the next basis.
    pub next_level_params: LevelParams,
    /// Full exact schedule for the root and recursive suffix when the generated
    /// table has an entry for this runtime key.
    pub exact_plan: Option<HachiSchedulePlan>,
}

impl HachiRootRuntimePlan {
    /// Exact root lookup key for this runtime context.
    pub const fn lookup_key(&self) -> HachiScheduleLookupKey {
        HachiScheduleLookupKey::with_batch(
            self.max_num_vars,
            self.num_vars,
            self.layout_num_claims,
            self.batch,
        )
    }

    /// Recursive witness length after the root fold.
    pub fn next_w_len(&self) -> usize {
        self.next_inputs.current_w_len
    }

    /// Shape of the serialized root proof body for this runtime context.
    #[cfg(test)]
    pub(crate) fn level_proof_shape(&self) -> LevelProofShape {
        let rounds = sumcheck_rounds(self.root_lp.ring_dimension, self.next_w_len());
        let b = 1usize << self.level_lp.log_basis;
        LevelProofShape {
            y_ring_coeffs: self.batch.num_points * self.root_lp.ring_dimension,
            v_coeffs: self.root_lp.d_key.row_len() * self.root_lp.ring_dimension,
            stage1_stages: stage1_tree_stage_shapes(rounds, b),
            stage2_sumcheck: (rounds, 3),
            next_commit_coeffs: self.next_level_params.b_key.row_len()
                * self.next_level_params.ring_dimension,
        }
    }

    /// Exact bytes of the serialized root proof body for this runtime context.
    pub fn level_proof_bytes<Cfg: CommitmentConfig>(&self) -> usize {
        level_proof_bytes(
            field_bits(Cfg::decomposition()),
            &self.root_lp,
            &self.level_lp,
            &self.next_level_params,
            self.next_w_len(),
            self.batch.num_points,
        )
    }
}

/// Derive the canonical runtime context for a root opening from a caller-
/// supplied per-polynomial root layout.
///
/// This is the internal bridge that lets setup sizing, proof-shape logic, and
/// runtime prove/verify all share the same root transition semantics even when
/// a caller is exploring alternate root layouts.
///
/// # Errors
///
/// Returns an error if the batched layout scaling, next witness sizing, or
/// next-level basis selection fails.
pub(crate) fn hachi_root_runtime_plan_from_root_layout<Cfg, const D: usize>(
    key: HachiScheduleLookupKey,
    root_lp: &LevelParams,
) -> Result<HachiRootRuntimePlan, HachiError>
where
    Cfg: CommitmentConfig,
{
    let planning_envelope = HachiBatchPlanningEnvelope::homogeneous::<Cfg>(key.batch);
    let derivation = derive_batched_root_level_derivation::<Cfg, D>(
        key.max_num_vars,
        root_lp,
        key.batch.num_claims,
    )?;
    let next_w_ring =
        w_ring_element_count_with_batch_summary::<Cfg::Field>(&derivation.level_lp, key.batch);
    let next_w_len = next_w_ring
        .checked_mul(derivation.root_lp.ring_dimension)
        .ok_or_else(|| HachiError::InvalidSetup("root next witness length overflow".to_string()))?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 1,
        current_w_len: next_w_len,
    };
    let next_log_basis = planned_next_log_basis_with_current_basis_and_envelope::<Cfg>(
        key,
        next_inputs,
        derivation.root_lp.log_basis,
        planning_envelope,
    )?;
    hachi_root_runtime_plan_from_root_layout_with_next_log_basis::<Cfg, D>(
        key,
        root_lp,
        next_log_basis,
    )
}

pub(crate) fn hachi_root_runtime_plan_from_root_layout_with_next_log_basis<Cfg, const D: usize>(
    key: HachiScheduleLookupKey,
    root_lp: &LevelParams,
    next_log_basis: u32,
) -> Result<HachiRootRuntimePlan, HachiError>
where
    Cfg: CommitmentConfig,
{
    let planning_envelope = HachiBatchPlanningEnvelope::homogeneous::<Cfg>(key.batch);
    let derivation = derive_batched_root_level_derivation::<Cfg, D>(
        key.max_num_vars,
        root_lp,
        key.batch.num_claims,
    )?;
    let inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_lp),
    };
    let next_w_ring =
        w_ring_element_count_with_batch_summary::<Cfg::Field>(&derivation.level_lp, key.batch);
    let next_w_len = next_w_ring
        .checked_mul(derivation.root_lp.ring_dimension)
        .ok_or_else(|| HachiError::InvalidSetup("root next witness length overflow".to_string()))?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 1,
        current_w_len: next_w_len,
    };
    let next_level_params = Cfg::level_params_with_log_basis(next_inputs, next_log_basis);

    Ok(HachiRootRuntimePlan {
        max_num_vars: key.max_num_vars,
        num_vars: key.num_vars,
        layout_num_claims: key.layout_num_claims,
        batch: key.batch,
        planning_envelope,
        inputs,
        level_lp: derivation.level_lp,
        root_lp: derivation.root_lp,
        next_inputs,
        next_level_params,
        exact_plan: None,
    })
}

fn with_log_basis(mut decomp: DecompositionParams, log_basis: u32) -> DecompositionParams {
    decomp.log_basis = log_basis;
    decomp
}

pub(crate) fn main_level_decomposition_from_root(
    root_decomp: DecompositionParams,
    log_basis: u32,
) -> DecompositionParams {
    with_log_basis(root_decomp, log_basis)
}

pub(crate) fn recursive_level_decomposition_from_root(
    root_decomp: DecompositionParams,
    log_basis: u32,
) -> DecompositionParams {
    let parent_open = root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound);
    DecompositionParams {
        log_basis,
        log_commit_bound: log_basis,
        log_open_bound: Some(parent_open),
    }
}

pub(crate) fn recursive_level_decomposition<Cfg: CommitmentConfig>(
    lp: &LevelParams,
) -> DecompositionParams {
    recursive_level_decomposition_from_root(Cfg::decomposition(), lp.log_basis)
}

fn layout_from_params(
    m_vars: usize,
    r_vars: usize,
    lp: &LevelParams,
    decomp: DecompositionParams,
    num_ring: usize,
) -> Result<LevelParams, HachiError> {
    let depth_commit = num_digits_for_bound(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = num_digits_for_bound(open_bound, decomp.log_basis);
    let depth_fold = compute_num_digits_fold(r_vars, lp.challenge_l1_mass(), decomp.log_basis);
    lp.with_decomp(
        m_vars,
        r_vars,
        depth_commit,
        depth_open,
        depth_fold,
        num_ring,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Fully planned public data for one Hachi fold level.
pub struct HachiPlannedLevel {
    /// Public inputs that selected this level.
    pub inputs: HachiScheduleInputs,
    /// Active unified level params chosen for this level.
    pub lp: LevelParams,
    /// Public inputs for the next level after folding.
    pub next_inputs: HachiScheduleInputs,
    /// Planned log-basis of the next level.
    pub next_level_log_basis: u32,
    /// `n_b * d` of the next level, used for next_w_commitment shape.
    pub next_commit_coeffs: usize,
    /// Exact bytes contributed by this level to the proof.
    pub level_bytes: usize,
}

impl HachiPlannedLevel {
    /// Public state at the start of this fold level.
    pub fn input_state(&self) -> HachiPlannedState {
        HachiPlannedState {
            level: self.inputs.level,
            current_w_len: self.inputs.current_w_len,
            log_basis: self.lp.log_basis,
        }
    }

    /// Public state reached after this fold level.
    pub fn output_state(&self) -> HachiPlannedState {
        HachiPlannedState {
            level: self.next_inputs.level,
            current_w_len: self.next_inputs.current_w_len,
            log_basis: self.next_level_log_basis,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Public state after a planned prefix of Hachi fold levels.
pub struct HachiPlannedState {
    /// Next level index reached by the plan.
    pub level: usize,
    /// Witness length in field elements at this state.
    pub current_w_len: usize,
    /// Active log-basis for the witness at this state.
    pub log_basis: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Terminal direct packed-witness handoff in a planned opening proof.
pub struct HachiPlannedDirectStep {
    /// Public witness state carried by the direct handoff.
    pub state: HachiPlannedState,
    /// Serialized witness shape carried by the direct handoff.
    pub witness_shape: DirectWitnessShape,
    /// Exact bytes contributed by the packed direct witness.
    pub direct_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Exact current-step execution data recovered from a pinned schedule.
pub struct HachiPlannedLevelExecution {
    /// Planned fold level that matches the current public state.
    pub level: HachiPlannedLevel,
    /// Planned next-level params implied by the following schedule step.
    pub next_level_params: LevelParams,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One step in a planned opening proof.
pub enum HachiPlannedStep {
    /// A Hachi fold level with an explicit next-state handoff.
    Fold(Box<HachiPlannedLevel>),
    /// The terminal packed-witness direct handoff.
    Direct(HachiPlannedDirectStep),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Deterministic level-by-level schedule selected from public inputs.
pub struct HachiSchedulePlan {
    /// Planned opening-proof steps in execution order.
    ///
    /// The final step is always [`HachiPlannedStep::Direct`].
    pub steps: Vec<HachiPlannedStep>,
    /// Total proof bytes excluding the outer proof wrapper.
    pub no_wrapper_bytes: usize,
    /// Total proof bytes in the serialized singleton `HachiBatchedProof`
    /// wire format.
    ///
    /// The singleton batched proof is currently headerless, so this equals
    /// [`Self::no_wrapper_bytes`].
    pub exact_proof_bytes: usize,
}

impl HachiSchedulePlan {
    /// Iterate over all planned fold levels in execution order.
    pub fn fold_levels(&self) -> impl Iterator<Item = &HachiPlannedLevel> + '_ {
        self.steps.iter().filter_map(|step| match step {
            HachiPlannedStep::Fold(level) => Some(level.as_ref()),
            HachiPlannedStep::Direct(_) => None,
        })
    }

    /// Number of planned fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_levels().count()
    }

    /// Return the terminal direct packed-witness handoff.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without a trailing direct step.
    pub fn direct_step(&self) -> &HachiPlannedDirectStep {
        match self
            .steps
            .last()
            .expect("planned schedule always contains at least one step")
        {
            HachiPlannedStep::Direct(step) => step,
            HachiPlannedStep::Fold(_) => {
                panic!("planned schedule must end in a direct packed-witness step")
            }
        }
    }

    /// Return the initial public witness state before any proof steps run.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without any steps.
    pub fn initial_state(&self) -> HachiPlannedState {
        match self
            .steps
            .first()
            .expect("planned schedule always contains at least one step")
        {
            HachiPlannedStep::Fold(level) => level.input_state(),
            HachiPlannedStep::Direct(step) => step.state,
        }
    }

    /// Iterate over the planned witness states after each executed fold prefix.
    pub fn states(&self) -> impl Iterator<Item = HachiPlannedState> + '_ {
        std::iter::once(self.initial_state())
            .chain(self.fold_levels().map(|level| level.output_state()))
    }

    /// Return the public witness state after `prefix_len` fold levels.
    pub fn state_after_prefix(&self, prefix_len: usize) -> Option<HachiPlannedState> {
        if prefix_len == 0 {
            return Some(self.initial_state());
        }
        self.fold_levels()
            .nth(prefix_len - 1)
            .map(HachiPlannedLevel::output_state)
    }

    /// Return the final witness state after all planned Hachi levels.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without a trailing direct step.
    pub fn terminal_state(&self) -> HachiPlannedState {
        self.direct_step().state
    }
}

fn exact_planned_state_index(
    schedule: &HachiSchedulePlan,
    inputs: HachiScheduleInputs,
    log_basis: Option<u32>,
) -> Option<usize> {
    schedule.states().position(|state| {
        state.level == inputs.level
            && state.current_w_len == inputs.current_w_len
            && log_basis.is_none_or(|basis| state.log_basis == basis)
    })
}

pub(crate) fn exact_planned_level_execution<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Result<Option<HachiPlannedLevelExecution>, HachiError> {
    let Some(state_index) = exact_planned_state_index(schedule, inputs, Some(log_basis)) else {
        return Ok(None);
    };
    let Some(current_step) = schedule.steps.get(state_index) else {
        return Ok(None);
    };
    let HachiPlannedStep::Fold(current_level) = current_step else {
        return Ok(None);
    };
    let Some(next_step) = schedule.steps.get(state_index + 1) else {
        return Err(HachiError::InvalidSetup(
            "planned fold step must be followed by another schedule step".to_string(),
        ));
    };
    let next_level_params = match next_step {
        HachiPlannedStep::Fold(next_level) => next_level.lp.clone(),
        HachiPlannedStep::Direct(direct) => {
            let (d, n_b) = match direct.witness_shape {
                DirectWitnessShape::PackedDigits(_) => {
                    let entry_d = current_level.lp.ring_dimension;
                    let entry_nb = current_level.next_commit_coeffs / entry_d;
                    (entry_d, entry_nb)
                }
                DirectWitnessShape::FieldElements(_) => (current_level.lp.ring_dimension, 0),
            };
            LevelParams::params_only(
                d,
                direct.state.log_basis,
                0,
                n_b,
                0,
                Cfg::stage1_challenge_config(d),
            )
        }
    };
    Ok(Some(HachiPlannedLevelExecution {
        level: current_level.as_ref().clone(),
        next_level_params,
    }))
}

fn scheduled_suffix_bytes_from_index(schedule: &HachiSchedulePlan, state_index: usize) -> usize {
    debug_assert!(state_index <= schedule.num_fold_levels());
    schedule
        .fold_levels()
        .skip(state_index)
        .map(|level| level.level_bytes)
        .sum::<usize>()
        + schedule.direct_step().direct_bytes
}

fn generated_direct_witness_shape(shape: GeneratedDirectWitnessShape) -> DirectWitnessShape {
    match shape {
        GeneratedDirectWitnessShape::PackedDigits {
            num_elems,
            bits_per_elem,
        } => DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)),
        GeneratedDirectWitnessShape::FieldElements { num_elems } => {
            DirectWitnessShape::FieldElements(num_elems)
        }
    }
}

fn generated_direct_log_basis<Cfg: CommitmentConfig>(shape: GeneratedDirectWitnessShape) -> u32 {
    match shape {
        GeneratedDirectWitnessShape::PackedDigits { bits_per_elem, .. } => bits_per_elem,
        GeneratedDirectWitnessShape::FieldElements { .. } => Cfg::decomposition().log_basis,
    }
}

fn generated_step_current_w_len(step: &GeneratedStep) -> usize {
    match step {
        GeneratedStep::Fold(level) => level.current_w_len,
        GeneratedStep::Direct(direct) => direct.current_w_len,
    }
}

fn generated_level_params<Cfg: CommitmentConfig>(
    step: GeneratedFoldStep,
    context: &str,
) -> Result<LevelParams, HachiError> {
    let stage1_config = Cfg::stage1_challenge_config(step.d as usize);
    let params = LevelParams::params_only(
        step.d as usize,
        step.log_basis,
        step.n_a as usize,
        step.n_b as usize,
        step.n_d as usize,
        stage1_config,
    );
    if step.challenge_l1_mass != params.challenge_l1_mass() {
        return Err(HachiError::InvalidSetup(format!(
            "generated schedule {context} challenge L1 mass mismatch: pinned={}, runtime={}",
            step.challenge_l1_mass,
            params.challenge_l1_mass()
        )));
    }
    Ok(params)
}

fn schedule_plan_from_generated_entry<Cfg: CommitmentConfig, const D: usize>(
    key: HachiScheduleLookupKey,
    entry: &super::generated::GeneratedScheduleTableEntry,
) -> Result<HachiSchedulePlan, HachiError> {
    let Some(root_step) = entry.steps.first() else {
        return Err(HachiError::InvalidSetup(
            "generated schedule table entry must contain at least one step".to_string(),
        ));
    };
    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?;
    if generated_step_current_w_len(root_step) != expected_root_w_len {
        return Err(HachiError::InvalidSetup(format!(
            "generated root witness length {} does not match key={key:?}",
            generated_step_current_w_len(root_step)
        )));
    }

    let field_bits = field_bits(Cfg::decomposition());
    let mut steps = Vec::with_capacity(entry.steps.len().max(1));
    let mut fold_level = 0usize;

    for (step_index, generated_step) in entry.steps.iter().enumerate() {
        match generated_step {
            GeneratedStep::Fold(level) => {
                let Some(next_generated_step) = entry.steps.get(step_index + 1) else {
                    return Err(HachiError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    )));
                };
                let next_current_w_len = generated_step_current_w_len(next_generated_step);
                if level.next_w_len != next_current_w_len {
                    return Err(HachiError::InvalidSetup(format!(
                        "generated next_w_len mismatch at level {fold_level}: pinned={}, next step={next_current_w_len}",
                        level.next_w_len
                    )));
                }
                let next_log_basis = match next_generated_step {
                    GeneratedStep::Fold(next_level) => next_level.log_basis,
                    GeneratedStep::Direct(direct) => match direct.witness_shape {
                        GeneratedDirectWitnessShape::PackedDigits { bits_per_elem, .. } => {
                            bits_per_elem
                        }
                        GeneratedDirectWitnessShape::FieldElements { .. } => {
                            return Err(HachiError::InvalidSetup(format!(
                                "generated schedule level {fold_level} cannot transition into a field-element direct step"
                            )))
                        }
                    },
                };

                let inputs = HachiScheduleInputs {
                    max_num_vars: key.max_num_vars,
                    level: fold_level,
                    current_w_len: level.current_w_len,
                };
                let next_inputs = HachiScheduleInputs {
                    max_num_vars: key.max_num_vars,
                    level: fold_level + 1,
                    current_w_len: next_current_w_len,
                };
                let params = generated_level_params::<Cfg>(*level, &format!("level {fold_level}"))?;
                let level_decomp = if fold_level == 0 {
                    main_level_decomposition_from_root(Cfg::decomposition(), level.log_basis)
                } else {
                    recursive_level_decomposition_from_root(Cfg::decomposition(), level.log_basis)
                };
                let layout = layout_from_params(
                    level.m_vars as usize,
                    level.r_vars as usize,
                    &params,
                    level_decomp,
                    level.current_w_len / level.d as usize,
                )?;
                let root_is_batched =
                    fold_level == 0 && key.batch != HachiRootBatchSummary::singleton();
                let mut lp = params.with_layout(&layout);
                if root_is_batched {
                    lp.num_digits_fold = level.delta_fold;
                }
                debug_assert_eq!(
                    lp.num_digits_open, level.delta_open,
                    "generated delta_open mismatch at level {fold_level}"
                );
                debug_assert_eq!(
                    lp.num_digits_fold, level.delta_fold,
                    "generated delta_fold mismatch at level {fold_level}"
                );
                debug_assert_eq!(
                    lp.num_digits_commit, level.delta_commit,
                    "generated delta_commit mismatch at level {fold_level}"
                );
                let runtime_next_w_len = if fold_level == 0 {
                    let next_w_ring =
                        w_ring_element_count_with_batch_summary::<Cfg::Field>(&lp, key.batch);
                    next_w_ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        HachiError::InvalidSetup(
                            "generated root next witness length overflow".to_string(),
                        )
                    })?
                } else {
                    planned_next_w_len(field_bits, &lp)
                };
                if runtime_next_w_len != level.next_w_len {
                    return Err(HachiError::InvalidSetup(format!(
                        "generated next_w_len mismatch at level {fold_level}: pinned={}, runtime={runtime_next_w_len}",
                        level.next_w_len
                    )));
                }

                let (next_level_params, next_commit_coeffs) = match next_generated_step {
                    GeneratedStep::Fold(next_level) => {
                        let next_level_params = generated_level_params::<Cfg>(
                            *next_level,
                            &format!("next level {}", fold_level + 1),
                        )?;
                        let coeffs =
                            next_level_params.b_key.row_len() * next_level_params.ring_dimension;
                        (next_level_params, coeffs)
                    }
                    GeneratedStep::Direct(direct) => {
                        let (entry_d, entry_nb) = match (direct.entry_d, direct.entry_nb) {
                            (Some(entry_d), Some(entry_nb)) => (entry_d as usize, entry_nb as usize),
                            (None, None) => (lp.ring_dimension, 0),
                            _ => {
                                return Err(HachiError::InvalidSetup(
                                    "generated direct entry commitment must specify both D and n_b or neither"
                                        .to_string(),
                                ))
                            }
                        };
                        (
                            LevelParams::params_only(
                                entry_d,
                                next_log_basis,
                                0,
                                entry_nb,
                                0,
                                lp.stage1_config.clone(),
                            ),
                            entry_nb * entry_d,
                        )
                    }
                };
                let runtime_level_bytes = if fold_level == 0 {
                    level_proof_bytes(
                        field_bits,
                        &lp,
                        &lp,
                        &next_level_params,
                        next_inputs.current_w_len,
                        key.batch.num_points,
                    )
                } else {
                    hachi_level_proof_bytes(
                        field_bits,
                        &lp,
                        &next_level_params,
                        next_inputs.current_w_len,
                    )
                };

                steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
                    inputs,
                    lp,
                    next_inputs,
                    next_level_log_basis: next_log_basis,
                    next_commit_coeffs,
                    level_bytes: runtime_level_bytes,
                })));
                fold_level += 1;
            }
            GeneratedStep::Direct(direct) => {
                if step_index + 1 != entry.steps.len() {
                    return Err(HachiError::InvalidSetup(
                        "generated direct step must be terminal".to_string(),
                    ));
                }
                let witness_shape = generated_direct_witness_shape(direct.witness_shape);
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                if direct_bytes != direct.direct_bytes {
                    return Err(HachiError::InvalidSetup(format!(
                        "generated direct bytes mismatch at terminal step: pinned={}, runtime={direct_bytes}",
                        direct.direct_bytes
                    )));
                }
                if !matches!(
                    (direct.entry_d, direct.entry_nb),
                    (Some(_), Some(_)) | (None, None)
                ) {
                    return Err(HachiError::InvalidSetup(
                        "generated direct entry commitment must specify both D and n_b or neither"
                            .to_string(),
                    ));
                }

                let state = HachiPlannedState {
                    level: fold_level,
                    current_w_len: direct.current_w_len,
                    log_basis: generated_direct_log_basis::<Cfg>(direct.witness_shape),
                };
                steps.push(HachiPlannedStep::Direct(HachiPlannedDirectStep {
                    state,
                    witness_shape,
                    direct_bytes,
                }));
            }
        }
    }

    let no_wrapper_bytes = steps
        .iter()
        .map(|step| match step {
            HachiPlannedStep::Fold(level) => level.level_bytes,
            HachiPlannedStep::Direct(step) => step.direct_bytes,
        })
        .sum();
    Ok(HachiSchedulePlan {
        steps,
        no_wrapper_bytes,
        exact_proof_bytes: no_wrapper_bytes,
    })
}

pub(crate) fn generated_schedule_plan_from_table<Cfg: CommitmentConfig, const D: usize>(
    key: HachiScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    table_entry(table, generated_schedule_lookup_key(key))
        .map(|entry| schedule_plan_from_generated_entry::<Cfg, D>(key, entry))
        .transpose()
}

fn exact_plan_from_root_and_suffix<Cfg: CommitmentConfig>(
    root_plan: HachiRootRuntimePlan,
    suffix: HachiSchedulePlan,
) -> Result<HachiSchedulePlan, HachiError> {
    let root_level_bytes = root_plan.level_proof_bytes::<Cfg>();
    let next_inputs = root_plan.next_inputs;
    let next_commit_coeffs =
        root_plan.next_level_params.b_key.row_len() * root_plan.next_level_params.ring_dimension;
    let mut steps = Vec::with_capacity(1 + suffix.steps.len());
    steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
        inputs: root_plan.inputs,
        lp: root_plan.level_lp,
        next_inputs,
        next_level_log_basis: root_plan.next_level_params.log_basis,
        next_commit_coeffs,
        level_bytes: root_level_bytes,
    })));
    steps.extend(suffix.steps);

    let no_wrapper_bytes = root_level_bytes
        .checked_add(suffix.no_wrapper_bytes)
        .ok_or_else(|| HachiError::InvalidSetup("schedule byte count overflow".to_string()))?;
    Ok(HachiSchedulePlan {
        steps,
        no_wrapper_bytes,
        exact_proof_bytes: no_wrapper_bytes,
    })
}

fn exact_plan_from_root_and_direct<Cfg: CommitmentConfig>(
    root_plan: HachiRootRuntimePlan,
) -> Result<HachiSchedulePlan, HachiError> {
    let root_level_bytes = root_plan.level_proof_bytes::<Cfg>();
    let direct_state = HachiPlannedState {
        level: root_plan.next_inputs.level,
        current_w_len: root_plan.next_inputs.current_w_len,
        log_basis: root_plan.next_level_params.log_basis,
    };
    let direct_witness_shape =
        DirectWitnessShape::PackedDigits((direct_state.current_w_len, direct_state.log_basis));
    let direct_bytes =
        direct_witness_bytes(field_bits(Cfg::decomposition()), &direct_witness_shape);
    let next_commit_coeffs =
        root_plan.next_level_params.b_key.row_len() * root_plan.next_level_params.ring_dimension;
    let steps = vec![
        HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
            inputs: root_plan.inputs,
            lp: root_plan.level_lp,
            next_inputs: root_plan.next_inputs,
            next_level_log_basis: root_plan.next_level_params.log_basis,
            next_commit_coeffs,
            level_bytes: root_level_bytes,
        })),
        HachiPlannedStep::Direct(HachiPlannedDirectStep {
            state: direct_state,
            witness_shape: direct_witness_shape,
            direct_bytes,
        }),
    ];
    let no_wrapper_bytes = root_level_bytes
        .checked_add(direct_bytes)
        .ok_or_else(|| HachiError::InvalidSetup("schedule byte count overflow".to_string()))?;
    Ok(HachiSchedulePlan {
        steps,
        no_wrapper_bytes,
        exact_proof_bytes: no_wrapper_bytes,
    })
}

fn runtime_stops_after_batched_root(next_w_len: usize, root_w_len: usize) -> bool {
    const MIN_W_LEN_FOR_FOLDING: usize = 4096;
    next_w_len <= MIN_W_LEN_FOR_FOLDING || next_w_len >= root_w_len
}

fn batched_root_direct_suffix_bytes_if_runtime_stops<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
) -> Result<Option<usize>, HachiError> {
    if level != 1 || root_key.batch == HachiRootBatchSummary::singleton() {
        return Ok(None);
    }
    let alpha = Cfg::D.trailing_zeros() as usize;
    let root_w_len = if root_key.num_vars > alpha {
        1usize
            .checked_shl(root_key.num_vars as u32)
            .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?
    } else {
        Cfg::D
    };
    if !runtime_stops_after_batched_root(current_w_len, root_w_len) {
        return Ok(None);
    }
    Ok(Some(direct_witness_bytes(
        field_bits(Cfg::decomposition()),
        &DirectWitnessShape::PackedDigits((current_w_len, current_log_basis)),
    )))
}

#[doc(hidden)]
pub fn exact_schedule_plan_for_lookup_key<Cfg: CommitmentConfig, const D: usize>(
    key: HachiScheduleLookupKey,
) -> Result<HachiSchedulePlan, HachiError> {
    if key == HachiScheduleLookupKey::singleton(key.max_num_vars, key.max_num_vars, 1) {
        let current_w_len = 1usize
            .checked_shl(key.max_num_vars as u32)
            .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?;
        let inputs = HachiScheduleInputs {
            max_num_vars: key.max_num_vars,
            level: 0,
            current_w_len,
        };
        let planning_envelope = HachiBatchPlanningEnvelope::singleton::<Cfg>();
        let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(inputs);
        let direct_log_basis = Cfg::decomposition().log_basis;
        let direct_witness_shape = DirectWitnessShape::FieldElements(current_w_len);
        let direct_bytes =
            direct_witness_bytes(field_bits(Cfg::decomposition()), &direct_witness_shape);
        let mut best_plan = HachiSchedulePlan {
            steps: vec![HachiPlannedStep::Direct(HachiPlannedDirectStep {
                state: HachiPlannedState {
                    level: 0,
                    current_w_len,
                    log_basis: direct_log_basis,
                },
                witness_shape: direct_witness_shape,
                direct_bytes,
            })],
            no_wrapper_bytes: direct_bytes,
            exact_proof_bytes: direct_bytes,
        };

        for root_log_basis in min_log_basis..=max_log_basis {
            let Ok(root_lp) = current_level_layout_with_log_basis::<Cfg>(inputs, root_log_basis)
            else {
                continue;
            };
            let next_w_ring =
                w_ring_element_count_with_batch_summary::<Cfg::Field>(&root_lp, key.batch);
            let Some(next_w_len) = next_w_ring.checked_mul(root_lp.ring_dimension) else {
                return Err(HachiError::InvalidSetup(
                    "root next witness length overflow".to_string(),
                ));
            };
            let next_inputs = HachiScheduleInputs {
                max_num_vars: key.max_num_vars,
                level: 1,
                current_w_len: next_w_len,
            };
            let next_log_basis = dp_best_basis_with_current_basis_and_envelope::<Cfg>(
                next_inputs,
                root_lp.log_basis,
                planning_envelope,
            )?;
            let root_plan = hachi_root_runtime_plan_from_root_layout_with_next_log_basis::<Cfg, D>(
                key,
                &root_lp,
                next_log_basis,
            )?;
            let (suffix_min_log_basis, suffix_max_log_basis) =
                Cfg::log_basis_search_range(next_inputs);
            let cfg = PlannerConfig::from_cfg::<Cfg>(
                key.max_num_vars,
                suffix_min_log_basis,
                suffix_max_log_basis,
            );
            let suffix = dp_suffix_plan::<Cfg>(
                cfg,
                PlannerState {
                    level: next_inputs.level,
                    current_w_len: next_inputs.current_w_len,
                    log_basis: next_log_basis,
                },
            )?;
            let candidate_plan = exact_plan_from_root_and_suffix::<Cfg>(root_plan, suffix)?;
            if candidate_plan.exact_proof_bytes < best_plan.exact_proof_bytes {
                best_plan = candidate_plan;
            }
        }
        return Ok(best_plan);
    }
    let planning_envelope = HachiBatchPlanningEnvelope::homogeneous::<Cfg>(key.batch);
    let root_lp = hachi_batched_root_layout::<Cfg, D>(key.num_vars, key.layout_num_claims)?;
    let derivation = derive_batched_root_level_derivation::<Cfg, D>(
        key.max_num_vars,
        &root_lp,
        key.batch.num_claims,
    )?;
    let inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(&root_lp),
    };
    let next_w_ring =
        w_ring_element_count_with_batch_summary::<Cfg::Field>(&derivation.level_lp, key.batch);
    let next_w_len = next_w_ring
        .checked_mul(derivation.root_lp.ring_dimension)
        .ok_or_else(|| HachiError::InvalidSetup("root next witness length overflow".to_string()))?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 1,
        current_w_len: next_w_len,
    };
    let next_log_basis = dp_best_basis_with_current_basis_and_envelope::<Cfg>(
        next_inputs,
        derivation.root_lp.log_basis,
        planning_envelope,
    )?;
    let root_plan = hachi_root_runtime_plan_from_root_layout_with_next_log_basis::<Cfg, D>(
        key,
        &root_lp,
        next_log_basis,
    )?;
    let direct_plan = exact_plan_from_root_and_direct::<Cfg>(root_plan.clone())?;
    if runtime_stops_after_batched_root(next_inputs.current_w_len, inputs.current_w_len) {
        return Ok(direct_plan);
    }
    let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(next_inputs);
    let cfg = PlannerConfig::from_cfg::<Cfg>(key.max_num_vars, min_log_basis, max_log_basis);
    let suffix = dp_suffix_plan::<Cfg>(
        cfg,
        PlannerState {
            level: next_inputs.level,
            current_w_len: next_inputs.current_w_len,
            log_basis: next_log_basis,
        },
    )?;
    let suffix_plan = exact_plan_from_root_and_suffix::<Cfg>(root_plan, suffix)?;
    Ok(
        if direct_plan.exact_proof_bytes <= suffix_plan.exact_proof_bytes {
            direct_plan
        } else {
            suffix_plan
        },
    )
}

/// Build a schedule plan by simulating the level chain for any
/// `CommitmentConfig` whose basis choices are fully deterministic from
/// public inputs (e.g. static or test configs without generated tables).
///
/// The root fold (level 0) is always emitted because the commitment
/// opening protocol mandates at least one fold before the direct tail.
///
/// `root_layout` must match the layout that `Cfg::commitment_layout()`
/// would return.  The caller supplies it explicitly so that this function
/// never calls `Cfg::commitment_layout()` (which may itself call
/// `Cfg::schedule_plan()`, creating infinite recursion for configs that
/// use the default `commitment_layout` implementation).
pub(crate) fn build_schedule_plan_from_config<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    root_lp: &LevelParams,
) -> Result<HachiSchedulePlan, HachiError> {
    let fb = field_bits(Cfg::decomposition());

    let root_w_len = 1usize
        .checked_shl(max_num_vars as u32)
        .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?;

    let mut steps = Vec::new();
    let mut level = 0usize;
    let mut current_w_len = root_w_len;

    loop {
        let inputs = HachiScheduleInputs {
            max_num_vars,
            level,
            current_w_len,
        };
        let log_basis = Cfg::log_basis_at_level(inputs);
        let lp = if level == 0 {
            let params = Cfg::root_level_params_for_layout_with_log_basis(inputs, root_lp)?;
            params.with_layout(root_lp)
        } else {
            current_level_layout_with_log_basis::<Cfg>(inputs, log_basis)?
        };
        let next_w_len = planned_next_w_len(fb, &lp);

        let next_inputs = HachiScheduleInputs {
            max_num_vars,
            level: level + 1,
            current_w_len: next_w_len,
        };
        let next_log_basis = Cfg::log_basis_at_level(next_inputs);
        let next_level_params = Cfg::level_params_with_log_basis(next_inputs, next_log_basis);

        let continue_bytes = hachi_level_proof_bytes(fb, &lp, &next_level_params, next_w_len);

        let should_stop = level > 0
            && (next_w_len >= current_w_len
                || packed_digits_bytes(current_w_len, log_basis) <= continue_bytes);

        if should_stop {
            let witness_shape = DirectWitnessShape::PackedDigits((current_w_len, log_basis));
            let direct_bytes = direct_witness_bytes(fb, &witness_shape);
            steps.push(HachiPlannedStep::Direct(HachiPlannedDirectStep {
                state: HachiPlannedState {
                    level,
                    current_w_len,
                    log_basis,
                },
                witness_shape,
                direct_bytes,
            }));
            break;
        }

        let next_commit_coeffs =
            next_level_params.b_key.row_len() * next_level_params.ring_dimension;
        steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
            inputs,
            lp,
            next_inputs,
            next_level_log_basis: next_log_basis,
            next_commit_coeffs,
            level_bytes: continue_bytes,
        })));

        level += 1;
        current_w_len = next_w_len;
    }

    let no_wrapper_bytes: usize = steps
        .iter()
        .map(|step| match step {
            HachiPlannedStep::Fold(l) => l.level_bytes,
            HachiPlannedStep::Direct(d) => d.direct_bytes,
        })
        .sum();

    Ok(HachiSchedulePlan {
        steps,
        no_wrapper_bytes,
        exact_proof_bytes: no_wrapper_bytes,
    })
}

pub(crate) fn field_bits(root_decomp: DecompositionParams) -> u32 {
    root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound)
}

pub(super) fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

pub(crate) fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

pub(crate) fn direct_witness_bytes(field_bits: u32, shape: &DirectWitnessShape) -> usize {
    match shape {
        DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            packed_digits_bytes(*num_elems, *bits_per_elem)
        }
        DirectWitnessShape::FieldElements(num_coeffs) => {
            num_coeffs.saturating_mul(field_bytes(field_bits))
        }
    }
}

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

fn stage1_proof_bytes(rounds: usize, b: usize, elem_bytes: usize) -> usize {
    stage1_tree_stage_shapes(rounds, b)
        .into_iter()
        .map(|stage| {
            sumcheck_bytes(rounds, stage.sumcheck.1, elem_bytes) + stage.child_claims * elem_bytes
        })
        .sum::<usize>()
        + elem_bytes
}

/// Compute the number of digits needed when decomposing the `r` polynomial
/// at a recursive level (always full-field, so use asymmetric centering).
pub(crate) fn recursive_r_decomp_levels(field_bits: u32, log_basis: u32) -> usize {
    compute_num_digits_full_field(field_bits, log_basis).max(1)
}

pub(crate) fn planned_w_ring_element_count(field_bits: u32, lp: &LevelParams) -> usize {
    let w_hat_count = lp.num_blocks * lp.num_digits_open;
    let t_hat_count = lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre_count = lp.inner_width() * lp.num_digits_fold;
    let r_count = lp.m_row_count(1, 1) * recursive_r_decomp_levels(field_bits, lp.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

pub(crate) fn planned_next_w_len(field_bits: u32, lp: &LevelParams) -> usize {
    planned_w_ring_element_count(field_bits, lp) * lp.ring_dimension
}

fn sumcheck_rounds(level_d: usize, next_w_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / level_d;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}

pub(crate) fn hachi_level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = proof_ring_vec_bytes(1, lp.ring_dimension, elem_bytes);
    let v_bytes = proof_ring_vec_bytes(lp.d_key.row_len(), lp.ring_dimension, elem_bytes);
    let next_commit_bytes =
        proof_ring_vec_bytes(next_lp.b_key.row_len(), next_lp.ring_dimension, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let b = 1usize << lp.log_basis;
    let stage1_bytes = stage1_proof_bytes(rounds, b, elem_bytes);

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}

fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: (0..rounds)
            .map(|_| CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(); degree],
            })
            .collect(),
    }
}

fn dummy_eq_factored_sumcheck<F: FieldCore>(
    rounds: usize,
    degree: usize,
) -> EqFactoredSumcheckProof<F> {
    EqFactoredSumcheckProof {
        round_polys: (0..rounds)
            .map(|_| EqFactoredUniPoly {
                coeffs_except_linear_term: vec![
                    F::zero();
                    EqFactoredUniPoly::<F>::stored_coeff_count_for_degree(degree)
                ],
            })
            .collect(),
    }
}

fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> HachiStage1Proof<F> {
    HachiStage1Proof {
        stages: stage1_tree_stage_shapes(rounds, b)
            .into_iter()
            .map(|shape| HachiStage1StageProof {
                sumcheck: dummy_eq_factored_sumcheck(rounds, shape.sumcheck.1),
                child_claims: vec![F::zero(); shape.child_claims],
            })
            .collect(),
        s_claim: F::zero(),
    }
}

pub(super) fn exact_recursive_level_proof_bytes<F: FieldCore>(
    lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
) -> Result<usize, HachiError> {
    let current_coeffs = lp
        .d_key
        .row_len()
        .checked_mul(lp.ring_dimension)
        .ok_or_else(|| HachiError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let next_commit_coeffs = next_lp
        .b_key
        .row_len()
        .checked_mul(next_lp.ring_dimension)
        .ok_or_else(|| HachiError::InvalidSetup("recursive proof sizing overflow".to_string()))?;
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let b = 1usize << lp.log_basis;

    let proof = HachiLevelProof {
        y_ring: FlatRingVec::from_coeffs(vec![F::zero(); lp.ring_dimension]),
        v: FlatRingVec::from_coeffs(vec![F::zero(); current_coeffs]),
        stage1: dummy_stage1_proof(rounds, b),
        stage2: HachiStage2Proof {
            sumcheck: dummy_sumcheck(rounds, 3),
            next_w_commitment: FlatRingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
            next_w_eval: F::zero(),
        },
    };
    Ok(proof.serialized_size(Compress::No))
}

pub(crate) fn level_proof_bytes(
    field_bits: u32,
    lp: &LevelParams,
    level_lp: &LevelParams,
    next_lp: &LevelParams,
    next_w_len: usize,
    num_claims: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = proof_ring_vec_bytes(num_claims, lp.ring_dimension, elem_bytes);
    let v_bytes = proof_ring_vec_bytes(lp.d_key.row_len(), lp.ring_dimension, elem_bytes);
    let next_commit_bytes =
        proof_ring_vec_bytes(next_lp.b_key.row_len(), next_lp.ring_dimension, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
    let b = 1usize << level_lp.log_basis;
    let stage1_bytes = stage1_proof_bytes(rounds, b, elem_bytes);

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}

/// Derive the commitment layout for a recursive level at the given log-basis.
///
/// # Errors
///
/// Returns an error if the root or recursive layout derivation fails.
pub fn current_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, HachiError> {
    if inputs.level == 0 {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }
    let params = Cfg::level_params_with_log_basis(inputs, log_basis);
    let layout = hachi_recursive_level_layout_from_params::<Cfg>(&params, inputs.current_w_len)?;
    Ok(params.with_layout(&layout))
}

fn dp_recursive_suffix_bytes_with_log_basis_and_envelope<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<usize, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(inputs);
    let cfg = PlannerConfig::from_cfg::<Cfg>(max_num_vars, min_log_basis, max_log_basis);
    cached_dp_suffix_bytes::<Cfg>(
        cfg,
        planning_envelope,
        PlannerState {
            level,
            current_w_len,
            log_basis: current_log_basis,
        },
    )
}

fn actual_recursive_suffix_bytes_with_log_basis_and_envelope<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<usize, HachiError> {
    if let Some(direct_bytes) = batched_root_direct_suffix_bytes_if_runtime_stops::<Cfg>(
        root_key,
        level,
        current_w_len,
        current_log_basis,
    )? {
        return Ok(direct_bytes);
    }
    dp_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
        root_key.max_num_vars,
        level,
        current_w_len,
        current_log_basis,
        planning_envelope,
    )
}

fn dp_best_basis_with_current_basis_and_envelope<Cfg: CommitmentConfig>(
    next_inputs: HachiScheduleInputs,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<u32, HachiError> {
    let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(next_inputs);
    let lower_bound = current_log_basis.max(min_log_basis);
    let cfg =
        PlannerConfig::from_cfg::<Cfg>(next_inputs.max_num_vars, min_log_basis, max_log_basis);
    cached_dp_best_basis::<Cfg>(cfg, planning_envelope, next_inputs, lower_bound)
        .map(|(log_basis, _)| log_basis)
        .ok_or_else(|| HachiError::InvalidSetup("no valid next-level log basis found".to_string()))
}

fn dp_log_basis_at_level_from_schedule_and_envelope<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    min_log_basis: u32,
    max_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<u32, HachiError> {
    let cfg = PlannerConfig::from_cfg::<Cfg>(inputs.max_num_vars, min_log_basis, max_log_basis);
    cached_dp_best_basis::<Cfg>(cfg, planning_envelope, inputs, min_log_basis)
        .map(|(log_basis, _)| log_basis)
        .ok_or_else(|| HachiError::InvalidSetup("no valid log basis found".to_string()))
}

#[cfg(test)]
pub(crate) fn planned_recursive_suffix_bytes_with_log_basis<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
) -> Result<usize, HachiError> {
    planned_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
        root_key,
        level,
        current_w_len,
        current_log_basis,
        HachiBatchPlanningEnvelope::singleton::<Cfg>(),
    )
}

pub(crate) fn planned_recursive_suffix_bytes_with_log_basis_and_envelope<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<usize, HachiError> {
    if let Some(direct_bytes) = batched_root_direct_suffix_bytes_if_runtime_stops::<Cfg>(
        root_key,
        level,
        current_w_len,
        current_log_basis,
    )? {
        return Ok(direct_bytes);
    }
    if let Some(schedule) = Cfg::schedule_plan(root_key)? {
        return planned_recursive_suffix_bytes_with_log_basis_from_schedule_and_envelope::<Cfg>(
            &schedule,
            root_key.max_num_vars,
            level,
            current_w_len,
            current_log_basis,
            planning_envelope,
        );
    }
    actual_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
        root_key,
        level,
        current_w_len,
        current_log_basis,
        planning_envelope,
    )
}

#[cfg(test)]
pub(crate) fn planned_next_log_basis_with_current_basis<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    next_inputs: HachiScheduleInputs,
    current_log_basis: u32,
) -> Result<u32, HachiError> {
    planned_next_log_basis_with_current_basis_and_envelope::<Cfg>(
        root_key,
        next_inputs,
        current_log_basis,
        HachiBatchPlanningEnvelope::singleton::<Cfg>(),
    )
}

pub(crate) fn planned_next_log_basis_with_current_basis_and_envelope<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    next_inputs: HachiScheduleInputs,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<u32, HachiError> {
    if let Some(schedule) = Cfg::schedule_plan(root_key)? {
        if let Some(next_state_index) = exact_planned_state_index(&schedule, next_inputs, None) {
            if let Some(prev_state) = next_state_index
                .checked_sub(1)
                .and_then(|idx| schedule.state_after_prefix(idx))
            {
                if prev_state.log_basis == current_log_basis {
                    return Ok(schedule
                        .state_after_prefix(next_state_index)
                        .expect("exact planned next-state index must resolve to a state")
                        .log_basis);
                }
            }
        }
        return dp_best_basis_with_current_basis_and_envelope::<Cfg>(
            next_inputs,
            current_log_basis,
            planning_envelope,
        );
    }
    dp_best_basis_with_current_basis_and_envelope::<Cfg>(
        next_inputs,
        current_log_basis,
        planning_envelope,
    )
}

pub(crate) fn planned_log_basis_at_level_from_schedule<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    inputs: HachiScheduleInputs,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<u32, HachiError> {
    planned_log_basis_at_level_from_schedule_and_envelope::<Cfg>(
        schedule,
        inputs,
        min_log_basis,
        max_log_basis,
        HachiBatchPlanningEnvelope::singleton::<Cfg>(),
    )
}

fn planned_log_basis_at_level_from_schedule_and_envelope<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    inputs: HachiScheduleInputs,
    min_log_basis: u32,
    max_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<u32, HachiError> {
    if let Some(state_index) = exact_planned_state_index(schedule, inputs, None) {
        return Ok(schedule
            .state_after_prefix(state_index)
            .expect("exact planned state index must resolve to a state")
            .log_basis);
    }
    dp_log_basis_at_level_from_schedule_and_envelope::<Cfg>(
        inputs,
        min_log_basis,
        max_log_basis,
        planning_envelope,
    )
}

pub(crate) fn planned_schedule_key_from_schedule(
    lookup_key: HachiScheduleLookupKey,
    schedule: &HachiSchedulePlan,
) -> String {
    let mut key = format!(
        "planner_v3_nv{}_poly{}_layout{}_claims{}_groups{}_points{}",
        lookup_key.max_num_vars,
        lookup_key.num_vars,
        lookup_key.layout_num_claims,
        lookup_key.batch.num_claims,
        lookup_key.batch.num_commitment_groups,
        lookup_key.batch.num_points
    );
    for state in schedule.states() {
        let _ = write!(key, "_l{}b{}", state.level, state.log_basis);
    }
    key
}

/// Side-by-side recursive suffix estimates for reporting and regression tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HachiRecursiveSuffixEstimate {
    /// Bytes from the generated schedule table alone.
    pub table_bytes: usize,
    /// Bytes from planning the actual `(level, w_len, log_basis)` state.
    pub actual_state_bytes: usize,
    /// Whether the queried state was an exact generated-schedule hit.
    pub exact_state_match: bool,
    /// Whether the actual-state estimate came from the miss-path local planner.
    pub used_actual_state_planner: bool,
}

/// Compare the generated-table suffix estimate against the actual-state suffix
/// estimate for a specific recursive state.
///
/// # Errors
///
/// Returns an error if the schedule lookup or actual-state planner cannot
/// price the requested recursive suffix.
pub fn recursive_suffix_estimate_with_log_basis<Cfg: CommitmentConfig>(
    root_key: HachiScheduleLookupKey,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<HachiRecursiveSuffixEstimate, HachiError> {
    if let Some(schedule) = Cfg::schedule_plan(root_key)? {
        let inputs = HachiScheduleInputs {
            max_num_vars: root_key.max_num_vars,
            level,
            current_w_len,
        };
        let exact_state_match =
            exact_planned_state_index(&schedule, inputs, Some(current_log_basis)).is_some();
        let table_bytes = planned_recursive_suffix_bytes_with_log_basis_from_schedule::<Cfg>(
            &schedule,
            root_key.max_num_vars,
            level,
            current_w_len,
            current_log_basis,
        )?;
        let actual_state_bytes = actual_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
            root_key,
            level,
            current_w_len,
            current_log_basis,
            planning_envelope,
        )?;
        return Ok(HachiRecursiveSuffixEstimate {
            table_bytes,
            actual_state_bytes,
            exact_state_match,
            used_actual_state_planner: !exact_state_match,
        });
    }

    let actual_state_bytes = actual_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
        root_key,
        level,
        current_w_len,
        current_log_basis,
        planning_envelope,
    )?;
    Ok(HachiRecursiveSuffixEstimate {
        table_bytes: actual_state_bytes,
        actual_state_bytes,
        exact_state_match: false,
        used_actual_state_planner: true,
    })
}

pub(crate) fn planned_recursive_suffix_bytes_from_schedule<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<usize, HachiError> {
    planned_recursive_suffix_bytes_from_schedule_and_envelope::<Cfg>(
        schedule,
        max_num_vars,
        level,
        current_w_len,
        min_log_basis,
        max_log_basis,
        HachiBatchPlanningEnvelope::singleton::<Cfg>(),
    )
}

fn planned_recursive_suffix_bytes_from_schedule_and_envelope<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    min_log_basis: u32,
    max_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<usize, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    if let Some(state_index) = exact_planned_state_index(schedule, inputs, None) {
        return Ok(scheduled_suffix_bytes_from_index(schedule, state_index));
    }
    let current_log_basis = planned_log_basis_at_level_from_schedule_and_envelope::<Cfg>(
        schedule,
        inputs,
        min_log_basis,
        max_log_basis,
        planning_envelope,
    )?;
    dp_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
        max_num_vars,
        level,
        current_w_len,
        current_log_basis,
        planning_envelope,
    )
}

pub(crate) fn planned_recursive_suffix_bytes_with_log_basis_from_schedule<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
) -> Result<usize, HachiError> {
    planned_recursive_suffix_bytes_with_log_basis_from_schedule_and_envelope::<Cfg>(
        schedule,
        max_num_vars,
        level,
        current_w_len,
        current_log_basis,
        HachiBatchPlanningEnvelope::singleton::<Cfg>(),
    )
}

#[allow(clippy::too_many_arguments)]
fn planned_recursive_suffix_bytes_with_log_basis_from_schedule_and_envelope<
    Cfg: CommitmentConfig,
>(
    schedule: &HachiSchedulePlan,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    planning_envelope: HachiBatchPlanningEnvelope,
) -> Result<usize, HachiError> {
    if let Some(state_index) = exact_planned_state_index(
        schedule,
        HachiScheduleInputs {
            max_num_vars,
            level,
            current_w_len,
        },
        Some(current_log_basis),
    ) {
        return Ok(scheduled_suffix_bytes_from_index(schedule, state_index));
    }
    dp_recursive_suffix_bytes_with_log_basis_and_envelope::<Cfg>(
        max_num_vars,
        level,
        current_w_len,
        current_log_basis,
        planning_envelope,
    )
}

/// Derive the root level's active params and layout.
///
/// # Errors
///
/// Returns an error if the root variable split is invalid or overflows.
pub fn hachi_root_level_layout<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<LevelParams, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    };
    Cfg::root_level_layout_with_log_basis(inputs, Cfg::log_basis_at_level(inputs))
}

/// Derive the root commitment layout, allowing a zero-outer direct root.
///
/// Unlike [`hachi_root_level_layout`], this helper is for the commitment
/// surface rather than the fold surface, so it permits tiny roots that fit
/// entirely inside one padded ring element.
///
/// # Errors
///
/// Returns an error if `max_num_vars` underflows `alpha` or if the derived
/// layout overflows.
pub(crate) fn hachi_root_commitment_layout<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<LevelParams, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    };
    let log_basis = Cfg::log_basis_at_level(inputs);
    let alpha = Cfg::d_at_level(0, inputs.current_w_len).trailing_zeros() as usize;
    if max_num_vars > alpha {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }

    let d = Cfg::d_at_level(0, inputs.current_w_len);
    let stage1_config = Cfg::stage1_challenge_config(d);
    let mut params = LevelParams::params_only(d, log_basis, 1, 1, 1, stage1_config);
    let decomp = main_level_decomposition_from_root(Cfg::decomposition(), log_basis);
    for _ in 0..4 {
        let layout = layout_from_params(0, 0, &params, decomp, 0)?;
        let derived_params = Cfg::root_level_params_for_layout_with_log_basis(inputs, &layout)?;
        if (
            derived_params.a_key.row_len(),
            derived_params.b_key.row_len(),
            derived_params.d_key.row_len(),
        ) == (
            params.a_key.row_len(),
            params.b_key.row_len(),
            params.d_key.row_len(),
        ) {
            return Ok(derived_params.with_layout(&layout));
        }
        params = derived_params;
    }
    Err(HachiError::InvalidSetup(format!(
        "failed to converge on tiny-root params for {} at max_num_vars={max_num_vars}",
        std::any::type_name::<Cfg>()
    )))
}

/// Derive a recursive `w`-opening layout from the active level params.
///
/// # Errors
///
/// Returns an error if the witness length is incompatible with `params.d` or if
/// the recursive layout derivation overflows.
pub fn hachi_recursive_level_layout_from_params<Cfg: CommitmentConfig>(
    lp: &LevelParams,
    current_w_len: usize,
) -> Result<LevelParams, HachiError> {
    if !current_w_len.is_multiple_of(lp.ring_dimension) {
        return Err(HachiError::InvalidInput(format!(
            "witness length {current_w_len} is not divisible by D={}",
            lp.ring_dimension
        )));
    }
    let num_ring_elems = current_w_len / lp.ring_dimension;
    let total = num_ring_elems.next_power_of_two().max(1);
    let alpha = lp.ring_dimension.trailing_zeros() as usize;
    let reduced_vars = total.trailing_zeros() as usize;
    let max_num_vars = reduced_vars + alpha;
    let decomp = recursive_level_decomposition::<Cfg>(lp);
    let (m_vars, r_vars) = optimal_m_r_split_with_params(lp, decomp, reduced_vars, num_ring_elems);
    let layout = layout_from_params(m_vars, r_vars, lp, decomp, num_ring_elems)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}

// Ring-native §4.1 commitment layout helpers.
//
// These helpers used to back a `RingCommitmentScheme` trait that materialised
// commitments from explicit `t_hat` layouts. The production flow commits via
// `HachiPolyOps::commit_inner_witness` (see `commitment_scheme.rs`), so only
// the layout-selection helpers remain here.

pub(crate) fn root_current_w_len<const D: usize>(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(D))
        .unwrap_or(0)
}

pub(crate) fn scale_batched_root_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    if num_claims == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }

    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_lp),
    };
    let root_stage1_config =
        Cfg::stage1_challenge_config(Cfg::d_at_level(0, root_inputs.current_w_len));
    let mut scaled = root_lp.clone();
    let d = scaled.ring_dimension;
    // Root batching concatenates the outer binding roles across claims.
    // The inner A role stays per-claim, so only B and D widen here.
    scaled.b_key = AjtaiKeyParams::new(
        scaled.b_key.row_len(),
        root_lp
            .b_key
            .col_len()
            .checked_mul(num_claims)
            .ok_or_else(|| HachiError::InvalidSetup("batched outer width overflow".to_string()))?,
        scaled.b_key.collision_inf(),
        d,
    );
    scaled.d_key = AjtaiKeyParams::new(
        scaled.d_key.row_len(),
        root_lp
            .d_key
            .col_len()
            .checked_mul(num_claims)
            .ok_or_else(|| HachiError::InvalidSetup("batched D width overflow".to_string()))?,
        scaled.d_key.collision_inf(),
        d,
    );
    // `num_claims` amplifies the folded root witness bound. Public point count
    // is handled later when sizing the explicit y rows and serialized y_rings.
    scaled.num_digits_fold = root_lp
        .num_digits_fold
        .max(compute_num_digits_fold_with_claims(
            root_lp.r_vars,
            root_stage1_config.l1_mass(),
            root_lp.log_basis,
            num_claims,
        ));
    Ok(scaled)
}

/// Shared batched-root derivation used by planner and runtime.
///
/// `level_lp` is the batch-effective root layout that widens the `B/D` widths
/// and fold-digit budget for the concrete root batch. `root_lp` is the active
/// root parameter set derived against that widened layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchedRootLevelDerivation {
    pub level_lp: LevelParams,
    pub root_lp: LevelParams,
}

pub(crate) fn derive_batched_root_level_derivation<Cfg, const D: usize>(
    max_num_vars: usize,
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<BatchedRootLevelDerivation, HachiError>
where
    Cfg: CommitmentConfig,
{
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_lp),
    };
    let level_lp = scale_batched_root_layout::<Cfg, D>(max_num_vars, root_lp, num_claims)?;
    let root_lp = Cfg::root_level_params_for_layout_with_log_basis(inputs, &level_lp)?;
    Ok(BatchedRootLevelDerivation { level_lp, root_lp })
}

/// Planner-derived batched root split parameters.
pub(crate) struct BatchedRootSplit {
    /// Per-polynomial root params/layout for the chosen `(log_basis, m, r)`.
    pub params: LevelParams,
}

/// Extract `BatchedRootSplit` from a pre-computed `HachiSchedulePlan`'s
/// first fold level, if one exists.
fn split_from_schedule_plan(plan: &HachiSchedulePlan) -> Option<BatchedRootSplit> {
    let root_level = plan.fold_levels().next()?;
    let per_poly_fold = compute_num_digits_fold(
        root_level.lp.r_vars,
        root_level.lp.challenge_l1_mass(),
        root_level.lp.log_basis,
    );
    let mut lp = root_level.lp.clone();
    lp.num_digits_fold = per_poly_fold;
    Some(BatchedRootSplit { params: lp })
}

pub(crate) fn fallback_batched_root_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(max_num_vars)?;
    let params = if num_claims <= 1 {
        root_lp
    } else {
        scale_batched_root_layout::<Cfg, D>(max_num_vars, &root_lp, num_claims)?
    };
    Ok(BatchedRootSplit { params })
}

fn per_poly_root_split_from_batched_level(
    root_lp: &LevelParams,
    per_poly_fold: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError> {
    if num_claims == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }
    let b_cols = root_lp
        .b_key
        .col_len()
        .checked_div(num_claims)
        .filter(|cols| cols.saturating_mul(num_claims) == root_lp.b_key.col_len())
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "batched root B width {} is not divisible by num_claims={num_claims}",
                root_lp.b_key.col_len()
            ))
        })?;
    let d_cols = root_lp
        .d_key
        .col_len()
        .checked_div(num_claims)
        .filter(|cols| cols.saturating_mul(num_claims) == root_lp.d_key.col_len())
        .ok_or_else(|| {
            HachiError::InvalidSetup(format!(
                "batched root D width {} is not divisible by num_claims={num_claims}",
                root_lp.d_key.col_len()
            ))
        })?;
    let d = root_lp.ring_dimension;
    let mut lp = root_lp.clone();
    lp.b_key = AjtaiKeyParams::new(lp.b_key.row_len(), b_cols, lp.b_key.collision_inf(), d);
    lp.d_key = AjtaiKeyParams::new(lp.d_key.row_len(), d_cols, lp.d_key.collision_inf(), d);
    lp.num_digits_fold = per_poly_fold;
    Ok(BatchedRootSplit { params: lp })
}

/// Find the optimal `(log_basis, m, r)` triple for a batched root opening.
///
/// First checks the pre-computed generated tables.  Falls back to the DP
/// planner only when no table entry exists.
pub(crate) fn optimal_root_batch_split<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<BatchedRootSplit, HachiError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = HachiScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        num_claims,
        HachiRootBatchSummary::new(num_claims, 1, 1)?,
    );
    if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
        if let Some(split) = split_from_schedule_plan(&plan) {
            tracing::info!(
                max_num_vars,
                num_claims,
                total_bytes = plan.exact_proof_bytes,
                root_m = split.params.log_block_len(),
                root_r = split.params.log_num_blocks(),
                root_lb = split.params.log_basis,
                "batched root split: read from pre-computed table"
            );
            return Ok(split);
        }
        let split = fallback_batched_root_split::<Cfg, D>(max_num_vars, 1)?;
        tracing::info!(
            max_num_vars,
            num_claims,
            "batched root split: schedule is direct-only, falling back to config root layout"
        );
        return Ok(split);
    }

    use crate::planner::schedule_params::{find_optimal_schedule, Step, WitnessShape};

    let shape = WitnessShape {
        num_claims,
        num_commitment_groups: 1,
        num_points: 1,
    };
    let schedule = find_optimal_schedule::<Cfg, D>(max_num_vars, shape)?;

    let root_step = match schedule.steps.first() {
        Some(Step::Fold(step)) => step,
        _ => return fallback_batched_root_split::<Cfg, D>(max_num_vars, 1),
    };

    let split = per_poly_root_split_from_batched_level(
        &root_step.params,
        root_step.delta_fold_per_poly,
        num_claims,
    )?;

    tracing::info!(
        max_num_vars,
        num_claims,
        total_bytes = schedule.total_bytes,
        root_m = split.params.log_block_len(),
        root_r = split.params.log_num_blocks(),
        root_lb = split.params.log_basis,
        "batched root split: computed from scratch by DP planner (no pre-computed table)"
    );

    Ok(split)
}

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `max_num_vars` variables.
///
/// When `num_claims <= 1` this returns the singleton layout from
/// [`CommitmentConfig::commitment_layout`]. For larger batches the
/// `m_vars`/`r_vars` split is optimized to minimize proof size.
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn hachi_batched_root_layout<Cfg, const D: usize>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    if num_claims <= 1 {
        return Cfg::commitment_layout(max_num_vars);
    }

    let split = optimal_root_batch_split::<Cfg, D>(max_num_vars, num_claims)?;
    Ok(split.params)
}

#[cfg(test)]
mod commit_tests {
    use crate::primitives::{HachiDeserialize, HachiSerialize};
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::setup::{HachiExpandedSetup, HachiProverSetup, HachiVerifierSetup};
    use crate::test_utils::{TinyConfig, F as TestF};
    use std::sync::Arc;

    #[test]
    fn expanded_setup_roundtrips_and_derives_same_verifier() {
        const TEST_D: usize = 64;
        let prover_setup = HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(16, 3, 1).unwrap();
        let verifier_setup = HachiVerifierSetup {
            expanded: Arc::clone(&prover_setup.expanded),
        };

        let mut bytes = Vec::new();
        prover_setup
            .expanded
            .serialize_compressed(&mut bytes)
            .unwrap();
        let decoded = HachiExpandedSetup::<TestF>::deserialize_compressed(&bytes[..], &()).unwrap();

        assert_eq!(decoded, prover_setup.expanded.as_ref().clone());
        assert_eq!(decoded.seed.max_num_batched_polys, 3);

        let derived_verifier = HachiVerifierSetup {
            expanded: Arc::new(decoded.clone()),
        };
        assert_eq!(derived_verifier, verifier_setup);
    }

    #[test]
    fn setup_accepts_field_coupled_presets() {
        HachiProverSetup::<fp128::Field, 128>::new::<fp128::D128Full>(12, 1, 1)
            .expect("legacy fp128 preset should accept the legacy field");

        HachiProverSetup::<fp128::Field, 128>::new::<fp128::D128Full>(12, 1, 1)
            .expect("default fp128 fixed-D preset should accept the default field");

        HachiProverSetup::<fp128::Field, 32>::new::<fp128::D32Full>(12, 1, 1)
            .expect("small-D fp128 preset should accept the default field");
    }

    #[cfg(feature = "disk-persistence")]
    mod disk_persistence {
        use super::*;
        use crate::protocol::commitment::CommitmentConfig;
        use crate::protocol::setup::{get_storage_path, load_expanded_setup};
        use std::fs;
        use std::sync::{LazyLock, Mutex};

        static DISK_TEST_ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

        fn cleanup_setup_file(max_num_vars: usize) {
            if let Some(path) = get_storage_path::<TinyConfig>(max_num_vars, 1, 1) {
                let _ = fs::remove_file(path);
            }
        }

        fn with_test_cache_dir<T>(test_name: &str, f: impl FnOnce() -> T) -> T {
            let _guard = DISK_TEST_ENV_LOCK.lock().unwrap();
            let cache_root = std::env::temp_dir().join(format!("hachi-disk-tests-{test_name}"));
            fs::create_dir_all(&cache_root).unwrap();

            let old_local_app_data = std::env::var_os("LOCALAPPDATA");
            std::env::set_var("LOCALAPPDATA", &cache_root);
            let out = f();
            match old_local_app_data {
                Some(path) => std::env::set_var("LOCALAPPDATA", path),
                None => std::env::remove_var("LOCALAPPDATA"),
            }
            out
        }

        #[test]
        fn save_and_load_roundtrips() {
            with_test_cache_dir("roundtrip", || {
                const TEST_D: usize = 64;
                const MAX_VARS: usize = 100;

                cleanup_setup_file(MAX_VARS);

                let prover_setup =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                let loaded = load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1, 1).unwrap();
                assert_eq!(loaded, prover_setup.expanded.as_ref().clone());

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn setup_uses_cache_on_second_call() {
            with_test_cache_dir("second-call", || {
                const TEST_D: usize = 64;
                const MAX_VARS: usize = 101;

                cleanup_setup_file(MAX_VARS);

                let first =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                let second =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                assert_eq!(first.expanded, second.expanded);

                cleanup_setup_file(MAX_VARS);
            });
        }

        #[test]
        fn ntt_caches_rebuilt_correctly_from_disk() {
            with_test_cache_dir("ntt-rebuild", || {
                use crate::algebra::CyclotomicRing;
                use crate::protocol::commitment::utils::linear::mat_vec_mul_ntt_single_i8;
                use crate::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};

                const TEST_D: usize = 64;
                const MAX_VARS: usize = 102;

                cleanup_setup_file(MAX_VARS);

                let fresh_setup =
                    HachiProverSetup::<TestF, TEST_D>::new::<TinyConfig>(MAX_VARS, 1, 1).unwrap();

                let loaded_expanded =
                    load_expanded_setup::<TestF, TinyConfig>(MAX_VARS, 1, 1).unwrap();
                let disk_setup =
                    HachiProverSetup::<TestF, TEST_D>::from_expanded(loaded_expanded).unwrap();

                let lp = TinyConfig::commitment_layout(MAX_VARS).unwrap();
                let num_coeffs = lp.num_blocks * lp.block_len;
                let coeffs = vec![CyclotomicRing::<TestF, TEST_D>::zero(); num_coeffs];
                let poly = DensePoly::<TestF, TEST_D>::from_ring_coeffs(coeffs);

                // Commit via the production path on both setups and compare.
                // Both should yield the same `u = B · t_hat` because the
                // disk-loaded expanded setup must rebuild its NTT caches to
                // match the fresh one exactly.
                let commit_u = |setup: &HachiProverSetup<TestF, TEST_D>| {
                    let inner = poly
                        .commit_inner_witness(
                            &setup.expanded.shared_matrix,
                            &setup.ntt_shared,
                            lp.a_key.row_len(),
                            lp.block_len,
                            lp.num_digits_commit,
                            lp.num_digits_open,
                            lp.log_basis,
                            setup.expanded.seed.max_stride,
                        )
                        .unwrap();
                    mat_vec_mul_ntt_single_i8::<TestF, TEST_D>(
                        &setup.ntt_shared,
                        lp.b_key.row_len(),
                        setup.expanded.seed.max_stride,
                        inner.t_hat.flat_digits(),
                    )
                };

                let fresh_u = commit_u(&fresh_setup);
                let disk_u = commit_u(&disk_setup);

                assert_eq!(fresh_u, disk_u);

                cleanup_setup_file(MAX_VARS);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{CyclotomicRing, SparseChallengeConfig};
    use crate::primitives::serialization::{Compress, HachiSerialize};
    use crate::protocol::commitment::generated::{
        fp128_d128_full_table, fp128_d32_full_table, fp128_d32_onehot_table, fp128_d64_full_table,
        fp128_d64_onehot_table, GeneratedScheduleTable,
    };
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::proof::{FlatRingVec, HachiBatchedRootProof};
    use crate::protocol::ring_switch::{
        w_ring_element_count, w_ring_element_count_with_claim_groups,
    };
    use crate::FieldCore;

    type F = fp128::Field;

    fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(max_num_vars: usize) {
        let key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
        let plan = Cfg::schedule_plan(key)
            .expect("planner should succeed")
            .expect("config should provide a planner");
        for level in plan.fold_levels() {
            let runtime_next_w_len =
                w_ring_element_count::<Cfg::Field>(&level.lp) * level.lp.ring_dimension;
            assert_eq!(
                runtime_next_w_len, level.next_inputs.current_w_len,
                "planner/runtime next_w_len mismatch at level {} for max_num_vars={max_num_vars}",
                level.inputs.level
            );
        }
    }

    fn assert_generated_table_matches_cfg_schedule<Cfg: CommitmentConfig, const D: usize>(
        table: GeneratedScheduleTable,
    ) {
        for entry in table.entries {
            let key = HachiScheduleLookupKey::with_batch(
                entry.key.max_num_vars,
                entry.key.num_vars,
                entry.key.layout_num_claims,
                HachiRootBatchSummary::new(
                    entry.key.batch_num_claims,
                    entry.key.batch_num_commitment_groups,
                    entry.key.batch_num_points,
                )
                .expect("generated batch summary"),
            );
            let generated = generated_schedule_plan_from_table::<Cfg, D>(key, table)
                .expect("generated table should materialize")
                .expect("entry should exist in generated table");
            let planned = Cfg::schedule_plan(key)
                .expect("config schedule should succeed")
                .expect("config should provide a generated schedule");
            assert_eq!(
                generated, planned,
                "generated schedule should match cfg-selected schedule for key={key:?}"
            );
        }
    }

    fn assert_exact_root_fold_matches_runtime_root_plan<Cfg: CommitmentConfig, const D: usize>(
        max_num_vars: usize,
    ) {
        let key = HachiScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
        let plan = Cfg::schedule_plan(key)
            .expect("config schedule should succeed")
            .expect("config should provide an exact schedule");
        let planned_root = exact_planned_level_execution::<Cfg>(
            &plan,
            HachiScheduleInputs {
                max_num_vars,
                level: 0,
                current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
            },
            plan.fold_levels()
                .next()
                .expect("exact schedule should begin with a fold")
                .lp
                .log_basis,
        )
        .expect("exact plan should resolve the root fold")
        .expect("exact plan should contain a matching root fold");
        let runtime_root = Cfg::get_params_for_prove::<D>(
            max_num_vars,
            max_num_vars,
            1,
            HachiRootBatchSummary::singleton(),
        )
        .expect("runtime root plan should succeed");
        assert_eq!(
            planned_root.level.inputs.current_w_len,
            runtime_root.inputs.current_w_len,
            "planned/runtime root current_w_len mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.lp,
            runtime_root.level_lp,
            "planned/runtime root lp mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.next_level_params,
            runtime_root.next_level_params,
            "planned/runtime next-level params mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.next_inputs.current_w_len,
            runtime_root.next_inputs.current_w_len,
            "planned/runtime next_w_len mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
    }

    #[test]
    fn generated_fp128_schedule_tables_match_cfg_schedule() {
        assert_generated_table_matches_cfg_schedule::<fp128::D32Full, 32>(fp128_d32_full_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D32OneHot, 32>(
            fp128_d32_onehot_table(),
        );
        assert_generated_table_matches_cfg_schedule::<fp128::D64Full, 64>(fp128_d64_full_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D64OneHot, 64>(
            fp128_d64_onehot_table(),
        );
        assert_generated_table_matches_cfg_schedule::<fp128::D128Full, 128>(fp128_d128_full_table());
    }

    #[test]
    fn generated_d32_full_root_fold_matches_runtime_root_plan() {
        assert_exact_root_fold_matches_runtime_root_plan::<fp128::D32Full, 32>(26);
    }

    #[test]
    fn generated_d128_full_table_materializes_valid_plans() {
        let table = fp128_d128_full_table();
        for entry in table.entries {
            let key = HachiScheduleLookupKey::with_batch(
                entry.key.max_num_vars,
                entry.key.num_vars,
                entry.key.layout_num_claims,
                HachiRootBatchSummary::new(
                    entry.key.batch_num_claims,
                    entry.key.batch_num_commitment_groups,
                    entry.key.batch_num_points,
                )
                .expect("generated batch summary"),
            );
            generated_schedule_plan_from_table::<fp128::D128Full, 128>(key, table)
                .expect("generated table should materialize")
                .expect("entry should exist in generated table");
        }
    }

    #[test]
    fn adaptive_bounded_plan_matches_runtime_next_w_len() {
        for max_num_vars in [14, 20, 30] {
            assert_plan_matches_runtime_w_sizes::<fp128::D128Full>(max_num_vars);
        }
    }

    #[test]
    fn adaptive_onehot_plan_matches_runtime_next_w_len() {
        for max_num_vars in [15, 30, 44] {
            assert_plan_matches_runtime_w_sizes::<fp128::D64OneHot>(max_num_vars);
        }
    }

    #[test]
    fn singleton_root_runtime_plan_matches_existing_root_layout() {
        type Cfg = fp128::D64OneHot;

        let runtime =
            Cfg::get_params_for_prove::<{ Cfg::D }>(30, 30, 1, HachiRootBatchSummary::singleton())
                .expect("singleton runtime plan");
        let root_lp = hachi_root_level_layout::<Cfg>(30).unwrap();

        assert_eq!(runtime.batch, HachiRootBatchSummary::singleton());
        assert_eq!(runtime.level_lp, root_lp);
        assert_eq!(runtime.inputs.level, 0);
        assert_eq!(runtime.next_inputs.level, 1);
        assert_eq!(runtime.level_proof_shape().y_ring_coeffs, Cfg::D);
    }

    #[test]
    fn adaptive_onehot_explicit_recursive_basis_beats_colliding_stateless_state() {
        type Cfg = fp128::D64OneHot;

        let current_inputs = HachiScheduleInputs {
            max_num_vars: 30,
            level: 4,
            current_w_len: 245_888,
        };
        let root_key = HachiScheduleLookupKey::singleton(30, 30, 1);
        let next_log_basis =
            planned_next_log_basis_with_current_basis::<Cfg>(root_key, current_inputs, 5).unwrap();
        let suffix_bytes = planned_recursive_suffix_bytes_with_log_basis::<Cfg>(
            root_key,
            current_inputs.level,
            current_inputs.current_w_len,
            5,
        )
        .unwrap();

        assert_eq!(next_log_basis, 5);
        assert!(suffix_bytes < packed_digits_bytes(current_inputs.current_w_len, 5));
    }

    fn assert_batched_off_table_state_uses_actual_state_planner<Cfg: CommitmentConfig>(
        max_num_vars: usize,
        level: usize,
        current_w_len: usize,
        current_log_basis: u32,
        num_claims: usize,
    ) {
        let estimate = recursive_suffix_estimate_with_log_basis::<Cfg>(
            HachiScheduleLookupKey::with_batch(
                max_num_vars,
                max_num_vars,
                num_claims,
                HachiRootBatchSummary::new(num_claims, 1, 1).expect("same-point batch summary"),
            ),
            level,
            current_w_len,
            current_log_basis,
            HachiBatchPlanningEnvelope::homogeneous::<Cfg>(
                HachiRootBatchSummary::new(num_claims, 1, 1).expect("same-point batch summary"),
            ),
        )
        .expect("recursive suffix estimate");

        assert!(
            !estimate.exact_state_match,
            "batched recursive state should land off the singleton generated path"
        );
        assert!(
            estimate.used_actual_state_planner,
            "off-table batched state should use the actual-state miss-path planner"
        );
    }

    #[test]
    fn batched_d32_onehot_off_table_state_uses_actual_state_planner() {
        assert_batched_off_table_state_uses_actual_state_planner::<fp128::D32OneHot>(
            32, 5, 129_216, 4, 4,
        );
    }

    #[test]
    fn batched_d64_onehot_off_table_state_uses_actual_state_planner() {
        assert_batched_off_table_state_uses_actual_state_planner::<fp128::D64OneHot>(
            32, 5, 87_744, 5, 4,
        );
    }

    #[test]
    fn blessed_d64_onehot_batched_states_use_actual_state_planner() {
        type Cfg = fp128::D64OneHot;
        for batch in [
            HachiRootBatchSummary::new(6, 3, 1).unwrap(),
            HachiRootBatchSummary::new(6, 3, 2).unwrap(),
        ] {
            let root_plan = Cfg::get_params_for_prove::<{ Cfg::D }>(20, 20, 6, batch).unwrap();
            let estimate = recursive_suffix_estimate_with_log_basis::<Cfg>(
                root_plan.lookup_key(),
                root_plan.next_inputs.level,
                root_plan.next_w_len(),
                root_plan.next_level_params.log_basis,
                root_plan.planning_envelope,
            )
            .unwrap();
            assert!(
                !estimate.exact_state_match,
                "blessed batch {batch:?} should remain off-table and use the exact miss-path planner"
            );
            assert!(
                estimate.used_actual_state_planner,
                "blessed batch {batch:?} should use the miss-path planner"
            );
            assert_eq!(
                estimate.table_bytes, estimate.actual_state_bytes,
                "off-table exact suffix accounting should agree with the measured DP fallback for {batch:?}"
            );
        }
    }

    #[test]
    fn recursive_onehot_split_matches_open_digit_witness_count() {
        type Cfg = fp128::D64OneHot;

        let inputs = HachiScheduleInputs {
            max_num_vars: 30,
            level: 1,
            current_w_len: 25_974_272,
        };
        let params = Cfg::level_params(inputs);
        let decomp =
            recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis);
        let num_ring = inputs.current_w_len / params.ring_dimension;
        let lp_12_7 = layout_from_params(12, 7, &params, decomp, num_ring).unwrap();
        let lp_11_8 = layout_from_params(11, 8, &params, decomp, num_ring).unwrap();
        let w_12_7 = planned_w_ring_element_count(field_bits(Cfg::decomposition()), &lp_12_7);
        let w_11_8 = planned_w_ring_element_count(field_bits(Cfg::decomposition()), &lp_11_8);
        let reduced_vars = (inputs.current_w_len / params.ring_dimension)
            .next_power_of_two()
            .trailing_zeros() as usize;

        assert!(w_12_7 < w_11_8);
        assert_eq!(
            optimal_m_r_split_with_params(&params, decomp, reduced_vars, num_ring),
            (12, 7)
        );
    }

    #[test]
    fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_lp = LevelParams::params_only(D, 2, 2, 3, 2, stage1_config.clone());
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(D, log_basis, 2, 2, 2, stage1_config.clone())
                .with_decomp(0, 0, 1, 1, 1, 0)
                .unwrap();
            assert_eq!(
                hachi_level_proof_bytes(128, &lp, &next_lp, next_w_len),
                exact_recursive_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
        use crate::protocol::params::AjtaiKeyParams;
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_lp = LevelParams::params_only(D, 2, 2, 3, 2, stage1_config.clone());
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams {
                ring_dimension: D,
                log_basis,
                a_key: AjtaiKeyParams::new(2, 1, 0, D),
                b_key: AjtaiKeyParams::new(2, 1, 0, D),
                d_key: AjtaiKeyParams::new(2, 1, 0, D),
                num_blocks: 1,
                block_len: 1,
                m_vars: 0,
                r_vars: 0,
                stage1_config: stage1_config.clone(),
                num_digits_commit: 1,
                num_digits_open: 1,
                num_digits_fold: 1,
            };
            let rounds = sumcheck_rounds(D, next_w_len);
            let b = 1usize << log_basis;
            let next_commitment = FlatRingVec::from_ring_elems(&vec![
                CyclotomicRing::<F, D>::zero();
                next_lp.b_key.row_len()
            ])
            .into_compact();
            let num_points = 5;
            let root_proof = HachiBatchedRootProof::new_two_stage::<D>(
                vec![CyclotomicRing::<F, D>::zero(); num_points],
                vec![CyclotomicRing::<F, D>::zero(); lp.d_key.row_len()],
                dummy_stage1_proof(rounds, b),
                dummy_sumcheck(rounds, 3),
                next_commitment,
                F::zero(),
            );

            assert_eq!(
                level_proof_bytes(128, &lp, &lp, &next_lp, next_w_len, num_points),
                root_proof.serialized_size(Compress::No),
                "planned batched root bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn tight_block_len_is_no_larger_than_pow2() {
        for max_num_vars in [14, 20, 30] {
            let plan = fp128::D128Full::schedule_plan(HachiScheduleLookupKey::singleton(
                max_num_vars,
                max_num_vars,
                1,
            ))
            .expect("planner should succeed")
            .expect("config should provide a planner");
            for level in plan.fold_levels() {
                let pow2_block = 1usize << level.lp.m_vars;
                assert!(
                    level.lp.block_len <= pow2_block,
                    "block_len {} should be <= 2^m_vars {} at level {} (num_vars={})",
                    level.lp.block_len,
                    pow2_block,
                    level.inputs.level,
                    max_num_vars
                );
                if level.inputs.level > 0 {
                    let num_ring = level.inputs.current_w_len / level.lp.ring_dimension;
                    let expected_tight = num_ring.div_ceil(level.lp.num_blocks);
                    assert_eq!(
                        level.lp.block_len, expected_tight,
                        "recursive level {} should use tight block_len = ceil({num_ring} / {})",
                        level.inputs.level, level.lp.num_blocks
                    );
                }
            }
        }
    }

    #[test]
    fn root_batch_summary_tracks_only_aggregate_counts() {
        let a = HachiRootBatchSummary::from_claim_group_sizes(&[1, 1, 4], 2).unwrap();
        let b = HachiRootBatchSummary::from_claim_group_sizes(&[2, 2, 2], 2).unwrap();
        let c = HachiRootBatchSummary::from_claim_group_sizes(&[3, 3], 2).unwrap();

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(HachiRootBatchSummary::singleton().num_claims, 1);
    }

    #[test]
    fn batched_root_layout_is_invariant_under_equivalent_partitions() {
        type Cfg = fp128::D64OneHot;

        let batch_a = HachiRootBatchSummary::from_claim_group_sizes(&[1, 1, 4], 2).unwrap();
        let batch_b = HachiRootBatchSummary::from_claim_group_sizes(&[2, 2, 2], 2).unwrap();

        let plan_a =
            Cfg::get_params_for_prove::<{ Cfg::D }>(30, 30, batch_a.num_claims, batch_a).unwrap();
        let plan_b =
            Cfg::get_params_for_prove::<{ Cfg::D }>(30, 30, batch_b.num_claims, batch_b).unwrap();

        assert_eq!(plan_a.level_lp, plan_b.level_lp);
        assert_eq!(plan_a.root_lp, plan_b.root_lp);
    }

    #[test]
    fn batched_root_next_w_len_and_shape_are_invariant_under_equivalent_partitions() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let claim_groups_a = [1usize, 1, 4];
        let claim_groups_b = [2usize, 2, 2];
        let batch_a = HachiRootBatchSummary::from_claim_group_sizes(&claim_groups_a, 2).unwrap();
        let batch_b = HachiRootBatchSummary::from_claim_group_sizes(&claim_groups_b, 2).unwrap();

        let plan_a = Cfg::get_params_for_prove::<{ Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            batch_a.num_claims,
            batch_a,
        )
        .unwrap();
        let plan_b = Cfg::get_params_for_prove::<{ Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            batch_b.num_claims,
            batch_b,
        )
        .unwrap();

        let next_w_ring_a = w_ring_element_count_with_claim_groups::<
            <Cfg as CommitmentConfig>::Field,
        >(&plan_a.level_lp, &claim_groups_a, batch_a.num_points);
        let next_w_ring_b = w_ring_element_count_with_claim_groups::<
            <Cfg as CommitmentConfig>::Field,
        >(&plan_b.level_lp, &claim_groups_b, batch_b.num_points);

        assert_eq!(next_w_ring_a, next_w_ring_b);
        assert_eq!(plan_a.next_w_len(), plan_b.next_w_len());
        assert_eq!(plan_a.level_proof_shape(), plan_b.level_proof_shape());
    }

    #[test]
    fn batched_root_next_w_len_requires_group_and_point_counts() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let singleton_groups = HachiRootBatchSummary::new(6, 6, 1).unwrap();
        let grouped_same_point = HachiRootBatchSummary::new(6, 3, 1).unwrap();
        let grouped_two_points = HachiRootBatchSummary::new(6, 3, 2).unwrap();

        let singleton_plan = Cfg::get_params_for_prove::<{ Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            singleton_groups.num_claims,
            singleton_groups,
        )
        .unwrap();
        let grouped_plan = Cfg::get_params_for_prove::<{ Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            grouped_same_point.num_claims,
            grouped_same_point,
        )
        .unwrap();
        let multipoint_plan = Cfg::get_params_for_prove::<{ Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            grouped_two_points.num_claims,
            grouped_two_points,
        )
        .unwrap();

        assert_eq!(singleton_plan.level_lp, grouped_plan.level_lp);
        assert_eq!(grouped_plan.level_lp, multipoint_plan.level_lp);
        assert_ne!(singleton_plan.next_w_len(), grouped_plan.next_w_len());
        assert_ne!(grouped_plan.next_w_len(), multipoint_plan.next_w_len());
        assert_eq!(singleton_plan.level_proof_shape().y_ring_coeffs, Cfg::D);
        assert_eq!(grouped_plan.level_proof_shape().y_ring_coeffs, Cfg::D);
        assert_eq!(
            multipoint_plan.level_proof_shape().y_ring_coeffs,
            2 * Cfg::D
        );
    }
}
