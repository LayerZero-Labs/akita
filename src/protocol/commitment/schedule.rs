use super::commit::{hachi_batched_root_layout, root_current_w_len, scale_batched_root_layout};
use super::config::{
    compute_num_digits, compute_num_digits_fold, optimal_m_r_split_with_params, CommitmentConfig,
    DecompositionParams, HachiCommitmentLayout,
};
use super::generated::{
    table_entry, GeneratedDirectWitnessShape, GeneratedFoldStep, GeneratedScheduleTable,
    GeneratedStep,
};
use crate::algebra::SparseChallengeConfig;
use crate::error::HachiError;
use crate::protocol::proof::{
    DirectWitnessShape, HachiProofShape, HachiProofStepShape, LevelProofShape,
};
use crate::protocol::ring_switch::w_ring_element_count_with_batch_summary;
use crate::protocol::sumcheck::hachi_stage1_tree::stage1_tree_stage_shapes;
use std::collections::HashMap;
use std::fmt::Write;

/// Public inputs that deterministically select one level's active Hachi params.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub(crate) struct HachiRootRuntimePlan {
    /// Setup/public schedule bucket that selects the root family policy.
    pub max_num_vars: usize,
    /// Actual root polynomial arity.
    pub num_vars: usize,
    /// Number of claims the root commitment layout was sized for.
    pub layout_num_claims: usize,
    /// Aggregate opening-batch summary for this root invocation.
    pub batch: HachiRootBatchSummary,
    /// Public inputs for the root level.
    pub inputs: HachiScheduleInputs,
    /// Per-polynomial root commitment layout chosen before root batching.
    pub root_layout: HachiCommitmentLayout,
    /// Actual runtime root-level layout after scaling for `batch`.
    pub level_layout: HachiCommitmentLayout,
    /// Active root params under `root_layout.log_basis`.
    pub params: HachiLevelParams,
    /// Public inputs for the first recursive level after the root fold.
    pub next_inputs: HachiScheduleInputs,
    /// Active params for the first recursive level, respecting the current
    /// root basis when selecting the next basis.
    pub next_level_params: HachiLevelParams,
}

impl HachiRootRuntimePlan {
    /// Recursive witness length after the root fold.
    pub(crate) fn next_w_len(&self) -> usize {
        self.next_inputs.current_w_len
    }

    /// Shape of the serialized root proof body for this runtime context.
    #[cfg(test)]
    pub(crate) fn level_proof_shape(&self) -> LevelProofShape {
        let rounds = sumcheck_rounds(self.params.d, self.next_w_len());
        let b = 1usize << self.level_layout.log_basis;
        LevelProofShape {
            y_ring_coeffs: self.batch.num_claims * self.params.d,
            v_coeffs: self.params.n_d * self.params.d,
            stage1_stages: stage1_tree_stage_shapes(rounds, b),
            stage2_sumcheck: (rounds, 3),
            next_commit_coeffs: self.next_level_params.n_b * self.next_level_params.d,
        }
    }

    /// Exact bytes of the serialized root proof body for this runtime context.
    pub(crate) fn level_proof_bytes<Cfg: CommitmentConfig>(&self) -> usize {
        batched_root_level_proof_bytes(
            field_bits(Cfg::decomposition()),
            &self.params,
            self.level_layout,
            &self.next_level_params,
            self.next_w_len(),
            self.batch.num_claims,
        )
    }
}

/// Canonical root schedule artifact used by dynamic root selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HachiRootScheduleArtifact<const D: usize> {
    /// Public lookup key for this root schedule.
    pub key: HachiScheduleLookupKey,
    /// Canonical root runtime plan.
    pub root_plan: HachiRootRuntimePlan,
    /// Recursive suffix bytes from the chosen next-level state.
    pub recursive_suffix_bytes: usize,
    /// Total estimated proof bytes for this root schedule.
    pub total_proof_bytes: usize,
}

/// Derive the canonical root schedule artifact for one root config.
///
/// # Errors
///
/// Returns an error if the root layout, next-level basis selection, or
/// recursive suffix sizing fails for the provided public root context.
pub(crate) fn hachi_root_schedule_artifact<Cfg, const D: usize>(
    key: HachiScheduleLookupKey,
) -> Result<HachiRootScheduleArtifact<D>, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_layout = hachi_batched_root_layout::<Cfg, D>(key.num_vars, key.layout_num_claims)?;
    let root_plan = hachi_root_runtime_plan_from_root_layout::<Cfg, D>(key, root_layout)?;
    let recursive_suffix_bytes = planned_recursive_suffix_bytes_with_log_basis::<Cfg>(
        key.max_num_vars,
        root_plan.next_inputs.level,
        root_plan.next_inputs.current_w_len,
        root_plan.next_level_params.log_basis,
    )?;
    let total_proof_bytes = root_plan.level_proof_bytes::<Cfg>() + recursive_suffix_bytes;
    Ok(HachiRootScheduleArtifact {
        key,
        root_plan,
        recursive_suffix_bytes,
        total_proof_bytes,
    })
}

/// Runtime source of truth for one Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiLevelParams {
    /// Ring dimension at this level.
    pub d: usize,
    /// Gadget base exponent.
    pub log_basis: u32,
    /// Active inner Ajtai rank.
    pub n_a: usize,
    /// Active outer commitment rank.
    pub n_b: usize,
    /// Active D-matrix rank.
    pub n_d: usize,
    /// Conservative sparse-challenge L1 mass used by folded-norm bounds.
    pub challenge_l1_mass: usize,
    /// Stage-1 challenge family sampled at this level.
    pub stage1_config: SparseChallengeConfig,
}

impl HachiLevelParams {
    /// Total number of quotient / relation rows in `M`.
    pub fn m_row_count(&self) -> usize {
        self.m_row_count_with_public_outputs(1)
    }

    /// Total number of quotient / relation rows when the root carries
    /// `num_public_outputs` public `y` rows.
    pub fn m_row_count_with_public_outputs(&self, num_public_outputs: usize) -> usize {
        self.n_d + self.n_b + num_public_outputs + 1 + self.n_a
    }

    /// Total number of quotient / relation rows when the root carries
    /// `num_commitments` explicit commitment vectors and `num_public_outputs`
    /// public `y` rows.
    pub fn m_row_count_with_commitments_and_public_outputs(
        &self,
        num_commitments: usize,
        num_public_outputs: usize,
    ) -> usize {
        self.n_d + self.n_b * num_commitments + num_public_outputs + 1 + self.n_a
    }

    /// Total number of root-batched quotient / relation rows when each claim
    /// keeps its own commitment vector.
    pub fn batched_root_m_row_count(&self, num_claims: usize) -> usize {
        self.m_row_count_with_commitments_and_public_outputs(num_claims, num_claims)
    }
}

/// Derive the canonical runtime context for a singleton root opening.
///
/// `layout_num_claims` is the root-commitment batch capacity the setup/layout
/// was chosen for, which can differ from the actual opening batch.
///
/// # Errors
///
/// Returns an error if the root layout, batched layout scaling, next witness
/// sizing, or next-level basis selection fails.
pub(crate) fn hachi_root_runtime_plan<Cfg, const D: usize>(
    max_num_vars: usize,
    num_vars: usize,
    layout_num_claims: usize,
) -> Result<HachiRootRuntimePlan, HachiError>
where
    Cfg: CommitmentConfig,
{
    hachi_root_runtime_plan_with_batch::<Cfg, D>(
        max_num_vars,
        num_vars,
        layout_num_claims,
        HachiRootBatchSummary::singleton(),
    )
}

/// Derive the canonical runtime context for a batched root opening.
///
/// `layout_num_claims` selects the per-polynomial root layout fixed at commit
/// time, while `batch` captures the concrete opening batch that determines the
/// actual root-level witness size and proof shape.
///
/// # Errors
///
/// Returns an error if the root layout, batched layout scaling, next witness
/// sizing, or next-level basis selection fails.
pub(crate) fn hachi_root_runtime_plan_with_batch<Cfg, const D: usize>(
    max_num_vars: usize,
    num_vars: usize,
    layout_num_claims: usize,
    batch: HachiRootBatchSummary,
) -> Result<HachiRootRuntimePlan, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_layout = hachi_batched_root_layout::<Cfg, D>(num_vars, layout_num_claims)?;
    hachi_root_runtime_plan_from_root_layout::<Cfg, D>(
        HachiScheduleLookupKey::with_batch(max_num_vars, num_vars, layout_num_claims, batch),
        root_layout,
    )
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
    root_layout: HachiCommitmentLayout,
) -> Result<HachiRootRuntimePlan, HachiError>
where
    Cfg: CommitmentConfig,
{
    let level_layout =
        scale_batched_root_layout::<Cfg, D>(key.max_num_vars, root_layout, key.batch.num_claims)?;
    let inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 0,
        current_w_len: root_current_w_len::<D>(root_layout),
    };
    let params = Cfg::level_params_with_log_basis(inputs, root_layout.log_basis);
    let next_w_ring =
        w_ring_element_count_with_batch_summary::<Cfg::Field>(&params, level_layout, key.batch);
    let next_w_len = next_w_ring
        .checked_mul(params.d)
        .ok_or_else(|| HachiError::InvalidSetup("root next witness length overflow".to_string()))?;
    let next_inputs = HachiScheduleInputs {
        max_num_vars: key.max_num_vars,
        level: 1,
        current_w_len: next_w_len,
    };
    let next_log_basis =
        planned_next_log_basis_with_current_basis::<Cfg>(next_inputs, params.log_basis)?;
    let next_level_params = Cfg::level_params_with_log_basis(next_inputs, next_log_basis);

    Ok(HachiRootRuntimePlan {
        max_num_vars: key.max_num_vars,
        num_vars: key.num_vars,
        layout_num_claims: key.layout_num_claims,
        batch: key.batch,
        inputs,
        root_layout,
        level_layout,
        params,
        next_inputs,
        next_level_params,
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

fn main_level_decomposition<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
) -> DecompositionParams {
    main_level_decomposition_from_root(Cfg::decomposition(), params.log_basis)
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
    params: &HachiLevelParams,
) -> DecompositionParams {
    recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis)
}

fn layout_from_params(
    m_vars: usize,
    r_vars: usize,
    params: &HachiLevelParams,
    decomp: DecompositionParams,
    num_ring: usize,
) -> Result<HachiCommitmentLayout, HachiError> {
    let depth_commit = compute_num_digits(decomp.log_commit_bound, decomp.log_basis);
    let open_bound = decomp.log_open_bound.unwrap_or(decomp.log_commit_bound);
    let depth_open = compute_num_digits(open_bound, decomp.log_basis);
    let depth_fold = compute_num_digits_fold(r_vars, params.challenge_l1_mass, decomp.log_basis);
    HachiCommitmentLayout::new_with_decomp(
        m_vars,
        r_vars,
        params.n_a,
        depth_commit,
        depth_open,
        depth_fold,
        decomp.log_basis,
        num_ring,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PlannerState {
    level: usize,
    current_w_len: usize,
    log_basis: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Fully planned public data for one Hachi fold level.
pub struct HachiPlannedLevel {
    /// Public inputs that selected this level.
    pub inputs: HachiScheduleInputs,
    /// Active Hachi parameters chosen for this level.
    pub params: HachiLevelParams,
    /// Runtime commitment layout used at this level.
    pub layout: HachiCommitmentLayout,
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
            log_basis: self.params.log_basis,
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
    /// Total proof bytes in the serialized `HachiProof` wire format.
    ///
    /// `HachiProof` is currently headerless, so this equals
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

    /// Derive the [`HachiProofShape`] needed for deserializing a proof
    /// produced under this schedule.
    pub fn to_proof_shape(&self) -> HachiProofShape {
        let mut step_shapes: Vec<HachiProofStepShape> = self
            .fold_levels()
            .map(|level| {
                let p = &level.params;
                let next_w_len = level.next_inputs.current_w_len;
                let rounds = sumcheck_rounds(p.d, next_w_len);
                let b = 1usize << level.layout.log_basis;

                HachiProofStepShape::Fold(LevelProofShape {
                    y_ring_coeffs: p.d,
                    v_coeffs: p.n_d * p.d,
                    stage1_stages: stage1_tree_stage_shapes(rounds, b),
                    stage2_sumcheck: (rounds, 3),
                    next_commit_coeffs: level.next_commit_coeffs,
                })
            })
            .collect();

        let terminal = self.direct_step();
        step_shapes.push(HachiProofStepShape::Direct(terminal.witness_shape.clone()));
        HachiProofShape { step_shapes }
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
) -> Result<HachiLevelParams, HachiError> {
    let params = HachiLevelParams {
        d: step.d as usize,
        log_basis: step.log_basis,
        n_a: step.n_a as usize,
        n_b: step.n_b as usize,
        n_d: step.n_d as usize,
        challenge_l1_mass: step.challenge_l1_mass,
        stage1_config: Cfg::stage1_challenge_config(step.d as usize),
    };
    if params.challenge_l1_mass != params.stage1_config.l1_mass() {
        return Err(HachiError::InvalidSetup(format!(
            "generated schedule {context} challenge L1 mass mismatch: pinned={}, runtime={}",
            params.challenge_l1_mass,
            params.stage1_config.l1_mass()
        )));
    }
    Ok(params)
}

fn schedule_plan_from_generated_entry<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    entry: &super::generated::GeneratedScheduleTableEntry,
) -> Result<HachiSchedulePlan, HachiError> {
    let Some(root_step) = entry.steps.first() else {
        return Err(HachiError::InvalidSetup(
            "generated schedule table entry must contain at least one step".to_string(),
        ));
    };
    let expected_root_w_len = 1usize
        .checked_shl(max_num_vars as u32)
        .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?;
    if generated_step_current_w_len(root_step) != expected_root_w_len {
        return Err(HachiError::InvalidSetup(format!(
            "generated root witness length {} does not match max_num_vars={max_num_vars}",
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
                    max_num_vars,
                    level: fold_level,
                    current_w_len: level.current_w_len,
                };
                let next_inputs = HachiScheduleInputs {
                    max_num_vars,
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
                let runtime_next_w_len =
                    planned_next_w_len(field_bits, Cfg::planner_half_field_bound(), &params, layout);
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
                        (
                            next_level_params.clone(),
                            next_level_params.n_b * next_level_params.d,
                        )
                    }
                    GeneratedStep::Direct(direct) => {
                        let (entry_d, entry_nb) = match (direct.entry_d, direct.entry_nb) {
                            (Some(entry_d), Some(entry_nb)) => (entry_d as usize, entry_nb as usize),
                            (None, None) => (params.d, 0),
                            _ => {
                                return Err(HachiError::InvalidSetup(
                                    "generated direct entry commitment must specify both D and n_b or neither"
                                        .to_string(),
                                ))
                            }
                        };
                        (
                            HachiLevelParams {
                                d: entry_d,
                                log_basis: next_log_basis,
                                n_a: 0,
                                n_b: entry_nb,
                                n_d: 0,
                                challenge_l1_mass: params.challenge_l1_mass,
                                stage1_config: params.stage1_config.clone(),
                            },
                            entry_nb * entry_d,
                        )
                    }
                };
                let runtime_level_bytes = hachi_level_proof_bytes(
                    field_bits,
                    &params,
                    layout,
                    &next_level_params,
                    next_inputs.current_w_len,
                );

                steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
                    inputs,
                    params,
                    layout,
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
                if !matches!((direct.entry_d, direct.entry_nb), (Some(_), Some(_)) | (None, None)) {
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

pub(crate) fn generated_schedule_plan_from_table<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    table: GeneratedScheduleTable,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    table_entry(table, max_num_vars)
        .map(|entry| schedule_plan_from_generated_entry::<Cfg>(max_num_vars, entry))
        .transpose()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedSuffix {
    steps: Vec<HachiPlannedStep>,
    no_wrapper_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlannerConfig {
    max_num_vars: usize,
    min_log_basis: u32,
    max_log_basis: u32,
    field_bits: u32,
    half_field_bound: u128,
}

fn field_bits(root_decomp: DecompositionParams) -> u32 {
    root_decomp
        .log_open_bound
        .unwrap_or(root_decomp.log_commit_bound)
}

fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

pub(crate) fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

fn direct_witness_bytes(field_bits: u32, shape: &DirectWitnessShape) -> usize {
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

pub(crate) fn recursive_r_decomp_levels_for_bound(
    field_bits: u32,
    half_field_bound: u128,
    log_basis: u32,
) -> usize {
    let bits = field_bits as usize;
    let lb = log_basis as usize;
    let mut levels = compute_num_digits(field_bits, log_basis);
    if levels == 0 {
        levels = 1;
    }

    let total_bits = levels * lb;
    if total_bits <= bits {
        let b = 1u128 << log_basis;
        let half_b_minus_1 = b / 2 - 1;
        let b_minus_1 = b - 1;
        let mut b_pow = 1u128;
        for _ in 0..levels {
            b_pow = b_pow.saturating_mul(b);
        }
        let max_positive = half_b_minus_1.saturating_mul((b_pow - 1) / b_minus_1);
        if max_positive < half_field_bound {
            levels += 1;
        }
    }

    levels
}

pub(crate) fn planned_w_ring_element_count(
    field_bits: u32,
    half_field_bound: u128,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> usize {
    let w_hat_count = layout.num_blocks * layout.num_digits_open;
    let t_hat_count = layout.num_blocks * level_params.n_a * layout.num_digits_open;
    let z_pre_count = layout.inner_width * layout.num_digits_fold;
    let r_count = level_params.m_row_count()
        * recursive_r_decomp_levels_for_bound(field_bits, half_field_bound, layout.log_basis);
    w_hat_count + t_hat_count + z_pre_count + r_count
}

pub(crate) fn planned_next_w_len(
    field_bits: u32,
    half_field_bound: u128,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
) -> usize {
    planned_w_ring_element_count(field_bits, half_field_bound, level_params, layout)
        * level_params.d
}

fn sumcheck_rounds(level_d: usize, next_w_len: usize) -> usize {
    let num_l = level_d.trailing_zeros() as usize;
    let num_ring_elems = next_w_len / level_d;
    let num_u = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    num_u + num_l
}

pub(crate) fn hachi_level_proof_bytes(
    field_bits: u32,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    next_level_params: &HachiLevelParams,
    next_w_len: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = proof_ring_vec_bytes(1, level_params.d, elem_bytes);
    let v_bytes = proof_ring_vec_bytes(level_params.n_d, level_params.d, elem_bytes);
    let next_commit_bytes =
        proof_ring_vec_bytes(next_level_params.n_b, next_level_params.d, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(level_params.d, next_w_len);
    let b = 1usize << layout.log_basis;
    let stage1_bytes = stage1_proof_bytes(rounds, b, elem_bytes);

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}

pub(crate) fn batched_root_level_proof_bytes(
    field_bits: u32,
    level_params: &HachiLevelParams,
    layout: HachiCommitmentLayout,
    next_level_params: &HachiLevelParams,
    next_w_len: usize,
    num_claims: usize,
) -> usize {
    let elem_bytes = field_bytes(field_bits);
    let y_bytes = proof_ring_vec_bytes(num_claims, level_params.d, elem_bytes);
    let v_bytes = proof_ring_vec_bytes(level_params.n_d, level_params.d, elem_bytes);
    let next_commit_bytes =
        proof_ring_vec_bytes(next_level_params.n_b, next_level_params.d, elem_bytes);
    let next_eval_bytes = elem_bytes;
    let rounds = sumcheck_rounds(level_params.d, next_w_len);
    let b = 1usize << layout.log_basis;
    let stage1_bytes = stage1_proof_bytes(rounds, b, elem_bytes);

    y_bytes
        + v_bytes
        + stage1_bytes
        + sumcheck_bytes(rounds, 3, elem_bytes)
        + next_commit_bytes
        + next_eval_bytes
}

fn current_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: HachiScheduleInputs,
    log_basis: u32,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params_with_log_basis(inputs, log_basis);
    let layout = if inputs.level == 0 {
        let alpha = params.d.trailing_zeros() as usize;
        let reduced_vars = inputs.max_num_vars.checked_sub(alpha).ok_or_else(|| {
            HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
        })?;
        if reduced_vars == 0 {
            return Err(HachiError::InvalidSetup(
                "max_num_vars must leave at least one outer variable".to_string(),
            ));
        }
        let decomp = main_level_decomposition_from_root(Cfg::decomposition(), log_basis);
        let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, reduced_vars, 0);
        layout_from_params(m_vars, r_vars, &params, decomp, 0)?
    } else {
        hachi_recursive_level_layout_from_params::<Cfg>(&params, inputs.current_w_len)?
    };
    Ok((params, layout))
}

fn best_recursive_suffix<Cfg: CommitmentConfig>(
    cfg: PlannerConfig,
    memo: &mut HashMap<PlannerState, PlannedSuffix>,
    state: PlannerState,
) -> Result<PlannedSuffix, HachiError> {
    if let Some(existing) = memo.get(&state) {
        return Ok(existing.clone());
    }

    let direct_state = HachiPlannedState {
        level: state.level,
        current_w_len: state.current_w_len,
        log_basis: state.log_basis,
    };
    let witness_shape = DirectWitnessShape::PackedDigits((state.current_w_len, state.log_basis));
    let direct_bytes = direct_witness_bytes(cfg.field_bits, &witness_shape);
    let mut best = PlannedSuffix {
        steps: vec![HachiPlannedStep::Direct(HachiPlannedDirectStep {
            state: direct_state,
            witness_shape,
            direct_bytes,
        })],
        no_wrapper_bytes: direct_bytes,
    };

    let inputs = HachiScheduleInputs {
        max_num_vars: cfg.max_num_vars,
        level: state.level,
        current_w_len: state.current_w_len,
    };
    if let Ok((params, layout)) =
        current_level_layout_with_log_basis::<Cfg>(inputs, state.log_basis)
    {
        let next_w_len = planned_next_w_len(cfg.field_bits, cfg.half_field_bound, &params, layout);
        if next_w_len < state.current_w_len {
            let next_level = state.level + 1;
            let next_inputs = HachiScheduleInputs {
                max_num_vars: cfg.max_num_vars,
                level: next_level,
                current_w_len: next_w_len,
            };
            for next_log_basis in state.log_basis.max(cfg.min_log_basis)..=cfg.max_log_basis {
                let next_level_params =
                    Cfg::level_params_with_log_basis(next_inputs, next_log_basis);
                let level_bytes = hachi_level_proof_bytes(
                    cfg.field_bits,
                    &params,
                    layout,
                    &next_level_params,
                    next_w_len,
                );
                let suffix = best_recursive_suffix::<Cfg>(
                    cfg,
                    memo,
                    PlannerState {
                        level: next_level,
                        current_w_len: next_w_len,
                        log_basis: next_log_basis,
                    },
                )?;
                let candidate_bytes = level_bytes + suffix.no_wrapper_bytes;
                if candidate_bytes < best.no_wrapper_bytes {
                    let mut steps = Vec::with_capacity(suffix.steps.len() + 1);
                    steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
                        inputs,
                        params: params.clone(),
                        layout,
                        next_inputs,
                        next_level_log_basis: next_log_basis,
                        next_commit_coeffs: next_level_params.n_b * next_level_params.d,
                        level_bytes,
                    })));
                    steps.extend(suffix.steps);
                    best = PlannedSuffix {
                        steps,
                        no_wrapper_bytes: candidate_bytes,
                    };
                }
            }
        }
    }

    memo.insert(state, best.clone());
    Ok(best)
}

#[cfg(test)]
pub(crate) fn planned_recursive_suffix_bytes<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<usize, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    if let Some(schedule) = Cfg::schedule_plan(max_num_vars)? {
        return planned_recursive_suffix_bytes_from_schedule::<Cfg>(
            &schedule,
            max_num_vars,
            level,
            current_w_len,
            min_log_basis,
            max_log_basis,
        );
    }
    let current_log_basis = Cfg::log_basis_at_level(inputs);
    let cfg = PlannerConfig {
        max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let suffix = best_recursive_suffix::<Cfg>(
        cfg,
        &mut memo,
        PlannerState {
            level,
            current_w_len,
            log_basis: current_log_basis,
        },
    )?;
    Ok(suffix.no_wrapper_bytes)
}

pub(crate) fn planned_recursive_suffix_bytes_with_log_basis<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
) -> Result<usize, HachiError> {
    if let Some(schedule) = Cfg::schedule_plan(max_num_vars)? {
        let inputs = HachiScheduleInputs {
            max_num_vars,
            level,
            current_w_len,
        };
        let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(inputs);
        return planned_recursive_suffix_bytes_with_log_basis_from_schedule::<Cfg>(
            &schedule,
            max_num_vars,
            level,
            current_w_len,
            current_log_basis,
            min_log_basis,
            max_log_basis,
        );
    }
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(inputs);
    let cfg = PlannerConfig {
        max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let suffix = best_recursive_suffix::<Cfg>(
        cfg,
        &mut memo,
        PlannerState {
            level,
            current_w_len,
            log_basis: current_log_basis,
        },
    )?;
    Ok(suffix.no_wrapper_bytes)
}

pub(crate) fn planned_next_log_basis_with_current_basis<Cfg: CommitmentConfig>(
    next_inputs: HachiScheduleInputs,
    current_log_basis: u32,
) -> Result<u32, HachiError> {
    if let Some(schedule) = Cfg::schedule_plan(next_inputs.max_num_vars)? {
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
    }
    let (min_log_basis, max_log_basis) = Cfg::log_basis_search_range(next_inputs);
    let lower_bound = current_log_basis.max(min_log_basis);
    let cfg = PlannerConfig {
        max_num_vars: next_inputs.max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let mut best: Option<(u32, usize)> = None;
    for log_basis in lower_bound..=max_log_basis {
        let suffix = best_recursive_suffix::<Cfg>(
            cfg,
            &mut memo,
            PlannerState {
                level: next_inputs.level,
                current_w_len: next_inputs.current_w_len,
                log_basis,
            },
        )?;
        if best
            .as_ref()
            .is_none_or(|(_, best_bytes)| suffix.no_wrapper_bytes < *best_bytes)
        {
            best = Some((log_basis, suffix.no_wrapper_bytes));
        }
    }
    best.map(|(log_basis, _)| log_basis)
        .ok_or_else(|| HachiError::InvalidSetup("no valid next-level log basis found".to_string()))
}

pub(crate) fn estimated_recursive_suffix_bytes<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
) -> Result<usize, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    let current_log_basis = Cfg::log_basis_at_level(inputs);
    let direct_bytes = packed_digits_bytes(current_w_len, current_log_basis);

    if let Some(planned_bytes) = Cfg::recursive_suffix_bytes(max_num_vars, level, current_w_len)? {
        return Ok(planned_bytes.min(direct_bytes));
    }

    let (params, layout) = current_level_layout_with_log_basis::<Cfg>(inputs, current_log_basis)?;
    let field_bits = field_bits(Cfg::decomposition());
    let next_w_len =
        planned_next_w_len(field_bits, Cfg::planner_half_field_bound(), &params, layout);
    if next_w_len >= current_w_len {
        return Ok(direct_bytes);
    }

    let next_inputs = HachiScheduleInputs {
        max_num_vars,
        level: level + 1,
        current_w_len: next_w_len,
    };
    let next_level_params = Cfg::level_params(next_inputs);
    let continue_bytes =
        hachi_level_proof_bytes(field_bits, &params, layout, &next_level_params, next_w_len)
            + packed_digits_bytes(next_w_len, next_level_params.log_basis);
    Ok(direct_bytes.min(continue_bytes))
}

pub(crate) fn planned_schedule<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<HachiSchedulePlan, HachiError> {
    let root_current_w_len = 1usize
        .checked_shl(max_num_vars as u32)
        .ok_or_else(|| HachiError::InvalidSetup("root witness length overflow".to_string()))?;
    let root_inputs = HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: root_current_w_len,
    };
    let cfg = PlannerConfig {
        max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let direct_state = HachiPlannedState {
        level: 0,
        current_w_len: root_current_w_len,
        log_basis: Cfg::decomposition().log_basis,
    };
    let direct_witness_shape = DirectWitnessShape::FieldElements(root_current_w_len);
    let direct_bytes = direct_witness_bytes(cfg.field_bits, &direct_witness_shape);
    let mut best: Option<PlannedSuffix> = Some(PlannedSuffix {
        steps: vec![HachiPlannedStep::Direct(HachiPlannedDirectStep {
            state: direct_state,
            witness_shape: direct_witness_shape,
            direct_bytes,
        })],
        no_wrapper_bytes: direct_bytes,
    });

    for root_log_basis in min_log_basis..=max_log_basis {
        let Ok((root_params, root_layout)) =
            current_level_layout_with_log_basis::<Cfg>(root_inputs, root_log_basis)
        else {
            continue;
        };
        let next_w_len = planned_next_w_len(
            cfg.field_bits,
            cfg.half_field_bound,
            &root_params,
            root_layout,
        );

        let next_level = 1usize;
        let next_inputs = HachiScheduleInputs {
            max_num_vars,
            level: next_level,
            current_w_len: next_w_len,
        };
        for next_log_basis in root_log_basis.max(min_log_basis)..=max_log_basis {
            let next_level_params = Cfg::level_params_with_log_basis(next_inputs, next_log_basis);
            let level_bytes = hachi_level_proof_bytes(
                cfg.field_bits,
                &root_params,
                root_layout,
                &next_level_params,
                next_w_len,
            );
            let next_state = HachiPlannedState {
                level: next_inputs.level,
                current_w_len: next_inputs.current_w_len,
                log_basis: next_log_basis,
            };
            let mut steps = Vec::new();
            steps.push(HachiPlannedStep::Fold(Box::new(HachiPlannedLevel {
                inputs: root_inputs,
                params: root_params.clone(),
                layout: root_layout,
                next_inputs,
                next_level_log_basis: next_log_basis,
                next_commit_coeffs: next_level_params.n_b * next_level_params.d,
                level_bytes,
            })));
            let suffix = if next_w_len < root_inputs.current_w_len {
                best_recursive_suffix::<Cfg>(
                    cfg,
                    &mut memo,
                    PlannerState {
                        level: next_level,
                        current_w_len: next_w_len,
                        log_basis: next_log_basis,
                    },
                )?
            } else {
                let witness_shape = DirectWitnessShape::PackedDigits((next_w_len, next_log_basis));
                let direct_bytes = direct_witness_bytes(cfg.field_bits, &witness_shape);
                PlannedSuffix {
                    steps: vec![HachiPlannedStep::Direct(HachiPlannedDirectStep {
                        state: next_state,
                        witness_shape,
                        direct_bytes,
                    })],
                    no_wrapper_bytes: direct_bytes,
                }
            };
            let candidate_bytes = level_bytes + suffix.no_wrapper_bytes;
            if best
                .as_ref()
                .is_none_or(|existing| candidate_bytes < existing.no_wrapper_bytes)
            {
                steps.extend(suffix.steps);
                best = Some(PlannedSuffix {
                    steps,
                    no_wrapper_bytes: candidate_bytes,
                });
            }
        }
    }

    let best = best.ok_or_else(|| {
        HachiError::InvalidSetup("adaptive schedule search found no valid root level".to_string())
    })?;

    Ok(HachiSchedulePlan {
        steps: best.steps,
        no_wrapper_bytes: best.no_wrapper_bytes,
        exact_proof_bytes: best.no_wrapper_bytes,
    })
}

pub(crate) fn planned_log_basis_at_level_from_schedule<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    inputs: HachiScheduleInputs,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<u32, HachiError> {
    if let Some(state_index) = exact_planned_state_index(schedule, inputs, None) {
        return Ok(schedule
            .state_after_prefix(state_index)
            .expect("exact planned state index must resolve to a state")
            .log_basis);
    }
    let state = schedule
        .state_after_prefix(inputs.level)
        .unwrap_or_else(|| schedule.terminal_state());
    debug_assert_eq!(
        state.level,
        inputs.level.min(schedule.terminal_state().level)
    );
    if inputs.level > 0 && state.current_w_len != inputs.current_w_len {
        let cfg = PlannerConfig {
            max_num_vars: inputs.max_num_vars,
            min_log_basis,
            max_log_basis,
            field_bits: field_bits(Cfg::decomposition()),
            half_field_bound: Cfg::planner_half_field_bound(),
        };
        let mut memo = HashMap::new();
        let mut best: Option<(u32, usize)> = None;
        for log_basis in min_log_basis..=max_log_basis {
            let suffix = best_recursive_suffix::<Cfg>(
                cfg,
                &mut memo,
                PlannerState {
                    level: inputs.level,
                    current_w_len: inputs.current_w_len,
                    log_basis,
                },
            )?;
            if best
                .as_ref()
                .is_none_or(|(_, best_bytes)| suffix.no_wrapper_bytes < *best_bytes)
            {
                best = Some((log_basis, suffix.no_wrapper_bytes));
            }
        }
        return best.map(|(log_basis, _)| log_basis).ok_or_else(|| {
            HachiError::InvalidSetup("no valid adaptive log basis found".to_string())
        });
    }
    Ok(state.log_basis)
}

pub(crate) fn planned_schedule_key_from_schedule(schedule: &HachiSchedulePlan) -> String {
    let mut key = String::from("planner_v2");
    for state in schedule.states() {
        let _ = write!(key, "_l{}b{}", state.level, state.log_basis);
    }
    key
}

pub(crate) fn planned_recursive_suffix_bytes_from_schedule<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    min_log_basis: u32,
    max_log_basis: u32,
) -> Result<usize, HachiError> {
    let inputs = HachiScheduleInputs {
        max_num_vars,
        level,
        current_w_len,
    };
    if let Some(state_index) = exact_planned_state_index(schedule, inputs, None) {
        return Ok(scheduled_suffix_bytes_from_index(schedule, state_index));
    }
    let current_log_basis = planned_log_basis_at_level_from_schedule::<Cfg>(
        schedule,
        inputs,
        min_log_basis,
        max_log_basis,
    )?;
    let cfg = PlannerConfig {
        max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let suffix = best_recursive_suffix::<Cfg>(
        cfg,
        &mut memo,
        PlannerState {
            level,
            current_w_len,
            log_basis: current_log_basis,
        },
    )?;
    Ok(suffix.no_wrapper_bytes)
}

pub(crate) fn planned_recursive_suffix_bytes_with_log_basis_from_schedule<Cfg: CommitmentConfig>(
    schedule: &HachiSchedulePlan,
    max_num_vars: usize,
    level: usize,
    current_w_len: usize,
    current_log_basis: u32,
    min_log_basis: u32,
    max_log_basis: u32,
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
    let cfg = PlannerConfig {
        max_num_vars,
        min_log_basis,
        max_log_basis,
        field_bits: field_bits(Cfg::decomposition()),
        half_field_bound: Cfg::planner_half_field_bound(),
    };
    let mut memo = HashMap::new();
    let suffix = best_recursive_suffix::<Cfg>(
        cfg,
        &mut memo,
        PlannerState {
            level,
            current_w_len,
            log_basis: current_log_basis,
        },
    )?;
    Ok(suffix.no_wrapper_bytes)
}

/// Derive the root level's active params and layout.
///
/// # Errors
///
/// Returns an error if the root variable split is invalid or overflows.
pub fn hachi_root_level_layout<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params(HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    });
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.checked_sub(alpha).ok_or_else(|| {
        HachiError::InvalidSetup("max_num_vars is smaller than alpha".to_string())
    })?;
    if reduced_vars == 0 {
        return Err(HachiError::InvalidSetup(
            "max_num_vars must leave at least one outer variable".to_string(),
        ));
    }
    let decomp = main_level_decomposition::<Cfg>(&params);
    let (m_vars, r_vars) = optimal_m_r_split_with_params(&params, decomp, reduced_vars, 0);
    let layout = layout_from_params(m_vars, r_vars, &params, decomp, 0)?;
    Ok((params, layout))
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
) -> Result<(HachiLevelParams, HachiCommitmentLayout), HachiError> {
    let params = Cfg::level_params(HachiScheduleInputs {
        max_num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
    });
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = max_num_vars.saturating_sub(alpha);
    let decomp = main_level_decomposition::<Cfg>(&params);
    let (m_vars, r_vars) = if reduced_vars == 0 {
        (0, 0)
    } else {
        optimal_m_r_split_with_params(&params, decomp, reduced_vars, 0)
    };
    let layout = layout_from_params(m_vars, r_vars, &params, decomp, 0)?;
    Ok((params, layout))
}

/// Derive a recursive `w`-opening layout from the active level params.
///
/// # Errors
///
/// Returns an error if the witness length is incompatible with `params.d` or if
/// the recursive layout derivation overflows.
pub fn hachi_recursive_level_layout_from_params<Cfg: CommitmentConfig>(
    params: &HachiLevelParams,
    current_w_len: usize,
) -> Result<HachiCommitmentLayout, HachiError> {
    if !current_w_len.is_multiple_of(params.d) {
        return Err(HachiError::InvalidInput(format!(
            "witness length {current_w_len} is not divisible by D={}",
            params.d
        )));
    }
    let num_ring_elems = current_w_len / params.d;
    let total = num_ring_elems.next_power_of_two().max(1);
    let alpha = params.d.trailing_zeros() as usize;
    let reduced_vars = total.trailing_zeros() as usize;
    let max_num_vars = reduced_vars + alpha;
    let decomp = recursive_level_decomposition::<Cfg>(params);
    let (m_vars, r_vars) =
        optimal_m_r_split_with_params(params, decomp, reduced_vars, num_ring_elems);
    let layout = layout_from_params(m_vars, r_vars, params, decomp, num_ring_elems)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{CyclotomicRing, SparseChallengeConfig};
    use crate::primitives::serialization::{Compress, HachiSerialize};
    use crate::protocol::commitment::generated::{
        fp128_adaptive_bounded_table, fp128_d128_full_table, GeneratedScheduleTable,
    };
    use crate::protocol::commitment::presets::fp128;
    use crate::protocol::proof::{
        FlatRingVec, HachiBatchedRootProof, HachiLevelProof, HachiStage1Proof,
        HachiStage1StageProof,
    };
    use crate::protocol::ring_switch::{
        w_ring_element_count, w_ring_element_count_with_point_claim_groups,
    };
    use crate::protocol::sumcheck::hachi_stage1_tree::stage1_tree_stage_shapes;
    use crate::protocol::sumcheck::{
        CompressedUniPoly, EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof,
    };
    use crate::FieldCore;

    type F = fp128::Field;

    fn dummy_sumcheck(rounds: usize, degree: usize) -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: (0..rounds)
                .map(|_| CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(); degree],
                })
                .collect(),
        }
    }

    fn dummy_eq_factored_sumcheck(rounds: usize, degree: usize) -> EqFactoredSumcheckProof<F> {
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

    fn dummy_stage1_proof(rounds: usize, b: usize) -> HachiStage1Proof<F> {
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

    fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(max_num_vars: usize) {
        let plan = Cfg::schedule_plan(max_num_vars)
            .expect("planner should succeed")
            .expect("config should provide a planner");
        for level in plan.fold_levels() {
            let runtime_next_w_len =
                w_ring_element_count::<Cfg::Field>(&level.params, level.layout) * level.params.d;
            assert_eq!(
                runtime_next_w_len, level.next_inputs.current_w_len,
                "planner/runtime next_w_len mismatch at level {} for max_num_vars={max_num_vars}",
                level.inputs.level
            );
        }
    }

    fn assert_generated_table_matches_cfg_schedule<Cfg: CommitmentConfig>(
        table: GeneratedScheduleTable,
    ) {
        for entry in table.entries {
            let generated = generated_schedule_plan_from_table::<Cfg>(entry.max_num_vars, table)
                .expect("generated table should materialize")
                .expect("entry should exist in generated table");
            let planned = Cfg::schedule_plan(entry.max_num_vars)
                .expect("config schedule should succeed")
                .expect("config should provide a generated schedule");
            assert_eq!(
                generated, planned,
                "generated schedule should match cfg-selected schedule for max_num_vars={}",
                entry.max_num_vars
            );
        }
    }

    #[test]
    fn generated_fp128_schedule_tables_match_cfg_schedule() {
        assert_generated_table_matches_cfg_schedule::<fp128::D32Full>(
            fp128_adaptive_bounded_table::<32, 128, 2, 2, 2>().unwrap(),
        );
        assert_generated_table_matches_cfg_schedule::<fp128::D32LogBasis>(
            fp128_adaptive_bounded_table::<32, 3, 2, 2, 2>().unwrap(),
        );
        assert_generated_table_matches_cfg_schedule::<fp128::D32OneHot>(
            fp128_adaptive_bounded_table::<32, 1, 2, 2, 2>().unwrap(),
        );
    }

    #[test]
    fn generated_d128_full_table_materializes_valid_plans() {
        let table = fp128_d128_full_table();
        for entry in table.entries {
            generated_schedule_plan_from_table::<fp128::D128Full>(entry.max_num_vars, table)
                .expect("generated table should materialize")
                .expect("entry should exist in generated table");
        }
    }

    #[test]
    fn d128_bounded_families_fall_back_to_runtime_planner() {
        assert!(fp128_adaptive_bounded_table::<128, 128, 1, 1, 1>().is_none());
        assert!(fp128_adaptive_bounded_table::<128, 3, 1, 1, 1>().is_none());
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
            hachi_root_runtime_plan::<Cfg, { Cfg::D }>(30, 30, 1).expect("singleton runtime plan");
        let (root_params, root_layout) = hachi_root_level_layout::<Cfg>(30).unwrap();

        assert_eq!(runtime.batch, HachiRootBatchSummary::singleton());
        assert_eq!(runtime.root_layout, root_layout);
        assert_eq!(runtime.level_layout, root_layout);
        assert_eq!(runtime.params, root_params);
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
        let next_log_basis =
            planned_next_log_basis_with_current_basis::<Cfg>(current_inputs, 5).unwrap();
        let suffix_bytes = planned_recursive_suffix_bytes_with_log_basis::<Cfg>(
            current_inputs.max_num_vars,
            current_inputs.level,
            current_inputs.current_w_len,
            5,
        )
        .unwrap();

        assert_eq!(next_log_basis, 5);
        assert!(suffix_bytes < packed_digits_bytes(current_inputs.current_w_len, 5));
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
        let num_ring = inputs.current_w_len / params.d;
        let layout_12_7 = layout_from_params(12, 7, &params, decomp, num_ring).unwrap();
        let layout_11_8 = layout_from_params(11, 8, &params, decomp, num_ring).unwrap();
        let w_12_7 = planned_w_ring_element_count(
            field_bits(Cfg::decomposition()),
            Cfg::planner_half_field_bound(),
            &params,
            layout_12_7,
        );
        let w_11_8 = planned_w_ring_element_count(
            field_bits(Cfg::decomposition()),
            Cfg::planner_half_field_bound(),
            &params,
            layout_11_8,
        );
        let reduced_vars = (inputs.current_w_len / params.d)
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
        let next_level_params = HachiLevelParams {
            d: D,
            log_basis: 2,
            n_a: 2,
            n_b: 3,
            n_d: 2,
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config: stage1_config.clone(),
        };
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let level_params = HachiLevelParams {
                d: D,
                log_basis,
                n_a: 2,
                n_b: 2,
                n_d: 2,
                challenge_l1_mass: stage1_config.l1_mass(),
                stage1_config: stage1_config.clone(),
            };
            let layout = HachiCommitmentLayout {
                m_vars: 0,
                r_vars: 0,
                num_blocks: 1,
                block_len: 1,
                inner_width: 1,
                outer_width: 1,
                d_matrix_width: 1,
                num_digits_commit: 1,
                num_digits_open: 1,
                num_digits_fold: 1,
                log_basis,
            };
            let rounds = sumcheck_rounds(D, next_w_len);
            let b = 1usize << log_basis;
            let next_commitment = FlatRingVec::from_ring_elems(&vec![
                CyclotomicRing::<F, D>::zero();
                next_level_params.n_b
            ])
            .to_proof_ring_vec();
            let level_proof = HachiLevelProof::new_two_stage::<D>(
                CyclotomicRing::<F, D>::zero(),
                vec![CyclotomicRing::<F, D>::zero(); level_params.n_d],
                dummy_stage1_proof(rounds, b),
                dummy_sumcheck(rounds, 3),
                next_commitment,
                F::zero(),
            );

            assert_eq!(
                hachi_level_proof_bytes(128, &level_params, layout, &next_level_params, next_w_len),
                level_proof.serialized_size(Compress::No),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_level_params = HachiLevelParams {
            d: D,
            log_basis: 2,
            n_a: 2,
            n_b: 3,
            n_d: 2,
            challenge_l1_mass: stage1_config.l1_mass(),
            stage1_config: stage1_config.clone(),
        };
        let next_w_len = D * 8;
        let num_claims = 5;

        for log_basis in 2..=6 {
            let level_params = HachiLevelParams {
                d: D,
                log_basis,
                n_a: 2,
                n_b: 2,
                n_d: 2,
                challenge_l1_mass: stage1_config.l1_mass(),
                stage1_config: stage1_config.clone(),
            };
            let layout = HachiCommitmentLayout {
                m_vars: 0,
                r_vars: 0,
                num_blocks: 1,
                block_len: 1,
                inner_width: 1,
                outer_width: 1,
                d_matrix_width: 1,
                num_digits_commit: 1,
                num_digits_open: 1,
                num_digits_fold: 1,
                log_basis,
            };
            let rounds = sumcheck_rounds(D, next_w_len);
            let b = 1usize << log_basis;
            let next_commitment = FlatRingVec::from_ring_elems(&vec![
                CyclotomicRing::<F, D>::zero();
                next_level_params.n_b
            ])
            .to_proof_ring_vec();
            let root_proof = HachiBatchedRootProof::new_two_stage::<D>(
                vec![CyclotomicRing::<F, D>::zero(); num_claims],
                vec![CyclotomicRing::<F, D>::zero(); level_params.n_d],
                dummy_stage1_proof(rounds, b),
                dummy_sumcheck(rounds, 3),
                next_commitment,
                F::zero(),
            );

            assert_eq!(
                batched_root_level_proof_bytes(
                    128,
                    &level_params,
                    layout,
                    &next_level_params,
                    next_w_len,
                    num_claims,
                ),
                root_proof.serialized_size(Compress::No),
                "planned batched root bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn tight_block_len_is_no_larger_than_pow2() {
        for max_num_vars in [14, 20, 30] {
            let plan = fp128::D128Full::schedule_plan(max_num_vars)
                .expect("planner should succeed")
                .expect("config should provide a planner");
            for level in plan.fold_levels() {
                let pow2_block = 1usize << level.layout.m_vars;
                assert!(
                    level.layout.block_len <= pow2_block,
                    "block_len {} should be <= 2^m_vars {} at level {} (num_vars={})",
                    level.layout.block_len,
                    pow2_block,
                    level.inputs.level,
                    max_num_vars
                );
                if level.inputs.level > 0 {
                    let num_ring = level.inputs.current_w_len / level.params.d;
                    let expected_tight = num_ring.div_ceil(level.layout.num_blocks);
                    assert_eq!(
                        level.layout.block_len, expected_tight,
                        "recursive level {} should use tight block_len = ceil({num_ring} / {})",
                        level.inputs.level, level.layout.num_blocks
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

        let plan_a = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            30,
            30,
            batch_a.num_claims,
            batch_a,
        )
        .unwrap();
        let plan_b = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            30,
            30,
            batch_b.num_claims,
            batch_b,
        )
        .unwrap();

        assert_eq!(plan_a.root_layout, plan_b.root_layout);
        assert_eq!(plan_a.level_layout, plan_b.level_layout);
        assert_eq!(plan_a.params, plan_b.params);
    }

    #[test]
    fn batched_root_next_w_len_and_shape_are_invariant_under_equivalent_partitions() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let claim_groups_a = [1usize, 1, 4];
        let claim_groups_b = [2usize, 2, 2];
        let batch_a = HachiRootBatchSummary::from_claim_group_sizes(&claim_groups_a, 2).unwrap();
        let batch_b = HachiRootBatchSummary::from_claim_group_sizes(&claim_groups_b, 2).unwrap();

        let plan_a = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            batch_a.num_claims,
            batch_a,
        )
        .unwrap();
        let plan_b = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            batch_b.num_claims,
            batch_b,
        )
        .unwrap();

        let layout = plan_a.level_layout;
        let next_w_ring_a = w_ring_element_count_with_point_claim_groups::<
            <Cfg as CommitmentConfig>::Field,
        >(&plan_a.params, layout, &claim_groups_a, batch_a.num_points);
        let next_w_ring_b = w_ring_element_count_with_point_claim_groups::<
            <Cfg as CommitmentConfig>::Field,
        >(&plan_b.params, layout, &claim_groups_b, batch_b.num_points);

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

        let singleton_plan = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            singleton_groups.num_claims,
            singleton_groups,
        )
        .unwrap();
        let grouped_plan = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            grouped_same_point.num_claims,
            grouped_same_point,
        )
        .unwrap();
        let multipoint_plan = hachi_root_runtime_plan_with_batch::<Cfg, { Cfg::D }>(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            grouped_two_points.num_claims,
            grouped_two_points,
        )
        .unwrap();

        assert_eq!(singleton_plan.level_layout, grouped_plan.level_layout);
        assert_eq!(grouped_plan.level_layout, multipoint_plan.level_layout);
        assert_ne!(singleton_plan.next_w_len(), grouped_plan.next_w_len());
        assert_ne!(grouped_plan.next_w_len(), multipoint_plan.next_w_len());
    }
}
