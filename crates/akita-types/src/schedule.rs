//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::generated::{GeneratedScheduleKey, GeneratedScheduleTable};
use crate::{DirectWitnessShape, LevelParams};
use akita_field::HachiError;

/// Public inputs that deterministically select one level's active Akita params.
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

/// Convert the public runtime lookup key into a generated-table lookup key.
pub const fn generated_schedule_lookup_key(key: HachiScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        max_num_vars: key.max_num_vars,
        num_vars: key.num_vars,
        layout_num_claims: key.layout_num_claims,
        batch_num_claims: key.batch.num_claims,
        batch_num_commitment_groups: key.batch.num_commitment_groups,
        batch_num_points: key.batch.num_points,
    }
}

/// Fully planned public data for one Akita fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Public state after a planned prefix of Akita fold levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HachiPlannedState {
    /// Next level index reached by the plan.
    pub level: usize,
    /// Witness length in field elements at this state.
    pub current_w_len: usize,
    /// Active log-basis for the witness at this state.
    pub log_basis: u32,
}

/// Terminal direct packed-witness handoff in a planned opening proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiPlannedDirectStep {
    /// Public witness state carried by the direct handoff.
    pub state: HachiPlannedState,
    /// Serialized witness shape carried by the direct handoff.
    pub witness_shape: DirectWitnessShape,
    /// Exact bytes contributed by the packed direct witness.
    pub direct_bytes: usize,
}

/// Exact current-step execution data recovered from a pinned schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiPlannedLevelExecution {
    /// Planned fold level that matches the current public state.
    pub level: HachiPlannedLevel,
    /// Planned next-level params implied by the following schedule step.
    pub next_level_params: LevelParams,
}

/// One step in a planned opening proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HachiPlannedStep {
    /// An Akita fold level with an explicit next-state handoff.
    Fold(Box<HachiPlannedLevel>),
    /// The terminal packed-witness direct handoff.
    Direct(HachiPlannedDirectStep),
}

/// Deterministic level-by-level schedule selected from public inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Return the final witness state after all planned Akita levels.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without a trailing direct step.
    pub fn terminal_state(&self) -> HachiPlannedState {
        self.direct_step().state
    }
}

/// Provider interface for generated or externally supplied schedule plans.
///
/// Runtime prover/verifier crates should depend on this provider-shaped
/// contract rather than on planner search. `akita-planner` can implement the
/// search side separately and publish generated tables or explicit plans.
pub trait ScheduleProvider {
    /// Pre-computed schedule table backing this provider, if any.
    fn schedule_table() -> Option<GeneratedScheduleTable>;

    /// Stable identity for the active schedule at `key`.
    fn schedule_key(key: HachiScheduleLookupKey) -> String;

    /// Optional full schedule plan for configs with an explicit provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider cannot materialize a valid schedule.
    fn schedule_plan(key: HachiScheduleLookupKey) -> Result<Option<HachiSchedulePlan>, HachiError>;
}

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Unified level parameters (ring dimension, Ajtai keys, block geometry,
    /// digit depths, challenge config).
    pub params: LevelParams,
    /// Witness length entering this level.
    pub current_w_len: usize,
    /// Per-polynomial fold digits (`num_claims=1`). Equal to
    /// `params.num_digits_fold` for singleton schedules; smaller for batched
    /// roots where the layout uses the batched bound.
    pub delta_fold_per_poly: usize,
    /// Ring-element count in the witness after ring-switching.
    pub w_ring: usize,
    /// Witness length leaving this level.
    pub next_w_len: usize,
    /// Proof bytes for this level.
    pub level_bytes: usize,
}

/// Terminal direct-send step.
#[derive(Clone, Debug)]
pub struct DirectStep {
    /// Witness length entering the direct step.
    pub current_w_len: usize,
    /// Packed bits per witness element.
    pub bits_per_elem: u32,
    /// Direct witness bytes.
    pub direct_bytes: usize,
}

/// A single step in the schedule.
#[derive(Clone, Debug)]
pub enum Step {
    /// Fold through one recursive level.
    Fold(FoldStep),
    /// Send the terminal witness directly.
    Direct(DirectStep),
}

/// Complete schedule with step-by-step parameters.
#[derive(Clone, Debug)]
pub struct Schedule {
    /// Ordered proof schedule steps.
    pub steps: Vec<Step>,
    /// Exact total proof bytes for the schedule.
    pub total_bytes: usize,
}

/// Aggregate witness-shape inputs that determine root-level sizing.
///
/// The root-level witness ring count is, for any `(K, G, P)`:
///
/// ```text
///   W(lp; K, G, P) = K · 2^r · δ_open                       // |ŵ|
///                  + K · 2^r · n_A · δ_open                 // |t̂|
///                  + P · 2^m · δ_commit · δ_fold            // |z_pre|
///                  + (n_D + n_B·G + P + 1 + n_A) · δ_R(b)   // |r|
/// ```
///
/// Singleton openings are simply the `K = G = P = 1` special case of this
/// formula; the planner does not need to branch on "batched vs non-batched"
/// — only on this aggregate shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WitnessShape {
    /// `K` — total number of polynomial claims (drives `|ŵ|`, `|t̂|`).
    pub num_claims: usize,
    /// `G` — number of commitment groups (drives the `n_B·G` term in `|r|).
    pub num_commitment_groups: usize,
    /// `P` — number of distinct opening points (drives `|z_pre|` and the
    /// `+P` term in `|r|).
    pub num_points: usize,
}

impl WitnessShape {
    /// Build a witness shape from explicit `(K, G, P)`.
    pub const fn new(num_claims: usize, num_commitment_groups: usize, num_points: usize) -> Self {
        Self {
            num_claims,
            num_commitment_groups,
            num_points,
        }
    }

    /// Singleton shape: one polynomial, one group, one point.
    pub const fn singleton() -> Self {
        Self {
            num_claims: 1,
            num_commitment_groups: 1,
            num_points: 1,
        }
    }

    /// Build a witness shape from per-group opening-point counts.
    ///
    /// Interprets `points_per_group[g]` as the number of distinct opening
    /// points associated with commitment group `g`. The aggregates are:
    ///
    /// * `G = points_per_group.len()`
    /// * `P = sum(points_per_group)`  (treats each group's points as
    ///   distinct from other groups')
    /// * `K = sum(points_per_group)`  (one claim per `(group, point)` pair)
    pub fn from_points_per_group(points_per_group: &[usize]) -> Self {
        let num_commitment_groups = points_per_group.len();
        let total_points: usize = points_per_group.iter().copied().sum();
        Self {
            num_claims: total_points,
            num_commitment_groups,
            num_points: total_points,
        }
    }
}
