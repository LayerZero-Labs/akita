//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::generated::GeneratedScheduleKey;
use crate::{
    ClaimIncidenceSummary, DirectWitnessShape, LevelParams, RingOpeningPoint, TerminalProofMode,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore};

/// Public inputs that deterministically select one level's active Akita params.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleInputs {
    /// Root polynomial variable count.
    pub num_vars: usize,
    /// Fold level, where `0` is the original polynomial.
    pub level: usize,
    /// Current witness length in field elements before this level runs.
    pub current_w_len: usize,
}

/// Validate ring-switch opening-point routing against a level layout.
///
/// # Errors
///
/// Returns an error when there are no opening points, the claim-to-point table
/// has the wrong length, an opening point does not match `lp`, or a routed
/// point index is out of range.
pub fn validate_opening_points_for_claims<F: FieldCore>(
    opening_points: &[RingOpeningPoint<F>],
    claim_to_point: &[usize],
    lp: &LevelParams,
    num_claims: usize,
) -> Result<(), AkitaError> {
    if opening_points.is_empty() {
        return Err(AkitaError::InvalidInput(
            "multipoint ring switch requires at least one opening point".to_string(),
        ));
    }
    if claim_to_point.len() != num_claims {
        return Err(AkitaError::InvalidSize {
            expected: num_claims,
            actual: claim_to_point.len(),
        });
    }
    for opening_point in opening_points {
        if opening_point.a.len() < lp.block_len || opening_point.b.len() != lp.num_blocks {
            return Err(AkitaError::InvalidInput(
                "multipoint ring switch m-eval opening-point layout mismatch".to_string(),
            ));
        }
    }
    if claim_to_point
        .iter()
        .any(|&point_idx| point_idx >= opening_points.len())
    {
        return Err(AkitaError::InvalidInput(
            "multipoint ring switch claim-to-point index out of range".to_string(),
        ));
    }
    Ok(())
}

/// Public runtime key that selects a concrete root schedule context.
///
/// This is intentionally narrower than a full schedule table entry: it records
/// only the public inputs that pick a root plan, not the resulting plan data.
///
/// Under the one-commitment-per-opening-point invariant, the number of
/// distinct point commitments equals the number of distinct opening points,
/// so the planner-facing projection records `num_points`. The generated
/// schedule table key still calls this field `num_commitment_groups` for ABI
/// stability; the translation happens in
/// `akita_planner::generated_schedule_lookup_key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Root polynomial arity.
    pub num_vars: usize,
    /// Number of distinct opening points (and therefore, distinct point
    /// commitments).
    pub num_points: usize,
    /// Number of commitment-side `t` protocol vectors.
    pub num_t_vectors: usize,
    /// Number of root relation `w` protocol vectors.
    pub num_w_vectors: usize,
    /// Number of distinct `z` protocol vectors.
    pub num_z_vectors: usize,
}

impl AkitaScheduleLookupKey {
    /// Singleton root-opening context.
    pub const fn singleton(num_vars: usize) -> Self {
        Self {
            num_vars,
            num_points: 1,
            num_t_vectors: 1,
            num_w_vectors: 1,
            num_z_vectors: 1,
        }
    }

    /// General root-opening context.
    pub const fn new(
        num_vars: usize,
        num_t_vectors: usize,
        num_w_vectors: usize,
        num_z_vectors: usize,
    ) -> Self {
        Self::new_with_points(
            num_vars,
            num_z_vectors,
            num_t_vectors,
            num_w_vectors,
            num_z_vectors,
        )
    }

    /// General root-opening context with an explicit opening-point count.
    pub const fn new_with_points(
        num_vars: usize,
        num_points: usize,
        num_t_vectors: usize,
        num_w_vectors: usize,
        num_z_vectors: usize,
    ) -> Self {
        Self {
            num_vars,
            num_points,
            num_t_vectors,
            num_w_vectors,
            num_z_vectors,
        }
    }

    /// Build a schedule lookup key from normalized opening incidence.
    ///
    /// Each opening point cites exactly one commitment, so the planner-facing
    /// projection carries only the per-point arities.
    ///
    /// # Errors
    ///
    /// Returns an error if the incidence routing tables are malformed.
    pub fn new_from_incidence(incidence: &ClaimIncidenceSummary) -> Result<Self, AkitaError> {
        let num_t_vectors = incidence.num_polynomials();
        if incidence.claim_to_point().len() != incidence.num_claims() {
            return Err(AkitaError::InvalidInput(
                "claim incidence summary lengths do not match aggregate counts".to_string(),
            ));
        }
        for &point_idx in incidence.claim_to_point() {
            if point_idx >= incidence.num_points() {
                return Err(AkitaError::InvalidInput(
                    "claim incidence summary contains out-of-range routing".to_string(),
                ));
            }
        }

        Ok(Self::new_with_points(
            incidence.num_vars(),
            incidence.num_points(),
            num_t_vectors,
            incidence.num_claims(),
            incidence.num_public_rows(),
        ))
    }
}

/// Convert the public runtime lookup key into a generated-table lookup key.
///
/// The generated-table key preserves the legacy `num_commitment_groups` field
/// name as part of its ABI; `num_points` is the runtime-facing alias under the
/// one-commitment-per-point invariant.
pub const fn generated_schedule_lookup_key(key: AkitaScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        num_vars: key.num_vars,
        num_commitment_groups: key.num_points,
        num_t_vectors: key.num_t_vectors,
        num_w_vectors: key.num_w_vectors,
        num_z_vectors: key.num_z_vectors,
    }
}

/// Fully planned public data for one Akita fold level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaPlannedLevel {
    /// Public inputs that selected this level.
    pub inputs: AkitaScheduleInputs,
    /// Active unified level params chosen for this level.
    pub lp: LevelParams,
    /// Public inputs for the next level after folding.
    pub next_inputs: AkitaScheduleInputs,
    /// Planned log-basis of the next level.
    pub next_level_log_basis: u32,
    /// `n_b * d` of the next level, used for next_w_commitment shape.
    pub next_commit_coeffs: usize,
    /// Exact bytes contributed by this level to the proof.
    pub level_bytes: usize,
}

impl AkitaPlannedLevel {
    /// Public state at the start of this fold level.
    pub fn input_state(&self) -> AkitaPlannedState {
        AkitaPlannedState {
            level: self.inputs.level,
            current_w_len: self.inputs.current_w_len,
            log_basis: self.lp.log_basis,
        }
    }

    /// Public state reached after this fold level.
    pub fn output_state(&self) -> AkitaPlannedState {
        AkitaPlannedState {
            level: self.next_inputs.level,
            current_w_len: self.next_inputs.current_w_len,
            log_basis: self.next_level_log_basis,
        }
    }
}

/// Public state after a planned prefix of Akita fold levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkitaPlannedState {
    /// Next level index reached by the plan.
    pub level: usize,
    /// Witness length in field elements at this state.
    pub current_w_len: usize,
    /// Active log-basis for the witness at this state.
    pub log_basis: u32,
}

/// Terminal direct packed-witness handoff in a planned opening proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaPlannedDirectStep {
    /// Public witness state carried by the direct handoff.
    pub state: AkitaPlannedState,
    /// Serialized witness shape carried by the direct handoff.
    pub witness_shape: DirectWitnessShape,
    /// Exact bytes contributed by the packed direct witness.
    pub direct_bytes: usize,
    /// Terminal proof mode for folded terminal direct steps.
    pub terminal_proof_mode: TerminalProofMode,
    /// Commit-layout params for the root-direct case (planned root
    /// step is `Direct`). See [`DirectStep::commit_params`] for the full
    /// three-state contract; the same rules apply here.
    pub commit_params: Option<LevelParams>,
    /// SIS-secure level params for the terminal `Direct(PackedDigits)`
    /// step that sits after one or more folds. `None` for root-direct
    /// (the root direct has no sumcheck-time level after itself).
    pub level_params: Option<LevelParams>,
}

/// Exact current-step execution data recovered from a pinned schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaPlannedLevelExecution {
    /// Planned fold level that matches the current public state.
    pub level: AkitaPlannedLevel,
    /// Planned next-level params implied by the following schedule step.
    pub next_level_params: LevelParams,
}

/// One step in a planned opening proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaPlannedStep {
    /// An Akita fold level with an explicit next-state handoff.
    Fold(Box<AkitaPlannedLevel>),
    /// The terminal packed-witness direct handoff. Boxed so the variant
    /// stays small after `AkitaPlannedDirectStep` gained a
    /// root-direct `commit_params: Option<LevelParams>` field.
    Direct(Box<AkitaPlannedDirectStep>),
}

/// Deterministic level-by-level schedule selected from public inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaSchedulePlan {
    /// Planned opening-proof steps in execution order.
    ///
    /// The final step is always [`AkitaPlannedStep::Direct`].
    pub steps: Vec<AkitaPlannedStep>,
    /// Total proof bytes excluding the outer proof wrapper.
    pub no_wrapper_bytes: usize,
    /// Total proof bytes in the serialized singleton `AkitaBatchedProof`
    /// wire format.
    ///
    /// The singleton batched proof is currently headerless, so this equals
    /// [`Self::no_wrapper_bytes`].
    pub exact_proof_bytes: usize,
}

impl AkitaSchedulePlan {
    /// Iterate over all planned fold levels in execution order.
    pub fn fold_levels(&self) -> impl Iterator<Item = &AkitaPlannedLevel> + '_ {
        self.steps.iter().filter_map(|step| match step {
            AkitaPlannedStep::Fold(level) => Some(level.as_ref()),
            AkitaPlannedStep::Direct(_) => None,
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
    pub fn direct_step(&self) -> &AkitaPlannedDirectStep {
        match self
            .steps
            .last()
            .expect("planned schedule always contains at least one step")
        {
            AkitaPlannedStep::Direct(step) => step,
            AkitaPlannedStep::Fold(_) => {
                panic!("planned schedule must end in a direct packed-witness step")
            }
        }
    }

    /// Return the initial public witness state before any proof steps run.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without any steps.
    pub fn initial_state(&self) -> AkitaPlannedState {
        match self
            .steps
            .first()
            .expect("planned schedule always contains at least one step")
        {
            AkitaPlannedStep::Fold(level) => level.input_state(),
            AkitaPlannedStep::Direct(step) => step.state,
        }
    }

    /// Iterate over the planned witness states after each executed fold prefix.
    pub fn states(&self) -> impl Iterator<Item = AkitaPlannedState> + '_ {
        std::iter::once(self.initial_state())
            .chain(self.fold_levels().map(|level| level.output_state()))
    }

    /// Return the public witness state after `prefix_len` fold levels.
    pub fn state_after_prefix(&self, prefix_len: usize) -> Option<AkitaPlannedState> {
        if prefix_len == 0 {
            return Some(self.initial_state());
        }
        self.fold_levels()
            .nth(prefix_len - 1)
            .map(AkitaPlannedLevel::output_state)
    }

    /// Return the final witness state after all planned Akita levels.
    ///
    /// # Panics
    ///
    /// Panics if the schedule was constructed without a trailing direct step.
    pub fn terminal_state(&self) -> AkitaPlannedState {
        self.direct_step().state
    }

    /// Return the exact planned-state index matching public inputs and,
    /// optionally, an expected log-basis.
    pub fn exact_state_index(
        &self,
        inputs: AkitaScheduleInputs,
        log_basis: Option<u32>,
    ) -> Option<usize> {
        self.states().position(|state| {
            state.level == inputs.level
                && state.current_w_len == inputs.current_w_len
                && log_basis.is_none_or(|basis| state.log_basis == basis)
        })
    }
}

/// Render a stable identity for a planned schedule selected by public inputs.
pub fn planned_schedule_key_from_schedule(
    lookup_key: AkitaScheduleLookupKey,
    schedule: &AkitaSchedulePlan,
) -> String {
    let mut key = format!(
        "planner_v5_nv{}_g{}_t{}_w{}_z{}",
        lookup_key.num_vars,
        lookup_key.num_points,
        lookup_key.num_t_vectors,
        lookup_key.num_w_vectors,
        lookup_key.num_z_vectors
    );
    for state in schedule.states() {
        let _ = write!(key, "_l{}b{}", state.level, state.log_basis);
    }
    key
}

/// Resolve the exact planned fold execution matching a runtime public state.
///
/// # Errors
///
/// Returns an error if the matching fold is not followed by another planned
/// step. Returns `Ok(None)` when the requested state is absent or is not a fold
/// step.
pub fn exact_planned_level_execution<Stage1Config>(
    schedule: &AkitaSchedulePlan,
    inputs: AkitaScheduleInputs,
    log_basis: u32,
    stage1_challenge_config: Stage1Config,
) -> Result<Option<AkitaPlannedLevelExecution>, AkitaError>
where
    Stage1Config: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
{
    let Some(state_index) = schedule.exact_state_index(inputs, Some(log_basis)) else {
        return Ok(None);
    };
    let Some(current_step) = schedule.steps.get(state_index) else {
        return Ok(None);
    };
    let AkitaPlannedStep::Fold(current_level) = current_step else {
        return Ok(None);
    };
    let Some(next_step) = schedule.steps.get(state_index + 1) else {
        return Err(AkitaError::InvalidSetup(
            "planned fold step must be followed by another schedule step".to_string(),
        ));
    };
    let next_level_params = match next_step {
        AkitaPlannedStep::Fold(next_level) => next_level.lp.clone(),
        AkitaPlannedStep::Direct(direct) => {
            let (d, n_b) = match direct.witness_shape {
                DirectWitnessShape::PackedDigits(_) => {
                    let entry_d = current_level.lp.ring_dimension;
                    let entry_nb = current_level.next_commit_coeffs / entry_d;
                    (entry_d, entry_nb)
                }
                DirectWitnessShape::FieldElements(_) => (current_level.lp.ring_dimension, 0),
            };
            LevelParams::params_only(
                current_level.lp.a_key.sis_family(),
                d,
                direct.state.log_basis,
                0,
                n_b,
                0,
                stage1_challenge_config(d)?,
            )
        }
    };
    Ok(Some(AkitaPlannedLevelExecution {
        level: current_level.as_ref().clone(),
        next_level_params,
    }))
}

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::sis::compute_num_digits_full_field(field_bits, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
pub fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Total ring elements in the recursive witness polynomial.
///
/// Components: `e_hat + t_hat + B-blinding + decomposed z_pre + decomposed r`.
pub fn w_ring_element_count<F: CanonicalField>(lp: &LevelParams) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts::<F>(lp, 1, 1, 1, 1)
}

/// Total ring elements in a recursive witness polynomial for explicit batch counts.
pub fn w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts_for_layout::<F>(
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::WithDBlock,
    )
}

/// Total ring elements in a recursive witness polynomial for an explicit
/// M-row layout. The terminal layout drops the D-block from the M-matrix,
/// which shrinks the per-row `r` quotients by `n_d * r_decomp_levels` ring
/// elements relative to the intermediate layout.
pub fn w_ring_element_count_with_counts_for_layout<F: CanonicalField>(
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    layout: crate::layout::MRowLayout,
) -> Result<usize, AkitaError> {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        layout,
    )
}

/// Non-generic variant of [`w_ring_element_count_with_counts`] for callers
/// that already know the effective field bit width.
pub fn w_ring_element_count_with_counts_bits(
    field_bits: u32,
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts_for_layout_bits(
        field_bits,
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        crate::layout::MRowLayout::WithDBlock,
    )
}

/// Non-generic variant of [`w_ring_element_count_with_counts_for_layout`] for
/// callers that already know the effective field bit width. The planner
/// search uses this to keep its API free of a base-field type parameter.
pub fn w_ring_element_count_with_counts_for_layout_bits(
    field_bits: u32,
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    layout: crate::layout::MRowLayout,
) -> Result<usize, AkitaError> {
    w_ring_element_count_with_counts_for_layout_bits_and_quotient(
        field_bits,
        lp,
        num_points,
        num_t_vectors,
        num_w_vectors,
        num_public_rows,
        layout,
        TerminalWitnessQuotient::IncludeRHat,
    )
}

/// Terminal witness quotient emission policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalWitnessQuotient {
    /// Include decomposed `r_hat` quotient ring elements.
    IncludeRHat,
    /// Omit decomposed `r_hat` quotient ring elements.
    OmitRHat,
}

impl TerminalProofMode {
    /// Witness quotient policy for this terminal proof mode.
    ///
    /// Direct row checks need no `r_hat` quotient digits; the sumcheck path
    /// keeps them.
    #[inline]
    pub const fn terminal_witness_quotient(self) -> TerminalWitnessQuotient {
        match self {
            Self::RingSwitchSumcheck => TerminalWitnessQuotient::IncludeRHat,
            Self::DirectRingRelations => TerminalWitnessQuotient::OmitRHat,
        }
    }
}

/// Non-generic witness ring-element count with an explicit terminal quotient
/// policy. The quotient policy only affects terminal direct relation mode;
/// callers using intermediate layouts must keep [`TerminalWitnessQuotient::IncludeRHat`].
#[allow(clippy::too_many_arguments)]
pub fn w_ring_element_count_with_counts_for_layout_bits_and_quotient(
    field_bits: u32,
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_public_rows: usize,
    layout: crate::layout::MRowLayout,
    quotient: TerminalWitnessQuotient,
) -> Result<usize, AkitaError> {
    if layout != crate::layout::MRowLayout::Terminal
        && quotient == TerminalWitnessQuotient::OmitRHat
    {
        return Err(AkitaError::InvalidSetup(
            "r_hat omission is only valid for terminal layout".to_string(),
        ));
    }
    let w_hat_count = num_w_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_t_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    let z_pre_count = num_public_rows
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    let r_count = match quotient {
        TerminalWitnessQuotient::IncludeRHat => {
            // One public y-row per packaged public opening row.
            let r_rows = lp.m_row_count_for(num_points, num_public_rows, layout)?;
            r_rows
                .checked_mul(crate::layout::digit_math::compute_num_digits_full_field(
                    field_bits,
                    lp.log_basis,
                ))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("witness r-tail width overflow".to_string())
                })?
        }
        TerminalWitnessQuotient::OmitRHat => 0,
    };
    #[cfg(feature = "zk")]
    {
        // Terminal layout drops the D-block from the relation entirely, so
        // its per-row blinding is also unused. Intermediate layout keeps the
        // D-block blinding as before.
        let d_blinding_count = match layout {
            crate::layout::MRowLayout::WithDBlock => crate::zk::blinding_column_count_from_bits(
                lp.d_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                field_bits as usize,
            ),
            crate::layout::MRowLayout::WithoutDBlock => 0,
        };
        let b_blinding_count = num_points
            .checked_mul(crate::zk::blinding_column_count_from_bits(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                field_bits as usize,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("ZK B-blinding width overflow".to_string()))?;
        e_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(b_blinding_count))
            .and_then(|n| n.checked_add(d_blinding_count))
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
    #[cfg(not(feature = "zk"))]
    {
        e_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
}

/// Parameters for one fold level in the computed schedule.
#[derive(Clone, Debug)]
pub struct FoldStep {
    /// Unified level parameters (ring dimension, Ajtai keys, block geometry,
    /// digit depths, challenge config).
    pub params: LevelParams,
    /// Witness length entering this level.
    pub current_w_len: usize,
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
    /// Serialized terminal witness payload shape.
    pub witness_shape: CleartextWitnessShape,
    /// Direct witness bytes.
    pub direct_bytes: usize,
    /// Terminal proof mode for this direct handoff.
    pub terminal_proof_mode: TerminalProofMode,
    /// Commit-layout params for the root-direct case (the schedule's
    /// first step is this `Direct`). Three states:
    ///
    /// `Some(_)` is the root commit layout — the verifier replays
    /// commitments against it and the transcript binds it through the
    /// per-proof effective-schedule digest (`PlanSection`). `None` is the
    /// *uncommittable* edge: a table-recorded large-`num_vars` entry
    /// whose singleton root layout exceeds the audited SIS floor. The
    /// schedule is intentionally usable for proof-size exploration and
    /// DP planning, but `get_params_for_batched_commitment` rejects it
    /// loudly and `setup_level_params_from_runtime_schedule` returns
    /// an empty list. Don't commit through such a schedule.
    ///
    /// Terminal-direct steps (`witness_shape = PackedDigits`, schedule
    /// is `[Fold, …, Fold, Direct]`) ship the cleartext witness without
    /// committing — the verifier absorbs the bytes into the transcript
    /// and re-evaluates the witness directly. They always carry
    /// `params = None`. The active `log_basis` lives on
    /// [`Self::witness_shape`]; `scheduled_next_level_params`
    /// synthesizes a [`LevelParams::log_basis_stub`] from it so the
    /// prover's terminal-fold path still receives a `LevelParams`-shaped
    /// successor (only `log_basis` is consulted there).
    pub params: Option<LevelParams>,
}

impl DirectStep {
    /// Active terminal log-basis for packed direct witnesses.
    pub fn log_basis(&self, field_bits: u32) -> u32 {
        match self.witness_shape {
            CleartextWitnessShape::PackedDigits((_, bits)) => bits,
            CleartextWitnessShape::FieldElements(_) => field_bits,
        }
    }
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

impl Schedule {
    /// Iterate over the fold steps in execution order.
    pub fn fold_steps(&self) -> impl Iterator<Item = &FoldStep> + '_ {
        self.steps.iter().filter_map(|step| match step {
            Step::Fold(fold) => Some(fold),
            Step::Direct(_) => None,
        })
    }

    /// Number of fold levels before the terminal direct step.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_steps().count()
    }

    /// Witness length (field elements) entering the first step, or `None`
    /// when the schedule has no steps.
    pub fn initial_w_len(&self) -> Option<usize> {
        self.steps.first().map(|step| match step {
            Step::Fold(fold) => fold.current_w_len,
            Step::Direct(direct) => direct.current_w_len,
        })
    }

    /// Append the descriptor digest encoding for this effective schedule.
    ///
    /// Kept next to [`Schedule`] so protocol-affecting step field changes are
    /// reviewed with their Fiat-Shamir binding.
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.steps.len());
        for step in &self.steps {
            match step {
                Step::Fold(fold) => {
                    bytes.push(0);
                    fold.params.append_descriptor_bytes(bytes);
                    push_usize(bytes, fold.current_w_len);
                    push_usize(bytes, fold.next_w_len);
                    push_usize(bytes, fold.level_bytes);
                }
                Step::Direct(direct) => {
                    bytes.push(1);
                    push_usize(bytes, direct.current_w_len);
                    append_direct_witness_shape_descriptor_bytes(bytes, &direct.witness_shape);
                    push_usize(bytes, direct.direct_bytes);
                    bytes.push(terminal_proof_mode_descriptor_tag(
                        direct.terminal_proof_mode,
                    ));
                }
            }
        }
        push_usize(bytes, self.total_bytes);
    }
}

fn append_direct_witness_shape_descriptor_bytes(
    bytes: &mut Vec<u8>,
    shape: &CleartextWitnessShape,
) {
    match shape {
        CleartextWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
            bytes.push(0);
            push_usize(bytes, *num_elems);
            push_u32(bytes, *bits_per_elem);
        }
        CleartextWitnessShape::FieldElements(coeff_len) => {
            bytes.push(1);
            push_usize(bytes, *coeff_len);
        }
    }
}

fn terminal_proof_mode_descriptor_tag(mode: TerminalProofMode) -> u8 {
    match mode {
        TerminalProofMode::RingSwitchSumcheck => 0,
        TerminalProofMode::DirectRingRelations => 1,
    }
}

/// Witness length entering the root fold, in field elements.
pub fn root_current_w_len(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(lp.ring_dimension))
        .unwrap_or(0)
}

/// Build the root-direct schedule for roots that do not admit a fold step.
///
/// `commit_params` carries the root commit layout that
/// `Cfg::get_params_for_batched_commitment` returns for this schedule shape.
///
/// # Errors
///
/// Returns an error if `num_vars` cannot be represented as a witness length.
pub fn root_direct_schedule(
    num_vars: usize,
    commit_params: LevelParams,
) -> Result<Schedule, AkitaError> {
    let current_w_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root-direct witness length overflow".to_string())
    })?;
    Ok(Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len,
            witness_shape: CleartextWitnessShape::FieldElements(current_w_len),
            direct_bytes: 0,
            terminal_proof_mode: TerminalProofMode::RingSwitchSumcheck,
            commit_params: Some(commit_params),
            // Root-direct never has a "next level after itself"; the
            // schedule walks the single direct step and stops.
            level_params: None,
        })],
        total_bytes: 0,
    })
}

/// Scale a per-polynomial root layout to a batched root layout.
///
/// # Errors
///
/// Returns an error when `num_claims` is zero or scaling overflows a layout
/// width.
pub fn scale_batched_root_layout(
    root_lp: &LevelParams,
    num_claims: usize,
    root_stage1_l1_mass: usize,
    field_bits: u32,
) -> Result<LevelParams, AkitaError> {
    if num_claims == 0 {
        return Err(AkitaError::InvalidSetup(
            "max_num_batched_polys must be at least 1".to_string(),
        ));
    }

    let mut scaled = root_lp.clone();
    let d = scaled.ring_dimension;
    scaled.b_key = crate::AjtaiKeyParams::try_new(
        scaled.b_key.sis_family(),
        scaled.b_key.row_len(),
        root_lp
            .b_key
            .col_len()
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("batched outer width overflow".to_string()))?,
        scaled.b_key.collision_inf(),
        d,
    )?;
    scaled.d_key = crate::AjtaiKeyParams::try_new(
        scaled.d_key.sis_family(),
        scaled.d_key.row_len(),
        root_lp
            .d_key
            .col_len()
            .checked_mul(num_claims)
            .ok_or_else(|| AkitaError::InvalidSetup("batched D width overflow".to_string()))?,
        scaled.d_key.collision_inf(),
        d,
    )?;
    scaled.num_digits_fold = root_lp.num_digits_fold.max(
        crate::layout::digit_math::compute_num_digits_fold_with_claims(
            root_lp.r_vars,
            root_stage1_l1_mass,
            root_lp.log_basis,
            num_claims,
            field_bits,
        ),
    );
    Ok(scaled)
}

/// Extract the per-polynomial layout from a batched root layout.
pub fn split_batched_root_params(root_lp: &LevelParams, field_bits: u32) -> LevelParams {
    let per_poly_fold = crate::layout::digit_math::compute_num_digits_fold_with_claims(
        root_lp.r_vars,
        root_lp.challenge_l1_mass(),
        root_lp.log_basis,
        1,
        field_bits,
    );
    let mut lp = root_lp.clone();
    lp.num_digits_fold = per_poly_fold;
    lp
}

/// Extract a per-polynomial batched root layout from the first fold level in a
/// pre-computed schedule plan.
pub fn split_batched_root_params_from_schedule_plan(
    plan: &AkitaSchedulePlan,
    field_bits: u32,
) -> Option<LevelParams> {
    let root_level = plan.fold_levels().next()?;
    Some(split_batched_root_params(&root_level.lp, field_bits))
}

/// Translate an offline [`AkitaSchedulePlan`] into the runtime [`Schedule`]
/// format.
///
/// `field_bits` is used only for terminal direct witnesses encoded as field
/// elements; packed-digit direct witnesses carry their own bit width.
pub fn schedule_from_plan(plan: &AkitaSchedulePlan, field_bits: u32) -> Schedule {
    let mut steps = Vec::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            AkitaPlannedStep::Fold(level) => {
                let lp = level.lp.clone();
                let delta_fold_per_poly =
                    crate::layout::digit_math::compute_num_digits_fold_with_claims(
                        lp.r_vars,
                        lp.challenge_l1_mass(),
                        lp.log_basis,
                        1,
                        field_bits,
                    );
                let ring_dim = lp.ring_dimension;
                let next_w_len = level.next_inputs.current_w_len;
                let w_ring = next_w_len / ring_dim;
                steps.push(Step::Fold(FoldStep {
                    params: lp,
                    current_w_len: level.inputs.current_w_len,
                    delta_fold_per_poly,
                    w_ring,
                    next_w_len,
                    level_bytes: level.level_bytes,
                }));
            }
            AkitaPlannedStep::Direct(direct) => {
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct.state.current_w_len,
                    witness_shape: direct.witness_shape.clone(),
                    direct_bytes: direct.direct_bytes,
                    terminal_proof_mode: direct.terminal_proof_mode,
                    commit_params: direct.commit_params.clone(),
                    level_params: direct.level_params.clone(),
                }));
            }
        }
    }
    Schedule {
        steps,
        total_bytes: plan.exact_proof_bytes,
    }
}

/// Return the number of fold levels in a runtime schedule.
pub fn schedule_num_fold_levels(schedule: &Schedule) -> usize {
    schedule
        .steps
        .iter()
        .filter(|step| matches!(step, Step::Fold(_)))
        .count()
}

/// Return whether a runtime schedule uses the root-direct fast path.
pub fn schedule_is_root_direct(schedule: &Schedule) -> bool {
    matches!(schedule.steps.first(), Some(Step::Direct(_)))
}

/// Return the root fold step when a runtime schedule starts with one.
pub fn schedule_root_fold_step(schedule: &Schedule) -> Option<&FoldStep> {
    match schedule.steps.first() {
        Some(Step::Fold(step)) => Some(step),
        Some(Step::Direct(_)) | None => None,
    }
}

/// Return the terminal direct witness shape from a runtime schedule.
///
/// # Errors
///
/// Returns an error if the schedule does not end in a direct witness handoff.
pub fn schedule_terminal_direct_witness_shape(
    schedule: &Schedule,
) -> Result<&CleartextWitnessShape, AkitaError> {
    match schedule.steps.last() {
        Some(Step::Direct(step)) => Ok(&step.witness_shape),
        Some(Step::Fold(_)) => Err(AkitaError::InvalidSetup(
            "schedule must end in a terminal direct witness step".to_string(),
        )),
        None => Err(AkitaError::InvalidSetup(
            "schedule is missing terminal direct witness step".to_string(),
        )),
    }
}

/// Resolve one scheduled level's active Akita params.
///
/// `Fold` steps return the baked-in `params` set by the planner DP and
/// table materializer. A terminal `Direct(PackedDigits)` step has no
/// commitment of its own (the cleartext witness is absorbed into the
/// transcript directly), so it ships no `LevelParams`; this function
/// instead returns a [`LevelParams::log_basis_stub`] carrying only the
/// active `log_basis` read off `witness_shape`. The only caller that
/// actually consumes a field of the terminal-Direct successor is the
/// prover's terminal-fold path, which reads `log_basis`.
///
/// # Errors
///
/// Returns an error when `step_index` is outside the schedule or when a
/// recursive schedule transitions into a `Direct(FieldElements)` (only
/// the *first* step of a root-direct schedule may carry that shape).
pub fn scheduled_next_level_params(
    schedule: &Schedule,
    step_index: usize,
) -> Result<LevelParams, AkitaError> {
    match schedule.steps.get(step_index) {
        Some(Step::Fold(step)) => Ok(step.params.clone()),
        Some(Step::Direct(step)) => match step.witness_shape {
            CleartextWitnessShape::PackedDigits((_, log_basis)) => {
                Ok(LevelParams::log_basis_stub(log_basis))
            }
            CleartextWitnessShape::FieldElements(_) => Err(AkitaError::InvalidSetup(
                "recursive schedule cannot transition into a field-element direct step".to_string(),
            )),
        },
        None => Err(AkitaError::InvalidSetup(
            "schedule is missing successor step".to_string(),
        )),
    }
}

/// Resolve the current fold params and successor params for a scheduled fold.
///
/// This validates that the runtime witness length and log-basis agree with the
/// selected planner schedule before deriving the next level params.
///
/// # Errors
///
/// Returns an error if `level` is not a fold step or if the runtime state does
/// not match the scheduled fold.
pub fn scheduled_fold_execution(
    schedule: &Schedule,
    level: usize,
    inputs: AkitaScheduleInputs,
    current_log_basis: u32,
) -> Result<(LevelParams, LevelParams), AkitaError> {
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != inputs.current_w_len || step.params.log_basis != current_log_basis {
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled recursive level {level} did not match runtime state: \
             expected_w_len={}, actual_w_len={}, expected_log_basis={}, actual_log_basis={}",
            step.current_w_len, inputs.current_w_len, step.params.log_basis, current_log_basis
        )));
    }
    let next_level_params = scheduled_next_level_params(schedule, level + 1)?;
    Ok((step.params.clone(), next_level_params))
}

#[cfg(test)]
mod tests;
