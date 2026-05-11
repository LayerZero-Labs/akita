//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::generated::{
    generated_direct_log_basis, generated_direct_witness_shape, generated_step_current_w_len,
    table_entry, GeneratedDirectWitnessShape, GeneratedFoldStep, GeneratedScheduleKey,
    GeneratedScheduleTable, GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::{
    direct_witness_bytes, level_layout_from_params, level_proof_bytes,
    recursive_level_decomposition_from_root, DecompositionParams, DirectWitnessShape, LevelParams,
    RingOpeningPoint, SisModulusFamily,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use std::fmt::Write;

/// Public inputs that deterministically select one level's active Akita params.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleInputs {
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
pub struct AkitaRootBatchSummary {
    /// Total number of flattened root claims.
    pub num_claims: usize,
    /// Number of committed root groups.
    pub num_commitment_groups: usize,
    /// Number of distinct opening points.
    pub num_points: usize,
}

impl AkitaRootBatchSummary {
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
    ) -> Result<Self, AkitaError> {
        if num_claims == 0 {
            return Err(AkitaError::InvalidInput(
                "root batching requires at least one claim".to_string(),
            ));
        }
        if num_commitment_groups == 0 {
            return Err(AkitaError::InvalidInput(
                "root batching requires at least one commitment group".to_string(),
            ));
        }
        if num_points == 0 {
            return Err(AkitaError::InvalidInput(
                "root batching requires at least one opening point".to_string(),
            ));
        }
        if num_commitment_groups > num_claims {
            return Err(AkitaError::InvalidInput(format!(
                "root batching has {num_commitment_groups} commitment groups but only {num_claims} claims"
            )));
        }
        if num_points > num_claims {
            return Err(AkitaError::InvalidInput(format!(
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
    pub fn from_group_poly_counts(
        group_poly_counts: &[usize],
        num_points: usize,
    ) -> Result<Self, AkitaError> {
        if group_poly_counts.is_empty() {
            return Err(AkitaError::InvalidInput(
                "root batching requires at least one commitment group".to_string(),
            ));
        }
        if let Some(group_idx) = group_poly_counts.iter().position(|&size| size == 0) {
            return Err(AkitaError::InvalidInput(format!(
                "root batching group {group_idx} must be nonempty"
            )));
        }
        let num_claims = group_poly_counts.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                AkitaError::InvalidInput("root batching total claim count overflow".to_string())
            })
        })?;
        Self::new(num_claims, group_poly_counts.len(), num_points)
    }
}

/// Return the total number of claims represented by nonempty claim groups.
///
/// # Errors
///
/// Returns an error when the group list is empty, contains an empty group, or
/// overflows `usize`.
pub fn checked_num_claims_from_group_sizes(
    group_poly_counts: &[usize],
) -> Result<usize, AkitaError> {
    if group_poly_counts.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "claim groups must be nonempty".to_string(),
        ));
    }
    group_poly_counts
        .iter()
        .try_fold(0usize, |acc, &group_size| {
            if group_size == 0 {
                return Err(AkitaError::InvalidSetup(
                    "claim groups must be nonempty".to_string(),
                ));
            }
            acc.checked_add(group_size)
                .ok_or_else(|| AkitaError::InvalidSetup("claim group count overflow".to_string()))
        })
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AkitaScheduleLookupKey {
    /// Setup/public schedule bucket.
    pub max_num_vars: usize,
    /// Actual root polynomial arity.
    pub num_vars: usize,
    /// Number of claims the root commitment layout was sized for at commit
    /// time. This can exceed `batch.num_claims`.
    pub layout_num_claims: usize,
    /// Aggregate opening-batch summary for the concrete invocation.
    pub batch: AkitaRootBatchSummary,
}

impl AkitaScheduleLookupKey {
    /// Singleton root-opening context.
    pub const fn singleton(max_num_vars: usize, num_vars: usize, layout_num_claims: usize) -> Self {
        Self {
            max_num_vars,
            num_vars,
            layout_num_claims,
            batch: AkitaRootBatchSummary::singleton(),
        }
    }

    /// General root-opening context.
    pub const fn with_batch(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
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
pub const fn generated_schedule_lookup_key(key: AkitaScheduleLookupKey) -> GeneratedScheduleKey {
    GeneratedScheduleKey {
        max_num_vars: key.max_num_vars,
        num_vars: key.num_vars,
        layout_num_claims: key.layout_num_claims,
        batch_num_claims: key.batch.num_claims,
        batch_num_commitment_groups: key.batch.num_commitment_groups,
        batch_num_points: key.batch.num_points,
    }
}

fn generated_level_params<Stage1Config>(
    sis_family: SisModulusFamily,
    step: GeneratedFoldStep,
    context: &str,
    stage1_challenge_config: &Stage1Config,
) -> Result<LevelParams, AkitaError>
where
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
{
    let stage1_config = stage1_challenge_config(step.d as usize);
    let params = LevelParams::params_only(
        sis_family,
        step.d as usize,
        step.log_basis,
        step.n_a as usize,
        step.n_b as usize,
        step.n_d as usize,
        stage1_config,
    );
    if step.challenge_l1_mass != params.challenge_l1_mass() {
        return Err(AkitaError::InvalidSetup(format!(
            "generated schedule {context} challenge L1 mass mismatch: pinned={}, runtime={}",
            step.challenge_l1_mass,
            params.challenge_l1_mass()
        )));
    }
    Ok(params)
}

fn w_ring_element_count_with_batch_summary_bits<F: CanonicalField>(
    field_bits: u32,
    lp: &LevelParams,
    batch: AkitaRootBatchSummary,
) -> usize {
    let _field_marker = core::marker::PhantomData::<F>;
    let w_hat_count = batch.num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat_count = batch.num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre_count = batch.num_points * lp.inner_width() * lp.num_digits_fold;
    let r_rows = lp.m_row_count(batch.num_commitment_groups, batch.num_points);
    let r_count =
        r_rows * crate::layout::digit_math::compute_num_digits_full_field(field_bits, lp.log_basis);
    #[cfg(feature = "zk")]
    {
        let blinding_count = batch.num_commitment_groups
            * crate::zk::blinding_column_count_from_bits(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                field_bits as usize,
            );
        w_hat_count + t_hat_count + blinding_count + z_pre_count + r_count
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat_count + t_hat_count + z_pre_count + r_count
    }
}

/// Materialize and validate a generated schedule-table entry into a planned
/// runtime schedule.
///
/// `stage1_challenge_config` and `scale_batched_root_layout` are the only
/// config-specific hooks: generated table validation, direct witness sizing,
/// level layout assembly, next-witness sizing, and proof-byte sizing are shared
/// by `akita-types`.
///
/// # Errors
///
/// Returns an error if the generated entry is structurally invalid, does not
/// match `key`, or does not agree with the supplied config policy callbacks.
pub fn schedule_plan_from_generated_entry<F, Stage1Config, ScaleBatchedRoot>(
    key: AkitaScheduleLookupKey,
    entry: &GeneratedScheduleTableEntry,
    sis_family: SisModulusFamily,
    root_decomp: DecompositionParams,
    stage1_challenge_config: Stage1Config,
    scale_batched_root_layout: ScaleBatchedRoot,
) -> Result<AkitaSchedulePlan, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
    ScaleBatchedRoot: Fn(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
{
    let Some(root_step) = entry.steps.first() else {
        return Err(AkitaError::InvalidSetup(
            "generated schedule table entry must contain at least one step".to_string(),
        ));
    };
    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;
    if generated_step_current_w_len(root_step) != expected_root_w_len {
        return Err(AkitaError::InvalidSetup(format!(
            "generated root witness length {} does not match key={key:?}",
            generated_step_current_w_len(root_step)
        )));
    }

    let field_bits = root_decomp.field_bits();
    let mut steps = Vec::with_capacity(entry.steps.len().max(1));
    let mut fold_level = 0usize;

    for (step_index, generated_step) in entry.steps.iter().enumerate() {
        match generated_step {
            GeneratedStep::Fold(level) => {
                let Some(next_generated_step) = entry.steps.get(step_index + 1) else {
                    return Err(AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    )));
                };
                let next_current_w_len = generated_step_current_w_len(next_generated_step);
                if level.next_w_len != next_current_w_len {
                    return Err(AkitaError::InvalidSetup(format!(
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
                            return Err(AkitaError::InvalidSetup(format!(
                                "generated schedule level {fold_level} cannot transition into a field-element direct step"
                            )))
                        }
                    },
                };

                let inputs = AkitaScheduleInputs {
                    max_num_vars: key.max_num_vars,
                    level: fold_level,
                    current_w_len: level.current_w_len,
                };
                let next_inputs = AkitaScheduleInputs {
                    max_num_vars: key.max_num_vars,
                    level: fold_level + 1,
                    current_w_len: next_current_w_len,
                };
                let params = generated_level_params(
                    sis_family,
                    *level,
                    &format!("level {fold_level}"),
                    &stage1_challenge_config,
                )?;
                let level_decomp = if fold_level == 0 {
                    DecompositionParams {
                        log_basis: level.log_basis,
                        ..root_decomp
                    }
                } else {
                    recursive_level_decomposition_from_root(root_decomp, level.log_basis)
                };
                let layout = level_layout_from_params(
                    level.m_vars as usize,
                    level.r_vars as usize,
                    &params,
                    level_decomp,
                    level.current_w_len / level.d as usize,
                )?;
                let root_is_batched =
                    fold_level == 0 && key.batch != AkitaRootBatchSummary::singleton();
                let mut lp = params.with_layout(&layout);
                if root_is_batched {
                    lp = scale_batched_root_layout(&lp, key.batch.num_claims)?;
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
                    let next_w_ring = w_ring_element_count_with_batch_summary_bits::<F>(
                        field_bits, &lp, key.batch,
                    );
                    next_w_ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated root next witness length overflow".to_string(),
                        )
                    })?
                } else {
                    w_ring_element_count_with_batch_summary_bits::<F>(
                        field_bits,
                        &lp,
                        AkitaRootBatchSummary::singleton(),
                    ) * lp.ring_dimension
                };
                if runtime_next_w_len != level.next_w_len {
                    return Err(AkitaError::InvalidSetup(format!(
                        "generated next_w_len mismatch at level {fold_level}: pinned={}, runtime={runtime_next_w_len}",
                        level.next_w_len
                    )));
                }

                let (next_level_params, next_commit_coeffs) = match next_generated_step {
                    GeneratedStep::Fold(next_level) => {
                        let next_level_params = generated_level_params(
                            sis_family,
                            *next_level,
                            &format!("next level {}", fold_level + 1),
                            &stage1_challenge_config,
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
                                return Err(AkitaError::InvalidSetup(
                                    "generated direct entry commitment must specify both D and n_b or neither"
                                        .to_string(),
                                ))
                            }
                        };
                        (
                            LevelParams::params_only(
                                sis_family,
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
                    level_proof_bytes(
                        field_bits,
                        &lp,
                        &lp,
                        &next_level_params,
                        next_inputs.current_w_len,
                        1,
                    )
                };

                steps.push(AkitaPlannedStep::Fold(Box::new(AkitaPlannedLevel {
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
                    return Err(AkitaError::InvalidSetup(
                        "generated direct step must be terminal".to_string(),
                    ));
                }
                let witness_shape = generated_direct_witness_shape(direct.witness_shape);
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);
                if direct_bytes != direct.direct_bytes {
                    return Err(AkitaError::InvalidSetup(format!(
                        "generated direct bytes mismatch at terminal step: pinned={}, runtime={direct_bytes}",
                        direct.direct_bytes
                    )));
                }
                if !matches!(
                    (direct.entry_d, direct.entry_nb),
                    (Some(_), Some(_)) | (None, None)
                ) {
                    return Err(AkitaError::InvalidSetup(
                        "generated direct entry commitment must specify both D and n_b or neither"
                            .to_string(),
                    ));
                }

                let state = AkitaPlannedState {
                    level: fold_level,
                    current_w_len: direct.current_w_len,
                    log_basis: generated_direct_log_basis(
                        direct.witness_shape,
                        root_decomp.log_basis,
                    ),
                };
                steps.push(AkitaPlannedStep::Direct(AkitaPlannedDirectStep {
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
            AkitaPlannedStep::Fold(level) => level.level_bytes,
            AkitaPlannedStep::Direct(step) => step.direct_bytes,
        })
        .sum();
    Ok(AkitaSchedulePlan {
        steps,
        no_wrapper_bytes,
        exact_proof_bytes: no_wrapper_bytes,
    })
}

/// Look up and materialize a generated schedule-table entry.
///
/// # Errors
///
/// Returns an error if a matching generated entry exists but fails validation
/// against the supplied config policy callbacks.
pub fn generated_schedule_plan_from_table<F, Stage1Config, ScaleBatchedRoot>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
    sis_family: SisModulusFamily,
    root_decomp: DecompositionParams,
    stage1_challenge_config: Stage1Config,
    scale_batched_root_layout: ScaleBatchedRoot,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
    ScaleBatchedRoot: Fn(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
{
    table_entry(table, generated_schedule_lookup_key(key))
        .map(|entry| {
            schedule_plan_from_generated_entry::<F, _, _>(
                key,
                entry,
                sis_family,
                root_decomp,
                stage1_challenge_config,
                scale_batched_root_layout,
            )
        })
        .transpose()
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
    /// The terminal packed-witness direct handoff.
    Direct(AkitaPlannedDirectStep),
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

/// Resolve the active planned log-basis for public schedule inputs.
///
/// # Errors
///
/// Returns an error when the schedule does not include the requested public
/// state.
pub fn planned_log_basis_at_level_from_schedule(
    schedule: &AkitaSchedulePlan,
    inputs: AkitaScheduleInputs,
) -> Result<u32, AkitaError> {
    if let Some(state) = schedule
        .exact_state_index(inputs, None)
        .and_then(|state_index| schedule.state_after_prefix(state_index))
    {
        return Ok(state.log_basis);
    }
    Err(AkitaError::InvalidSetup(format!(
        "no planned log basis for inputs={inputs:?}: schedule does not include this state"
    )))
}

/// Render a stable identity for a planned schedule selected by public inputs.
pub fn planned_schedule_key_from_schedule(
    lookup_key: AkitaScheduleLookupKey,
    schedule: &AkitaSchedulePlan,
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
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
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
                stage1_challenge_config(d),
            )
        }
    };
    Ok(Some(AkitaPlannedLevelExecution {
        level: current_level.as_ref().clone(),
        next_level_params,
    }))
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
    fn schedule_key(key: AkitaScheduleLookupKey) -> String;

    /// Optional full schedule plan for configs with an explicit provider.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider cannot materialize a valid schedule.
    fn schedule_plan(key: AkitaScheduleLookupKey) -> Result<Option<AkitaSchedulePlan>, AkitaError>;
}

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::layout::digit_math::compute_num_digits_full_field(field_bits, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
pub fn detect_field_modulus<F: CanonicalField>() -> u128 {
    (-F::one()).to_canonical_u128() + 1
}

/// Total ring elements in the recursive witness polynomial.
///
/// Components: `w_hat + t_hat + B-blinding + decomposed z_pre + decomposed r`.
pub fn w_ring_element_count<F: CanonicalField>(lp: &LevelParams) -> usize {
    w_ring_element_count_with_counts::<F>(lp, 1, 1, 1)
}

/// Total ring elements in a recursive witness polynomial for explicit batch counts.
pub fn w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &LevelParams,
    num_claims: usize,
    num_commitment_groups: usize,
    num_points: usize,
) -> usize {
    let w_hat_count = num_claims * lp.num_blocks * lp.num_digits_open;
    let t_hat_count = num_claims * lp.num_blocks * lp.a_key.row_len() * lp.num_digits_open;
    let z_pre_count = num_points * lp.inner_width() * lp.num_digits_fold;
    // One public y-row per distinct opening point (batched_cwss_proof.tex §6).
    let r_rows = lp.m_row_count(num_commitment_groups, num_points);
    let r_count = r_rows * r_decomp_levels::<F>(lp.log_basis);
    #[cfg(feature = "zk")]
    {
        let blinding_count = num_commitment_groups
            * crate::zk::blinding_column_count::<F>(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
            );
        w_hat_count + t_hat_count + blinding_count + z_pre_count + r_count
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat_count + t_hat_count + z_pre_count + r_count
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

/// Witness length entering the root fold, in field elements.
pub fn root_current_w_len(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(lp.ring_dimension))
        .unwrap_or(0)
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
                let bits_per_elem = match direct.witness_shape {
                    DirectWitnessShape::PackedDigits((_, bits)) => bits,
                    DirectWitnessShape::FieldElements(_) => field_bits,
                };
                steps.push(Step::Direct(DirectStep {
                    current_w_len: direct.state.current_w_len,
                    bits_per_elem,
                    direct_bytes: direct.direct_bytes,
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

/// Resolve one scheduled level's active Akita params.
///
/// Fold steps carry concrete params in the schedule. Direct steps only carry
/// the terminal packed basis, so callers provide the config-specific direct
/// param derivation callback.
///
/// # Errors
///
/// Returns an error when `step_index` is outside the schedule.
pub fn scheduled_next_level_params<DirectParams>(
    schedule: &Schedule,
    step_index: usize,
    inputs: AkitaScheduleInputs,
    direct_params: DirectParams,
) -> Result<LevelParams, AkitaError>
where
    DirectParams: FnOnce(AkitaScheduleInputs, u32) -> LevelParams,
{
    match schedule.steps.get(step_index) {
        Some(Step::Fold(step)) => Ok(step.params.clone()),
        Some(Step::Direct(step)) => Ok(direct_params(inputs, step.bits_per_elem)),
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
pub fn scheduled_fold_execution<DirectParams>(
    schedule: &Schedule,
    level: usize,
    inputs: AkitaScheduleInputs,
    current_log_basis: u32,
    direct_params: DirectParams,
) -> Result<(LevelParams, LevelParams), AkitaError>
where
    DirectParams: FnOnce(AkitaScheduleInputs, u32) -> LevelParams,
{
    let Some(Step::Fold(step)) = schedule.steps.get(level) else {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule is missing fold step at level {level}"
        )));
    };
    if step.current_w_len != inputs.current_w_len || step.params.log_basis != current_log_basis {
        return Err(AkitaError::InvalidSetup(
            "scheduled recursive level did not match runtime state".to_string(),
        ));
    }
    let next_inputs = AkitaScheduleInputs {
        max_num_vars: inputs.max_num_vars,
        level: level + 1,
        current_w_len: step.next_w_len,
    };
    let next_level_params =
        scheduled_next_level_params(schedule, level + 1, next_inputs, direct_params)?;
    Ok((step.params.clone(), next_level_params))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        stage1_tree_stage_shapes, sumcheck_rounds, AjtaiKeyParams, AkitaBatchedRootProof,
        AkitaLevelProof, AkitaStage1Proof, AkitaStage1StageProof, AkitaStage2Proof, FlatRingVec,
    };
    use akita_algebra::CyclotomicRing;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::FieldCore;
    use akita_field::Prime128OffsetA7F7;
    use akita_serialization::{AkitaSerialize, Compress};
    use akita_sumcheck::{
        CompressedUniPoly, EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof,
    };

    type F = Prime128OffsetA7F7;

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

    fn dummy_stage1_proof<F: FieldCore>(rounds: usize, b: usize) -> AkitaStage1Proof<F> {
        AkitaStage1Proof {
            stages: stage1_tree_stage_shapes(rounds, b)
                .into_iter()
                .map(|shape| AkitaStage1StageProof {
                    sumcheck: dummy_eq_factored_sumcheck(rounds, shape.sumcheck.1),
                    child_claims: vec![F::zero(); shape.child_claims],
                })
                .collect(),
            s_claim: F::zero(),
        }
    }

    fn exact_level_proof_bytes<F: FieldCore + AkitaSerialize>(
        lp: &LevelParams,
        next_lp: &LevelParams,
        next_w_len: usize,
    ) -> Result<usize, AkitaError> {
        let current_coeffs = lp
            .d_key
            .row_len()
            .checked_mul(lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let next_commit_coeffs = next_lp
            .b_key
            .row_len()
            .checked_mul(next_lp.ring_dimension)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("recursive proof sizing overflow".to_string())
            })?;
        let rounds = sumcheck_rounds(lp.ring_dimension, next_w_len);
        let b = 1usize << lp.log_basis;

        let proof = AkitaLevelProof {
            y_ring: FlatRingVec::from_coeffs(vec![F::zero(); lp.ring_dimension]),
            v: FlatRingVec::from_coeffs(vec![F::zero(); current_coeffs]),
            stage1: dummy_stage1_proof(rounds, b),
            stage2: AkitaStage2Proof {
                sumcheck: dummy_sumcheck(rounds, 3),
                next_w_commitment: FlatRingVec::from_coeffs(vec![F::zero(); next_commit_coeffs]),
                next_w_eval: F::zero(),
            },
        };
        Ok(proof.serialized_size(Compress::No))
    }

    #[test]
    fn planned_level_bytes_match_two_stage_payload_at_all_bases() {
        const D: usize = 64;
        let stage1_config = SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        };
        let next_lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, stage1_config.clone());
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams::params_only(
                SisModulusFamily::Q128,
                D,
                log_basis,
                2,
                2,
                2,
                stage1_config.clone(),
            )
            .with_decomp(0, 0, 1, 1, 1, 0)
            .unwrap();
            assert_eq!(
                level_proof_bytes(128, &lp, &lp, &next_lp, next_w_len, 1),
                exact_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
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
        let next_lp =
            LevelParams::params_only(SisModulusFamily::Q128, D, 2, 2, 3, 2, stage1_config.clone());
        let next_w_len = D * 8;

        for log_basis in 2..=6 {
            let lp = LevelParams {
                ring_dimension: D,
                log_basis,
                a_key: AjtaiKeyParams::new(SisModulusFamily::Q128, 2, 1, 0, D),
                b_key: AjtaiKeyParams::new(SisModulusFamily::Q128, 2, 1, 0, D),
                d_key: AjtaiKeyParams::new(SisModulusFamily::Q128, 2, 1, 0, D),
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
            let root_proof = AkitaBatchedRootProof::new_two_stage::<D>(
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
    fn root_batch_summary_tracks_only_aggregate_counts() {
        let a = AkitaRootBatchSummary::from_group_poly_counts(&[1, 1, 4], 2).unwrap();
        let b = AkitaRootBatchSummary::from_group_poly_counts(&[2, 2, 2], 2).unwrap();
        let c = AkitaRootBatchSummary::from_group_poly_counts(&[3, 3], 2).unwrap();

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(AkitaRootBatchSummary::singleton().num_claims, 1);
    }
}
