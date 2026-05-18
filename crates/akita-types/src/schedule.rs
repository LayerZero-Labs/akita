//! Runtime schedule shapes shared by configs, prover, verifier, and planner.

use crate::generated::{
    table_entry, GeneratedFoldStep, GeneratedScheduleKey, GeneratedScheduleTable,
    GeneratedScheduleTableEntry, GeneratedStep,
};
use crate::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, level_layout_from_params,
    level_proof_bytes, recursive_level_decomposition_from_root, ClaimIncidenceSummary,
    DecompositionParams, DirectWitnessShape, LevelParams, RingOpeningPoint, SisModulusFamily,
};
use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use std::fmt::Write;

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
/// stability; the translation happens in `generated_schedule_lookup_key`.
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

fn generated_level_params<Stage1Config>(
    sis_family: SisModulusFamily,
    step: GeneratedFoldStep,
    stage1_challenge_config: &Stage1Config,
) -> LevelParams
where
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
{
    let stage1_config = stage1_challenge_config(step.ring_d as usize);
    LevelParams::params_only(
        sis_family,
        step.ring_d as usize,
        step.log_basis,
        step.n_a as usize,
        step.n_b as usize,
        step.n_d as usize,
        stage1_config,
    )
}

fn w_ring_element_count_with_vector_counts_bits<F: CanonicalField>(
    field_bits: u32,
    lp: &LevelParams,
    num_points: usize,
    num_t_vectors: usize,
    num_w_vectors: usize,
    num_z_vectors: usize,
) -> Result<usize, AkitaError> {
    let _field_marker = core::marker::PhantomData::<F>;
    let w_hat_count = num_w_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_t_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let z_pre_count = num_z_vectors
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(lp.num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    let r_rows = lp.m_row_count(num_points, num_z_vectors)?;
    let r_count = r_rows
        .checked_mul(crate::layout::digit_math::compute_num_digits_full_field(
            field_bits,
            lp.log_basis,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("witness r-tail width overflow".to_string()))?;
    #[cfg(feature = "zk")]
    {
        let d_blinding_count = crate::zk::blinding_column_count_from_bits(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
            field_bits as usize,
        );
        let b_blinding_count = num_points
            .checked_mul(crate::zk::blinding_column_count_from_bits(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
                field_bits as usize,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("ZK B-blinding width overflow".to_string()))?;
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(b_blinding_count))
            .and_then(|n| n.checked_add(d_blinding_count))
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
}

/// Config-specific policy hooks needed to materialize generated schedule-table
/// entries into runtime schedules.
pub struct GeneratedSchedulePlanPolicy<Stage1Config, ScaleBatchedRoot, DirectLevelParams> {
    /// SIS modulus family used by generated fold levels.
    pub sis_family: SisModulusFamily,
    /// Root-level digit decomposition used to interpret generated entries.
    pub root_decomp: DecompositionParams,
    /// Challenge-field width used for verifier challenges and proof-byte accounting.
    pub challenge_field_bits: u32,
    /// Number of public rows in recursive fold levels.
    pub recursive_public_rows: usize,
    /// Base-field width of the logical extension opening. This is `1` for the
    /// ordinary base-field path, which has no extension-opening reduction.
    pub extension_opening_width: usize,
    /// Stage-1 sparse challenge policy for each ring dimension.
    pub stage1_challenge_config: Stage1Config,
    /// Root-layout scaler for batched committed openings.
    pub scale_batched_root_layout: ScaleBatchedRoot,
    /// Direct terminal layout policy for a schedule state and log-basis.
    pub direct_level_params: DirectLevelParams,
}

fn padded_boolean_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: AkitaScheduleLookupKey,
    current_w_len: usize,
) -> Result<usize, AkitaError> {
    if extension_opening_width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (
            key.num_w_vectors
                .checked_mul(extension_opening_width)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "root extension-opening partial count overflow".to_string(),
                    )
                })?,
            key.num_vars,
        )
    } else {
        (extension_opening_width, padded_boolean_vars(current_w_len)?)
    };
    extension_opening_reduction_proof_bytes(
        challenge_field_bits,
        partials,
        opening_vars,
        extension_opening_width,
    )
}

/// Materialize and validate a generated schedule-table entry into a planned
/// runtime schedule.
///
/// The policy hooks are the only config-specific inputs: generated table
/// validation, direct witness sizing, level layout assembly, next-witness sizing,
/// and proof-byte sizing are shared by `akita-types`.
///
/// # Errors
///
/// Returns an error if the generated entry is structurally invalid, does not
/// match `key`, or does not agree with the supplied config policy callbacks.
pub fn schedule_plan_from_generated_entry<F, Stage1Config, ScaleBatchedRoot, DirectLevelParams>(
    key: AkitaScheduleLookupKey,
    entry: &GeneratedScheduleTableEntry,
    policy: GeneratedSchedulePlanPolicy<Stage1Config, ScaleBatchedRoot, DirectLevelParams>,
) -> Result<AkitaSchedulePlan, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
    ScaleBatchedRoot: Fn(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    DirectLevelParams: Fn(AkitaScheduleInputs, u32) -> Result<LevelParams, AkitaError>,
{
    let GeneratedSchedulePlanPolicy {
        sis_family,
        root_decomp,
        challenge_field_bits,
        recursive_public_rows,
        extension_opening_width,
        stage1_challenge_config,
        scale_batched_root_layout,
        direct_level_params,
    } = policy;

    if entry.steps.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "generated schedule table entry must contain at least one step".to_string(),
        ));
    }
    if recursive_public_rows == 0 {
        return Err(AkitaError::InvalidSetup(
            "recursive public row count must be nonzero".to_string(),
        ));
    }
    let expected_root_w_len = 1usize
        .checked_shl(key.num_vars as u32)
        .ok_or_else(|| AkitaError::InvalidSetup("root witness length overflow".to_string()))?;

    let field_bits = root_decomp.field_bits();
    let mut steps = Vec::with_capacity(entry.steps.len().max(1));
    let mut fold_level = 0usize;
    let mut current_w_len = expected_root_w_len;
    let mut current_log_basis = root_decomp.log_basis;

    for (step_index, generated_step) in entry.steps.iter().enumerate() {
        match generated_step {
            GeneratedStep::Fold(level) => {
                let Some(next_generated_step) = entry.steps.get(step_index + 1) else {
                    return Err(AkitaError::InvalidSetup(format!(
                        "generated schedule ended with a fold step at level {fold_level}"
                    )));
                };
                let next_log_basis = match next_generated_step {
                    GeneratedStep::Fold(next_level) => next_level.log_basis,
                    GeneratedStep::Direct(_) => level.log_basis,
                };

                let inputs = AkitaScheduleInputs {
                    num_vars: key.num_vars,
                    level: fold_level,
                    current_w_len,
                };
                let params = generated_level_params(sis_family, *level, &stage1_challenge_config);
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
                    current_w_len / level.ring_d as usize,
                )?;
                let root_is_batched = fold_level == 0
                    && (key.num_points != 1
                        || key.num_t_vectors != 1
                        || key.num_w_vectors != 1
                        || key.num_z_vectors != 1);
                let mut lp = params.with_layout(&layout);
                if root_is_batched {
                    lp = scale_batched_root_layout(&lp, key.num_t_vectors)?;
                }
                let runtime_next_w_len = if fold_level == 0 {
                    let next_w_ring = w_ring_element_count_with_vector_counts_bits::<F>(
                        field_bits,
                        &lp,
                        key.num_points,
                        key.num_t_vectors,
                        key.num_w_vectors,
                        key.num_z_vectors,
                    )?;
                    next_w_ring.checked_mul(lp.ring_dimension).ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated root next witness length overflow".to_string(),
                        )
                    })?
                } else {
                    w_ring_element_count_with_vector_counts_bits::<F>(
                        field_bits,
                        &lp,
                        1,
                        1,
                        recursive_public_rows,
                        recursive_public_rows,
                    )?
                    .checked_mul(lp.ring_dimension)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "generated recursive next witness length overflow".to_string(),
                        )
                    })?
                };
                let next_inputs = AkitaScheduleInputs {
                    num_vars: key.num_vars,
                    level: fold_level + 1,
                    current_w_len: runtime_next_w_len,
                };

                let (next_level_params, next_commit_coeffs) = match next_generated_step {
                    GeneratedStep::Fold(next_level) => {
                        let next_level_params = generated_level_params(
                            sis_family,
                            *next_level,
                            &stage1_challenge_config,
                        );
                        let coeffs =
                            next_level_params.b_key.row_len() * next_level_params.ring_dimension;
                        (next_level_params, coeffs)
                    }
                    GeneratedStep::Direct(_) => {
                        let next_level_params = direct_level_params(next_inputs, next_log_basis)?;
                        let coeffs =
                            next_level_params.b_key.row_len() * next_level_params.ring_dimension;
                        (next_level_params, coeffs)
                    }
                };
                let base_level_bytes = if fold_level == 0 {
                    level_proof_bytes(
                        field_bits,
                        challenge_field_bits,
                        &lp,
                        &lp,
                        &next_level_params,
                        next_inputs.current_w_len,
                        key.num_z_vectors,
                    )
                } else {
                    level_proof_bytes(
                        field_bits,
                        challenge_field_bits,
                        &lp,
                        &lp,
                        &next_level_params,
                        next_inputs.current_w_len,
                        recursive_public_rows,
                    )
                };
                let runtime_level_bytes = base_level_bytes
                    + extension_opening_reduction_level_bytes(
                        challenge_field_bits,
                        extension_opening_width,
                        fold_level,
                        key,
                        current_w_len,
                    )?;

                steps.push(AkitaPlannedStep::Fold(Box::new(AkitaPlannedLevel {
                    inputs,
                    lp,
                    next_inputs,
                    next_level_log_basis: next_log_basis,
                    next_commit_coeffs,
                    level_bytes: runtime_level_bytes,
                })));
                fold_level += 1;
                current_w_len = runtime_next_w_len;
                current_log_basis = next_log_basis;
            }
            GeneratedStep::Direct(_) => {
                if step_index + 1 != entry.steps.len() {
                    return Err(AkitaError::InvalidSetup(
                        "generated direct step must be terminal".to_string(),
                    ));
                }
                let witness_shape = if fold_level == 0 {
                    DirectWitnessShape::FieldElements(current_w_len)
                } else {
                    DirectWitnessShape::PackedDigits((current_w_len, current_log_basis))
                };
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);

                let state = AkitaPlannedState {
                    level: fold_level,
                    current_w_len,
                    log_basis: current_log_basis,
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
pub fn generated_schedule_plan_from_table<F, Stage1Config, ScaleBatchedRoot, DirectLevelParams>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
    policy: GeneratedSchedulePlanPolicy<Stage1Config, ScaleBatchedRoot, DirectLevelParams>,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> SparseChallengeConfig,
    ScaleBatchedRoot: Fn(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    DirectLevelParams: Fn(AkitaScheduleInputs, u32) -> Result<LevelParams, AkitaError>,
{
    if table.sis_family != policy.sis_family {
        return Err(AkitaError::InvalidSetup(format!(
            "generated schedule SIS family mismatch: table={:?}, config={:?}",
            table.sis_family, policy.sis_family
        )));
    }
    match table_entry(table, generated_schedule_lookup_key(key)) {
        Some(entry) => {
            schedule_plan_from_generated_entry::<F, _, _, _>(key, entry, policy).map(Some)
        }
        None => Ok(None),
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
    let w_hat_count = num_w_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_t_vectors
        .checked_mul(lp.num_blocks)
        .and_then(|n| n.checked_mul(lp.a_key.row_len()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let z_pre_count = num_public_rows
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(lp.num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    // One public y-row per packaged public opening row.
    let r_rows = lp.m_row_count(num_points, num_public_rows)?;
    let r_count = r_rows
        .checked_mul(r_decomp_levels::<F>(lp.log_basis))
        .ok_or_else(|| AkitaError::InvalidSetup("witness r-tail width overflow".to_string()))?;
    #[cfg(feature = "zk")]
    {
        let d_blinding_count = crate::zk::blinding_column_count::<F>(
            lp.d_key.row_len(),
            lp.ring_dimension,
            lp.log_basis,
        );
        let b_blinding_count = num_points
            .checked_mul(crate::zk::blinding_column_count::<F>(
                lp.b_key.row_len(),
                lp.ring_dimension,
                lp.log_basis,
            ))
            .ok_or_else(|| AkitaError::InvalidSetup("ZK B-blinding width overflow".to_string()))?;
        w_hat_count
            .checked_add(t_hat_count)
            .and_then(|n| n.checked_add(b_blinding_count))
            .and_then(|n| n.checked_add(d_blinding_count))
            .and_then(|n| n.checked_add(z_pre_count))
            .and_then(|n| n.checked_add(r_count))
            .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
    }
    #[cfg(not(feature = "zk"))]
    {
        w_hat_count
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
    /// Serialized terminal witness payload shape.
    pub witness_shape: DirectWitnessShape,
    /// Direct witness bytes.
    pub direct_bytes: usize,
}

impl DirectStep {
    /// Active terminal log-basis for packed direct witnesses.
    pub fn log_basis(&self, field_bits: u32) -> u32 {
        match self.witness_shape {
            DirectWitnessShape::PackedDigits((_, bits)) => bits,
            DirectWitnessShape::FieldElements(_) => field_bits,
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

/// Witness length entering the root fold, in field elements.
pub fn root_current_w_len(lp: &LevelParams) -> usize {
    lp.num_blocks
        .checked_mul(lp.block_len)
        .and_then(|len| len.checked_mul(lp.ring_dimension))
        .unwrap_or(0)
}

/// Build the root-direct schedule for roots that do not admit a fold step.
///
/// # Errors
///
/// Returns an error if `num_vars` cannot be represented as a witness length.
pub fn root_direct_schedule(num_vars: usize) -> Result<Schedule, AkitaError> {
    let current_w_len = 1usize.checked_shl(num_vars as u32).ok_or_else(|| {
        AkitaError::InvalidSetup("root-direct witness length overflow".to_string())
    })?;
    Ok(Schedule {
        steps: vec![Step::Direct(DirectStep {
            current_w_len,
            witness_shape: DirectWitnessShape::FieldElements(current_w_len),
            direct_bytes: 0,
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

/// Return the root fold params when a runtime schedule starts with a fold.
pub fn schedule_root_fold_params(schedule: &Schedule) -> Option<&LevelParams> {
    schedule_root_fold_step(schedule).map(|step| &step.params)
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
        Some(Step::Direct(step)) => match step.witness_shape {
            DirectWitnessShape::PackedDigits((_, bits_per_elem)) => {
                Ok(direct_params(inputs, bits_per_elem))
            }
            DirectWitnessShape::FieldElements(_) => Err(AkitaError::InvalidSetup(
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
        return Err(AkitaError::InvalidSetup(format!(
            "scheduled recursive level {level} did not match runtime state: \
             expected_w_len={}, actual_w_len={}, expected_log_basis={}, actual_log_basis={}",
            step.current_w_len, inputs.current_w_len, step.params.log_basis, current_log_basis
        )));
    }
    let next_inputs = AkitaScheduleInputs {
        num_vars: inputs.num_vars,
        level: level + 1,
        current_w_len: step.next_w_len,
    };
    let next_level_params =
        scheduled_next_level_params(schedule, level + 1, next_inputs, direct_params)?;
    Ok((step.params.clone(), next_level_params))
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

    #[test]
    fn root_direct_schedule_uses_field_element_payload() {
        let schedule = root_direct_schedule(3).expect("root-direct schedule");
        assert_eq!(schedule.total_bytes, 0);

        let [Step::Direct(step)] = schedule.steps.as_slice() else {
            panic!("root-direct schedule should contain one direct step");
        };
        assert_eq!(step.current_w_len, 8);
        assert_eq!(step.witness_shape, DirectWitnessShape::FieldElements(8));
        assert_eq!(step.direct_bytes, 0);
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
            extension_opening_reduction: None,
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
    fn generated_schedule_key_preserves_commitment_group_count() {
        let one_group = AkitaScheduleLookupKey::new_with_points(16, 1, 4, 4, 1);
        let four_groups = AkitaScheduleLookupKey::new_with_points(16, 4, 4, 4, 1);

        assert_ne!(
            generated_schedule_lookup_key(one_group),
            generated_schedule_lookup_key(four_groups),
            "generated schedule lookup must not alias differently grouped commitment shapes"
        );
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
                level_proof_bytes(128, 128, &lp, &lp, &next_lp, next_w_len, 1),
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
                level_proof_bytes(128, 128, &lp, &lp, &next_lp, next_w_len, num_points),
                root_proof.serialized_size(Compress::No),
                "planned batched root bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }
}
