//! Materialize generated schedule-table entries into runtime
//! `AkitaSchedulePlan` values.
//!
//! This module owns the policy-driven materializer that the runtime config
//! invokes for table hits. The verifier reaches it transitively through the
//! single `CommitmentConfig::schedule_plan` entry point; planner DP only runs
//! on table miss when the `planner` feature is enabled.

use akita_challenges::SparseChallengeConfig;
use akita_field::{AkitaError, CanonicalField};
use akita_types::generated::{
    table_entry, GeneratedFoldStep, GeneratedScheduleTable, GeneratedScheduleTableEntry,
    GeneratedStep,
};
use akita_types::{
    direct_witness_bytes, extension_opening_reduction_proof_bytes, generated_schedule_lookup_key,
    level_layout_from_params, level_proof_bytes, recursive_level_decomposition_from_root,
    root_extension_opening_partials, terminal_level_proof_bytes,
    w_ring_element_count_with_counts_bits, w_ring_element_count_with_counts_for_layout_bits,
    AkitaPlannedDirectStep, AkitaPlannedLevel, AkitaPlannedState, AkitaPlannedStep,
    AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan, DecompositionParams,
    DirectWitnessShape, LevelParams, MRowLayout, SisModulusFamily,
};

/// Policy hooks needed to materialize generated schedule-table entries into
/// runtime schedules.
///
/// This is the renamed `GeneratedSchedulePlanPolicy` from `akita-types`.
pub struct PlanPolicy<Stage1Config, ScaleBatchedRoot, DirectLevelParams> {
    /// SIS modulus family used by generated fold levels.
    pub sis_family: SisModulusFamily,
    /// Cyclotomic ring dimension `D` for this config's root commitments.
    /// Mirrors `Cfg::D`.
    pub ring_dimension: usize,
    /// Root-level digit decomposition used to interpret generated entries.
    pub root_decomp: DecompositionParams,
    /// Challenge-field width used for verifier challenges and proof-byte accounting.
    pub challenge_field_bits: u32,
    /// Number of public rows in recursive fold levels.
    pub recursive_public_rows: usize,
    /// Base-field width of the logical extension opening. This is `1` for the
    /// ordinary base-field path, which has no extension-opening reduction.
    pub extension_opening_width: usize,
    /// Stage-1 sparse challenge policy for each ring dimension. The hook is
    /// Result-returning so config-side validation propagates instead of
    /// panicking on the verifier replay path.
    pub stage1_challenge_config: Stage1Config,
    /// Root-layout scaler for batched committed openings.
    pub scale_batched_root_layout: ScaleBatchedRoot,
    /// Direct terminal layout policy for a schedule state and log-basis.
    pub direct_level_params: DirectLevelParams,
    /// Infinity-norm expansion introduced when claim-field coordinates are
    /// embedded into the ring subfield via `psi`. Mirrors
    /// `CommitmentConfig::ring_subfield_embedding_norm_bound`. Consumed by
    /// [`crate::root_direct_commit_layout`] when materializing a root-direct
    /// table entry.
    pub ring_subfield_norm_bound: u32,
}

fn generated_level_params<Stage1Config>(
    sis_family: SisModulusFamily,
    step: GeneratedFoldStep,
    stage1_challenge_config: &Stage1Config,
) -> Result<LevelParams, AkitaError>
where
    Stage1Config: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
{
    let stage1_config = stage1_challenge_config(step.ring_d as usize)?;
    Ok(LevelParams::params_only(
        sis_family,
        step.ring_d as usize,
        step.log_basis,
        step.n_a as usize,
        step.n_b as usize,
        step.n_d as usize,
        stage1_config,
    ))
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
            root_extension_opening_partials(extension_opening_width, key.num_w_vectors),
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
/// validation, direct witness sizing, level layout assembly, next-witness
/// sizing, and proof-byte sizing are shared by `akita-types`.
///
/// # Errors
///
/// Returns an error if the generated entry is structurally invalid, does not
/// match `key`, or does not agree with the supplied config policy callbacks.
pub fn schedule_plan_from_table_entry<F, Stage1Config, ScaleBatchedRoot, DirectLevelParams>(
    key: AkitaScheduleLookupKey,
    entry: &GeneratedScheduleTableEntry,
    policy: PlanPolicy<Stage1Config, ScaleBatchedRoot, DirectLevelParams>,
) -> Result<AkitaSchedulePlan, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ScaleBatchedRoot: Fn(&LevelParams, usize) -> Result<LevelParams, AkitaError>,
    DirectLevelParams: Fn(AkitaScheduleInputs, u32) -> Result<LevelParams, AkitaError>,
{
    let PlanPolicy {
        sis_family,
        ring_dimension,
        root_decomp,
        challenge_field_bits,
        recursive_public_rows,
        extension_opening_width,
        stage1_challenge_config,
        scale_batched_root_layout,
        direct_level_params,
        ring_subfield_norm_bound,
    } = policy;

    if entry.steps.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "generated schedule table entry must contain at least one step".to_string(),
        ));
    }
    if recursive_public_rows != 1 {
        return Err(AkitaError::InvalidSetup(
            "recursive generated schedules currently require exactly one public row".to_string(),
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
    let mut terminal_witness_field_len: Option<usize> = None;

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
                let params = generated_level_params(sis_family, *level, &stage1_challenge_config)?;
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
                    let next_w_ring = w_ring_element_count_with_counts_bits(
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
                    w_ring_element_count_with_counts_bits(field_bits, &lp, 1, 1, 1, 1)?
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
                        )?;
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
                let is_terminal = matches!(next_generated_step, GeneratedStep::Direct(_));
                let runtime_next_w_len = if is_terminal {
                    let (num_points, num_t_vectors, num_w_vectors, num_public_rows) =
                        if fold_level == 0 {
                            (
                                key.num_points,
                                key.num_t_vectors,
                                key.num_w_vectors,
                                key.num_z_vectors,
                            )
                        } else {
                            (1, 1, 1, 1)
                        };
                    let terminal_ring_count = w_ring_element_count_with_counts_for_layout_bits(
                        field_bits,
                        &lp,
                        num_points,
                        num_t_vectors,
                        num_w_vectors,
                        num_public_rows,
                        MRowLayout::Terminal,
                    )?;
                    let terminal_field_len = terminal_ring_count
                        .checked_mul(lp.ring_dimension)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "terminal recursive witness length overflow".to_string(),
                            )
                        })?;
                    terminal_witness_field_len = Some(terminal_field_len);
                    terminal_field_len
                } else {
                    runtime_next_w_len
                };
                let next_inputs = AkitaScheduleInputs {
                    num_vars: key.num_vars,
                    level: fold_level + 1,
                    current_w_len: runtime_next_w_len,
                };
                let num_claims_here = if fold_level == 0 {
                    key.num_z_vectors
                } else {
                    1
                };
                let base_level_bytes = if is_terminal {
                    terminal_level_proof_bytes(
                        field_bits,
                        challenge_field_bits,
                        &lp,
                        next_inputs.current_w_len,
                        num_claims_here,
                    )
                } else {
                    level_proof_bytes(
                        field_bits,
                        challenge_field_bits,
                        &lp,
                        &lp,
                        &next_level_params,
                        next_inputs.current_w_len,
                        num_claims_here,
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
                // Root-direct entries (`fold_level == 0`) carry the root
                // commit layout so `Cfg::get_params_for_batched_commitment`
                // can read it straight from the materialized step.
                // Terminal direct (after one or more folds) leaves
                // `commit_params` as `None`; the root commit layout lives
                // on the first fold step instead.
                // Root-direct entries (`fold_level == 0`) record the root
                // commit layout when it lies inside the audited SIS-floor
                // table. The generated tables also contain large-`num_vars`
                // root-direct edge entries whose root-commit layout exceeds
                // the audited SIS-floor (no production caller asks for
                // their singleton commit params). Materialize those with
                // `commit_params: None`; the corresponding
                // `get_params_for_batched_commitment` call surfaces the
                // missing-layout error if anything ever does ask.
                // Terminal direct (after one or more folds, `fold_level > 0`)
                // always leaves `commit_params` as `None`; the root commit
                // layout lives on the first fold step instead.
                let commit_params = if fold_level == 0 {
                    crate::root_direct_commit_layout(
                        sis_family,
                        ring_dimension,
                        root_decomp,
                        stage1_challenge_config(ring_dimension)?,
                        ring_subfield_norm_bound,
                        key.num_vars,
                        root_decomp.log_basis,
                    )
                    .ok()
                } else {
                    None
                };
                let witness_shape = if fold_level == 0 {
                    DirectWitnessShape::FieldElements(current_w_len)
                } else {
                    let terminal_field_len = terminal_witness_field_len.ok_or_else(|| {
                        AkitaError::InvalidSetup(
                            "terminal direct step missing precomputed witness length".to_string(),
                        )
                    })?;
                    DirectWitnessShape::PackedDigits((terminal_field_len, current_log_basis))
                };
                let direct_bytes = direct_witness_bytes(field_bits, &witness_shape);

                let direct_current_w_len = match &witness_shape {
                    DirectWitnessShape::PackedDigits((len, _)) => *len,
                    DirectWitnessShape::FieldElements(len) => *len,
                };
                let state = AkitaPlannedState {
                    level: fold_level,
                    current_w_len: direct_current_w_len,
                    log_basis: current_log_basis,
                };
                steps.push(AkitaPlannedStep::Direct(Box::new(AkitaPlannedDirectStep {
                    state,
                    witness_shape,
                    direct_bytes,
                    commit_params,
                })));
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
pub fn schedule_plan_from_table<F, Stage1Config, ScaleBatchedRoot, DirectLevelParams>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
    policy: PlanPolicy<Stage1Config, ScaleBatchedRoot, DirectLevelParams>,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
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
        Some(entry) => schedule_plan_from_table_entry::<F, _, _, _>(key, entry, policy).map(Some),
        None => Ok(None),
    }
}
