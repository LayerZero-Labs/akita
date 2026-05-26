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
    level_layout_from_params, level_proof_bytes, root_extension_opening_partials,
    terminal_level_proof_bytes, w_ring_element_count_with_counts_bits,
    w_ring_element_count_with_counts_for_layout_bits, AkitaPlannedDirectStep, AkitaPlannedLevel,
    AkitaPlannedState, AkitaPlannedStep, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaSchedulePlan, CommitmentEnvelope, DecompositionParams, DirectWitnessShape, LevelParams,
    MRowLayout, SisModulusFamily,
};

/// Policy hooks needed to materialize generated schedule-table entries into
/// runtime schedules.
///
/// This is the renamed `GeneratedSchedulePlanPolicy` from `akita-types`.
pub struct PlanPolicy<Stage1Config> {
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
    /// Pre-computed commitment envelope for `key.num_vars`. Consumed by
    /// [`crate::direct_level_params_with_log_basis`] when materializing a
    /// terminal-direct step. Caller computes it once via
    /// `Cfg::envelope(num_vars)` at policy construction.
    pub envelope: CommitmentEnvelope,
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
pub fn schedule_plan_from_table_entry<F, Stage1Config>(
    key: AkitaScheduleLookupKey,
    entry: &GeneratedScheduleTableEntry,
    policy: PlanPolicy<Stage1Config>,
) -> Result<AkitaSchedulePlan, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
{
    let PlanPolicy {
        sis_family,
        ring_dimension,
        root_decomp,
        challenge_field_bits,
        recursive_public_rows,
        extension_opening_width,
        stage1_challenge_config,
        envelope,
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
                    // Recursive level: balanced-digit `w` entries collapse
                    // `log_commit_bound` to `log_basis`.
                    DecompositionParams {
                        log_basis: level.log_basis,
                        log_commit_bound: level.log_basis,
                        log_open_bound: Some(
                            root_decomp
                                .log_open_bound
                                .unwrap_or(root_decomp.log_commit_bound),
                        ),
                    }
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
                    // Inlined `scale_batched_root_layout_with_config` — the
                    // policy already exposes the two Cfg knobs we need
                    // (`stage1_challenge_config(ring_dimension)` and
                    // `root_decomp.field_bits()`), so no fn-pointer hop.
                    lp = akita_types::scale_batched_root_layout(
                        &lp,
                        key.num_t_vectors,
                        stage1_challenge_config(ring_dimension)?.l1_norm(),
                        root_decomp.field_bits(),
                    )?;
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
                        let next_level_params = crate::direct_level_params_with_log_basis(
                            sis_family,
                            ring_dimension,
                            root_decomp,
                            stage1_challenge_config(ring_dimension)?,
                            ring_subfield_norm_bound,
                            &envelope,
                            next_inputs,
                            next_log_basis,
                        )?;
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
                // commit layout — same role as `FoldStep.params` for
                // Fold-root entries, so the same batched scaling applies
                // when the entry is for a non-singleton incidence.
                //
                // Tables also contain large-`num_vars` root-direct edge
                // entries whose singleton root layout exceeds the audited
                // SIS floor. We materialize those with `commit_params:
                // None` *deliberately*: such schedules are valid for
                // `proof_size` exploration and DP planning but must not
                // be committed against. The contract is documented on
                // [`DirectStep::commit_params`]; concretely it means
                // `Cfg::get_params_for_batched_commitment` rejects the
                // schedule with `InvalidSetup("root-direct schedule is
                // missing commit params")` and
                // `setup_level_params_from_runtime_schedule` returns
                // an empty list. See `commit_params_uncommittable_root_direct`
                // tests in `proof_optimized.rs` for the locked-in
                // behavior.
                //
                // Terminal direct (`fold_level > 0`) leaves `commit_params`
                // as `None`; the root commit layout lives on the first
                // fold step instead.
                let commit_params = if fold_level == 0 {
                    let singleton = crate::root_direct_commit_layout(
                        sis_family,
                        ring_dimension,
                        root_decomp,
                        stage1_challenge_config(ring_dimension)?,
                        ring_subfield_norm_bound,
                        key.num_vars,
                        root_decomp.log_basis,
                    )
                    .ok();
                    let root_is_batched = key.num_points != 1
                        || key.num_t_vectors != 1
                        || key.num_w_vectors != 1
                        || key.num_z_vectors != 1;
                    match singleton {
                        Some(lp) if root_is_batched => {
                            Some(akita_types::scale_batched_root_layout(
                                &lp,
                                key.num_t_vectors,
                                stage1_challenge_config(ring_dimension)?.l1_norm(),
                                root_decomp.field_bits(),
                            )?)
                        }
                        other => other,
                    }
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
                // Bake terminal-direct level params onto the planned step
                // for `fold_level > 0` (i.e. terminal `Direct(PackedDigits)`
                // after at least one fold), so prover/verifier can read
                // the next-level params straight from the schedule.
                // Root-direct (`fold_level == 0`) has no next level, so
                // `level_params` stays `None`.
                let level_params = if fold_level > 0 {
                    Some(crate::direct_level_params_with_log_basis(
                        sis_family,
                        ring_dimension,
                        root_decomp,
                        stage1_challenge_config(ring_dimension)?,
                        ring_subfield_norm_bound,
                        &envelope,
                        AkitaScheduleInputs {
                            num_vars: key.num_vars,
                            level: fold_level,
                            current_w_len: direct_current_w_len,
                        },
                        current_log_basis,
                    )?)
                } else {
                    None
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
                    level_params,
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
pub fn schedule_plan_from_table<F, Stage1Config>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
    policy: PlanPolicy<Stage1Config>,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    F: CanonicalField,
    Stage1Config: Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
{
    if table.sis_family != policy.sis_family {
        return Err(AkitaError::InvalidSetup(format!(
            "generated schedule SIS family mismatch: table={:?}, config={:?}",
            table.sis_family, policy.sis_family
        )));
    }
    match table_entry(table, generated_schedule_lookup_key(key)) {
        Some(entry) => schedule_plan_from_table_entry::<F, _>(key, entry, policy).map(Some),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    //! Malformed-table-entry contract tests for the materializer.
    //!
    //! These exercise the `AkitaError` paths the materializer must
    //! return for inputs the table generator should never produce, so
    //! a future regression that panics or silently accepts bad data
    //! fails here instead of at the verifier boundary. They are direct
    //! unit tests on `schedule_plan_from_table_entry` (and `_from_table`
    //! for the SIS-family check) and don't go through any preset Cfg.
    use super::*;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Fp32;
    use akita_types::generated::{GeneratedScheduleKey, GeneratedScheduleTable};
    use akita_types::CommitmentEnvelope;

    type Field = Fp32<251>;
    const RING_DIMENSION: usize = 8;
    const FIELD_BITS: u32 = 32;

    fn fold_step(ring_d: u32, log_basis: u32) -> GeneratedStep {
        GeneratedStep::Fold(GeneratedFoldStep {
            ring_d,
            log_basis,
            m_vars: 0,
            r_vars: 0,
            n_a: 1,
            n_b: 1,
            n_d: 1,
        })
    }

    fn entry_from_steps(
        key: AkitaScheduleLookupKey,
        steps: &'static [GeneratedStep],
    ) -> GeneratedScheduleTableEntry {
        GeneratedScheduleTableEntry {
            key: GeneratedScheduleKey {
                num_vars: key.num_vars,
                num_commitment_groups: key.num_t_vectors,
                num_t_vectors: key.num_t_vectors,
                num_w_vectors: key.num_w_vectors,
                num_z_vectors: key.num_z_vectors,
            },
            steps,
        }
    }

    fn supported_stage1(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        if d != RING_DIMENSION {
            return Err(AkitaError::InvalidSetup(format!(
                "unsupported ring_d={d} in test stage1 chooser"
            )));
        }
        Ok(SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        })
    }

    fn default_policy() -> PlanPolicy<fn(usize) -> Result<SparseChallengeConfig, AkitaError>> {
        PlanPolicy {
            sis_family: SisModulusFamily::Q32,
            ring_dimension: RING_DIMENSION,
            root_decomp: DecompositionParams {
                log_basis: 3,
                log_commit_bound: FIELD_BITS,
                log_open_bound: Some(FIELD_BITS),
            },
            challenge_field_bits: FIELD_BITS,
            recursive_public_rows: 1,
            extension_opening_width: 1,
            stage1_challenge_config: supported_stage1,
            envelope: CommitmentEnvelope {
                max_n_a: 1,
                max_n_b: 1,
                max_n_d: 1,
            },
            ring_subfield_norm_bound: 1,
        }
    }

    fn assert_invalid_setup_contains(err: AkitaError, needle: &str) {
        let s = err.to_string();
        assert!(
            s.contains(needle),
            "expected InvalidSetup containing {needle:?}, got: {s}"
        );
    }

    #[test]
    fn empty_entry_rejected() {
        let key = AkitaScheduleLookupKey::singleton(8);
        let entry = entry_from_steps(key, Box::leak(Box::new([])));
        let err = schedule_plan_from_table_entry::<Field, _>(key, &entry, default_policy())
            .expect_err("empty entry must error");
        assert_invalid_setup_contains(err, "must contain at least one step");
    }

    #[test]
    fn fold_without_terminal_direct_rejected() {
        let key = AkitaScheduleLookupKey::singleton(8);
        let entry = entry_from_steps(
            key,
            Box::leak(Box::new([fold_step(RING_DIMENSION as u32, 3)])),
        );
        let err = schedule_plan_from_table_entry::<Field, _>(key, &entry, default_policy())
            .expect_err("fold-only entry must error: no follower for the last fold");
        assert_invalid_setup_contains(err, "ended with a fold step");
    }

    #[test]
    fn nonterminal_direct_rejected() {
        let key = AkitaScheduleLookupKey::singleton(8);
        let entry = entry_from_steps(
            key,
            Box::leak(Box::new([
                GeneratedStep::Direct(akita_types::generated::GeneratedDirectStep),
                fold_step(RING_DIMENSION as u32, 3),
            ])),
        );
        let err = schedule_plan_from_table_entry::<Field, _>(key, &entry, default_policy())
            .expect_err("Direct step in non-terminal position must error");
        assert_invalid_setup_contains(err, "generated direct step must be terminal");
    }

    #[test]
    fn unsupported_ring_dimension_propagates_stage1_error() {
        // Stage-1 chooser hard-fails for ring_d != RING_DIMENSION.
        // A fold step that names a different ring_d should surface that
        // stage1 error rather than silently fall through to envelope
        // params.
        let key = AkitaScheduleLookupKey::singleton(8);
        let entry = entry_from_steps(
            key,
            Box::leak(Box::new([
                fold_step(16, 3), // ring_d=16, but supported_stage1 only accepts 8
                GeneratedStep::Direct(akita_types::generated::GeneratedDirectStep),
            ])),
        );
        let err = schedule_plan_from_table_entry::<Field, _>(key, &entry, default_policy())
            .expect_err("unsupported ring_d in fold step must propagate stage1 error");
        assert_invalid_setup_contains(err, "unsupported ring_d");
    }

    #[test]
    fn overflow_shaped_key_rejected() {
        // num_vars = 64 makes 1usize.checked_shl(64) overflow on a
        // 64-bit target; the materializer must reject the shape rather
        // than panic.
        let key = AkitaScheduleLookupKey::singleton(64);
        let entry = entry_from_steps(
            key,
            Box::leak(Box::new([GeneratedStep::Direct(
                akita_types::generated::GeneratedDirectStep,
            )])),
        );
        let err = schedule_plan_from_table_entry::<Field, _>(key, &entry, default_policy())
            .expect_err("num_vars=64 must overflow the witness-length shift and error");
        assert_invalid_setup_contains(err, "root witness length overflow");
    }

    #[test]
    fn sis_family_mismatch_rejected_at_table_layer() {
        // `schedule_plan_from_table` is the only call site that checks
        // table-vs-policy SIS family, so we exercise it directly here.
        let key = AkitaScheduleLookupKey::singleton(8);
        let entry = entry_from_steps(
            key,
            Box::leak(Box::new([GeneratedStep::Direct(
                akita_types::generated::GeneratedDirectStep,
            )])),
        );
        let table = GeneratedScheduleTable {
            sis_family: SisModulusFamily::Q128, // table claims Q128
            entries: Box::leak(Box::new([entry])),
        };
        // Policy says Q32 (mismatch).
        let err = schedule_plan_from_table::<Field, _>(key, table, default_policy())
            .expect_err("table-vs-policy SIS family mismatch must error");
        assert_invalid_setup_contains(err, "SIS family mismatch");
    }

    #[test]
    fn unsupported_recursive_public_rows_rejected() {
        // The materializer only supports `recursive_public_rows = 1`
        // today; a config that asks for anything else must be rejected
        // up front.
        let key = AkitaScheduleLookupKey::singleton(8);
        let entry = entry_from_steps(
            key,
            Box::leak(Box::new([GeneratedStep::Direct(
                akita_types::generated::GeneratedDirectStep,
            )])),
        );
        let mut policy = default_policy();
        policy.recursive_public_rows = 2;
        let err = schedule_plan_from_table_entry::<Field, _>(key, &entry, policy)
            .expect_err("recursive_public_rows != 1 must error");
        assert_invalid_setup_contains(err, "exactly one public row");
    }
}
