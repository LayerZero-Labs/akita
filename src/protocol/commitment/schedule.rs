use crate::protocol::config::CommitmentConfig;
use akita_field::HachiError;
use akita_types::digit_math::optimal_m_r_split;
use akita_types::generated::{
    table_entry, GeneratedDirectWitnessShape, GeneratedFoldStep, GeneratedScheduleTable,
    GeneratedStep,
};
use akita_types::DecompositionParams;
use akita_types::LevelParams;
use akita_types::{
    direct_witness_bytes, generated_schedule_lookup_key, level_layout_from_params,
    level_proof_bytes, planned_next_w_len, recursive_level_decomposition_from_root,
    recursive_level_proof_bytes, w_ring_element_count_with_batch_summary, DirectWitnessShape,
    HachiPlannedDirectStep, HachiPlannedLevel, HachiPlannedLevelExecution, HachiPlannedState,
    HachiPlannedStep, HachiRootBatchSummary, HachiScheduleInputs, HachiScheduleLookupKey,
    HachiSchedulePlan, WitnessShape,
};
use std::fmt::Write;

#[cfg(test)]
use crate::FieldCore;
#[cfg(test)]
use akita_serialization::{Compress, HachiSerialize};
#[cfg(test)]
use akita_sumcheck::{
    CompressedUniPoly, EqFactoredSumcheckProof, EqFactoredUniPoly, SumcheckProof,
};
#[cfg(test)]
use akita_types::{planned_w_ring_element_count, stage1_tree_stage_shapes, sumcheck_rounds};
#[cfg(test)]
use akita_types::{
    FlatRingVec, HachiLevelProof, HachiStage1Proof, HachiStage1StageProof, HachiStage2Proof,
};

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

fn schedule_plan_from_generated_entry<Cfg: CommitmentConfig>(
    key: HachiScheduleLookupKey,
    entry: &akita_types::generated::GeneratedScheduleTableEntry,
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

    let field_bits = Cfg::decomposition().field_bits();
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
                    DecompositionParams {
                        log_basis: level.log_basis,
                        ..Cfg::decomposition()
                    }
                } else {
                    recursive_level_decomposition_from_root(Cfg::decomposition(), level.log_basis)
                };
                let layout = level_layout_from_params(
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
                    lp = scale_batched_root_layout::<Cfg>(&lp, key.batch.num_claims)?;
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
                    recursive_level_proof_bytes(
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

pub(crate) fn generated_schedule_plan_from_table<Cfg: CommitmentConfig>(
    key: HachiScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<Option<HachiSchedulePlan>, HachiError> {
    table_entry(table, generated_schedule_lookup_key(key))
        .map(|entry| schedule_plan_from_generated_entry::<Cfg>(key, entry))
        .transpose()
}

#[cfg(test)]
fn dummy_sumcheck<F: FieldCore>(rounds: usize, degree: usize) -> SumcheckProof<F> {
    SumcheckProof {
        round_polys: (0..rounds)
            .map(|_| CompressedUniPoly {
                coeffs_except_linear_term: vec![F::zero(); degree],
            })
            .collect(),
    }
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

pub(crate) fn planned_log_basis_at_level_from_schedule(
    schedule: &HachiSchedulePlan,
    inputs: HachiScheduleInputs,
) -> Result<u32, HachiError> {
    if let Some(state_index) = exact_planned_state_index(schedule, inputs, None) {
        return Ok(schedule
            .state_after_prefix(state_index)
            .expect("exact planned state index must resolve to a state")
            .log_basis);
    }
    Err(HachiError::InvalidSetup(format!(
        "no planned log basis for inputs={inputs:?}: schedule does not include this state"
    )))
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

/// Derive the root commitment layout, allowing a zero-outer direct root.
///
/// This helper is for the commitment surface rather than the fold surface,
/// so it permits tiny roots that fit entirely inside one padded ring
/// element.
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
    let alpha = Cfg::D.trailing_zeros() as usize;
    if max_num_vars > alpha {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }

    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let mut params = LevelParams::params_only(d, log_basis, 1, 1, 1, stage1_config);
    let decomp = DecompositionParams {
        log_basis,
        ..Cfg::decomposition()
    };
    for _ in 0..4 {
        let layout = level_layout_from_params(0, 0, &params, decomp, 0)?;
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
    let decomp = recursive_level_decomposition_from_root(Cfg::decomposition(), lp.log_basis);
    let (m_vars, r_vars) = optimal_m_r_split(
        lp.a_key.row_len() as u32,
        lp.challenge_l1_mass(),
        decomp.log_commit_bound,
        decomp.log_basis,
        reduced_vars,
        num_ring_elems,
    );
    let layout = level_layout_from_params(m_vars, r_vars, lp, decomp, num_ring_elems)?;
    debug_assert_eq!(layout.m_vars + layout.r_vars + alpha, max_num_vars);
    Ok(layout)
}

// Ring-native §4.1 commitment layout helpers.
//
// These helpers used to back a `RingCommitmentScheme` trait that materialised
// commitments from explicit `t_hat` layouts. The production flow commits via
// `HachiPolyOps::commit_inner_witness` (see `commitment_scheme.rs`), so only
// the layout-selection helpers remain here.

#[cfg(test)]
pub(crate) use akita_types::root_current_w_len;

pub(crate) fn scale_batched_root_layout<Cfg>(
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    akita_types::scale_batched_root_layout(
        root_lp,
        num_claims,
        Cfg::stage1_challenge_config(Cfg::D).l1_mass(),
    )
}

/// Extract a per-poly batched root layout from a pre-computed schedule plan's
/// first fold level, if one exists.
fn split_from_schedule_plan(plan: &HachiSchedulePlan) -> Option<LevelParams> {
    let root_level = plan.fold_levels().next()?;
    Some(akita_types::split_batched_root_params(&root_level.lp))
}

pub(crate) fn fallback_batched_root_split<Cfg>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(max_num_vars)?;
    if num_claims <= 1 {
        Ok(root_lp)
    } else {
        scale_batched_root_layout::<Cfg>(&root_lp, num_claims)
    }
}

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `max_num_vars` variables.
///
/// First checks the pre-computed generated tables. When no table entry exists,
/// it falls back to the config-derived root split without running offline
/// planner search in the runtime crate. The returned layout has per-polynomial
/// `B`/`D` widths and per-polynomial `num_digits_fold`; callers that want the
/// batched root layout scale it themselves (internally via
/// `scale_batched_root_layout`).
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn hachi_batched_root_layout<Cfg>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, HachiError>
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
                root_m = split.log_block_len(),
                root_r = split.log_num_blocks(),
                root_lb = split.log_basis,
                "batched root split: read from pre-computed table"
            );
            return Ok(split);
        }
        tracing::info!(
            max_num_vars,
            num_claims,
            "batched root split: schedule is direct-only, falling back to config root layout"
        );
        return fallback_batched_root_split::<Cfg>(max_num_vars, 1);
    }

    tracing::info!(
        max_num_vars,
        num_claims,
        "batched root split: generated table miss, using planner fallback"
    );

    let schedule = akita_planner::find_optimal_schedule::<Cfg>(
        max_num_vars,
        WitnessShape::new(num_claims, 1, 1),
    )?;
    match schedule.steps.first() {
        Some(akita_types::Step::Fold(root_step)) => {
            Ok(akita_types::split_batched_root_params(&root_step.params))
        }
        Some(akita_types::Step::Direct(_)) | None => {
            fallback_batched_root_split::<Cfg>(max_num_vars, num_claims)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::config::proof_optimized::fp128;
    use crate::FieldCore;
    use akita_algebra::{CyclotomicRing, SparseChallengeConfig};
    use akita_serialization::{Compress, HachiSerialize};
    use akita_types::generated::{
        fp128_d128_full_table, fp128_d32_full_table, fp128_d32_onehot_table, fp128_d64_full_table,
        fp128_d64_onehot_table, GeneratedScheduleTable,
    };
    use akita_types::{
        w_ring_element_count, w_ring_element_count_with_claim_groups, FlatRingVec,
        HachiBatchedRootProof, ScheduleProvider,
    };

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

    fn assert_generated_table_matches_cfg_schedule<Cfg: CommitmentConfig>(
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
            let generated = generated_schedule_plan_from_table::<Cfg>(key, table)
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

    fn assert_generated_batched_roots_are_scaled<Cfg: CommitmentConfig>(
        table: GeneratedScheduleTable,
    ) {
        let mut checked_folded_entry = false;
        for entry in table
            .entries
            .iter()
            .filter(|entry| entry.key.batch_num_claims > 1)
        {
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
            let generated = generated_schedule_plan_from_table::<Cfg>(key, table)
                .expect("generated table should materialize")
                .expect("entry should exist in generated table");
            let Some(root) = generated.fold_levels().next() else {
                continue;
            };
            checked_folded_entry = true;
            let singleton_outer_width =
                root.lp.a_key.row_len() * root.lp.num_digits_open * root.lp.num_blocks;
            let singleton_d_width = root.lp.num_digits_open * root.lp.num_blocks;
            assert_eq!(
                root.lp.outer_width(),
                singleton_outer_width * entry.key.batch_num_claims,
                "generated batched root B width should be claim-scaled for key={key:?}"
            );
            assert_eq!(
                root.lp.d_matrix_width(),
                singleton_d_width * entry.key.batch_num_claims,
                "generated batched root D width should be claim-scaled for key={key:?}"
            );
        }
        assert!(
            checked_folded_entry,
            "generated table should include at least one folded batched entry"
        );
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
        let runtime_root = Cfg::get_params_for_prove(
            max_num_vars,
            max_num_vars,
            1,
            HachiRootBatchSummary::singleton(),
        )
        .expect("runtime root plan should succeed");
        let Some(akita_types::Step::Fold(runtime_root_step)) = runtime_root.steps.first() else {
            panic!("runtime root schedule should start with a fold");
        };
        assert_eq!(
            planned_root.level.inputs.current_w_len,
            runtime_root_step.current_w_len,
            "planned/runtime root current_w_len mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.lp,
            runtime_root_step.params,
            "planned/runtime root lp mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.next_inputs.current_w_len,
            runtime_root_step.next_w_len,
            "planned/runtime next_w_len mismatch for {} at max_num_vars={max_num_vars}",
            std::any::type_name::<Cfg>()
        );
    }

    #[test]
    fn generated_fp128_schedule_tables_match_cfg_schedule() {
        assert_generated_table_matches_cfg_schedule::<fp128::D32Full>(fp128_d32_full_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D32OneHot>(fp128_d32_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D64Full>(fp128_d64_full_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D64OneHot>(fp128_d64_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D128Full>(fp128_d128_full_table());
    }

    #[test]
    fn generated_batched_roots_restore_scaled_widths() {
        assert_generated_batched_roots_are_scaled::<fp128::D32Full>(fp128_d32_full_table());
        assert_generated_batched_roots_are_scaled::<fp128::D32OneHot>(fp128_d32_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp128::D64Full>(fp128_d64_full_table());
        assert_generated_batched_roots_are_scaled::<fp128::D64OneHot>(fp128_d64_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp128::D128Full>(fp128_d128_full_table());
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
            generated_schedule_plan_from_table::<fp128::D128Full>(key, table)
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

        let runtime = Cfg::get_params_for_prove(30, 30, 1, HachiRootBatchSummary::singleton())
            .expect("singleton runtime plan");
        let root_inputs = HachiScheduleInputs {
            max_num_vars: 30,
            level: 0,
            current_w_len: 1usize << 30,
        };
        let root_lp = Cfg::root_level_layout_with_log_basis(
            root_inputs,
            Cfg::log_basis_at_level(root_inputs),
        )
        .unwrap();
        let Some(akita_types::Step::Fold(runtime_root_step)) = runtime.steps.first() else {
            panic!("singleton schedule should start with a fold");
        };

        assert_eq!(runtime_root_step.params, root_lp);
        assert_eq!(runtime_root_step.current_w_len, 1usize << 30);
        assert_eq!(runtime_root_step.next_w_len % Cfg::D, 0);
    }

    #[test]
    fn recursive_onehot_split_matches_open_digit_witness_count() {
        type Cfg = fp128::D64OneHot;

        // Use the root decomposition basis directly: this test exercises the
        // tight (m, r) split optimizer at a recursive state that is not part of
        // the canonical schedule, so we don't rely on `log_basis_at_level`.
        let log_basis = Cfg::decomposition().log_basis;
        let inputs = HachiScheduleInputs {
            max_num_vars: 30,
            level: 1,
            current_w_len: 25_974_272,
        };
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        let decomp =
            recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis);
        let num_ring = inputs.current_w_len / params.ring_dimension;
        let lp_12_7 = level_layout_from_params(12, 7, &params, decomp, num_ring).unwrap();
        let lp_11_8 = level_layout_from_params(11, 8, &params, decomp, num_ring).unwrap();
        let w_12_7 = planned_w_ring_element_count(Cfg::decomposition().field_bits(), &lp_12_7);
        let w_11_8 = planned_w_ring_element_count(Cfg::decomposition().field_bits(), &lp_11_8);
        let reduced_vars = (inputs.current_w_len / params.ring_dimension)
            .next_power_of_two()
            .trailing_zeros() as usize;

        assert!(w_12_7 < w_11_8);
        assert_eq!(
            optimal_m_r_split(
                params.a_key.row_len() as u32,
                params.challenge_l1_mass(),
                decomp.log_commit_bound,
                decomp.log_basis,
                reduced_vars,
                num_ring,
            ),
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
                recursive_level_proof_bytes(128, &lp, &next_lp, next_w_len),
                exact_recursive_level_proof_bytes::<F>(&lp, &next_lp, next_w_len).unwrap(),
                "planned level bytes should match the serialized two-stage body at log_basis={log_basis}"
            );
        }
    }

    #[test]
    fn planned_batched_root_bytes_match_two_stage_payload_at_all_bases() {
        use akita_types::AjtaiKeyParams;
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

        let plan_a = Cfg::get_params_for_prove(30, 30, batch_a.num_claims, batch_a).unwrap();
        let plan_b = Cfg::get_params_for_prove(30, 30, batch_b.num_claims, batch_b).unwrap();
        let Some(akita_types::Step::Fold(root_a)) = plan_a.steps.first() else {
            panic!("batch A schedule should start with a fold");
        };
        let Some(akita_types::Step::Fold(root_b)) = plan_b.steps.first() else {
            panic!("batch B schedule should start with a fold");
        };

        assert_eq!(root_a.params, root_b.params);
    }

    #[test]
    fn batched_root_next_w_len_and_shape_are_invariant_under_equivalent_partitions() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let claim_groups_a = [1usize, 1, 4];
        let claim_groups_b = [2usize, 2, 2];
        let batch_a = HachiRootBatchSummary::from_claim_group_sizes(&claim_groups_a, 2).unwrap();
        let batch_b = HachiRootBatchSummary::from_claim_group_sizes(&claim_groups_b, 2).unwrap();

        let plan_a =
            Cfg::get_params_for_prove(MAX_NUM_VARS, MAX_NUM_VARS, batch_a.num_claims, batch_a)
                .unwrap();
        let plan_b =
            Cfg::get_params_for_prove(MAX_NUM_VARS, MAX_NUM_VARS, batch_b.num_claims, batch_b)
                .unwrap();
        let Some(akita_types::Step::Fold(root_a)) = plan_a.steps.first() else {
            panic!("batch A schedule should start with a fold");
        };
        let Some(akita_types::Step::Fold(root_b)) = plan_b.steps.first() else {
            panic!("batch B schedule should start with a fold");
        };

        let next_w_ring_a = w_ring_element_count_with_claim_groups::<
            <Cfg as CommitmentConfig>::Field,
        >(&root_a.params, &claim_groups_a, batch_a.num_points);
        let next_w_ring_b = w_ring_element_count_with_claim_groups::<
            <Cfg as CommitmentConfig>::Field,
        >(&root_b.params, &claim_groups_b, batch_b.num_points);

        assert_eq!(next_w_ring_a, next_w_ring_b);
        assert_eq!(root_a.next_w_len, root_b.next_w_len);
        assert_eq!(root_a.level_bytes, root_b.level_bytes);
    }

    #[test]
    fn batched_root_next_w_len_requires_group_and_point_counts() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let singleton_groups = HachiRootBatchSummary::new(6, 6, 1).unwrap();
        let grouped_same_point = HachiRootBatchSummary::new(6, 3, 1).unwrap();
        let grouped_two_points = HachiRootBatchSummary::new(6, 3, 2).unwrap();

        let singleton_plan = Cfg::get_params_for_prove(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            singleton_groups.num_claims,
            singleton_groups,
        )
        .unwrap();
        let grouped_plan = Cfg::get_params_for_prove(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            grouped_same_point.num_claims,
            grouped_same_point,
        )
        .unwrap();
        let multipoint_plan = Cfg::get_params_for_prove(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            grouped_two_points.num_claims,
            grouped_two_points,
        )
        .unwrap();
        let Some(akita_types::Step::Fold(singleton_root)) = singleton_plan.steps.first() else {
            panic!("singleton schedule should start with a fold");
        };
        let Some(akita_types::Step::Fold(grouped_root)) = grouped_plan.steps.first() else {
            panic!("grouped schedule should start with a fold");
        };
        let Some(akita_types::Step::Fold(multipoint_root)) = multipoint_plan.steps.first() else {
            panic!("multipoint schedule should start with a fold");
        };

        assert_eq!(singleton_root.params, grouped_root.params);
        assert_eq!(grouped_root.params, multipoint_root.params);
        assert_ne!(singleton_root.next_w_len, grouped_root.next_w_len);
        assert_ne!(grouped_root.next_w_len, multipoint_root.next_w_len);
        assert_eq!(singleton_groups.num_points * Cfg::D, Cfg::D);
        assert_eq!(grouped_same_point.num_points * Cfg::D, Cfg::D);
        assert_eq!(grouped_two_points.num_points * Cfg::D, 2 * Cfg::D);
    }
}
