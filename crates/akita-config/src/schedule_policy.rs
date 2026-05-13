use crate::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::{GeneratedScheduleTable, GeneratedStage1ChallengeShape};
use akita_types::DecompositionParams;
use akita_types::LevelParams;
#[cfg(feature = "planner")]
use akita_types::WitnessShape;
use akita_types::{
    level_layout_from_params, AkitaRootBatchSummary, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaSchedulePlan,
};

#[cfg(test)]
use akita_types::layout::digit_math::optimal_m_r_split;
#[cfg(test)]
use akita_types::{planned_w_ring_element_count, recursive_level_decomposition_from_root};

pub(crate) fn generated_schedule_plan_from_table<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
    if matches!(
        table.stage1_challenge_shape,
        GeneratedStage1ChallengeShape::Tensor
    ) && !Cfg::allow_tensor_stage1_schedules()
    {
        return Err(AkitaError::InvalidSetup(format!(
            "tensor stage-1 generated schedules require an audited tensor opt-in for {}",
            std::any::type_name::<Cfg>()
        )));
    }
    akita_types::generated_schedule_plan_from_table(
        key,
        table,
        Cfg::decomposition(),
        Cfg::stage1_challenge_config,
        |root_lp, num_claims| {
            akita_types::scale_batched_root_layout(
                root_lp,
                num_claims,
                Cfg::decomposition().field_bits(),
            )
        },
    )
}

/// Derive the commitment layout for a recursive level at the given log-basis.
///
/// # Errors
///
/// Returns an error if the root or recursive layout derivation fails.
pub fn current_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    if inputs.level == 0 {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }
    let params = Cfg::level_params_with_log_basis(inputs, log_basis);
    let layout = akita_types::recursive_level_layout_from_params(
        &params,
        inputs.current_w_len,
        Cfg::decomposition(),
    )?;
    Ok(params.with_layout(&layout))
}

/// Shape-aware variant of `current_level_layout_with_log_basis`.
///
/// Pre-sets `params.stage1_challenge_shape = shape` BEFORE invoking
/// `recursive_level_layout_from_params`, so the (m_vars, r_vars,
/// num_blocks, block_len, inner_width, num_digits_fold) split is
/// derived against the chosen shape's effective L1 mass from the start
/// — rather than the default shape's mass with a post-hoc shape patch
/// that leaves the split inconsistent with `num_digits_fold`.
///
/// Mirrors `current_level_layout_with_log_basis` so `inputs.level == 0`
/// still defers to `root_level_layout_with_log_basis` (root layout
/// already handles its own shape derivation through `params_only` +
/// `apply_stage1_challenge_shape` in proof_optimized).
///
/// # Errors
///
/// Returns the same errors as `recursive_level_layout_from_params`.
pub fn current_level_layout_with_log_basis_for_shape<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
    shape: akita_challenges::Stage1ChallengeShape,
) -> Result<LevelParams, AkitaError> {
    if inputs.level == 0 {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }
    let mut params = Cfg::level_params_with_log_basis(inputs, log_basis);
    params.stage1_challenge_shape = shape;
    let layout = akita_types::recursive_level_layout_from_params(
        &params,
        inputs.current_w_len,
        Cfg::decomposition(),
    )?;
    Ok(params.with_layout(&layout))
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
pub(crate) fn akita_root_commitment_layout<Cfg: CommitmentConfig>(
    max_num_vars: usize,
) -> Result<LevelParams, AkitaError> {
    let inputs = AkitaScheduleInputs {
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
    Err(AkitaError::InvalidSetup(format!(
        "failed to converge on tiny-root params for {} at max_num_vars={max_num_vars}",
        std::any::type_name::<Cfg>()
    )))
}

// Ring-native §4.1 commitment layout helpers.
//
// These helpers used to back a `RingCommitmentScheme` trait that materialised
// commitments from explicit `t_hat` layouts. The production flow commits via
// `AkitaPolyOps::commit_inner_witness` (see `commitment_scheme.rs`), so only
// the layout-selection helpers remain here.

pub(crate) fn fallback_batched_root_split<Cfg>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(max_num_vars)?;
    if num_claims <= 1 {
        Ok(root_lp)
    } else {
        akita_types::scale_batched_root_layout(
            &root_lp,
            num_claims,
            Cfg::decomposition().field_bits(),
        )
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
/// `akita_types::scale_batched_root_layout`).
///
/// # Errors
///
/// Returns an error if the layout parameters overflow or are invalid.
pub fn akita_batched_root_layout<Cfg>(
    max_num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = AkitaScheduleLookupKey::with_batch(
        max_num_vars,
        max_num_vars,
        num_claims,
        AkitaRootBatchSummary::new(num_claims, 1, 1)?,
    );
    if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
        if let Some(split) = akita_types::split_batched_root_params_from_schedule_plan(
            &plan,
            Cfg::decomposition().field_bits(),
        ) {
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

    #[cfg(feature = "planner")]
    {
        let schedule = akita_planner::find_optimal_schedule::<Cfg>(
            max_num_vars,
            WitnessShape::new(num_claims, 1, 1),
        )?;
        match schedule.steps.first() {
            Some(akita_types::Step::Fold(root_step)) => Ok(akita_types::split_batched_root_params(
                &root_step.params,
                Cfg::decomposition().field_bits(),
            )),
            Some(akita_types::Step::Direct(_)) | None => {
                fallback_batched_root_split::<Cfg>(max_num_vars, 1)
            }
        }
    }

    #[cfg(not(feature = "planner"))]
    {
        let _ = num_claims;
        Err(crate::missing_generated_schedule(
            "batched root layout",
            lookup_key,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof_optimized::fp128;
    use akita_types::generated::{
        fp128_d128_full_table, fp128_d32_full_table, fp128_d32_onehot_table, fp128_d64_full_table,
        fp128_d64_onehot_table, GeneratedScheduleTable,
    };
    #[cfg(feature = "planner")]
    use akita_types::w_ring_element_count_with_claim_groups;
    use akita_types::{w_ring_element_count, ScheduleProvider};

    fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(max_num_vars: usize) {
        let key = AkitaScheduleLookupKey::singleton(max_num_vars, max_num_vars, 1);
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
            let key = AkitaScheduleLookupKey::with_batch(
                entry.key.max_num_vars,
                entry.key.num_vars,
                entry.key.layout_num_claims,
                AkitaRootBatchSummary::new(
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
            let key = AkitaScheduleLookupKey::with_batch(
                entry.key.max_num_vars,
                entry.key.num_vars,
                entry.key.layout_num_claims,
                AkitaRootBatchSummary::new(
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
        let _ = checked_folded_entry;
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
    fn generated_d32_full_tensor_plan_materializes() {
        let key = AkitaScheduleLookupKey::singleton(26, 26, 1);
        let plan = fp128::D32Full::schedule_plan(key)
            .expect("D32 tensor schedule lookup should succeed")
            .expect("D32 tensor table should contain the key");
        let runtime =
            fp128::D32Full::get_params_for_prove(26, 26, 1, AkitaRootBatchSummary::singleton())
                .expect("runtime D32 tensor plan should succeed");
        assert_eq!(runtime.steps.len(), plan.steps.len());
    }

    #[test]
    fn generated_d128_full_table_materializes_valid_plans() {
        let table = fp128_d128_full_table();
        for entry in table.entries {
            let key = AkitaScheduleLookupKey::with_batch(
                entry.key.max_num_vars,
                entry.key.num_vars,
                entry.key.layout_num_claims,
                AkitaRootBatchSummary::new(
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
    fn generated_tensor_tables_are_audited_opted_in() {
        assert!(fp128::D128Full::allow_tensor_stage1_schedules());
        assert!(matches!(
            fp128_d128_full_table().stage1_challenge_shape,
            GeneratedStage1ChallengeShape::Tensor
        ));
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

        let runtime = Cfg::get_params_for_prove(30, 30, 1, AkitaRootBatchSummary::singleton())
            .expect("singleton runtime plan");
        let root_inputs = AkitaScheduleInputs {
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

        // Verify the planner's `optimal_m_r_split` returns a (m, r) split
        // whose `planned_w_ring_element_count` is minimal among all valid
        // splits at the same reduced-var count. The optimal split itself
        // depends on the level's params (specifically `n_a`), which is now
        // chosen by the shape-aware-tentative + iterated fixed-point SIS
        // derivation — so we can't pin the exact (m, r) the test produces.
        // We instead check the optimality invariant directly.
        let log_basis = Cfg::decomposition().log_basis;
        let inputs = AkitaScheduleInputs {
            max_num_vars: 30,
            level: 1,
            current_w_len: 25_974_272,
        };
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        let decomp =
            recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis);
        let num_ring = inputs.current_w_len / params.ring_dimension;
        let reduced_vars = num_ring.next_power_of_two().trailing_zeros() as usize;

        let (best_m, best_r) = optimal_m_r_split(
            params.a_key.row_len() as u32,
            params.challenge_l1_mass(),
            decomp.log_commit_bound,
            decomp.log_basis,
            reduced_vars,
            num_ring,
            decomp.field_bits(),
        );
        assert_eq!(best_m + best_r, reduced_vars);

        let best_lp = level_layout_from_params(best_m, best_r, &params, decomp, num_ring).unwrap();
        let best_w = planned_w_ring_element_count(Cfg::decomposition().field_bits(), &best_lp);

        // Try every alternative valid split and assert the optimal split has
        // the minimum (or tied-minimum) w_ring count.
        for m in 1..reduced_vars {
            let r = reduced_vars - m;
            let Ok(other) = level_layout_from_params(m, r, &params, decomp, num_ring) else {
                continue;
            };
            let other_w = planned_w_ring_element_count(Cfg::decomposition().field_bits(), &other);
            assert!(
                best_w <= other_w,
                "(m={best_m}, r={best_r}) w={best_w} should be <= (m={m}, r={r}) w={other_w}",
            );
        }
    }

    #[test]
    fn tight_block_len_is_no_larger_than_pow2() {
        for max_num_vars in [14, 20, 30] {
            let plan = fp128::D128Full::schedule_plan(AkitaScheduleLookupKey::singleton(
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

    #[cfg(feature = "planner")]
    #[test]
    fn batched_root_layout_is_invariant_under_equivalent_partitions() {
        type Cfg = fp128::D64OneHot;

        let batch_a = AkitaRootBatchSummary::from_claim_group_sizes(&[1, 1, 4], 2).unwrap();
        let batch_b = AkitaRootBatchSummary::from_claim_group_sizes(&[2, 2, 2], 2).unwrap();

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

    #[cfg(feature = "planner")]
    #[test]
    fn batched_root_next_w_len_and_shape_are_invariant_under_equivalent_partitions() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let claim_groups_a = [1usize, 1, 4];
        let claim_groups_b = [2usize, 2, 2];
        let batch_a = AkitaRootBatchSummary::from_claim_group_sizes(&claim_groups_a, 2).unwrap();
        let batch_b = AkitaRootBatchSummary::from_claim_group_sizes(&claim_groups_b, 2).unwrap();

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

    #[cfg(feature = "planner")]
    #[test]
    fn batched_root_next_w_len_requires_group_and_point_counts() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        let singleton_groups = AkitaRootBatchSummary::new(6, 6, 1).unwrap();
        let grouped_same_point = AkitaRootBatchSummary::new(6, 3, 1).unwrap();
        let grouped_two_points = AkitaRootBatchSummary::new(6, 3, 2).unwrap();

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

    #[cfg(feature = "planner")]
    #[test]
    fn batched_root_layout_planner_direct_fallback_is_per_polynomial() {
        type Cfg = fp128::D32OneHot;
        const MAX_NUM_VARS: usize = 1;
        const NUM_CLAIMS: usize = 3;

        let table_miss_key = AkitaScheduleLookupKey::with_batch(
            MAX_NUM_VARS,
            MAX_NUM_VARS,
            NUM_CLAIMS,
            AkitaRootBatchSummary::new(NUM_CLAIMS, 1, 1).unwrap(),
        );
        assert!(
            Cfg::schedule_plan(table_miss_key).unwrap().is_none(),
            "test must exercise the planner fallback, not a generated table entry"
        );

        let planner_schedule = akita_planner::find_optimal_schedule::<Cfg>(
            MAX_NUM_VARS,
            WitnessShape::new(NUM_CLAIMS, 1, 1),
        )
        .expect("planner fallback");
        assert!(
            !planner_schedule
                .steps
                .iter()
                .any(|step| matches!(step, akita_types::Step::Fold(_))),
            "test must exercise the direct/empty fallback path"
        );

        let singleton = fallback_batched_root_split::<Cfg>(MAX_NUM_VARS, 1).unwrap();
        let scaled = fallback_batched_root_split::<Cfg>(MAX_NUM_VARS, NUM_CLAIMS).unwrap();
        let actual = akita_batched_root_layout::<Cfg>(MAX_NUM_VARS, NUM_CLAIMS).unwrap();

        assert_eq!(actual, singleton);
        assert_ne!(actual.outer_width(), scaled.outer_width());
        assert_ne!(actual.d_matrix_width(), scaled.d_matrix_width());
    }
}
