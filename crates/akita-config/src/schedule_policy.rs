use crate::sis_policy::sis_derived_recursive_params;
use crate::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::generated::GeneratedScheduleTable;
use akita_types::DecompositionParams;
use akita_types::LevelParams;
use akita_types::{
    level_layout_from_params, AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan,
    GeneratedSchedulePlanPolicy,
};

#[cfg(test)]
use akita_types::layout::digit_math::optimal_m_r_split;
#[cfg(any(all(test, not(feature = "zk")), all(test, feature = "planner")))]
use akita_types::ClaimIncidenceSummary;
#[cfg(test)]
use akita_types::{planned_w_ring_element_count, recursive_level_decomposition_from_root};

pub(crate) fn generated_schedule_plan_from_table<Cfg>(
    key: AkitaScheduleLookupKey,
    table: GeneratedScheduleTable,
) -> Result<Option<AkitaSchedulePlan>, AkitaError>
where
    Cfg: CommitmentConfig,
{
    akita_types::generated_schedule_plan_from_table::<<Cfg as CommitmentConfig>::Field, _, _, _>(
        key,
        table,
        GeneratedSchedulePlanPolicy {
            sis_family: Cfg::sis_modulus_family(),
            root_decomp: Cfg::decomposition(),
            challenge_field_bits: Cfg::decomposition().field_bits() * Cfg::CHAL_EXT_DEGREE as u32,
            recursive_public_rows: 1,
            extension_opening_width: Cfg::CLAIM_EXT_DEGREE,
            stage1_challenge_config: Cfg::stage1_challenge_config,
            scale_batched_root_layout: scale_batched_root_layout_with_config::<Cfg>,
            direct_level_params: direct_level_params_with_log_basis::<Cfg>,
        },
    )
}

fn scale_batched_root_layout_with_config<Cfg: CommitmentConfig>(
    root_lp: &LevelParams,
    num_claims: usize,
) -> Result<LevelParams, AkitaError> {
    akita_types::scale_batched_root_layout(
        root_lp,
        num_claims,
        Cfg::stage1_challenge_config(Cfg::D).l1_norm(),
        Cfg::decomposition().field_bits(),
    )
}

fn direct_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, AkitaError> {
    if inputs.level == 0 {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }

    let envelope = Cfg::envelope(inputs.num_vars);
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let params = sis_derived_recursive_params::<Cfg>(
        Cfg::D,
        log_basis,
        inputs.current_w_len,
        &stage1_config,
        &envelope,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "failed to derive direct terminal params for level {} at num_vars={}",
            inputs.level, inputs.num_vars
        ))
    })?;
    akita_types::recursive_level_layout_from_params(
        &params,
        inputs.current_w_len,
        Cfg::decomposition(),
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

/// Derive the root commitment layout, allowing a zero-outer direct root.
///
/// This helper is for the commitment surface rather than the fold surface,
/// so it permits tiny roots that fit entirely inside one padded ring
/// element.
///
/// # Errors
///
/// Returns an error if `num_vars` underflows `alpha` or if the derived
/// layout overflows.
pub(crate) fn akita_root_commitment_layout<Cfg: CommitmentConfig>(
    num_vars: usize,
) -> Result<LevelParams, AkitaError> {
    let inputs = AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
    };
    let log_basis = Cfg::log_basis_at_level(inputs);
    let alpha = Cfg::D.trailing_zeros() as usize;
    if num_vars > alpha {
        return Cfg::root_level_layout_with_log_basis(inputs, log_basis);
    }

    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let mut params = LevelParams::params_only(
        Cfg::sis_modulus_family(),
        d,
        log_basis,
        1,
        1,
        1,
        stage1_config,
    );
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
        "failed to converge on tiny-root params for {} at num_vars={num_vars}",
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
    num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let root_lp = Cfg::commitment_layout(num_vars)?;
    if num_claims <= 1 {
        Ok(root_lp)
    } else {
        akita_types::scale_batched_root_layout(
            &root_lp,
            num_claims,
            Cfg::stage1_challenge_config(Cfg::D).l1_norm(),
            Cfg::decomposition().field_bits(),
        )
    }
}

/// Derive the per-polynomial commitment layout optimized for a batch of
/// `num_claims` polynomials with `num_vars` variables.
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
    num_vars: usize,
    num_claims: usize,
) -> Result<LevelParams, AkitaError>
where
    Cfg: CommitmentConfig,
{
    let lookup_key = AkitaScheduleLookupKey::new(num_vars, num_claims, num_claims, 1);
    if let Some(plan) = Cfg::schedule_plan(lookup_key)? {
        if let Some(split) = akita_types::split_batched_root_params_from_schedule_plan(
            &plan,
            Cfg::decomposition().field_bits(),
        ) {
            tracing::info!(
                num_vars,
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
            num_vars,
            num_claims,
            "batched root split: schedule is direct-only, falling back to config root layout"
        );
        return fallback_batched_root_split::<Cfg>(num_vars, 1);
    }

    tracing::info!(
        num_vars,
        num_claims,
        "batched root split: generated table miss, using planner fallback"
    );

    #[cfg(feature = "planner")]
    {
        let schedule = akita_planner::find_optimal_schedule::<Cfg>(lookup_key)?;
        match schedule.steps.first() {
            Some(akita_types::Step::Fold(root_step)) => Ok(akita_types::split_batched_root_params(
                &root_step.params,
                Cfg::decomposition().field_bits(),
            )),
            Some(akita_types::Step::Direct(_)) | None => {
                fallback_batched_root_split::<Cfg>(num_vars, 1)
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
    #[cfg(not(feature = "zk"))]
    use crate::proof_optimized::{fp32, fp64};
    #[cfg(not(feature = "zk"))]
    use akita_types::generated::{
        fp128_d128_full_table, fp128_d32_full_table, fp128_d32_onehot_table, fp128_d64_full_table,
        fp128_d64_onehot_table, fp32_d128_onehot_table, fp32_d128_table, fp32_d256_onehot_table,
        fp32_d256_table, fp32_d512_onehot_table, fp32_d512_table, fp32_d64_onehot_table,
        fp32_d64_table, fp64_d128_onehot_table, fp64_d128_table, fp64_d256_onehot_table,
        fp64_d256_table, fp64_d32_onehot_table, fp64_d32_table, fp64_d64_onehot_table,
        fp64_d64_table, GeneratedScheduleTable,
    };
    #[cfg(not(feature = "zk"))]
    use akita_types::w_ring_element_count;
    #[cfg(feature = "planner")]
    use akita_types::w_ring_element_count_with_counts;
    #[cfg(any(not(feature = "zk"), feature = "planner"))]
    use akita_types::ScheduleProvider;

    #[cfg(feature = "planner")]
    fn point_local_incidence_summary(
        num_vars: usize,
        num_polys_per_point: &[usize],
    ) -> ClaimIncidenceSummary {
        ClaimIncidenceSummary::from_point_polys(num_vars, num_polys_per_point.to_vec())
            .expect("schedule-policy incidence inputs are nonempty and point-local")
    }

    #[cfg(not(feature = "zk"))]
    fn assert_plan_matches_runtime_w_sizes<Cfg: CommitmentConfig>(num_vars: usize) {
        let key = AkitaScheduleLookupKey::singleton(num_vars);
        let plan = Cfg::schedule_plan(key)
            .expect("planner should succeed")
            .expect("config should provide a planner");
        let num_fold_levels = plan.num_fold_levels();
        for (idx, level) in plan.fold_levels().enumerate() {
            // The last fold in a fold-then-direct schedule is the terminal
            // recursive fold and ships its W in cleartext under
            // MRowLayout::Terminal (drops the D-block from the per-row `r`
            // quotients), so its `next_w_len` is smaller than what the
            // intermediate-layout helper would report.
            let is_terminal_fold = idx + 1 == num_fold_levels;
            let layout = if is_terminal_fold {
                akita_types::MRowLayout::Terminal
            } else {
                akita_types::MRowLayout::Intermediate
            };
            let runtime_next_w_len = akita_types::w_ring_element_count_with_counts_for_layout::<
                Cfg::Field,
            >(&level.lp, 1, 1, 1, 1, layout)
                * level.lp.ring_dimension;
            assert_eq!(
                runtime_next_w_len, level.next_inputs.current_w_len,
                "planner/runtime next_w_len mismatch at level {} for num_vars={num_vars}",
                level.inputs.level
            );
        }
    }

    #[cfg(not(feature = "zk"))]
    fn assert_generated_table_matches_cfg_schedule<Cfg: CommitmentConfig>(
        table: GeneratedScheduleTable,
    ) {
        for entry in table.entries {
            let key = AkitaScheduleLookupKey::new_with_points(
                entry.key.num_vars,
                entry.key.num_commitment_groups,
                entry.key.num_t_vectors,
                entry.key.num_w_vectors,
                entry.key.num_z_vectors,
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

    #[cfg(not(feature = "zk"))]
    fn assert_generated_batched_roots_are_scaled<Cfg: CommitmentConfig>(
        table: GeneratedScheduleTable,
    ) {
        let mut checked_folded_entry = false;
        for entry in table
            .entries
            .iter()
            .filter(|entry| entry.key.num_t_vectors > 1)
        {
            let key = AkitaScheduleLookupKey::new_with_points(
                entry.key.num_vars,
                entry.key.num_commitment_groups,
                entry.key.num_t_vectors,
                entry.key.num_w_vectors,
                entry.key.num_z_vectors,
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
                singleton_outer_width * entry.key.num_t_vectors,
                "generated batched root B width should be claim-scaled for key={key:?}"
            );
            assert_eq!(
                root.lp.d_matrix_width(),
                singleton_d_width * entry.key.num_t_vectors,
                "generated batched root D width should be claim-scaled for key={key:?}"
            );
        }
        assert!(
            checked_folded_entry,
            "generated table should include at least one folded batched entry"
        );
    }

    #[cfg(not(feature = "zk"))]
    fn assert_exact_root_fold_matches_runtime_root_plan<Cfg: CommitmentConfig, const D: usize>(
        num_vars: usize,
    ) {
        let key = AkitaScheduleLookupKey::singleton(num_vars);
        let plan = Cfg::schedule_plan(key)
            .expect("config schedule should succeed")
            .expect("config should provide an exact schedule");
        let planned_root = akita_types::exact_planned_level_execution(
            &plan,
            AkitaScheduleInputs {
                num_vars,
                level: 0,
                current_w_len: 1usize.checked_shl(num_vars as u32).unwrap_or(0),
            },
            plan.fold_levels()
                .next()
                .expect("exact schedule should begin with a fold")
                .lp
                .log_basis,
            Cfg::stage1_challenge_config,
        )
        .expect("exact plan should resolve the root fold")
        .expect("exact plan should contain a matching root fold");
        let incidence =
            ClaimIncidenceSummary::same_point(num_vars, 1).expect("singleton incidence");
        let runtime_root =
            Cfg::get_params_for_prove(&incidence).expect("runtime root plan should succeed");
        let Some(akita_types::Step::Fold(runtime_root_step)) = runtime_root.steps.first() else {
            panic!("runtime root schedule should start with a fold");
        };
        assert_eq!(
            planned_root.level.inputs.current_w_len,
            runtime_root_step.current_w_len,
            "planned/runtime root current_w_len mismatch for {} at num_vars={num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.lp,
            runtime_root_step.params,
            "planned/runtime root lp mismatch for {} at num_vars={num_vars}",
            std::any::type_name::<Cfg>()
        );
        assert_eq!(
            planned_root.level.next_inputs.current_w_len,
            runtime_root_step.next_w_len,
            "planned/runtime next_w_len mismatch for {} at num_vars={num_vars}",
            std::any::type_name::<Cfg>()
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_fp128_schedule_tables_match_cfg_schedule() {
        assert_generated_table_matches_cfg_schedule::<fp128::D32Full>(fp128_d32_full_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D32OneHot>(fp128_d32_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D64Full>(fp128_d64_full_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D64OneHot>(fp128_d64_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp128::D128Full>(fp128_d128_full_table());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_small_field_schedule_tables_match_cfg_schedule() {
        assert_generated_table_matches_cfg_schedule::<fp32::D64Full>(fp32_d64_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D64OneHot>(fp32_d64_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D128Full>(fp32_d128_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D128OneHot>(fp32_d128_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D256Full>(fp32_d256_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D256OneHot>(fp32_d256_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D512Full>(fp32_d512_table());
        assert_generated_table_matches_cfg_schedule::<fp32::D512OneHot>(fp32_d512_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D32Full>(fp64_d32_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D32OneHot>(fp64_d32_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D64Full>(fp64_d64_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D64OneHot>(fp64_d64_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D128Full>(fp64_d128_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D128OneHot>(fp64_d128_onehot_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D256Full>(fp64_d256_table());
        assert_generated_table_matches_cfg_schedule::<fp64::D256OneHot>(fp64_d256_onehot_table());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_batched_roots_restore_scaled_widths() {
        assert_generated_batched_roots_are_scaled::<fp128::D32Full>(fp128_d32_full_table());
        assert_generated_batched_roots_are_scaled::<fp128::D32OneHot>(fp128_d32_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp128::D64Full>(fp128_d64_full_table());
        assert_generated_batched_roots_are_scaled::<fp128::D64OneHot>(fp128_d64_onehot_table());
        assert_generated_batched_roots_are_scaled::<fp128::D128Full>(fp128_d128_full_table());
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_d32_full_root_fold_matches_runtime_root_plan() {
        assert_exact_root_fold_matches_runtime_root_plan::<fp128::D32Full, 32>(26);
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_d128_full_table_materializes_valid_plans() {
        let table = fp128_d128_full_table();
        for entry in table.entries {
            let key = AkitaScheduleLookupKey::new(
                entry.key.num_vars,
                entry.key.num_t_vectors,
                entry.key.num_w_vectors,
                entry.key.num_z_vectors,
            );
            generated_schedule_plan_from_table::<fp128::D128Full>(key, table)
                .expect("generated table should materialize")
                .expect("entry should exist in generated table");
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn generated_table_rejects_sis_family_mismatch() {
        let table = fp128_d128_full_table();
        let mismatched = GeneratedScheduleTable {
            sis_family: akita_types::SisModulusFamily::Q32,
            entries: table.entries,
        };
        let entry = mismatched
            .entries
            .iter()
            .find(|entry| entry.key.num_t_vectors == 1)
            .expect("fp128 table should contain singleton rows");
        let key = AkitaScheduleLookupKey::new_with_points(
            entry.key.num_vars,
            entry.key.num_commitment_groups,
            entry.key.num_t_vectors,
            entry.key.num_w_vectors,
            entry.key.num_z_vectors,
        );
        let err = generated_schedule_plan_from_table::<fp128::D128Full>(key, mismatched)
            .expect_err("mismatched SIS family must be rejected");
        assert!(
            err.to_string().contains("SIS family mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn adaptive_bounded_plan_matches_runtime_next_w_len() {
        for num_vars in [14, 20, 30] {
            assert_plan_matches_runtime_w_sizes::<fp128::D128Full>(num_vars);
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn adaptive_onehot_plan_matches_runtime_next_w_len() {
        for num_vars in [15, 30, 44] {
            assert_plan_matches_runtime_w_sizes::<fp128::D64OneHot>(num_vars);
        }
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn singleton_root_runtime_plan_matches_existing_root_layout() {
        type Cfg = fp128::D64OneHot;

        let incidence = ClaimIncidenceSummary::same_point(30, 1).expect("singleton incidence");
        let runtime = Cfg::get_params_for_prove(&incidence).expect("singleton runtime plan");
        let root_inputs = AkitaScheduleInputs {
            num_vars: 30,
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
        let inputs = AkitaScheduleInputs {
            num_vars: 30,
            level: 1,
            current_w_len: 25_974_272,
        };
        let params = Cfg::level_params_with_log_basis(inputs, log_basis);
        let decomp =
            recursive_level_decomposition_from_root(Cfg::decomposition(), params.log_basis);
        let num_ring = inputs.current_w_len / params.ring_dimension;
        let lp_12_7 = level_layout_from_params(12, 7, &params, decomp, num_ring).unwrap();
        let lp_11_8 = level_layout_from_params(11, 8, &params, decomp, num_ring).unwrap();
        let w_12_7 = planned_w_ring_element_count::<<Cfg as CommitmentConfig>::Field>(
            Cfg::decomposition().field_bits(),
            &lp_12_7,
        );
        let w_11_8 = planned_w_ring_element_count::<<Cfg as CommitmentConfig>::Field>(
            Cfg::decomposition().field_bits(),
            &lp_11_8,
        );
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
                decomp.field_bits(),
            ),
            (12, 7)
        );
    }

    #[test]
    #[cfg(not(feature = "zk"))]
    fn tight_block_len_is_no_larger_than_pow2() {
        for num_vars in [14, 20, 30] {
            let plan = fp128::D128Full::schedule_plan(AkitaScheduleLookupKey::singleton(num_vars))
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
                    num_vars
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
    fn batched_root_next_w_len_and_shape_track_total_claims() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        // Two opening-point shapes with the same total claim count: a 4+2
        // partition and a 3+3 partition. Under one-commit-per-point the planner
        // key projects only `num_polys_per_point`, so total `num_w_vectors`
        // matches between the two shapes.
        let incidence_a = point_local_incidence_summary(MAX_NUM_VARS, &[4, 2]);
        let incidence_b = point_local_incidence_summary(MAX_NUM_VARS, &[3, 3]);

        let plan_a = Cfg::get_params_for_prove(&incidence_a).unwrap();
        let plan_b = Cfg::get_params_for_prove(&incidence_b).unwrap();
        let Some(akita_types::Step::Fold(root_a)) = plan_a.steps.first() else {
            panic!("batch A schedule should start with a fold");
        };
        let Some(akita_types::Step::Fold(root_b)) = plan_b.steps.first() else {
            panic!("batch B schedule should start with a fold");
        };

        let next_w_ring_a = w_ring_element_count_with_counts::<<Cfg as CommitmentConfig>::Field>(
            &root_a.params,
            incidence_a.num_points(),
            incidence_a.num_polynomials(),
            incidence_a.num_claims(),
            incidence_a.num_public_rows(),
        );
        let next_w_ring_b = w_ring_element_count_with_counts::<<Cfg as CommitmentConfig>::Field>(
            &root_b.params,
            incidence_b.num_points(),
            incidence_b.num_polynomials(),
            incidence_b.num_claims(),
            incidence_b.num_public_rows(),
        );

        assert_eq!(next_w_ring_a, next_w_ring_b);
    }

    #[cfg(feature = "planner")]
    #[test]
    fn batched_root_next_w_len_depends_on_projected_schedule_key() {
        type Cfg = fp128::D64OneHot;
        const MAX_NUM_VARS: usize = 30;

        // Six total polynomials distributed differently across opening points.
        let one_point_six_polys = point_local_incidence_summary(MAX_NUM_VARS, &[6]);
        let six_points_one_poly = point_local_incidence_summary(MAX_NUM_VARS, &[1, 1, 1, 1, 1, 1]);

        let singleton_plan = Cfg::get_params_for_prove(&one_point_six_polys).unwrap();
        let multipoint_plan = Cfg::get_params_for_prove(&six_points_one_poly).unwrap();
        let Some(akita_types::Step::Fold(singleton_root)) = singleton_plan.steps.first() else {
            panic!("singleton schedule should start with a fold");
        };
        let Some(akita_types::Step::Fold(multipoint_root)) = multipoint_plan.steps.first() else {
            panic!("multipoint schedule should start with a fold");
        };

        assert_ne!(
            AkitaScheduleLookupKey::new_from_incidence(&one_point_six_polys).unwrap(),
            AkitaScheduleLookupKey::new_from_incidence(&six_points_one_poly).unwrap(),
        );
        assert_ne!(singleton_root.next_w_len, multipoint_root.next_w_len);
        assert_eq!(one_point_six_polys.num_points(), 1);
        assert_eq!(six_points_one_poly.num_points(), 6);
    }

    #[cfg(feature = "planner")]
    #[test]
    fn batched_root_layout_planner_direct_fallback_is_per_polynomial() {
        type Cfg = fp128::D32OneHot;
        const MAX_NUM_VARS: usize = 1;
        const NUM_CLAIMS: usize = 3;

        let table_miss_key = AkitaScheduleLookupKey::new(MAX_NUM_VARS, NUM_CLAIMS, NUM_CLAIMS, 1);
        assert!(
            Cfg::schedule_plan(table_miss_key).unwrap().is_none(),
            "test must exercise the planner fallback, not a generated table entry"
        );

        let planner_schedule =
            akita_planner::find_optimal_schedule::<Cfg>(table_miss_key).expect("planner fallback");
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
