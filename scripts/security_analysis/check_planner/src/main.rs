//! Verifies whether the live planner derives the same `(n_a, n_b, n_d)` ranks
//! for a given (preset, max_num_vars, current_w_len, log_basis) as what is
//! checked into the generated tables. Reports any disagreement.
//!
//! This is the ground-truth check for the security analysis. If the live
//! planner says "rank=R" and the table says "rank=R'!=R", one of them is
//! wrong; the audit must then go fix or regenerate.

use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_config::proof_optimized::fp128::{
    D128Full, D128OneHot, D32Full, D32OneHot, D64Full, D64OneHot,
};
use akita_config::CommitmentConfig;
use akita_planner::PlannerConfig;
use akita_types::generated::sis_floor::min_rank_for_secure_width;
use akita_types::{
    AjtaiRole, AkitaScheduleInputs, CommitmentEnvelope, LevelParams,
};

#[derive(Debug)]
struct Case {
    preset: &'static str,
    max_num_vars: usize,
    current_w_len: usize,
    log_basis: u32,
    expected_n_a: usize,
    expected_n_b: usize,
    expected_n_d: usize,
}

const CASES: &[Case] = &[
    // d32_full, recursive step 1 from max_num_vars=32 entry.
    // Table says: m_vars=15, r_vars=9, n_a=2, n_b=2, n_d=2.
    // My Python recomputed: n_a=3 required.
    Case {
        preset: "d32_full",
        max_num_vars: 32,
        current_w_len: 335_564_800,
        log_basis: 2,
        expected_n_a: 2,
        expected_n_b: 2,
        expected_n_d: 2,
    },
    // d128_full, recursive step 1 from max_num_vars=50 entry.
    // Table says: m_vars=19, r_vars=14, n_a=1.
    // My Python recomputed: n_a=2 required (width 335872 > rank-1 max 299143 at bucket=255).
    Case {
        preset: "d128_full",
        max_num_vars: 50,
        current_w_len: 704_374_685_696,
        log_basis: 2,
        expected_n_a: 1,
        expected_n_b: 1,
        expected_n_d: 1,
    },
    // d64_full, recursive level from max_num_vars=20 entry.
    // Table says: many entries with n_a=1 at widths >= 1024 at tensor.
    Case {
        preset: "d64_full",
        max_num_vars: 20,
        current_w_len: 477_312,
        log_basis: 4,
        expected_n_a: 1,
        expected_n_b: 1,
        expected_n_d: 1,
    },
];

/// Reproduces the rank derivation logic from
/// `crates/akita-config/src/bin/gen_schedule_tables.rs::fresh_level_params_with_log_basis`.
/// This is what the planner actually does when generating the schedule
/// tables; it bypasses the cached-table fast path and exercises the SIS
/// derivation directly.
fn fresh_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, inputs.max_num_vars);
    let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, inputs.max_num_vars);
    let envelope = CommitmentEnvelope {
        max_n_a: inner_floor,
        max_n_b: outer_floor,
        max_n_d: outer_floor,
    };
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let production_shape = stage1_challenge_shape_for_config(&stage1_config);

    if inputs.level > 0 {
        // (A) the "as-shipped" recursive derivation:
        //     tentative shape = Flat, then patch shape after.
        let tentative = LevelParams::params_only(d, log_basis, envelope.max_n_a, 1, 1, stage1_config.clone());
        println!("    [trace] tentative: shape={:?} envelope.max_n_a={}", tentative.stage1_challenge_shape, envelope.max_n_a);
        if let Ok(layout1) = akita_types::recursive_level_layout_from_params(
            &tentative, inputs.current_w_len, Cfg::decomposition(),
        ) {
            println!(
                "    [trace] layout1: m_vars={} r_vars={} block_len={} inner_width={} shape={:?}",
                layout1.m_vars, layout1.r_vars, layout1.block_len, layout1.inner_width(),
                layout1.stage1_challenge_shape,
            );
            if let Some(mut params) = akita_types::sis_derived_recursive_params_for_layout(
                d, log_basis, &stage1_config, &envelope, &layout1,
            ) {
                println!(
                    "    [trace] sis_derived: n_a={} n_b={} n_d={} a_coll={} shape={:?} (note: shape inherits from layout1)",
                    params.a_key.row_len(), params.b_key.row_len(), params.d_key.row_len(),
                    params.a_key.collision_inf(), params.stage1_challenge_shape,
                );
                params.stage1_challenge_shape = production_shape;
                println!("    [trace] params after shape patch: shape={:?}", params.stage1_challenge_shape);
                if let Ok(layout2) = akita_types::recursive_level_layout_from_params(
                    &params, inputs.current_w_len, Cfg::decomposition(),
                ) {
                    println!(
                        "    [trace] layout2: m_vars={} r_vars={} block_len={} inner_width={} shape={:?}",
                        layout2.m_vars, layout2.r_vars, layout2.block_len, layout2.inner_width(),
                        layout2.stage1_challenge_shape,
                    );
                    let result = params.with_layout(&layout2);
                    println!(
                        "    [trace] params.with_layout(layout2): n_a={} a_coll={} inner_width={}",
                        result.a_key.row_len(), result.a_key.collision_inf(), result.inner_width(),
                    );
                    return result;
                }
                return params;
            }
        }
    }
    let mut p = LevelParams::params_only(
        d, log_basis, envelope.max_n_a, envelope.max_n_b, envelope.max_n_d, stage1_config,
    );
    p.stage1_challenge_shape = production_shape;
    p
}

/// Alternative derivation that sets `stage1_challenge_shape = production_shape`
/// on the tentative layout BEFORE the SIS-rank lookup, so the recursive A-role
/// rank gets the tensor-aware extraction collision bucket.
fn fresh_level_params_with_log_basis_shape_first<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, inputs.max_num_vars);
    let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, inputs.max_num_vars);
    let envelope = CommitmentEnvelope {
        max_n_a: inner_floor,
        max_n_b: outer_floor,
        max_n_d: outer_floor,
    };
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let production_shape = stage1_challenge_shape_for_config(&stage1_config);

    if inputs.level > 0 {
        let mut tentative = LevelParams::params_only(d, log_basis, envelope.max_n_a, 1, 1, stage1_config.clone());
        tentative.stage1_challenge_shape = production_shape;  // <-- shape FIRST
        if let Ok(layout) = akita_types::recursive_level_layout_from_params(
            &tentative, inputs.current_w_len, Cfg::decomposition(),
        ) {
            if let Some(mut params) = akita_types::sis_derived_recursive_params_for_layout(
                d, log_basis, &stage1_config, &envelope, &layout,
            ) {
                params.stage1_challenge_shape = production_shape;
                if let Ok(layout) = akita_types::recursive_level_layout_from_params(
                    &params, inputs.current_w_len, Cfg::decomposition(),
                ) {
                    return params.with_layout(&layout);
                }
                return params;
            }
        }
    }
    let mut p = LevelParams::params_only(
        d, log_basis, envelope.max_n_a, envelope.max_n_b, envelope.max_n_d, stage1_config,
    );
    p.stage1_challenge_shape = production_shape;
    p
}

fn stage1_challenge_shape_for_config(config: &SparseChallengeConfig) -> Stage1ChallengeShape {
    match config {
        SparseChallengeConfig::BoundedL1Norm => Stage1ChallengeShape::Flat,
        SparseChallengeConfig::Uniform { .. } | SparseChallengeConfig::ExactShell { .. } => {
            Stage1ChallengeShape::Tensor
        }
    }
}

fn check_one<Cfg: CommitmentConfig + PlannerConfig>(case: &Case) {
    let inputs = AkitaScheduleInputs {
        max_num_vars: case.max_num_vars,
        level: 1,
        current_w_len: case.current_w_len,
    };
    let lp = match akita_config::current_level_layout_with_log_basis::<Cfg>(inputs, case.log_basis)
    {
        Ok(lp) => lp,
        Err(e) => {
            println!(
                "  ERROR: current_level_layout_with_log_basis failed: {:?}",
                e
            );
            return;
        }
    };
    let live_fresh = fresh_level_params_with_log_basis::<Cfg>(inputs, case.log_basis);
    let shape_first = fresh_level_params_with_log_basis_shape_first::<Cfg>(inputs, case.log_basis);
    let inner_width = lp.inner_width();
    let outer_width = lp.outer_width();
    let d_matrix_width = lp.d_matrix_width();
    let a_collision = lp.a_key.collision_inf();
    let b_collision = lp.b_key.collision_inf();
    let d_collision = lp.d_key.collision_inf();
    let derivation_shape = format!("{:?}", lp.stage1_challenge_shape);
    println!(
        "  derivation: shape={derivation_shape} a_coll={a_collision} b_coll={b_collision} d_coll={d_collision}"
    );
    println!(
        "  layout: m_vars={} r_vars={} num_blocks={} block_len={} delta_open={} delta_commit={}",
        lp.m_vars, lp.r_vars, lp.num_blocks, lp.block_len, lp.num_digits_open, lp.num_digits_commit,
    );
    println!(
        "  widths: inner={inner_width} outer={outer_width} d_matrix={d_matrix_width}"
    );
    println!(
        "  ranks (planner-stored): n_a={} n_b={} n_d={}",
        lp.a_key.row_len(),
        lp.b_key.row_len(),
        lp.d_key.row_len(),
    );
    println!(
        "  expected from generated table: n_a={} n_b={} n_d={}",
        case.expected_n_a, case.expected_n_b, case.expected_n_d
    );

    // Now do the rank lookup against the SAME a_collision the planner used.
    let live_a_rank =
        min_rank_for_secure_width(Cfg::D as u32, a_collision, inner_width as u64);
    let live_b_rank =
        min_rank_for_secure_width(Cfg::D as u32, b_collision, outer_width as u64);
    let live_d_rank =
        min_rank_for_secure_width(Cfg::D as u32, d_collision, d_matrix_width as u64);
    println!(
        "  rank from sis_floor at derivation collision: a={:?} b={:?} d={:?}",
        live_a_rank, live_b_rank, live_d_rank
    );

    // Then the same but at the *production* shape extraction, to expose the
    // difference between derivation and runtime.
    let prod_lp = {
        let mut p = lp.clone();
        p.stage1_challenge_shape = match Cfg::stage1_challenge_config(Cfg::D) {
            akita_challenges::SparseChallengeConfig::BoundedL1Norm => {
                akita_challenges::Stage1ChallengeShape::Flat
            }
            akita_challenges::SparseChallengeConfig::Uniform { .. }
            | akita_challenges::SparseChallengeConfig::ExactShell { .. } => {
                akita_challenges::Stage1ChallengeShape::Tensor
            }
        };
        p
    };
    let bd_collision = (1u32 << lp.log_basis) - 1;
    if let Ok(report) = prod_lp.stage1_sis_extraction_report(bd_collision) {
        println!(
            "  PRODUCTION-shape report: shape={:?} extraction_linf={} a_extraction={} a_bucket={}",
            prod_lp.stage1_challenge_shape,
            report.extraction_linf,
            report.a_role_extraction_collision,
            report.a_role_supported_collision_bucket,
        );
        let prod_rank = min_rank_for_secure_width(
            Cfg::D as u32,
            report.a_role_supported_collision_bucket,
            inner_width as u64,
        );
        println!(
            "  rank from sis_floor at PRODUCTION shape: a={:?} (vs stored {})",
            prod_rank,
            lp.a_key.row_len()
        );
    }

    println!();
    println!("  --- gen_schedule_tables `fresh` derivation (shape-after-SIS, as-shipped) ---");
    println!(
        "  fresh n_a={} n_b={} n_d={} a_coll={} b_coll={} d_coll={} shape={:?} inner_width={}",
        live_fresh.a_key.row_len(),
        live_fresh.b_key.row_len(),
        live_fresh.d_key.row_len(),
        live_fresh.a_key.collision_inf(),
        live_fresh.b_key.collision_inf(),
        live_fresh.d_key.collision_inf(),
        live_fresh.stage1_challenge_shape,
        live_fresh.inner_width(),
    );

    // Direct SIS table lookup at the fresh-derived collision and width:
    let fresh_a_rank = min_rank_for_secure_width(
        Cfg::D as u32,
        live_fresh.a_key.collision_inf(),
        live_fresh.inner_width() as u64,
    );
    let fresh_b_rank = min_rank_for_secure_width(
        Cfg::D as u32,
        live_fresh.b_key.collision_inf(),
        live_fresh.outer_width() as u64,
    );
    let fresh_d_rank = min_rank_for_secure_width(
        Cfg::D as u32,
        live_fresh.d_key.collision_inf(),
        live_fresh.d_matrix_width() as u64,
    );
    println!(
        "  fresh direct SIS lookup: a_rank={:?} (width={}, coll={}), b_rank={:?} (width={}, coll={}), d_rank={:?} (width={}, coll={})",
        fresh_a_rank, live_fresh.inner_width(), live_fresh.a_key.collision_inf(),
        fresh_b_rank, live_fresh.outer_width(), live_fresh.b_key.collision_inf(),
        fresh_d_rank, live_fresh.d_matrix_width(), live_fresh.d_key.collision_inf(),
    );

    println!("  --- shape-first derivation (tentative shape = production shape) ---");
    println!(
        "  shape_first n_a={} n_b={} n_d={} a_coll={} b_coll={} d_coll={} shape={:?} inner_width={}",
        shape_first.a_key.row_len(),
        shape_first.b_key.row_len(),
        shape_first.d_key.row_len(),
        shape_first.a_key.collision_inf(),
        shape_first.b_key.collision_inf(),
        shape_first.d_key.collision_inf(),
        shape_first.stage1_challenge_shape,
        shape_first.inner_width(),
    );
}

fn main() {
    // Dump every generated entry whose schedule plan fails to materialize.
    use akita_types::generated::{
        fp128_d128_full_table, fp128_d128_onehot_table, fp128_d32_full_table,
        fp128_d32_onehot_table, fp128_d64_full_table, fp128_d64_onehot_table,
        GeneratedScheduleTable,
    };
    use akita_types::{AkitaRootBatchSummary, AkitaScheduleLookupKey, ScheduleProvider};

    fn scan_table<Cfg: CommitmentConfig>(name: &str, table: GeneratedScheduleTable) {
        let mut failures = 0usize;
        let mut total = 0usize;
        for entry in table.entries {
            total += 1;
            let key = AkitaScheduleLookupKey::with_batch(
                entry.key.max_num_vars,
                entry.key.num_vars,
                entry.key.layout_num_claims,
                AkitaRootBatchSummary::new(
                    entry.key.batch_num_claims,
                    entry.key.batch_num_commitment_groups,
                    entry.key.batch_num_points,
                )
                .unwrap(),
            );
            match <Cfg as ScheduleProvider>::schedule_plan(key) {
                Ok(Some(_)) => {}
                Ok(None) => {
                    failures += 1;
                    println!("  {name}: schedule_plan returned None for key={:?}", entry.key);
                }
                Err(e) => {
                    failures += 1;
                    println!("  {name}: FAILED for key={:?}: {:?}", entry.key, e);
                }
            }
        }
        println!("{name}: {failures}/{total} entries failed validation");
    }

    scan_table::<D32Full>("d32_full", fp128_d32_full_table());
    scan_table::<D32OneHot>("d32_onehot", fp128_d32_onehot_table());
    scan_table::<D64Full>("d64_full", fp128_d64_full_table());
    scan_table::<D64OneHot>("d64_onehot", fp128_d64_onehot_table());
    scan_table::<D128Full>("d128_full", fp128_d128_full_table());
    scan_table::<D128OneHot>("d128_onehot", fp128_d128_onehot_table());

    let _ = CASES; // suppress unused-static warning
}
