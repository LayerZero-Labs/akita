//! Diagnostic: print limited-vs-full basis planner layout summaries for
//! multi-group sizing exploration.
//!
//! Marked `#[ignore]` so it never runs in default CI; invoke with
//! `cargo test -p akita-config --test commitment_group_layout_probe -- --ignored --nocapture`.

use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_planner::{find_group_batch_schedule, PlannerPolicy};
use akita_types::{AkitaScheduleLookupKey, LevelParams, PolynomialGroupLayout};

type Cfg = fp128::D64OneHot;

struct LayoutSummary {
    position_index_bits: usize,
    block_index_bits: usize,
    log_basis_inner: u32,
    log_basis_outer: u32,
    log_basis_open: u32,
    n_a: usize,
    n_b: usize,
    t_hat_g: usize,
}

fn root_params(schedule: &akita_types::Schedule) -> Result<&LevelParams, AkitaError> {
    schedule
        .folds
        .first()
        .map(|fold| &fold.params)
        .ok_or_else(|| AkitaError::InvalidSetup("schedule is missing its root fold".to_string()))
}

fn layout_summary(policy: &PlannerPolicy, num_vars: usize) -> Result<LayoutSummary, AkitaError> {
    let key = PolynomialGroupLayout::new(num_vars, 1);
    let schedule = find_group_batch_schedule(
        &AkitaScheduleLookupKey::single(key),
        policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    let params = root_params(&schedule)?;
    let t_hat_g = params
        .num_live_blocks
        .checked_mul(params.inner_commit_matrix.output_rank())
        .and_then(|n| n.checked_mul(params.num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("t_hat_g overflow".to_string()))?;

    Ok(LayoutSummary {
        position_index_bits: params.position_index_bits(),
        block_index_bits: params.block_index_bits(),
        log_basis_inner: params.log_basis_inner,
        log_basis_outer: params.log_basis_outer,
        log_basis_open: params.log_basis_open,
        n_a: params.inner_commit_matrix.output_rank(),
        n_b: params.outer_commit_matrix.output_rank(),
        t_hat_g,
    })
}

fn print_layout(result: Result<LayoutSummary, AkitaError>) {
    match result {
        Ok(layout) => print!(
            ",ok,{},{},{},{},{},{},{},{}",
            layout.position_index_bits,
            layout.block_index_bits,
            layout.log_basis_inner,
            layout.log_basis_outer,
            layout.log_basis_open,
            layout.n_a,
            layout.n_b,
            layout.t_hat_g
        ),
        Err(err) => print!(",error:{err:?},,,,,,,,,",),
    }
}

#[test]
#[ignore = "diagnostic"]
fn print_commitment_group_layout_probe() -> Result<(), AkitaError> {
    let mut limited_policy = policy_of::<Cfg>();
    let full_policy = policy_of::<Cfg>();
    let (min_basis, max_basis) = Cfg::basis_range();
    limited_policy.basis_range = (min_basis, min_basis);

    println!("cfg=fp128::D64OneHot K_g=1 min_basis={min_basis} max_basis={max_basis}");
    println!(
        "num_vars,\
limited_status,limited_m,limited_r,limited_log_basis_inner,limited_log_basis_outer,limited_log_basis_open,limited_n_a,limited_n_b,limited_t_hat_g,\
full_status,full_m,full_r,full_log_basis_inner,full_log_basis_outer,full_log_basis_open,full_n_a,full_n_b,full_t_hat_g"
    );

    for num_vars in 20..=60 {
        print!("{num_vars}");
        print_layout(layout_summary(&limited_policy, num_vars));
        print_layout(layout_summary(&full_policy, num_vars));
        println!();
    }

    Ok(())
}
