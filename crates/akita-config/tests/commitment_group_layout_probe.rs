use akita_config::proof_optimized::fp128;
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_planner::{find_schedule, PlannerPolicy};
use akita_types::sis::{min_secure_rank, rounded_up_collision_norm_t};
use akita_types::{AkitaScheduleLookupKey, LevelParams, Step};

type Cfg = fp128::D64OneHot;

struct LayoutSummary {
    m_vars: usize,
    r_vars: usize,
    log_basis: u32,
    n_a: usize,
    n_b_at_layout_basis: usize,
    conservative_n_b: usize,
    t_hat_g: usize,
}

fn root_params(schedule: &akita_types::Schedule) -> Result<&LevelParams, AkitaError> {
    match schedule.steps.first() {
        Some(Step::Fold(fold)) => Ok(&fold.params),
        Some(Step::Direct(direct)) => direct.params.as_ref().ok_or_else(|| {
            AkitaError::InvalidSetup("root-direct schedule has no commit params".to_string())
        }),
        None => Err(AkitaError::InvalidSetup("empty schedule".to_string())),
    }
}

fn layout_summary(
    policy: &PlannerPolicy,
    num_vars: usize,
    max_basis: u32,
) -> Result<LayoutSummary, AkitaError> {
    let key = AkitaScheduleLookupKey::new(num_vars, 1, 1, 1);
    let schedule = find_schedule(
        key,
        policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;
    let params = root_params(&schedule)?;
    let b_width = params.b_key.col_len();
    let norm_at_lmax = rounded_up_collision_norm_t(Cfg::sis_modulus_family(), Cfg::D, max_basis)
        .ok_or_else(|| AkitaError::InvalidSetup("B norm overflow".to_string()))?;
    let conservative_n_b = min_secure_rank(
        Cfg::sis_modulus_family(),
        Cfg::D as u32,
        norm_at_lmax,
        b_width as u64,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("B rank lookup failed".to_string()))?;
    let t_hat_g = params
        .num_blocks
        .checked_mul(params.a_key.row_len())
        .and_then(|n| n.checked_mul(params.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("t_hat_g overflow".to_string()))?;

    Ok(LayoutSummary {
        m_vars: params.m_vars,
        r_vars: params.r_vars,
        log_basis: params.log_basis,
        n_a: params.a_key.row_len(),
        n_b_at_layout_basis: params.b_key.row_len(),
        conservative_n_b,
        t_hat_g,
    })
}

fn print_layout(result: Result<LayoutSummary, AkitaError>) {
    match result {
        Ok(layout) => print!(
            ",ok,{},{},{},{},{},{},{}",
            layout.m_vars,
            layout.r_vars,
            layout.log_basis,
            layout.n_a,
            layout.n_b_at_layout_basis,
            layout.conservative_n_b,
            layout.t_hat_g
        ),
        Err(err) => print!(",error:{err:?},,,,,,,",),
    }
}

#[test]
fn print_commitment_group_layout_probe() -> Result<(), AkitaError> {
    let mut limited_policy = policy_of::<Cfg>();
    let full_policy = policy_of::<Cfg>();
    let (min_basis, max_basis) = Cfg::basis_range();
    limited_policy.basis_range = (min_basis, min_basis);

    println!("cfg=fp128::D64OneHot K_g=1 min_basis={min_basis} max_basis={max_basis}");
    println!(
        "num_vars,\
limited_status,limited_m,limited_r,limited_log_basis,limited_n_a,limited_b_width,limited_n_b_at_basis,limited_conservative_n_b,limited_t_hat_g,\
full_status,full_m,full_r,full_log_basis,full_n_a,full_b_width,full_n_b_at_basis,full_conservative_n_b,full_t_hat_g"
    );

    for num_vars in 20..=60 {
        print!("{num_vars}");
        print_layout(layout_summary(&limited_policy, num_vars, max_basis));
        print_layout(layout_summary(&full_policy, num_vars, max_basis));
        println!();
    }

    Ok(())
}
