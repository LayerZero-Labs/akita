use akita_config::proof_optimized::fp32;
use akita_config::{policy_of, CommitmentConfig};
use akita_field::{AkitaError, CanonicalField};
use akita_planner::find_schedule;
use akita_types::sis::{
    choose_op_norm_rejection_for_a_role, decomposed_s_block_ring_count, decomposed_t_ring_count,
    decomposed_w_ring_count, fold_challenge_norms, folded_witness_public_linf_cap, min_secure_rank,
    num_digits_open, num_digits_s_commit, rounded_up_collision_norm_t, rounded_up_collision_norm_w,
    AjtaiKeyParams, FoldWitnessLinfCapConfig, FoldWitnessNorms,
};
use akita_types::{
    w_ring_element_count_with_counts_for_layout_bits, AkitaScheduleInputs, AkitaScheduleLookupKey,
    DecompositionParams, LevelParams, MRowLayout, Step,
};

type Cfg = fp32::D256Full;

fn step_params(step: &Step) -> Option<&LevelParams> {
    match step {
        Step::Fold(fold) => Some(&fold.params),
        Step::Direct(direct) => direct.params.as_ref(),
    }
}

fn print_schedule(num_vars: usize, num_polynomials: usize) -> Result<(), AkitaError> {
    let key = AkitaScheduleLookupKey::new(num_vars, num_polynomials);
    let policy = policy_of::<Cfg>();
    let schedule = find_schedule(
        key,
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?;

    println!(
        "nv={num_vars} polys={num_polynomials} total_bytes={} steps={}",
        schedule.total_bytes,
        schedule.steps.len()
    );
    for (idx, step) in schedule.steps.iter().enumerate() {
        match step {
            Step::Fold(fold) => {
                let lp = &fold.params;
                let z_cap = lp.fold_witness_linf_cap_for_claims(num_polynomials)?;
                let depth_fold = lp.num_digits_fold(
                    num_polynomials,
                    <Cfg as CommitmentConfig>::Field::modulus_bits(),
                )?;
                println!(
                    "  fold#{idx}: in={} out={} bytes={} lb={} m={} r={} blocks={} block_len={} A(n={},w={},bucket={}) B(n={},w={}) D(n={},w={}) z_cap={} delta_fold={} opnorm={} tier={}",
                    fold.current_w_len,
                    fold.next_w_len,
                    fold.level_bytes,
                    lp.log_basis,
                    lp.m_vars,
                    lp.r_vars,
                    lp.num_blocks,
                    lp.block_len,
                    lp.a_key.row_len(),
                    lp.inner_width(),
                    lp.a_key.collision_linf(),
                    lp.b_key.row_len(),
                    lp.outer_width(),
                    lp.d_key.row_len(),
                    lp.d_matrix_width(),
                    z_cap,
                    depth_fold,
                    lp.op_norm_rejection,
                    lp.tier_split,
                );
            }
            Step::Direct(direct) => {
                let root = step_params(step)
                    .map(|lp| format!(" root_A(n={},w={})", lp.a_key.row_len(), lp.inner_width()))
                    .unwrap_or_default();
                println!(
                    "  direct#{idx}: in={} bytes={} shape={:?}{}",
                    direct.current_w_len, direct.direct_bytes, direct.witness_shape, root
                );
            }
        }
    }
    Ok(())
}

fn scan_root_candidates(num_vars: usize, num_polynomials: usize) -> Result<(), AkitaError> {
    let policy = policy_of::<Cfg>();
    let ring_challenge_cfg = Cfg::ring_challenge_config(policy.ring_dimension)?;
    let fold_challenge_shape = Cfg::fold_challenge_shape_at_level(AkitaScheduleInputs {
        num_vars,
        level: 0,
        current_w_len: 1usize << num_vars,
    });
    let alpha = (policy.ring_dimension as u32).trailing_zeros() as usize;
    let reduced_vars = num_vars.saturating_sub(alpha);
    let min_r_vars = if reduced_vars >= 3 { 1 } else { 0 };
    let max_r_vars = reduced_vars.saturating_sub(1);
    let field_bits = <Cfg as CommitmentConfig>::Field::modulus_bits();
    let mut fail_a = 0usize;
    let mut fail_b = 0usize;
    let mut fail_d = 0usize;
    let mut fail_shrink = 0usize;
    let mut viable = 0usize;
    let mut a_rejects: Vec<(u128, u128, u64, u32, usize, usize, usize)> = Vec::new();

    println!("root-scan nv={num_vars} reduced_vars={reduced_vars}");
    for log_basis in Cfg::basis_range().0..=Cfg::basis_range().1 {
        let level_decomp = DecompositionParams {
            log_basis,
            ..policy.decomposition
        };
        let depth_commit = num_digits_s_commit(level_decomp, true);
        let depth_open = num_digits_open(level_decomp);
        for r_vars in (min_r_vars..=max_r_vars).rev() {
            let m_vars = reduced_vars - r_vars;
            let block_len = 1usize << m_vars;
            let num_blocks = 1usize << r_vars;
            let Some(width_s) = decomposed_s_block_ring_count(block_len, depth_commit) else {
                fail_a += 1;
                continue;
            };
            let Some((op_norm_rejection, norm_s, n_a)) = choose_op_norm_rejection_for_a_role(
                policy.sis_family,
                policy.ring_dimension,
                level_decomp,
                &ring_challenge_cfg,
                fold_challenge_shape,
                true,
                policy.onehot_chunk_size,
                policy.ring_subfield_norm_bound,
                r_vars,
                num_polynomials,
                width_s as u64,
            ) else {
                let challenge = fold_challenge_norms(&ring_challenge_cfg, fold_challenge_shape);
                let witness = FoldWitnessNorms::new(
                    log_basis,
                    policy.ring_dimension,
                    policy.onehot_chunk_size,
                    false,
                );
                if let Ok(cap_config) = FoldWitnessLinfCapConfig::for_fold_level(
                    &ring_challenge_cfg,
                    fold_challenge_shape,
                    policy.ring_dimension,
                    false,
                    width_s,
                ) {
                    if let Ok(z_cap) = folded_witness_public_linf_cap(
                        challenge,
                        witness,
                        r_vars,
                        num_polynomials,
                        &cap_config,
                    ) {
                        let mass =
                            fold_challenge_shape.effective_l1_mass(&ring_challenge_cfg) as u128;
                        let collision_linf = 8u128
                            .saturating_mul(mass)
                            .saturating_mul(z_cap)
                            .saturating_mul(u128::from(policy.ring_subfield_norm_bound));
                        a_rejects.push((
                            collision_linf,
                            z_cap,
                            width_s as u64,
                            log_basis,
                            m_vars,
                            r_vars,
                            block_len,
                        ));
                    }
                }
                fail_a += 1;
                continue;
            };
            let a_key = AjtaiKeyParams::try_new(
                policy.sis_family,
                n_a,
                width_s,
                norm_s,
                policy.ring_dimension,
            )?;
            let Some(norm_t) =
                rounded_up_collision_norm_t(policy.sis_family, policy.ring_dimension, log_basis)
            else {
                fail_b += 1;
                continue;
            };
            let Some(width_t) =
                decomposed_t_ring_count(n_a, depth_open, num_blocks, num_polynomials)
            else {
                fail_b += 1;
                continue;
            };
            let Some(n_b) = min_secure_rank(
                policy.sis_family,
                policy.ring_dimension as u32,
                norm_t,
                width_t as u64,
            ) else {
                fail_b += 1;
                continue;
            };
            let b_key = AjtaiKeyParams::try_new(
                policy.sis_family,
                n_b,
                width_t,
                norm_t,
                policy.ring_dimension,
            )?;
            let Some(norm_w) =
                rounded_up_collision_norm_w(policy.sis_family, policy.ring_dimension, log_basis)
            else {
                fail_d += 1;
                continue;
            };
            let Some(width_w) = decomposed_w_ring_count(depth_open, num_blocks, num_polynomials)
            else {
                fail_d += 1;
                continue;
            };
            let Some(n_d) = min_secure_rank(
                policy.sis_family,
                policy.ring_dimension as u32,
                norm_w,
                width_w as u64,
            ) else {
                fail_d += 1;
                continue;
            };
            let d_key = AjtaiKeyParams::try_new(
                policy.sis_family,
                n_d,
                width_w,
                norm_w,
                policy.ring_dimension,
            )?;
            let Ok(lp) = LevelParams {
                ring_dimension: policy.ring_dimension,
                log_basis,
                a_key,
                b_key,
                d_key,
                num_blocks,
                block_len,
                m_vars,
                r_vars,
                stage1_config: ring_challenge_cfg.clone(),
                op_norm_rejection,
                fold_challenge_shape,
                num_digits_commit: depth_commit,
                num_digits_open: depth_open,
                onehot_chunk_size: 0,
                tier_split: 1,
                f_key: None,
                fold_linf_cap_config: FoldWitnessLinfCapConfig::worst_case_beta_only(),
                num_digits_fold_one: 1,
                field_bits_hint: 0,
                cached_num_digits_fold_claims: 0,
                cached_num_digits_fold_value: 1,
            }
            .with_fold_linf_cap_config(field_bits, num_polynomials) else {
                fail_a += 1;
                continue;
            };
            let next_w_len = w_ring_element_count_with_counts_for_layout_bits(
                field_bits,
                &lp,
                num_polynomials,
                1,
                MRowLayout::WithDBlock,
            )? * policy.ring_dimension;
            if next_w_len * log_basis as usize >= (1usize << num_vars) * field_bits as usize {
                fail_shrink += 1;
                continue;
            }
            viable += 1;
            let z_cap = lp.fold_witness_linf_cap_for_claims(num_polynomials)?;
            println!(
                "  viable lb={log_basis} m={m_vars} r={r_vars} A(n={},w={},bucket={}) B(n={},w={}) D(n={},w={}) next={} z_cap={}",
                lp.a_key.row_len(),
                lp.inner_width(),
                lp.a_key.collision_linf(),
                lp.b_key.row_len(),
                lp.outer_width(),
                lp.d_key.row_len(),
                lp.d_matrix_width(),
                next_w_len,
                z_cap,
            );
        }
    }
    println!("root-scan summary: viable={viable} fail_a={fail_a} fail_b={fail_b} fail_d={fail_d} fail_shrink={fail_shrink}");
    a_rejects.sort_by_key(|&(collision_linf, _, width, _, _, _, _)| (collision_linf, width));
    for &(collision_linf, z_cap, width, log_basis, m_vars, r_vars, block_len) in
        a_rejects.iter().take(8)
    {
        println!(
            "  best-A-reject lb={log_basis} m={m_vars} r={r_vars} block_len={block_len} width={width} z_cap={z_cap} collision_linf={collision_linf}"
        );
    }
    a_rejects.sort_by_key(|&(_, _, width, _, _, _, _)| width);
    for (collision_linf, z_cap, width, log_basis, m_vars, r_vars, block_len) in
        a_rejects.iter().take(8)
    {
        println!(
            "  narrow-A-reject lb={log_basis} m={m_vars} r={r_vars} block_len={block_len} width={width} z_cap={z_cap} collision_linf={collision_linf}"
        );
    }
    a_rejects.sort_by_key(|&(collision_linf, _, width, _, _, _, _)| (width > 8192, collision_linf));
    for (collision_linf, z_cap, width, log_basis, m_vars, r_vars, block_len) in a_rejects
        .iter()
        .filter(|(_, _, width, _, _, _, _)| *width <= 8192)
        .take(8)
    {
        println!(
            "  estimable-A-reject lb={log_basis} m={m_vars} r={r_vars} block_len={block_len} width={width} z_cap={z_cap} collision_linf={collision_linf}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "diagnostic: run with --ignored --nocapture to inspect fp32 dense planner output"]
fn fp32_d256_dense_nv24_to_nv32() -> Result<(), AkitaError> {
    for num_vars in 24..=32 {
        print_schedule(num_vars, 1)?;
    }
    scan_root_candidates(31, 1)?;
    scan_root_candidates(32, 1)?;
    Ok(())
}
