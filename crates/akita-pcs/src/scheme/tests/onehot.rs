use super::*;

#[test]
fn profile_native_commit_group_returns_exact_frozen_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let profile =
        akita_config::committed_group_profile::<OneHotCfg>(&key).expect("precommit profile");
    let total_field = (profile.num_live_blocks * profile.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_0001)];

    let setup = OneHotScheme::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let (commitment, _hint) =
        OneHotScheme::commit_group(&setup, &polys, &stack).expect("precommit");
    let frozen_layout = commitment.profile;

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.num_positions_per_block,
        profile.num_positions_per_block
    );
    assert_eq!(frozen_layout.num_live_blocks, profile.num_live_blocks);
    assert_eq!(
        frozen_layout.log_basis_outer,
        OneHotCfg::opening_basis_range().0
    );
    assert_eq!(
        frozen_layout.inner_commit_matrix.output_rank(),
        profile.inner_commit_matrix.output_rank()
    );
    assert_eq!(
        frozen_layout.outer_commit_matrix.output_rank(),
        profile.outer_commit_matrix.output_rank()
    );
    assert_eq!(
        commitment.rows().count(),
        frozen_layout.outer_commit_matrix.output_rank()
    );
}

fn multi_group_root_params(schedule: &akita_types::FoldSchedule) -> &CommittedGroupParams {
    &schedule.root.params.final_group.commitment
}

fn with_precommit_stack<R>(
    max_num_vars: usize,
    max_num_polys: usize,
    run: impl FnOnce(
        &akita_prover::AkitaProverSetup<OneHotF>,
        &akita_prover::UniformProverStack<'_, OneHotF, CpuBackend>,
    ) -> R,
) -> R {
    let setup = OneHotScheme::setup_prover(max_num_vars, max_num_polys).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    run(&setup, &stack)
}

#[test]
fn profile_native_commit_group_allows_independent_groups() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;
    // Precommitted groups are committed independently, so setup only needs to
    // cover the largest standalone group rather than the sum of all groups.
    const SETUP_CAPACITY_SIZE: usize = PRE_B_SIZE;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(NV, PRE_B_SIZE);
    let pre_a_profile = akita_config::committed_group_profile::<OneHotCfg>(&pre_a_key)
        .expect("precommit A profile");
    let pre_b_profile = akita_config::committed_group_profile::<OneHotCfg>(&pre_b_key)
        .expect("precommit B profile");
    let pre_a_polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_1001)];
    let pre_b_polys = [
        debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_2001),
        debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_2002),
    ];

    with_precommit_stack(NV, SETUP_CAPACITY_SIZE, |setup, stack| {
        let (pre_a_commitment, _pre_a_hint) =
            OneHotScheme::commit_group(setup, &pre_a_polys, stack).expect("precommit A");
        let (pre_b_commitment, _pre_b_hint) =
            OneHotScheme::commit_group(setup, &pre_b_polys, stack).expect("precommit B");
        let pre_a_frozen = pre_a_commitment.profile;
        let pre_b_frozen = pre_b_commitment.profile;

        assert_eq!(pre_a_frozen.group, pre_a_key);
        assert_eq!(pre_b_frozen.group, pre_b_key);
        assert_eq!(
            pre_a_commitment.rows().count(),
            pre_a_frozen.outer_commit_matrix.output_rank()
        );
        assert_eq!(
            pre_b_commitment.rows().count(),
            pre_b_frozen.outer_commit_matrix.output_rank()
        );
        assert_ne!(pre_a_frozen.group, pre_b_frozen.group);
        assert_eq!(pre_a_frozen, pre_a_profile);
        assert_eq!(pre_b_frozen, pre_b_profile);
    });
}

#[test]
fn group_batch_schedule_preserves_precommitted_order() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 1;
    const PRE_C_SIZE: usize = 1;
    const MAIN_SIZE: usize = 4;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_B_SIZE);
    let pre_c_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_C_SIZE);
    let pre_a_frozen = akita_config::committed_group_profile::<OneHotCfg>(&pre_a_key)
        .expect("precommit A profile");
    let pre_b_frozen = akita_config::committed_group_profile::<OneHotCfg>(&pre_b_key)
        .expect("precommit B profile");
    let pre_c_frozen = akita_config::committed_group_profile::<OneHotCfg>(&pre_c_key)
        .expect("precommit C profile");
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, MAIN_SIZE),
        precommitteds: vec![pre_a_frozen, pre_b_frozen, pre_c_frozen],
    };

    let schedule =
        OneHotCfg::runtime_schedule(multi_group_key.clone()).expect("multi-group runtime schedule");
    let root = multi_group_root_params(&schedule);
    let main_params = schedule.root.params.final_group.commitment.clone();

    assert_eq!(multi_group_key.num_commitment_groups(), 4);
    assert_eq!(
        multi_group_key
            .num_polynomials()
            .expect("multi-group polynomial count"),
        PRE_A_SIZE + PRE_B_SIZE + PRE_C_SIZE + MAIN_SIZE
    );
    assert_eq!(main_params, *root);
    assert_eq!(schedule.root.params.precommitted_groups.len(), 3);
    assert_eq!(
        schedule.root.params.precommitted_groups[0].descriptor,
        pre_a_frozen
    );
    assert_eq!(
        schedule.root.params.precommitted_groups[1].descriptor,
        pre_b_frozen
    );
    assert_eq!(
        schedule.root.params.precommitted_groups[2].descriptor,
        pre_c_frozen
    );
}

#[test]
fn group_batch_commits_independent_arity_precommitteds() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;
    const GROUP_SIZE: usize = 1;
    const FINAL_SIZE: usize = 4;
    const SETUP_CAPACITY_SIZE: usize = FINAL_SIZE + 2 * GROUP_SIZE;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_a_frozen = akita_config::committed_group_profile::<OneHotCfg>(&pre_a_key)
        .expect("precommit A profile");
    let pre_b_frozen = akita_config::committed_group_profile::<OneHotCfg>(&pre_b_key)
        .expect("precommit B profile");
    let pre_a_polys = [debug_make_onehot_poly(
        PRE_NV,
        ONEHOT_D,
        0x0bee_fcaf_9a77_5001,
    )];
    let pre_b_polys = [debug_make_onehot_poly(
        PRE_NV,
        ONEHOT_D,
        0x0bee_fcaf_9a77_6001,
    )];

    let setup = OneHotScheme::setup_prover(FINAL_NV, SETUP_CAPACITY_SIZE).expect("protocol setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared protocol setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("protocol stack");
    let (pre_a_commitment, _pre_a_hint) =
        OneHotScheme::commit_group::<_, _>(&setup, &pre_a_polys, &stack).expect("precommit A");
    let (pre_b_commitment, _pre_b_hint) =
        OneHotScheme::commit_group::<_, _>(&setup, &pre_b_polys, &stack).expect("precommit B");
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, FINAL_SIZE),
        precommitteds: vec![pre_a_frozen, pre_b_frozen],
    };
    assert!(multi_group_key
        .fits_setup_capacity(FINAL_NV, SETUP_CAPACITY_SIZE)
        .expect("setup capacity"));

    let multi_group_schedule =
        OneHotCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let main_params = multi_group_root_params(&multi_group_schedule);
    let final_polys = [
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7001),
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7002),
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7003),
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7004),
    ];
    let (final_commitment, final_hint, _selection) = OneHotScheme::commit_final_group(
        &setup,
        &final_polys,
        &stack,
        vec![pre_a_commitment.profile, pre_b_commitment.profile],
    )
    .expect("final multi-group commitment");

    assert_eq!(
        pre_a_commitment.rows().count(),
        pre_a_frozen.outer_commit_matrix.output_rank()
    );
    assert_eq!(
        pre_b_commitment.rows().count(),
        pre_b_frozen.outer_commit_matrix.output_rank()
    );
    assert_eq!(
        final_commitment.rows().count(),
        main_params.outer_commit_matrix.output_rank()
    );
    assert_eq!(final_hint.inner_rows().len(), FINAL_SIZE);
    assert_eq!(
        akita_prover::RootPolyMeta::num_vars(&final_polys[0]),
        FINAL_NV,
        "final one-hot group should retain its native variable domain"
    );
    assert_eq!(
        multi_group_schedule.root.params.precommitted_groups.len(),
        2
    );
    assert_eq!(
        multi_group_schedule.root.params.precommitted_groups[0].descriptor,
        pre_a_frozen
    );
    assert_eq!(
        multi_group_schedule.root.params.precommitted_groups[1].descriptor,
        pre_b_frozen
    );
}

#[test]
fn commit_group_returns_frozen_exact_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let profile =
        akita_config::committed_group_profile::<OneHotCfg>(&key).expect("group commit profile");
    let total_field = (profile.num_live_blocks * profile.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_0001)];

    let setup = OneHotScheme::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let (commitment, _hint) =
        OneHotScheme::commit_group(&setup, &polys, &stack).expect("commit group");
    let frozen_layout = commitment.profile;

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.num_positions_per_block,
        profile.num_positions_per_block
    );
    assert_eq!(frozen_layout.num_live_blocks, profile.num_live_blocks);
    assert_eq!(frozen_layout.log_basis_outer, profile.log_basis_outer);
    assert_eq!(
        frozen_layout.inner_commit_matrix.output_rank(),
        profile.inner_commit_matrix.output_rank()
    );
    assert_eq!(
        frozen_layout.outer_commit_matrix.output_rank(),
        profile.outer_commit_matrix.output_rank()
    );
    assert_eq!(
        commitment.rows().count(),
        frozen_layout.outer_commit_matrix.output_rank()
    );
}
