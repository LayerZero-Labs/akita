use super::*;

type PrecommitCommitter = PrecommittedOneHotScheme;

#[test]
fn precommit_config_commit_returns_exact_frozen_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let opening_batch = OpeningClaimsLayout::new(NV, GROUP_SIZE).expect("opening batch");
    let layout = PrecommittedOneHotCfg::get_params_for_batched_commitment(&opening_batch)
        .expect("precommit layout");
    let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = PrecommitCommitter::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let (commitment, _hint) =
        PrecommitCommitter::commit(&setup, &polys, &stack).expect("precommit");
    let frozen_layout = akita_types::CommittedGroupProfile::from_params(key, &layout);

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.num_positions_per_block,
        layout.num_positions_per_block
    );
    assert_eq!(frozen_layout.num_live_blocks, layout.num_live_blocks);
    assert_eq!(frozen_layout.log_basis_outer, OneHotCfg::basis_range().0);
    assert_eq!(
        frozen_layout.inner_commit_matrix.output_rank(),
        layout.inner_commit_matrix.output_rank()
    );
    assert_eq!(
        frozen_layout.outer_commit_matrix.output_rank(),
        layout.outer_commit_matrix.output_rank()
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
    let setup = PrecommitCommitter::setup_prover(max_num_vars, max_num_polys).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    run(&setup, &stack)
}

#[test]
fn precommit_config_allows_independent_precommitted_groups() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;
    // Precommitted groups are committed independently, so setup only needs to
    // cover the largest standalone group rather than the sum of all groups.
    const SETUP_CAPACITY_SIZE: usize = PRE_B_SIZE;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(NV, PRE_B_SIZE);
    let pre_a_opening_batch = OpeningClaimsLayout::new(NV, PRE_A_SIZE).expect("precommit A batch");
    let pre_b_opening_batch = OpeningClaimsLayout::new(NV, PRE_B_SIZE).expect("precommit B batch");
    let pre_a_layout =
        PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_a_opening_batch)
            .expect("precommit A layout");
    let pre_b_layout =
        PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_b_opening_batch)
            .expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_1001)];
    let pre_b_polys = [
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_2001),
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_2002),
    ];

    with_precommit_stack(NV, SETUP_CAPACITY_SIZE, |setup, stack| {
        let (pre_a_commitment, _pre_a_hint) =
            PrecommitCommitter::commit(setup, &pre_a_polys, stack).expect("precommit A");
        let (pre_b_commitment, _pre_b_hint) =
            PrecommitCommitter::commit(setup, &pre_b_polys, stack).expect("precommit B");
        let pre_a_frozen =
            akita_types::CommittedGroupProfile::from_params(pre_a_key, &pre_a_layout);
        let pre_b_frozen =
            akita_types::CommittedGroupProfile::from_params(pre_b_key, &pre_b_layout);

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
    let pre_a_opening_batch =
        OpeningClaimsLayout::new(PRE_NV, PRE_A_SIZE).expect("precommit A batch");
    let pre_b_opening_batch =
        OpeningClaimsLayout::new(PRE_NV, PRE_B_SIZE).expect("precommit B batch");
    let pre_c_opening_batch =
        OpeningClaimsLayout::new(PRE_NV, PRE_C_SIZE).expect("precommit C batch");
    let pre_a_layout =
        PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_a_opening_batch)
            .expect("precommit A layout");
    let pre_b_layout =
        PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_b_opening_batch)
            .expect("precommit B layout");
    let pre_c_layout =
        PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_c_opening_batch)
            .expect("precommit C layout");
    let pre_a_frozen = akita_types::CommittedGroupProfile::from_params(pre_a_key, &pre_a_layout);
    let pre_b_frozen = akita_types::CommittedGroupProfile::from_params(pre_b_key, &pre_b_layout);
    let pre_c_frozen = akita_types::CommittedGroupProfile::from_params(pre_c_key, &pre_c_layout);
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
    let pre_opening_batch = OpeningClaimsLayout::new(PRE_NV, GROUP_SIZE).expect("precommit batch");
    let pre_a_layout = PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit A layout");
    let pre_b_layout = PrecommittedOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_5001)];
    let pre_b_polys = [debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_6001)];

    let setup = OneHotScheme::setup_prover(FINAL_NV, SETUP_CAPACITY_SIZE).expect("protocol setup");
    let prepared = CpuBackend
        .prepare_setup(&setup)
        .expect("prepared protocol setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("protocol stack");
    let (pre_a_commitment, _pre_a_hint) =
        PrecommitCommitter::commit::<_, _>(&setup, &pre_a_polys, &stack).expect("precommit A");
    let (pre_b_commitment, _pre_b_hint) =
        PrecommitCommitter::commit::<_, _>(&setup, &pre_b_polys, &stack).expect("precommit B");
    let pre_a_frozen = akita_types::CommittedGroupProfile::from_params(pre_a_key, &pre_a_layout);
    let pre_b_frozen = akita_types::CommittedGroupProfile::from_params(pre_b_key, &pre_b_layout);
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
        debug_make_onehot_poly(main_params, 0x0bee_fcaf_9a77_7001),
        debug_make_onehot_poly(main_params, 0x0bee_fcaf_9a77_7002),
        debug_make_onehot_poly(main_params, 0x0bee_fcaf_9a77_7003),
        debug_make_onehot_poly(main_params, 0x0bee_fcaf_9a77_7004),
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
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(NV, GROUP_SIZE).expect("opening batch");
    // `commit_group` freezes the standalone precommit layout (root basis pinned),
    // so size the setup and expected layout with the precommit config, not the
    // main runtime config (which resolves a different, single-group root split).
    let layout = PrecommittedOneHotCfg::get_params_for_batched_commitment(&opening_batch)
        .expect("group commit layout");
    let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = PrecommitCommitter::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let (commitment, _hint) =
        OneHotScheme::commit_group(&setup, &polys, &stack).expect("commit group");
    let frozen_layout = commitment.profile;

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.num_positions_per_block,
        layout.num_positions_per_block
    );
    assert_eq!(frozen_layout.num_live_blocks, layout.num_live_blocks);
    assert_eq!(frozen_layout.log_basis_outer, layout.log_basis_outer);
    assert_eq!(
        frozen_layout.inner_commit_matrix.output_rank(),
        layout.inner_commit_matrix.output_rank()
    );
    assert_eq!(
        frozen_layout.outer_commit_matrix.output_rank(),
        layout.outer_commit_matrix.output_rank()
    );
    assert_eq!(
        commitment.rows().count(),
        frozen_layout.outer_commit_matrix.output_rank()
    );
}

/// Produce and verify a folded multi-group-root one-hot same-point proof for the
/// given precommitted group sizes plus a final group size, exercising unequal
/// `K_g`. Precommitted groups use the exact fixed-root precommit config; the
/// final group is committed with `commit_final_group`; the multi-group root folds
/// into a singleton recursive suffix.
fn multi_group_root_round_trip_onehot<TestCfg, ProtocolCfg>(
    pre_num_vars: usize,
    final_num_vars: usize,
    pre_sizes: &[usize],
    final_size: usize,
    check_group_binding: bool,
) where
    TestCfg: CommitmentConfig<Field = OneHotF, ExtField = OneHotF>,
    ProtocolCfg: CommitmentConfig<Field = OneHotF, ExtField = OneHotF>,
{
    let total: usize = pre_sizes.iter().sum::<usize>() + final_size;
    let opening_num_vars = pre_num_vars.max(final_num_vars);

    let setup =
        AkitaCommitmentScheme::<ProtocolCfg>::setup_prover(opening_num_vars, total).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    // Commit every precommitted group under the exact precommit config; keep the
    // polynomials alive so the prover/verifier can borrow references.
    let mut pre_keys = Vec::new();
    let mut pre_frozen = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_layouts = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<OneHotPoly<OneHotF, u8>>> = Vec::new();
    for (group_idx, &num_polynomials) in pre_sizes.iter().enumerate() {
        let key = akita_types::PolynomialGroupLayout::new(pre_num_vars, num_polynomials);
        let layout =
            akita_config::committed_group_params::<ProtocolCfg>(&key).expect("precommit params");
        let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..num_polynomials)
            .map(|poly_idx| {
                debug_make_onehot_poly(
                    &layout,
                    0x0bee_fcaf_1a00_0000 + ((group_idx as u64) << 8) + poly_idx as u64,
                )
            })
            .collect();
        let (commitment, hint) =
            AkitaCommitmentScheme::<ProtocolCfg>::commit_group(&setup, &polys, &stack)
                .expect("precommit");
        pre_frozen.push(commitment.profile);
        pre_keys.push(key);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_layouts.push(layout);
        pre_polys_by_group.push(polys);
    }

    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(final_num_vars, final_size),
        precommitteds: pre_frozen,
    };
    let opening_layout = multi_group_key
        .opening_layout()
        .expect("multi-group opening layout");
    let multi_group_schedule =
        ProtocolCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let main_params = multi_group_root_params(&multi_group_schedule);
    assert!(
        multi_group_schedule
            .root
            .params
            .precommitted_groups
            .iter()
            .zip(&pre_commitments)
            .all(|(group, commitment)| group.descriptor == commitment.profile),
        "precommitted groups must retain their exact native profiles"
    );
    if TestCfg::chunked_witness_cfg().uses_multi_chunk() {
        let root = &multi_group_schedule.root;
        let root_commitment = &root.params.final_group.commitment;
        assert!(!root.params.precommitted_groups.is_empty());
        assert_eq!(
            root.params.witness_partition.num_chunks(),
            TestCfg::chunked_witness_cfg().num_chunks,
            "root fold must retain the configured chunk count"
        );
        let witness_layout = akita_types::WitnessLayout::new(
            root_commitment,
            &opening_layout,
            root.params.witness_partition.num_chunks(),
            akita_types::r_decomp_levels::<OneHotF>(root_commitment.log_basis_open),
        )
        .expect("group-by-chunk witness layout");
        assert_eq!(
            witness_layout.units().len(),
            opening_layout.num_groups() * root.params.witness_partition.num_chunks(),
        );
    }
    let final_polys: Vec<OneHotPoly<OneHotF, u8>> = (0..final_size)
        .map(|poly_idx| {
            debug_make_onehot_poly(main_params, 0x0bee_fcaf_f100_0000 + poly_idx as u64)
        })
        .collect();
    let (final_commitment, final_hint, _selection) =
        AkitaCommitmentScheme::<ProtocolCfg>::commit_final_group(
            &setup,
            &final_polys,
            &stack,
            pre_commitments.iter().map(|group| group.profile).collect(),
        )
        .expect("final multi-group commitment");

    let mut pre_point = debug_random_point(pre_num_vars);
    pre_point[0] += OneHotF::one();
    let final_point = debug_random_point(final_num_vars);
    let pre_openings: Vec<Vec<OneHotF>> = pre_polys_by_group
        .iter()
        .zip(pre_layouts.iter())
        .map(|(polys, layout)| {
            polys
                .iter()
                .map(|poly| opening_from_poly(poly, &pre_point, layout))
                .collect()
        })
        .collect();
    let final_openings: Vec<OneHotF> = final_polys
        .iter()
        .map(|poly| opening_from_poly(poly, &final_point, main_params))
        .collect();

    let pre_refs_by_group: Vec<Vec<&OneHotPoly<OneHotF, u8>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&OneHotPoly<OneHotF, u8>> = final_polys.iter().collect();

    let mut prover_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        prover_groups.push(
            PolynomialGroupClaims::new(
                pre_point.clone(),
                openings.clone(),
                pre_commitments[group_idx].clone(),
            )
            .expect("pre prover group"),
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            final_point.clone(),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    );

    let mut prover_polys: Vec<&[&OneHotPoly<OneHotF, u8>]> = Vec::new();
    for refs in &pre_refs_by_group {
        prover_polys.push(&refs[..]);
    }
    prover_polys.push(&final_refs[..]);
    let mut prover_hints = pre_hints;
    prover_hints.push(final_hint);

    let prover_claims = selected_prover_data::<ProtocolCfg, _>(
        OpeningClaims::from_groups(prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
    )
    .expect("multi-group prover data");
    let selection = prover_claims.0;

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    let proof = AkitaCommitmentScheme::<ProtocolCfg>::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("multi-group prove");
    assert!(proof.num_fold_levels() >= 2);
    let planned_stage3 = multi_group_schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.incoming_setup_prefix.is_some())
        .count();
    let proved_stage3 = proof
        .nonterminal_folds()
        .filter(|fold| fold.stage3_sumcheck_proof().is_some())
        .count();
    assert_eq!(
        proved_stage3, planned_stage3,
        "proof stage-3 payloads must follow the config-selected schedule"
    );

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize multi-group proof");
    let decoded = akita_types::AkitaBatchedProof::<OneHotF, OneHotF>::deserialize_uncompressed(
        &bytes[..],
        &shape,
    )
    .expect("deserialize multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup =
        AkitaCommitmentScheme::<ProtocolCfg>::setup_verifier(&setup).expect("verifier setup");
    let mut verifier_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        verifier_groups.push(
            PolynomialGroupClaims::new(
                pre_point.clone(),
                openings.clone(),
                &pre_commitments[group_idx],
            )
            .expect("pre verifier group"),
        );
    }
    verifier_groups.push(
        PolynomialGroupClaims::new(
            final_point.clone(),
            final_openings.clone(),
            &final_commitment,
        )
        .expect("final verifier group"),
    );
    let verify_claims =
        OpeningClaims::from_groups(verifier_groups).expect("multi-group verifier claims");
    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    AkitaCommitmentScheme::<ProtocolCfg>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        GroupBatchStatement::new(selection, verify_claims).expect("multi-group statement"),
        BasisMode::Lagrange,
    )
    .expect("multi-group verify");

    if check_group_binding {
        assert_eq!(pre_commitments.len(), 1, "binding fixture uses two groups");
        let swapped_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                pre_point.clone(),
                pre_openings[0].clone(),
                &final_commitment,
            )
            .expect("swapped pre verifier group"),
            PolynomialGroupClaims::new(
                final_point.clone(),
                final_openings.clone(),
                &pre_commitments[0],
            )
            .expect("swapped final verifier group"),
        ])
        .expect("swapped verifier claims");
        let mut swapped_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
        assert!(
            AkitaCommitmentScheme::<ProtocolCfg>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut swapped_transcript,
                GroupBatchStatement::new(selection, swapped_claims)
                    .expect("swapped-group statement"),
                BasisMode::Lagrange,
            )
            .is_err(),
            "swapped group commitments must reject"
        );

        let mut tampered_final_openings = final_openings.clone();
        tampered_final_openings[0] += OneHotF::one();
        let tampered_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(pre_point, pre_openings[0].clone(), &pre_commitments[0])
                .expect("pre verifier group"),
            PolynomialGroupClaims::new(final_point, tampered_final_openings, &final_commitment)
                .expect("tampered final verifier group"),
        ])
        .expect("tampered verifier claims");
        let mut tampered_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
        assert!(
            AkitaCommitmentScheme::<ProtocolCfg>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut tampered_transcript,
                GroupBatchStatement::new(selection, tampered_claims)
                    .expect("tampered-opening statement"),
                BasisMode::Lagrange,
            )
            .is_err(),
            "tampered group opening must reject"
        );
    }
}

#[test]
fn multi_group_root_folded_group_binding_round_trips() {
    multi_group_root_round_trip_onehot::<OneHotCfg, OneHotCfg>(14, 20, &[1], 2, true);
}

#[test]
fn multi_group_root_allows_precommitted_arity_above_final_group() {
    type PlannerCfg = crate::test_support::EnvelopeFinalGroupConfig<OneHotCfg, OneHotCfg>;

    multi_group_root_round_trip_onehot::<OneHotCfg, PlannerCfg>(20, 14, &[1], 1, false);
}

#[test]
fn multi_group_root_opens_multi_polynomial_precommitted_group() {
    multi_group_root_round_trip_onehot::<OneHotCfg, OneHotCfg>(14, 20, &[2], 1, false);
}

#[test]
fn multi_group_root_allows_final_a_smaller_than_precommitted_a() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 24;
    type ProtocolCfg =
        crate::test_support::EnvelopeFinalGroupConfig<fp128::D256OneHot, fp128::D64OneHot>;

    let pre_group = akita_types::PolynomialGroupLayout::new(PRE_NV, 1);
    let precommitted = akita_planner::derive_standalone_precommit_profile(
        pre_group,
        &akita_config::policy_of::<fp128::D256OneHot>(),
        fp128::D256OneHot::root_honest_fold_policy(),
        fp128::D256OneHot::ring_challenge_config,
    )
    .expect("synthetic D256 precommit profile");
    let key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, 2),
        precommitteds: vec![precommitted],
    };
    let opening_layout = key.opening_layout().unwrap();
    let schedule = ProtocolCfg::runtime_schedule(key).expect("descending-A schedule");
    let root = multi_group_root_params(&schedule);
    assert_eq!(root.d_a(), 64);
    assert_eq!(root.group_role_dims(&opening_layout, 0).unwrap().d_a(), 256);
}

#[test]
#[cfg(feature = "profile-ci")]
fn multi_group_multi_chunk_fold_round_trips() {
    multi_group_root_round_trip_onehot::<fp128::OneHotMultiChunkW2R2, fp128::OneHotMultiChunkW2R2>(
        14,
        14,
        &[1],
        1,
        false,
    );
}

#[test]
fn batched_onehot_roundtrip_matches_public_shape_context() {
    // NV chosen large enough that the runtime schedule yields at least two
    // fold steps so the proof is fold-rooted (not terminal-rooted). Under
    // the post-soundness-fix proof shape, a single-fold schedule emits a
    // `Terminal` root with no recursive suffix, which this test does not
    // exercise.
    const NV: usize = 20;
    const BATCH_SIZE: usize = 2;

    let layout = akita_batched_root_layout::<OneHotCfg>(NV, BATCH_SIZE).expect("layout");
    let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    let total_chunks = total_field / BENCH_ONEHOT_K;
    assert_eq!(total_chunks * BENCH_ONEHOT_K, total_field);

    let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..BATCH_SIZE)
        .map(|poly_idx| debug_make_onehot_poly(&layout, 0x0bee_fcaf_e000_1500 + poly_idx as u64))
        .collect();
    let poly_refs: Vec<&OneHotPoly<OneHotF, u8>> = polys.iter().collect();
    let point = debug_random_point(NV);
    let openings: Vec<OneHotF> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &point, &layout))
        .collect();

    let setup = OneHotScheme::setup_prover(NV, BATCH_SIZE).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = OneHotScheme::setup_verifier(&setup).expect("verifier setup");
    let (commitment, hint) =
        OneHotScheme::commit::<_, _>(&setup, &polys, &stack).expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let prover_group = PolynomialGroupClaims::new(
        point.clone(),
        vec![OneHotF::zero(); poly_refs.len()],
        commitments[0].clone(),
    )
    .expect("valid one-hot prover group");
    let proof = OneHotScheme::batched_prove::<_, _, _>(
        &setup,
        selected_prover_data::<OneHotCfg, _>(
            OpeningClaims::from_groups(vec![prover_group]).expect("valid one-hot prover claims"),
            hints,
            vec![&poly_refs[..]],
        )
        .expect("valid one-hot prover opening data"),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched onehot prove");

    let expected_shape = expected_same_point_batched_shape(NV, BATCH_SIZE, &proof);
    let actual_shape = proof.shape();
    assert_eq!(
        expected_shape.root.opening_payload_coeffs,
        actual_shape.root.opening_payload_coeffs
    );
    assert_eq!(
        expected_shape.root.stage1_stages,
        actual_shape.root.stage1_stages
    );
    assert_eq!(
        expected_shape.root.stage2_sumcheck_proof,
        actual_shape.root.stage2_sumcheck_proof
    );
    assert_eq!(
        expected_shape.root.next_witness_binding,
        actual_shape.root.next_witness_binding
    );
    assert_eq!(expected_shape.recursive_folds, actual_shape.recursive_folds);
    assert_eq!(
        expected_shape.terminal.extension_opening_reduction,
        actual_shape.terminal.extension_opening_reduction
    );
    assert!(
        expected_shape
            .terminal
            .terminal_response
            .admits_realized(&actual_shape.terminal.terminal_response),
        "terminal witness shape {:?} does not admit {:?}",
        expected_shape.terminal.terminal_response,
        actual_shape.terminal.terminal_response
    );
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let decoded =
        AkitaBatchedProof::<OneHotF, OneHotF>::deserialize_uncompressed(&*bytes, &actual_shape)
            .expect("deserialize batched proof with derived shape");
    assert_eq!(decoded, proof);

    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    OneHotScheme::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        selected_statement::<OneHotCfg>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point,
                openings,
                &commitments[0],
            )
            .expect("valid one-hot verifier group")])
            .expect("valid one-hot verifier claims"),
        )
        .expect("valid one-hot verifier statement"),
        BasisMode::Lagrange,
    )
    .expect("batched onehot verify");
}
