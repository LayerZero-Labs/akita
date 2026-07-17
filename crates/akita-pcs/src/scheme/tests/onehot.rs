use super::*;

type ConservativeCommitter = ConservativeOneHotScheme;
type RegularCommitter = OneHotScheme;

#[test]
fn conservative_config_commit_returns_frozen_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let opening_batch = OpeningClaimsLayout::new(NV, GROUP_SIZE).expect("opening batch");
    let layout = ConservativeOneHotCfg::get_params_for_batched_commitment(&opening_batch)
        .expect("conservative commit layout");
    let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = ConservativeCommitter::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let (commitment, _hint) =
        ConservativeCommitter::commit(&setup, &polys, &stack).expect("conservative commit");
    let frozen_layout = akita_types::PrecommittedGroupParams::from_params(key, &layout);

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.num_positions_per_block,
        layout.num_positions_per_block
    );
    assert_eq!(frozen_layout.num_live_blocks, layout.num_live_blocks);
    assert_eq!(
        frozen_layout.log_basis_outer,
        ConservativeOneHotCfg::basis_range().0
    );
    assert_eq!(frozen_layout.n_a, layout.a_key.row_len());
    assert_eq!(frozen_layout.n_b, layout.b_key.row_len());
    assert_eq!(commitment.rows().count(), frozen_layout.n_b);
}

fn multi_group_root_params(schedule: &akita_types::Schedule) -> &LevelParams {
    match schedule.steps.first().expect("multi-group schedule step") {
        Step::Direct(direct) => direct.params.as_ref().expect("multi-group root params"),
        Step::Fold(fold) => &fold.params,
    }
}

fn with_conservative_commit_stack<R>(
    max_num_vars: usize,
    max_num_polys: usize,
    run: impl FnOnce(
        &akita_prover::AkitaProverSetup<OneHotF>,
        &akita_prover::UniformProverStack<'_, OneHotF, CpuBackend>,
    ) -> R,
) -> R {
    let setup = ConservativeCommitter::setup_prover(max_num_vars, max_num_polys).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    run(&setup, &stack)
}

#[test]
fn conservative_config_allows_independent_precommitted_groups() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(NV, PRE_B_SIZE);
    let pre_a_opening_batch = OpeningClaimsLayout::new(NV, PRE_A_SIZE).expect("precommit A batch");
    let pre_b_opening_batch = OpeningClaimsLayout::new(NV, PRE_B_SIZE).expect("precommit B batch");
    let pre_a_layout =
        ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_a_opening_batch)
            .expect("precommit A layout");
    let pre_b_layout =
        ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_b_opening_batch)
            .expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_1001)];
    let pre_b_polys = [
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_2001),
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_2002),
    ];

    with_conservative_commit_stack(NV, PRE_A_SIZE + PRE_B_SIZE, |setup, stack| {
        let (pre_a_commitment, _pre_a_hint) =
            ConservativeCommitter::commit(setup, &pre_a_polys, stack).expect("precommit A");
        let (pre_b_commitment, _pre_b_hint) =
            ConservativeCommitter::commit(setup, &pre_b_polys, stack).expect("precommit B");
        let pre_a_frozen =
            akita_types::PrecommittedGroupParams::from_params(pre_a_key, &pre_a_layout);
        let pre_b_frozen =
            akita_types::PrecommittedGroupParams::from_params(pre_b_key, &pre_b_layout);

        assert_eq!(pre_a_frozen.group, pre_a_key);
        assert_eq!(pre_b_frozen.group, pre_b_key);
        assert_eq!(pre_a_commitment.rows().count(), pre_a_frozen.n_b);
        assert_eq!(pre_b_commitment.rows().count(), pre_b_frozen.n_b);
        assert_ne!(pre_a_frozen.group, pre_b_frozen.group);
    });
}

#[test]
fn group_batch_schedule_preserves_precommitted_order() {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = PRE_NV * 2;
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
        ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_a_opening_batch)
            .expect("precommit A layout");
    let pre_b_layout =
        ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_b_opening_batch)
            .expect("precommit B layout");
    let pre_c_layout =
        ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_c_opening_batch)
            .expect("precommit C layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_3001)];
    let pre_b_polys = [debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_4001)];
    let pre_c_polys = [debug_make_onehot_poly(&pre_c_layout, 0x0bee_fcaf_9a77_4002)];

    with_conservative_commit_stack(
        FINAL_NV,
        PRE_A_SIZE + PRE_B_SIZE + PRE_C_SIZE + MAIN_SIZE,
        |setup, stack| {
            ConservativeCommitter::commit(setup, &pre_a_polys, stack).expect("precommit A");
            ConservativeCommitter::commit(setup, &pre_b_polys, stack).expect("precommit B");
            ConservativeCommitter::commit(setup, &pre_c_polys, stack).expect("precommit C");
            let pre_a_frozen =
                akita_types::PrecommittedGroupParams::from_params(pre_a_key, &pre_a_layout);
            let pre_b_frozen =
                akita_types::PrecommittedGroupParams::from_params(pre_b_key, &pre_b_layout);
            let pre_c_frozen =
                akita_types::PrecommittedGroupParams::from_params(pre_c_key, &pre_c_layout);
            let multi_group_key = akita_types::AkitaScheduleLookupKey {
                final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, MAIN_SIZE),
                precommitteds: vec![pre_a_frozen, pre_b_frozen, pre_c_frozen],
            };

            let schedule = OneHotCfg::runtime_schedule(multi_group_key.clone())
                .expect("multi-group runtime schedule");
            let root = multi_group_root_params(&schedule);
            let main_params = akita_types::multi_group_root_commit_params(&schedule)
                .expect("main multi-group commit params");

            assert_eq!(multi_group_key.num_commitment_groups(), 4);
            assert_eq!(
                multi_group_key
                    .num_polynomials()
                    .expect("multi-group polynomial count"),
                PRE_A_SIZE + PRE_B_SIZE + PRE_C_SIZE + MAIN_SIZE
            );
            assert_eq!(main_params, *root);
            assert_eq!(root.precommitted_groups.len(), 3);
            assert_eq!(root.precommitted_groups[0].layout, pre_a_frozen);
            assert_eq!(root.precommitted_groups[1].layout, pre_b_frozen);
            assert_eq!(root.precommitted_groups[2].layout, pre_c_frozen);
        },
    );
}

#[test]
fn group_batch_commits_precommitteds_then_double_size_final_group() {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = PRE_NV * 2;
    const GROUP_SIZE: usize = 1;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_opening_batch = OpeningClaimsLayout::new(PRE_NV, GROUP_SIZE).expect("precommit batch");
    let pre_a_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit A layout");
    let pre_b_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_5001)];
    let pre_b_polys = [debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_6001)];

    with_conservative_commit_stack(FINAL_NV, GROUP_SIZE, |setup, stack| {
        let (pre_a_commitment, _pre_a_hint) =
            ConservativeCommitter::commit::<_, _>(setup, &pre_a_polys, stack).expect("precommit A");
        let (pre_b_commitment, _pre_b_hint) =
            ConservativeCommitter::commit::<_, _>(setup, &pre_b_polys, stack).expect("precommit B");
        let pre_a_frozen =
            akita_types::PrecommittedGroupParams::from_params(pre_a_key, &pre_a_layout);
        let pre_b_frozen =
            akita_types::PrecommittedGroupParams::from_params(pre_b_key, &pre_b_layout);
        let multi_group_key = akita_types::AkitaScheduleLookupKey {
            final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, GROUP_SIZE),
            precommitteds: vec![pre_a_frozen, pre_b_frozen],
        };

        let multi_group_schedule =
            OneHotCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
        let main_params = multi_group_root_params(&multi_group_schedule);
        let final_polys = [debug_make_onehot_poly(main_params, 0x0bee_fcaf_9a77_7001)];
        let (final_commitment, final_hint) = OneHotScheme::commit_final_group::<_, _>(
            setup,
            &final_polys,
            stack,
            vec![pre_a_key, pre_b_key],
        )
        .expect("final multi-group commitment");

        assert_eq!(pre_a_commitment.rows().count(), pre_a_frozen.n_b);
        assert_eq!(pre_b_commitment.rows().count(), pre_b_frozen.n_b);
        assert_eq!(final_commitment.rows().count(), main_params.b_key.row_len());
        assert_eq!(final_hint.decomposed_inner_rows.len(), GROUP_SIZE);
        assert_eq!(
            akita_prover::RootPolyMeta::num_vars(&final_polys[0]),
            FINAL_NV,
            "final one-hot group should live on the doubled variable domain"
        );
        assert_eq!(main_params.precommitted_groups.len(), 2);
        assert_eq!(main_params.precommitted_groups[0].layout, pre_a_frozen);
        assert_eq!(main_params.precommitted_groups[1].layout, pre_b_frozen);
    });
}

#[test]
fn commit_group_returns_frozen_conservative_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let opening_batch =
        akita_types::OpeningClaimsLayout::new(NV, GROUP_SIZE).expect("opening batch");
    let layout =
        OneHotCfg::get_params_for_batched_commitment(&opening_batch).expect("group commit layout");
    let total_field = (layout.num_live_blocks * layout.num_positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = OneHotScheme::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let (frozen_layout, commitment, _hint) =
        OneHotScheme::commit_group::<_, _>(&setup, &polys, &stack).expect("commit group");

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.num_positions_per_block,
        layout.num_positions_per_block
    );
    assert_eq!(frozen_layout.num_live_blocks, layout.num_live_blocks);
    assert_eq!(frozen_layout.log_basis_outer, layout.log_basis_outer);
    assert_eq!(frozen_layout.n_a, layout.a_key.row_len());
    assert_eq!(frozen_layout.n_b, layout.b_key.row_len());
    assert_eq!(commitment.rows().count(), frozen_layout.n_b);
}

/// Produce and verify a folded multi-group-root one-hot same-point proof for the
/// given precommitted group sizes plus a final group size, exercising unequal
/// `K_g`. Precommitted groups are committed under the conservative config; the
/// final group is committed with `commit_final_group`; the multi-group root folds
/// into a singleton recursive suffix.
fn multi_group_root_round_trip_onehot<TestCfg>(
    pre_sizes: &[usize],
    final_size: usize,
    setup_contribution_mode: akita_types::SetupContributionMode,
) where
    TestCfg: CommitmentConfig<Field = OneHotF, ExtField = OneHotF>,
{
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = PRE_NV * 2;
    let total: usize = pre_sizes.iter().sum::<usize>() + final_size;

    let setup = match setup_contribution_mode {
        akita_types::SetupContributionMode::Direct => AkitaCommitmentScheme::<
            ConservativeCommitmentConfig<TestCfg>,
        >::setup_prover(FINAL_NV, total),
        akita_types::SetupContributionMode::Recursive => crate::AkitaCommitmentScheme::<
            akita_config::RecursiveCommitmentConfig<ConservativeCommitmentConfig<TestCfg>>,
        >::setup_prover(FINAL_NV, total),
    }
    .expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    // Commit every precommitted group under the conservative config; keep the
    // polynomials alive so the prover/verifier can borrow references.
    let mut pre_keys = Vec::new();
    let mut pre_frozen = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_layouts = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<OneHotPoly<OneHotF, u8>>> = Vec::new();
    for (group_idx, &k) in pre_sizes.iter().enumerate() {
        let key = akita_types::PolynomialGroupLayout::new(PRE_NV, k);
        let opening_batch = OpeningClaimsLayout::new(PRE_NV, k).expect("precommit batch");
        let layout = ConservativeCommitmentConfig::<TestCfg>::get_params_for_batched_commitment(
            &opening_batch,
        )
        .expect("precommit layout");
        let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..k)
            .map(|poly_idx| {
                debug_make_onehot_poly(
                    &layout,
                    0x0bee_fcaf_1a00_0000 + ((group_idx as u64) << 8) + poly_idx as u64,
                )
            })
            .collect();
        let (commitment, hint) =
            AkitaCommitmentScheme::<ConservativeCommitmentConfig<TestCfg>>::batched_commit(
                &setup,
                &polys[..],
                &stack,
            )
            .expect("precommit");
        pre_frozen.push(akita_types::PrecommittedGroupParams::from_params(
            key, &layout,
        ));
        pre_keys.push(key);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_layouts.push(layout);
        pre_polys_by_group.push(polys);
    }

    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, final_size),
        precommitteds: pre_frozen,
    };
    let opening_layout = multi_group_key
        .opening_layout()
        .expect("multi-group opening layout");
    let multi_group_schedule =
        TestCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let main_params = multi_group_root_params(&multi_group_schedule);
    if TestCfg::chunked_witness_cfg().uses_multi_chunk() {
        let Step::Fold(root) = multi_group_schedule
            .steps
            .first()
            .expect("chunked multi-group root step")
        else {
            panic!("chunked multi-group regression must force a fold");
        };
        assert!(root.params.has_precommitted_groups());
        assert_eq!(
            root.params.witness_chunk,
            TestCfg::chunked_witness_cfg(),
            "root fold must retain the configured chunk count"
        );
        let relation_rows = root
            .params
            .relation_matrix_row_count_for(
                opening_layout.num_groups(),
                RelationMatrixRowLayout::WithDBlock,
            )
            .expect("grouped relation rows");
        let witness_layout = akita_types::WitnessLayout::new(
            &root.params,
            &opening_layout,
            root.params.witness_chunk.num_chunks,
            relation_rows,
            akita_types::r_decomp_levels::<OneHotF>(root.params.log_basis_open),
        )
        .expect("group-by-chunk witness layout");
        assert_eq!(
            witness_layout.units().len(),
            opening_layout.num_groups() * root.params.witness_chunk.num_chunks,
        );
    }
    let final_polys: Vec<OneHotPoly<OneHotF, u8>> = (0..final_size)
        .map(|poly_idx| {
            debug_make_onehot_poly(main_params, 0x0bee_fcaf_f100_0000 + poly_idx as u64)
        })
        .collect();
    let (final_commitment, final_hint) = AkitaCommitmentScheme::<TestCfg>::commit_final_group(
        &setup,
        &final_polys,
        &stack,
        pre_keys,
    )
    .expect("final multi-group commitment");

    let point = debug_random_point(FINAL_NV);
    let pre_openings: Vec<Vec<OneHotF>> = pre_polys_by_group
        .iter()
        .zip(pre_layouts.iter())
        .map(|(polys, layout)| {
            polys
                .iter()
                .map(|poly| opening_from_poly(poly, &point[..PRE_NV], layout))
                .collect()
        })
        .collect();
    let final_openings: Vec<OneHotF> = final_polys
        .iter()
        .map(|poly| opening_from_poly(poly, &point, main_params))
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
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                openings.clone(),
                pre_commitments[group_idx].clone(),
            )
            .expect("pre prover group"),
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
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

    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
    )
    .expect("multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    let proof = AkitaCommitmentScheme::<TestCfg>::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        setup_contribution_mode,
    )
    .expect("multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));
    if matches!(proof.root, akita_types::AkitaBatchedRootProof::Fold(_)) {
        assert!(
            !proof.steps.is_empty(),
            "intermediate multi-group root must hand off to a suffix"
        );
    }

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

    let verifier_setup = AkitaCommitmentScheme::<TestCfg>::setup_verifier(&setup);
    let mut verifier_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        verifier_groups.push(
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                openings.clone(),
                &pre_commitments[group_idx],
            )
            .expect("pre verifier group"),
        );
    }
    verifier_groups.push(
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            final_openings.clone(),
            &final_commitment,
        )
        .expect("final verifier group"),
    );
    let verify_claims = OpeningClaims::from_groups(point.clone(), verifier_groups)
        .expect("multi-group verifier claims");
    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    AkitaCommitmentScheme::<TestCfg>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_claims,
        BasisMode::Lagrange,
        setup_contribution_mode,
    )
    .expect("multi-group verify");
}

#[test]
fn multi_group_root_folded_unequal_one_three_round_trips() {
    multi_group_root_round_trip_onehot::<OneHotCfg>(
        &[1],
        3,
        akita_types::SetupContributionMode::Direct,
    );
}

#[test]
fn multi_group_root_folded_unequal_two_one_round_trips() {
    multi_group_root_round_trip_onehot::<OneHotCfg>(
        &[1, 1],
        1,
        akita_types::SetupContributionMode::Direct,
    );
}

#[test]
fn multi_group_root_folded_recursive_setup_round_trips() {
    multi_group_root_round_trip_onehot::<OneHotCfg>(
        &[1],
        3,
        akita_types::SetupContributionMode::Recursive,
    );
}

#[test]
fn multi_group_multi_chunk_fold_round_trips() {
    multi_group_root_round_trip_onehot::<fp128::D64OneHotMultiChunkW2R2>(
        &[1],
        3,
        akita_types::SetupContributionMode::Direct,
    );
}

#[test]
fn multi_group_root_folded_two_group_onehot_round_trips() {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = PRE_NV * 2;
    const PRE_SIZE: usize = 1;
    const FINAL_SIZE: usize = 1;
    const TOTAL_SIZE: usize = PRE_SIZE + FINAL_SIZE;

    let pre_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_SIZE);
    let pre_opening_batch = OpeningClaimsLayout::new(PRE_NV, PRE_SIZE).expect("precommit batch");
    let pre_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit layout");
    let pre_polys = [debug_make_onehot_poly(&pre_layout, 0x0bee_fcaf_9a77_8001)];

    let setup = ConservativeCommitter::setup_prover(FINAL_NV, TOTAL_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let (pre_commitment, pre_hint) =
        ConservativeCommitter::batched_commit(&setup, &pre_polys, &stack).expect("precommit");
    let pre_frozen = akita_types::PrecommittedGroupParams::from_params(pre_key, &pre_layout);
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, FINAL_SIZE),
        precommitteds: vec![pre_frozen],
    };
    let multi_group_schedule =
        OneHotCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let final_layout = multi_group_root_params(&multi_group_schedule);
    let final_polys = [debug_make_onehot_poly(final_layout, 0x0bee_fcaf_9a77_9001)];
    let (final_commitment, final_hint) =
        RegularCommitter::commit_final_group(&setup, &final_polys, &stack, vec![pre_key])
            .expect("final multi-group commitment");

    let point = debug_random_point(FINAL_NV);
    let pre_opening = opening_from_poly(&pre_polys[0], &point[..PRE_NV], &pre_layout);
    let final_opening = opening_from_poly(&final_polys[0], &point, final_layout);
    let pre_refs: Vec<&OneHotPoly<OneHotF, u8>> = pre_polys.iter().collect();
    let final_refs: Vec<&OneHotPoly<OneHotF, u8>> = final_polys.iter().collect();
    let prover_groups = vec![
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
            vec![pre_opening],
            pre_commitment.clone(),
        )
        .expect("pre prover group"),
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            vec![final_opening],
            final_commitment.clone(),
        )
        .expect("final prover group"),
    ];
    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
        vec![pre_hint, final_hint],
        vec![&pre_refs[..], &final_refs[..]],
    )
    .expect("multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-root-direct");
    let proof = RegularCommitter::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));
    if matches!(proof.root, akita_types::AkitaBatchedRootProof::Fold(_)) {
        assert!(
            !proof.steps.is_empty(),
            "intermediate multi-group root must hand off to a suffix"
        );
    }

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

    let verifier_setup = RegularCommitter::setup_verifier(&setup);
    let verify_claims = || {
        OpeningClaims::from_groups(
            point.clone(),
            vec![
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                    vec![pre_opening],
                    &pre_commitment,
                )
                .expect("pre verifier group"),
                PolynomialGroupClaims::new(
                    PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                    vec![final_opening],
                    &final_commitment,
                )
                .expect("final verifier group"),
            ],
        )
        .expect("multi-group verifier claims")
    };
    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-root-direct");
    RegularCommitter::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_claims(),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("multi-group verify");

    // Negative: swapping the two group commitments must reject.
    let mut swapped_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-root-direct");
    let swapped_claims = OpeningClaims::from_groups(
        point.clone(),
        vec![
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                vec![pre_opening],
                &final_commitment,
            )
            .expect("pre verifier group"),
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                vec![final_opening],
                &pre_commitment,
            )
            .expect("final verifier group"),
        ],
    )
    .expect("swapped verifier claims");
    assert!(
        RegularCommitter::batched_verify(
            &decoded,
            &verifier_setup,
            &mut swapped_transcript,
            swapped_claims,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "swapped group commitments must reject"
    );

    // Negative: tampering the final group's opening must reject.
    let mut tampered_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-root-direct");
    let tampered_claims = OpeningClaims::from_groups(
        point.clone(),
        vec![
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre point vars"),
                vec![pre_opening],
                &pre_commitment,
            )
            .expect("pre verifier group"),
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                vec![final_opening + OneHotF::one()],
                &final_commitment,
            )
            .expect("final verifier group"),
        ],
    )
    .expect("tampered verifier claims");
    assert!(
        RegularCommitter::batched_verify(
            &decoded,
            &verifier_setup,
            &mut tampered_transcript,
            tampered_claims,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .is_err(),
        "tampered group opening must reject"
    );
}

#[test]
fn multi_group_root_folded_three_group_onehot_round_trips() {
    const PRE_NV: usize = 8;
    const FINAL_NV: usize = PRE_NV * 2;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 1;
    const FINAL_SIZE: usize = 1;
    const TOTAL_SIZE: usize = PRE_A_SIZE + PRE_B_SIZE + FINAL_SIZE;

    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_B_SIZE);
    let pre_opening_batch = OpeningClaimsLayout::new(PRE_NV, PRE_A_SIZE).expect("precommit batch");
    let pre_a_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit A layout");
    let pre_b_layout = ConservativeOneHotCfg::get_params_for_batched_commitment(&pre_opening_batch)
        .expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_5001)];
    let pre_b_polys = [debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_6001)];

    let setup = ConservativeCommitter::setup_prover(FINAL_NV, TOTAL_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");

    let (pre_a_commitment, pre_a_hint) =
        ConservativeCommitter::batched_commit(&setup, &pre_a_polys, &stack).expect("precommit A");
    let (pre_b_commitment, pre_b_hint) =
        ConservativeCommitter::batched_commit(&setup, &pre_b_polys, &stack).expect("precommit B");
    let pre_a_frozen = akita_types::PrecommittedGroupParams::from_params(pre_a_key, &pre_a_layout);
    let pre_b_frozen = akita_types::PrecommittedGroupParams::from_params(pre_b_key, &pre_b_layout);
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, FINAL_SIZE),
        precommitteds: vec![pre_a_frozen, pre_b_frozen],
    };
    let multi_group_schedule =
        OneHotCfg::runtime_schedule(multi_group_key).expect("multi-group runtime schedule");
    let main_params = multi_group_root_params(&multi_group_schedule);
    let final_polys = [debug_make_onehot_poly(main_params, 0x0bee_fcaf_9a77_7001)];
    let (final_commitment, final_hint) = RegularCommitter::commit_final_group(
        &setup,
        &final_polys,
        &stack,
        vec![pre_a_key, pre_b_key],
    )
    .expect("final multi-group commitment");

    let point = debug_random_point(FINAL_NV);
    let pre_a_opening = opening_from_poly(&pre_a_polys[0], &point[..PRE_NV], &pre_a_layout);
    let pre_b_opening = opening_from_poly(&pre_b_polys[0], &point[..PRE_NV], &pre_b_layout);
    let final_opening = opening_from_poly(&final_polys[0], &point, main_params);

    let pre_a_refs: Vec<&OneHotPoly<OneHotF, u8>> = pre_a_polys.iter().collect();
    let pre_b_refs: Vec<&OneHotPoly<OneHotF, u8>> = pre_b_polys.iter().collect();
    let final_refs: Vec<&OneHotPoly<OneHotF, u8>> = final_polys.iter().collect();
    let prover_groups = vec![
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre A point vars"),
            vec![pre_a_opening],
            pre_a_commitment.clone(),
        )
        .expect("pre A prover group"),
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre B point vars"),
            vec![pre_b_opening],
            pre_b_commitment.clone(),
        )
        .expect("pre B prover group"),
        PolynomialGroupClaims::new(
            PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
            vec![final_opening],
            final_commitment.clone(),
        )
        .expect("final prover group"),
    ];
    let prover_claims = ProverOpeningData::new(
        OpeningClaims::from_groups(point.clone(), prover_groups).expect("prover claims"),
        vec![pre_a_hint, pre_b_hint, final_hint],
        vec![&pre_a_refs[..], &pre_b_refs[..], &final_refs[..]],
    )
    .expect("multi-group prover data");

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-root-direct-3");
    let proof = RegularCommitter::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("multi-group prove");
    assert!(!matches!(
        proof.root,
        akita_types::AkitaBatchedRootProof::ZeroFold { .. }
    ));
    if matches!(proof.root, akita_types::AkitaBatchedRootProof::Fold(_)) {
        assert!(
            !proof.steps.is_empty(),
            "intermediate multi-group root must hand off to a suffix"
        );
    }

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

    let verifier_setup = RegularCommitter::setup_verifier(&setup);
    let verify_claims = OpeningClaims::from_groups(
        point.clone(),
        vec![
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre A point vars"),
                vec![pre_a_opening],
                &pre_a_commitment,
            )
            .expect("pre A verifier group"),
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(PRE_NV, FINAL_NV).expect("pre B point vars"),
                vec![pre_b_opening],
                &pre_b_commitment,
            )
            .expect("pre B verifier group"),
            PolynomialGroupClaims::new(
                PointVariableSelection::prefix(FINAL_NV, FINAL_NV).expect("final point vars"),
                vec![final_opening],
                &final_commitment,
            )
            .expect("final verifier group"),
        ],
    )
    .expect("multi-group verifier claims");
    let mut verifier_transcript =
        AkitaTranscript::<OneHotF>::new(b"test/multi-group-root-direct-3");
    RegularCommitter::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verify_claims,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("multi-group verify");
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
    let verifier_setup = OneHotScheme::setup_verifier(&setup);
    let (commitment, hint) =
        OneHotScheme::commit::<_, _>(&setup, &polys, &stack).expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let proof = OneHotScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(
            &point[..],
            &poly_refs[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched onehot prove");

    let expected_shape = expected_same_point_batched_shape(NV, BATCH_SIZE, &proof);
    let actual_shape = proof.shape();
    // The expected and actual shapes must match in their root variant: either
    // both `Fold` (multi-fold schedules) or both `Terminal` (1-fold schedules).
    match (&expected_shape, &actual_shape) {
        (
            AkitaBatchedProofShape::Fold {
                root_shape: expected_root,
                step_shapes: expected_steps,
            },
            AkitaBatchedProofShape::Fold {
                root_shape: actual_root,
                step_shapes: actual_steps,
            },
        ) => {
            assert_eq!(expected_root.v_coeffs, actual_root.v_coeffs);
            assert_eq!(expected_root.stage1_stages, actual_root.stage1_stages);
            assert_eq!(
                expected_root.stage2_sumcheck_proof,
                actual_root.stage2_sumcheck_proof
            );
            assert_eq!(
                expected_root.next_commit_coeffs,
                actual_root.next_commit_coeffs
            );
            assert_eq!(expected_steps.len(), actual_steps.len());
            for (expected_step, actual_step) in expected_steps.iter().zip(actual_steps.iter()) {
                match (expected_step, actual_step) {
                    (
                        AkitaProofStepShape::Terminal(expected_terminal),
                        AkitaProofStepShape::Terminal(actual_terminal),
                    ) => {
                        assert_eq!(
                            expected_terminal.extension_opening_reduction,
                            actual_terminal.extension_opening_reduction
                        );
                        assert_eq!(
                            expected_terminal.stage2_sumcheck.len(),
                            actual_terminal.stage2_sumcheck.len(),
                            "terminal stage-2 round count"
                        );
                        assert!(
                            expected_terminal
                                .final_witness
                                .admits_realized(&actual_terminal.final_witness),
                            "terminal witness shape {:?} does not admit {:?}",
                            expected_terminal.final_witness,
                            actual_terminal.final_witness
                        );
                    }
                    _ => assert_eq!(expected_step, actual_step),
                }
            }
        }
        (
            AkitaBatchedProofShape::Terminal(expected_terminal),
            AkitaBatchedProofShape::Terminal(actual_terminal),
        ) => {
            assert_eq!(
                expected_terminal.extension_opening_reduction,
                actual_terminal.extension_opening_reduction
            );
            assert_eq!(
                expected_terminal.stage2_sumcheck,
                actual_terminal.stage2_sumcheck
            );
            assert!(
                expected_terminal
                    .final_witness
                    .admits_realized(&actual_terminal.final_witness),
                "terminal witness shape {:?} does not admit {:?}",
                expected_terminal.final_witness,
                actual_terminal.final_witness
            );
        }
        _ => panic!(
            "expected and actual shape root variants disagree: expected={expected_shape:?}, actual={actual_shape:?}"
        ),
    }
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
        verifier_claims(&point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched onehot verify");
}
