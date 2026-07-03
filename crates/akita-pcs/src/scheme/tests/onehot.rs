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
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = RegularCommitter::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let (commitment, _hint) =
        ConservativeCommitter::commit(&setup, &polys, &stack).expect("conservative commit");
    let frozen_layout = akita_types::PrecommittedGroupParams::from_params(key, &layout);

    assert_eq!(frozen_layout.group, key);
    assert_eq!(frozen_layout.m_vars, layout.m_vars);
    assert_eq!(frozen_layout.r_vars, layout.r_vars);
    assert_eq!(
        frozen_layout.log_basis,
        ConservativeOneHotCfg::basis_range().0
    );
    assert_eq!(frozen_layout.n_a, layout.a_key.row_len());
    assert_eq!(frozen_layout.conservative_n_b, layout.b_key.row_len());
    assert_eq!(commitment.rows().count(), frozen_layout.conservative_n_b);
}

fn grouped_root_params(schedule: &akita_types::Schedule) -> &LevelParams {
    match schedule.steps.first().expect("grouped schedule step") {
        Step::Direct(direct) => direct.params.as_ref().expect("grouped root params"),
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
    let setup = RegularCommitter::setup_prover(max_num_vars, max_num_polys).expect("setup");
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
        assert_eq!(
            pre_a_commitment.rows().count(),
            pre_a_frozen.conservative_n_b
        );
        assert_eq!(
            pre_b_commitment.rows().count(),
            pre_b_frozen.conservative_n_b
        );
        assert_ne!(pre_a_frozen.group, pre_b_frozen.group);
    });
}

#[test]
fn group_batch_schedule_preserves_precommitted_order() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;
    const MAIN_SIZE: usize = 3;

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
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_3001)];
    let pre_b_polys = [
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_4001),
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_4002),
    ];

    with_conservative_commit_stack(NV, PRE_A_SIZE + PRE_B_SIZE + MAIN_SIZE, |setup, stack| {
        ConservativeCommitter::commit(setup, &pre_a_polys, stack).expect("precommit A");
        ConservativeCommitter::commit(setup, &pre_b_polys, stack).expect("precommit B");
        let pre_a_frozen =
            akita_types::PrecommittedGroupParams::from_params(pre_a_key, &pre_a_layout);
        let pre_b_frozen =
            akita_types::PrecommittedGroupParams::from_params(pre_b_key, &pre_b_layout);
        let grouped_key = akita_types::AkitaScheduleLookupKey {
            final_group: akita_types::PolynomialGroupLayout::new(NV, MAIN_SIZE),
            precommitteds: vec![pre_a_frozen.clone(), pre_b_frozen.clone()],
        };

        let schedule =
            OneHotCfg::runtime_schedule(grouped_key.clone()).expect("grouped runtime schedule");
        let root = grouped_root_params(&schedule);
        let main_params = OneHotCfg::get_params_for_grouped_batched_commitment(&grouped_key)
            .expect("main grouped commit params");

        assert_eq!(grouped_key.num_commitment_groups(), 3);
        assert_eq!(
            grouped_key
                .num_polynomials()
                .expect("grouped polynomial count"),
            PRE_A_SIZE + PRE_B_SIZE + MAIN_SIZE
        );
        assert_eq!(main_params, *root);
        assert_eq!(root.precommitted_groups.len(), 2);
        assert_eq!(root.precommitted_groups[0].layout, pre_a_frozen);
        assert_eq!(root.precommitted_groups[1].layout, pre_b_frozen);
    });
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
        let grouped_key = akita_types::AkitaScheduleLookupKey {
            final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, GROUP_SIZE),
            precommitteds: vec![pre_a_frozen.clone(), pre_b_frozen.clone()],
        };

        let main_params = OneHotCfg::get_params_for_grouped_batched_commitment(&grouped_key)
            .expect("main grouped commit params");
        let final_polys = [debug_make_onehot_poly(&main_params, 0x0bee_fcaf_9a77_7001)];
        let (final_commitment, final_hint) = OneHotScheme::commit_final_group::<_, _>(
            setup,
            &final_polys,
            stack,
            vec![pre_a_key, pre_b_key],
        )
        .expect("final grouped commitment");

        assert_eq!(
            pre_a_commitment.rows().count(),
            pre_a_frozen.conservative_n_b
        );
        assert_eq!(
            pre_b_commitment.rows().count(),
            pre_b_frozen.conservative_n_b
        );
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
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = OneHotScheme::setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let handle = OneHotScheme::commit_group::<_, _>(&setup, &polys, &stack).expect("commit group");

    assert_eq!(handle.schedule.layout.group, key);
    assert_eq!(handle.schedule.layout.m_vars, layout.m_vars);
    assert_eq!(handle.schedule.layout.r_vars, layout.r_vars);
    assert_eq!(handle.schedule.layout.log_basis, OneHotCfg::basis_range().0);
    assert_eq!(handle.schedule.layout.n_a, layout.a_key.row_len());
    assert_eq!(
        handle.schedule.layout.conservative_n_b,
        layout.b_key.row_len()
    );
    assert_eq!(
        handle.commitment.rows().count(),
        handle.schedule.layout.conservative_n_b
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
    let total_field = (layout.num_blocks * layout.block_len)
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
