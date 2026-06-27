use super::*;

#[test]
fn commit_group_returns_frozen_conservative_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::AkitaScheduleLookupKey::new(NV, GROUP_SIZE);
    let layout = OneHotCfg::get_params_for_group_commit(&key).expect("group commit layout");
    let total_field = (layout.num_blocks * layout.block_len)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(&layout, 0x0bee_fcaf_9a77_0001)];

    let setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, GROUP_SIZE)
        .expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let handle =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit_group(&setup, &polys, &stack)
            .expect("commit group");

    assert_eq!(handle.schedule.layout.key, key);
    assert_eq!(handle.schedule.layout.m_vars, layout.m_vars);
    assert_eq!(handle.schedule.layout.r_vars, layout.r_vars);
    assert_eq!(handle.schedule.layout.log_basis, OneHotCfg::basis_range().0);
    assert_eq!(handle.schedule.layout.n_a, layout.a_key.row_len());
    assert_eq!(
        handle.schedule.layout.conservative_n_b,
        layout.b_key.row_len()
    );
    assert_eq!(
        handle.commitment.u.len(),
        handle.schedule.layout.conservative_n_b
    );
}

fn grouped_root_params(schedule: &akita_types::Schedule) -> &LevelParams {
    match schedule.steps.first().expect("grouped schedule step") {
        Step::Direct(direct) => direct.params.as_ref().expect("grouped root params"),
        Step::Fold(fold) => &fold.params,
    }
}

fn with_group_commit_stack<R>(
    max_num_vars: usize,
    max_num_polys: usize,
    run: impl FnOnce(
        &akita_prover::AkitaProverSetup<OneHotF, ONEHOT_D>,
        &akita_prover::UniformProverStack<'_, OneHotF, CpuBackend, ONEHOT_D>,
    ) -> R,
) -> R {
    let setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(
        max_num_vars,
        max_num_polys,
    )
    .expect("setup");
    let prepared = CpuBackend.prepare_setup(&setup).expect("prepared setup");
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    run(&setup, &stack)
}

#[test]
fn commit_group_allows_independent_precommitted_groups() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;

    let pre_a_key = akita_types::AkitaScheduleLookupKey::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::AkitaScheduleLookupKey::new(NV, PRE_B_SIZE);
    let pre_a_layout =
        OneHotCfg::get_params_for_group_commit(&pre_a_key).expect("precommit A layout");
    let pre_b_layout =
        OneHotCfg::get_params_for_group_commit(&pre_b_key).expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_1001)];
    let pre_b_polys = [
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_2001),
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_2002),
    ];

    with_group_commit_stack(NV, PRE_A_SIZE + PRE_B_SIZE, |setup, stack| {
        let pre_a = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit_group(
            setup,
            &pre_a_polys,
            stack,
        )
        .expect("precommit A");
        let pre_b = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit_group(
            setup,
            &pre_b_polys,
            stack,
        )
        .expect("precommit B");

        assert_eq!(pre_a.schedule.layout.key, pre_a_key);
        assert_eq!(pre_b.schedule.layout.key, pre_b_key);
        assert_eq!(
            pre_a.commitment.u.len(),
            pre_a.schedule.layout.conservative_n_b
        );
        assert_eq!(
            pre_b.commitment.u.len(),
            pre_b.schedule.layout.conservative_n_b
        );
        assert_ne!(pre_a.schedule.layout.key, pre_b.schedule.layout.key);
    });
}

#[test]
fn group_batch_schedule_preserves_precommitted_order() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;
    const MAIN_SIZE: usize = 3;

    let pre_a_key = akita_types::AkitaScheduleLookupKey::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::AkitaScheduleLookupKey::new(NV, PRE_B_SIZE);
    let pre_a_layout =
        OneHotCfg::get_params_for_group_commit(&pre_a_key).expect("precommit A layout");
    let pre_b_layout =
        OneHotCfg::get_params_for_group_commit(&pre_b_key).expect("precommit B layout");
    let pre_a_polys = [debug_make_onehot_poly(&pre_a_layout, 0x0bee_fcaf_9a77_3001)];
    let pre_b_polys = [
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_4001),
        debug_make_onehot_poly(&pre_b_layout, 0x0bee_fcaf_9a77_4002),
    ];

    with_group_commit_stack(NV, PRE_A_SIZE + PRE_B_SIZE + MAIN_SIZE, |setup, stack| {
        let pre_a = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit_group(
            setup,
            &pre_a_polys,
            stack,
        )
        .expect("precommit A");
        let pre_b = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit_group(
            setup,
            &pre_b_polys,
            stack,
        )
        .expect("precommit B");
        let grouped_key = akita_types::GroupBatchAkitaScheduleLookupKey {
            main: akita_types::AkitaScheduleLookupKey::new(NV, MAIN_SIZE),
            precommitteds: vec![pre_a.schedule.layout.clone(), pre_b.schedule.layout.clone()],
        };

        let schedule = OneHotCfg::get_group_batch_schedule(&grouped_key).expect("grouped schedule");
        let root = grouped_root_params(&schedule);

        assert_eq!(grouped_key.num_commitment_groups(), 3);
        assert_eq!(
            grouped_key
                .num_polynomials()
                .expect("grouped polynomial count"),
            PRE_A_SIZE + PRE_B_SIZE + MAIN_SIZE
        );
        assert_eq!(root.precommitted_groups.len(), 2);
        assert_eq!(root.precommitted_groups[0].layout, pre_a.schedule.layout);
        assert_eq!(root.precommitted_groups[1].layout, pre_b.schedule.layout);
    });
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

    let polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
        .map(|poly_idx| debug_make_onehot_poly(&layout, 0x0bee_fcaf_e000_1500 + poly_idx as u64))
        .collect();
    let poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> = polys.iter().collect();
    let point = debug_random_point(NV);
    let openings: Vec<OneHotF> = polys
        .iter()
        .map(|poly| opening_from_poly(poly, &point, &layout))
        .collect();

    let setup = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE)
        .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(&setup, &polys, &stack)
            .expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
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
    <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched onehot verify");
}
