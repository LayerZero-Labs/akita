use super::*;

/// Direct-terminal onehot fixture. The inner [`DirectTerminalCfg`] flips the
/// terminal proof mode and re-materializes the inner table under
/// `DirectRingRelations` (keeping the table's root/fold structure and envelope
/// floor); the outer [`PlannerCfg`] additionally covers any table-miss
/// incidence through the direct-mode planner DP. No hand surgery: this
/// exercises the production schedule-construction path with only the terminal
/// mode flipped.
///
/// [`DirectTerminalCfg`]: akita_planner::test_utils::DirectTerminalCfg
/// [`PlannerCfg`]: akita_planner::test_utils::PlannerCfg
type DirectRecursiveOneHotCfg = akita_planner::test_utils::PlannerCfg<
    akita_planner::test_utils::DirectTerminalCfg<fp128::D64OneHot>,
>;

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
        .map(|poly| debug_opening_from_poly(poly, &point, &layout))
        .collect();

    let setup =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE, 1)
            .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup =
        <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .expect("batched onehot commit");
    let commitments = [commitment];
    let hints = vec![hint];

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let proof = <OneHotScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
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
            assert_eq!(expected_root.y_ring_coeffs, actual_root.y_ring_coeffs);
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
            assert_eq!(expected_steps, actual_steps);
        }
        (
            AkitaBatchedProofShape::Terminal(expected_terminal),
            AkitaBatchedProofShape::Terminal(actual_terminal),
        ) => {
            assert_eq!(expected_terminal, actual_terminal);
        }
        _ => panic!(
            "expected and actual shape root variants disagree: expected={expected_shape:?}, actual={actual_shape:?}"
        ),
    }
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let decoded =
        AkitaBatchedProof::<OneHotF, OneHotF>::deserialize_uncompressed(&*bytes, &expected_shape)
            .expect("deserialize batched proof with derived shape");
    assert_eq!(decoded, proof);

    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    <OneHotScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched onehot verify");
}

#[test]
fn direct_recursive_terminal_onehot_roundtrip() {
    const NV: usize = 20;
    const BATCH_SIZE: usize = 2;
    type DirectScheme = AkitaCommitmentScheme<ONEHOT_D, DirectRecursiveOneHotCfg>;

    // Direct mode legitimately shifts the DP-optimal root split (the cheaper
    // terminal step changes the cost landscape), so size the polynomials from
    // the direct config's actual prove schedule rather than the table-only
    // `akita_batched_root_layout`, which would disagree with what `commit` uses.
    let incidence = ClaimIncidenceSummary::same_point(NV, BATCH_SIZE).expect("incidence");
    let batched_root = DirectRecursiveOneHotCfg::get_params_for_batched_commitment(&incidence)
        .expect("direct batched root layout");
    let layout = akita_types::split_batched_root_params(
        &batched_root,
        DirectRecursiveOneHotCfg::decomposition().field_bits(),
    );
    let polys: Vec<OneHotPoly<OneHotF, ONEHOT_D, u8>> = (0..BATCH_SIZE)
        .map(|poly_idx| debug_make_onehot_poly(&layout, 0x0bee_fcaf_d1ec_7000 + poly_idx as u64))
        .collect();
    let poly_refs: Vec<&OneHotPoly<OneHotF, ONEHOT_D, u8>> = polys.iter().collect();
    let point = debug_random_point(NV);
    let openings: Vec<OneHotF> = polys
        .iter()
        .map(|poly| debug_opening_from_poly(poly, &point, &layout))
        .collect();

    let setup =
        <DirectScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_prover(NV, BATCH_SIZE, 1)
            .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup =
        <DirectScheme as CommitmentProver<OneHotF, ONEHOT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <DirectScheme as CommitmentProver<OneHotF, ONEHOT_D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        &poly_refs,
    )
    .expect("batched onehot commit");
    let commitments = [commitment];

    let mut prover_transcript =
        AkitaTranscript::<OneHotF>::new(b"test/direct-recursive-terminal-onehot");
    let proof = <DirectScheme as CommitmentProver<OneHotF, ONEHOT_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched onehot direct recursive terminal prove");

    let AkitaBatchedProofShape::Fold { step_shapes, .. } = proof.shape() else {
        panic!("direct recursive fixture should keep a recursive suffix");
    };
    let Some(AkitaProofStepShape::Terminal(terminal_shape)) = step_shapes.last() else {
        panic!("direct recursive fixture should end with a terminal proof step");
    };
    assert!(matches!(
        terminal_shape.relation,
        TerminalRelationProofShape::DirectRingRelations
    ));
    let Some(akita_types::AkitaProofStep::Terminal(terminal)) = proof.steps.last() else {
        panic!("direct recursive fixture should carry a terminal proof step");
    };
    assert!(terminal.stage2_sumcheck().is_none());

    let opening_groups = [&openings[..]];
    let mut verifier_transcript =
        AkitaTranscript::<OneHotF>::new(b"test/direct-recursive-terminal-onehot");
    <DirectScheme as CommitmentVerifier<OneHotF, ONEHOT_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
    )
    .expect("batched onehot direct recursive terminal verify");
}
