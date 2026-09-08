use super::*;

#[test]
fn selective_l2_proof_rejects_transcript_mutations() {
    const NV: usize = 30;
    const BATCH_SIZE: usize = 4;
    const TRANSCRIPT_LABEL: &[u8] = b"test/selective-l2-mutations";
    type L2Cfg = OneHotCfg;

    let scheme = workspace_scheme::<L2Cfg>().expect("workspace schedule artifact");
    let layout = catalog_root_layout(&scheme, NV, BATCH_SIZE);
    let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..BATCH_SIZE)
        .map(|index| debug_make_onehot_poly(NV, layout.d_a(), 0x0bee_fcaf_1200_0000 + index as u64))
        .collect();
    let poly_refs: Vec<&OneHotPoly<OneHotF, u8>> = polys.iter().collect();
    let point = debug_random_point(NV);
    let openings: Vec<OneHotF> = polys
        .iter()
        .map(|poly| {
            opening_from_poly(
                poly,
                &point,
                layout.d_a(),
                layout.blocks().positions_per_block,
                layout.blocks().live_blocks,
            )
        })
        .collect();

    let setup = scheme.setup_prover(NV, BATCH_SIZE).expect("L2 setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared L2 setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("L2 stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("L2 verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            &polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("L2 commitment");
    let commitments = [commitment];
    let prover_group = PolynomialGroupClaims::new(
        point.clone(),
        vec![OneHotF::zero(); BATCH_SIZE],
        commitments[0].clone(),
    )
    .expect("L2 prover group");
    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(TRANSCRIPT_LABEL);
    let proof = scheme
        .batched_prove::<_, _, _>(
            &setup,
            selected_prover_data::<L2Cfg, _>(
                &scheme,
                OpeningClaims::from_groups(vec![prover_group]).expect("L2 prover claims"),
                vec![hint],
                vec![&poly_refs],
            )
            .expect("L2 opening data"),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("L2 proof");

    let verify = |candidate: &AkitaBatchedProof<OneHotF, OneHotF>| {
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            openings.clone(),
            &commitments[0],
        )
        .expect("L2 verifier group")])
        .expect("L2 verifier claims");
        let mut transcript = AkitaTranscript::<OneHotF>::new(TRANSCRIPT_LABEL);
        scheme.batched_verify(
            candidate,
            &verifier_setup,
            &mut transcript,
            selected_statement::<L2Cfg>(&scheme, claims).expect("L2 verifier statement"),
            BasisMode::Lagrange,
        )
    };
    verify(&proof).expect("valid L2 proof");

    let l2_index = proof
        .recursive_folds
        .iter()
        .position(|fold| fold.stage1.norm_proof.is_some())
        .expect("generated schedule must select one L2 fold");
    let mut bad_norm = proof.clone();
    bad_norm.recursive_folds[l2_index]
        .stage1
        .norm_proof
        .as_mut()
        .expect("L2 norm")
        .response_l2_sq += 1;
    assert!(verify(&bad_norm).is_err());

    let mut over_cap = proof.clone();
    over_cap.recursive_folds[l2_index]
        .stage1
        .norm_proof
        .as_mut()
        .expect("L2 norm")
        .response_l2_sq = u128::MAX;
    assert!(verify(&over_cap).is_err());

    let mut bad_virtual = proof.clone();
    bad_virtual.recursive_folds[l2_index]
        .stage1
        .norm_proof
        .as_mut()
        .expect("L2 norm")
        .virtual_evaluations[0] += OneHotF::one();
    assert!(verify(&bad_virtual).is_err());

    let mut bad_sumcheck = proof.clone();
    bad_sumcheck.recursive_folds[l2_index]
        .stage1
        .norm_proof
        .as_mut()
        .expect("L2 norm")
        .sumcheck
        .round_polys[0]
        .coeffs_except_linear_term[0] += OneHotF::one();
    assert!(verify(&bad_sumcheck).is_err());

    let mut bad_nonce = proof;
    let mut nonce_bytes = bad_nonce.nonce_stream.as_bytes().to_vec();
    nonce_bytes[0] ^= 1;
    bad_nonce.nonce_stream = akita_types::TranscriptNonceStream::from_bytes(
        nonce_bytes,
        bad_nonce.nonce_stream.bit_len(),
    )
    .unwrap();
    assert!(verify(&bad_nonce).is_err());
}
