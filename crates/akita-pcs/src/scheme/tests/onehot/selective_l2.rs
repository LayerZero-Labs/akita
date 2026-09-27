use super::*;

#[test]
fn selective_l2_proof_rejects_transcript_mutations() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(selective_l2_proof_rejects_transcript_mutations_inner)
        .expect("selective L2 test thread")
        .join()
        .expect("selective L2 test thread panicked");
}

fn selective_l2_proof_rejects_transcript_mutations_inner() {
    const NV: usize = 30;
    const BATCH_SIZE: usize = 4;
    const TRANSCRIPT_LABEL: &[u8] = b"test/selective-l2-mutations";
    type L2Cfg = OneHotCfg;

    let scheme = workspace_scheme::<L2Cfg>().expect("workspace schedule artifact");
    let layout = catalog_root_layout(&scheme, NV, BATCH_SIZE);
    let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..BATCH_SIZE)
        .map(|index| debug_make_onehot_poly(NV, layout.d_a(), 0x0bee_fcaf_1200_0000 + index as u64))
        .collect();

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
    let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
    let verifier_setup = scheme.setup_verifier(&setup).expect("L2 verifier setup");
    let akita_cpu_backend::CommitOutput {
        committed_group: commitment,
        private_handle: hint,
    } = stack
        .commit(
            scheme.schedules(),
            &stack.import_source(polys.to_vec()).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("L2 commitment");
    let commitments = [commitment];
    let prover_group =
        PolynomialGroupClaims::new(point.clone(), openings.clone(), commitments[0].clone())
            .expect("L2 prover group");
    let proof = scheme
        .batched_prove(
            &setup,
            selected_prover_data::<L2Cfg, _>(
                &scheme,
                OpeningClaims::from_groups(vec![prover_group]).expect("L2 prover claims"),
                vec![hint],
            )
            .expect("L2 opening data"),
            &stack,
            TRANSCRIPT_LABEL,
            BasisMode::Lagrange,
        )
        .expect("L2 proof");

    let verify = |candidate: &[u8]| {
        let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            openings.clone(),
            &commitments[0],
        )
        .expect("L2 verifier group")])
        .expect("L2 verifier claims");
        scheme.batched_verify(
            candidate,
            &verifier_setup,
            TRANSCRIPT_LABEL,
            selected_statement::<L2Cfg>(&scheme, claims).expect("L2 verifier statement"),
            BasisMode::Lagrange,
        )
    };
    verify(&proof).expect("valid L2 proof");

    for offset in [0, proof.len() / 4, proof.len() / 2, proof.len() - 1] {
        let mut mutated = proof.clone();
        mutated[offset] ^= 1;
        assert!(verify(&mutated).is_err(), "mutated proof accepted");
    }

    let truncated = &proof[..proof.len() - 1];
    assert!(verify(truncated).is_err());
}
