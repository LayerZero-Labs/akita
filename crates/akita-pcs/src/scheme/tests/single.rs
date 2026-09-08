use super::*;

#[test]
fn reduced_relation_catalog_roundtrip_reaches_production_verifier() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| {
            const NUM_VARS: usize = 16;

            let (scheme, verifier_setup, commitment, mut proof, opening_point, opening, _) =
                make_verify_fixture(NUM_VARS);
            let key = akita_types::AkitaScheduleLookupKey::single(
                akita_types::PolynomialGroupLayout::new(NUM_VARS, 1),
            );
            let selection = scheme
                .schedules()
                .resolve_key(&key)
                .expect("shipped reduced-relation schedule");
            let schedule = selection.schedule();
            let first_reduced_index = schedule
                .recursive_folds
                .iter()
                .position(|fold| fold.params.ring_relation_mode.is_reduced_evaluation())
                .expect("fixture reduced cutover");
            let reduced = &schedule.recursive_folds[first_reduced_index..];
            assert!(
                reduced.len() >= 2,
                "fixture must execute more than one reduced recursive fold"
            );
            assert!(reduced
                .iter()
                .any(|fold| fold.params.payload_mode.is_compressed()));
            assert!(reduced.iter().any(|fold| matches!(
                fold.params.payload_mode,
                akita_types::CommitmentPayloadMode::Raw
            )));
            assert!(schedule
                .recursive_folds
                .iter()
                .skip_while(|fold| !fold.params.ring_relation_mode.is_reduced_evaluation())
                .all(|fold| fold.params.ring_relation_mode.is_reduced_evaluation()));

            let commitments = [commitment];
            let openings = [opening];
            let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
            scheme
                .batched_verify(
                    &proof,
                    &verifier_setup,
                    &mut verifier_transcript,
                    verifier_claims(&scheme, &opening_point, &openings, &commitments[0]),
                    BasisMode::Lagrange,
                )
                .expect("production verifier must replay the reduced-relation suffix");

            let first_round = proof.recursive_folds[first_reduced_index]
                .stage2
                .sumcheck_proof
                .round_polys
                .first_mut()
                .expect("reduced stage2 sumcheck round");
            let coefficient = first_round
                .coeffs_except_linear_term
                .first_mut()
                .expect("reduced stage2 sumcheck coefficient");
            *coefficient += F::one();
            let mut tampered_transcript = AkitaTranscript::<F>::new(b"test/prove");
            scheme
                .batched_verify(
                    &proof,
                    &verifier_setup,
                    &mut tampered_transcript,
                    verifier_claims(&scheme, &opening_point, &openings, &commitments[0]),
                    BasisMode::Lagrange,
                )
                .expect_err("production verifier must reject a tampered reduced stage2 proof");
        })
        .expect("reduced-relation test thread")
        .join()
        .expect("reduced-relation test thread panicked");
}

#[test]
fn verify_rejects_wrong_opening() {
    let scheme = workspace_scheme::<Cfg>().expect("workspace schedule artifact");
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout(&scheme, 16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = scheme.setup_prover(num_vars, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            std::slice::from_ref(&poly),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = scheme
        .batched_prove::<_, _, _>(
            &setup,
            prover_claims(
                &scheme,
                &opening_point[..],
                &poly_refs[..],
                &commitments[0],
                hint,
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .unwrap();

    let wrong_opening = opening + F::one();
    let wrong_openings = [wrong_opening];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = scheme.batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(
            &scheme,
            &opening_point[..],
            &wrong_openings[..],
            &commitments[0],
        ),
        BasisMode::Lagrange,
    );

    assert!(
        result.is_err(),
        "verify must reject an incorrect opening value"
    );
}

#[test]
fn verify_rejects_malformed_v_dimension_without_panicking() {
    let (scheme, verifier_setup, commitment, mut proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let root_fold = &mut proof.root;
    let mut coeffs = root_fold.opening_payload.coeffs().to_vec();
    let _ = coeffs.pop().expect("expected non-empty v");
    root_fold.opening_payload = RingVec::from_coeffs(coeffs);

    let commitments = [commitment];
    let openings = [opening];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
        scheme.batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&scheme, &opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        )
    }));

    assert!(
        matches!(result, Ok(Err(_))),
        "malformed opening payload must be rejected without panicking"
    );
}

#[test]
fn folded_payload_commitments_and_digits_stay_base_field() {
    fn assert_base_flat_ring_vec(_: &RingVec<F>) {}
    fn assert_base_direct_witness(_: &akita_types::TerminalResponse<F>) {}

    let (_, _, _, proof, _, _, _) = make_verify_fixture(16);
    let root = &proof.root;
    assert_base_flat_ring_vec(&root.opening_payload);
    if let Some(commitment) = root.stage2.next_witness_binding.outer_payload() {
        assert_base_flat_ring_vec(commitment);
    }

    for level in proof.nonterminal_folds() {
        assert_base_flat_ring_vec(&level.opening_payload);
        if let Some(commitment) = level.stage2.next_witness_binding.outer_payload() {
            assert_base_flat_ring_vec(commitment);
        }
    }
    assert_base_direct_witness(proof.terminal_response());
}

#[test]
fn folded_root_rejects_unchecked_extension_opening_reduction_payload() {
    let (scheme, verifier_setup, commitment, mut proof, opening_point, opening, _) =
        make_verify_fixture(16);
    let dummy_sumcheck = akita_sumcheck::SumcheckProof {
        round_polys: proof.root.stage2.sumcheck_proof.round_polys.to_vec(),
    };
    proof.root.extension_opening_reduction = Some(ExtensionOpeningReductionProof {
        partials: vec![F::zero()],
        sumcheck: dummy_sumcheck,
        final_claims: vec![F::zero()],
    });

    let openings = [opening];
    let commitments = [commitment];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    scheme
        .batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&scheme, &opening_point[..], &openings[..], &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect_err("unchecked extension-opening payload must be rejected");
}
