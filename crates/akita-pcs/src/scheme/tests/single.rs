use super::*;

#[test]
fn verify_rejects_wrong_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = Scheme::setup_prover(num_vars, 1).unwrap();
    let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");

    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = Scheme::commit::<_, _>(
        &setup,
        std::slice::from_ref(&poly),
        &stack,
        akita_prover::GroupContext::scheduler_without_prior_groups(),
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
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&opening_point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    let wrong_opening = opening + F::one();
    let wrong_openings = [wrong_opening];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &wrong_openings[..], &commitments[0]),
        BasisMode::Lagrange,
    );

    assert!(
        result.is_err(),
        "verify must reject an incorrect opening value"
    );
}

#[test]
fn verify_rejects_malformed_v_dimension_without_panicking() {
    let (verifier_setup, commitment, mut proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let root_fold = &mut proof.root;
    let mut coeffs = root_fold.opening_payload.coeffs().to_vec();
    let _ = coeffs.pop().expect("expected non-empty v");
    root_fold.opening_payload = RingVec::from_coeffs(coeffs);

    let commitments = [commitment];
    let openings = [opening];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
        Scheme::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
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

    let (_, _, proof, _, _, _) = make_verify_fixture(16);
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
    let (verifier_setup, commitment, mut proof, opening_point, opening, _) =
        make_verify_fixture(16);
    let dummy_sumcheck = proof.root.stage2.sumcheck_proof.clone();
    proof.root.extension_opening_reduction = Some(ExtensionOpeningReductionProof {
        partials: vec![F::zero()],
        sumcheck: dummy_sumcheck,
    });

    let openings = [opening];
    let commitments = [commitment];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
    )
    .expect_err("unchecked extension-opening payload must be rejected");
}
