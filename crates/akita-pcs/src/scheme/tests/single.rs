use super::*;

#[test]
fn verify_passes_for_consistent_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = Scheme::setup_prover(num_vars, 1).unwrap();
    let prepared = CpuBackend
        .prepare_setup(
            &setup,
            &akita_types::PreparedNttPlan::base_envelope(setup.expanded.as_ref()).unwrap(),
        )
        .unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup);

    let (commitment, hint) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&opening_point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );

    assert!(result.is_ok());
}

#[test]
fn verify_rejects_wrong_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = Scheme::setup_prover(num_vars, 1).unwrap();
    let prepared = CpuBackend
        .prepare_setup(
            &setup,
            &akita_types::PreparedNttPlan::base_envelope(setup.expanded.as_ref()).unwrap(),
        )
        .unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup);

    let (commitment, hint) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

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
        akita_types::SetupContributionMode::Direct,
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
        akita_types::SetupContributionMode::Direct,
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
    let root_fold = proof
        .root
        .as_fold_mut()
        .expect("expected a fold-rooted batched proof");
    let mut coeffs = root_fold.v.coeffs().to_vec();
    let _ = coeffs.pop().expect("expected non-empty v");
    root_fold.v = RingVec::from_coeffs(coeffs);

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
            akita_types::SetupContributionMode::Direct,
        )
    }));

    assert!(matches!(result, Ok(Err(AkitaError::InvalidProof))));
}

#[test]
fn fp128_degree_one_batched_proof_roundtrip_is_stable() {
    let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let shape = proof.shape();

    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let mut repeated_bytes = Vec::new();
    proof.serialize_uncompressed(&mut repeated_bytes).unwrap();
    assert_eq!(bytes, repeated_bytes);

    let decoded = AkitaBatchedProof::<F, F>::deserialize_uncompressed(&*bytes, &shape)
        .expect("degree-one proof should roundtrip");
    assert_eq!(decoded, proof);

    let commitments = [commitment];
    let openings = [opening];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    Scheme::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("degree-one roundtrip proof should verify");
}

#[test]
fn folded_payload_commitments_and_digits_stay_base_field() {
    fn assert_base_flat_ring_vec(_: &RingVec<F>) {}
    fn assert_base_direct_witness(_: &akita_types::CleartextWitnessProof<F>) {}

    let (_, _, proof, _, _, _) = make_verify_fixture(16);
    let root = proof
        .root
        .as_fold()
        .expect("fixture should use folded root proof");
    assert_base_flat_ring_vec(&root.v);
    assert_base_flat_ring_vec(
        &root
            .stage2
            .as_intermediate()
            .expect("fold root proof must carry intermediate stage-2 proof")
            .next_w_commitment,
    );

    for level in proof.fold_levels() {
        assert_base_flat_ring_vec(level.v());
        assert_base_flat_ring_vec(level.next_w_commitment());
    }
    assert_base_direct_witness(proof.final_witness());
}

#[test]
fn folded_root_rejects_unchecked_extension_opening_reduction_payload() {
    let (verifier_setup, commitment, mut proof, opening_point, opening, _) =
        make_verify_fixture(16);
    let dummy_sumcheck = proof
        .root
        .as_fold()
        .expect("fixture should use folded root proof")
        .stage2
        .as_intermediate()
        .expect("fold root proof must carry intermediate stage-2 proof")
        .sumcheck_proof
        .clone();
    proof
        .root
        .as_fold_mut()
        .expect("fixture should use folded root proof")
        .extension_opening_reduction = Some(ExtensionOpeningReductionProof {
        partials: vec![F::zero()],
        sumcheck: dummy_sumcheck,
    });

    let openings = [opening];
    let commitments = [commitment];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let err = Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap_err();
    assert!(matches!(err, AkitaError::InvalidProof));
}

#[test]
fn monomial_basis_prove_verify_round_trip() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;

    let coeffs: Vec<F> = (0..len).map(|i| F::from_u64(i as u64)).collect();
    let poly = DensePoly::<F>::from_field_evals(num_vars, D, &coeffs).unwrap();

    let setup = Scheme::setup_prover(num_vars, 1).unwrap();
    let prepared = CpuBackend
        .prepare_setup(
            &setup,
            &akita_types::PreparedNttPlan::base_envelope(setup.expanded.as_ref()).unwrap(),
        )
        .unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup);

    let (commitment, hint) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

    let mw = monomial_weights(&opening_point).unwrap();
    let opening: F = coeffs
        .iter()
        .zip(mw.iter())
        .fold(F::zero(), |acc, (&c, &w)| acc + c * w);

    let poly_refs: [&DensePoly<F>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/monomial");
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&opening_point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Monomial,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/monomial");
    let result = Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Monomial,
        akita_types::SetupContributionMode::Direct,
    );

    assert!(
        result.is_ok(),
        "monomial-basis proof should verify: {result:?}"
    );
}

#[test]
fn tiny_d64_root_direct_helpers_accept_valid_proof() {
    type DirectCfg = fp128::D64Full;
    type DirectF = fp128::Field;
    const DIRECT_D: usize = DirectCfg::D;
    type DirectScheme = AkitaCommitmentScheme<DirectCfg>;

    let num_vars = 4usize;
    let evals: Vec<DirectF> = (0..(1usize << num_vars))
        .map(|i| DirectF::from_u64((i + 1) as u64))
        .collect();
    let poly = DensePoly::<DirectF>::from_field_evals(num_vars, DIRECT_D, &evals).unwrap();
    let opening_point = vec![DirectF::zero(); num_vars];
    let opening = evals[0];

    let setup = DirectScheme::setup_prover(num_vars, 1).unwrap();
    let prepared = CpuBackend
        .prepare_setup(
            &setup,
            &akita_types::PreparedNttPlan::base_envelope(setup.expanded.as_ref()).unwrap(),
        )
        .unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = DirectScheme::setup_verifier(&setup);
    let (commitment, hint) =
        DirectScheme::commit::<_, _>(&setup, std::slice::from_ref(&poly), &stack).unwrap();

    let poly_refs: [&DensePoly<DirectF>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];

    let mut prover_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    let proof = DirectScheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(&opening_point[..], &poly_refs[..], &commitments[0], hint),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    assert!(proof.is_root_direct());
    assert_eq!(proof.num_fold_levels(), 0);
    let witnesses = proof
        .root
        .as_zero_fold()
        .expect("root-direct batched proof expected");
    assert_eq!(witnesses.len(), 1);
    assert!(cleartext_witness_opening_matches::<DirectF, DirectF>(
        &witnesses[0],
        &opening_point,
        &opening,
        BasisMode::Lagrange,
    )
    .unwrap());

    let mut verifier_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    DirectScheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();
}
