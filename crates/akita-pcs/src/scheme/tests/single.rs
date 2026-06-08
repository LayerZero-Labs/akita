use super::*;

#[test]
fn verify_passes_for_consistent_opening() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;

    let (poly, evals) = make_dense_poly(num_vars);

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::from_ref(&poly),
        &CpuBackend,
        &prepared,
    )
    .unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let prove_stack = uniform_prove_stack(&setup, &CpuBackend, &prepared);

    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &prove_stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
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

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::from_ref(&poly),
        &CpuBackend,
        &prepared,
    )
    .unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let lw = lagrange_weights(&opening_point).unwrap();
    let opening: F = evals
        .iter()
        .zip(lw.iter())
        .fold(F::zero(), |a, (&c, &w)| a + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let prove_stack = uniform_prove_stack(&setup, &CpuBackend, &prepared);

    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &prove_stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let wrong_opening = opening + F::one();
    let wrong_openings = [wrong_opening];
    let wrong_opening_groups = [&wrong_openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: wrong_opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );

    assert!(
        result.is_err(),
        "verify must reject an incorrect opening value"
    );
}

#[test]
fn verify_rejects_malformed_y_ring_dimension_without_panicking() {
    let (verifier_setup, commitment, mut proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let root_fold = proof
        .root
        .as_fold_mut()
        .expect("expected a fold-rooted batched proof");
    let mut coeffs = root_fold.y_rings.coeffs().to_vec();
    let _ = coeffs.pop().expect("expected non-empty y_rings");
    root_fold.y_rings = FlatRingVec::from_coeffs(coeffs);

    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            vec![(
                &opening_point[..],
                CommittedOpenings {
                    openings: opening_groups[0],
                    commitment: &commitments[0],
                },
            )],
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
    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("degree-one roundtrip proof should verify");
}

#[test]
fn folded_payload_commitments_and_digits_stay_base_field() {
    fn assert_base_flat_ring_vec(_: &FlatRingVec<F>) {}
    fn assert_base_direct_witness(_: &akita_types::CleartextWitnessProof<F>) {}

    let (_, _, proof, _, _, _) = make_verify_fixture(16);
    let root = proof
        .root
        .as_fold()
        .expect("fixture should use folded root proof");
    assert_base_flat_ring_vec(&root.y_rings);
    assert_base_flat_ring_vec(&root.v);
    assert_base_flat_ring_vec(&root.stage2.next_w_commitment);

    for level in proof.fold_levels() {
        assert_base_flat_ring_vec(&level.y_ring);
        assert_base_flat_ring_vec(&level.v);
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
    let err = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: &openings[..],
                commitment: &commitments[0],
            },
        )],
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
    let poly = DensePoly::<F, D>::from_field_evals(num_vars, &coeffs).unwrap();

    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 1, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);

    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::from_ref(&poly),
        &CpuBackend,
        &prepared,
    )
    .unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();

    let mw = monomial_weights(&opening_point).unwrap();
    let opening: F = coeffs
        .iter()
        .zip(mw.iter())
        .fold(F::zero(), |acc, (&c, &w)| acc + c * w);

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/monomial");
    let prove_stack = uniform_prove_stack(&setup, &CpuBackend, &prepared);

    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &prove_stack,
        &mut prover_transcript,
        BasisMode::Monomial,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/monomial");
    let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Monomial,
        akita_types::SetupContributionMode::Direct,
    );

    assert!(
        result.is_ok(),
        "monomial-basis proof should verify: {result:?}"
    );
}

#[test]
fn tiny_d32_root_direct_helpers_accept_valid_proof() {
    type DirectCfg = fp128::D32Full;
    type DirectF = fp128::Field;
    const DIRECT_D: usize = DirectCfg::D;
    type DirectScheme = AkitaCommitmentScheme<DIRECT_D, DirectCfg>;

    let num_vars = 4usize;
    let evals: Vec<DirectF> = (0..(1usize << num_vars))
        .map(|i| DirectF::from_u64((i + 1) as u64))
        .collect();
    let poly = DensePoly::<DirectF, DIRECT_D>::from_field_evals(num_vars, &evals).unwrap();
    let opening_point = vec![DirectF::zero(); num_vars];
    let opening = evals[0];

    let setup = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_prover(num_vars, 1, 1)
        .unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup =
        <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::setup_verifier(&setup);
    let (commitment, hint) = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::commit(
        &setup,
        RootCommitPolys::from_ref(&poly),
        &CpuBackend,
        &prepared,
    )
    .unwrap();

    let poly_refs: [&DensePoly<DirectF, DIRECT_D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    let prove_stack = uniform_prove_stack(&setup, &CpuBackend, &prepared);

    let proof = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &prove_stack,
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
    <DirectScheme as CommitmentVerifier<DirectF, DIRECT_D>>::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();
}
