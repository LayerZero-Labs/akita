use super::*;

#[test]
fn batched_commit_matches_individual_commits() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
    let poly_a = DensePoly::<F>::from_field_evals(num_vars, D, &evals_a).unwrap();
    let poly_b = DensePoly::<F>::from_field_evals(num_vars, D, &evals_b).unwrap();
    let setup = Scheme::setup_prover(num_vars, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
        .iter()
        .map(|group| Scheme::commit::<_, _>(&setup, group, &stack))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .unzip();
    let (commitment_a, hint_a) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly_a), &stack).unwrap();
    let (commitment_b, hint_b) =
        Scheme::commit::<_, _>(&setup, std::slice::from_ref(&poly_b), &stack).unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
    assert_eq!(batched_hints, vec![hint_a, hint_b]);
}

#[test]
fn batched_verify_accepts_consistent_openings_and_rejects_bad_inputs() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.position_index_bits() + layout.block_index_bits() + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
    let poly_a = DensePoly::<F>::from_field_evals(num_vars, D, &evals_a).unwrap();
    let poly_b = DensePoly::<F>::from_field_evals(num_vars, D, &evals_b).unwrap();
    let setup = Scheme::setup_prover(num_vars, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let stack =
        akita_prover::UniformProverStack::uniform(&CpuBackend, &prepared, setup.expanded.as_ref())
            .expect("stack");
    let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
    let poly_group = [&poly_a, &poly_b];
    let (commitment, hint) =
        Scheme::commit::<_, _>(&setup, &[poly_a.clone(), poly_b.clone()], &stack).unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 9) as u64)).collect();
    let openings = [
        dense_opening(&evals_a, &opening_point),
        dense_opening(&evals_b, &opening_point),
    ];

    const TRANSCRIPT_LABEL: &[u8] = b"test/batched-prove";

    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let proof = Scheme::batched_prove::<_, _, _>(
        &setup,
        prover_claims(
            &opening_point[..],
            &poly_group[..],
            &commitments[0],
            hints.into_iter().next().unwrap(),
        ),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .unwrap();

    let mut bytes = Vec::new();
    let shape = proof.shape();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let proof = AkitaBatchedProof::<F, F>::deserialize_uncompressed(&*bytes, &shape).unwrap();

    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &openings[..], &commitments[0]),
        BasisMode::Lagrange,
    )
    .expect("batched verify should accept consistent openings");

    let mut wrong_openings = openings;
    wrong_openings[1] += F::one();
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let wrong_opening_result = Scheme::batched_verify(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &wrong_openings[..], &commitments[0]),
        BasisMode::Lagrange,
    );
    assert!(matches!(
        wrong_opening_result,
        Err(AkitaError::InvalidProof)
    ));

    let mut oversized_proof = proof.clone();
    {
        let fold = &mut oversized_proof.root;
        let mut oversized_v_coeffs = fold.v.coeffs().to_vec();
        oversized_v_coeffs.extend(vec![F::zero(); D]);
        fold.v = RingVec::from_coeffs(oversized_v_coeffs);
    }

    let mut oversized_openings = openings.to_vec();
    oversized_openings.push(F::zero());
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let oversized_result = Scheme::batched_verify(
        &oversized_proof,
        &verifier_setup,
        &mut verifier_transcript,
        verifier_claims(&opening_point[..], &oversized_openings[..], &commitments[0]),
        BasisMode::Lagrange,
    );

    assert!(matches!(oversized_result, Err(AkitaError::InvalidProof)));
}
