use super::*;

#[test]
#[cfg(not(feature = "zk"))]
fn batched_commit_matches_individual_commits() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 1) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 3 + 7) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let poly_groups = [std::slice::from_ref(&poly_a), std::slice::from_ref(&poly_b)];

    let (batched_commitments, batched_hints): (Vec<_>, Vec<_>) = poly_groups
        .iter()
        .map(|group| {
            <Scheme as CommitmentProver<F, D>>::commit(
                &setup,
                RootCommitPolys::new(group),
                &CpuBackend,
                &prepared,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .unzip();
    let (commitment_a, hint_a) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::from_ref(&poly_a),
        &CpuBackend,
        &prepared,
    )
    .unwrap();
    let (commitment_b, hint_b) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::from_ref(&poly_b),
        &CpuBackend,
        &prepared,
    )
    .unwrap();

    assert_eq!(batched_commitments, vec![commitment_a, commitment_b]);
    assert_eq!(batched_hints, vec![hint_a, hint_b]);
}

/// Exercise the batched root-direct fast path: for a layout/batch shape
/// whose offline-planned schedule has zero fold levels, the prover must
/// emit a [`AkitaBatchedRootProof::ZeroFold`] variant with no recursive
/// suffix, and the verifier must accept it via the batched root-direct
/// checks (per-claim opening + joint per-group re-commit).
#[test]
fn batched_root_direct_fast_path_round_trip() {
    // For Cfg = fp128::D64Full with num_t_vectors = 4 and a same-
    // point batch of 4 claims, the generated schedule table is
    // direct-only up to num_vars = 12.
    const NUM_VARS: usize = 8;
    const NUM_POLYS: usize = 4;

    let len = 1usize << NUM_VARS;
    let polys: Vec<DensePoly<F, D>> = (0..NUM_POLYS)
        .map(|poly_idx| {
            let evals: Vec<F> = (0..len)
                .map(|i| F::from_u64((i * (poly_idx + 1) + 17) as u64))
                .collect();
            DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).unwrap()
        })
        .collect();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::new(&polys),
        &CpuBackend,
        &prepared,
    )
    .unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 3) as u64)).collect();
    let openings: Vec<F> = polys
        .iter()
        .map(|poly| {
            let mut evals = vec![F::zero(); len];
            for (i, ring) in poly.coeffs.iter().enumerate() {
                let base = i * D;
                let take = (len.saturating_sub(base)).min(D);
                if take == 0 {
                    break;
                }
                evals[base..base + take].copy_from_slice(&ring.coefficients()[..take]);
            }
            let lw = lagrange_weights(&opening_point).unwrap();
            evals
                .iter()
                .zip(lw.iter())
                .fold(F::zero(), |a, (&c, &w)| a + c * w)
        })
        .collect();

    let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &CpuBackend,
        &prepared,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched root-direct prove");

    assert!(
        proof.is_root_direct(),
        "expected a root-direct batched proof at num_vars={NUM_VARS}, num_t_vectors={NUM_POLYS}"
    );
    let direct_witnesses = proof
        .root
        .as_zero_fold()
        .expect("root-direct variant must expose per-claim direct witnesses");
    assert_eq!(direct_witnesses.len(), NUM_POLYS);
    assert!(
        proof.steps.is_empty(),
        "root-direct batched proof must not carry recursive-suffix steps"
    );

    let mut bytes = Vec::new();
    let shape = proof.shape();
    assert!(matches!(shape, AkitaBatchedProofShape::ZeroFold { .. }));
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let round_trip = AkitaBatchedProof::<F, F>::deserialize_uncompressed(&*bytes, &shape).unwrap();
    assert_eq!(round_trip, proof);

    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct");
    let opening_groups = [&openings[..]];
    <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &round_trip,
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
    .expect("batched root-direct verify");
}

/// The verifier must reject a root-direct batched proof whose
/// per-claim direct witnesses disagree with the claimed opening.
#[test]
fn batched_root_direct_rejects_wrong_opening() {
    const NUM_VARS: usize = 8;
    const NUM_POLYS: usize = 4;
    let len = 1usize << NUM_VARS;
    let polys: Vec<DensePoly<F, D>> = (0..NUM_POLYS)
        .map(|poly_idx| {
            let evals: Vec<F> = (0..len)
                .map(|i| F::from_u64((i + poly_idx + 11) as u64))
                .collect();
            DensePoly::<F, D>::from_field_evals(NUM_VARS, &evals).unwrap()
        })
        .collect();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NUM_VARS, NUM_POLYS, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::new(&polys),
        &CpuBackend,
        &prepared,
    )
    .unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..NUM_VARS).map(|i| F::from_u64((i + 2) as u64)).collect();
    let openings: Vec<F> = (0..NUM_POLYS).map(|_| F::from_u64(999_999)).collect();

    let poly_group = [&polys[0], &polys[1], &polys[2], &polys[3]];

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_group[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &CpuBackend,
        &prepared,
        &mut prover_transcript,
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    )
    .expect("batched root-direct prove");
    assert!(proof.is_root_direct());

    let mut verifier_transcript =
        AkitaTranscript::<F>::new(b"test/batched-root-direct-bad-opening");
    let opening_groups = [&openings[..]];
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
    assert!(result.is_err(), "verifier must reject bogus openings");
}

#[test]
fn batched_verify_accepts_consistent_openings_and_rejects_bad_inputs() {
    let alpha = D.trailing_zeros() as usize;
    let layout = singleton_layout::<Cfg>(16);
    let num_vars = layout.m_vars + layout.r_vars + alpha;
    let len = 1usize << num_vars;
    let evals_a: Vec<F> = (0..len).map(|i| F::from_u64((i + 5) as u64)).collect();
    let evals_b: Vec<F> = (0..len).map(|i| F::from_u64((i * 7 + 3) as u64)).collect();
    let poly_a = DensePoly::<F, D>::from_field_evals(num_vars, &evals_a).unwrap();
    let poly_b = DensePoly::<F, D>::from_field_evals(num_vars, &evals_b).unwrap();
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 1).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let poly_group = [poly_a, poly_b];
    let poly_refs = [&poly_group[0], &poly_group[1]];
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        RootCommitPolys::new(&poly_group),
        &CpuBackend,
        &prepared,
    )
    .unwrap();
    let commitments = [commitment];
    let hints = vec![hint];

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 9) as u64)).collect();
    let openings = [
        dense_opening(&evals_a, &opening_point),
        dense_opening(&evals_b, &opening_point),
    ];

    const TRANSCRIPT_LABEL: &[u8] = b"test/batched-prove";

    let mut prover_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint: hints.into_iter().next().unwrap(),
            },
        )],
        &CpuBackend,
        &prepared,
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
    let opening_groups = [&openings[..]];
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
    .expect("batched verify should accept consistent openings");

    let mut wrong_openings = openings;
    wrong_openings[1] += F::one();
    let wrong_opening_groups = [&wrong_openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let wrong_opening_result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
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
    assert!(matches!(
        wrong_opening_result,
        Err(AkitaError::InvalidProof)
    ));

    let mut oversized_proof = proof.clone();
    {
        let fold = oversized_proof
            .root
            .as_fold_mut()
            .expect("oversized-y-rings test expects a fold-rooted batched proof");
        let mut oversized_y_coeffs = fold.y_rings.coeffs().to_vec();
        oversized_y_coeffs.extend(vec![F::zero(); D]);
        fold.y_rings = FlatRingVec::from_coeffs(oversized_y_coeffs);
    }

    let mut oversized_openings = openings.to_vec();
    oversized_openings.push(F::zero());
    let oversized_opening_groups = [&oversized_openings[..]];

    let mut verifier_transcript = AkitaTranscript::<F>::new(TRANSCRIPT_LABEL);
    let oversized_result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
        &oversized_proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: oversized_opening_groups[0],
                commitment: &commitments[0],
            },
        )],
        BasisMode::Lagrange,
        akita_types::SetupContributionMode::Direct,
    );

    assert!(matches!(oversized_result, Err(AkitaError::InvalidProof)));
}
