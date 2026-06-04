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
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
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
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
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
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
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
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
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
        )
    }));

    assert!(matches!(result, Ok(Err(AkitaError::InvalidProof))));
}

#[test]
fn fp128_degree_one_batched_proof_roundtrip_is_stable() {
    let (verifier_setup, commitment, proof, opening_point, opening, _layout) =
        make_verify_fixture(16);
    let (_, _, same_proof, _, _, _) = make_verify_fixture(16);
    let shape = proof.shape();
    assert_eq!(shape, same_proof.shape());

    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let mut same_bytes = Vec::new();
    same_proof.serialize_uncompressed(&mut same_bytes).unwrap();
    #[cfg(not(feature = "zk"))]
    assert_eq!(bytes, same_bytes);

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
        &decoded,
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
    )
    .expect("degree-one roundtrip proof should verify");
}

fn schedule_with_setup_prefix_carried_suffix(
    schedule: akita_types::Schedule,
    num_vars: usize,
) -> Result<akita_types::Schedule, AkitaError> {
    let root_step = schedule
        .steps
        .first()
        .cloned()
        .ok_or(AkitaError::InvalidProof)?;
    let akita_types::Step::Fold(root_fold) = &root_step else {
        return Ok(schedule);
    };
    let root_next_w_len = root_fold.next_w_len;
    let root_log_basis = root_fold.params.log_basis;
    let root_level_bytes = root_fold.level_bytes;
    let next_params = akita_types::scheduled_next_level_params(&schedule, 1)?;
    let carried_key = akita_types::AkitaScheduleLookupKey::new_with_points(num_vars, 2, 2, 2, 2);
    let carried_suffix = akita_config::test_support::recursive_carried_suffix_schedule::<Cfg>(
        next_params.ring_dimension,
        carried_key,
        1,
        root_next_w_len,
        root_log_basis,
    )?;
    if !matches!(
        carried_suffix.steps.first(),
        Some(akita_types::Step::Fold(_))
    ) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix carry needs a recursive fold to consume it".to_string(),
        ));
    }
    let mut steps = Vec::with_capacity(carried_suffix.steps.len() + 1);
    steps.push(root_step);
    steps.extend(carried_suffix.steps);
    Ok(akita_types::Schedule {
        steps,
        total_bytes: root_level_bytes
            .checked_add(carried_suffix.total_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("carried schedule size overflow".into()))?,
    })
}

#[test]
fn recursive_suffix_verifies_witness_plus_dummy_setup_carried_batch() {
    let num_vars = 15;
    let (poly, evals) = make_dense_poly(num_vars);
    let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(num_vars, 2, 2).unwrap();
    let prepared = CpuBackend.prepare_setup(&setup).unwrap();
    let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <Scheme as CommitmentProver<F, D>>::commit(
        &setup,
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

    let opening_point: Vec<F> = (0..num_vars).map(|i| F::from_u64((i + 2) as u64)).collect();
    let opening = dense_opening(&evals, &opening_point);
    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment.clone()];
    let expanded_arc = setup.expanded.clone();
    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let proof =
        akita_prover::prove_batched_with_policy::<F, F, F, _, _, D, _, _, _, _, _>(
            setup.expanded.as_ref(),
            vec![(
                &opening_point[..],
                CommittedPolynomials {
                    polynomials: &poly_refs[..],
                    commitment: &commitments[0],
                    hint,
                },
            )],
            &mut prover_transcript,
            BasisMode::Lagrange,
            |incidence_summary| {
                let schedule = Cfg::get_params_for_prove(incidence_summary)?;
                schedule_with_setup_prefix_carried_suffix(schedule, incidence_summary.num_vars())
            },
            Cfg::get_params_for_batched_commitment,
            |schedule, _next_inputs| akita_types::scheduled_next_level_params(schedule, 1),
            |transcript, incidence_summary, schedule, basis| {
                akita_config::bind_transcript_instance_descriptor::<F, _, D, Cfg>(
                    setup.expanded.as_ref(),
                    incidence_summary,
                    schedule,
                    basis,
                    transcript,
                )
            },
            |prepared_claims, schedule, next_params, transcript, basis| {
                let num_vars = prepared_claims.incidence_summary.num_vars();
                akita_prover::prove_folded_batched_with_policy::<F, F, F, _, _, _, D, _, _, _>(
                setup.expanded.as_ref(),
                &CpuBackend,
                &prepared,
                transcript,
                prepared_claims,
                &schedule,
                basis,
                &next_params,
                |w| {
                    akita_prover::commit_next_w_with_policy::<F, F, _, _, _, D>(
                        &next_params,
                        &expanded_arc,
                        &CpuBackend,
                        &prepared,
                        w,
                        |params, current_w_len| {
                            akita_types::recursive_level_layout_from_params(
                                params,
                                current_w_len,
                                Cfg::decomposition(),
                                Cfg::ring_subfield_embedding_norm_bound(),
                            )
                        },
                        recursive_w_commit_layout_for_d::<Cfg>,
                    )
                },
                |next_state, schedule, transcript| {
                    prove_recursive_suffix::<F, _, _, D, Cfg>(
                        &expanded_arc,
                        &CpuBackend,
                        &prepared,
                        num_vars,
                        transcript,
                        next_state,
                        schedule,
                    )
                },
                |raw| {
                    let carried_claim = raw.next_state.carried_openings[0].clone();
                    let point = carried_claim.opening_point.clone();
                    let dummy_logical_w = akita_prover::RecursiveWitnessFlat::from_i8_digits(
                        vec![0; carried_claim.natural_len],
                    );
                    let dummy_next = akita_prover::commit_next_w_with_policy::<F, F, _, _, _, D>(
                        &next_params,
                        &expanded_arc,
                        &CpuBackend,
                        &prepared,
                        &dummy_logical_w,
                        |params, current_w_len| {
                            akita_types::recursive_level_layout_from_params(
                                params,
                                current_w_len,
                                Cfg::decomposition(),
                                Cfg::ring_subfield_embedding_norm_bound(),
                            )
                        },
                        recursive_w_commit_layout_for_d::<Cfg>,
                    )?;
                    let dummy_commitment = dummy_next.commitment;
                    assert_ne!(dummy_commitment, raw.next_state.commitment);
                    let (dummy_w, dummy_logical_w) = match dummy_next.witness {
                        Some(committed_w) => (committed_w, Some(dummy_logical_w)),
                        None => (dummy_logical_w, None),
                    };
                    raw.extra_carried_sources
                        .push(akita_types::CarriedOpeningSourceProof {
                            commitment: dummy_commitment.clone(),
                        });
                    raw.extra_carried_openings
                        .push(akita_types::CarriedOpeningProof {
                            source_idx: 1,
                            point: point.clone(),
                            value: F::zero(),
                            basis: BasisMode::Lagrange,
                            natural_len: dummy_w.len(),
                            padded_len: carried_claim.padded_len,
                            kind: akita_types::CarriedOpeningKind::SetupPrefix,
                        });
                    raw.next_state.extra_carried_sources.push(
                        akita_prover::RecursiveCarriedSource {
                            w: dummy_w.clone(),
                            logical_w: dummy_logical_w,
                            commitment: dummy_commitment,
                            hint: dummy_next.hint,
                        },
                    );
                    raw.next_state
                        .carried_openings
                        .push(akita_prover::RecursiveCarriedOpening {
                            source_idx: 1,
                            opening_point: point,
                            opening: F::zero(),
                            basis: BasisMode::Lagrange,
                            natural_len: dummy_w.len(),
                            padded_len: carried_claim.padded_len,
                            kind: akita_types::CarriedOpeningKind::SetupPrefix,
                        });
                    Ok(())
                },
            )
            .map(|(proof, _)| proof)
            },
        )
        .unwrap();

    let root = proof
        .root
        .as_fold()
        .expect("fixture should use folded root");
    assert_eq!(root.stage2.extra_carried_sources.len(), 1);
    assert_eq!(root.stage2.extra_carried_openings.len(), 1);
    assert!(!proof.steps.is_empty());

    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    akita_verifier::verify_batched_with_policy::<F, F, F, _, D, _, _, _, _, _>(
        &proof,
        &verifier_setup,
        &mut verifier_transcript,
        vec![(
            &opening_point[..],
            CommittedOpenings {
                openings: opening_groups[0],
                commitment: &commitment,
            },
        )],
        BasisMode::Lagrange,
        |incidence_summary| {
            let schedule = Cfg::get_params_for_prove(incidence_summary)?;
            schedule_with_setup_prefix_carried_suffix(schedule, incidence_summary.num_vars())
        },
        |schedule, _next_inputs| akita_types::scheduled_next_level_params(schedule, 1),
        Cfg::get_params_for_batched_commitment,
        |transcript, incidence_summary, schedule, basis| {
            akita_config::bind_transcript_instance_descriptor::<F, _, D, Cfg>(
                &verifier_setup.expanded,
                incidence_summary,
                schedule,
                basis,
                transcript,
            )
        },
        |witnesses, setup, commitments, incidence_summary, params, direct_commitment_payload| {
            akita_verifier::verify_root_direct_commitments_with_params::<F, D>(
                witnesses,
                setup,
                commitments,
                incidence_summary,
                params,
                direct_commitment_payload,
            )
        },
    )
    .expect("witness plus setup-prefix carried batch should verify");
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
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
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
    let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Monomial,
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
        &CpuBackend,
        &prepared,
        std::slice::from_ref(&poly),
    )
    .unwrap();

    let poly_refs: [&DensePoly<DirectF, DIRECT_D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    let mut prover_transcript = AkitaTranscript::<DirectF>::new(b"test/tiny-direct");
    let proof = <DirectScheme as CommitmentProver<DirectF, DIRECT_D>>::batched_prove(
        &setup,
        &CpuBackend,
        &prepared,
        vec![(
            &opening_point[..],
            CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            },
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
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
    )
    .unwrap();
}
