use super::*;
use akita_field::AkitaError;
use akita_prover::protocol::flow::{RecursiveCarriedOpening, RecursiveCarriedSource};
use akita_prover::AkitaProverSetup;
use akita_prover::{
    build_folded_batched_proof_with_suffix, commit_next_w, prepare_batched_prove_inputs,
    prove_root_fold_with_params, prove_suffix, RecursiveWitnessFlat,
};
use akita_types::{
    schedule_num_fold_levels, schedule_root_fold_step, AjtaiKeyParams, AkitaScheduleLookupKey,
    CarriedOpeningKind, CarriedOpeningProof, CarriedOpeningSourceProof, Schedule,
    SetupContributionMode, SetupMatrixEnvelope,
};

/// `num_vars` for the carried-batch exit test. nv=15 is too small after the
/// committed-fold A-role SIS pricing on main; nv=30 is the first planner hit.
const CARRIED_SUFFIX_TEST_NUM_VARS: usize = 30;

fn carried_suffix_test_num_vars() -> usize {
    carried_suffix_schedule(CARRIED_SUFFIX_TEST_NUM_VARS)
        .expect("carried suffix schedule for test num_vars");
    CARRIED_SUFFIX_TEST_NUM_VARS
}

/// Build a `[root fold, carried fold, direct]` schedule whose single recursive
/// fold consumes a two-claim carried batch (recursive witness + one extra
/// source). The production planner never emits this shape on its own, so the
/// carried-batch exit test drives it through the test-only verifier seam.
fn carried_suffix_schedule(num_vars: usize) -> Result<Schedule, AkitaError> {
    let incidence = ClaimIncidenceSummary::same_point(num_vars, 1)?;
    let base = Cfg::get_params_for_prove(&incidence)?;
    let root_step = base
        .steps
        .first()
        .cloned()
        .ok_or(AkitaError::InvalidProof)?;
    let Step::Fold(root_fold) = &root_step else {
        return Err(AkitaError::InvalidProof);
    };
    let root_next_w_len = root_fold.next_w_len;
    let root_log_basis = root_fold.params.log_basis;
    let root_level_bytes = root_fold.level_bytes;
    let next_params = scheduled_next_level_params(&base, 1)?;
    let carried_key = AkitaScheduleLookupKey::new_with_points(num_vars, 2, 2, 2, 2);
    let carried_suffix =
        akita_prover::dispatch_ring_dim_result!(next_params.ring_dimension, |D_LEVEL| {
            akita_config::test_support::recursive_carried_suffix_schedule::<
                akita_config::WCommitmentConfig<{ D_LEVEL }, Cfg>,
            >(carried_key, 1, root_next_w_len, root_log_basis)
        })?;
    if !matches!(carried_suffix.steps.first(), Some(Step::Fold(_))) {
        return Err(AkitaError::InvalidSetup(
            "carried batch needs a recursive fold to consume it".to_string(),
        ));
    }
    let mut steps = Vec::with_capacity(carried_suffix.steps.len() + 1);
    steps.push(root_step);
    steps.extend(carried_suffix.steps);
    Ok(Schedule {
        steps,
        total_bytes: root_level_bytes
            .checked_add(carried_suffix.total_bytes)
            .ok_or_else(|| AkitaError::InvalidSetup("carried schedule overflow".to_string()))?,
    })
}

/// Maximum A/B/D Ajtai footprint (in setup ring elements) used by one level.
fn level_setup_footprint(params: &LevelParams) -> usize {
    let key = |k: &AjtaiKeyParams| k.row_len().saturating_mul(k.col_len());
    key(&params.a_key)
        .max(key(&params.b_key))
        .max(key(&params.d_key))
}

/// Setup capacity (shared-matrix ring elements) large enough for every level
/// of a carried-batch schedule. The production envelope only plans singleton
/// recursion, so a two-source carried fold needs a larger recursive Ajtai key
/// than `Cfg::max_setup_matrix_size` reserves; this walks the actual schedule.
fn carried_setup_envelope(num_vars: usize, schedule: &Schedule) -> SetupMatrixEnvelope {
    let mut envelope = Cfg::max_setup_matrix_size(num_vars, 2, 2).expect("baseline setup envelope");
    let mut needed = envelope.max_setup_len;
    for step in &schedule.steps {
        match step {
            Step::Fold(fold) => needed = needed.max(level_setup_footprint(&fold.params)),
            Step::Direct(direct) => {
                if let Some(params) = direct.params.as_ref() {
                    needed = needed.max(level_setup_footprint(params));
                }
            }
        }
    }
    for level in 1..schedule_num_fold_levels(schedule) {
        if let Ok(next) = scheduled_next_level_params(schedule, level) {
            needed = needed.max(level_setup_footprint(&next));
        }
    }
    envelope.max_setup_len = needed;
    envelope
}

/// Prove a witness opening plus one dummy extra carried source through the
/// recursive fold, returning the assembled proof together with the schedule,
/// verifier setup, commitment, point, and opening needed to replay it.
///
/// The dummy source is an all-zero witness committed at the recursive level and
/// carried as a `SetupPrefix` claim opening to zero at the recursive-witness
/// point. It is genuinely folded and opened by the fold relation (claim 1), not
/// just serialized.
#[allow(clippy::type_complexity)]
fn prove_witness_plus_dummy_carried_batch(
    num_vars: usize,
) -> (
    AkitaBatchedProof<F, F>,
    Schedule,
    AkitaVerifierSetup<F>,
    RingCommitment<F, D>,
    Vec<F>,
    F,
) {
    let (poly, evals) = make_dense_poly(num_vars);

    let schedule = carried_suffix_schedule(num_vars).expect("carried suffix schedule");
    // Root fold + one recursive (terminal) carried fold.
    assert_eq!(schedule_num_fold_levels(&schedule), 2);

    // The production setup envelope only reserves space for singleton
    // recursion; the carried fold's two-source recursive commit needs a larger
    // shared matrix, so size capacity from the actual carried schedule.
    let envelope = carried_setup_envelope(num_vars, &schedule);
    let setup = AkitaProverSetup::<F, D>::generate_with_capacity(num_vars, 2, 2, envelope).unwrap();
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
    let setup_mode = SetupContributionMode::Direct;

    let mut prover_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let claims = vec![(
        &opening_point[..],
        CommittedPolynomials {
            polynomials: &poly_refs[..],
            commitment: &commitments[0],
            hint,
        },
    )];
    let prepared_claims =
        prepare_batched_prove_inputs::<F, F, _, D>(setup.expanded.as_ref(), claims).unwrap();
    let num_vars_inc = prepared_claims.incidence_summary.num_vars();

    akita_config::bind_transcript_instance_descriptor::<F, _, D, Cfg>(
        setup.expanded.as_ref(),
        &prepared_claims.incidence_summary,
        &schedule,
        BasisMode::Lagrange,
        &mut prover_transcript,
    )
    .unwrap();

    let root_step = schedule_root_fold_step(&schedule).unwrap().clone();
    let root_next_params = scheduled_next_level_params(&schedule, 1).unwrap();

    let mut root = prove_root_fold_with_params::<F, F, F, _, _, _, Cfg, D>(
        &setup.expanded,
        &setup.prefix_slots,
        &CpuBackend,
        &prepared,
        &mut prover_transcript,
        &prepared_claims.flat_polys,
        &prepared_claims.incidence_summary,
        &prepared_claims.opening_points,
        &prepared_claims.commitments_by_point,
        prepared_claims.flat_hints,
        &root_step.params,
        root_step.next_w_len,
        &root_next_params,
        BasisMode::Lagrange,
        setup_mode,
    )
    .unwrap();

    // Inject a dummy second carried source into the level-1 batch.
    let carried_claim = root.next_state.carried_openings[0].clone();
    let point = carried_claim.opening_point.clone();
    let dummy_logical_w = RecursiveWitnessFlat::from_i8_digits(vec![0; carried_claim.natural_len]);
    let dummy_next = commit_next_w::<Cfg, _, D>(
        &root_next_params,
        &setup.expanded,
        &CpuBackend,
        &prepared,
        &dummy_logical_w,
    )
    .unwrap();
    let dummy_commitment = dummy_next.commitment;
    assert_ne!(dummy_commitment, root.next_state.commitment);
    let (dummy_w, dummy_logical_w) = match dummy_next.witness {
        Some(committed_w) => (committed_w, Some(dummy_logical_w)),
        None => (dummy_logical_w, None),
    };
    // A genuine all-zero witness opens to zero at every point.
    let dummy_value = F::zero();
    root.extra_carried_sources.push(CarriedOpeningSourceProof {
        commitment: dummy_commitment.clone(),
    });
    root.extra_carried_openings.push(CarriedOpeningProof {
        source_idx: 1,
        point: point.clone(),
        value: dummy_value,
        basis: BasisMode::Lagrange,
        natural_len: dummy_w.len(),
        padded_len: carried_claim.padded_len,
        kind: CarriedOpeningKind::SetupPrefix,
    });
    root.next_state
        .extra_carried_sources
        .push(RecursiveCarriedSource {
            w: dummy_w.clone(),
            logical_w: dummy_logical_w,
            commitment: dummy_commitment,
            hint: dummy_next.hint,
        });
    root.next_state
        .carried_openings
        .push(RecursiveCarriedOpening {
            source_idx: 1,
            opening_point: point,
            opening: dummy_value,
            basis: BasisMode::Lagrange,
            natural_len: dummy_w.len(),
            padded_len: carried_claim.padded_len,
            kind: CarriedOpeningKind::SetupPrefix,
        });

    let (proof, _levels) =
        build_folded_batched_proof_with_suffix::<F, F, D, _>(root, |next_state| {
            prove_suffix::<Cfg, _, _, D>(
                &setup.expanded,
                &setup.prefix_slots,
                &CpuBackend,
                &prepared,
                num_vars_inc,
                &mut prover_transcript,
                next_state,
                &schedule,
                setup_mode,
            )
        })
        .unwrap();

    (
        proof,
        schedule,
        verifier_setup,
        commitment,
        opening_point,
        opening,
    )
}

#[test]
fn recursive_suffix_verifies_witness_plus_dummy_carried_batch() {
    let (proof, schedule, verifier_setup, commitment, opening_point, opening) =
        prove_witness_plus_dummy_carried_batch(carried_suffix_test_num_vars());

    let root = proof
        .root
        .as_fold()
        .expect("fixture should use a folded root");
    assert_eq!(root.stage2.extra_carried_sources.len(), 1);
    assert_eq!(root.stage2.extra_carried_openings.len(), 1);
    assert!(!proof.steps.is_empty());

    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    akita_verifier::verify_batched_with_schedule::<Cfg, _, D>(
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
        SetupContributionMode::Direct,
        &schedule,
    )
    .expect("witness plus dummy carried batch should verify");
}

#[test]
fn recursive_suffix_rejects_inconsistent_dummy_carried_opening() {
    let (mut proof, schedule, verifier_setup, commitment, opening_point, opening) =
        prove_witness_plus_dummy_carried_batch(carried_suffix_test_num_vars());

    // Tamper the carried claim's bound opening value: the verifier replays the
    // fold relation against the proof-visible value and must reject without
    // panicking (verifier no-panic contract).
    {
        let fold = proof.root.as_fold_mut().expect("folded root");
        fold.stage2.extra_carried_openings[0].value += F::one();
    }

    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut verifier_transcript = AkitaTranscript::<F>::new(b"test/prove");
    let result = akita_verifier::verify_batched_with_schedule::<Cfg, _, D>(
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
        SetupContributionMode::Direct,
        &schedule,
    );
    // Reaching this assertion (rather than panicking) already proves the
    // verifier honored the no-panic contract; it must also reject.
    assert!(
        result.is_err(),
        "verifier must reject an inconsistent carried opening, got {result:?}"
    );
}

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
