use super::*;

/// Final-group variables of the recursive multi-group round trip.
pub(crate) const RECURSIVE_ROUND_TRIP_NV: usize = 32;
/// Polynomials across the round trip's two singleton precommitted groups and
/// its two-polynomial final group.
pub(crate) const RECURSIVE_ROUND_TRIP_POLYS: usize = 4;

pub(crate) fn recursive_multi_group_round_trip<BaseCfg>(
    transcript_domain: &'static [u8],
    on_schedule: fn(&FoldSchedule),
) where
    BaseCfg: CommitmentConfig<Field = F, ExtField = F>
        + akita_config::recursive_commitment::RecursiveScheduleConfig,
{
    init_rayon_pool();
    run_on_large_stack(move || {
        let setup = load_workspace_scheme::<RecursiveCommitmentConfig<BaseCfg>>()
            .expect("workspace recursive schedule catalog")
            .setup_prover(RECURSIVE_ROUND_TRIP_NV, RECURSIVE_ROUND_TRIP_POLYS)
            .expect("recursive setup");
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        recursive_multi_group_round_trip_on::<BaseCfg>(
            &setup,
            &stack,
            transcript_domain,
            on_schedule,
        );
    });
}

/// Run the recursive multi-group round trip on a caller-built `setup` and
/// `stack`, which must cover `BaseCfg`'s recursive rows at
/// `(RECURSIVE_ROUND_TRIP_NV, RECURSIVE_ROUND_TRIP_POLYS)`.
pub(crate) fn recursive_multi_group_round_trip_on<BaseCfg>(
    setup: &AkitaProverSetup<F>,
    stack: &CpuBackend<F, F>,
    transcript_domain: &[u8],
    on_schedule: fn(&FoldSchedule),
) where
    BaseCfg: CommitmentConfig<Field = F, ExtField = F>
        + akita_config::recursive_commitment::RecursiveScheduleConfig,
{
    const PRE_NV: usize = 16;
    const FINAL_NV: usize = RECURSIVE_ROUND_TRIP_NV;
    const PRE_GROUPS: usize = 2;
    const PRE_GROUP_SIZE: usize = 1;
    const FINAL_GROUP_SIZE: usize = 2;
    const _: () =
        assert!(PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE == RECURSIVE_ROUND_TRIP_POLYS);

    let base_scheme = load_workspace_scheme::<BaseCfg>().expect("workspace base schedule catalog");
    let recursive_scheme = load_workspace_scheme::<RecursiveCommitmentConfig<BaseCfg>>()
        .expect("workspace recursive schedule catalog");
    let pre_key = PolynomialGroupLayout::new(PRE_NV, PRE_GROUP_SIZE);
    let pre_frozen = base_scheme
        .schedules()
        .resolve_key(&AkitaScheduleLookupKey::single(pre_key))
        .expect("independent profile")
        .profiles()
        .final_group;
    let schedule_key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(FINAL_NV, FINAL_GROUP_SIZE),
        precommitteds: vec![pre_frozen, pre_frozen],
    };
    let opening_layout = schedule_key.opening_layout().expect("opening layout");
    let schedule = recursive_scheme
        .schedules()
        .resolve_key(&schedule_key)
        .expect("recursive profile schedule resolves")
        .schedule()
        .clone();
    assert!(
        schedule_uses_setup_prefix(&schedule),
        "recursive profile must carry setup-prefix metadata"
    );
    on_schedule(&schedule);

    assert!(
        !setup.prefix_slots.is_empty(),
        "recursive setup must precompute setup-prefix slots for the generated profile"
    );

    let mut pre_polys_by_group = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    for group_idx in 0..PRE_GROUPS {
        let poly = make_onehot_poly::<BaseCfg>(PRE_NV, 0x0bee_fcaf_2026_0000 + group_idx as u64);
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = stack
            .commit(
                recursive_scheme.schedules(),
                &stack.import_source(vec![poly.clone()]).expect("source"),
                akita_cpu_backend::GroupContext::explicit(&pre_frozen),
            )
            .expect("precommit group");
        pre_polys_by_group.push(vec![poly]);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
    }

    let final_polys: Vec<OneHotPoly<F, u8>> = (0..FINAL_GROUP_SIZE)
        .map(|poly_idx| {
            make_onehot_poly::<BaseCfg>(FINAL_NV, 0x0bee_fcaf_2026_1000 + poly_idx as u64)
        })
        .collect();
    let precommitteds = PrecommittedGroupProfiles::from_ordered_groups(pre_commitments.iter())
        .expect("nonempty precommitted groups");
    let akita_cpu_backend::CommitOutput {
        committed_group: final_commitment,
        private_handle: final_hint,
    } = stack
        .commit(
            recursive_scheme.schedules(),
            &stack.import_source(final_polys.clone()).expect("source"),
            akita_cpu_backend::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )
        .expect("final generated-profile commitment");

    let point = random_point(FINAL_NV, 0xcafe_2026_0001);
    // Independent oracles: sums of Lagrange weights at the hot indices.
    let pre_openings: Vec<Vec<F>> = pre_polys_by_group
        .iter()
        .map(|polys| {
            polys
                .iter()
                .map(|poly| onehot_opening_lagrange(poly, &point[..PRE_NV]))
                .collect()
        })
        .collect();
    let final_openings: Vec<F> = final_polys
        .iter()
        .map(|poly| onehot_opening_lagrange(poly, &point))
        .collect();

    let mut prover_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        prover_groups.push(
            PolynomialGroupClaims::new(
                point[..PRE_NV].to_vec(),
                openings.clone(),
                pre_commitments[group_idx].clone(),
            )
            .expect("pre prover group"),
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            point.clone(),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    );

    let mut prover_hints = pre_hints;
    prover_hints.push(final_hint);

    let prover_claims = selected_prover_data::<RecursiveCommitmentConfig<BaseCfg>>(
        OpeningClaims::from_groups(prover_groups).expect("prover claims"),
        prover_hints,
        recursive_scheme.schedules(),
    );
    let selection = prover_claims.selection();

    let proof = recursive_scheme
        .batched_prove(
            setup,
            prover_claims,
            stack,
            transcript_domain,
            BasisMode::Lagrange,
        )
        .expect("generated-profile recursive proof");

    let verifier_setup = recursive_scheme
        .setup_verifier_for_schedule(setup, &schedule, &opening_layout)
        .expect("verifier setup");
    let verify_claims = |final_openings: Vec<F>| {
        let mut verifier_groups = Vec::new();
        for (group_idx, openings) in pre_openings.iter().enumerate() {
            verifier_groups.push(
                PolynomialGroupClaims::new(
                    point[..PRE_NV].to_vec(),
                    openings.clone(),
                    &pre_commitments[group_idx],
                )
                .expect("pre verifier group"),
            );
        }
        verifier_groups.push(
            PolynomialGroupClaims::new(point.clone(), final_openings, &final_commitment)
                .expect("final verifier group"),
        );
        let claims = OpeningClaims::from_groups(verifier_groups).expect("verifier claims");
        GroupBatchStatement::new(selection, claims).expect("verifier statement")
    };

    recursive_scheme
        .batched_verify(
            &proof,
            &verifier_setup,
            transcript_domain,
            verify_claims(final_openings.clone()),
            BasisMode::Lagrange,
        )
        .expect("generated-profile recursive verify");

    if let Some(alternate_verifier_setup) = verifier_setup_with_alternate_full_prefix(
        setup,
        &verifier_setup,
        &first_setup_prefix_slot(&schedule),
    ) {
        let alternate_result = recursive_scheme.batched_verify(
            &proof,
            &alternate_verifier_setup,
            transcript_domain,
            verify_claims(final_openings.clone()),
            BasisMode::Lagrange,
        );
        assert!(
            alternate_result.is_err(),
            "successor grouped opening must reject a full-prefix commitment whose active prefix agrees but tail differs"
        );
    }

    let reject_stage3_tamper = |tampered_proof: Vec<u8>, label: &str| {
        let result = recursive_scheme.batched_verify(
            &tampered_proof,
            &verifier_setup,
            transcript_domain,
            verify_claims(final_openings.clone()),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "{label} must be rejected without panicking"
        );
    };

    let mut tampered_claim = proof.clone();
    let claim_probe = tampered_claim.len() / 3;
    tampered_claim[claim_probe] ^= 1;
    reject_stage3_tamper(tampered_claim, "tampered Stage 3 claim");

    let mut tampered_prefix_eval = proof.clone();
    let prefix_probe = tampered_prefix_eval.len() / 2;
    tampered_prefix_eval[prefix_probe] ^= 1;
    reject_stage3_tamper(
        tampered_prefix_eval,
        "tampered Stage 3 setup-prefix evaluation",
    );

    let mut tampered_round = proof.clone();
    let round_probe = tampered_round.len() * 2 / 3;
    tampered_round[round_probe] ^= 1;
    reject_stage3_tamper(
        tampered_round,
        "tampered Stage 3 round polynomial and derived point",
    );

    let mut tampered = final_openings;
    tampered[0] += F::from_u128_reduced(1);
    let tampered_result = recursive_scheme.batched_verify(
        &proof,
        &verifier_setup,
        transcript_domain,
        verify_claims(tampered),
        BasisMode::Lagrange,
    );
    assert!(
        tampered_result.is_err(),
        "recursive verify must reject a tampered final opening"
    );
}
