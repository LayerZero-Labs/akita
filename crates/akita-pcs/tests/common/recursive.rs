use super::*;

pub(crate) fn recursive_multi_group_round_trip<BaseCfg>(
    transcript_domain: &'static [u8],
    on_schedule: fn(&FoldSchedule),
) where
    BaseCfg: CommitmentConfig<Field = F, ExtField = F>
        + akita_config::recursive_commitment::RecursiveScheduleConfig,
{
    type Recursive<BaseCfg> = AkitaCommitmentScheme<RecursiveCommitmentConfig<BaseCfg>>;

    const PRE_NV: usize = 16;
    const FINAL_NV: usize = 32;
    const PRE_GROUPS: usize = 2;
    const PRE_GROUP_SIZE: usize = 1;
    const FINAL_GROUP_SIZE: usize = 2;
    const TOTAL_GROUP_SIZE: usize = PRE_GROUPS * PRE_GROUP_SIZE + FINAL_GROUP_SIZE;

    init_rayon_pool();
    run_on_large_stack(move || {
        let base_scheme =
            load_workspace_scheme::<BaseCfg>().expect("workspace base schedule catalog");
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

        let setup = recursive_scheme
            .setup_prover(FINAL_NV, TOTAL_GROUP_SIZE)
            .expect("recursive setup");
        assert!(
            !setup.prefix_slots.is_empty(),
            "recursive setup must precompute setup-prefix slots for the generated profile"
        );
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");

        let mut pre_polys_by_group = Vec::new();
        let mut pre_commitments = Vec::new();
        let mut pre_hints = Vec::new();
        for group_idx in 0..PRE_GROUPS {
            let poly =
                make_onehot_poly::<BaseCfg>(PRE_NV, 0x0bee_fcaf_2026_0000 + group_idx as u64);
            let akita_prover::CommitOutput {
                committed_group: commitment,
                prover_state: hint,
            } = base_scheme
                .commit(
                    &setup,
                    std::slice::from_ref(&poly),
                    stack.commitment(),
                    akita_prover::GroupContext::scheduler_without_precommitted_groups(),
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
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            prover_state: final_hint,
        } = recursive_scheme
            .commit(
                &setup,
                &final_polys,
                stack.commitment(),
                akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
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

        let pre_refs_by_group: Vec<Vec<&OneHotPoly<F, u8>>> = pre_polys_by_group
            .iter()
            .map(|polys| polys.iter().collect())
            .collect();
        let final_refs: Vec<&OneHotPoly<F, u8>> = final_polys.iter().collect();

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

        let mut prover_polys: Vec<&[&OneHotPoly<F, u8>]> = Vec::new();
        for refs in &pre_refs_by_group {
            prover_polys.push(&refs[..]);
        }
        prover_polys.push(&final_refs[..]);
        let mut prover_hints = pre_hints;
        prover_hints.push(final_hint);

        let prover_claims = selected_prover_data::<RecursiveCommitmentConfig<BaseCfg>, _>(
            OpeningClaims::from_groups(prover_groups).expect("prover claims"),
            prover_hints,
            prover_polys,
            recursive_scheme.schedules(),
        );
        let selection = prover_claims.selection();

        let proof = recursive_scheme
            .batched_prove(
                &setup,
                prover_claims,
                &stack,
                transcript_domain,
                BasisMode::Lagrange,
            )
            .expect("generated-profile recursive proof");

        let verifier_setup = recursive_scheme
            .setup_verifier_for_schedule(&setup, &schedule, &opening_layout)
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
            &setup,
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
    });
}
