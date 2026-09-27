use super::*;

fn fp32_l2_onehot_poly(
    params: &CommittedGroupParams,
    seed: usize,
) -> akita_cpu_backend::OneHotPoly<fp32::Field, u8> {
    let onehot_k = akita_config::unit_onehot_source_chunk_size::<fp32::OneHot>()
        .expect("fp32 one-hot fixture requires a unit-one-hot config");
    let total_field = params
        .blocks()
        .live_blocks
        .checked_mul(params.blocks().positions_per_block)
        .and_then(|count| count.checked_mul(params.d_a()))
        .expect("fp32 L2 fixture length");
    assert_eq!(total_field % onehot_k, 0);
    let indices = (0..total_field / onehot_k)
        .map(|chunk| Some(((chunk * 29 + seed * 41 + 7) % onehot_k) as u8))
        .collect();
    akita_cpu_backend::OneHotPoly::new(onehot_k, indices).expect("fp32 L2 one-hot polynomial")
}

#[test]
fn fp32_ext4_l2_pcs_roundtrip_and_stage2_rejections() {
    type Cfg = fp32::OneHot;
    type E = fp32::ExtensionField;
    const NUM_VARS: usize = 28;
    const LABEL: &[u8] = b"test/fp32-ext4-multiblock-l2-pcs";

    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");
        let opening_layout = OpeningClaimsLayout::new(NUM_VARS, 1).expect("L2 opening layout");
        let schedule = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(
                opening_layout
                    .root_final_group_layout()
                    .expect("singleton group layout"),
            ))
            .expect("shipped L2 schedule")
            .schedule()
            .clone();
        let l2_step = schedule
            .recursive_folds
            .iter()
            .find(|step| {
                matches!(
                    step.params.inner().matrix.security_route(),
                    akita_types::InnerCommitSecurityRoute::L2 { .. }
                )
            })
            .expect("schedule-selected small-field L2 fold");
        assert_eq!(l2_step.params.d_a(), 128);
        assert_eq!(
            l2_step.params.fold_challenge_config(),
            akita_challenges::D128_SELECTIVE_L2_CHALLENGE_CONFIG,
        );
        assert_eq!(
            akita_challenges::selective_l2_operator_norm_rejection(
                128,
                &l2_step.params.fold_challenge_config(),
            ),
            Some(akita_challenges::OperatorNormRejection::D128_SELECTIVE_L2),
        );
        let akita_types::InnerCommitSecurityRoute::L2 {
            norm_proof_shape, ..
        } = l2_step.params.inner().matrix.security_route()
        else {
            unreachable!("selected route checked above")
        };
        norm_proof_shape
            .validate()
            .expect("shipped small-field norm-proof shape");

        let poly = fp32_l2_onehot_poly(&schedule.root.params, 3);
        let point = (0..NUM_VARS)
            .map(|i| E::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        let opening = onehot_opening_lagrange(&poly, &point);
        let setup = scheme.setup_prover(NUM_VARS, 1).expect("L2 prover setup");
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        let verifier_setup = scheme.setup_verifier(&setup).expect("L2 verifier setup");
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = stack
            .commit(
                scheme.schedules(),
                &stack.import_source(vec![poly.clone()]).expect("source"),
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("L2 commitment");

        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![opening],
            commitment.clone(),
        )
        .expect("L2 prover group")])
        .expect("L2 prover claims");
        let proof = scheme
            .batched_prove(
                &setup,
                selected_prover_data::<Cfg>(prover_claims, vec![hint], scheme.schedules()),
                &stack,
                LABEL,
                BasisMode::Lagrange,
            )
            .expect("small-field L2 proof");

        let verify = |candidate: &[u8]| {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .expect("L2 verifier group")])
            .expect("L2 verifier claims");
            scheme
                .verifier(verifier_setup.clone())
                .and_then(|verifier| {
                    verifier.batched_verify(
                        candidate,
                        LABEL,
                        selected_statement::<Cfg>(claims, scheme.schedules()),
                        BasisMode::Lagrange,
                    )
                })
        };
        verify(&proof).expect("verify native small-field L2 PCS proof");
        for offset in [0, proof.len() / 3, proof.len() * 2 / 3, proof.len() - 1] {
            let mut mutated = proof.clone();
            mutated[offset] ^= 1;
            assert!(verify(&mutated).is_err());
        }
        assert!(verify(&proof[..proof.len() - 1]).is_err());
    });
}

#[test]
fn fp32_nv20_shipped_terminal_route_roundtrip_and_rejections() {
    type Cfg = fp32::OneHot;
    type E = fp32::ExtensionField;
    const NUM_VARS: usize = 20;
    const LABEL: &[u8] = b"test/fp32-nv20-shipped-terminal-route";

    init_rayon_pool();
    run_on_large_stack(|| {
        let scheme = load_workspace_scheme::<Cfg>().expect("workspace schedule catalog");
        let opening_layout = OpeningClaimsLayout::new(NUM_VARS, 1).expect("terminal L2 layout");
        let schedule = scheme
            .schedules()
            .resolve_key(&AkitaScheduleLookupKey::single(
                opening_layout
                    .root_final_group_layout()
                    .expect("singleton group layout"),
            ))
            .expect("shipped fp32 schedule")
            .schedule()
            .clone();
        let terminal_params = &schedule.terminal;
        assert!(
            terminal_params.response_l2_sq_cap().is_some()
                || terminal_params.response_shape.layout.groups[0]
                    .z_linf_cap
                    .is_some(),
            "terminal route must enforce an L2 or Linf response bound"
        );

        let poly = fp32_l2_onehot_poly(&schedule.root.params, 9);
        let point = (0..NUM_VARS)
            .map(|i| E::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        let opening = onehot_opening_lagrange(&poly, &point);
        let setup = scheme
            .setup_prover(NUM_VARS, 1)
            .expect("terminal L2 prover setup");
        let stack = CpuBackend::new(setup.expanded.clone()).expect("backend");
        let verifier_setup = scheme
            .setup_verifier(&setup)
            .expect("terminal L2 verifier setup");
        let akita_cpu_backend::CommitOutput {
            committed_group: commitment,
            private_handle: hint,
        } = stack
            .commit(
                scheme.schedules(),
                &stack.import_source(vec![poly.clone()]).expect("source"),
                akita_cpu_backend::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("terminal L2 commitment");

        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![opening],
            commitment.clone(),
        )
        .expect("terminal L2 prover group")])
        .expect("terminal L2 prover claims");
        let proof = scheme
            .batched_prove(
                &setup,
                selected_prover_data::<Cfg>(prover_claims, vec![hint], scheme.schedules()),
                &stack,
                LABEL,
                BasisMode::Lagrange,
            )
            .expect("shipped terminal proof");

        let verify = |candidate: &[u8]| {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .expect("terminal L2 verifier group")])
            .expect("terminal L2 verifier claims");
            scheme
                .verifier(verifier_setup.clone())
                .and_then(|verifier| {
                    verifier.batched_verify(
                        candidate,
                        LABEL,
                        selected_statement::<Cfg>(claims, scheme.schedules()),
                        BasisMode::Lagrange,
                    )
                })
        };
        verify(&proof).expect("verify shipped terminal proof");
        for offset in [0, proof.len() / 2, proof.len() - 1] {
            let mut mutated = proof.clone();
            mutated[offset] ^= 1;
            assert!(verify(&mutated).is_err());
        }
        assert!(verify(&proof[..proof.len() - 1]).is_err());
    });
}
