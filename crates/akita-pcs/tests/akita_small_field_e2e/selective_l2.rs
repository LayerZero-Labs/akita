use super::*;

fn fp32_l2_onehot_poly(
    params: &CommittedGroupParams,
    seed: usize,
) -> akita_prover::OneHotPoly<fp32::Field, u8> {
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
    akita_prover::OneHotPoly::new(onehot_k, indices).expect("fp32 L2 one-hot polynomial")
}

fn encode_test_golomb_rice(values: &[i64], rice_low_bits: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut bit_position = 0usize;
    let mut write_bit = |bit: bool| {
        let byte_index = bit_position / 8;
        if byte_index == bytes.len() {
            bytes.push(0);
        }
        if bit {
            bytes[byte_index] |= 1 << (bit_position % 8);
        }
        bit_position += 1;
    };
    for &value in values {
        let zigzag = ((value << 1) ^ (value >> 63)) as u64;
        let quotient = zigzag >> rice_low_bits;
        for _ in 0..quotient {
            write_bit(true);
        }
        write_bit(false);
        let remainder = zigzag & ((1u64 << rice_low_bits) - 1);
        for bit in 0..rice_low_bits {
            write_bit((remainder >> bit) & 1 == 1);
        }
    }
    bytes
}

#[test]
fn fp32_ext4_l2_pcs_roundtrip_and_stage2_rejections() {
    type Cfg = fp32::OneHot;
    type F = fp32::Field;
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
            response_l2_sq_cap,
            norm_proof_shape,
            ..
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
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared L2 setup");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("L2 prover stack");
        let verifier_setup = scheme.setup_verifier(&setup).expect("L2 verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("L2 commitment");
        let poly_refs = [&poly];
        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![E::zero()],
            commitment.clone(),
        )
        .expect("L2 prover group")])
        .expect("L2 prover claims");
        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = scheme
            .batched_prove(
                &setup,
                selected_prover_data::<Cfg, _>(
                    prover_claims,
                    vec![hint],
                    vec![&poly_refs],
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("small-field L2 proof");
        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof
            .serialize_uncompressed(&mut bytes)
            .expect("serialize small-field L2 PCS proof");
        let proof = AkitaBatchedProof::<F, E>::deserialize_uncompressed(&bytes[..], &shape)
            .expect("deserialize small-field L2 PCS proof");
        let l2_index = proof
            .recursive_folds
            .iter()
            .position(|fold| fold.stage1.norm_proof.is_some())
            .expect("proof must carry the selected L2 norm");
        let norm_proof = proof.recursive_folds[l2_index]
            .stage1
            .norm_proof
            .as_ref()
            .expect("L2 norm proof");
        assert_eq!(
            norm_proof.subclaims.len(),
            norm_proof_shape.subclaim_count().expect("valid norm shape")
        );
        assert_eq!(
            norm_proof.virtual_evaluations.len(),
            norm_proof_shape.virtual_evaluation_count()
        );

        let verify = |candidate: &AkitaBatchedProof<F, E>| {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .expect("L2 verifier group")])
            .expect("L2 verifier claims");
            let mut transcript = AkitaTranscript::<F>::new(LABEL);
            scheme.batched_verify(
                candidate,
                &verifier_setup,
                &mut transcript,
                selected_statement::<Cfg>(claims, scheme.schedules()),
                BasisMode::Lagrange,
            )
        };
        verify(&proof).expect("verify serialized small-field L2 PCS proof");

        let mut over_cap = proof.clone();
        over_cap.recursive_folds[l2_index]
            .stage1
            .norm_proof
            .as_mut()
            .expect("L2 norm proof")
            .response_l2_sq = response_l2_sq_cap + 1;
        assert!(verify(&over_cap).is_err());

        if !norm_proof.subclaims.is_empty() {
            let mut bad_subclaim = proof.clone();
            bad_subclaim.recursive_folds[l2_index]
                .stage1
                .norm_proof
                .as_mut()
                .expect("L2 norm proof")
                .subclaims[0] += E::one();
            assert!(verify(&bad_subclaim).is_err());
        }

        let mut bad_virtual = proof.clone();
        bad_virtual.recursive_folds[l2_index]
            .stage1
            .norm_proof
            .as_mut()
            .expect("L2 norm proof")
            .virtual_evaluations[0] += E::one();
        assert!(verify(&bad_virtual).is_err());

        let mut bad_nonce = proof.clone();
        let mut nonce_bytes = bad_nonce.nonce_stream.as_bytes().to_vec();
        nonce_bytes[0] ^= 1;
        bad_nonce.nonce_stream = akita_types::TranscriptNonceStream::from_bytes(
            nonce_bytes,
            bad_nonce.nonce_stream.bit_len(),
        )
        .unwrap();
        assert!(verify(&bad_nonce).is_err());

        let mut bad_stage2 = proof;
        bad_stage2.recursive_folds[l2_index]
            .stage2
            .sumcheck_proof
            .round_polys[0]
            .coeffs_except_linear_term[0] += E::one();
        assert!(verify(&bad_stage2).is_err());
    });
}

#[test]
fn fp32_nv20_shipped_terminal_route_roundtrip_and_rejections() {
    type Cfg = fp32::OneHot;
    type F = fp32::Field;
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
        let response_l2_sq_cap = terminal_params.response_l2_sq_cap();

        let poly = fp32_l2_onehot_poly(&schedule.root.params, 9);
        let point = (0..NUM_VARS)
            .map(|i| E::from_u64((i as u64).wrapping_mul(5).wrapping_add(1)))
            .collect::<Vec<_>>();
        let opening = onehot_opening_lagrange(&poly, &point);
        let setup = scheme
            .setup_prover(NUM_VARS, 1)
            .expect("terminal L2 prover setup");
        let prepared = CpuBackend::DEFAULT
            .prepare_setup(&setup)
            .expect("prepared terminal L2 setup");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("terminal L2 prover stack");
        let verifier_setup = scheme
            .setup_verifier(&setup)
            .expect("terminal L2 verifier setup");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = scheme
            .commit(
                &setup,
                std::slice::from_ref(&poly),
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("terminal L2 commitment");
        let poly_refs = [&poly];
        let prover_claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
            point.clone(),
            vec![E::zero()],
            commitment.clone(),
        )
        .expect("terminal L2 prover group")])
        .expect("terminal L2 prover claims");
        let mut prover_transcript = AkitaTranscript::<F>::new(LABEL);
        let proof = scheme
            .batched_prove(
                &setup,
                selected_prover_data::<Cfg, _>(
                    prover_claims,
                    vec![hint],
                    vec![&poly_refs],
                    scheme.schedules(),
                ),
                &stack,
                &mut prover_transcript,
                BasisMode::Lagrange,
            )
            .expect("shipped terminal proof");

        let verify = |candidate: &AkitaBatchedProof<F, E>| {
            let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                point.clone(),
                vec![opening],
                &commitment,
            )
            .expect("terminal L2 verifier group")])
            .expect("terminal L2 verifier claims");
            let mut transcript = AkitaTranscript::<F>::new(LABEL);
            scheme.batched_verify(
                candidate,
                &verifier_setup,
                &mut transcript,
                selected_statement::<Cfg>(claims, scheme.schedules()),
                BasisMode::Lagrange,
            )
        };
        verify(&proof).expect("verify shipped terminal proof");

        let mut bad_nonce = proof.clone();
        let mut nonce_bytes = bad_nonce.nonce_stream.as_bytes().to_vec();
        nonce_bytes[0] ^= 1;
        bad_nonce.nonce_stream = akita_types::TranscriptNonceStream::from_bytes(
            nonce_bytes,
            bad_nonce.nonce_stream.bit_len(),
        )
        .unwrap();
        assert!(verify(&bad_nonce).is_err());

        let mut over_cap = proof;
        let group = *over_cap
            .terminal
            .terminal_response
            .layout
            .groups
            .first()
            .expect("single terminal group");
        let payload = over_cap
            .terminal
            .terminal_response
            .z_payloads
            .first_mut()
            .expect("terminal z payload");
        let mut values = akita_types::decode_terminal_z_golomb_payload(payload, &group)
            .expect("honest terminal z decode")
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>();
        if let Some(response_l2_sq_cap) = response_l2_sq_cap {
            assert!(
                group.z_linf_cap.is_none(),
                "terminal L2 routes must not enforce a separate Linf cap"
            );
            // Stay comfortably inside the signed terminal wire type while
            // making the complete decoded response exceed the scheduled cap.
            let coordinate = i64::from(i16::MAX / 2);
            let coordinate_sq = u128::try_from(coordinate * coordinate).expect("positive square");
            let mut forced_l2_sq = 0u128;
            for value in &mut values {
                *value = coordinate;
                forced_l2_sq += coordinate_sq;
                if forced_l2_sq > response_l2_sq_cap {
                    break;
                }
            }
            assert!(forced_l2_sq > response_l2_sq_cap);
        } else {
            let linf_cap = group
                .z_linf_cap
                .expect("terminal Linf routes must carry a coefficient cap");
            let over_linf = i64::try_from(
                linf_cap
                    .checked_add(1)
                    .expect("terminal Linf cap increment"),
            )
            .expect("terminal Linf cap fits the signed wire type");
            *values.first_mut().expect("nonempty terminal response") = over_linf;
        }
        *payload = encode_test_golomb_rice(&values, group.z_rice_low_bits);
        assert!(payload.len() <= group.z_payload_bytes);
        assert!(verify(&over_cap).is_err());
    });
}
