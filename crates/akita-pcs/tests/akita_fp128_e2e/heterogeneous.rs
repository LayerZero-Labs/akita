use super::*;

// ============================================================================
// GROUP E — Heterogeneous configurations (fp128)
//
// Tests that span multiple commitment groups with different polynomial types or
// compute backends.  Orthogonal to the Group B matrix.
// ============================================================================

// fp128: three commitment groups with heterogeneous polynomial types
// (one-hot precommit + dense precommit + one-hot final), proved jointly.
// This is the key test for the heterogeneous-group code path.
#[test]
fn heterogeneous_group_types() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const ONEHOT_PRE_NV: usize = 14;
        const DENSE_PRE_NV: usize = 15;
        const FINAL_NV: usize = 16;

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(FINAL_NV, 4).expect("setup");
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");
        let stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("stack");

        // Derive the OneHot pre-commit ring_d from the row without prior
        // groups, so the polynomial matches what `commit` selects below.
        let pre_d = OneHotCfg::profile_without_precommitted_groups(
            akita_types::PolynomialGroupLayout::unit_one_hot(ONEHOT_PRE_NV, 1, 256),
        )
        .expect("onehot pre profile without precommitted groups")
        .inner_commit_matrix
        .ring_dimension();
        let onehot_k_pre = 256usize;
        let pre_chunks = (1usize << ONEHOT_PRE_NV) / onehot_k_pre;
        let onehot_pre = akita_prover::OneHotPoly::<F, u8>::new(
            onehot_k_pre,
            pre_d,
            (0..pre_chunks)
                .map(|i| (i % 3 == 0).then_some((i % onehot_k_pre) as u8))
                .collect(),
        )
        .expect("K=256 precommitted poly");

        let dense_evals_a = (0..(1usize << DENSE_PRE_NV))
            .map(|i| F::from_u64((i % 257) as u64))
            .collect::<Vec<_>>();
        let dense_evals_b = (0..(1usize << DENSE_PRE_NV))
            .map(|i| F::from_u64((i % 509) as u64))
            .collect::<Vec<_>>();
        let dense_a =
            akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, DENSE_D, &dense_evals_a)
                .expect("dense a");
        let dense_b =
            akita_prover::DensePoly::from_field_evals(DENSE_PRE_NV, DENSE_D, &dense_evals_b)
                .expect("dense b");

        let final_onehot = make_onehot_poly(FINAL_NV, 0x1701_0000);

        let dense_polys = [dense_a.clone(), dense_b.clone()];
        let final_polys = [MultilinearPolynomial::onehot(final_onehot.clone())];

        // OneHot pre-group committed with OneHotCfg (matches catalog descriptor[0]).
        let akita_prover::CommitOutput {
            committed_group: onehot_pre_commitment,
            hint: onehot_pre_hint,
        } = AkitaCommitmentScheme::<OneHotCfg>::commit(
            &setup,
            std::slice::from_ref(&onehot_pre),
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("K=256 precommit");

        // Dense pre-group committed with DenseCfg so its profile matches the
        // Dense descriptor in catalog entry {final_nv=16, pre=[onehot(14,1), dense(15,2)]}.
        let dense_setup =
            AkitaCommitmentScheme::<DenseCfg>::setup_prover(DENSE_PRE_NV, 2).expect("dense setup");
        let dense_prepared = CpuBackend::DEFAULT
            .prepare_setup(&dense_setup)
            .expect("dense prepared");
        let dense_stack = UniformProverStack::uniform(
            &CpuBackend::DEFAULT,
            &dense_prepared,
            dense_setup.expanded.as_ref(),
        )
        .expect("dense stack");
        let akita_prover::CommitOutput {
            committed_group: dense_commitment,
            hint: dense_hint,
        } = AkitaCommitmentScheme::<DenseCfg>::commit(
            &dense_setup,
            &dense_polys,
            &dense_stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("dense precommit");

        let precommitteds = PrecommittedGroupProfiles::from_profiles(vec![
            onehot_pre_commitment.profile,
            dense_commitment.profile,
        ])
        .expect("nonempty precommitted groups");
        let akita_prover::CommitOutput {
            committed_group: final_commitment,
            hint: final_hint,
        } = AkitaCommitmentScheme::<OneHotCfg>::commit(
            &setup,
            &final_polys,
            &stack,
            akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )
        .expect("final commit");

        let onehot_pre_point: Vec<F> = (0..ONEHOT_PRE_NV)
            .map(|i| F::from_u64((i + 2) as u64))
            .collect();
        let dense_point: Vec<F> = (0..DENSE_PRE_NV)
            .map(|i| F::from_u64((i + 37) as u64))
            .collect();
        let final_point: Vec<F> = (0..FINAL_NV)
            .map(|i| F::from_u64((i + 71) as u64))
            .collect();

        // Independent oracles for every group in the heterogeneous batch.
        let onehot_pre_opening = onehot_opening_lagrange(&onehot_pre, &onehot_pre_point);
        let dense_opening_a = dense_opening_lagrange(&dense_evals_a, &dense_point);
        let dense_opening_b = dense_opening_lagrange(&dense_evals_b, &dense_point);
        let final_opening = onehot_opening_lagrange(&final_onehot, &final_point);

        let onehot_pre_refs = [&MultilinearPolynomial::onehot(onehot_pre.clone())];
        let dense_refs = [
            &MultilinearPolynomial::dense(dense_a.clone()),
            &MultilinearPolynomial::dense(dense_b.clone()),
        ];
        let final_refs = [&final_polys[0]];

        let prover_data = selected_prover_data::<OneHotCfg, _>(
            OpeningClaims::from_groups(vec![
                PolynomialGroupClaims::new(
                    onehot_pre_point.clone(),
                    vec![onehot_pre_opening],
                    onehot_pre_commitment.clone(),
                )
                .expect("K=256 prover group"),
                PolynomialGroupClaims::new(
                    dense_point.clone(),
                    vec![dense_opening_a, dense_opening_b],
                    dense_commitment.clone(),
                )
                .expect("dense prover group"),
                PolynomialGroupClaims::new(
                    final_point.clone(),
                    vec![final_opening],
                    final_commitment.clone(),
                )
                .expect("final prover group"),
            ])
            .expect("prover claims"),
            vec![onehot_pre_hint, dense_hint, final_hint],
            vec![&onehot_pre_refs, &dense_refs, &final_refs],
        );
        let selection = prover_data.selection();

        // The openings below come from independent oracles, so the resolved
        // schedule is no longer needed to project them. Keep the resolution as
        // a structural check that the heterogeneous selection binds to the
        // two-precommit catalog entry.
        let schedule = OneHotCfg::resolve_schedule_selection(selection)
            .expect("heterogeneous schedule")
            .schedule()
            .clone();
        assert_eq!(
            schedule.root.params.precommitted_groups.len(),
            2,
            "heterogeneous selection must resolve to the two-precommit entry"
        );

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_group_types");
        let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove(
            &setup,
            prover_data,
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("heterogeneous prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let verifier_setup =
            AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup).expect("verifier setup");
        let verify_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                onehot_pre_point,
                vec![onehot_pre_opening],
                &onehot_pre_commitment,
            )
            .expect("K=256 verifier group"),
            PolynomialGroupClaims::new(
                dense_point,
                vec![dense_opening_a, dense_opening_b],
                &dense_commitment,
            )
            .expect("dense verifier group"),
            PolynomialGroupClaims::new(final_point, vec![final_opening], &final_commitment)
                .expect("final verifier group"),
        ])
        .expect("verifier claims");
        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_group_types");
        AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(selection, verify_claims).expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("heterogeneous verify");
    });
}

// Compute backend heterogeneity: commit uses CpuBackend, prove uses a split
// ProverComputeStack with separate backends for each phase.
#[test]
fn heterogeneous_compute_backends() {
    init_rayon_pool();
    run_on_large_stack(|| {
        const NV: usize = 16;
        type Cfg = fp128::Dense;
        type Scheme = AkitaCommitmentScheme<Cfg>;

        let evals: Vec<F> = (0..(1usize << NV)).map(|i| F::from_u64(i as u64)).collect();
        let poly = akita_prover::DensePoly::<F>::from_field_evals(NV, DENSE_D, &evals).unwrap();

        let setup = Scheme::setup_prover(NV, 1).unwrap();
        let prepared = CpuBackend::DEFAULT.prepare_setup(&setup).expect("prepared");

        let commit_backend = CommitCluster;
        let opening_backend = OpeningCluster;
        let tensor = TensorCluster;
        let ring = RingSwitchCluster;
        let stack = ProverComputeStack::new(
            (&commit_backend, &prepared),
            (&opening_backend, &prepared),
            (&tensor, &prepared),
            (&ring, &prepared),
            setup.expanded.as_ref(),
        )
        .expect("heterogeneous stack");

        let verifier_setup = Scheme::setup_verifier(&setup).expect("verifier setup");
        let commit_stack =
            UniformProverStack::uniform(&CpuBackend::DEFAULT, &prepared, setup.expanded.as_ref())
                .expect("commit stack");
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = akita_prover::commit::<Cfg, akita_prover::DensePoly<F>, CpuBackend>(
            std::slice::from_ref(&poly),
            setup.expanded.as_ref(),
            &commit_stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");

        let pt: Vec<F> = (0..NV).map(|i| F::from_u64((i + 2) as u64)).collect();
        let expected_opening = dense_opening_lagrange(&evals, &pt);

        let poly_refs = [&poly];
        let commitments = [commitment];
        let prover_data = selected_prover_data::<Cfg, _>(
            OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                pt.clone(),
                vec![expected_opening],
                commitments[0].clone(),
            )
            .expect("prover group")])
            .expect("prover claims"),
            vec![hint],
            vec![&poly_refs[..]],
        );
        let selection = prover_data.selection();

        let mut prover_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_compute_backends");
        let proof = batched_prove::<Cfg, _, _, _, _, _, _>(
            &setup.expanded,
            &setup.prefix_slots,
            &stack,
            prover_data,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("heterogeneous prove");

        let shape = proof.shape();
        let mut bytes = Vec::new();
        proof.serialize_compressed(&mut bytes).expect("serialize");
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
            &mut std::io::Cursor::new(bytes),
            &shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            AkitaTranscript::<F>::new(b"completeness/heterogeneous_compute_backends");
        Scheme::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            GroupBatchStatement::new(
                selection,
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    pt.clone(),
                    vec![expected_opening],
                    &commitments[0],
                )
                .expect("verifier group")])
                .expect("verifier claims"),
            )
            .expect("statement"),
            BasisMode::Lagrange,
        )
        .expect("heterogeneous verify");
    });
}
