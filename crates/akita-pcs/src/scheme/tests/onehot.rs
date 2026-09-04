use super::*;

#[test]
fn profile_native_commit_group_returns_exact_frozen_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let profile = catalog_profile(&scheme, key);
    let total_field = (profile.blocks.live_blocks * profile.blocks.positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % onehot_source_chunk_size::<OneHotCfg>(), 0);
    let polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_0001)];

    let setup = scheme.setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint: _hint,
    } = scheme
        .commit(
            &setup,
            &polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit");
    let frozen_layout = commitment.profile;

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.blocks.positions_per_block,
        profile.blocks.positions_per_block
    );
    assert_eq!(frozen_layout.blocks.live_blocks, profile.blocks.live_blocks);
    assert_eq!(
        frozen_layout.outer.digits.log_basis,
        OneHotCfg::opening_basis_range().0
    );
    assert_eq!(
        frozen_layout.inner.matrix.output_rank(),
        profile.inner.matrix.output_rank()
    );
    assert_eq!(
        frozen_layout.outer.matrix.output_rank(),
        profile.outer.matrix.output_rank()
    );
    assert_eq!(
        commitment.rows().count(),
        frozen_layout.outer.matrix.output_rank()
    );
}

fn multi_group_root_params(schedule: &akita_types::FoldSchedule) -> &CommittedGroupParams {
    &schedule.root.params
}

fn with_precommit_stack<R>(
    scheme: &OneHotScheme,
    max_num_vars: usize,
    max_num_polys: usize,
    run: impl FnOnce(
        &akita_prover::AkitaProverSetup<OneHotF>,
        &akita_prover::UniformProverStack<'_, OneHotF, CpuBackend>,
    ) -> R,
) -> R {
    let setup = scheme
        .setup_prover(max_num_vars, max_num_polys)
        .expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    run(&setup, &stack)
}

#[test]
fn profile_native_commit_group_allows_independent_groups() {
    const NV: usize = 16;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 2;
    // Precommitted groups are committed independently, so setup only needs to
    // cover the largest standalone group rather than the sum of all groups.
    const SETUP_CAPACITY_SIZE: usize = PRE_B_SIZE;

    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let pre_a_key = akita_types::PolynomialGroupLayout::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(NV, PRE_B_SIZE);
    let pre_a_profile = catalog_profile(&scheme, pre_a_key);
    let pre_b_profile = catalog_profile(&scheme, pre_b_key);
    let pre_a_polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_1001)];
    let pre_b_polys = [
        debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_2001),
        debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_2002),
    ];

    with_precommit_stack(&scheme, NV, SETUP_CAPACITY_SIZE, |setup, stack| {
        let akita_prover::CommitOutput {
            committed_group: pre_a_commitment,
            hint: _pre_a_hint,
        } = scheme
            .commit(
                setup,
                &pre_a_polys,
                stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("precommit A");
        let akita_prover::CommitOutput {
            committed_group: pre_b_commitment,
            hint: _pre_b_hint,
        } = scheme
            .commit(
                setup,
                &pre_b_polys,
                stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("precommit B");
        let pre_a_frozen = pre_a_commitment.profile;
        let pre_b_frozen = pre_b_commitment.profile;

        assert_eq!(pre_a_frozen.group, pre_a_key);
        assert_eq!(pre_b_frozen.group, pre_b_key);
        assert_eq!(
            pre_a_commitment.rows().count(),
            pre_a_frozen.outer.matrix.output_rank()
        );
        assert_eq!(
            pre_b_commitment.rows().count(),
            pre_b_frozen.outer.matrix.output_rank()
        );
        assert_ne!(pre_a_frozen.group, pre_b_frozen.group);
        assert_eq!(pre_a_frozen, pre_a_profile);
        assert_eq!(pre_b_frozen, pre_b_profile);
    });
}

#[test]
fn group_batch_schedule_preserves_precommitted_order() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;
    const PRE_A_SIZE: usize = 1;
    const PRE_B_SIZE: usize = 1;
    const PRE_C_SIZE: usize = 1;
    const MAIN_SIZE: usize = 4;

    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_B_SIZE);
    let pre_c_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_C_SIZE);
    let pre_a_frozen = catalog_profile(&scheme, pre_a_key);
    let pre_b_frozen = catalog_profile(&scheme, pre_b_key);
    let pre_c_frozen = catalog_profile(&scheme, pre_c_key);
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, MAIN_SIZE),
        precommitteds: vec![pre_a_frozen, pre_b_frozen, pre_c_frozen],
    };

    let schedule = scheme
        .schedules()
        .resolve_key(&multi_group_key)
        .expect("multi-group runtime schedule")
        .schedule()
        .clone();
    let root = multi_group_root_params(&schedule);
    let main_params = schedule.root.params.clone();

    assert_eq!(multi_group_key.num_commitment_groups(), 4);
    assert_eq!(
        multi_group_key
            .num_polynomials()
            .expect("multi-group polynomial count"),
        PRE_A_SIZE + PRE_B_SIZE + PRE_C_SIZE + MAIN_SIZE
    );
    assert_eq!(main_params, *root);
    assert_eq!(schedule.root.params.precommitted_groups().len(), 3);
    assert_eq!(
        schedule.root.params.precommitted_groups()[0].profile,
        pre_a_frozen
    );
    assert_eq!(
        schedule.root.params.precommitted_groups()[1].profile,
        pre_b_frozen
    );
    assert_eq!(
        schedule.root.params.precommitted_groups()[2].profile,
        pre_c_frozen
    );
}

#[test]
fn group_batch_commits_independent_arity_precommitted_groups() {
    const PRE_NV: usize = 14;
    const FINAL_NV: usize = 20;
    const GROUP_SIZE: usize = 1;
    const FINAL_SIZE: usize = 4;
    const SETUP_CAPACITY_SIZE: usize = FINAL_SIZE + 2 * GROUP_SIZE;

    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_a_frozen = catalog_profile(&scheme, pre_a_key);
    let pre_b_frozen = catalog_profile(&scheme, pre_b_key);
    let pre_a_polys = [debug_make_onehot_poly(
        PRE_NV,
        ONEHOT_D,
        0x0bee_fcaf_9a77_5001,
    )];
    let pre_b_polys = [debug_make_onehot_poly(
        PRE_NV,
        ONEHOT_D,
        0x0bee_fcaf_9a77_6001,
    )];

    let setup = scheme
        .setup_prover(FINAL_NV, SETUP_CAPACITY_SIZE)
        .expect("protocol setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared protocol setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("protocol stack");
    let akita_prover::CommitOutput {
        committed_group: pre_a_commitment,
        hint: _pre_a_hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            &pre_a_polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit A");
    let akita_prover::CommitOutput {
        committed_group: pre_b_commitment,
        hint: _pre_b_hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            &pre_b_polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit B");
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, FINAL_SIZE),
        precommitteds: vec![pre_a_frozen, pre_b_frozen],
    };
    assert!(multi_group_key
        .fits_setup_capacity(FINAL_NV, SETUP_CAPACITY_SIZE)
        .expect("setup capacity"));

    let multi_group_schedule = scheme
        .schedules()
        .resolve_key(&multi_group_key)
        .expect("multi-group runtime schedule")
        .schedule()
        .clone();
    let main_params = multi_group_root_params(&multi_group_schedule);
    let final_polys = [
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7001),
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7002),
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7003),
        debug_make_onehot_poly(FINAL_NV, main_params.d_a(), 0x0bee_fcaf_9a77_7004),
    ];
    let precommitteds = akita_types::PrecommittedGroupProfiles::from_profiles(vec![
        pre_a_commitment.profile,
        pre_b_commitment.profile,
    ])
    .expect("nonempty precommitted groups");
    let akita_prover::CommitOutput {
        committed_group: final_commitment,
        hint: final_hint,
    } = scheme
        .commit(
            &setup,
            &final_polys,
            &stack,
            akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
        )
        .expect("final multi-group commitment");
    let explicit_output = scheme
        .commit(
            &setup,
            &final_polys,
            &stack,
            akita_prover::GroupContext::explicit(&main_params.own_group().profile),
        )
        .expect("explicit final multi-group commitment");

    assert_eq!(explicit_output.committed_group, final_commitment);
    assert_eq!(explicit_output.hint, final_hint);

    assert_eq!(
        pre_a_commitment.rows().count(),
        pre_a_frozen.outer.matrix.output_rank()
    );
    assert_eq!(
        pre_b_commitment.rows().count(),
        pre_b_frozen.outer.matrix.output_rank()
    );
    assert_eq!(
        final_commitment.rows().count(),
        main_params.outer().matrix.output_rank()
    );
    assert_eq!(final_hint.inner_rows().len(), FINAL_SIZE);
    assert_eq!(
        akita_prover::RootPolyMeta::num_vars(&final_polys[0]),
        FINAL_NV,
        "final one-hot group should retain its native variable domain"
    );
    assert_eq!(
        multi_group_schedule.root.params.precommitted_groups().len(),
        2
    );
    assert_eq!(
        multi_group_schedule.root.params.precommitted_groups()[0].profile,
        pre_a_frozen
    );
    assert_eq!(
        multi_group_schedule.root.params.precommitted_groups()[1].profile,
        pre_b_frozen
    );
}

#[test]
fn commit_group_returns_frozen_exact_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let profile = catalog_profile(&scheme, key);
    let total_field = (profile.blocks.live_blocks * profile.blocks.positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % onehot_source_chunk_size::<OneHotCfg>(), 0);
    let polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_0001)];

    let setup = scheme.setup_prover(NV, GROUP_SIZE).expect("setup");
    let prepared = CpuBackend::DEFAULT
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &CpuBackend::DEFAULT,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint: _hint,
    } = scheme
        .commit(
            &setup,
            &polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit group");
    let frozen_layout = commitment.profile;

    assert_eq!(frozen_layout.group, key);
    assert_eq!(
        frozen_layout.blocks.positions_per_block,
        profile.blocks.positions_per_block
    );
    assert_eq!(frozen_layout.blocks.live_blocks, profile.blocks.live_blocks);
    assert_eq!(
        frozen_layout.outer.digits.log_basis,
        profile.outer.digits.log_basis
    );
    assert_eq!(
        frozen_layout.inner.matrix.output_rank(),
        profile.inner.matrix.output_rank()
    );
    assert_eq!(
        frozen_layout.outer.matrix.output_rank(),
        profile.outer.matrix.output_rank()
    );
    assert_eq!(
        commitment.rows().count(),
        frozen_layout.outer.matrix.output_rank()
    );
}

mod multi_group;
#[test]
fn batched_onehot_roundtrip_matches_public_shape_context() {
    // NV chosen large enough that the runtime schedule yields at least two
    // fold steps so the proof is fold-rooted (not terminal-rooted). Under
    // the post-soundness-fix proof shape, a single-fold schedule emits a
    // `Terminal` root with no recursive suffix, which this test does not
    // exercise.
    const NV: usize = 20;
    const BATCH_SIZE: usize = 2;

    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let layout = catalog_root_layout(&scheme, NV, BATCH_SIZE);
    let total_field = (layout.blocks().live_blocks * layout.blocks().positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    let onehot_k = onehot_source_chunk_size::<OneHotCfg>();
    let total_chunks = total_field / onehot_k;
    assert_eq!(total_chunks * onehot_k, total_field);

    let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..BATCH_SIZE)
        .map(|poly_idx| {
            debug_make_onehot_poly(NV, layout.d_a(), 0x0bee_fcaf_e000_1500 + poly_idx as u64)
        })
        .collect();
    let poly_refs: Vec<&OneHotPoly<OneHotF, u8>> = polys.iter().collect();
    let point = debug_random_point(NV);
    let openings: Vec<OneHotF> = polys
        .iter()
        .map(|poly| {
            opening_from_poly(
                poly,
                &point,
                layout.d_a(),
                layout.blocks().positions_per_block,
                layout.blocks().live_blocks,
            )
        })
        .collect();

    let setup = scheme.setup_prover(NV, BATCH_SIZE).unwrap();
    let cached_backend = CpuBackend::with_resource_limits(
        usize::MAX,
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
    )
    .unwrap();
    let prepared = cached_backend.prepare_setup(&setup).unwrap();
    let stack = akita_prover::UniformProverStack::uniform(
        &cached_backend,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = scheme
        .commit::<_, _>(
            &setup,
            &polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("batched onehot commit");
    let commitments = [commitment];
    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    let prover_group = PolynomialGroupClaims::new(
        point.clone(),
        vec![OneHotF::zero(); poly_refs.len()],
        commitments[0].clone(),
    )
    .expect("valid one-hot prover group");
    let proof = scheme
        .batched_prove::<_, _, _>(
            &setup,
            selected_prover_data::<OneHotCfg, _>(
                &scheme,
                OpeningClaims::from_groups(vec![prover_group])
                    .expect("valid one-hot prover claims"),
                vec![hint],
                vec![&poly_refs[..]],
            )
            .expect("valid one-hot prover opening data"),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("batched onehot prove");

    let expected_shape = expected_same_point_batched_shape(&scheme, NV, BATCH_SIZE, &proof);
    let actual_shape = proof.shape();
    assert_eq!(
        expected_shape.nonce_stream_bits,
        actual_shape.nonce_stream_bits
    );
    assert_eq!(
        expected_shape.root.opening_payload_coeffs,
        actual_shape.root.opening_payload_coeffs
    );
    assert_eq!(
        expected_shape.root.stage1_stages,
        actual_shape.root.stage1_stages
    );
    assert_eq!(
        expected_shape.root.stage2_sumcheck_proof,
        actual_shape.root.stage2_sumcheck_proof
    );
    assert_eq!(
        expected_shape.root.next_witness_binding,
        actual_shape.root.next_witness_binding
    );
    assert_eq!(expected_shape.recursive_folds, actual_shape.recursive_folds);
    assert_eq!(
        expected_shape.terminal.extension_opening_reduction,
        actual_shape.terminal.extension_opening_reduction
    );
    assert!(
        expected_shape
            .terminal
            .terminal_response
            .admits_realized(&actual_shape.terminal.terminal_response),
        "terminal witness shape {:?} does not admit {:?}",
        expected_shape.terminal.terminal_response,
        actual_shape.terminal.terminal_response
    );
    let mut bytes = Vec::new();
    proof.serialize_uncompressed(&mut bytes).unwrap();
    let decoded =
        AkitaBatchedProof::<OneHotF, OneHotF>::deserialize_uncompressed(&*bytes, &actual_shape)
            .expect("deserialize batched proof with derived shape");
    assert_eq!(decoded, proof);

    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/batched-onehot-shape");
    scheme
        .batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            selected_statement::<OneHotCfg>(
                &scheme,
                OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
                    point,
                    openings,
                    &commitments[0],
                )
                .expect("valid one-hot verifier group")])
                .expect("valid one-hot verifier claims"),
            )
            .expect("valid one-hot verifier statement"),
            BasisMode::Lagrange,
        )
        .expect("batched onehot verify");
}

#[path = "onehot/selective_l2.rs"]
mod selective_l2;
