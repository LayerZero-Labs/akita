use super::*;

#[test]
fn profile_native_commit_group_returns_exact_frozen_layout() {
    const NV: usize = 16;
    const GROUP_SIZE: usize = 1;

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let profile = OneHotCfg::profile_without_precommitted_groups(key).expect("independent profile");
    let total_field = (profile.blocks.live_blocks * profile.blocks.positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_0001)];

    let setup = OneHotScheme::setup_prover(NV, GROUP_SIZE).expect("setup");
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
    } = OneHotScheme::commit(
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
    max_num_vars: usize,
    max_num_polys: usize,
    run: impl FnOnce(
        &akita_prover::AkitaProverSetup<OneHotF>,
        &akita_prover::UniformProverStack<'_, OneHotF, CpuBackend>,
    ) -> R,
) -> R {
    let setup = OneHotScheme::setup_prover(max_num_vars, max_num_polys).expect("setup");
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

    let pre_a_key = akita_types::PolynomialGroupLayout::new(NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(NV, PRE_B_SIZE);
    let pre_a_profile =
        OneHotCfg::profile_without_precommitted_groups(pre_a_key).expect("independent profile");
    let pre_b_profile =
        OneHotCfg::profile_without_precommitted_groups(pre_b_key).expect("independent profile");
    let pre_a_polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_1001)];
    let pre_b_polys = [
        debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_2001),
        debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_2002),
    ];

    with_precommit_stack(NV, SETUP_CAPACITY_SIZE, |setup, stack| {
        let akita_prover::CommitOutput {
            committed_group: pre_a_commitment,
            hint: _pre_a_hint,
        } = OneHotScheme::commit(
            setup,
            &pre_a_polys,
            stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit A");
        let akita_prover::CommitOutput {
            committed_group: pre_b_commitment,
            hint: _pre_b_hint,
        } = OneHotScheme::commit(
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

    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_A_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_B_SIZE);
    let pre_c_key = akita_types::PolynomialGroupLayout::new(PRE_NV, PRE_C_SIZE);
    let pre_a_frozen =
        OneHotCfg::profile_without_precommitted_groups(pre_a_key).expect("independent profile");
    let pre_b_frozen =
        OneHotCfg::profile_without_precommitted_groups(pre_b_key).expect("independent profile");
    let pre_c_frozen =
        OneHotCfg::profile_without_precommitted_groups(pre_c_key).expect("independent profile");
    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(FINAL_NV, MAIN_SIZE),
        precommitteds: vec![pre_a_frozen, pre_b_frozen, pre_c_frozen],
    };

    let schedule = OneHotCfg::resolve_catalog_row_for_key(&multi_group_key)
        .expect("multi-group runtime schedule")
        .into_schedule();
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

    let pre_a_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_b_key = akita_types::PolynomialGroupLayout::new(PRE_NV, GROUP_SIZE);
    let pre_a_frozen =
        OneHotCfg::profile_without_precommitted_groups(pre_a_key).expect("independent profile");
    let pre_b_frozen =
        OneHotCfg::profile_without_precommitted_groups(pre_b_key).expect("independent profile");
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

    let setup = OneHotScheme::setup_prover(FINAL_NV, SETUP_CAPACITY_SIZE).expect("protocol setup");
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
    } = OneHotScheme::commit::<_, _>(
        &setup,
        &pre_a_polys,
        &stack,
        akita_prover::GroupContext::scheduler_without_precommitted_groups(),
    )
    .expect("precommit A");
    let akita_prover::CommitOutput {
        committed_group: pre_b_commitment,
        hint: _pre_b_hint,
    } = OneHotScheme::commit::<_, _>(
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

    let multi_group_schedule = OneHotCfg::resolve_catalog_row_for_key(&multi_group_key)
        .expect("multi-group runtime schedule")
        .into_schedule();
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
    } = OneHotScheme::commit(
        &setup,
        &final_polys,
        &stack,
        akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
    )
    .expect("final multi-group commitment");
    let explicit_output = OneHotScheme::commit(
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

    let key = akita_types::PolynomialGroupLayout::new(NV, GROUP_SIZE);
    let profile = OneHotCfg::profile_without_precommitted_groups(key).expect("independent profile");
    let total_field = (profile.blocks.live_blocks * profile.blocks.positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    assert_eq!(total_field % BENCH_ONEHOT_K, 0);
    let polys = [debug_make_onehot_poly(NV, ONEHOT_D, 0x0bee_fcaf_9a77_0001)];

    let setup = OneHotScheme::setup_prover(NV, GROUP_SIZE).expect("setup");
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
    } = OneHotScheme::commit(
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

/// Produce and verify a folded multi-group-root one-hot same-point proof for the
/// given precommitted group sizes plus a final group size, exercising unequal
/// `K_g`. Precommitted groups use exact generated standalone profiles; the
/// final group uses a scheduled context with precommitted groups; the multi-group root folds
/// into a singleton recursive suffix.
fn multi_group_root_round_trip_onehot<TestCfg, ProtocolCfg>(
    pre_num_vars: usize,
    final_num_vars: usize,
    pre_sizes: &[usize],
    final_size: usize,
    check_group_binding: bool,
    max_cached_ring_switch_elements: usize,
) -> AkitaBatchedProof<OneHotF, OneHotF>
where
    TestCfg: CommitmentConfig<Field = OneHotF, ExtField = OneHotF>,
    ProtocolCfg: CommitmentConfig<Field = OneHotF, ExtField = OneHotF>,
{
    let total: usize = pre_sizes.iter().sum::<usize>() + final_size;
    let opening_num_vars = pre_num_vars.max(final_num_vars);

    let setup =
        AkitaCommitmentScheme::<ProtocolCfg>::setup_prover(opening_num_vars, total).expect("setup");
    let cached_backend = CpuBackend::with_resource_limits(
        max_cached_ring_switch_elements,
        CpuBackend::DEFAULT_COMMIT_SCRATCH_BYTES_PER_WORKER,
    )
    .expect("cached backend");
    let prepared = cached_backend
        .prepare_setup(&setup)
        .expect("prepared setup");
    let stack = akita_prover::UniformProverStack::uniform(
        &cached_backend,
        &prepared,
        setup.expanded.as_ref(),
    )
    .expect("stack");
    // Commit every precommitted group from its exact generated profile; keep the
    // polynomials alive so the prover/verifier can borrow references.
    let mut pre_keys = Vec::new();
    let mut pre_frozen = Vec::new();
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_layouts = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<OneHotPoly<OneHotF, u8>>> = Vec::new();
    for (group_idx, &num_polynomials) in pre_sizes.iter().enumerate() {
        let key = akita_types::PolynomialGroupLayout::new(pre_num_vars, num_polynomials);
        let profile =
            ProtocolCfg::profile_without_precommitted_groups(key).expect("independent profile");
        let polys: Vec<OneHotPoly<OneHotF, u8>> = (0..num_polynomials)
            .map(|poly_idx| {
                debug_make_onehot_poly(
                    pre_num_vars,
                    profile.inner.matrix.ring_dimension(),
                    0x0bee_fcaf_1a00_0000 + ((group_idx as u64) << 8) + poly_idx as u64,
                )
            })
            .collect();
        let akita_prover::CommitOutput {
            committed_group: commitment,
            hint,
        } = AkitaCommitmentScheme::<ProtocolCfg>::commit(
            &setup,
            &polys,
            &stack,
            akita_prover::GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("precommit");
        pre_frozen.push(commitment.profile);
        pre_keys.push(key);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_layouts.push(profile);
        pre_polys_by_group.push(polys);
    }

    let multi_group_key = akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(final_num_vars, final_size),
        precommitteds: pre_frozen,
    };
    let opening_layout = multi_group_key
        .opening_layout()
        .expect("multi-group opening layout");
    let multi_group_schedule = ProtocolCfg::resolve_catalog_row_for_key(&multi_group_key)
        .expect("multi-group runtime schedule")
        .into_schedule();
    let main_params = multi_group_root_params(&multi_group_schedule);
    assert_eq!(
        multi_group_schedule
            .root
            .params
            .precommitted_groups()
            .iter()
            .map(|group| group.profile)
            .collect::<Vec<_>>(),
        pre_layouts,
        "precommitted groups must retain their native descriptors"
    );
    if TestCfg::chunked_witness_cfg().uses_multi_chunk() {
        let root = &multi_group_schedule.root;
        let root_commitment = &root.params;
        assert!(!root.params.precommitted_groups().is_empty());
        assert_eq!(
            root_commitment.witness_chunk.num_chunks,
            TestCfg::chunked_witness_cfg().num_chunks,
            "root fold must retain the configured chunk count"
        );
        let relation_geometry =
            akita_types::RelationWitnessGeometry::for_evaluation_trace_execution(
                root_commitment,
                &opening_layout,
            )
            .expect("evaluation-trace relation geometry");
        let witness_layout = akita_types::WitnessLayout::new(
            root_commitment,
            &opening_layout,
            &relation_geometry,
            root_commitment.witness_chunk.num_chunks,
            akita_types::RelationQuotientPlan::quotient_lift(
                akita_types::r_decomp_levels::<OneHotF>(root_commitment.open().digits.log_basis),
            )
            .unwrap(),
        )
        .expect("group-by-chunk witness layout");
        assert_eq!(
            witness_layout.units().len(),
            opening_layout.num_groups() * root_commitment.witness_chunk.num_chunks,
        );
    }
    let final_polys: Vec<OneHotPoly<OneHotF, u8>> = (0..final_size)
        .map(|poly_idx| {
            debug_make_onehot_poly(
                final_num_vars,
                main_params.d_a(),
                0x0bee_fcaf_f100_0000 + poly_idx as u64,
            )
        })
        .collect();
    let precommitteds =
        akita_types::PrecommittedGroupProfiles::from_ordered_groups(pre_commitments.iter())
            .expect("nonempty precommitted groups");
    let akita_prover::CommitOutput {
        committed_group: final_commitment,
        hint: final_hint,
    } = AkitaCommitmentScheme::<ProtocolCfg>::commit(
        &setup,
        &final_polys,
        &stack,
        akita_prover::GroupContext::scheduler_with_precommitted_groups(&precommitteds),
    )
    .expect("final multi-group commitment");

    let mut pre_point = debug_random_point(pre_num_vars);
    pre_point[0] += OneHotF::one();
    let final_point = debug_random_point(final_num_vars);
    let pre_openings: Vec<Vec<OneHotF>> = pre_polys_by_group
        .iter()
        .zip(pre_layouts.iter())
        .map(|(polys, layout)| {
            polys
                .iter()
                .map(|poly| {
                    opening_from_poly(
                        poly,
                        &pre_point,
                        layout.inner.matrix.ring_dimension(),
                        layout.blocks.positions_per_block,
                        layout.blocks.live_blocks,
                    )
                })
                .collect()
        })
        .collect();
    let final_openings: Vec<OneHotF> = final_polys
        .iter()
        .map(|poly| {
            opening_from_poly(
                poly,
                &final_point,
                main_params.d_a(),
                main_params.blocks().positions_per_block,
                main_params.blocks().live_blocks,
            )
        })
        .collect();

    let pre_refs_by_group: Vec<Vec<&OneHotPoly<OneHotF, u8>>> = pre_polys_by_group
        .iter()
        .map(|polys| polys.iter().collect())
        .collect();
    let final_refs: Vec<&OneHotPoly<OneHotF, u8>> = final_polys.iter().collect();

    let mut prover_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        prover_groups.push(
            PolynomialGroupClaims::new(
                pre_point.clone(),
                openings.clone(),
                pre_commitments[group_idx].clone(),
            )
            .expect("pre prover group"),
        );
    }
    prover_groups.push(
        PolynomialGroupClaims::new(
            final_point.clone(),
            final_openings.clone(),
            final_commitment.clone(),
        )
        .expect("final prover group"),
    );

    let mut prover_polys: Vec<&[&OneHotPoly<OneHotF, u8>]> = Vec::new();
    for refs in &pre_refs_by_group {
        prover_polys.push(&refs[..]);
    }
    prover_polys.push(&final_refs[..]);
    let mut prover_hints = pre_hints;
    prover_hints.push(final_hint);

    let prover_claims = selected_prover_data::<ProtocolCfg, _>(
        OpeningClaims::from_groups(prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
    )
    .expect("multi-group prover data");
    let selection = prover_claims.selection();

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    let proof = AkitaCommitmentScheme::<ProtocolCfg>::batched_prove(
        &setup,
        prover_claims,
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("multi-group prove");
    assert!(proof.num_fold_levels() >= 2);
    let planned_stage3 = multi_group_schedule
        .recursive_folds
        .iter()
        .filter(|fold| fold.params.setup_prefix().is_some())
        .count();
    let proved_stage3 = proof
        .nonterminal_folds()
        .filter(|fold| fold.stage3_sumcheck_proof().is_some())
        .count();
    assert_eq!(
        proved_stage3, planned_stage3,
        "proof stage-3 payloads must follow the config-selected schedule"
    );

    let shape = proof.shape();
    let mut bytes = Vec::new();
    proof
        .serialize_uncompressed(&mut bytes)
        .expect("serialize multi-group proof");
    let decoded = akita_types::AkitaBatchedProof::<OneHotF, OneHotF>::deserialize_uncompressed(
        &bytes[..],
        &shape,
    )
    .expect("deserialize multi-group proof");
    assert_eq!(decoded, proof);

    let verifier_setup =
        AkitaCommitmentScheme::<ProtocolCfg>::setup_verifier(&setup).expect("verifier setup");
    let mut verifier_groups = Vec::new();
    for (group_idx, openings) in pre_openings.iter().enumerate() {
        verifier_groups.push(
            PolynomialGroupClaims::new(
                pre_point.clone(),
                openings.clone(),
                &pre_commitments[group_idx],
            )
            .expect("pre verifier group"),
        );
    }
    verifier_groups.push(
        PolynomialGroupClaims::new(
            final_point.clone(),
            final_openings.clone(),
            &final_commitment,
        )
        .expect("final verifier group"),
    );
    let verify_claims =
        OpeningClaims::from_groups(verifier_groups).expect("multi-group verifier claims");
    let mut verifier_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    AkitaCommitmentScheme::<ProtocolCfg>::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        GroupBatchStatement::new(selection, verify_claims).expect("multi-group statement"),
        BasisMode::Lagrange,
    )
    .expect("multi-group verify");

    if check_group_binding {
        assert_eq!(pre_commitments.len(), 1, "binding fixture uses two groups");
        let swapped_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(
                pre_point.clone(),
                pre_openings[0].clone(),
                &final_commitment,
            )
            .expect("swapped pre verifier group"),
            PolynomialGroupClaims::new(
                final_point.clone(),
                final_openings.clone(),
                &pre_commitments[0],
            )
            .expect("swapped final verifier group"),
        ])
        .expect("swapped verifier claims");
        let mut swapped_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
        assert!(
            AkitaCommitmentScheme::<ProtocolCfg>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut swapped_transcript,
                GroupBatchStatement::new(selection, swapped_claims)
                    .expect("swapped-group statement"),
                BasisMode::Lagrange,
            )
            .is_err(),
            "swapped group commitments must reject"
        );

        let mut tampered_final_openings = final_openings.clone();
        tampered_final_openings[0] += OneHotF::one();
        let tampered_claims = OpeningClaims::from_groups(vec![
            PolynomialGroupClaims::new(pre_point, pre_openings[0].clone(), &pre_commitments[0])
                .expect("pre verifier group"),
            PolynomialGroupClaims::new(final_point, tampered_final_openings, &final_commitment)
                .expect("tampered final verifier group"),
        ])
        .expect("tampered verifier claims");
        let mut tampered_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
        assert!(
            AkitaCommitmentScheme::<ProtocolCfg>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut tampered_transcript,
                GroupBatchStatement::new(selection, tampered_claims)
                    .expect("tampered-opening statement"),
                BasisMode::Lagrange,
            )
            .is_err(),
            "tampered group opening must reject"
        );
    }
    proof
}

#[test]
fn multi_group_root_folded_group_binding_round_trips() {
    multi_group_root_round_trip_onehot::<OneHotCfg, OneHotCfg>(14, 20, &[1], 2, true, usize::MAX);
}

#[test]
fn multi_group_root_allows_precommitted_arity_above_final_group() {
    type PlannerCfg = crate::test_support::EnvelopeFinalGroupConfig<OneHotCfg, OneHotCfg>;

    multi_group_root_round_trip_onehot::<OneHotCfg, PlannerCfg>(20, 14, &[1], 1, false, usize::MAX);
}

#[test]
fn multi_group_root_opens_multi_polynomial_precommitted_group() {
    multi_group_root_round_trip_onehot::<OneHotCfg, OneHotCfg>(14, 20, &[2], 1, false, usize::MAX);
}

#[test]
fn three_group_cached_and_streamed_proofs_are_identical() {
    let cached = multi_group_root_round_trip_onehot::<OneHotCfg, OneHotCfg>(
        14,
        20,
        &[1, 1],
        4,
        false,
        usize::MAX,
    );
    let streamed =
        multi_group_root_round_trip_onehot::<OneHotCfg, OneHotCfg>(14, 20, &[1, 1], 4, false, 0);
    assert_eq!(streamed, cached, "cached and streamed proofs differ");
}

#[test]
#[cfg(feature = "profile-ci")]
fn multi_group_multi_chunk_fold_round_trips() {
    multi_group_root_round_trip_onehot::<fp128::OneHotMultiChunkW2R2, fp128::OneHotMultiChunkW2R2>(
        14,
        14,
        &[1],
        1,
        false,
        usize::MAX,
    );
}

#[test]
fn batched_onehot_roundtrip_matches_public_shape_context() {
    // NV chosen large enough that the runtime schedule yields at least two
    // fold steps so the proof is fold-rooted (not terminal-rooted). Under
    // the post-soundness-fix proof shape, a single-fold schedule emits a
    // `Terminal` root with no recursive suffix, which this test does not
    // exercise.
    const NV: usize = 20;
    const BATCH_SIZE: usize = 2;

    let layout = akita_batched_root_layout::<OneHotCfg>(NV, BATCH_SIZE).expect("layout");
    let total_field = (layout.blocks().live_blocks * layout.blocks().positions_per_block)
        .checked_mul(ONEHOT_D)
        .expect("total field size overflow");
    let total_chunks = total_field / BENCH_ONEHOT_K;
    assert_eq!(total_chunks * BENCH_ONEHOT_K, total_field);

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

    let setup = OneHotScheme::setup_prover(NV, BATCH_SIZE).unwrap();
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
    let verifier_setup = OneHotScheme::setup_verifier(&setup).expect("verifier setup");
    let akita_prover::CommitOutput {
        committed_group: commitment,
        hint,
    } = OneHotScheme::commit::<_, _>(
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
    let proof = OneHotScheme::batched_prove::<_, _, _>(
        &setup,
        selected_prover_data::<OneHotCfg, _>(
            OpeningClaims::from_groups(vec![prover_group]).expect("valid one-hot prover claims"),
            vec![hint],
            vec![&poly_refs[..]],
        )
        .expect("valid one-hot prover opening data"),
        &stack,
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .expect("batched onehot prove");

    let expected_shape = expected_same_point_batched_shape(NV, BATCH_SIZE, &proof);
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
    OneHotScheme::batched_verify(
        &decoded,
        &verifier_setup,
        &mut verifier_transcript,
        selected_statement::<OneHotCfg>(
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
