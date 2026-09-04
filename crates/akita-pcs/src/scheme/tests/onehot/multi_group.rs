use super::*;

fn multi_group_key(
    schedules: &akita_config::TrustedScheduleCatalog,
    pre_num_vars: usize,
    final_num_vars: usize,
    pre_sizes: &[usize],
    final_size: usize,
) -> akita_types::AkitaScheduleLookupKey {
    let precommitteds = pre_sizes
        .iter()
        .map(|&num_polynomials| {
            schedules
                .resolve_key(&akita_types::AkitaScheduleLookupKey::single(
                    akita_types::PolynomialGroupLayout::new(pre_num_vars, num_polynomials),
                ))
                .expect("independent row")
                .profiles()
                .final_group
        })
        .collect();
    akita_types::AkitaScheduleLookupKey {
        final_group: akita_types::PolynomialGroupLayout::new(final_num_vars, final_size),
        precommitteds,
    }
}

/// Produce and verify a folded multi-group-root one-hot same-point proof from
/// one caller-materialized scheme and exact lookup key.
fn multi_group_root_round_trip_onehot<ProtocolCfg>(
    scheme: &AkitaCommitmentScheme<ProtocolCfg>,
    multi_group_key: &akita_types::AkitaScheduleLookupKey,
    check_group_binding: bool,
    max_cached_ring_switch_elements: usize,
) -> AkitaBatchedProof<OneHotF, OneHotF>
where
    ProtocolCfg: CommitmentConfig<Field = OneHotF, ExtField = OneHotF>,
{
    let pre_layouts = multi_group_key.precommitteds.clone();
    let pre_num_vars = pre_layouts
        .first()
        .expect("multi-group fixture requires a precommitted group")
        .group
        .num_vars();
    assert!(
        pre_layouts
            .iter()
            .all(|profile| profile.group.num_vars() == pre_num_vars),
        "multi-group fixture requires one shared precommitted point domain"
    );
    let pre_sizes = pre_layouts
        .iter()
        .map(|profile| profile.group.num_polynomials())
        .collect::<Vec<_>>();
    let final_num_vars = multi_group_key.final_group.num_vars();
    let final_size = multi_group_key.final_group.num_polynomials();
    let total = multi_group_key
        .num_polynomials()
        .expect("multi-group polynomial count");
    let opening_num_vars = pre_num_vars.max(final_num_vars);
    let multi_group_schedule = scheme
        .schedules()
        .resolve_key(multi_group_key)
        .expect("multi-group runtime schedule")
        .schedule()
        .clone();

    let setup = scheme.setup_prover(opening_num_vars, total).expect("setup");
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
    let mut pre_commitments = Vec::new();
    let mut pre_hints = Vec::new();
    let mut pre_polys_by_group: Vec<Vec<OneHotPoly<OneHotF, u8>>> = Vec::new();
    for (group_idx, (&num_polynomials, profile)) in
        pre_sizes.iter().zip(pre_layouts.iter()).enumerate()
    {
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
        } = scheme
            .commit(
                &setup,
                &polys,
                &stack,
                akita_prover::GroupContext::scheduler_without_precommitted_groups(),
            )
            .expect("precommit");
        assert_eq!(commitment.profile, *profile);
        pre_commitments.push(commitment);
        pre_hints.push(hint);
        pre_polys_by_group.push(polys);
    }

    let opening_layout = multi_group_key
        .opening_layout()
        .expect("multi-group opening layout");
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
    if ProtocolCfg::chunked_witness_cfg().uses_multi_chunk() {
        let root = &multi_group_schedule.root;
        let root_commitment = &root.params;
        assert!(!root.params.precommitted_groups().is_empty());
        assert_eq!(
            root_commitment.witness_chunk.num_chunks,
            ProtocolCfg::chunked_witness_cfg().num_chunks,
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
            .expect("quotient-lift relation plan"),
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
    } = scheme
        .commit(
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

    let prover_claims = SelectedProverOpeningData::from_committed_claims::<ProtocolCfg>(
        OpeningClaims::from_groups(prover_groups).expect("prover claims"),
        prover_hints,
        prover_polys,
        scheme.schedules(),
    )
    .expect("multi-group prover data");
    let selection = prover_claims.selection();

    let mut prover_transcript = AkitaTranscript::<OneHotF>::new(b"test/multi-group-unequal");
    let proof = scheme
        .batched_prove(
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

    let verifier_setup = scheme.setup_verifier(&setup).expect("verifier setup");
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
    scheme
        .batched_verify(
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
            scheme
                .batched_verify(
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
            scheme
                .batched_verify(
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
    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let key = multi_group_key(scheme.schedules(), 14, 20, &[1], 2);
    multi_group_root_round_trip_onehot(&scheme, &key, true, usize::MAX);
}

#[test]
fn multi_group_root_allows_precommitted_arity_above_final_group() {
    type PlannerCfg = crate::test_support::EnvelopeFinalGroupConfig<OneHotCfg, OneHotCfg>;

    let workspace_catalog = akita_config::test_support::workspace_schedule_catalog::<PlannerCfg>()
        .expect("workspace schedule artifact");
    let key = multi_group_key(&workspace_catalog, 20, 14, &[1], 1);
    workspace_catalog
        .resolve_key(&key)
        .expect_err("synthetic planner row must not be shipped");
    let synthetic = PlannerCfg::derive_catalog_row(&key).expect("synthetic planner row");
    let schedules = akita_config::TrustedScheduleCatalog::try_new(
        PlannerCfg::schedule_family_name(),
        workspace_catalog
            .rows()
            .map(|row| (row.profiles().clone(), row.schedule().clone()))
            .chain(std::iter::once((
                synthetic.profiles().clone(),
                synthetic.schedule().clone(),
            ))),
        &akita_config::policy_of::<PlannerCfg>(),
        PlannerCfg::ring_challenge_config,
    )
    .expect("synthetic planner catalog");
    let scheme = AkitaCommitmentScheme::<PlannerCfg>::new(schedules).expect("planner test scheme");
    multi_group_root_round_trip_onehot(&scheme, &key, false, usize::MAX);
}

#[test]
fn multi_group_root_opens_multi_polynomial_precommitted_group() {
    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let key = multi_group_key(scheme.schedules(), 14, 20, &[2], 1);
    multi_group_root_round_trip_onehot(&scheme, &key, false, usize::MAX);
}

#[test]
fn three_group_cached_and_streamed_proofs_are_identical() {
    let scheme = workspace_scheme::<OneHotCfg>().expect("workspace schedule artifact");
    let key = multi_group_key(scheme.schedules(), 14, 20, &[1, 1], 4);
    let cached = multi_group_root_round_trip_onehot(&scheme, &key, false, usize::MAX);
    let streamed = multi_group_root_round_trip_onehot(&scheme, &key, false, 0);
    assert_eq!(streamed, cached, "cached and streamed proofs differ");
}

#[test]
#[cfg(feature = "profile-ci")]
fn multi_group_multi_chunk_fold_round_trips() {
    type MultiChunkCfg = fp128::OneHotMultiChunkW2R2;
    let scheme = workspace_scheme::<MultiChunkCfg>().expect("workspace schedule artifact");
    let key = multi_group_key(scheme.schedules(), 14, 14, &[1], 1);
    multi_group_root_round_trip_onehot(&scheme, &key, false, usize::MAX);
}
