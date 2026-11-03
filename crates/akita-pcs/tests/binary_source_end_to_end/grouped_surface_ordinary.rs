//! Several ordinary groups and polynomial owners in one binary opening.
use super::*;

#[test]
fn grouped_surface_multiple_ordinary_groups_and_polynomials() {
    init_rayon_pool();
    run_on_large_stack(|| mixed_ordinary(false));
}

/// The catalog's singleton profile has native degree 1024, above the root's
/// 648 budget. Both sides reject it as setup before any opening byte.
#[test]
fn grouped_surface_unsupported_ordinary_profile_rejects_before_opening_bytes() {
    init_rayon_pool();
    run_on_large_stack(|| mixed_ordinary(true));
}

fn mixed_catalog(
    unsupported: bool,
) -> (
    TrustedScheduleCatalog<BinaryTestSuffix>,
    [akita_params::GroupCommitPhaseParams; 2],
) {
    let base = test_catalog();
    let singleton = |n| ScheduleLookupKey::single(PolynomialGroupLayout::singleton(n));
    let carrier = base.catalog().resolve_key(&singleton(23)).unwrap();
    let baseline_key = ScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(18),
        precommitteds: vec![carrier.profiles().final_group],
    };
    let baseline = base.catalog().resolve_key(&baseline_key).unwrap();
    let final_row = base.catalog().resolve_key(&singleton(18)).unwrap();
    let mut ordinary_profile = if unsupported {
        final_row.profiles().final_group
    } else {
        baseline.profiles().final_group
    };
    ordinary_profile.group = PolynomialGroupLayout::new(18, 2);
    let b = ordinary_profile.outer.matrix;
    ordinary_profile.outer.matrix = akita_params::OuterCommitMatrixParams::try_new(
        b.security_policy(),
        b.sis_table_key().table_digest,
        b.sis_modulus_profile(),
        b.output_rank(),
        ordinary_profile
            .derive_slice_geometry()
            .unwrap()
            .physical_input_width(),
        b.coeff_linf_bound(),
        b.ring_dimension(),
    )
    .unwrap();
    ordinary_profile.validate_frozen_precommit(128).unwrap();
    let key = ScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(18),
        precommitteds: vec![carrier.profiles().final_group, ordinary_profile],
    };
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = akita_config::policy_of::<BinaryTestSuffix>(chunking);
    let contract = BinaryTestSuffix::committed_source_contract().unwrap();
    let schedule = akita_planner::find_schedule_with_root_setup_mode(
        &key,
        contract,
        &[contract; 2],
        &policy,
        BinaryTestSuffix::ring_challenge_config,
        akita_params::SetupContributionMode::Recursive,
        Some(&baseline.schedule().root.params),
    )
    .unwrap()
    .schedule;
    let final_profile = akita_params::GroupCommitPhaseParams::try_from_params(
        key.final_group,
        &schedule.root.params,
    )
    .unwrap();
    let profiles = akita_params::CommittedGroupBatchProfile {
        final_group: final_profile,
        precommitteds: key.precommitteds,
    };
    let rows = vec![
        (carrier.profiles().clone(), carrier.schedule().clone()),
        (final_row.profiles().clone(), final_row.schedule().clone()),
        (profiles, schedule),
    ];
    let catalog = TrustedScheduleCatalog::new(
        akita_schedules::ValidatedScheduleCatalog::try_new(
            BinaryTestSuffix::schedule_family_name(),
            rows,
            &policy,
            BinaryTestSuffix::ring_challenge_config,
        )
        .unwrap(),
        chunking,
    )
    .unwrap();
    (catalog, [ordinary_profile, final_profile])
}

fn mixed_ordinary(unsupported: bool) {
    let (catalog, ordinary_profiles) = mixed_catalog(unsupported);
    let scheme = AkitaCommitmentScheme::<BinaryTestSuffix>::new(catalog);
    let carrier = scheme
        .schedules()
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(23),
        ))
        .unwrap()
        .profiles()
        .final_group;
    let setup = scheme.setup_prover(23, 4).unwrap();
    let backend = CpuBackend::new(setup.expanded.clone()).unwrap();
    let index = BinaryIndexMap::new(8, 128, 8, 2).unwrap();
    let descriptor = BinarySourceDescriptor::new(
        index.clone(),
        vec![[71; 32]],
        BinaryCommittedGroupManifest::commitment_profiles(&index, 1, 0).unwrap(),
    )
    .unwrap();
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier).unwrap();
    let words = claims::words(&index, 0);
    let claim = claims::word_claim(&descriptor, 0, &words);
    let source = backend
        .import_binary_source(manifest.clone(), carrier, vec![words])
        .unwrap();
    let binary = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )
        .unwrap();
    let mut ordinary = Vec::new();
    let mut points = Vec::new();
    let mut evaluations = Vec::new();
    for (group, profile) in ordinary_profiles.iter().enumerate() {
        let point = (0..18)
            .map(|axis| F::from_u64(13 + axis * 7 + group as u64))
            .collect::<Vec<_>>();
        let mut polynomials = Vec::new();
        let mut values = Vec::new();
        for owner in 0..profile.group.num_polynomials() {
            let coefficients = (0..1usize << 18)
                .map(|i| F::from_u64(((i ^ (i >> (owner + 1)) ^ (group + owner)) & 1) as u64))
                .collect::<Vec<_>>();
            let mut reduced = coefficients.clone();
            for &coordinate in &point {
                reduced = reduced
                    .chunks_exact(2)
                    .map(|pair| (F::one() - coordinate) * pair[0] + coordinate * pair[1])
                    .collect();
            }
            values.push(reduced[0]);
            polynomials.push(DensePoly::from_field_evals(18, &coefficients).unwrap());
        }
        let source = backend.import_source(polynomials).unwrap();
        ordinary.push(
            backend
                .commit(scheme.schedules(), &source, GroupContext::explicit(profile))
                .unwrap(),
        );
        points.push(point);
        evaluations.push(values);
    }
    let mut groups =
        vec![
            PolynomialGroupClaims::new_binary(binary.committed_group.clone(), claim.clone())
                .unwrap(),
        ];
    let mut handles = vec![binary.private_handle];
    for ((committed, point), values) in ordinary.iter().zip(&points).zip(&evaluations) {
        groups.push(
            PolynomialGroupClaims::new(
                point.clone(),
                values.clone(),
                committed.committed_group.clone(),
            )
            .unwrap(),
        );
        handles.push(committed.private_handle.clone());
    }
    let data = selected_prover_data::<BinaryTestSuffix>(
        OpeningClaims::from_groups(groups).unwrap(),
        handles,
        scheme.schedules(),
    );
    let resolved = scheme
        .schedules()
        .resolve_selection(data.selection())
        .unwrap();
    let price = akita_planner::prime_planes::price_binary_root_planes(
        &manifest,
        true,
        &PrimePlaneManifest::new(&descriptor, &[]).unwrap(),
        resolved.schedule().recursive_folds[0].params.d_a(),
        ordinary_profiles,
    );
    let first_prefix = resolved.schedule().recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .slot_id()
        .unwrap();
    let later = resolved
        .schedule()
        .recursive_folds
        .iter()
        .skip(1)
        .filter_map(|fold| fold.params.setup_prefix().and_then(|p| p.slot_id()))
        .collect::<Vec<_>>();
    let prefixes = backend
        .prepare_binary_a_prefixes(&manifest, &first_prefix, &later)
        .unwrap();
    let verifier_setup = backend
        .prepare_binary_a_verifier_setup(
            &scheme.setup_verifier(&setup).unwrap(),
            &manifest,
            &first_prefix,
            &prefixes.get(&first_prefix).unwrap().public,
        )
        .unwrap();
    let verifier = scheme.verifier(verifier_setup).unwrap();
    let verify = |proof: &[u8], evaluations: &[Vec<F>]| {
        let mut groups =
            vec![
                PolynomialGroupClaims::new_binary(binary.committed_group.clone(), claim.clone())
                    .unwrap()
                    .map_commitment(|_| &binary.committed_group),
            ];
        for ((committed, point), values) in ordinary.iter().zip(&points).zip(evaluations) {
            groups.push(
                PolynomialGroupClaims::new(
                    point.clone(),
                    values.clone(),
                    &committed.committed_group,
                )
                .unwrap(),
            );
        }
        let statement = selected_statement::<BinaryTestSuffix>(
            OpeningClaims::from_groups(groups).unwrap(),
            scheme.schedules(),
        );
        let mut replay = new_verifier_channel(b"mixed-ordinary-surface", b"caller", proof).unwrap();
        verifier.batched_verify_in_session(&mut replay, statement, BasisMode::Lagrange)?;
        replay.check_eof()
    };
    let mut prover = new_prover_channel(b"mixed-ordinary-surface", b"caller").unwrap();
    let mut registry = akita_prover::BackendRegistry::<BinaryTestSuffix>::new().unwrap();
    let root = registry.register(&backend, &prefixes).unwrap();
    let mut route =
        akita_prover::FixedFoldRoute::new(vec![root; resolved.schedule().num_fold_levels()]);
    let before = prover.narg_string().len();
    let proved = akita_prover::batched_prove_in_session::<BinaryTestSuffix, _>(
        setup.expanded.descriptor(),
        scheme.schedules(),
        &registry,
        data,
        &mut prover,
        BasisMode::Lagrange,
        &mut route,
    );
    if unsupported {
        assert!(matches!(price, Err(AkitaError::InvalidSetup(_))));
        // The prover sends no field-switch partials, and the verifier reads
        // none: an empty proof rejects as setup, not as a malformed proof.
        assert!(matches!(proved, Err(AkitaError::InvalidSetup(_))));
        assert_eq!(prover.narg_string().len(), before);
        assert!(matches!(
            verify(&[], &evaluations),
            Err(AkitaError::InvalidSetup(_))
        ));
        return;
    }
    proved.unwrap();
    let price = price.unwrap();
    assert_eq!(
        price.segment_physical_cells,
        vec![
            4096, 49152, 512, 2304, 1536, 524288, 352256, 65536, 4096, 262144, 176128, 65536, 4096
        ]
    );
    assert_eq!(price.outgoing_physical_cells, 1511680);
    assert_eq!(price.outgoing_domain_len, 2097152);
    assert!(
        price.outgoing_domain_len
            <= akita_params::witness_commitment_domain_len(
                resolved.schedule().root.output_witness_len,
                resolved.schedule().recursive_folds[0].params.d_a()
            )
            .unwrap()
    );
    let proof = prover.narg_string().to_vec();
    eprintln!("grouped_surface ordinary dev={} groups=2 polynomials=2,1 proof={} physical={} domain={} segments={:?}",
        akita_params::DEV_PROTOCOL, proof.len(), price.outgoing_physical_cells, price.outgoing_domain_len, price.segment_physical_cells);
    verify(&proof, &evaluations).unwrap();
    for group in 0..evaluations.len() {
        for owner in 0..evaluations[group].len() {
            let mut altered = evaluations.clone();
            altered[group][owner] += F::one();
            assert!(verify(&proof, &altered).is_err());
        }
    }
}
