use super::*;

#[cfg(feature = "catalog-gen")]
fn policy_for_domain(
    mut policy: PlannerPolicy,
    domain: &RingDimensionSearchDomain,
) -> PlannerPolicy {
    policy.ring_dimension_candidates = Box::leak(domain.candidates().to_vec().into_boxed_slice());
    policy.selection_policy = if domain.candidates().len() > 1 {
        crate::SelectionPolicyId::MinSetupMatrixFieldElementsThenProofPayload
    } else {
        crate::SelectionPolicyId::MinEstimatedProofPayload
    };
    policy
}

#[test]
fn tensor_low_length_is_selected_independently() {
    assert_eq!(
        optimize_fold_challenge_shape(TensorChallengeShape::Tensor { fold_low_len: 1 }, 13,)
            .unwrap(),
        TensorChallengeShape::Tensor { fold_low_len: 4 },
    );
}

#[test]
fn balanced_chunk_geometry_prices_exact_work_and_residual_imbalance() {
    let flat = TensorChallengeShape::Flat;
    assert_eq!(
        layout_candidate_score(100, 13, 3, flat).unwrap(),
        (127, 100, 13, 1)
    );
    assert_eq!(
        layout_candidate_score(100, 12, 3, flat).unwrap(),
        (124, 100, 12, 0)
    );
}

#[test]
fn ring_dimension_domain_is_canonical_and_rejects_invalid_carriers() {
    let domain = RingDimensionSearchDomain::new(
        256,
        [
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
        ],
    )
    .unwrap();
    assert_eq!(
        domain.candidates(),
        &[
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64
            },
        ]
    );
    assert!(RingDimensionSearchDomain::new(
        256,
        [CommitmentRingDims {
            inner: 64,
            outer: 128,
            opening: 64
        }]
    )
    .is_err());
    assert!(RingDimensionSearchDomain::new(128, [CommitmentRingDims::uniform(256)]).is_err());
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_domain_search_beats_or_ties_uniform_d64() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let base_policy = policy_of::<D256OneHot>();
    let dimensions = [
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ];
    let domain = RingDimensionSearchDomain::new(base_policy.ring_dimension, dimensions).unwrap();
    let policy = policy_for_domain(base_policy, &domain);
    let key = PolynomialGroupLayout::singleton(16);
    let selected = find_schedule_mixed_ring(
        key,
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let selected_score = (
        selected.estimate.estimated_setup_envelope_ring_elements,
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
    );

    let uniform = RingDimensionSearchDomain::new(policy.ring_dimension, [dimensions[0]]).unwrap();
    let uniform_policy = policy_for_domain(policy_of::<D256OneHot>(), &uniform);
    let candidate = find_schedule_mixed_ring(
        key,
        &uniform_policy,
        &uniform,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    assert!(
        selected_score
            <= (
                candidate.estimate.estimated_setup_envelope_ring_elements,
                candidate.estimate.estimated_proof_payload_bytes().unwrap(),
            )
    );

    let schedule = &selected.schedule;
    assert!(domain
        .candidates()
        .contains(&schedule.root.params.final_group.commitment.role_dims()));
    let mut previous = schedule.root.params.final_group.commitment.role_dims();
    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        let current = fold.params.witness.role_dims();
        assert!(componentwise_dimensions_at_most(current, previous));
        if index + 1 >= MIXED_SEARCH_FOLD_LEVELS {
            assert_eq!(
                current,
                CommitmentRingDims::uniform(MIXED_SEARCH_SUFFIX_RING_DIMENSION)
            );
        }
        previous = current;
    }
    assert_eq!(
        schedule.terminal.params.witness.d_a(),
        MIXED_SEARCH_SUFFIX_RING_DIMENSION
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn grouped_scalar_fallback_preserves_mixed_domain() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};
    use akita_types::AkitaScheduleLookupKey;

    let base_policy = policy_of::<D256OneHot>();
    let dimensions = [
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ];
    let domain = RingDimensionSearchDomain::new(base_policy.ring_dimension, dimensions).unwrap();
    let policy = policy_for_domain(base_policy, &domain);
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(16));

    let grouped = crate::find_schedule(
        &key,
        &policy,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let direct = find_schedule_mixed_ring(
        key.final_group,
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();

    assert_eq!(
        grouped.schedule.canonical_descriptor_bytes(),
        direct.schedule.canonical_descriptor_bytes()
    );
    assert_eq!(
        grouped.estimate.estimated_proof_payload_bytes(),
        direct.estimate.estimated_proof_payload_bytes()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn pruned_mixed_search_matches_unpruned_traversal_and_is_canonical() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let base_policy = policy_of::<D256OneHot>();
    let d64 = CommitmentRingDims::uniform(64);
    let a128 = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let reversed_with_duplicate =
        RingDimensionSearchDomain::new(base_policy.ring_dimension, [a128, d64, a128]).unwrap();
    let canonical =
        RingDimensionSearchDomain::new(base_policy.ring_dimension, [d64, a128]).unwrap();
    let policy = policy_for_domain(base_policy, &canonical);
    let key = PolynomialGroupLayout::singleton(16);

    let selected = find_schedule_mixed_ring(
        key,
        &policy,
        &reversed_with_duplicate,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let unpruned = unpruned_search::find_schedule(
        key,
        &policy,
        &canonical,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let repeated = find_schedule_mixed_ring(
        key,
        &policy,
        &canonical,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();

    assert_eq!(
        (
            selected.estimate.estimated_setup_envelope_ring_elements,
            selected.estimate.estimated_proof_payload_bytes().unwrap(),
        ),
        (
            unpruned.estimate.estimated_setup_envelope_ring_elements,
            unpruned.estimate.estimated_proof_payload_bytes().unwrap(),
        )
    );
    let selected_descriptor = selected.schedule.canonical_descriptor_bytes();
    assert_eq!(
        selected_descriptor,
        unpruned.schedule.canonical_descriptor_bytes()
    );
    assert_eq!(
        selected_descriptor,
        repeated.schedule.canonical_descriptor_bytes()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_search_parallel_generation_is_descriptor_deterministic() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let handles = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                let base_policy = policy_of::<D256OneHot>();
                let domain = RingDimensionSearchDomain::new(
                    base_policy.ring_dimension,
                    [
                        CommitmentRingDims {
                            inner: 128,
                            outer: 64,
                            opening: 64,
                        },
                        CommitmentRingDims::uniform(64),
                    ],
                )
                .expect("mixed dimension domain");
                let policy = policy_for_domain(base_policy, &domain);
                find_schedule_mixed_ring(
                    PolynomialGroupLayout::singleton(16),
                    &policy,
                    &domain,
                    D256OneHot::ring_challenge_config,
                    D256OneHot::fold_challenge_shape_at_level,
                )
                .expect("parallel mixed planner run")
                .schedule
                .canonical_descriptor_bytes()
            })
        })
        .collect::<Vec<_>>();
    let descriptors = handles
        .into_iter()
        .map(|handle| handle.join().expect("planner thread"))
        .collect::<Vec<_>>();
    assert!(descriptors.windows(2).all(|pair| pair[0] == pair[1]));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_root_prices_eor_at_candidate_a_dimension() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let mut policy = policy_of::<D256OneHot>();
    // D256 enables root projection at this width while the D64 candidate does not.
    policy.claim_ext_degree = 64;
    let candidate_dimensions = CommitmentRingDims::uniform(64);
    let domain = RingDimensionSearchDomain::new(policy.ring_dimension, [candidate_dimensions])
        .expect("mixed dimension domain");
    policy = policy_for_domain(policy, &domain);
    let key = PolynomialGroupLayout::singleton(16);
    let selected = find_schedule_mixed_ring(
        key,
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .expect("mixed planner boundary schedule");
    let schedule = &selected.schedule;
    let root_params = &schedule.root.params.final_group.commitment;
    assert_eq!(root_params.role_dims(), candidate_dimensions);

    let challenge_field_bits = policy.challenge_field_bits().expect("valid policy");
    let candidate_eor_bytes = extension_opening_reduction_level_bytes(
        challenge_field_bits,
        policy.claim_ext_degree,
        0,
        key,
        schedule.root.input_witness_len,
        candidate_dimensions.d_a(),
    )
    .expect("candidate EOR bytes");
    let setup_generation_eor_bytes = extension_opening_reduction_level_bytes(
        challenge_field_bits,
        policy.claim_ext_degree,
        0,
        key,
        schedule.root.input_witness_len,
        policy.ring_dimension,
    )
    .expect("setup-generation EOR bytes");
    assert_eq!(candidate_eor_bytes, 0);
    assert!(setup_generation_eor_bytes > 0);

    let next_params = schedule
        .recursive_folds
        .first()
        .map(|step| &step.params.witness);
    let next_binding = if next_params.is_some() {
        akita_types::NextWitnessBindingPolicy::OuterCommitment
    } else {
        akita_types::NextWitnessBindingPolicy::TerminalInnerState
    };
    let root_without_eor = level_proof_bytes(
        policy.decomposition.field_bits(),
        challenge_field_bits,
        root_params,
        next_params,
        schedule.root.output_witness_len,
        Some(next_binding),
    )
    .expect("root bytes without EOR");
    assert_eq!(
        selected.estimate.estimated_root_direct_payload_bytes,
        root_without_eor + candidate_eor_bytes,
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_search_skips_an_unsupported_sis_candidate_and_keeps_its_sibling() {
    use akita_config::{policy_of, proof_optimized::fp128::D512OneHot, CommitmentConfig};

    let base_policy = policy_of::<D512OneHot>();
    let d64 = CommitmentRingDims::uniform(64);
    let unsupported_uniform_d512 = CommitmentRingDims::uniform(512);
    let domain =
        RingDimensionSearchDomain::new(base_policy.ring_dimension, [d64, unsupported_uniform_d512])
            .expect("mixed domain");
    let policy = policy_for_domain(base_policy, &domain);
    let selected = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &policy,
        &domain,
        D512OneHot::ring_challenge_config,
        D512OneHot::fold_challenge_shape_at_level,
    )
    .expect("the supported D64 sibling must survive");

    assert_eq!(
        selected
            .schedule
            .root
            .params
            .final_group
            .commitment
            .role_dims(),
        d64
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_nv36_benchmark_policy_selects_minimum_setup_schedule() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let base_policy = policy_of::<D256OneHot>();
    let d64 = CommitmentRingDims::uniform(64);
    let d128_mixed = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let d128 = CommitmentRingDims::uniform(128);
    let d256_mixed = CommitmentRingDims {
        inner: 256,
        outer: 128,
        opening: 128,
    };
    let domain = RingDimensionSearchDomain::new(
        base_policy.ring_dimension,
        [d64, d128_mixed, d128, d256_mixed],
    )
    .expect("benchmark dimension domain");
    let policy = policy_for_domain(base_policy, &domain);
    let selected = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(36),
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .expect("nv36 mixed planner");
    let rank_one_capped_domain =
        RingDimensionSearchDomain::new(policy.ring_dimension, [d64, d128_mixed, d128])
            .expect("rank-one-capped comparison domain");
    let mut comparison_policy =
        policy_for_domain(policy_of::<D256OneHot>(), &rank_one_capped_domain);
    comparison_policy.max_setup_envelope_field_elements = usize::MAX;
    let rank_one_capped = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(36),
        &comparison_policy,
        &rank_one_capped_domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .expect("rank-one-capped nv36 planner");
    let selected_root = &selected.schedule.root.params.final_group.commitment;
    let rank_one_capped_root = &rank_one_capped.schedule.root.params.final_group.commitment;

    assert_eq!(selected_root.role_dims(), d256_mixed);
    assert_eq!(
        selected.schedule.recursive_folds[0]
            .params
            .witness
            .role_dims(),
        CommitmentRingDims::uniform(64)
    );
    assert_eq!(
        selected.estimate.estimated_setup_envelope_ring_elements,
        262_144
    );
    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        99_368
    );
    assert_eq!(rank_one_capped_root.inner_commit_matrix.output_rank(), 3);
    assert_eq!(selected_root.inner_commit_matrix.output_rank(), 2);
    assert_eq!(rank_one_capped_root.outer_commit_matrix.output_rank(), 1);
    assert_eq!(selected_root.outer_commit_matrix.output_rank(), 1);
    assert!(
        selected_root.outer_commit_matrix.input_width()
            < rank_one_capped_root.outer_commit_matrix.input_width(),
        "the lower D256 A rank must reduce B width despite both B matrices having rank one"
    );
    assert!(
        selected.estimate.estimated_setup_envelope_ring_elements
            < rank_one_capped
                .estimate
                .estimated_setup_envelope_ring_elements,
        "the rank-two D256 candidate must beat the rank-one-capped search on setup"
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_search_requires_a_monotonic_d64_suffix_domain() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let base_policy = policy_of::<D256OneHot>();
    let missing_d64 = RingDimensionSearchDomain::new(
        base_policy.ring_dimension,
        [CommitmentRingDims::uniform(128)],
    )
    .unwrap();
    let missing_policy = policy_for_domain(base_policy, &missing_d64);
    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &missing_policy,
        &missing_d64,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("requires the D64 uniform candidate"));

    let below_d64 = RingDimensionSearchDomain::new(
        base_policy.ring_dimension,
        [
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 32,
                opening: 64,
            },
        ],
    )
    .unwrap();
    let below_policy = policy_for_domain(base_policy, &below_d64);
    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &below_policy,
        &below_d64,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error.to_string().contains("component-wise at least D64"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_search_rejects_direct_multi_chunk_policy() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let mut policy = policy_of::<D256OneHot>();
    policy.witness_chunk = akita_types::ChunkedWitnessCfg::d64_production();
    let domain =
        RingDimensionSearchDomain::new(policy.ring_dimension, [CommitmentRingDims::uniform(64)])
            .unwrap();
    policy = policy_for_domain(policy, &domain);
    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("does not yet support direct multi-chunk planning"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_search_validates_key_and_policy_at_entry() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let base_policy = policy_of::<D256OneHot>();
    let domain = RingDimensionSearchDomain::new(
        base_policy.ring_dimension,
        [
            CommitmentRingDims::uniform(64),
            CommitmentRingDims::uniform(base_policy.ring_dimension),
        ],
    )
    .unwrap();
    let policy = policy_for_domain(base_policy, &domain);

    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::new(16, 0),
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("opening group layouts must be nonempty"));

    let mut invalid_policy = policy;
    invalid_policy.max_setup_envelope_field_elements = 0;
    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &invalid_policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("maximum setup envelope must be positive"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn dimension_domain_is_bound_to_setup_generation_dimension() {
    use akita_config::{policy_of, proof_optimized::fp128::D128OneHot, CommitmentConfig};

    let policy = policy_of::<D128OneHot>();
    let domain = RingDimensionSearchDomain::uniform(256).unwrap();
    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &policy,
        &domain,
        D128OneHot::ring_challenge_config,
        D128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("domain uses setup generation D256, but policy uses D128"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_search_applies_setup_budget_in_physical_fields() {
    use akita_config::{policy_of, proof_optimized::fp128::D256OneHot, CommitmentConfig};

    let mut policy = policy_of::<D256OneHot>();
    let domain = RingDimensionSearchDomain::new(
        policy.ring_dimension,
        [
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
        ],
    )
    .unwrap();
    policy = policy_for_domain(policy, &domain);
    let selected = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let exact_fields =
        akita_types::setup_matrix_field_elements_for_schedule(&selected.schedule).unwrap();
    policy.max_setup_envelope_field_elements = exact_fields - 1;

    let error = find_schedule_mixed_ring(
        PolynomialGroupLayout::singleton(16),
        &policy,
        &domain,
        D256OneHot::ring_challenge_config,
        D256OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error.to_string().contains("no mixed-D schedule"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn preserved_recursive_proof_size_is_documented() {
    use akita_config::{
        policy_of, proof_optimized::fp128::D64OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::{AkitaScheduleLookupKey, PrecommittedGroupDescriptor};

    type Recursive = RecursiveCommitmentConfig<D64OneHot>;
    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let precommit_policy = policy_of::<D64OneHot>();
    let precommit = find_schedule_singular(
        precommit_layout,
        &precommit_policy,
        D64OneHot::ring_challenge_config,
        D64OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let descriptor = PrecommittedGroupDescriptor::from_params(
        precommit_layout,
        &precommit.schedule.root.params.final_group.commitment,
    );
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    let planned = crate::find_schedule(
        &key,
        &policy_of::<Recursive>(),
        Recursive::ring_challenge_config,
        Recursive::fold_challenge_shape_at_level,
    )
    .unwrap();

    assert_eq!(
        planned.estimate.estimated_proof_payload_bytes().unwrap(),
        102_732
    );
}
