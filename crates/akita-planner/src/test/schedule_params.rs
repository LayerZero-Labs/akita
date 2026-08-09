use super::*;
#[cfg(feature = "catalog-gen")]
use akita_types::extension_opening_reduction_level_bytes;

#[cfg(test)]
fn find_schedule(
    key: PolynomialGroupLayout,
    policy: &PlannerPolicy,
    honest_fold_policy: HonestFoldPolicySpec,
    dimensions: &RingDimensionSearchDomain,
    ring_challenge_config: impl Fn(usize) -> Result<akita_challenges::SparseChallengeConfig, AkitaError>,
) -> Result<PlannedFoldSchedule, AkitaError> {
    dimensions.validate_for_policy(policy)?;
    crate::planner::find_schedule(
        &akita_types::AkitaScheduleLookupKey::single(key),
        honest_fold_policy,
        &[],
        policy,
        ring_challenge_config,
    )
}

#[cfg(test)]
fn policy_for_domain(
    mut policy: PlannerPolicy,
    domain: &RingDimensionSearchDomain,
) -> PlannerPolicy {
    let is_uniform =
        domain.candidates() == [CommitmentRingDims::uniform(policy.uniform_ring_dimension)];
    policy.ring_dimension_schedule_mode = if is_uniform {
        crate::RingDimensionScheduleMode::UniformDimension {
            ring_dimension: policy.uniform_ring_dimension,
        }
    } else {
        let mut a = domain
            .candidates()
            .iter()
            .map(|dims| dims.d_a())
            .collect::<Vec<_>>();
        let mut b = domain
            .candidates()
            .iter()
            .map(|dims| dims.d_b())
            .collect::<Vec<_>>();
        let mut d = domain
            .candidates()
            .iter()
            .map(|dims| dims.d_d())
            .collect::<Vec<_>>();
        for dimensions in [&mut a, &mut b, &mut d] {
            dimensions.sort_unstable();
            dimensions.dedup();
        }
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels: 2,
            uniform_suffix_dimension: 64,
            potential_a_dimensions: Box::leak(a.into_boxed_slice()),
            potential_b_dimensions: Box::leak(b.into_boxed_slice()),
            potential_d_dimensions: Box::leak(d.into_boxed_slice()),
        }
    };
    policy.selection_policy = crate::SelectionPolicyId::for_policy(
        policy.recursive_setup_planning,
        policy.ring_dimension_schedule_mode,
    );
    policy
}

#[test]
fn dyadic_chunk_geometry_prices_exact_work_and_residual_imbalance() {
    assert_eq!(
        layout_candidate_score(100, 13, 4).unwrap(),
        (127, 100, 13, 1)
    );
    assert_eq!(
        layout_candidate_score(100, 12, 4).unwrap(),
        (124, 100, 12, 0)
    );
}

#[test]
fn ring_dimension_domain_is_canonical_and_rejects_invalid_carriers() {
    let domain = RingDimensionSearchDomain::new([
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
    ])
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
    assert!(RingDimensionSearchDomain::new([CommitmentRingDims {
        inner: 64,
        outer: 128,
        opening: 64
    }])
    .is_err());
    assert!(RingDimensionSearchDomain::new([CommitmentRingDims::uniform(256)]).is_ok());
}

#[cfg(feature = "catalog-gen")]
#[test]
fn mixed_domain_search_beats_or_ties_uniform_d64() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let dimensions = [
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ];
    let domain = RingDimensionSearchDomain::new(dimensions).unwrap();
    let policy = policy_for_domain(base_policy, &domain);
    let key = PolynomialGroupLayout::singleton(16);
    let selected = find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let selected_score = (
        selected
            .estimate
            .estimated_num_setup_field_elements
            .div_ceil(policy.uniform_ring_dimension),
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
    );

    let uniform = RingDimensionSearchDomain::uniform(dimensions[0].d_a()).unwrap();
    let mut uniform_policy = policy_of::<OneHot>();
    uniform_policy.uniform_ring_dimension = dimensions[0].d_a();
    uniform_policy.ring_dimension_schedule_mode =
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension: 64 };
    uniform_policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayload;
    let candidate = find_schedule(
        key,
        &uniform_policy,
        OneHot::root_honest_fold_policy(),
        &uniform,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(
        selected_score
            <= (
                candidate
                    .estimate
                    .estimated_num_setup_field_elements
                    .div_ceil(uniform_policy.uniform_ring_dimension),
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
        if index + 1 >= akita_schedules::ADAPTIVE_SEARCH_LEVELS {
            assert_eq!(
                current,
                CommitmentRingDims::uniform(ADAPTIVE_SUFFIX_RING_DIMENSION)
            );
        }
        previous = current;
    }
    assert_eq!(
        schedule.terminal.params.witness.d_a(),
        ADAPTIVE_SUFFIX_RING_DIMENSION
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_dimension_search_is_canonical() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let d64 = CommitmentRingDims::uniform(64);
    let a128 = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let reversed_with_duplicate = RingDimensionSearchDomain::new([a128, d64, a128]).unwrap();
    let canonical = RingDimensionSearchDomain::new([d64, a128]).unwrap();
    let policy = policy_for_domain(base_policy, &canonical);
    let key = PolynomialGroupLayout::singleton(16);

    let selected = find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &reversed_with_duplicate,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let repeated = find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &canonical,
        OneHot::ring_challenge_config,
    )
    .unwrap();

    let selected_descriptor = selected.schedule.canonical_descriptor_bytes();
    assert_eq!(
        selected_descriptor,
        repeated.schedule.canonical_descriptor_bytes()
    );
}

#[test]
fn uniform_suffix_dp_matches_unpruned_exact_cutover_search() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domain = RingDimensionSearchDomain::uniform(64).unwrap();
    let mut base_policy = policy_of::<OneHot>();
    base_policy.uniform_ring_dimension = 64;
    let policy = policy_for_domain(base_policy, &domain);
    let key = PolynomialGroupLayout::singleton(16);
    let selected = find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let unpruned = unpruned_search::find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();

    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        unpruned.estimate.estimated_proof_payload_bytes().unwrap()
    );
    assert_eq!(
        selected.schedule.canonical_descriptor_bytes(),
        unpruned.schedule.canonical_descriptor_bytes()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_frontier_matches_unpruned_l0_l1_search() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ])
    .expect("representative adaptive domain");
    let policy = policy_for_domain(policy_of::<OneHot>(), &domain);
    let key = PolynomialGroupLayout::singleton(16);
    let selected = find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("frontier search");
    let unpruned = unpruned_search::find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("unpruned adaptive search");

    assert_eq!(
        selected.estimate.estimated_num_setup_field_elements,
        unpruned.estimate.estimated_num_setup_field_elements
    );
    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        unpruned.estimate.estimated_proof_payload_bytes().unwrap()
    );
    assert_eq!(
        selected.schedule.canonical_descriptor_bytes(),
        unpruned.schedule.canonical_descriptor_bytes()
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_parallel_generation_is_descriptor_deterministic() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let handles = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                let base_policy = policy_of::<OneHot>();
                let domain = RingDimensionSearchDomain::new([
                    CommitmentRingDims {
                        inner: 128,
                        outer: 64,
                        opening: 64,
                    },
                    CommitmentRingDims::uniform(64),
                ])
                .expect("mixed dimension domain");
                let policy = policy_for_domain(base_policy, &domain);
                find_schedule(
                    PolynomialGroupLayout::singleton(16),
                    &policy,
                    OneHot::root_honest_fold_policy(),
                    &domain,
                    OneHot::ring_challenge_config,
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
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let mut policy = policy_of::<OneHot>();
    // D256 enables root projection at this width while the D64 candidate does not.
    policy.claim_ext_degree = 64;
    let candidate_dimensions = CommitmentRingDims::uniform(64);
    let domain =
        RingDimensionSearchDomain::new([candidate_dimensions]).expect("mixed dimension domain");
    policy = policy_for_domain(policy, &domain);
    let key = PolynomialGroupLayout::singleton(16);
    let selected = find_schedule(
        key,
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
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
    let uniform_eor_bytes = extension_opening_reduction_level_bytes(
        challenge_field_bits,
        policy.claim_ext_degree,
        0,
        key,
        schedule.root.input_witness_len,
        policy.uniform_ring_dimension,
    )
    .expect("setup-generation EOR bytes");
    assert_eq!(candidate_eor_bytes, 0);
    assert!(uniform_eor_bytes > 0);

    let next_params = schedule
        .recursive_folds
        .first()
        .map(|step| &step.params.witness);
    let next_binding = if next_params.is_some() {
        akita_types::NextWitnessBindingPolicy::OuterPayload
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
fn adaptive_search_rejects_an_advertised_unsupported_role_dimension() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let mut base_policy = policy_of::<OneHot>();
    base_policy.uniform_ring_dimension = 512;
    let d64 = CommitmentRingDims::uniform(64);
    let unsupported_uniform_d512 = CommitmentRingDims::uniform(512);
    let domain =
        RingDimensionSearchDomain::new([d64, unsupported_uniform_d512]).expect("mixed domain");
    let policy = policy_for_domain(base_policy, &domain);
    let error = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect_err("an unsupported advertised B/D dimension must reject the policy");
    assert!(error.to_string().contains("scheduled B dimension D512"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_nv36_minimizes_setup_before_proof_bytes() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
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
    let domain = RingDimensionSearchDomain::new([d64, d128_mixed, d128, d256_mixed])
        .expect("benchmark dimension domain");
    let policy = policy_for_domain(base_policy, &domain);
    let selected = find_schedule(
        PolynomialGroupLayout::singleton(36),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("nv36 mixed planner");
    let rank_one_capped_domain = RingDimensionSearchDomain::new([d64, d128_mixed, d128])
        .expect("rank-one-capped comparison domain");
    let mut comparison_policy = policy_for_domain(policy_of::<OneHot>(), &rank_one_capped_domain);
    comparison_policy.setup_field_budget = None;
    let rank_one_capped = find_schedule(
        PolynomialGroupLayout::singleton(36),
        &comparison_policy,
        OneHot::root_honest_fold_policy(),
        &rank_one_capped_domain,
        OneHot::ring_challenge_config,
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
        d64
    );
    assert_eq!(rank_one_capped_root.inner_commit_matrix.output_rank(), 3);
    assert_eq!(selected_root.inner_commit_matrix.output_rank(), 1);
    assert_eq!(rank_one_capped_root.outer_commit_matrix.output_rank(), 1);
    assert_eq!(selected_root.outer_commit_matrix.output_rank(), 1);
    assert!(
        selected_root.outer_commit_matrix.input_width()
            < rank_one_capped_root.outer_commit_matrix.input_width(),
        "the lower D256 A rank must reduce B width despite both B matrices having rank one"
    );
    assert!(
        selected.estimate.estimated_num_setup_field_elements
            < rank_one_capped.estimate.estimated_num_setup_field_elements,
        "the D256-root candidate must beat the restricted domain on setup fields"
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_requires_a_monotonic_d64_suffix_domain() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let missing_d64 = RingDimensionSearchDomain::new([CommitmentRingDims::uniform(128)]).unwrap();
    let missing_policy = policy_for_domain(base_policy, &missing_d64);
    let error = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &missing_policy,
        OneHot::root_honest_fold_policy(),
        &missing_d64,
        OneHot::ring_challenge_config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("must contain suffix D64"));

    let below_d64 = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 32,
            opening: 64,
        },
    ])
    .unwrap();
    let below_policy = policy_for_domain(base_policy, &below_d64);
    let error = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &below_policy,
        OneHot::root_honest_fold_policy(),
        &below_d64,
        OneHot::ring_challenge_config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("scheduled B dimension D32"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_supports_direct_multi_chunk_policy() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let mut policy = policy_of::<OneHot>();
    policy.witness_chunk = akita_types::ChunkedWitnessCfg::d64_production();
    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims::uniform(128),
        CommitmentRingDims::uniform(256),
    ])
    .unwrap();
    policy = policy_for_domain(policy, &domain);
    let schedule = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(!schedule.schedule.recursive_folds.is_empty());
    assert_eq!(
        schedule
            .schedule
            .root
            .params
            .final_group
            .commitment
            .witness_chunk
            .num_chunks,
        8
    );
    assert_eq!(
        schedule.schedule.recursive_folds[0]
            .params
            .witness
            .witness_chunk
            .num_chunks,
        8
    );
    assert!(schedule
        .schedule
        .recursive_folds
        .iter()
        .skip(1)
        .all(|fold| fold.params.witness.witness_chunk.num_chunks == 1));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_validates_key_and_policy_at_entry() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims::uniform(base_policy.uniform_ring_dimension),
    ])
    .unwrap();
    let policy = policy_for_domain(base_policy, &domain);

    let error = find_schedule(
        PolynomialGroupLayout::new(16, 0),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("opening group layouts must be nonempty"));

    let mut invalid_policy = policy;
    invalid_policy.setup_field_budget = Some(0);
    let error = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &invalid_policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("explicit setup field budget must be positive"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_root_domain_is_independent_of_uniform_config_dimension() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 64,
        },
    ])
    .expect("supported fp128 adaptive domain");
    let mut base_policy = policy_of::<OneHot>();
    base_policy.uniform_ring_dimension = 64;
    let policy = policy_for_domain(base_policy, &domain);
    domain.validate_for_policy(&policy).unwrap();

    let selected = find_schedule(
        PolynomialGroupLayout::singleton(36),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("D256 A search must not be capped by uniform D64");
    assert_eq!(
        selected.schedule.root.params.final_group.commitment.d_a(),
        256
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_applies_setup_budget_in_physical_fields() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let mut policy = policy_of::<OneHot>();
    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ])
    .unwrap();
    policy = policy_for_domain(policy, &domain);
    let selected = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let exact_fields =
        akita_types::setup_matrix_field_elements_for_schedule(&selected.schedule).unwrap();
    policy.setup_field_budget = Some(exact_fields - 1);

    let error = find_schedule(
        PolynomialGroupLayout::singleton(16),
        &policy,
        OneHot::root_honest_fold_policy(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap_err();
    assert!(error.to_string().contains("no mixed-D schedule"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn exact_payload_ties_prefer_the_smaller_setup_envelope() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHotMultiChunkW4R2, CommitmentConfig};

    let domain = RingDimensionSearchDomain::uniform(64).unwrap();
    // Production W4R2 is adaptive now. Keep this regression on its original
    // fixed-D64 domain, where two equal-payload schedules differ in setup size.
    let mut base_policy = policy_of::<OneHotMultiChunkW4R2>();
    base_policy.uniform_ring_dimension = 64;
    let policy = policy_for_domain(base_policy, &domain);
    let selected = find_schedule(
        PolynomialGroupLayout::singleton(32),
        &policy,
        OneHotMultiChunkW4R2::root_honest_fold_policy(),
        &domain,
        OneHotMultiChunkW4R2::ring_challenge_config,
    )
    .expect("W4R2 schedule");

    assert_eq!(
        selected.estimate.estimated_num_setup_field_elements,
        22_544_384
    );
    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        88_888
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn recursive_exact_cutover_proof_size_is_documented() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let descriptor = derive_standalone_precommit_profile(
        precommit_layout,
        &policy_of::<OneHot>(),
        OneHot::root_honest_fold_policy(),
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert_eq!(descriptor.inner_commit_matrix.ring_dimension(), 64);
    assert_eq!(descriptor.outer_commit_matrix.ring_dimension(), 64);
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    let planned = crate::find_schedule(
        &key,
        Recursive::root_honest_fold_policy(),
        &[
            OneHot::root_honest_fold_policy(),
            OneHot::root_honest_fold_policy(),
        ],
        &policy_of::<Recursive>(),
        Recursive::ring_challenge_config,
    )
    .unwrap();

    assert_eq!(
        planned.estimate.estimated_proof_payload_bytes().unwrap(),
        91_832
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn recursive_adaptive_search_selects_schedule_dimensions_and_setup_prefixes() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let descriptor = derive_standalone_precommit_profile(
        precommit_layout,
        &policy_of::<OneHot>(),
        OneHot::root_honest_fold_policy(),
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert_eq!(descriptor.inner_commit_matrix.ring_dimension(), 64);
    assert_eq!(descriptor.outer_commit_matrix.ring_dimension(), 64);
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let planned = crate::find_schedule(
        &key,
        Recursive::root_honest_fold_policy(),
        &[
            OneHot::root_honest_fold_policy(),
            OneHot::root_honest_fold_policy(),
        ],
        &policy_of::<Recursive>(),
        Recursive::ring_challenge_config,
    )
    .unwrap();

    assert!(planned.estimate.selected_offload_edges > 0);
    assert_eq!(
        planned
            .schedule
            .root
            .params
            .final_group
            .commitment
            .role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(
        planned
            .schedule
            .root
            .params
            .final_group
            .commitment
            .open_commit_matrix
            .input_width(),
        176_472,
        "root D width projects the main group once and then adds both frozen precommit segments"
    );
    assert_eq!(
        planned.schedule.recursive_folds[0]
            .params
            .witness
            .role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(
        planned.schedule.recursive_folds[1]
            .params
            .witness
            .role_dims(),
        CommitmentRingDims::uniform(64)
    );
    let mut previous = planned
        .schedule
        .root
        .params
        .final_group
        .commitment
        .role_dims();
    for fold in &planned.schedule.recursive_folds {
        let current = fold.params.witness.role_dims();
        assert!(current.d_a() <= previous.d_a());
        assert!(current.d_b() <= previous.d_b());
        assert!(current.d_d() <= previous.d_d());
        if let Some(prefix) = &fold.params.witness.setup_prefix {
            assert_eq!(
                prefix
                    .commitment_params
                    .layout
                    .inner_commit_matrix
                    .ring_dimension(),
                current.d_a()
            );
            assert_eq!(
                prefix
                    .commitment_params
                    .layout
                    .outer_commit_matrix
                    .ring_dimension(),
                current.d_b()
            );
        }
        previous = current;
    }
}
