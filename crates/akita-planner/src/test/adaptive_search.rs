use super::*;
#[path = "adaptive_search/relation_cutover.rs"]
mod relation_cutover;
#[path = "adaptive_search/selective_l2.rs"]
mod selective_l2;
#[cfg(feature = "catalog-gen")]
use akita_types::extension_opening_reduction_level_bytes;

fn onehot_group(num_vars: usize, num_polynomials: usize) -> PolynomialGroupLayout {
    PolynomialGroupLayout::new(num_vars, num_polynomials)
}

fn estimated_first_direct_setup_capacity(planned: &PlannedFoldSchedule) -> usize {
    akita_types::padded_setup_prefix_len(
        planned
            .estimate
            .first_direct_setup_field_len
            .expect("setup-first plan must report its first direct setup length"),
    )
}

fn materialized_first_direct_setup_capacity(
    planned: &PlannedFoldSchedule,
    key: PolynomialGroupLayout,
) -> usize {
    akita_schedules::planner_support::first_direct_setup_capacity_for_schedule(
        &planned.schedule,
        &akita_types::AkitaScheduleLookupKey::single(key)
            .opening_layout()
            .expect("opening layout"),
    )
    .expect("materialized first direct setup capacity")
}

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
    let uniform_dimension = domain.candidates().first().and_then(|first| {
        domain
            .candidates()
            .iter()
            .all(|candidate| {
                candidate == first && first.inner == first.outer && first.outer == first.opening
            })
            .then_some(first.d_a())
    });
    policy.ring_dimension_schedule_mode = if let Some(ring_dimension) = uniform_dimension {
        crate::RingDimensionScheduleMode::UniformDimension { ring_dimension }
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
            suffix_dimensions: &[64],
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
    let key = onehot_group(15, 4);
    let selected = find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let selected_score = (
        selected.estimate.estimated_num_setup_field_elements,
        estimated_first_direct_setup_capacity(&selected),
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
    );

    let uniform = RingDimensionSearchDomain::uniform(dimensions[0].d_a()).unwrap();
    let mut uniform_policy = policy_of::<OneHot>();
    uniform_policy.ring_dimension_schedule_mode =
        crate::RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels: 2,
            suffix_dimensions: &[64],
            potential_a_dimensions: &[64],
            potential_b_dimensions: &[64],
            potential_d_dimensions: &[64],
        };
    uniform_policy.selection_policy = crate::SelectionPolicyId::for_policy(
        uniform_policy.recursive_setup_planning,
        uniform_policy.ring_dimension_schedule_mode,
    );
    let candidate = find_schedule(
        key,
        &uniform_policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &uniform,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(
        selected_score
            <= (
                candidate.estimate.estimated_num_setup_field_elements,
                materialized_first_direct_setup_capacity(&candidate, key),
                candidate.estimate.estimated_proof_payload_bytes().unwrap(),
            )
    );

    let schedule = &selected.schedule;
    assert!(domain
        .candidates()
        .contains(&schedule.root.params.role_dims()));
    let mut previous = schedule.root.params.role_dims();
    for (index, fold) in schedule.recursive_folds.iter().enumerate() {
        let current = fold.params.role_dims();
        assert!(componentwise_dimensions_at_most(current, previous));
        if index + 1 >= akita_schedules::ADAPTIVE_SEARCH_LEVELS {
            assert_eq!(
                current,
                CommitmentRingDims::uniform(ADAPTIVE_SUFFIX_RING_DIMENSION)
            );
        }
        previous = current;
    }
    if schedule.recursive_folds.len() + 1 >= akita_schedules::ADAPTIVE_SEARCH_LEVELS {
        assert_eq!(
            schedule.terminal.d_a(),
            ADAPTIVE_SUFFIX_RING_DIMENSION,
            "a terminal beyond the adaptive prefix must use the audited suffix dimension"
        );
    }
}

#[cfg(feature = "catalog-gen")]
#[test]
fn proof_first_uniform_search_matches_unpruned_descriptor() {
    use akita_config::{policy_of, proof_optimized::fp32::OneHot, CommitmentConfig};

    // fp32 has extension degree four, so production s >= 64 requires d_A >= 256.
    let dimensions = RingDimensionSearchDomain::uniform(256).unwrap();
    let mut policy = policy_of::<OneHot>();
    policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::UniformDimension {
        ring_dimension: 256,
    };
    policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayloadV2;
    policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
    let selected = find_schedule(
        onehot_group(14, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &dimensions,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(
        selected.schedule.recursive_folds.len() < unpruned_search::MAX_ORACLE_RECURSION_DEPTH,
        "bounded oracle must cover the production winner's recursion depth",
    );
    let unpruned = unpruned_search::find_schedule(
        onehot_group(14, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let unpruned = &unpruned.planned;
    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        unpruned.estimate.estimated_proof_payload_bytes().unwrap(),
    );
    assert_eq!(
        selected.estimate.first_direct_setup_field_len,
        unpruned.estimate.first_direct_setup_field_len,
    );
    assert_eq!(selected.estimate.first_direct_setup_field_len, None);
    assert_eq!(
        selected.estimate.estimated_num_setup_field_elements,
        unpruned.estimate.estimated_num_setup_field_elements,
    );
    assert_eq!(
        selected.schedule.canonical_descriptor_bytes(),
        unpruned.schedule.canonical_descriptor_bytes(),
    );
    let root = &selected.schedule.root.params;
    assert!(matches!(
        root.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ));
    let terminal_eor = extension_opening_reduction_level_bytes(
        policy.challenge_field_bits().unwrap(),
        policy.claim_ext_degree,
        akita_types::PolynomialGroupLayout::singleton(
            akita_types::padded_boolean_opening_vars(selected.schedule.terminal.input_witness_len)
                .unwrap(),
        ),
    )
    .unwrap();
    assert!(terminal_eor > 0, "the ET terminal must retain its EOR");
    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        akita_schedules::expanded_schedule_proof_payload_bytes(
            &akita_types::AkitaScheduleLookupKey::single(onehot_group(14, 1)),
            &selected.schedule,
            &policy,
        )
        .unwrap(),
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn statically_infeasible_early_packing_domain_is_unsupported() {
    use akita_config::{policy_of, proof_optimized::fp32::OneHot, CommitmentConfig};

    let dimensions = RingDimensionSearchDomain::uniform(128).unwrap();
    let mut policy = policy_of::<OneHot>();
    policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::UniformDimension {
        ring_dimension: 128,
    };
    policy.selection_policy = crate::SelectionPolicyId::MinEstimatedProofPayloadV2;
    policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
    let error = find_schedule(
        onehot_group(14, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &dimensions,
        OneHot::ring_challenge_config,
    )
    .expect_err("an early fold without packing geometry must be unsupported");
    assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
    let error = unpruned_search::find_schedule(
        onehot_group(14, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        OneHot::ring_challenge_config,
    )
    .expect_err("the bounds-disabled oracle must use the same hard packing policy");
    assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn feasible_packing_dimension_ignores_infeasible_smaller_dimensions() {
    use akita_config::{policy_of, proof_optimized::fp32::OneHot, CommitmentConfig};

    let dimensions = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims::uniform(128),
        CommitmentRingDims::uniform(256),
    ])
    .unwrap();
    let mut policy = policy_for_domain(policy_of::<OneHot>(), &dimensions);
    policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
    let admitted_dimensions =
        dimension_candidates(&policy, 0, initial_dimension_ceiling(&policy).unwrap())
            .expect("adaptive dimension domain");
    assert!(
        admitted_dimensions
            .iter()
            .any(|dims| dims.d_a() != dims.d_b() || dims.d_a() != dims.d_d()),
        "the adaptive policy must include mixed A/B/D tuples"
    );
    let key = onehot_group(14, 1);
    let selected = find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &dimensions,
        OneHot::ring_challenge_config,
    )
    .expect("packing schedule from mixed domain");
    assert!(matches!(
        selected.schedule.root.params.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ));
    let unpruned = unpruned_search::find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        OneHot::ring_challenge_config,
    )
    .expect("unpruned packing schedule from mixed domain");
    let unpruned = &unpruned.planned;
    assert!(matches!(
        unpruned.schedule.root.params.opening_method(),
        akita_types::OpeningMethod::SubringCoefficientPacking { .. }
    ));
    assert_eq!(
        selected.estimate.first_direct_setup_field_len,
        unpruned.estimate.first_direct_setup_field_len,
    );
    assert_eq!(
        estimated_first_direct_setup_capacity(&selected),
        estimated_first_direct_setup_capacity(unpruned),
    );
    assert_eq!(
        selected.estimate.estimated_num_setup_field_elements,
        unpruned.estimate.estimated_num_setup_field_elements,
    );
    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        unpruned.estimate.estimated_proof_payload_bytes().unwrap(),
    );
    assert_eq!(
        selected.schedule.canonical_descriptor_bytes(),
        unpruned.schedule.canonical_descriptor_bytes(),
    );
    assert!(selected
        .schedule
        .recursive_folds
        .iter()
        .all(|fold| matches!(
            fold.params.opening_method(),
            akita_types::OpeningMethod::SubringCoefficientPacking { .. }
        )));
}

#[test]
fn adaptive_initial_ceiling_is_componentwise() {
    use akita_config::{policy_of, proof_optimized::fp128::Dense};

    const A: &[usize] = &[64, 256, 1024];
    const B: &[usize] = &[64, 128, 256];
    const D: &[usize] = &[64, 128];
    let mut policy = policy_of::<Dense>();
    policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels: 2,
        suffix_dimensions: &[64],
        potential_a_dimensions: A,
        potential_b_dimensions: B,
        potential_d_dimensions: D,
    };

    assert_eq!(
        initial_dimension_ceiling(&policy).unwrap(),
        CommitmentRingDims {
            inner: 1024,
            outer: 256,
            opening: 128,
        }
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
    let key = onehot_group(16, 1);

    let selected = find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &reversed_with_duplicate,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let repeated = find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
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

#[cfg(feature = "catalog-gen")]
#[test]
fn production_suffix_selects_l2_with_the_typed_response_model() {
    use akita_config::{policy_of, proof_optimized::fp128, CommitmentConfig};
    use akita_types::InnerCommitSecurityRoute;

    let domain = RingDimensionSearchDomain::uniform(64).expect("test domain");
    let fp128_policy = policy_of::<fp128::OneHot>();
    let selected = find_schedule(
        onehot_group(40, 1),
        &fp128_policy,
        akita_config::honest_fold_policy_of::<fp128::OneHot>(),
        &domain,
        fp128::OneHot::ring_challenge_config,
    )
    .expect("shipped fp128 selective L2 schedule");
    assert!(selected.schedule.recursive_folds.iter().any(|step| {
        matches!(
            step.params.inner().matrix.security_route(),
            InnerCommitSecurityRoute::L2 { .. }
        )
    }));
    assert!(
        selected.schedule.terminal.response_l2_sq_cap().is_some(),
        "terminal-only planning must preserve the PR369 selective-L2 route",
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn bounded_recursive_setup_search_matches_exhaustive_on_small_fixture() {
    use akita_config::{
        honest_fold_policy_of, policy_of, proof_optimized::fp128::OneHot, CommitmentConfig,
        RecursiveCommitmentConfig,
    };

    type Recursive = RecursiveCommitmentConfig<OneHot>;

    let domain = RingDimensionSearchDomain::uniform(64).unwrap();
    let bounded = policy_for_domain(policy_of::<Recursive>(), &domain);
    assert_eq!(
        bounded.recursive_setup_search_policy,
        crate::RecursiveSetupSearchPolicy::RootAndFirstChildV1
    );
    let mut exhaustive = bounded;
    exhaustive.recursive_setup_search_policy = crate::RecursiveSetupSearchPolicy::Exhaustive;
    let key = onehot_group(16, 1);
    let bounded_schedule = find_schedule(
        key,
        &bounded,
        honest_fold_policy_of::<Recursive>(),
        &domain,
        Recursive::ring_challenge_config,
    )
    .expect("bounded recursive setup search");
    let exhaustive_schedule = find_schedule(
        key,
        &exhaustive,
        honest_fold_policy_of::<Recursive>(),
        &domain,
        Recursive::ring_challenge_config,
    )
    .expect("exhaustive recursive setup search");

    assert_eq!(
        bounded_schedule.schedule.canonical_descriptor_bytes(),
        exhaustive_schedule.schedule.canonical_descriptor_bytes()
    );
    assert_eq!(bounded_schedule.estimate, exhaustive_schedule.estimate);
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_frontier_matches_unpruned_traversal_and_hand_priced_role_optima() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domains = [
        RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            CommitmentRingDims {
                inner: 128,
                outer: 128,
                opening: 64,
            },
        ])
        .expect("B-varying adaptive domain"),
        RingDimensionSearchDomain::new([
            CommitmentRingDims::uniform(64),
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 128,
            },
        ])
        .expect("D-varying adaptive domain"),
    ];

    // This isolates adaptive frontier pruning from selective-L2 candidate
    // enumeration, which has separate global-selection regressions below.
    let expected_root_dimensions = [
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
    ];

    for (domain_index, (domain, expected_root)) in domains
        .into_iter()
        .zip(expected_root_dimensions)
        .enumerate()
    {
        let mut base_policy = policy_of::<OneHot>();
        base_policy.selective_l2_response_model = crate::SelectiveL2ResponseModelId::Disabled;
        let policy = policy_for_domain(base_policy, &domain);
        let key = onehot_group(14, 1);
        let selected = find_schedule(
            key,
            &policy,
            akita_config::honest_fold_policy_of::<OneHot>(),
            &domain,
            OneHot::ring_challenge_config,
        )
        .expect("frontier search");
        let unpruned = unpruned_search::find_schedule(
            key,
            &policy,
            akita_config::honest_fold_policy_of::<OneHot>(),
            OneHot::ring_challenge_config,
        )
        .expect("unpruned adaptive search");
        let unpruned = &unpruned.planned;

        assert_eq!(
            selected.schedule.root.params.role_dims(),
            expected_root,
            "hand-priced role optimum changed"
        );

        assert_eq!(
            selected.estimate.first_direct_setup_field_len,
            unpruned.estimate.first_direct_setup_field_len,
            "domain {domain_index} first direct natural setup length"
        );
        assert_eq!(
            estimated_first_direct_setup_capacity(&selected),
            estimated_first_direct_setup_capacity(unpruned),
            "domain {domain_index} first direct padded setup capacity"
        );
        assert_eq!(
            selected.estimate.estimated_num_setup_field_elements,
            unpruned.estimate.estimated_num_setup_field_elements,
            "domain {domain_index} setup"
        );
        assert_eq!(
            selected.estimate.estimated_proof_payload_bytes().unwrap(),
            unpruned.estimate.estimated_proof_payload_bytes().unwrap(),
            "domain {domain_index} payload"
        );
        assert_eq!(
            selected.schedule.canonical_descriptor_bytes(),
            unpruned.schedule.canonical_descriptor_bytes()
        );
    }
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
                    onehot_group(16, 1),
                    &policy,
                    akita_config::honest_fold_policy_of::<OneHot>(),
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
fn adaptive_search_rejects_an_advertised_unsupported_role_dimension() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let d64 = CommitmentRingDims::uniform(64);
    let unsupported_uniform_d512 = CommitmentRingDims::uniform(512);
    let domain =
        RingDimensionSearchDomain::new([d64, unsupported_uniform_d512]).expect("mixed domain");
    let policy = policy_for_domain(base_policy, &domain);
    let error = find_schedule(
        onehot_group(16, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect_err("an unsupported advertised B/D dimension must reject the policy");
    assert!(error.to_string().contains("scheduled B dimension D512"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_nv36_minimizes_setup_envelope_before_first_direct_setup() {
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
        onehot_group(36, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("nv36 mixed planner");
    let rank_one_capped_domain = RingDimensionSearchDomain::new([d64, d128_mixed, d128])
        .expect("rank-one-capped comparison domain");
    let mut comparison_policy = policy_for_domain(policy_of::<OneHot>(), &rank_one_capped_domain);
    comparison_policy.setup_field_budget = None;
    let rank_one_capped = find_schedule(
        onehot_group(36, 1),
        &comparison_policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &rank_one_capped_domain,
        OneHot::ring_challenge_config,
    )
    .expect("rank-one-capped nv36 planner");
    let selected_root = &selected.schedule.root.params;
    assert_eq!(
        selected_root.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(
        selected.schedule.recursive_folds[0].params.role_dims(),
        selected_root.role_dims(),
        "exact packed grinding cost keeps the D256 A-role through the first packing fold"
    );
    let opening_methods = std::iter::once(selected_root.opening_method()).chain(
        selected
            .schedule
            .recursive_folds
            .iter()
            .map(|fold| fold.params.opening_method()),
    );
    for (level, opening_method) in opening_methods.enumerate() {
        if level <= 1 {
            assert!(matches!(
                opening_method,
                akita_types::OpeningMethod::SubringCoefficientPacking { .. }
            ));
        } else {
            assert_eq!(opening_method, akita_types::OpeningMethod::EvaluationTrace);
        }
    }
    let selected_score = (
        estimated_first_direct_setup_capacity(&selected),
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        selected.estimate.estimated_num_setup_field_elements,
    );
    let rank_one_capped_score = (
        estimated_first_direct_setup_capacity(&rank_one_capped),
        rank_one_capped
            .estimate
            .estimated_proof_payload_bytes()
            .unwrap(),
        rank_one_capped.estimate.estimated_num_setup_field_elements,
    );
    assert!(
        selected_score <= rank_one_capped_score,
        "the expanded domain must not lose on the adaptive direct objective"
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_requires_a_monotonic_d64_suffix_domain() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let missing_d64 = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(128),
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 128,
        },
    ])
    .unwrap();
    let missing_policy = policy_for_domain(base_policy, &missing_d64);
    let error = find_schedule(
        onehot_group(16, 1),
        &missing_policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
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
        onehot_group(16, 1),
        &below_policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
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
        onehot_group(16, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(!schedule.schedule.recursive_folds.is_empty());
    assert_eq!(schedule.schedule.root.params.witness_chunk.num_chunks, 8);
    assert_eq!(
        schedule.schedule.recursive_folds[0]
            .params
            .witness_chunk
            .num_chunks,
        8
    );
    assert!(schedule
        .schedule
        .recursive_folds
        .iter()
        .skip(1)
        .all(|fold| fold.params.witness_chunk.num_chunks == 1));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_validates_key_and_policy_at_entry() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let base_policy = policy_of::<OneHot>();
    let domain = RingDimensionSearchDomain::new([
        CommitmentRingDims::uniform(64),
        CommitmentRingDims::uniform(256),
    ])
    .unwrap();
    let policy = policy_for_domain(base_policy, &domain);

    let error = find_schedule(
        onehot_group(16, 0),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
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
        onehot_group(16, 1),
        &invalid_policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
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
    use akita_config::{policy_of, proof_optimized::fp128::OneHot};

    let ceiling = CommitmentRingDims {
        inner: 256,
        outer: 64,
        opening: 64,
    };
    let base_policy = policy_of::<OneHot>();
    let candidates = dimension_candidates(&base_policy, 0, ceiling)
        .expect("D256 A search must not be capped by uniform D64");
    assert!(candidates.contains(&ceiling));
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
        onehot_group(16, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    let exact_fields =
        akita_types::setup_matrix_field_elements_for_schedule(&selected.schedule).unwrap();
    policy.setup_field_budget = Some(exact_fields);

    let budgeted = find_schedule(
        onehot_group(16, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("the exact setup budget should retain the setup-minimal schedule");
    let budgeted_fields =
        akita_types::setup_matrix_field_elements_for_schedule(&budgeted.schedule).unwrap();
    assert_eq!(budgeted_fields, exact_fields);

    let smaller_budget = exact_fields - 1;
    policy.setup_field_budget = Some(smaller_budget);
    let tighter = find_schedule(
        onehot_group(16, 1),
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("a tighter feasible budget should select an admitted alternative");
    let tighter_fields =
        akita_types::setup_matrix_field_elements_for_schedule(&tighter.schedule).unwrap();
    assert!(tighter_fields <= smaller_budget);
}
