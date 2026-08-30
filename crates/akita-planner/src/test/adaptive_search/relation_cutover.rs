use super::*;

#[test]
fn bounded_suffix_dp_matches_unpruned_fixed_cutover_search() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domain = RingDimensionSearchDomain::uniform(256).unwrap();
    let base_policy = policy_of::<OneHot>();
    let mut policy = policy_for_domain(base_policy, &domain);
    // The oracle enumerates every complete suffix, so keep this correctness
    // fixture focused on relation cutovers rather than multiplying it by the
    // independent basis-search axis.
    policy.inner_basis_range.1 = policy.inner_basis_range.0;
    policy.opening_basis_range.1 = policy.opening_basis_range.0;
    let key = onehot_group(20, 1);
    let selected = find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(
        selected.schedule.recursive_folds.len() < unpruned_search::MAX_ORACLE_RECURSION_DEPTH,
        "bounded oracle must cover the production winner's recursion depth",
    );
    let unpruned = unpruned_search::find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        OneHot::ring_challenge_config,
    )
    .unwrap();
    assert!(unpruned.reduced_fold_candidates > 0);
    assert!(unpruned.suffix_states <= unpruned_search::MAX_ORACLE_SUFFIX_STATES);
    assert!(unpruned.complete_schedules <= unpruned_search::MAX_ORACLE_COMPLETE_SCHEDULES);
    let unpruned = &unpruned.planned;

    let relation_modes = std::iter::once(unpruned.schedule.root.params.ring_relation_mode)
        .chain(
            unpruned
                .schedule
                .recursive_folds
                .iter()
                .map(|fold| fold.params.ring_relation_mode),
        )
        .collect::<Vec<_>>();
    let cutover = relation_modes
        .iter()
        .position(|mode| mode.is_reduced_evaluation())
        .expect("the fixture must select a nonempty reduced suffix");
    assert!(
        cutover > 0,
        "the fixture must retain a nonempty quotient prefix"
    );
    assert!(relation_modes[..cutover]
        .iter()
        .all(|mode| *mode == akita_types::RingRelationMode::QuotientLift));
    assert!(relation_modes[cutover..]
        .iter()
        .all(|mode| mode.is_reduced_evaluation()));

    assert_eq!(
        selected.estimate.estimated_proof_payload_bytes().unwrap(),
        unpruned.estimate.estimated_proof_payload_bytes().unwrap()
    );
    assert_eq!(
        selected.estimate.estimated_num_setup_field_elements,
        unpruned.estimate.estimated_num_setup_field_elements,
    );
    assert_eq!(
        selected.schedule.canonical_descriptor_bytes(),
        unpruned.schedule.canonical_descriptor_bytes()
    );
}

#[test]
fn selected_cutover_is_invariant_under_relation_traversal_order() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domain = RingDimensionSearchDomain::uniform(256).unwrap();
    let mut policy = policy_for_domain(policy_of::<OneHot>(), &domain);
    policy.inner_basis_range.1 = policy.inner_basis_range.0;
    policy.opening_basis_range.1 = policy.opening_basis_range.0;
    let key = onehot_group(20, 1);
    let lookup_key = akita_types::AkitaScheduleLookupKey::single(key);
    let canonical = crate::planner::find_schedule_in_relation_order(
        &lookup_key,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &[],
        &policy,
        OneHot::ring_challenge_config,
        RelationTraversalOrder::Canonical,
        RelationModeFilter::All,
    )
    .unwrap();
    let reversed = crate::planner::find_schedule_in_relation_order(
        &lookup_key,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &[],
        &policy,
        OneHot::ring_challenge_config,
        RelationTraversalOrder::Reversed,
        RelationModeFilter::All,
    )
    .unwrap();

    assert!(canonical
        .schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation()));
    assert_eq!(
        canonical.estimate.estimated_proof_payload_bytes().unwrap(),
        reversed.estimate.estimated_proof_payload_bytes().unwrap()
    );
    assert_eq!(
        canonical.schedule.canonical_descriptor_bytes(),
        reversed.schedule.canonical_descriptor_bytes(),
        "relation candidate enumeration order must not affect the exact production DP winner"
    );
}
