use super::*;

#[test]
fn unpruned_search_covers_selective_l2_candidate_domain() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot, CommitmentConfig};

    let domain = RingDimensionSearchDomain::uniform(64).expect("D64 test domain");
    let mut policy = policy_of::<OneHot>();
    policy.inner_basis_range.1 = policy.inner_basis_range.0;
    policy.opening_basis_range.1 = policy.opening_basis_range.0;
    let key = onehot_group(12, 1);
    let selected = find_schedule(
        key,
        &policy,
        akita_config::honest_fold_policy_of::<OneHot>(),
        &domain,
        OneHot::ring_challenge_config,
    )
    .expect("production selective-L2 schedule");
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
    .expect("unpruned selective-L2 schedule");
    assert!(unpruned.linf_candidates > 0, "oracle must enumerate Linf");
    assert!(unpruned.l2_candidates > 0, "oracle must enumerate L2");
    assert_eq!(selected.estimate, unpruned.planned.estimate);
    assert_eq!(
        selected.schedule.canonical_descriptor_bytes(),
        unpruned.planned.schedule.canonical_descriptor_bytes(),
    );
}
