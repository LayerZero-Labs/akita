use super::*;
use akita_config::CommitmentConfig;

#[test]
fn candidates_enumerate_exact_role_cartesian_product() {
    use akita_config::{policy_of, proof_optimized::fp128::OneHot};

    let policy = policy_of::<OneHot>();
    let candidates = dimension_candidates(
        &policy,
        0,
        CommitmentRingDims {
            inner: 128,
            outer: 128,
            opening: 128,
        },
    )
    .expect("adaptive dimension candidates");
    assert_eq!(
        candidates,
        vec![
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
            CommitmentRingDims {
                inner: 128,
                outer: 128,
                opening: 64,
            },
            CommitmentRingDims::uniform(128),
        ]
    );
}

#[test]
fn fp32_suffix_candidates_are_uniform_and_monotone() {
    use akita_config::{policy_of, proof_optimized::fp32::OneHot};

    let policy = policy_of::<OneHot>();
    let candidates = dimension_candidates(
        &policy,
        akita_schedules::ADAPTIVE_SEARCH_LEVELS,
        CommitmentRingDims::uniform(256),
    )
    .expect("fp32 suffix candidates");
    assert_eq!(
        candidates,
        vec![
            CommitmentRingDims::uniform(64),
            CommitmentRingDims::uniform(128)
        ]
    );
    let after_drop = dimension_candidates(
        &policy,
        akita_schedules::ADAPTIVE_SEARCH_LEVELS + 1,
        CommitmentRingDims::uniform(64),
    )
    .expect("fp32 suffix candidates after D64 transition");
    assert_eq!(after_drop, vec![CommitmentRingDims::uniform(64)]);
}

#[test]
fn fp32_dense_nv20_uses_corrected_physical_a_bound() {
    use akita_config::proof_optimized::fp32;
    use akita_types::InnerCommitSecurityRoute;

    let policy = akita_config::policy_of::<fp32::Dense>();
    let planned = crate::planner::find_schedule(
        &akita_types::AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(20)),
        fp32::Dense::root_honest_fold_policy(),
        &[],
        &policy,
        fp32::Dense::ring_challenge_config,
    )
    .expect("physical A sizing no longer double-counts the subfield embedding norm");
    let root = &planned.schedule.root.params.final_group.commitment;
    assert_eq!(
        root.inner_commit_matrix.coeff_linf_bound(),
        Some(33_554_431)
    );
    assert!(matches!(
        root.inner_commit_matrix.security_route(),
        InnerCommitSecurityRoute::Linf(_)
    ));
    assert!(planned.schedule.recursive_folds.iter().any(|step| matches!(
        step.params.witness.inner_commit_matrix.security_route(),
        InnerCommitSecurityRoute::L2 { .. }
    )));
}
