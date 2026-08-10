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
fn fp32_dense_rejects_the_insecure_nv14_shape() {
    use akita_config::proof_optimized::fp32;

    let policy = akita_config::policy_of::<fp32::Dense>();
    let error = crate::planner::find_schedule(
        &akita_types::AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(14)),
        fp32::Dense::root_honest_fold_policy(),
        &[],
        &policy,
        fp32::Dense::ring_challenge_config,
    )
    .expect_err("fp32 dense nv14 must remain outside the securable catalog");
    assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
}
