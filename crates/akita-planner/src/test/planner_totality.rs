use super::*;

use akita_config::{
    honest_fold_policy_of, policy_of, proof_optimized::fp64::Dense, CommitmentConfig,
};

#[test]
fn valid_small_scalar_root_has_a_schedule() {
    let policy = policy_of::<Dense>();
    for num_vars in 8..=9 {
        let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
        let schedule = find_schedule(
            &key,
            honest_fold_policy_of::<Dense>(),
            &[],
            &policy,
            Dense::ring_challenge_config,
        )
        .unwrap_or_else(|error| panic!("valid nv={num_vars} D64-root request: {error}"));

        if num_vars == 8 {
            let root = &schedule.schedule.root;
            assert!(
                root.output_witness_len
                    * root.params.final_group.commitment.log_basis_open as usize
                    >= root.input_witness_len * policy.decomposition.field_bits() as usize,
                "the regression must exercise the previously rejected noncontractive root"
            );
            let cleartext_source_bytes =
                (1usize << num_vars) * (policy.decomposition.field_bits() as usize).div_ceil(8);
            assert!(
                schedule
                    .estimate
                    .estimated_proof_payload_bytes()
                    .expect("fallback proof size")
                    > cleartext_source_bytes,
                "planner totality must not depend on beating cleartext transmission"
            );
        }
        schedule
            .schedule
            .validate_structure()
            .expect("the fallback schedule must pass structural validation");
    }
}

#[test]
fn valid_small_grouped_root_has_a_schedule() {
    let precommitted_group = PolynomialGroupLayout::singleton(16);
    let policy = policy_of::<Dense>();
    let producer_key = AkitaScheduleLookupKey::single(precommitted_group);
    let producer_dimensions = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let producer_opening = PlannerOpeningCandidate::coefficient_packing(
        0,
        policy.claim_ext_degree,
        producer_dimensions,
        64,
    )
    .expect("D64 producer opening");
    let producer = root_level_candidates_for_basis(
        &producer_key,
        honest_fold_policy_of::<Dense>(),
        &[],
        &policy,
        producer_dimensions,
        producer_opening,
        &[],
        Dense::inner_basis_range().0,
        Dense::opening_basis_range().0,
        false,
    )
    .expect("scalar producer candidates")
    .into_iter()
    .next()
    .expect("scalar producer candidate")
    .0;
    let precommitted_profile =
        CommittedGroupProfile::try_from_params(precommitted_group, &producer)
            .expect("scalar producer profile");
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(8),
        precommitteds: vec![precommitted_profile],
    };
    let schedule = find_schedule(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("a valid grouped D64-root request must have a schedule");

    assert_eq!(schedule.schedule.root.params.precommitted_groups.len(), 1);
    assert!(
        !schedule.schedule.recursive_folds.is_empty(),
        "a grouped root must retain its required child fold"
    );
    schedule
        .schedule
        .validate_structure()
        .expect("the grouped fallback schedule must pass structural validation");
}

#[test]
fn exact_partial_ring_precommit_has_a_grouped_schedule() {
    let policy = policy_of::<Dense>();
    let precommitted_group = PolynomialGroupLayout::singleton(4);
    let profile = plan_precommit_profile(
        precommitted_group,
        honest_fold_policy_of::<Dense>(),
        &policy,
    )
    .expect("plan an exact sixteen-cell commitment profile");

    assert_eq!(profile.group, precommitted_group);
    assert_eq!(profile.num_live_ring_elements_per_claim, 1);
    assert!(profile.inner_commit_matrix.ring_dimension() > 16);

    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(16),
        precommitteds: vec![profile],
    };
    let schedule = find_schedule(
        &key,
        honest_fold_policy_of::<Dense>(),
        &[honest_fold_policy_of::<Dense>()],
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("open the exact partial-ring precommit under a grouped schedule");

    assert_eq!(schedule.schedule.root.params.precommitted_groups.len(), 1);
    schedule
        .schedule
        .validate_structure()
        .expect("partial-ring grouped schedule must validate");
}
