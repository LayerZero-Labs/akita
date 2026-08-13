use super::*;

/// Plan the profile a group commits with when it has no precommitted groups.
///
/// These tests assert planner output, so they plan the row rather than read a
/// catalog, matching what `generated_families` does at generation time.
fn planned_profile_without_precommitted_groups<Cfg: akita_config::CommitmentConfig>(
    group: PolynomialGroupLayout,
) -> CommittedGroupProfile {
    let planned = crate::find_schedule(
        &AkitaScheduleLookupKey::single(group),
        Cfg::root_honest_fold_policy(),
        &[],
        &akita_config::policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("independent schedule");
    CommittedGroupProfile::from_params(group, &planned.schedule.root.params.final_group.commitment)
}

#[test]
fn recursive_adaptive_search_selects_schedule_dimensions_and_setup_prefixes() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let descriptor = planned_profile_without_precommitted_groups::<OneHot>(precommit_layout);
    assert_eq!(descriptor.inner_commit_matrix.ring_dimension(), 256);
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
        88_752,
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
