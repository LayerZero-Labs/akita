use super::*;

#[test]
fn recursive_exact_cutover_proof_size_is_documented() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let descriptor = plan_standalone_precommit(
        precommit_layout,
        &policy_of::<OneHot>(),
        OneHot::root_honest_fold_policy(),
        OneHot::ring_challenge_config,
    )
    .unwrap()
    .selected
    .profile;
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
        86_275
    );
}

#[test]
fn scalar_recursive_nv36_selects_offloaded_schedule() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    type Recursive = RecursiveCommitmentConfig<OneHot>;
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(36));
    let policy = policy_of::<Recursive>();
    assert_eq!(
        policy.selection_policy,
        crate::SelectionPolicyId::MinFirstDirectSetupThenPayload
    );

    let planned = crate::find_schedule(
        &key,
        Recursive::root_honest_fold_policy(),
        &[],
        &policy,
        Recursive::ring_challenge_config,
    )
    .expect("scalar recursive schedule");

    assert!(planned.schedule.root.params.precommitted_groups.is_empty());
    assert!(planned.estimate.selected_offload_edges > 0);
    assert!(planned
        .schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.incoming_setup_prefix.is_some()));
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
            outer: 128,
            opening: 128,
        }
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
        planned.estimate.estimated_proof_payload_bytes().unwrap(),
        88_730
    );
}

#[test]
fn recursive_adaptive_search_selects_schedule_dimensions_and_setup_prefixes() {
    use akita_config::{
        policy_of, proof_optimized::fp128::OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::AkitaScheduleLookupKey;

    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let descriptor = plan_standalone_precommit(
        precommit_layout,
        &policy_of::<OneHot>(),
        OneHot::root_honest_fold_policy(),
        OneHot::ring_challenge_config,
    )
    .unwrap()
    .selected
    .profile;
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
