use super::*;

#[test]
fn tensor_low_length_is_selected_independently() {
    assert_eq!(
        optimize_fold_challenge_shape(TensorChallengeShape::Tensor { fold_low_len: 1 }, 13)
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

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_policy_rejects_role_dimensions_below_suffix() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot};

    static INVALID_B_DIMENSIONS: &[usize] = &[32, 64, 128];
    let mut policy = policy_of::<MixedDimFp128OneHot>();
    let crate::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels,
        uniform_suffix_dimension,
        potential_a_dimensions,
        potential_d_dimensions,
        ..
    } = policy.ring_dimension_schedule_mode
    else {
        panic!("test preset must be adaptive");
    };
    policy.ring_dimension_schedule_mode = crate::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels,
        uniform_suffix_dimension,
        potential_a_dimensions,
        potential_b_dimensions: INVALID_B_DIMENSIONS,
        potential_d_dimensions,
    };
    assert!(validate_policy(&policy).is_err());
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_schedule_obeys_search_window_and_uniform_suffix() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let policy = policy_of::<MixedDimFp128OneHot>();
    let planned = find_schedule(
        PolynomialGroupLayout::singleton(18),
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let crate::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels,
        uniform_suffix_dimension,
        potential_a_dimensions,
        potential_b_dimensions,
        potential_d_dimensions,
    } = policy.ring_dimension_schedule_mode
    else {
        panic!("test preset must be adaptive");
    };

    let mut levels = Vec::with_capacity(planned.schedule.recursive_folds.len() + 1);
    levels.push(
        planned
            .schedule
            .root
            .params
            .final_group
            .commitment
            .role_dims(),
    );
    levels.extend(
        planned
            .schedule
            .recursive_folds
            .iter()
            .map(|fold| fold.params.witness.role_dims()),
    );
    for (level, dimensions) in levels.iter().copied().enumerate() {
        if level < num_search_levels {
            assert!(potential_a_dimensions.contains(&dimensions.d_a()));
            assert!(potential_b_dimensions.contains(&dimensions.d_b()));
            assert!(potential_d_dimensions.contains(&dimensions.d_d()));
        } else {
            assert_eq!(
                dimensions,
                CommitmentRingDims::uniform(uniform_suffix_dimension)
            );
        }
        if let Some(previous) = level.checked_sub(1).map(|index| levels[index]) {
            assert!(dimensions.d_a() <= previous.d_a());
            assert!(dimensions.d_b() <= previous.d_b());
            assert!(dimensions.d_d() <= previous.d_d());
        }
    }
    assert_eq!(
        planned.schedule.terminal.params.witness.d_a(),
        uniform_suffix_dimension
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_role_selection_minimizes_rank_then_dimension() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let policy = policy_of::<MixedDimFp128OneHot>();
    let plan = |num_vars| {
        find_schedule(
            PolynomialGroupLayout::singleton(num_vars),
            &policy,
            MixedDimFp128OneHot::root_honest_fold_policy(),
            MixedDimFp128OneHot::ring_challenge_config,
            MixedDimFp128OneHot::fold_challenge_shape_at_level,
        )
        .unwrap()
    };
    let nv18_plan = plan(18);
    let nv18 = &nv18_plan.schedule.root.params.final_group.commitment;
    assert_eq!(
        nv18.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(nv18.outer_commit_matrix.output_rank(), 1);
    assert_eq!(nv18.open_commit_matrix.output_rank(), 1);

    let nv36_plan = plan(36);
    let nv36 = &nv36_plan.schedule.root.params.final_group.commitment;
    assert_eq!(
        nv36.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 128,
            opening: 128,
        }
    );
    assert_eq!(nv36.outer_commit_matrix.output_rank(), 1);
    assert_eq!(nv36.open_commit_matrix.output_rank(), 1);
    let nv36_l1 = &nv36_plan.schedule.recursive_folds[0].params.witness;
    assert_eq!(
        nv36_l1.role_dims(),
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(nv36_l1.inner_commit_matrix.output_rank(), 2);
    assert_eq!(nv36_l1.outer_commit_matrix.output_rank(), 1);
    assert_eq!(nv36_l1.open_commit_matrix.output_rank(), 1);

    let rank_at = |commitment: &CommittedGroupParams,
                   role: akita_types::SisMatrixRole,
                   candidate_dimension: usize| {
        let selected_dimension = match role {
            akita_types::SisMatrixRole::Outer => commitment.role_dims().d_b(),
            akita_types::SisMatrixRole::Open => commitment.role_dims().d_d(),
            akita_types::SisMatrixRole::Inner => unreachable!("test prices collision roles"),
        };
        let selected_width = match role {
            akita_types::SisMatrixRole::Outer => commitment.outer_commit_matrix.input_width(),
            akita_types::SisMatrixRole::Open => commitment.open_commit_matrix.input_width(),
            akita_types::SisMatrixRole::Inner => unreachable!("test prices collision roles"),
        };
        let native_width = selected_width / (commitment.role_dims().d_a() / selected_dimension);
        let (key, width) = akita_schedules::planner_support::projected_collision_role_price(
            &policy,
            role,
            commitment.role_dims().d_a(),
            candidate_dimension,
            native_width,
            commitment.log_basis_outer,
        )
        .unwrap();
        akita_types::sis::min_secure_rank(key, u64::try_from(width).unwrap()).unwrap()
    };
    assert_eq!(rank_at(nv18, akita_types::SisMatrixRole::Outer, 64), 1);
    assert_eq!(rank_at(nv18, akita_types::SisMatrixRole::Open, 64), 1);
    assert_eq!(rank_at(nv36, akita_types::SisMatrixRole::Outer, 64), 2);
    assert_eq!(rank_at(nv36, akita_types::SisMatrixRole::Open, 64), 2);

    assert_eq!(
        nv36_plan.estimate.estimated_setup_envelope_ring_elements,
        176_128
    );
    assert_eq!(
        nv36_plan.estimate.estimated_proof_payload_bytes().unwrap(),
        99_512
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_a_rank_fallback_accepts_rank_above_one() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let mut policy = policy_of::<MixedDimFp128OneHot>();
    policy.max_setup_envelope_field_elements = usize::MAX;
    let planned = find_schedule(
        PolynomialGroupLayout::singleton(40),
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let root = &planned.schedule.root.params.final_group.commitment;
    let l1 = &planned.schedule.recursive_folds[0].params.witness;

    assert_eq!(root.role_dims().d_a(), 256);
    assert_eq!(root.inner_commit_matrix.output_rank(), 2);
    assert_eq!(l1.role_dims().d_a(), 256);
    assert_eq!(l1.inner_commit_matrix.output_rank(), 2);
    assert_eq!(
        akita_types::setup_matrix_field_elements_for_schedule(&planned.schedule).unwrap(),
        268_435_456
    );
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_rejects_multi_chunk_policy() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let mut policy = policy_of::<MixedDimFp128OneHot>();
    policy.witness_chunk = akita_types::ChunkedWitnessCfg::d64_production();
    let error = find_schedule(
        PolynomialGroupLayout::singleton(18),
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("does not support direct multi-chunk planning"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_search_validates_key_policy_and_physical_setup_budget() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let mut policy = policy_of::<MixedDimFp128OneHot>();
    let error = find_schedule(
        PolynomialGroupLayout::new(18, 0),
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("opening group layouts must be nonempty"));

    policy.max_setup_envelope_field_elements = 1;
    assert!(find_schedule(
        PolynomialGroupLayout::singleton(18),
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .is_err());

    policy.max_setup_envelope_field_elements = 0;
    let error = find_schedule(
        PolynomialGroupLayout::singleton(18),
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("maximum setup envelope must be positive"));
}

#[cfg(feature = "catalog-gen")]
#[test]
fn preserved_recursive_proof_size_is_unchanged() {
    use akita_config::{
        policy_of, proof_optimized::fp128::D64OneHot, CommitmentConfig, RecursiveCommitmentConfig,
    };
    use akita_types::{AkitaScheduleLookupKey, CommittedGroupProfile};

    type Recursive = RecursiveCommitmentConfig<D64OneHot>;
    let precommit_layout = PolynomialGroupLayout::singleton(16);
    let precommit = find_schedule(
        precommit_layout,
        &policy_of::<D64OneHot>(),
        D64OneHot::root_honest_fold_policy(),
        D64OneHot::ring_challenge_config,
        D64OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let descriptor = CommittedGroupProfile::from_params(
        precommit_layout,
        &precommit.schedule.root.params.final_group.commitment,
    );
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    let planned = crate::find_group_batch_schedule(
        &key,
        Recursive::root_honest_fold_policy(),
        &[
            D64OneHot::root_honest_fold_policy(),
            D64OneHot::root_honest_fold_policy(),
        ],
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

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_schedule_is_descriptor_deterministic() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let policy = policy_of::<MixedDimFp128OneHot>();
    let plan = || {
        find_schedule(
            PolynomialGroupLayout::singleton(18),
            &policy,
            MixedDimFp128OneHot::root_honest_fold_policy(),
            MixedDimFp128OneHot::ring_challenge_config,
            MixedDimFp128OneHot::fold_challenge_shape_at_level,
        )
        .unwrap()
        .schedule
        .canonical_descriptor_bytes()
    };
    assert_eq!(plan(), plan());
}

#[cfg(feature = "catalog-gen")]
#[test]
fn adaptive_frontier_matches_unpruned_traversal() {
    use akita_config::{policy_of, proof_optimized::fp128::MixedDimFp128OneHot, CommitmentConfig};

    let policy = policy_of::<MixedDimFp128OneHot>();
    let key = PolynomialGroupLayout::singleton(18);
    let pruned = find_schedule(
        key,
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    let unpruned = unpruned_search::find_schedule(
        key,
        &policy,
        MixedDimFp128OneHot::root_honest_fold_policy(),
        MixedDimFp128OneHot::ring_challenge_config,
        MixedDimFp128OneHot::fold_challenge_shape_at_level,
    )
    .unwrap();
    assert_eq!(
        pruned.schedule.canonical_descriptor_bytes(),
        unpruned.schedule.canonical_descriptor_bytes()
    );
    assert_eq!(pruned.estimate, unpruned.estimate);
}
