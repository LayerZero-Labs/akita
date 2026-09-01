use super::*;

#[test]
fn canonical_shape_follows_successor_padded_relation_domain() {
    let mut schedule = recursive_schedule(64, 128, false);
    schedule.root.output_witness_len = 64;
    schedule.recursive_folds[0].input_witness_len = 64;
    let root_layout = OpeningClaimsLayout::new(6, 1).expect("root opening layout");

    let grinding_plan = GrindingPlan::new(Vec::new(), 1).expect("empty grinding plan");
    let shape = canonical_proof_shape(&schedule, &root_layout, 2, &grinding_plan)
        .expect("successor-aware proof shape");
    assert_eq!(shape.root.stage2_sumcheck_proof.len(), 7);
    assert_ne!(
        shape.root.stage2_sumcheck_proof.len(),
        sumcheck_rounds(schedule.root.params.d_a(), schedule.root.output_witness_len),
        "the fixture must distinguish successor padding from the retired shortcut"
    );
    assert_eq!(
        shape
            .terminal
            .extension_opening_reduction
            .expect("extension terminal reduction")
            .sumcheck
            .len(),
        6,
        "terminal EOR must inherit the seven-variable predecessor relation"
    );
}

#[test]
fn base_field_proof_shape_rejects_mixed_opening_families() {
    let mut schedule = recursive_schedule(64, 64, false);
    let precommitted_group = PolynomialGroupLayout::singleton(8);
    let mut precommitted = preceding_group_params(&schedule.root.params, precommitted_group);
    precommitted.opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    precommitted.opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(64).expect("production challenge family");
    schedule
        .root
        .params
        .insert_precommitted_group(precommitted)
        .expect("mixed-family precommit");
    let layout = OpeningClaimsLayout::from_groups(vec![
        precommitted_group,
        schedule.root.params.final_group().profile.group,
    ])
    .expect("grouped opening layout");

    let grinding_plan = GrindingPlan::new(Vec::new(), 1).expect("empty grinding plan");
    let error = canonical_proof_shape(&schedule, &layout, 1, &grinding_plan)
        .expect_err("base-field proof shape must reject mixed opening families");
    assert!(
        matches!(
            &error,
            AkitaError::InvalidSetup(message)
                if message.contains("cannot mix opening-method families")
        ),
        "unexpected mixed-family error: {error:?}"
    );
}

#[test]
fn proof_shape_accepts_group_local_subring_dimensions_within_packing_family() {
    let mut schedule = recursive_schedule(256, 64, false);
    retarget_outer_dimension(&mut schedule.root.params, 64).expect("root B dimension");
    retarget_open_dimension(&mut schedule.root.params, 64).expect("root D dimension");
    schedule.root.params.own_group_mut().opening.opening_method =
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        };
    schedule
        .root
        .params
        .own_group_mut()
        .opening
        .fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(64).expect("64 challenge family");

    let precommitted_group = PolynomialGroupLayout::singleton(8);
    let mut precommitted = preceding_group_params(&schedule.root.params, precommitted_group);
    precommitted.opening.opening_method = OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 128,
    };
    precommitted.opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(128).expect("128 challenge family");
    schedule
        .root
        .params
        .insert_precommitted_group(precommitted)
        .expect("packing-family precommit");
    let layout = OpeningClaimsLayout::from_groups(vec![
        precommitted_group,
        schedule.root.params.final_group().profile.group,
    ])
    .expect("grouped opening layout");

    let grinding_plan = GrindingPlan::new(Vec::new(), 1).expect("empty grinding plan");
    let shape = canonical_proof_shape(&schedule, &layout, 2, &grinding_plan)
        .expect("group-local packing dimensions share one opening family");
    assert!(shape.root.extension_opening_reduction.is_none());
}
