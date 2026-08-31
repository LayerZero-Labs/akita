use super::*;

#[test]
fn prepared_relation_address_clones_share_the_equality_window() {
    let point = (0..12)
        .map(|index| test_scalar(17 + index as u128))
        .collect::<Vec<_>>();
    let prepared = PreparedRelationAddress::new(&point).unwrap();
    let shared = prepared.clone();
    assert!(std::sync::Arc::ptr_eq(
        &prepared.equality_window,
        &shared.equality_window,
    ));
    assert!(std::sync::Arc::ptr_eq(&prepared.point, &shared.point));
}

#[test]
fn dense_z_eq_slice_uses_relative_high_carry() {
    let num_positions_per_block = 16;
    let depth_commit = 3;
    let depth_fold = 2;
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let inputs = test_inputs(
        1,
        1,
        1,
        1,
        4,
        num_positions_per_block,
        16,
        depth_commit,
        depth_fold,
        4,
        (0..8).map(|index| test_scalar(11 + index)).collect(),
    );
    let joint_geometry = crate::RelationWitnessGeometry::for_evaluation_trace_execution(
        &inputs.level_params,
        &inputs.opening_batch,
    )
    .unwrap();
    let layout = WitnessLayout::new(
        &inputs.level_params,
        &inputs.opening_batch,
        &joint_geometry,
        1,
        crate::RelationQuotientPlan::quotient_lift(inputs.depth_fold().unwrap()).unwrap(),
    )
    .unwrap();
    let relation_geometry = inputs
        .level_params
        .relation_address_geometry(&inputs.opening_batch, 1, TEST_D, layout.live_coeff_len())
        .unwrap();
    let full_vec_randomness = (0..relation_geometry.relation_lane_variable_count())
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let plan =
        prepare_single_group_plan(&inputs, &full_vec_randomness, &fold_gadget, &layout).unwrap();
    let expected = expected_z_setup_weights(
        &layout,
        layout.live_coeff_len(),
        0,
        num_positions_per_block,
        depth_commit,
        &fold_gadget,
        &full_vec_randomness,
    );
    assert_eq!(plan.group_column_eq_slices(0).unwrap().2, expected);
}

#[test]
fn prepare_accepts_exact_non_pow2_fold_count() {
    let mut lp = CommittedGroupParams::params_only(
        crate::SisModulusProfileId::Q128OffsetA7F7,
        64,
        3,
        1,
        1,
        1,
        akita_challenges::SparseChallengeConfig::production_for_ring_dim(64)
            .expect("supported test ring dimension"),
    )
    .with_decomp(8, 24, 2, 3, 3)
    .expect("valid test level params");
    lp.own_group_mut().profile.inner.matrix = crate::InnerCommitMatrixParams::new_unchecked(
        crate::sis::DEFAULT_SIS_SECURITY_POLICY,
        crate::sis::SisTableDigest::CURRENT,
        crate::sis::SisModulusProfileId::Q128OffsetA7F7,
        1,
        16,
        1,
        64,
    );
    lp.own_group_mut().profile.outer.matrix = crate::OuterCommitMatrixParams::new_unchecked(
        crate::sis::DEFAULT_SIS_SECURITY_POLICY,
        crate::sis::SisTableDigest::CURRENT,
        crate::sis::SisModulusProfileId::Q128OffsetA7F7,
        1,
        18,
        1,
        64,
    );
    lp.own_group_mut().opening.num_digits_fold = 2;
    let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
    let depth_fold = lp.num_digits_fold();
    let rows = lp
        .relation_matrix_row_count(opening_batch.num_groups())
        .unwrap();
    let group = SetupContributionGroupInputs {
        group_id: 0,
        num_claims: 2,
        depth_fold,
        a_row_start: 1,
        b_row_start: 2,
    };
    let joint_geometry =
        crate::RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &opening_batch)
            .unwrap();
    let witness_layout = WitnessLayout::new(
        &lp,
        &opening_batch,
        &joint_geometry,
        1,
        crate::RelationQuotientPlan::quotient_lift(2).unwrap(),
    )
    .unwrap();
    let opening_source_len = witness_layout.live_coeff_len();
    let eq_tau1 = (0..rows.next_power_of_two())
        .map(|idx| test_scalar(11 + idx as u128))
        .collect::<Vec<_>>()
        .into();
    let relation_address_geometry = crate::RelationAddressGeometry::new(
        CommitmentRingDims::uniform(TEST_D),
        TEST_D,
        opening_source_len,
    )
    .unwrap();
    let full_vec_randomness =
        vec![F::one(); relation_address_geometry.relation_lane_variable_count()];
    let prepared = SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        1,
        eq_tau1,
        &witness_layout,
        &[group],
        PreparedRelationAddress::new(&full_vec_randomness).unwrap(),
        None,
        relation_address_geometry,
    );
    assert!(prepared.is_ok(), "{:#?}", prepared.err());
}

#[test]
fn deferred_structured_setup_supports_empty_chunk_slots() {
    let num_live_blocks = 3;
    let num_chunks = 8;
    let num_claims = 3;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let num_positions_per_block = 4;
    let n_a = 2;
    let n_b = 2;
    let n_d = 1;
    let log_basis = 4;
    let rows = 1 + n_a + n_b + n_d;
    let layout = test_witness_layout(
        num_claims,
        num_live_blocks,
        num_positions_per_block,
        depth_open,
        depth_commit,
        depth_fold,
        n_a,
        num_chunks,
        n_d,
        depth_fold,
    );
    assert_eq!(
        layout
            .units()
            .iter()
            .map(WitnessUnitLayout::global_block_range)
            .collect::<Vec<_>>(),
        vec![0..0, 0..0, 0..1, 1..1, 1..1, 1..2, 2..2, 2..3]
    );
    let opening_source_len = layout.live_coeff_len();
    let groups = vec![SetupContributionGroupInputs {
        group_id: 0,
        num_claims,
        depth_fold,
        a_row_start: 1,
        b_row_start: 1 + n_a,
    }];
    let inputs = test_inputs(
        n_a,
        n_b,
        n_d,
        num_claims,
        num_live_blocks,
        num_positions_per_block,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        (0..rows.next_power_of_two())
            .map(|index| test_scalar(11 + index as u128))
            .collect(),
    );
    let role_dims = CommitmentRingDims::uniform(TEST_D);
    let address_bits = crate::RelationAddressGeometry::new(role_dims, TEST_D, opening_source_len)
        .unwrap()
        .relation_lane_variable_count();
    let full_vec_randomness = (0..address_bits)
        .map(|index| test_scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);

    let direct = prepare_test_plan(
        &inputs,
        &layout,
        opening_source_len,
        &groups,
        &full_vec_randomness,
        Some(&fold_gadget),
        role_dims,
    )
    .unwrap();
    let deferred = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        &layout,
        &groups,
        PreparedRelationAddress::new(&full_vec_randomness).unwrap(),
        Some(&fold_gadget),
        crate::RelationAddressGeometry::new(role_dims, TEST_D, opening_source_len).unwrap(),
    )
    .unwrap();

    let deferred_group = &deferred.groups[0];
    assert_eq!(deferred_group.active_unit_ranges.len(), num_live_blocks);
    assert_eq!(deferred_group.num_physical_units, num_chunks);
    assert_eq!(deferred_group.d_tensors.len(), num_claims * num_live_blocks);
    assert_eq!(deferred_group.a_tensors.len(), num_chunks);

    let block_challenges = (0..num_claims * num_live_blocks)
        .map(|index| test_scalar(1501 + index as u128))
        .collect::<Vec<_>>();
    let opening_a_evals = (0..num_positions_per_block)
        .map(|index| test_scalar(1601 + index as u128))
        .collect::<Vec<_>>();
    let alpha = test_scalar(3);
    let expected = span_evaluators::structured_slice_reference(
        &direct.groups[0],
        direct.direct_scan_state.weights(0).unwrap(),
        &block_challenges,
        &opening_a_evals,
        alpha,
    );
    assert_eq!(
        deferred
            .evaluate_structured_group::<F>(0, &block_challenges, &opening_a_evals, alpha)
            .unwrap(),
        expected
    );
}
