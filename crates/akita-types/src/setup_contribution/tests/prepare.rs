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
    let full_vec_randomness = (0..8)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let inputs = test_inputs(
        1,
        0,
        0,
        1,
        4,
        num_positions_per_block,
        16,
        depth_commit,
        depth_fold,
        4,
        vec![test_scalar(11), test_scalar(12)],
    );
    let layout = test_witness_layout(
        inputs.num_claims(),
        inputs.num_live_blocks(),
        inputs.num_positions_per_block(),
        inputs.depth_open(),
        inputs.depth_commit(),
        inputs.depth_fold().unwrap(),
        inputs.n_a(),
        1,
        1,
        inputs.depth_fold().unwrap(),
    );
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
    assert_eq!(plan.groups[0].column_eq_slices().unwrap().2, expected);
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
    lp.inner_commit_matrix = crate::InnerCommitMatrixParams::new_unchecked(
        crate::sis::DEFAULT_SIS_SECURITY_POLICY,
        crate::sis::SisTableDigest::CURRENT,
        crate::sis::SisModulusProfileId::Q128OffsetA7F7,
        1,
        16,
        1,
        64,
    );
    lp.outer_commit_matrix = crate::OuterCommitMatrixParams::new_unchecked(
        crate::sis::DEFAULT_SIS_SECURITY_POLICY,
        crate::sis::SisTableDigest::CURRENT,
        crate::sis::SisModulusProfileId::Q128OffsetA7F7,
        1,
        18,
        1,
        64,
    );
    let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
    let depth_fold = lp
        .num_digits_fold_for_params(&lp, 2, lp.field_bits_for_cache())
        .unwrap();
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
    let witness_layout = WitnessLayout::new(&lp, &opening_batch, 1, 2).unwrap();
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
        eq_tau1,
        &witness_layout,
        &[group],
        PreparedRelationAddress::new(&full_vec_randomness).unwrap(),
        None,
        relation_address_geometry,
    );
    assert!(prepared.is_ok(), "{:#?}", prepared.err());
}
