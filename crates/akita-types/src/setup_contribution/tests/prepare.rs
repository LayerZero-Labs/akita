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
