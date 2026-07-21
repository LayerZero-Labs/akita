use super::*;

#[test]
fn prepare_accepts_exact_non_pow2_fold_count() {
    let mut lp = LevelParams::params_only(
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
    lp.cached_num_digits_block_claims = 2;
    lp.cached_num_digits_fold_value = 2;
    let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
    let relation_matrix_row_layout = RelationMatrixRowLayout::WithDBlock;
    let rows = lp
        .relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)
        .unwrap();
    let group = SetupContributionGroupInputs {
        group_id: 0,
        num_claims: 2,
        depth_fold: 2,
        a_row_start: 1,
        b_row_start: 2,
    };
    let witness_layout = test_witness_layout(2, 3, 8, 3, 2, 2, 1, 1, rows, 2);
    let opening_source_len = witness_layout.total_len();
    let eq_tau1 = (0..rows.next_power_of_two())
        .map(|idx| test_scalar(11 + idx as u128))
        .collect::<Vec<_>>()
        .into();
    assert!(SetupContributionPlan::prepare::<F>(
        &lp,
        &opening_batch,
        relation_matrix_row_layout,
        eq_tau1,
        &witness_layout,
        opening_source_len,
        &[group],
        &[],
        None,
        CommitmentRingDims::uniform(TEST_D),
    )
    .is_ok());
}
