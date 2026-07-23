use super::*;
use akita_algebra::offset_eq::MAX_COMPACT_STRIDE_TERMS;

fn prepare_budget_fixture(ownership_units: usize) -> Result<SetupContributionPlan<F>, AkitaError> {
    const POSITIONS_PER_BLOCK: usize = 2048;
    const ROLE_DIMS: CommitmentRingDims = CommitmentRingDims {
        inner: 2048,
        outer: 16,
        opening: 16,
    };
    let inputs = test_inputs(
        1,
        0,
        0,
        1,
        ownership_units,
        POSITIONS_PER_BLOCK,
        1,
        1,
        1,
        4,
        vec![test_scalar(11); 4],
    );
    let rows = inputs
        .level_params
        .relation_matrix_row_count(inputs.opening_batch.num_groups())?;
    let witness_layout = test_witness_layout(
        1,
        ownership_units,
        POSITIONS_PER_BLOCK,
        1,
        1,
        1,
        1,
        ownership_units,
        rows,
        1,
    );
    let opening_source_len = witness_layout.total_len();
    let randomness_bits = crate::opening_domain_len(opening_source_len)?.trailing_zeros() as usize;
    let full_vec_randomness = (0..randomness_bits)
        .map(|index| test_scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let group = test_single_group_descriptor(&inputs)?;
    prepare_test_plan(
        &inputs,
        &witness_layout,
        opening_source_len,
        &[group],
        &full_vec_randomness,
        None,
        ROLE_DIMS,
    )
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
    lp.cached_num_digits_block_claims = 2;
    lp.cached_num_digits_fold_value = 2;
    let opening_batch = OpeningClaimsLayout::new(0, 2).expect("opening batch");
    let rows = lp
        .relation_matrix_row_count(opening_batch.num_groups())
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

#[test]
fn prepare_enforces_setup_point_evaluation_budget() {
    const TERMS_PER_UNIT: usize = 2048 * (2048 / 16);
    const UNITS_AT_CAP: usize = MAX_COMPACT_STRIDE_TERMS / TERMS_PER_UNIT;
    assert_eq!(UNITS_AT_CAP * TERMS_PER_UNIT, MAX_COMPACT_STRIDE_TERMS);

    let plan = prepare_budget_fixture(UNITS_AT_CAP).expect("budget cap accepted");
    assert_eq!(
        plan.projection_geometry().evaluation_terms(),
        MAX_COMPACT_STRIDE_TERMS
    );

    assert!(matches!(
        prepare_budget_fixture(UNITS_AT_CAP + 1),
        Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual,
        }) if actual > MAX_COMPACT_STRIDE_TERMS
    ));
}
