use super::*;

#[test]
fn packed_direct_matches_row_fallback_with_d_offset() {
    let plan = finalize_test_plan(SetupContributionPlan {
        d_rows: 2,
        d_physical_cols: 5,
        groups: vec![SetupContributionGroupPlan {
            e_col_offset: 2,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
    });
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multi_group_packed_direct_matches_row_fallback() {
    let plan = finalize_test_plan(SetupContributionPlan {
        d_rows: 2,
        d_physical_cols: 5,
        groups: vec![
            SetupContributionGroupPlan {
                e_col_offset: 2,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
                b_weights: vec![test_scalar(37), test_scalar(41)].into(),
                d_weights: vec![test_scalar(43), test_scalar(47)].into(),
            },
            SetupContributionGroupPlan {
                e_col_offset: 0,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                t_eq_slice: vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                ],
                z_eq_slice: vec![test_scalar(79), test_scalar(83), test_scalar(89)],
                a_row_weights: vec![test_scalar(97), test_scalar(101)].into(),
                b_weights: vec![test_scalar(103), test_scalar(107)].into(),
                d_weights: vec![test_scalar(109), test_scalar(113)].into(),
            },
        ],
    });
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);

    let setup_index_weight = plan.materialize_setup_index_weights().unwrap();
    let setup_view = setup
        .shared_matrix()
        .ring_view::<TEST_D>(1, setup_index_weight.len())
        .unwrap();
    let tie: F = setup_index_weight
        .iter()
        .zip(setup_view.as_slice())
        .map(|(w, ring)| eval_ring_at_pows(ring, &alpha_pows) * *w)
        .sum();
    assert_eq!(tie, got);
}

#[test]
fn packed_direct_matches_row_fallback_with_nested_role_dims() {
    let plan = finalize_test_plan(SetupContributionPlan {
        d_rows: 2,
        d_physical_cols: 5,
        groups: vec![SetupContributionGroupPlan {
            e_col_offset: 2,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
    });
    const D: usize = 64;
    const D_B: usize = 32;
    const D_D: usize = 32;
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            D,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn packed_direct_rejects_non_decomposable_role_alpha_pows() {
    let plan = finalize_test_plan(SetupContributionPlan {
        d_rows: 2,
        d_physical_cols: 5,
        groups: vec![SetupContributionGroupPlan {
            e_col_offset: 2,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
    });
    const D_A: usize = 64;
    const D_B: usize = 32;
    const D_D: usize = 32;
    let setup_len = 10;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D_A,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * D_A)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            D_A,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let mut alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    alpha_pows_b[1] += test_scalar(1);

    assert!(matches!(
        plan.evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn packed_direct_accepts_d_footprint_at_nested_d_d() {
    // D-role columns are counted at d_d; comparing `required` against
    // total_ring_elements_at_dyn(d_a) falsely rejects valid setups when
    // d_d < d_a and the D footprint dominates.
    let plan = finalize_test_plan(SetupContributionPlan {
        d_rows: 2,
        d_physical_cols: 11,
        groups: vec![SetupContributionGroupPlan {
            e_col_offset: 0,
            t_cols: 4,
            z_cols: 3,
            n_a: 2,
            n_b: 2,
            required: 0,
            segments: Vec::new().into(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
            b_weights: vec![test_scalar(37), test_scalar(41)].into(),
            d_weights: vec![test_scalar(43), test_scalar(47)].into(),
        }],
    });
    const D_A: usize = 64;
    const D_B: usize = 64;
    const D_D: usize = 32;
    let setup_ring_elements = 20usize;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: D_A,
            max_setup_len: setup_ring_elements,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * D_A)
                .map(|idx| test_scalar(311 + idx as u128))
                .collect(),
            D_A,
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D_A)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
        .unwrap();
    assert_eq!(got, expected);
}

#[test]
fn multi_group_packed_direct_matches_row_fallback_with_mismatched_t_cols() {
    let plan = finalize_test_plan(SetupContributionPlan {
        d_rows: 2,
        d_physical_cols: 5,
        groups: vec![
            SetupContributionGroupPlan {
                e_col_offset: 2,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_row_weights: vec![test_scalar(29), test_scalar(31)].into(),
                b_weights: vec![test_scalar(37), test_scalar(41)].into(),
                d_weights: vec![test_scalar(43), test_scalar(47)].into(),
            },
            SetupContributionGroupPlan {
                e_col_offset: 0,
                t_cols: 6,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new().into(),
                e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                t_eq_slice: vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                    test_scalar(79),
                    test_scalar(83),
                ],
                z_eq_slice: vec![test_scalar(89), test_scalar(97), test_scalar(101)],
                a_row_weights: vec![test_scalar(103), test_scalar(107)].into(),
                b_weights: vec![test_scalar(109), test_scalar(113)].into(),
                d_weights: vec![test_scalar(127), test_scalar(131)].into(),
            },
        ],
    });
    let setup_len = 12;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            gen_ring_dim: TEST_D,
            max_setup_len: setup_len,
            public_matrix_seed: [0u8; 32],
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
            TEST_D,
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
        .unwrap();
    assert_eq!(got, expected);
}
