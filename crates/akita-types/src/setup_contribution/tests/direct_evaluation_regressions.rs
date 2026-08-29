use super::*;

#[test]
fn multi_group_packed_direct_matches_row_fallback_with_mismatched_t_cols() {
    let plan = finalize_test_plan(
        2,
        5,
        vec![
            test_group_plan(
                2..4,
                4,
                3,
                2,
                2,
                vec![test_scalar(2), test_scalar(3)],
                vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                vec![test_scalar(29), test_scalar(31)],
                vec![test_scalar(37), test_scalar(41)],
            ),
            test_group_plan(
                0..2,
                6,
                3,
                2,
                2,
                vec![test_scalar(53), test_scalar(59)],
                vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                    test_scalar(79),
                    test_scalar(83),
                ],
                vec![test_scalar(89), test_scalar(97), test_scalar(101)],
                vec![test_scalar(103), test_scalar(107)],
                vec![test_scalar(109), test_scalar(113)],
            ),
        ],
        CommitmentRingDims::uniform(TEST_D),
    );
    let setup_len = 12;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_len * TEST_D,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_len * TEST_D)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
        ),
    );
    let alpha_pows = scalar_powers(test_scalar(3), TEST_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);
}
