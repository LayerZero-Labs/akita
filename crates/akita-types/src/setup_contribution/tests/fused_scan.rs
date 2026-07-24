use super::*;

#[test]
fn multi_group_packed_direct_matches_row_fallback_with_nested_role_dims() {
    const D_A: usize = 64;
    const D_B: usize = 32;
    const D_D: usize = 32;
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
                4,
                3,
                2,
                2,
                vec![test_scalar(43), test_scalar(47)],
                vec![
                    test_scalar(53),
                    test_scalar(59),
                    test_scalar(61),
                    test_scalar(67),
                ],
                vec![test_scalar(71), test_scalar(73), test_scalar(79)],
                vec![test_scalar(83), test_scalar(89)],
                vec![test_scalar(97), test_scalar(101)],
            ),
        ],
        CommitmentRingDims {
            inner: D_A,
            outer: D_B,
            opening: D_D,
        },
    );
    let setup_ring_elements = plan.required().div_ceil(D_A / D_D);
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
                .map(|idx| test_scalar(211 + idx as u128))
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
