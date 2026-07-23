use super::*;

#[allow(clippy::too_many_arguments)]
fn weighted_test_group_plan(
    d_col_range: std::ops::Range<usize>,
    t_cols: usize,
    z_cols: usize,
    n_a: usize,
    n_b: usize,
    e_eq_slice: Vec<F>,
    t_eq_slice: Vec<F>,
    z_eq_slice: Vec<F>,
    a_row_weights: Vec<F>,
    b_weights: Vec<F>,
) -> SetupContributionGroupPlan<F> {
    let mut group = test_group_plan(d_col_range, t_cols, z_cols, n_a, n_b);
    group.e_eq_slice = e_eq_slice;
    group.t_eq_slice = t_eq_slice;
    group.z_eq_slice = z_eq_slice;
    group.a_row_weights = a_row_weights.into();
    group.b_weights = b_weights.into();
    group
}

#[test]
fn fused_multi_group_scan_matches_separate_scans_with_nested_role_dims() {
    const D_A: usize = 64;
    const D_B: usize = 32;
    const D_D: usize = 32;
    let role_dims = CommitmentRingDims {
        inner: D_A,
        outer: D_B,
        opening: D_D,
    };
    let first_group = weighted_test_group_plan(
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
    );
    let second_group = weighted_test_group_plan(
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
    );
    let plan = finalize_test_plan(
        2,
        5,
        vec![first_group.clone(), second_group.clone()],
        role_dims,
    );
    let separate_plans = [
        finalize_test_plan(2, 5, vec![first_group], role_dims),
        finalize_test_plan(2, 5, vec![second_group], role_dims),
    ];
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
    let expected = separate_plans
        .iter()
        .map(|group_plan| {
            group_plan
                .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
                .unwrap()
        })
        .sum();
    let got = plan
        .evaluate_direct::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d)
        .unwrap();
    assert_eq!(got, expected);
}
