use super::*;

#[test]
fn multi_group_packed_direct_matches_row_fallback_with_nested_role_dims() {
    const D_A: usize = 128;
    const D_B: usize = 64;
    const D_D: usize = 64;
    let mut plan = finalize_test_plan(
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
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_ring_elements * D_A,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * D_A)
                .map(|idx| test_scalar(211 + idx as u128))
                .collect(),
        ),
    );
    let alpha = test_scalar(3);
    let alpha_pows_a = scalar_powers(alpha, D_A);
    let alpha_pows_b = scalar_powers(alpha, D_B);
    let alpha_pows_d = scalar_powers(alpha, D_D);
    let expected = plan
        .evaluate_direct_by_rows::<F>(&setup, &alpha_pows_a, &alpha_pows_b, &alpha_pows_d, D_A)
        .unwrap();
    let got = plan.evaluate_direct::<F>(&setup).unwrap();
    assert_eq!(got, expected);

    let a_functional: std::sync::Arc<[F]> = akita_algebra::ring::terminal_residue_kernel(
        &(0..D_A)
            .map(|index| test_scalar(601 + index as u128))
            .collect::<Vec<_>>(),
        test_scalar(5),
    )
    .unwrap()
    .into();
    let projected_functional: std::sync::Arc<[F]> = akita_algebra::ring::terminal_residue_kernel(
        &(0..D_B)
            .map(|index| test_scalar(809 + index as u128))
            .collect::<Vec<_>>(),
        test_scalar(5),
    )
    .unwrap()
    .into();
    for group in &mut plan.groups {
        group
            .direct_scan_weights
            .as_mut()
            .unwrap()
            .reduced_functionals = Some([
            a_functional.clone(),
            projected_functional.clone(),
            projected_functional.clone(),
        ]);
    }
    plan.direct_scan_functional = Some(
        PreparedCoefficientFunctional::reduced_evaluation(
            test_scalar(5),
            &[test_scalar(3); 6],
            plan.relation_address_geometry(),
        )
        .unwrap(),
    );
    let reduced_expected = plan
        .evaluate_direct_by_rows::<F>(
            &setup,
            &a_functional,
            &projected_functional,
            &projected_functional,
            D_A,
        )
        .unwrap();
    assert_eq!(plan.evaluate_direct::<F>(&setup).unwrap(), reduced_expected);
}

#[test]
fn reduced_fused_scan_matches_dense_rows_for_mixed_dimensions_and_chunks() {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let (inputs, groups, layout, lifted_plan, _, relation_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(5, &[2, 2, 1], role_dims, 32);
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1,
        &layout,
        &groups,
        PreparedRelationAddress::new(&relation_point).unwrap(),
        Some(&fold_gadget),
        lifted_plan.relation_address_geometry(),
    )
    .unwrap();
    let coefficient_variables = plan
        .relation_address_geometry()
        .relation_coefficient_variable_count();
    let coefficient_point = (0..coefficient_variables)
        .map(|index| test_scalar(401 + index as u128))
        .collect::<Vec<_>>();
    plan.materialize_direct_scan(
        PreparedCoefficientFunctional::reduced_evaluation(
            test_scalar(7),
            &coefficient_point,
            plan.relation_address_geometry(),
        )
        .unwrap(),
    )
    .unwrap();

    let base_dimension = plan.projection_geometry().base_ring_dim();
    let setup_coefficients = plan.required() * base_dimension;
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_coefficients,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_coefficients)
                .map(|index| test_scalar(503 + index as u128))
                .collect(),
        ),
    );
    let direct = plan.groups[0].direct_scan_weights.as_ref().unwrap();
    let [a_functional, b_functional, d_functional] = direct.reduced_functionals.as_ref().unwrap();
    assert!(std::sync::Arc::ptr_eq(b_functional, d_functional));
    let expected = plan
        .evaluate_direct_by_rows::<F>(
            &setup,
            a_functional,
            b_functional,
            d_functional,
            role_dims.d_a(),
        )
        .unwrap();
    assert_eq!(plan.evaluate_direct::<F>(&setup).unwrap(), expected);
}
