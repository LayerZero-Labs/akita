use super::super::test_oracle_weights::{
    setup_e_col_weights, setup_t_col_weights, setup_z_col_weights, RoleLaneSpec, RoleLaneWeighting,
};
use super::*;
use akita_algebra::offset_eq::OffsetEqWindow;
use jolt_field::{ExtField, MulBaseUnreduced, Ring};

fn literal_terminal_functional<E: Field>(point: &[E], dimension: usize, alpha: E) -> Vec<E> {
    assert_eq!(point.len(), dimension.trailing_zeros() as usize);
    let equality = (0..dimension)
        .map(|index| eq_eval_at_index(point, index))
        .collect::<Vec<_>>();
    let powers = scalar_powers(alpha, dimension);
    (0..dimension)
        .map(|multiplier_coefficient| {
            (0..dimension).fold(E::zero(), |sum, witness_coefficient| {
                let exponent = multiplier_coefficient + witness_coefficient;
                let term = equality[witness_coefficient] * powers[exponent % dimension];
                if exponent < dimension {
                    sum + term
                } else {
                    sum - term
                }
            })
        })
        .collect()
}

fn literal_native_functional<E: Field>(
    plan: &SetupContributionPlan<E>,
    coefficient_point: &[E],
    dimension: usize,
    alpha: E,
) -> Vec<E> {
    let coefficient_dimension = plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let ratio = dimension / coefficient_dimension;
    let mut native_point = coefficient_point.to_vec();
    native_point
        .extend_from_slice(&plan.relation_address.point()[..ratio.trailing_zeros() as usize]);
    literal_terminal_functional(&native_point, dimension, alpha)
}

fn literal_ring_dot<E>(ring: &[F], functional: &[E]) -> E
where
    E: ExtField<F>,
{
    assert_eq!(ring.len(), functional.len());
    ring.iter()
        .zip(functional)
        .fold(E::zero(), |sum, (&coefficient, &weight)| {
            sum + weight.mul_base(coefficient)
        })
}

fn naive_physical_b_weights<E: Field>(
    group: &SetupContributionGroupPlan<E>,
    logical: &[E],
    row_weights: &[E],
    b_row_start: usize,
) -> Vec<E> {
    let slice_count = group.physical_b.geometry().slice_count().get();
    let rows = group.physical_b.physical_rows();
    let columns = group.physical_b.physical_input_width();
    let maximum_blocks = group.num_live_blocks.div_ceil(slice_count);
    let per_block = columns / (group.num_claims * maximum_blocks);
    let mut physical = vec![E::zero(); rows * columns];
    for slice in 0..slice_count {
        let block_start = slice * group.num_live_blocks / slice_count;
        let block_end = (slice + 1) * group.num_live_blocks / slice_count;
        for row in 0..rows {
            let row_weight = row_weights[b_row_start + slice * rows + row];
            for claim in 0..group.num_claims {
                for block in block_start..block_end {
                    for offset in 0..per_block {
                        let physical_column =
                            (claim * maximum_blocks + block - block_start) * per_block + offset;
                        let logical_column =
                            (claim * group.num_live_blocks + block) * per_block + offset;
                        physical[row * columns + physical_column] +=
                            row_weight * logical[logical_column];
                    }
                }
            }
        }
    }
    physical
}

#[allow(clippy::too_many_arguments)]
fn reduced_direct_literal_oracle<E>(
    plan: &SetupContributionPlan<E>,
    setup: &AkitaExpandedSetup<F>,
    inputs: &TestSetupInputs,
    group_inputs: &[SetupContributionGroupInputs],
    layout: &WitnessLayout,
    fold_gadget: &[F],
    row_weights: &[E],
    coefficient_point: &[E],
    alpha: E,
) -> E
where
    E: ExtField<F> + MulBaseUnreduced<F>,
{
    let mut evaluation = E::zero();
    let row_families =
        crate::RelationWitnessGeometry::for_level(&inputs.level_params, &inputs.opening_batch, 1)
            .unwrap()
            .rhs_layout()
            .row_families()
            .unwrap();
    let d_row_start = row_families
        .iter()
        .position(|family| matches!(family, crate::RelationRowFamily::Opening { .. }))
        .unwrap();
    let coefficient_dimension = plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let mut d_column_start = 0usize;
    for group in &plan.groups {
        let group_input = group_inputs
            .iter()
            .find(|candidate| candidate.group_id == group.group_id)
            .unwrap();
        let role_spec = |dimension: usize| {
            let role_lanes = dimension / coefficient_dimension;
            RoleLaneSpec {
                a_ratio: group.role_dims.d_a() / coefficient_dimension,
                role_subcolumns: group.role_dims.d_a() / dimension,
                role_lanes,
                weighting: RoleLaneWeighting::ReducedHigh,
            }
        };
        let high_equality = |dimension: usize| {
            let role_lanes = dimension / coefficient_dimension;
            OffsetEqWindow::new(
                &plan.relation_address.point()[role_lanes.trailing_zeros() as usize..],
            )
            .unwrap()
        };
        let e_weights = setup_e_col_weights(
            layout,
            layout.live_coeff_len(),
            group.group_id,
            group.num_live_blocks,
            group.num_claims,
            group.depth_open,
            &high_equality(group.role_dims.d_d()),
            &role_spec(group.role_dims.d_d()),
        )
        .unwrap();
        let t_weights = setup_t_col_weights(
            layout,
            layout.live_coeff_len(),
            group.group_id,
            group.num_live_blocks,
            group.depth_commit,
            group.n_a,
            group.num_claims,
            &high_equality(group.role_dims.d_b()),
            &role_spec(group.role_dims.d_b()),
        )
        .unwrap();
        let mut z_weights = vec![E::zero(); group.z_cols];
        setup_z_col_weights(
            layout,
            layout.live_coeff_len(),
            group.group_id,
            group.num_positions_per_block,
            group.depth_witness,
            fold_gadget.len(),
            &high_equality(group.role_dims.d_a()),
            fold_gadget,
            &role_spec(group.role_dims.d_a()),
            &mut z_weights,
        )
        .unwrap();
        let a_functional =
            literal_native_functional(plan, coefficient_point, group.role_dims.d_a(), alpha);
        let b_functional =
            literal_native_functional(plan, coefficient_point, group.role_dims.d_b(), alpha);
        let d_functional =
            literal_native_functional(plan, coefficient_point, group.role_dims.d_d(), alpha);

        let d_view = setup
            .shared_matrix()
            .ring_view_dyn(plan.d_rows, plan.d_physical_cols, group.role_dims.d_d())
            .unwrap();
        for row in 0..plan.d_rows {
            for (local_column, &column_weight) in e_weights.iter().enumerate() {
                let column = d_column_start + local_column;
                let ring = d_view.row_flat(row).unwrap()
                    [column * group.role_dims.d_d()..(column + 1) * group.role_dims.d_d()]
                    .as_ref();
                evaluation += row_weights[d_row_start + row]
                    * column_weight
                    * literal_ring_dot(ring, &d_functional);
            }
        }

        let a_view = setup
            .shared_matrix()
            .ring_view_dyn(group.n_a, group.z_cols, group.role_dims.d_a())
            .unwrap();
        for row in 0..group.n_a {
            for (column, &column_weight) in z_weights.iter().enumerate() {
                let ring = &a_view.row_flat(row).unwrap()
                    [column * group.role_dims.d_a()..(column + 1) * group.role_dims.d_a()];
                evaluation += row_weights[group_input.a_row_start + row]
                    * column_weight
                    * literal_ring_dot(ring, &a_functional);
            }
        }

        let b_weights =
            naive_physical_b_weights(group, &t_weights, row_weights, group_input.b_row_start);
        let b_view = setup
            .shared_matrix()
            .ring_view_dyn(
                group.physical_b.physical_rows(),
                group.physical_b.physical_input_width(),
                group.role_dims.d_b(),
            )
            .unwrap();
        for row in 0..group.physical_b.physical_rows() {
            for column in 0..group.physical_b.physical_input_width() {
                let ring = &b_view.row_flat(row).unwrap()
                    [column * group.role_dims.d_b()..(column + 1) * group.role_dims.d_b()];
                evaluation += b_weights[row * group.physical_b.physical_input_width() + column]
                    * literal_ring_dot(ring, &b_functional);
            }
        }
        d_column_start += e_weights.len();
    }
    assert_eq!(d_column_start, plan.d_physical_cols);
    evaluation
}

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
    let lifted_groups =
        match std::mem::replace(&mut plan.direct_scan_state, DirectScanState::Unprepared) {
            DirectScanState::Lifted { groups, .. } => groups,
            _ => panic!("fixture must start with lifted direct-scan state"),
        };
    let reduced_groups = lifted_groups
        .into_iter()
        .map(|weights| ReducedDirectScanWeights {
            weights,
            roles: [
                ReducedRoleCoefficientState {
                    functional: a_functional.clone(),
                    equality: vec![F::one(); D_A].into(),
                },
                ReducedRoleCoefficientState {
                    functional: projected_functional.clone(),
                    equality: vec![F::one(); D_B].into(),
                },
                ReducedRoleCoefficientState {
                    functional: projected_functional.clone(),
                    equality: vec![F::one(); D_D].into(),
                },
            ],
        })
        .collect();
    plan.direct_scan_state = DirectScanState::Reduced {
        alpha: test_scalar(5),
        coefficient_point: vec![test_scalar(3); 6].into(),
        groups: reduced_groups,
    };
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
        inputs.eq_tau1.clone(),
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
    let DirectScanState::Reduced { groups: direct, .. } = &plan.direct_scan_state else {
        panic!("reduced fixture must prepare reduced direct-scan state");
    };
    let [_a_role, b_role, d_role] = &direct[0].roles;
    assert!(std::sync::Arc::ptr_eq(
        &b_role.functional,
        &d_role.functional
    ));
    assert!(std::sync::Arc::ptr_eq(&b_role.equality, &d_role.equality));
    let expected = reduced_direct_literal_oracle(
        &plan,
        &setup,
        &inputs,
        &groups,
        &layout,
        &fold_gadget,
        &inputs.eq_tau1,
        &coefficient_point,
        test_scalar(7),
    );
    assert_eq!(plan.evaluate_direct::<F>(&setup).unwrap(), expected);
}

#[test]
fn reduced_fused_scan_matches_independent_heterogeneous_two_group_oracle() {
    let HeterogeneousSetupFixture {
        inputs,
        groups,
        witness_layout,
        relation_address_geometry,
        relation_point,
        fold_gadget,
    } = heterogeneous_setup_fixture();
    let alpha = test_scalar(23);
    let coefficient_point = (0..relation_address_geometry.relation_coefficient_variable_count())
        .map(|index| test_scalar(1701 + index as u128))
        .collect::<Vec<_>>();
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        &witness_layout,
        &groups,
        PreparedRelationAddress::new(&relation_point).unwrap(),
        Some(&fold_gadget),
        relation_address_geometry,
    )
    .unwrap();
    plan.materialize_direct_scan(
        PreparedCoefficientFunctional::reduced_evaluation(
            alpha,
            &coefficient_point,
            relation_address_geometry,
        )
        .unwrap(),
    )
    .unwrap();
    let setup_coefficients = plan.required() * plan.projection_geometry().base_ring_dim();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_coefficients,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_coefficients)
                .map(|index| test_scalar(1801 + index as u128))
                .collect(),
        ),
    );
    assert_eq!(
        plan.evaluate_direct::<F>(&setup).unwrap(),
        reduced_direct_literal_oracle(
            &plan,
            &setup,
            &inputs,
            &groups,
            &witness_layout,
            &fold_gadget,
            &inputs.eq_tau1,
            &coefficient_point,
            alpha,
        )
    );
}

#[test]
fn reduced_fused_scan_matches_independent_oracle_over_extension_field() {
    type X = jolt_field::Ext2<F>;
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let (inputs, groups, layout, base_plan, _, base_relation_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(5, &[2, 2, 1], role_dims, 32);
    let extension =
        |value: u64| X::from_base_slice(&[F::from_u64(value), F::from_u64(value + 10_000)]);
    let relation_point = base_relation_point
        .iter()
        .enumerate()
        .map(|(index, _)| extension(601 + index as u64))
        .collect::<Vec<_>>();
    let geometry = base_plan.relation_address_geometry();
    let coefficient_point = (0..geometry.relation_coefficient_variable_count())
        .map(|index| extension(701 + index as u64))
        .collect::<Vec<_>>();
    let row_weights = inputs
        .eq_tau1
        .iter()
        .copied()
        .map(X::lift_base)
        .collect::<Vec<_>>();
    let alpha = extension(809);
    let mut plan = SetupContributionPlan::<X>::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        row_weights.clone().into(),
        &layout,
        &groups,
        PreparedRelationAddress::new(&relation_point).unwrap(),
        Some(&fold_gadget),
        geometry,
    )
    .unwrap();
    plan.materialize_direct_scan(
        PreparedCoefficientFunctional::reduced_evaluation(alpha, &coefficient_point, geometry)
            .unwrap(),
    )
    .unwrap();
    let setup_coefficients = plan.required() * plan.projection_geometry().base_ring_dim();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_coefficients,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_coefficients)
                .map(|index| test_scalar(901 + index as u128))
                .collect(),
        ),
    );
    assert_eq!(
        plan.evaluate_direct::<F>(&setup).unwrap(),
        reduced_direct_literal_oracle(
            &plan,
            &setup,
            &inputs,
            &groups,
            &layout,
            &fold_gadget,
            &row_weights,
            &coefficient_point,
            alpha,
        )
    );
}
