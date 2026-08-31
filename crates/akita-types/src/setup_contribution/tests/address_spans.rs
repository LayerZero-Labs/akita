use super::*;

#[test]
fn uniform_current_roles_do_not_split_at_the_outgoing_dimension() {
    let role_dims = CommitmentRingDims::uniform(TEST_D);
    let outgoing_ring_dim = TEST_D / 2;
    let (inputs, groups, witness_layout, plan, _, address_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(8, &[3, 5], role_dims, outgoing_ring_dim);
    let geometry = plan.relation_address_geometry();
    assert_eq!(geometry.relation_coefficient_block_len(), TEST_D);
    assert_eq!(geometry.role_relation_lane_count(RingRole::Inner), 1);

    let lane_weight = |witness_column: usize| eq_eval_at_index(&address_point, witness_column);
    let group = &groups[0];
    let (e_eq_slice, t_eq_slice, z_eq_slice) = plan.group_column_eq_slices(0).unwrap();
    let first_unit = witness_layout
        .units_for_group(group.group_id)
        .unwrap()
        .next()
        .unwrap();
    let first_e = first_unit
        .e_coefficient_index(
            role_dims.d_d(),
            group.num_claims,
            inputs.depth_open(),
            0,
            first_unit.global_block_start(),
            0,
            0,
            0,
        )
        .unwrap()
        / geometry.relation_coefficient_block_len();
    assert_eq!(e_eq_slice[0], lane_weight(first_e));

    let first_t = first_unit
        .t_coefficient_index(
            role_dims.d_a(),
            role_dims.d_b(),
            group.num_claims,
            inputs.n_a(),
            inputs.depth_commit(),
            0,
            first_unit.global_block_start(),
            0,
            0,
            0,
            0,
        )
        .unwrap()
        / geometry.relation_coefficient_block_len();
    assert_eq!(t_eq_slice[0], lane_weight(first_t));

    let mut expected_z = F::zero();
    for unit in witness_layout.units_for_group(group.group_id).unwrap() {
        for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
            let z = unit
                .z_coefficient_index(
                    role_dims.d_a(),
                    inputs.num_positions_per_block(),
                    inputs.depth_commit(),
                    group.depth_fold,
                    0,
                    0,
                    fold_digit,
                    0,
                )
                .unwrap()
                / geometry.relation_coefficient_block_len();
            expected_z -= lane_weight(z) * fold;
        }
    }
    assert_eq!(z_eq_slice[0], expected_z);
}

#[test]
fn mixed_current_roles_ignore_outgoing_repacking() {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let alpha = test_scalar(3);
    let mut expected = None;
    for outgoing_ring_dim in [16, 32, 64] {
        let (inputs, groups, witness_layout, plan, _, address_point, fold_gadget) =
            structured_weight_fixture_with_outgoing(8, &[3, 5], role_dims, outgoing_ring_dim);
        let geometry = plan.relation_address_geometry();
        assert_eq!(geometry.relation_coefficient_block_len(), 64);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Inner), 2);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Outer), 1);
        assert_eq!(geometry.role_relation_lane_count(RingRole::Opening), 1);
        assert_eq!(
            geometry.digit_witness_domain().live_len(),
            witness_layout.live_coeff_len()
        );

        let lane_alpha = [F::one(), scalar_powers(alpha, 65)[64]];
        let lane_weight = |lane_start: usize, lane_count: usize| {
            (0..lane_count)
                .map(|lane| eq_eval_at_index(&address_point, lane_start + lane) * lane_alpha[lane])
                .sum::<F>()
        };
        let group = &groups[0];
        let (e_eq_slice, t_eq_slice, z_eq_slice) = plan.group_column_eq_slices(0).unwrap();
        let first_unit = witness_layout
            .units_for_group(group.group_id)
            .unwrap()
            .next()
            .unwrap();
        let first_e = first_unit
            .e_coefficient_index(
                role_dims.d_d(),
                group.num_claims,
                inputs.depth_open(),
                0,
                first_unit.global_block_start(),
                0,
                0,
                0,
            )
            .unwrap()
            / geometry.relation_coefficient_block_len();
        let second_e = first_unit
            .e_coefficient_index(
                role_dims.d_d(),
                group.num_claims,
                inputs.depth_open(),
                0,
                first_unit.global_block_start(),
                1,
                0,
                0,
            )
            .unwrap()
            / geometry.relation_coefficient_block_len();
        assert_eq!(e_eq_slice[0], lane_weight(first_e, 1));
        assert_eq!(e_eq_slice[inputs.depth_open()], lane_weight(second_e, 1));

        let first_t = first_unit
            .t_coefficient_index(
                role_dims.d_a(),
                role_dims.d_b(),
                group.num_claims,
                inputs.n_a(),
                inputs.depth_commit(),
                0,
                first_unit.global_block_start(),
                0,
                0,
                0,
                0,
            )
            .unwrap()
            / geometry.relation_coefficient_block_len();
        let second_t = first_unit
            .t_coefficient_index(
                role_dims.d_a(),
                role_dims.d_b(),
                group.num_claims,
                inputs.n_a(),
                inputs.depth_commit(),
                0,
                first_unit.global_block_start(),
                0,
                1,
                0,
                0,
            )
            .unwrap()
            / geometry.relation_coefficient_block_len();
        assert_eq!(t_eq_slice[0], lane_weight(first_t, 1));
        assert_eq!(t_eq_slice[inputs.depth_commit()], lane_weight(second_t, 1));

        let mut expected_z = F::zero();
        for unit in witness_layout.units_for_group(group.group_id).unwrap() {
            for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                let z = unit
                    .z_coefficient_index(
                        role_dims.d_a(),
                        inputs.num_positions_per_block(),
                        inputs.depth_commit(),
                        group.depth_fold,
                        0,
                        0,
                        fold_digit,
                        0,
                    )
                    .unwrap()
                    / geometry.relation_coefficient_block_len();
                expected_z -= lane_weight(z, 2) * fold;
            }
        }
        assert_eq!(z_eq_slice[0], expected_z);

        let rho = (0..plan.required().next_power_of_two().trailing_zeros() as usize)
            .map(|index| test_scalar(701 + index as u128))
            .collect::<Vec<_>>();
        let observed = (
            e_eq_slice.to_vec(),
            t_eq_slice.to_vec(),
            z_eq_slice.to_vec(),
            plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap(),
        );
        if let Some(expected) = &expected {
            assert_eq!(&observed, expected);
        } else {
            expected = Some(observed);
        }
    }
}
