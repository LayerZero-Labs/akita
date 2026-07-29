use super::*;

#[test]
fn follows_outgoing_aware_relation_lanes() {
    let role_dims = CommitmentRingDims::uniform(TEST_D);
    let outgoing_ring_dim = TEST_D / 2;
    let (inputs, groups, witness_layout, plan, _, address_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(8, &[3, 5], role_dims, outgoing_ring_dim);
    let geometry = plan.relation_address_geometry();
    assert_eq!(
        geometry.common_relation_witness_coeff_count(),
        outgoing_ring_dim
    );
    assert_eq!(geometry.role_relation_lane_count(RingRole::Inner), 2);

    let alpha = test_scalar(3);
    let lane_alpha = scalar_powers(alpha, TEST_D)[outgoing_ring_dim];
    let lane_weight = |witness_column: usize| {
        let lane_start = witness_column * 2;
        eq_eval_at_index(&address_point, lane_start)
            + eq_eval_at_index(&address_point, lane_start + 1) * lane_alpha
    };
    let group = &groups[0];
    let first_unit = witness_layout.units_for_group(group.group_id).unwrap()[0];
    let first_e = first_unit
        .e_index(
            group.num_claims,
            inputs.depth_open(),
            0,
            first_unit.global_block_start(),
            0,
        )
        .unwrap();
    assert_eq!(plan.groups[0].e_eq_slice[0], lane_weight(first_e));

    let first_t = first_unit
        .t_index(
            group.num_claims,
            inputs.n_a(),
            inputs.depth_commit(),
            0,
            first_unit.global_block_start(),
            0,
            0,
        )
        .unwrap();
    assert_eq!(plan.groups[0].t_eq_slice[0], lane_weight(first_t));

    let mut expected_z = F::zero();
    for unit in witness_layout.units_for_group(group.group_id).unwrap() {
        for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
            let z = unit
                .z_index(
                    inputs.num_positions_per_block(),
                    inputs.depth_commit(),
                    group.depth_fold,
                    0,
                    0,
                    fold_digit,
                )
                .unwrap();
            expected_z -= lane_weight(z) * fold;
        }
    }
    assert_eq!(plan.groups[0].z_eq_slice[0], expected_z);
}

#[test]
fn composes_mixed_roles_with_smaller_outgoing_rings() {
    let role_dims = CommitmentRingDims {
        inner: 64,
        outer: 32,
        opening: 32,
    };
    let outgoing_ring_dim = 16;
    let (inputs, groups, witness_layout, plan, _, address_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(8, &[3, 5], role_dims, outgoing_ring_dim);
    let geometry = plan.relation_address_geometry();
    assert_eq!(geometry.role_relation_lane_count(RingRole::Inner), 4);
    assert_eq!(geometry.role_relation_lane_count(RingRole::Outer), 2);
    assert_eq!(geometry.role_relation_lane_count(RingRole::Opening), 2);

    let alpha = test_scalar(3);
    let alpha_base = scalar_powers(alpha, outgoing_ring_dim + 1)[outgoing_ring_dim];
    let lane_alpha = scalar_powers(alpha_base, 4);
    let lane_weight = |witness_column: usize, lane_offset: usize, lane_count: usize| {
        let lane_start = witness_column * 4 + lane_offset;
        (0..lane_count)
            .map(|lane| eq_eval_at_index(&address_point, lane_start + lane) * lane_alpha[lane])
            .sum::<F>()
    };
    let group = &groups[0];
    let first_unit = witness_layout.units_for_group(group.group_id).unwrap()[0];
    let first_e = first_unit
        .e_index(
            group.num_claims,
            inputs.depth_open(),
            0,
            first_unit.global_block_start(),
            0,
        )
        .unwrap();
    assert_eq!(plan.groups[0].e_eq_slice[0], lane_weight(first_e, 0, 2));
    assert_eq!(plan.groups[0].e_eq_slice[2], lane_weight(first_e, 2, 2));

    let first_t = first_unit
        .t_index(
            group.num_claims,
            inputs.n_a(),
            inputs.depth_commit(),
            0,
            first_unit.global_block_start(),
            0,
            0,
        )
        .unwrap();
    assert_eq!(plan.groups[0].t_eq_slice[0], lane_weight(first_t, 0, 2));
    assert_eq!(plan.groups[0].t_eq_slice[1], lane_weight(first_t, 2, 2));

    let mut expected_z = F::zero();
    for unit in witness_layout.units_for_group(group.group_id).unwrap() {
        for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
            let z = unit
                .z_index(
                    inputs.num_positions_per_block(),
                    inputs.depth_commit(),
                    group.depth_fold,
                    0,
                    0,
                    fold_digit,
                )
                .unwrap();
            expected_z -= lane_weight(z, 0, 4) * fold;
        }
    }
    assert_eq!(plan.groups[0].z_eq_slice[0], expected_z);
}
