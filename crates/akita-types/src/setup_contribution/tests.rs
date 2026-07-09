use super::plan::SetupContributionGroupPlan;
use super::*;
use crate::{
    gadget_row_scalars, AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix, LevelParams,
    RelationMatrixRowLayout,
};
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::{eval_ring_at_pows, scalar_powers};
use akita_field::Prime128OffsetA7F7;

type F = Prime128OffsetA7F7;
const TEST_D: usize = 64;

fn test_scalar(value: u128) -> F {
    F::from_canonical_u128(value)
}

fn finalize_test_plan(mut plan: SetupContributionPlan<F>) -> SetupContributionPlan<F> {
    for group in &mut plan.groups {
        group
            .refresh_segments(plan.d_rows, plan.d_physical_cols)
            .expect("valid cached setup scan segments");
    }
    plan
}

fn prepare_single_group_plan(
    inputs: &SetupContributionPlanInputs<F>,
    full_vec_randomness: &[F],
    eq_low: Option<&[F]>,
    z_block_low_eq: Option<&[F]>,
    fold_gadget: &[F],
    chunk_layout: &crate::WitnessLayout,
) -> Result<SetupContributionPlan<F>, AkitaError> {
    let single_group = SetupContributionGroupInputs::single_group_layout(inputs, chunk_layout, 0)?;
    let groups = std::slice::from_ref(&single_group.group);
    let static_plan = SetupContributionPlan::prepare_static(
        inputs,
        groups,
        single_group.d_row_start,
        single_group.d_rows,
        single_group.d_physical_cols,
    )?;
    SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        full_vec_randomness,
        eq_low,
        z_block_low_eq,
        Some(fold_gadget),
        groups,
    )
}

#[test]
fn dense_z_eq_slice_uses_relative_high_carry() {
    let block_len = 12;
    let depth_commit = 3;
    let depth_fold = 2;
    let num_points = 1;
    let z_range = block_len * depth_commit;
    let offset_z = 0;
    let full_vec_randomness = (0..9)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, 4);
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
        rows: 2,
        n_a: 1,
        n_b: 0,
        n_d: 0,
        num_groups: num_points,
        num_polys_per_group: vec![0],
        num_t_vectors: 0,
        num_claims: 1,
        num_blocks: 4,
        block_len,
        depth_open: 16,
        depth_commit,
        depth_fold,
        inner_width: z_range,
        eq_tau1: vec![test_scalar(11), test_scalar(12)],
    };

    let chunk_layout = crate::WitnessLayout {
        blocks_per_chunk: 4,
        chunks: vec![crate::WitnessChunkLayout {
            offset_z,
            offset_e: 0,
            offset_t: 64,
            offset_r: Some(0),
            global_block_base: 0,
        }],
        chunk_lengths: vec![crate::WitnessChunkLengths {
            z_len: z_range,
            e_len: 0,
            t_len: 0,
            r_len: Some(0),
        }],
    };
    let plan = prepare_single_group_plan(
        &inputs,
        &full_vec_randomness,
        None,
        None,
        &fold_gadget,
        &chunk_layout,
    )
    .unwrap();

    let expected = (0..z_range)
        .map(|c| {
            let dc = c % depth_commit;
            let blk = c / depth_commit;
            let mut acc = F::zero();
            for pt in 0..num_points {
                for (df, &fg) in fold_gadget.iter().enumerate().take(depth_fold) {
                    let x = blk
                        + block_len * pt
                        + block_len * num_points * df
                        + block_len * num_points * depth_fold * dc;
                    acc += eq_eval_at_index(&full_vec_randomness, offset_z + x) * fg;
                }
            }
            -acc
        })
        .collect::<Vec<_>>();

    assert_eq!(plan.groups[0].z_eq_slice, expected);
}

#[test]
fn setup_a_z_weights_do_not_include_commit_gadget() {
    let block_len = 8;
    let depth_commit = 3;
    let depth_fold = 2;
    let num_points = 1;
    let log_basis = 4;
    let z_range = block_len * depth_commit;
    let offset_z = 0;
    let full_vec_randomness = (0..8)
        .map(|idx| test_scalar(701 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let commit_gadget = gadget_row_scalars::<F>(depth_commit, log_basis);
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithoutDBlock,
        rows: 2,
        n_a: 1,
        n_b: 0,
        n_d: 0,
        num_groups: num_points,
        num_polys_per_group: vec![0],
        num_t_vectors: 0,
        num_claims: 1,
        num_blocks: 4,
        block_len,
        depth_open: 16,
        depth_commit,
        depth_fold,
        inner_width: z_range,
        eq_tau1: vec![test_scalar(11), test_scalar(12)],
    };
    let chunk_layout = crate::WitnessLayout {
        blocks_per_chunk: 4,
        chunks: vec![crate::WitnessChunkLayout {
            offset_z,
            offset_e: z_range,
            offset_t: z_range,
            offset_r: Some(0),
            global_block_base: 0,
        }],
        chunk_lengths: vec![crate::WitnessChunkLengths {
            z_len: z_range,
            e_len: 0,
            t_len: 0,
            r_len: Some(0),
        }],
    };

    let plan = prepare_single_group_plan(
        &inputs,
        &full_vec_randomness,
        None,
        None,
        &fold_gadget,
        &chunk_layout,
    )
    .unwrap();

    let expected = (0..z_range)
        .map(|k| {
            let blk = k / depth_commit;
            let dc = k % depth_commit;
            let mut acc = F::zero();
            for pt in 0..num_points {
                for (df, &fold) in fold_gadget.iter().enumerate().take(depth_fold) {
                    let x = blk + block_len * (pt + num_points * (df + depth_fold * dc));
                    acc += eq_eval_at_index(&full_vec_randomness, offset_z + x) * fold;
                }
            }
            -acc
        })
        .collect::<Vec<_>>();
    let wrong_with_commit_gadget = expected
        .iter()
        .enumerate()
        .map(|(k, &weight)| weight * commit_gadget[k % depth_commit])
        .collect::<Vec<_>>();

    assert_eq!(plan.groups[0].z_eq_slice, expected);
    assert_ne!(
        plan.groups[0].z_eq_slice, wrong_with_commit_gadget,
        "A setup weights are for A * G_fold * z_hat, not A * G_commit * G_fold * z_hat"
    );
}

#[test]
fn single_group_plan_supports_multi_chunk_weights() {
    let num_blocks = 4;
    let blocks_per_chunk = 2;
    let num_claims = 3;
    let depth_open = 2;
    let depth_commit = 2;
    let depth_fold = 2;
    let block_len = 4;
    let n_a = 2;
    let n_b = 2;
    let n_d = 1;
    let log_basis = 4;
    let z_range = block_len * depth_commit;
    let e_len_per_chunk = num_claims * depth_open * blocks_per_chunk;
    let t_len_per_chunk = n_a * num_claims * depth_open * blocks_per_chunk;
    let chunk_stride = z_range + e_len_per_chunk + t_len_per_chunk;
    let chunks = (0..2)
        .map(|idx| {
            let base = idx * chunk_stride;
            let offset_e = base + z_range;
            let offset_t = offset_e + e_len_per_chunk;
            crate::WitnessChunkLayout {
                offset_z: base,
                offset_e,
                offset_t,
                offset_r: (idx == 1).then_some(offset_t + t_len_per_chunk),
                global_block_base: idx * blocks_per_chunk,
            }
        })
        .collect::<Vec<_>>();
    let rows = 1 + n_a + n_b + n_d;
    let inputs = SetupContributionPlanInputs {
        relation_matrix_row_layout: RelationMatrixRowLayout::WithDBlock,
        rows,
        n_a,
        n_b,
        n_d,
        num_groups: 1,
        num_polys_per_group: vec![num_claims],
        num_t_vectors: num_claims,
        num_claims,
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        inner_width: z_range,
        eq_tau1: (0..rows.next_power_of_two())
            .map(|idx| test_scalar(11 + idx as u128))
            .collect(),
    };
    let full_vec_randomness = (0..10)
        .map(|idx| test_scalar(101 + idx as u128))
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, log_basis);
    let groups = [SetupContributionGroupInputs {
        e_col_offset: 0,
        num_claims,
        num_blocks,
        block_len,
        depth_open,
        depth_commit,
        depth_fold,
        log_basis,
        n_a,
        n_b,
        t_cols_per_vector: n_a * depth_open * num_blocks,
        a_row_start: 1,
        b_row_start: 1 + n_a,
        blocks_per_chunk,
        chunks,
    }];
    let static_plan = SetupContributionPlan::prepare_static(
        &inputs,
        &groups,
        rows - n_d,
        n_d,
        num_claims * num_blocks * depth_open,
    )
    .unwrap();
    let plan = SetupContributionPlan::finish_plan::<F>(
        &static_plan,
        &full_vec_randomness,
        None,
        None,
        Some(&fold_gadget),
        &groups,
    )
    .unwrap();

    let setup_len = plan.required().unwrap();
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
            segments: Vec::new(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)],
            b_weights: vec![test_scalar(37), test_scalar(41)],
            d_weights: vec![test_scalar(43), test_scalar(47)],
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
                segments: Vec::new(),
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_row_weights: vec![test_scalar(29), test_scalar(31)],
                b_weights: vec![test_scalar(37), test_scalar(41)],
                d_weights: vec![test_scalar(43), test_scalar(47)],
            },
            SetupContributionGroupPlan {
                e_col_offset: 0,
                t_cols: 4,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new(),
                e_eq_slice: vec![test_scalar(53), test_scalar(59)],
                t_eq_slice: vec![
                    test_scalar(61),
                    test_scalar(67),
                    test_scalar(71),
                    test_scalar(73),
                ],
                z_eq_slice: vec![test_scalar(79), test_scalar(83), test_scalar(89)],
                a_row_weights: vec![test_scalar(97), test_scalar(101)],
                b_weights: vec![test_scalar(103), test_scalar(107)],
                d_weights: vec![test_scalar(109), test_scalar(113)],
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
            segments: Vec::new(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)],
            b_weights: vec![test_scalar(37), test_scalar(41)],
            d_weights: vec![test_scalar(43), test_scalar(47)],
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
            segments: Vec::new(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)],
            b_weights: vec![test_scalar(37), test_scalar(41)],
            d_weights: vec![test_scalar(43), test_scalar(47)],
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
            segments: Vec::new(),
            e_eq_slice: vec![test_scalar(2), test_scalar(3)],
            t_eq_slice: vec![
                test_scalar(5),
                test_scalar(7),
                test_scalar(11),
                test_scalar(13),
            ],
            z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
            a_row_weights: vec![test_scalar(29), test_scalar(31)],
            b_weights: vec![test_scalar(37), test_scalar(41)],
            d_weights: vec![test_scalar(43), test_scalar(47)],
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
                segments: Vec::new(),
                e_eq_slice: vec![test_scalar(2), test_scalar(3)],
                t_eq_slice: vec![
                    test_scalar(5),
                    test_scalar(7),
                    test_scalar(11),
                    test_scalar(13),
                ],
                z_eq_slice: vec![test_scalar(17), test_scalar(19), test_scalar(23)],
                a_row_weights: vec![test_scalar(29), test_scalar(31)],
                b_weights: vec![test_scalar(37), test_scalar(41)],
                d_weights: vec![test_scalar(43), test_scalar(47)],
            },
            SetupContributionGroupPlan {
                e_col_offset: 0,
                t_cols: 6,
                z_cols: 3,
                n_a: 2,
                n_b: 2,
                required: 0,
                segments: Vec::new(),
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
                a_row_weights: vec![test_scalar(103), test_scalar(107)],
                b_weights: vec![test_scalar(109), test_scalar(113)],
                d_weights: vec![test_scalar(127), test_scalar(131)],
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

#[test]
fn from_level_params_rejects_non_pow2_num_blocks() {
    let mut lp = LevelParams::log_basis_stub(3);
    lp.ring_dimension = 64;
    lp.role_dims = crate::CommitmentRingDims::uniform(64);
    lp.num_blocks = 3;
    lp.block_len = 8;
    lp.num_digits_commit = 2;
    lp.num_digits_open = 3;
    assert!(SetupContributionPlanInputs::<F>::from_level_params(
        &lp,
        &[2],
        RelationMatrixRowLayout::WithoutDBlock,
        2,
    )
    .is_err());
}
