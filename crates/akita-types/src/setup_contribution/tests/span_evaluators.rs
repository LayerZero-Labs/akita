use super::*;

fn rho_for_required(required: usize) -> Vec<F> {
    let bits = required.next_power_of_two().trailing_zeros() as usize;
    (0..bits)
        .map(|idx| test_scalar(901 + idx as u128))
        .collect()
}
fn projection_scales(alpha: F, base_d: usize, role_d: usize) -> Vec<F> {
    scalar_powers(alpha, role_d)
        .chunks(base_d)
        .map(|chunk| chunk[0])
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn projected_setup_weight_reference(
    plan: &SetupContributionPlan<F>,
    rho: &[F],
    required: usize,
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
    a_scales: &[F],
    b_scales: &[F],
    d_scales: &[F],
) -> F {
    let mut acc = F::zero();
    for base_idx in 0..required {
        let mut weight = F::zero();
        for group in &plan.groups {
            let (e_eq_slice, _t_eq_slice, z_eq_slice) = group.column_eq_slices().unwrap();
            let d_idx = base_idx / d_ratio;
            if d_idx < plan.d_rows * plan.d_physical_cols {
                let d_col = d_idx % plan.d_physical_cols;
                let d_row = d_idx / plan.d_physical_cols;
                if group.d_col_range.contains(&d_col) {
                    weight += d_scales[base_idx % d_ratio]
                        * plan.d_weights[d_row]
                        * e_eq_slice[d_col - group.d_col_range.start];
                }
            }
            let b_idx = base_idx / b_ratio;
            if b_idx < group.physical_n_b * group.physical_t_cols {
                weight += b_scales[base_idx % b_ratio]
                    * group.direct_scan_weights.as_ref().unwrap().b_setup[b_idx];
            }
            let a_idx = base_idx / a_ratio;
            if a_idx < group.n_a * group.z_cols {
                let a_col = a_idx % group.z_cols;
                let a_row = a_idx / group.z_cols;
                weight +=
                    a_scales[base_idx % a_ratio] * group.a_row_weights[a_row] * z_eq_slice[a_col];
            }
        }
        acc += eq_eval_at_index(rho, base_idx) * weight;
    }
    acc
}
fn assert_span_mle_matches_dense(plan: &SetupContributionPlan<F>, rho: &[F], alpha: F) {
    let dense = plan
        .materialize_setup_index_weights(alpha)
        .unwrap()
        .into_iter()
        .enumerate()
        .fold(F::zero(), |acc, (index, weight)| {
            acc + eq_eval_at_index(rho, index) * weight
        });
    assert_eq!(
        plan.evaluate_setup_index_weight_mle(rho, alpha).unwrap(),
        dense
    );
}

fn assert_fixture_setup_index_mle_matches_dense(
    ownership_widths: &[usize],
    role_dims: CommitmentRingDims,
    outgoing_ring_dim: usize,
) {
    let alpha = test_scalar(3);
    let (_, _, _, plan, _, _, _) =
        structured_weight_fixture_with_outgoing(8, ownership_widths, role_dims, outgoing_ring_dim);
    let rho = rho_for_required(plan.required());
    assert_span_mle_matches_dense(&plan, &rho, alpha);
}

pub(super) fn structured_slice_reference(
    group: &SetupContributionGroupPlan<F>,
    block_challenges: &[F],
    opening_a_evals: &[F],
    alpha: F,
) -> F {
    let (e_eq_slice, t_eq_slice, z_eq_slice) = group.column_eq_slices().unwrap();
    let (outer_subcolumns, opening_subcolumns) =
        SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims).unwrap();
    let role_dims = group.role_dims;
    let alpha_powers = scalar_powers(alpha, role_dims.d_a());
    let opening_gadget = gadget_row_scalars::<F>(group.depth_open, group.log_basis_open);
    let commitment_gadget = gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
    let witness_gadget = gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
    let mut evaluation = F::zero();
    for claim in 0..group.num_claims {
        for block in 0..group.num_live_blocks {
            let challenge = block_challenges[claim * group.num_live_blocks + block];
            for subcolumn in 0..opening_subcolumns {
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    let column = (((claim * group.num_live_blocks + block) * opening_subcolumns
                        + subcolumn)
                        * group.depth_open)
                        + digit;
                    evaluation += challenge
                        * group.consistency_weight
                        * e_eq_slice[column]
                        * gadget
                        * alpha_powers[subcolumn * role_dims.d_d()];
                }
            }
            for row in 0..group.n_a {
                for subcolumn in 0..outer_subcolumns {
                    for (digit, &gadget) in commitment_gadget.iter().enumerate() {
                        let column = ((((claim * group.num_live_blocks + block) * group.n_a
                            + row)
                            * outer_subcolumns
                            + subcolumn)
                            * group.depth_commit)
                            + digit;
                        evaluation += challenge
                            * group.a_row_weights[row]
                            * t_eq_slice[column]
                            * gadget
                            * alpha_powers[subcolumn * role_dims.d_b()];
                    }
                }
            }
        }
    }
    for (position, &opening) in opening_a_evals.iter().enumerate() {
        for (digit, &gadget) in witness_gadget.iter().enumerate() {
            evaluation += group.consistency_weight
                * opening
                * z_eq_slice[position * group.depth_witness + digit]
                * gadget;
        }
    }
    evaluation
}

#[test]
fn canonical_tensors_match_dense_oracles_across_geometries() {
    let cases = [
        (&[8][..], CommitmentRingDims::uniform(TEST_D), TEST_D),
        (
            &[2, 2, 2, 2][..],
            CommitmentRingDims::uniform(TEST_D),
            TEST_D,
        ),
        (
            &[3, 5][..],
            CommitmentRingDims {
                inner: 128,
                outer: 64,
                opening: 64,
            },
            16,
        ),
    ];
    let alpha = test_scalar(3);
    for (ownership_widths, role_dims, outgoing_ring_dim) in cases {
        let (_, _, layout, full, _, _, _) = structured_weight_fixture_with_outgoing(
            8,
            ownership_widths,
            role_dims,
            outgoing_ring_dim,
        );
        let rho = rho_for_required(full.required());
        let dense = full
            .materialize_setup_index_weights(alpha)
            .unwrap()
            .into_iter()
            .enumerate()
            .fold(F::zero(), |acc, (index, weight)| {
                acc + eq_eval_at_index(&rho, index) * weight
            });
        assert_eq!(
            full.evaluate_setup_index_weight_mle(&rho, alpha).unwrap(),
            dense
        );
        let group = &full.groups[0];
        let expected_families = layout.units_for_group(group.group_id).unwrap().count();
        assert_eq!(group.a_tensors.len(), expected_families);
        assert!(group.a_tensors.iter().all(|family| {
            family
                .axes
                .iter()
                .any(|axis| axis.left_stride == 0 && axis.len == group.fold_gadget.len())
        }));
        let block_challenges = (0..group.num_claims * group.num_live_blocks)
            .map(|index| test_scalar(401 + index as u128))
            .collect::<Vec<_>>();
        let opening_a_evals = (0..group.num_positions_per_block)
            .map(|index| test_scalar(501 + index as u128))
            .collect::<Vec<_>>();
        let reference =
            structured_slice_reference(group, &block_challenges, &opening_a_evals, alpha);
        assert_eq!(
            full.evaluate_structured_group::<F>(
                group.group_id,
                &block_challenges,
                &opening_a_evals,
                alpha,
            )
            .unwrap(),
            reference
        );
    }
}

#[test]
fn span_setup_index_mle_matches_dense_single_chunk() {
    let (_, _, _, plan, _, _, _) =
        structured_weight_fixture(8, &[8], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    assert_span_mle_matches_dense(&plan, &rho, alpha);
}
#[test]
fn span_setup_index_mle_matches_dense_multi_chunk() {
    let (_, _, _, plan, _, _, _) =
        structured_weight_fixture(8, &[2, 2, 2, 2], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    assert_span_mle_matches_dense(&plan, &rho, alpha);
}

#[test]
fn sliced_b_setup_weights_contract_logical_rows_onto_one_physical_matrix() {
    let slice_count = crate::CommitmentSliceCount::try_new(4).unwrap();
    let (_, _, _, plan, _, _, _) = structured_weight_fixture_with_slices(
        8,
        &[3, 5],
        CommitmentRingDims::uniform(TEST_D),
        TEST_D,
        slice_count,
    );
    let group = &plan.groups[0];
    let (_, logical_t, _) = group.column_eq_slices().unwrap();
    let physical = &group.direct_scan_weights.as_ref().unwrap().b_setup;
    let ranges = slice_count.block_ranges(group.num_live_blocks).unwrap();
    let max_blocks = ranges.iter().map(std::ops::Range::len).max().unwrap();
    let per_block = group.physical_t_cols / (group.num_claims * max_blocks);
    let mut expected = vec![F::zero(); group.physical_n_b * group.physical_t_cols];
    for row in 0..group.physical_n_b {
        for (slice_index, range) in ranges.iter().enumerate() {
            let row_weight = group.b_weights[slice_index * group.physical_n_b + row];
            for claim in 0..group.num_claims {
                for local_block in 0..range.len() {
                    for offset in 0..per_block {
                        let physical_col = (claim * max_blocks + local_block) * per_block + offset;
                        let logical_col =
                            (claim * group.num_live_blocks + range.start + local_block) * per_block
                                + offset;
                        expected[row * group.physical_t_cols + physical_col] +=
                            row_weight * logical_t[logical_col];
                    }
                }
            }
        }
    }
    assert_eq!(physical, &expected);

    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    assert_span_mle_matches_dense(&plan, &rho, alpha);
    let setup_ring_elements = plan.required();
    let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: setup_ring_elements * TEST_D,
            setup_seed: [0u8; 32].into(),
        },
        FlatMatrix::from_flat_data(
            (0..setup_ring_elements * TEST_D)
                .map(|idx| test_scalar(1201 + idx as u128))
                .collect(),
        ),
    );
    let alpha_pows = scalar_powers(alpha, TEST_D);
    assert_eq!(
        plan.evaluate_direct::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows)
            .unwrap(),
        plan.evaluate_direct_by_rows::<F>(&setup, &alpha_pows, &alpha_pows, &alpha_pows, TEST_D,)
            .unwrap()
    );
}

#[test]
fn uniform_setup_index_mle_matches_single_chunk_plan() {
    assert_fixture_setup_index_mle_matches_dense(&[8], CommitmentRingDims::uniform(TEST_D), TEST_D);
}

#[test]
fn uniform_setup_index_mle_matches_multi_chunk_plan() {
    assert_fixture_setup_index_mle_matches_dense(
        &[2, 2, 2, 2],
        CommitmentRingDims::uniform(TEST_D),
        TEST_D,
    );
}

#[test]
fn uniform_setup_index_mle_ignores_outgoing_repacking() {
    assert_fixture_setup_index_mle_matches_dense(
        &[2, 2, 2, 2],
        CommitmentRingDims::uniform(TEST_D),
        TEST_D * 2,
    );
}

#[test]
fn setup_index_mle_matches_mixed_role_plans() {
    for role_dims in [
        CommitmentRingDims {
            inner: 128,
            outer: 64,
            opening: 64,
        },
        CommitmentRingDims {
            inner: 256,
            outer: 64,
            opening: 128,
        },
    ] {
        for ownership_widths in [&[8][..], &[2, 2, 2, 2][..], &[3, 5][..]] {
            assert_fixture_setup_index_mle_matches_dense(ownership_widths, role_dims, 16);
        }
    }
}

#[test]
fn span_setup_index_mle_supports_non_power_of_two_ownership_widths() {
    let (_, _, _, plan, _, _, _) =
        structured_weight_fixture(8, &[3, 5], CommitmentRingDims::uniform(TEST_D));
    let alpha = test_scalar(3);
    let rho = rho_for_required(plan.required());
    assert_span_mle_matches_dense(&plan, &rho, alpha);
}
#[test]
fn span_setup_index_mle_applies_mixed_role_projection_lanes() {
    let alpha = test_scalar(3);
    let role_dims = crate::CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let setup_ring_dim = 64;
    for ownership_widths in [&[8][..], &[2, 2, 2, 2][..], &[3, 5][..]] {
        let (_, _, _, plan, _, _, _) = structured_weight_fixture(8, ownership_widths, role_dims);
        let rho = rho_for_required(plan.required());
        let got = plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap();
        let expected = projected_setup_weight_reference(
            &plan,
            &rho,
            plan.required(),
            role_dims.d_a() / setup_ring_dim,
            role_dims.d_b() / setup_ring_dim,
            role_dims.d_d() / setup_ring_dim,
            &projection_scales(alpha, setup_ring_dim, role_dims.d_a()),
            &projection_scales(alpha, setup_ring_dim, role_dims.d_b()),
            &projection_scales(alpha, setup_ring_dim, role_dims.d_d()),
        );
        assert_eq!(got, expected, "ownership widths {ownership_widths:?}");
    }
}
