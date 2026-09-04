use super::*;
use jolt_field::Ring;

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
    physical_b_override: Option<&[F]>,
    a_ratio: usize,
    b_ratio: usize,
    d_ratio: usize,
    a_scales: &[F],
    b_scales: &[F],
    d_scales: &[F],
) -> F {
    let materialized_b = plan
        .groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            let direct = plan.direct_scan_state.weights(group_index).unwrap();
            group.physical_b.contract_logical_column_weights(&direct.t)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut acc = F::zero();
    for base_idx in 0..required {
        let mut weight = F::zero();
        for (group_index, group) in plan.groups.iter().enumerate() {
            let (e_eq_slice, _t_eq_slice, z_eq_slice) = plan
                .direct_scan_state
                .weights(group_index)
                .unwrap()
                .slices();
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
            if b_idx < group.physical_b.physical_footprint().unwrap() {
                let physical_b = physical_b_override.unwrap_or(&materialized_b[group_index]);
                weight += b_scales[base_idx % b_ratio] * physical_b[b_idx];
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

fn naive_sliced_physical_b_weights(
    group: &SetupContributionGroupPlan<F>,
    logical_t: &[F],
) -> Vec<F> {
    let slice_count = group.physical_b.geometry().slice_count().get();
    let physical_rows = group.physical_b.physical_rows();
    let physical_cols = group.physical_b.physical_input_width();
    let max_blocks_per_slice = group.num_live_blocks.div_ceil(slice_count);
    let per_block = physical_cols / (group.num_claims * max_blocks_per_slice);
    let mut expected = vec![F::zero(); physical_rows * physical_cols];
    for slice_index in 0..slice_count {
        let slice_start = slice_index * group.num_live_blocks / slice_count;
        let slice_end = (slice_index + 1) * group.num_live_blocks / slice_count;
        for row in 0..physical_rows {
            // Logical B rows are slice-major by specification. Keep this
            // oracle independent of production row-coordinate helpers.
            let row_weight =
                group.physical_b.logical_row_weights()[slice_index * physical_rows + row];
            for claim in 0..group.num_claims {
                for block in slice_start..slice_end {
                    let local_block = block - slice_start;
                    for offset in 0..per_block {
                        let physical_col =
                            (claim * max_blocks_per_slice + local_block) * per_block + offset;
                        let logical_col =
                            (claim * group.num_live_blocks + block) * per_block + offset;
                        expected[row * physical_cols + physical_col] +=
                            row_weight * logical_t[logical_col];
                    }
                }
            }
        }
    }
    expected
}

pub(super) fn structured_slice_reference(
    group: &SetupContributionGroupPlan<F>,
    direct: &DirectScanWeights<F>,
    block_challenges: &[F],
    opening_a_evals: &[F],
    alpha: F,
) -> F {
    let (e_eq_slice, t_eq_slice, z_eq_slice) = direct.slices();
    let (outer_subcolumns, _) =
        SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims).unwrap();
    let opening_subcolumns = group.opening_subcolumns;
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

#[allow(clippy::too_many_arguments)]
fn reduced_structured_slice_reference(
    group: &SetupContributionGroupPlan<F>,
    layout: &WitnessLayout,
    fold_gadget: &[F],
    block_challenges: &Challenges,
    opening_base_weights: &[F],
    coefficient_point: &[F],
    relation_point: &[F],
    alpha: F,
) -> F {
    let mut full_point = coefficient_point.to_vec();
    full_point.extend_from_slice(relation_point);
    let (outer_subcolumns, _) =
        SetupProjectionGeometry::native_role_subcolumn_counts(group.role_dims).unwrap();
    let opening_gadget = gadget_row_scalars::<F>(group.depth_open, group.log_basis_open);
    let commitment_gadget = gadget_row_scalars::<F>(group.depth_commit, group.log_basis_outer);
    let witness_gadget = gadget_row_scalars::<F>(group.depth_witness, group.log_basis_inner);
    let alpha_powers = scalar_powers(alpha, group.role_dims.d_a());
    let mut evaluation = F::zero();
    for claim in 0..group.num_claims {
        for global_block in 0..group.num_live_blocks {
            let block_claim = claim * group.num_live_blocks + global_block;
            let block_challenge = &block_challenges.as_slice()[block_claim];
            let unit = layout.unit_for_block(group.group_id, global_block).unwrap();
            for subcolumn in 0..group.opening_subcolumns {
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    let physical_start = unit
                        .e_coefficient_index(
                            group.role_dims.d_d(),
                            group.num_claims,
                            group.depth_open,
                            claim,
                            global_block,
                            subcolumn,
                            digit,
                            0,
                        )
                        .unwrap();
                    for coefficient in 0..group.role_dims.d_d() {
                        let ambient_coefficient = subcolumn * group.role_dims.d_d() + coefficient;
                        let multiplier_weight = block_challenge
                            .positions
                            .iter()
                            .zip(&block_challenge.coeffs)
                            .fold(F::zero(), |sum, (&position, &value)| {
                                let exponent = position as usize + ambient_coefficient;
                                let term = F::from_i64(i64::from(value))
                                    * alpha_powers[exponent % group.role_dims.d_a()];
                                if exponent < group.role_dims.d_a() {
                                    sum + term
                                } else {
                                    sum - term
                                }
                            });
                        evaluation += group.consistency_weight
                            * gadget
                            * multiplier_weight
                            * eq_eval_at_index(&full_point, physical_start + coefficient);
                    }
                }
            }
            for row in 0..group.n_a {
                for subcolumn in 0..outer_subcolumns {
                    for (digit, &gadget) in commitment_gadget.iter().enumerate() {
                        let physical_start = unit
                            .t_coefficient_index(
                                group.role_dims.d_a(),
                                group.role_dims.d_b(),
                                group.num_claims,
                                group.n_a,
                                group.depth_commit,
                                claim,
                                global_block,
                                row,
                                subcolumn,
                                digit,
                                0,
                            )
                            .unwrap();
                        for coefficient in 0..group.role_dims.d_b() {
                            let ambient_coefficient =
                                subcolumn * group.role_dims.d_b() + coefficient;
                            let multiplier_weight = block_challenge
                                .positions
                                .iter()
                                .zip(&block_challenge.coeffs)
                                .fold(F::zero(), |sum, (&position, &value)| {
                                    let exponent = position as usize + ambient_coefficient;
                                    let term = F::from_i64(i64::from(value))
                                        * alpha_powers[exponent % group.role_dims.d_a()];
                                    if exponent < group.role_dims.d_a() {
                                        sum + term
                                    } else {
                                        sum - term
                                    }
                                });
                            evaluation += group.a_row_weights[row]
                                * gadget
                                * multiplier_weight
                                * eq_eval_at_index(&full_point, physical_start + coefficient);
                        }
                    }
                }
            }
        }
    }
    for (position, &opening) in opening_base_weights.iter().enumerate() {
        for (commit_digit, &gadget) in witness_gadget.iter().enumerate() {
            for unit in layout.units_for_group(group.group_id).unwrap() {
                for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                    let physical_start = unit
                        .z_coefficient_index(
                            group.role_dims.d_a(),
                            group.num_positions_per_block,
                            group.depth_witness,
                            fold_gadget.len(),
                            position,
                            commit_digit,
                            fold_digit,
                            0,
                        )
                        .unwrap();
                    let scalar = -(group.consistency_weight * opening * gadget * fold);
                    for (coefficient, &power) in
                        alpha_powers.iter().enumerate().take(group.role_dims.d_a())
                    {
                        evaluation += scalar
                            * eq_eval_at_index(&full_point, physical_start + coefficient)
                            * power;
                    }
                }
            }
        }
    }
    evaluation
}

#[test]
fn reduced_structured_terms_use_complete_native_terminal_functionals() {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let outgoing_ring_dim = 32;
    let (inputs, groups, layout, _, _, relation_point, fold_gadget) =
        structured_weight_fixture_with_outgoing(8, &[3, 5], role_dims, outgoing_ring_dim);
    let geometry =
        crate::RelationAddressGeometry::new(role_dims, outgoing_ring_dim, layout.live_coeff_len())
            .unwrap();
    let coefficient_point = (0..geometry.relation_coefficient_variable_count())
        .map(|index| test_scalar(701 + index as u128))
        .collect::<Vec<_>>();
    let alpha = test_scalar(17);
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1,
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
    let group_id = plan.groups[0].group_id;
    let block_claim_count = plan.groups[0].num_claims * plan.groups[0].num_live_blocks;
    let sparse_challenges = (0..block_claim_count)
        .map(|index| SparseChallenge {
            positions: vec![(127 - index % 5) as u32].into(),
            coeffs: vec![if index % 2 == 0 { 1 } else { -1 }].into(),
        })
        .collect::<Vec<_>>();
    let block_challenges = Challenges::from_sparse(
        sparse_challenges,
        plan.groups[0].num_live_blocks,
        plan.groups[0].num_claims,
    )
    .unwrap();
    let opening_base_weights = (0..plan.groups[0].num_positions_per_block)
        .map(|index| test_scalar(901 + index as u128))
        .collect::<Vec<_>>();
    let assert_literal = |plan: &SetupContributionPlan<F>, blocks: &Challenges, openings: &[F]| {
        let opening = crate::RingMultiplierOpeningPoint::from_base(&crate::RingOpeningPoint {
            position_weights: openings.to_vec(),
            live_block_weights: vec![F::zero(); plan.groups[0].num_live_blocks],
        })
        .prepare_functional_multiplier();
        let expected = reduced_structured_slice_reference(
            &plan.groups[0],
            &layout,
            &fold_gadget,
            blocks,
            openings,
            &coefficient_point,
            &relation_point,
            alpha,
        );
        assert_ne!(expected, F::zero());
        assert_eq!(
            plan.evaluate_reduced_structured_group::<F>(group_id, blocks, &opening)
                .unwrap(),
            expected
        );
    };

    let zero_challenges = Challenges::from_sparse(
        (0..block_claim_count)
            .map(|_| SparseChallenge {
                positions: Vec::new().into(),
                coeffs: Vec::new().into(),
            })
            .collect(),
        plan.groups[0].num_live_blocks,
        plan.groups[0].num_claims,
    )
    .unwrap();

    let original_a_weights = plan.groups[0].a_row_weights.to_vec();
    let original_consistency = plan.groups[0].consistency_weight;
    std::sync::Arc::make_mut(&mut plan.groups[0].a_row_weights).fill(F::zero());
    assert_literal(
        &plan,
        &block_challenges,
        &vec![F::zero(); opening_base_weights.len()],
    );

    std::sync::Arc::make_mut(&mut plan.groups[0].a_row_weights)
        .copy_from_slice(&original_a_weights);
    plan.groups[0].consistency_weight = F::zero();
    assert_literal(
        &plan,
        &block_challenges,
        &vec![F::zero(); opening_base_weights.len()],
    );

    plan.groups[0].consistency_weight = original_consistency;
    assert_literal(&plan, &zero_challenges, &opening_base_weights);
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
        let direct = full.direct_scan_state.weights(0).unwrap();
        let reference =
            structured_slice_reference(group, direct, &block_challenges, &opening_a_evals, alpha);
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
fn setup_index_mle_bridges_smaller_relation_blocks_to_native_setup_blocks() {
    let role_dims = CommitmentRingDims::uniform(128);
    let (inputs, groups, layout, _, _, _, fold_gadget) =
        structured_weight_fixture(8, &[3, 5], role_dims);
    let relation_geometry = crate::RelationAddressGeometry::new_with_coefficient_block(
        role_dims,
        64,
        128,
        layout.live_coeff_len(),
    )
    .unwrap();
    let relation_point = (0..relation_geometry.relation_lane_variable_count())
        .map(|index| test_scalar(101 + index as u128))
        .collect::<Vec<_>>();
    let mut plan = SetupContributionPlan::prepare::<F>(
        &inputs.level_params,
        &inputs.opening_batch,
        1,
        inputs.eq_tau1.clone(),
        &layout,
        &groups,
        PreparedRelationAddress::new(&relation_point).unwrap(),
        Some(&fold_gadget),
        relation_geometry,
    )
    .unwrap();
    let alpha = test_scalar(3);
    plan.materialize_direct_scan(PreparedCoefficientFunctional::lifted_power(alpha))
        .unwrap();

    assert_eq!(
        plan.relation_address_geometry()
            .relation_coefficient_block_len(),
        64
    );
    assert_eq!(plan.projection_geometry().base_ring_dim(), 128);
    let rho = rho_for_required(plan.required());
    assert_span_mle_matches_dense(&plan, &rho, alpha);
}

#[test]
fn sliced_b_setup_weights_contract_logical_rows_onto_one_physical_matrix() {
    let role_dims = CommitmentRingDims {
        inner: 128,
        outer: 64,
        opening: 64,
    };
    let setup_ring_dim = 64;
    for slice_count in [2, 4, 8].map(|count| crate::CommitmentSliceCount::try_new(count).unwrap()) {
        let (_, _, _, plan, _, _, _) = structured_weight_fixture_with_slices(
            11,
            &[3, 5, 3],
            role_dims,
            setup_ring_dim,
            slice_count,
        );
        let group = &plan.groups[0];
        let direct = plan.direct_scan_state.weights(0).unwrap();
        let expected = naive_sliced_physical_b_weights(group, &direct.t);
        assert_eq!(
            group
                .physical_b
                .contract_logical_column_weights(&direct.t)
                .unwrap(),
            expected
        );

        let setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
            AkitaSetupDescriptor {
                max_num_vars: 0,
                max_num_batched_polys: 0,
                num_field_elements: plan.required() * role_dims.d_a(),
                setup_seed: [0u8; 32].into(),
            },
            FlatMatrix::from_flat_data(
                (0..plan.required() * role_dims.d_a())
                    .map(|index| test_scalar(1_201 + index as u128))
                    .collect(),
            ),
        );
        let alpha = test_scalar(3);
        let alpha_pows_a = scalar_powers(alpha, role_dims.d_a());
        let alpha_pows_b = scalar_powers(alpha, role_dims.d_b());
        let alpha_pows_d = scalar_powers(alpha, role_dims.d_d());
        assert_eq!(
            plan.evaluate_direct::<F>(&setup).unwrap(),
            plan.evaluate_direct_by_rows::<F>(
                &setup,
                &alpha_pows_a,
                &alpha_pows_b,
                &alpha_pows_d,
                role_dims.d_a(),
            )
            .unwrap(),
            "factorized direct B scan mismatch for S={}",
            slice_count.get(),
        );

        let rho = rho_for_required(plan.required());
        assert_eq!(
            plan.evaluate_setup_index_weight_mle(&rho, alpha).unwrap(),
            projected_setup_weight_reference(
                &plan,
                &rho,
                plan.required(),
                Some(&expected),
                role_dims.d_a() / setup_ring_dim,
                role_dims.d_b() / setup_ring_dim,
                role_dims.d_d() / setup_ring_dim,
                &projection_scales(alpha, setup_ring_dim, role_dims.d_a()),
                &projection_scales(alpha, setup_ring_dim, role_dims.d_b()),
                &projection_scales(alpha, setup_ring_dim, role_dims.d_d()),
            ),
            "structured sliced B tensor mismatch for S={}",
            slice_count.get(),
        );
    }
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
            None,
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
