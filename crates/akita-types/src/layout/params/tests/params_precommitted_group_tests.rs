use super::*;
use crate::schedule::CommittedGroupProfile;
use crate::WitnessLayout;

#[test]
fn multi_group_m_row_count_matches_canonical_layout() {
    let (lp, _) = sample_multi_group_root_params();
    let n_a_final = lp.inner_commit_matrix.output_rank();
    let n_b_final = lp.outer_commit_matrix.output_rank();
    let n_a_pre = lp.precommitted_groups[0]
        .layout
        .inner_commit_matrix
        .output_rank();
    let n_b_pre = lp.precommitted_groups[0]
        .layout
        .outer_commit_matrix
        .output_rank();
    let n_d = lp.open_commit_matrix.output_rank();

    assert_eq!(
        lp.relation_matrix_row_count(2).unwrap(),
        1 + n_a_final + n_b_final + 1 + n_a_pre + n_b_pre + n_d
    );
}

#[test]
fn multi_group_row_offsets_match_a_before_b_layout() {
    let (lp, batch) = sample_multi_group_root_params();
    let n_a_final = lp.inner_commit_matrix.output_rank();
    let n_b_final = lp.outer_commit_matrix.output_rank();
    let n_a_pre = lp.precommitted_groups[0]
        .layout
        .inner_commit_matrix
        .output_rank();
    let n_b_pre = lp.precommitted_groups[0]
        .layout
        .outer_commit_matrix
        .output_rank();
    let final_group = batch.root_final_group_index().expect("final group");

    assert_eq!(
        lp.a_row_range(&batch, final_group).unwrap(),
        1..1 + n_a_final
    );
    assert_eq!(
        lp.commitment_row_range(&batch, final_group).unwrap(),
        1 + n_a_final..1 + n_a_final + n_b_final
    );
    assert_eq!(
        lp.a_row_range(&batch, 0).unwrap(),
        2 + n_a_final + n_b_final..2 + n_a_final + n_b_final + n_a_pre
    );
    assert_eq!(
        lp.commitment_row_range(&batch, 0).unwrap(),
        2 + n_a_final + n_b_final + n_a_pre..2 + n_a_final + n_b_final + n_a_pre + n_b_pre
    );
    assert_eq!(lp.consistency_row_index(&batch, final_group).unwrap(), 0);
    assert_eq!(
        lp.consistency_row_index(&batch, 0).unwrap(),
        1 + n_a_final + n_b_final
    );
}

#[test]
fn multi_group_root_accepts_multi_chunk_witness_layout() {
    let (mut lp, batch) = sample_multi_group_root_params();
    lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
        num_chunks: 2,
        num_activated_levels: 1,
    };
    lp.evaluation_trace_row_index(&batch)
        .expect("canonical product layout supports grouped chunks");
}

#[test]
fn group_role_dims_use_group_a_b_and_level_shared_d() {
    let (mut lp, batch) = sample_multi_group_root_params();
    let precommitted = &mut lp.precommitted_groups[0];
    let outer = &precommitted.layout.outer_commit_matrix;
    precommitted.layout.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * 2,
        outer.coeff_linf_bound(),
        32,
    );
    let dims = lp
        .group_role_dims(&batch, 0)
        .expect("precommitted group role dimensions");
    assert_eq!(
        dims,
        CommitmentRingDims {
            inner: 64,
            outer: 32,
            opening: 64,
        }
    );
    let final_group = batch.root_final_group_index().expect("final group");
    assert_eq!(
        lp.group_role_dims(&batch, final_group)
            .expect("final group role dimensions"),
        lp.role_dims()
    );
}

#[test]
fn precommitted_params_reject_frozen_matrix_dimension_mismatch() {
    let (mut lp, _) = sample_multi_group_root_params();
    let precommitted = &mut lp.precommitted_groups[0];
    precommitted
        .layout
        .outer_commit_matrix
        .sis_table_key
        .ring_dimension /= 2;
    let err = precommitted
        .validate()
        .expect_err("frozen B dimension must match the serialized B matrix");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn native_group_dimensions_are_independent_of_final_group_order() {
    use akita_field::Prime128OffsetA7F7;

    let (mut lp, batch) = sample_multi_group_root_params();
    let precommitted = &mut lp.precommitted_groups[0];
    let inner = &precommitted.layout.inner_commit_matrix;
    precommitted.layout.inner_commit_matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner.coeff_linf_bound(),
        128,
    );
    let outer = &precommitted.layout.outer_commit_matrix;
    precommitted.layout.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * 2,
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );
    precommitted.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(128).expect("D128 challenge");

    assert_eq!(lp.d_a(), 64, "the final group remains native at A=64");
    assert_eq!(
        lp.group_role_dims(&batch, 0)
            .expect("precommitted group dimensions")
            .d_a(),
        128
    );
    let witness_layout = WitnessLayout::new(
        &lp,
        &batch,
        lp.witness_chunk.num_chunks,
        crate::r_decomp_levels::<Prime128OffsetA7F7>(lp.log_basis_open),
    )
    .expect("witness layout");
    assert_eq!(
        lp.output_witness_len::<Prime128OffsetA7F7>(&batch)
            .expect("output witness length"),
        witness_layout.live_coeff_len()
    );
    assert!(witness_layout
        .units_for_group(0)
        .expect("precommitted units")
        .all(|unit| unit.z_range().len().is_multiple_of(128)));
}

fn configure_test_role_dims(lp: &mut CommittedGroupParams, d_b: usize, d_d: usize) {
    let d_a = lp.d_a();
    assert!(d_a.is_multiple_of(d_b));
    assert!(d_a.is_multiple_of(d_d));
    let outer = &lp.outer_commit_matrix;
    lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * (d_a / d_b),
        outer.coeff_linf_bound(),
        d_b,
    );
    let open = &lp.open_commit_matrix;
    lp.open_commit_matrix = OpenCommitMatrixParams::new_unchecked(
        open.security_policy(),
        open.sis_table_key().table_digest,
        open.sis_modulus_profile(),
        open.output_rank(),
        open.input_width() * (d_a / d_d),
        open.coeff_linf_bound(),
        d_d,
    );
}

fn address_oracle_group_params(
    d_a: usize,
    d_b: usize,
    d_d: usize,
    blocks: usize,
) -> CommittedGroupParams {
    let mut lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q128OffsetA7F7,
        d_a,
        2,
        3,
        2,
        3,
        SparseChallengeConfig::production_for_ring_dim(d_a).expect("test challenge"),
    )
    .with_decomp(4, blocks * 4, 2, 2, 2)
    .expect("address-oracle params");
    configure_test_role_dims(&mut lp, d_b, d_d);
    lp
}

fn address_oracle_precommit(
    d_a: usize,
    d_b: usize,
    d_d: usize,
    blocks: usize,
    claims: usize,
) -> PrecommittedLevelParams {
    let mut lp = address_oracle_group_params(d_a, d_b, d_d, blocks);
    certify_test_sis_bounds(&mut lp);
    let outer = &lp.outer_commit_matrix;
    lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * claims,
        outer.coeff_linf_bound(),
        d_b,
    );
    let layout = CommittedGroupProfile::from_params(PolynomialGroupLayout::new(4, claims), &lp);
    PrecommittedLevelParams {
        layout,
        log_basis_open: lp.log_basis_open,
        fold_challenge_config: lp.fold_challenge_config,
        num_digits_open: lp.num_digits_open,
        num_digits_fold: lp.num_digits_fold,
    }
}

fn address_oracle_fixture(group_count: usize) -> (CommittedGroupParams, OpeningClaimsLayout) {
    let (final_dims, precommitted) = match group_count {
        1 => ((64, 64, 64, 8, 2), Vec::new()),
        2 => ((64, 64, 32, 8, 2), vec![(128, 64, 32, 16, 1)]),
        3 => (
            (64, 32, 64, 8, 2),
            vec![(128, 32, 64, 16, 1), (64, 64, 64, 8, 3)],
        ),
        _ => panic!("address-oracle fixture supports one to three groups"),
    };
    let (d_a, d_b, d_d, blocks, final_claims) = final_dims;
    let mut lp = address_oracle_group_params(d_a, d_b, d_d, blocks);
    lp.precommitted_groups = precommitted
        .iter()
        .map(|&(a, b, d, blocks, claims)| address_oracle_precommit(a, b, d, blocks, claims))
        .collect();
    let precommitted_layouts = lp
        .precommitted_groups
        .iter()
        .map(|group| group.layout.group)
        .collect::<Vec<_>>();
    let batch = OpeningClaimsLayout::from_root_groups(
        &precommitted_layouts,
        PolynomialGroupLayout::new(4, final_claims),
    )
    .expect("address-oracle opening layout");
    (lp, batch)
}

#[test]
fn compact_witness_addresses_match_independent_formula_matrix() {
    use akita_field::Prime128OffsetA7F7;

    for group_count in [1usize, 2, 3] {
        let (base_lp, batch) = address_oracle_fixture(group_count);
        let group_order = batch.root_group_order().expect("authenticated group order");
        for num_chunks in [1usize, 2, 4, 8] {
            let mut lp = base_lp.clone();
            lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
                num_chunks,
                num_activated_levels: usize::from(num_chunks > 1),
            };
            let quotient_depth = crate::r_decomp_levels::<Prime128OffsetA7F7>(lp.log_basis_open);
            let layout = WitnessLayout::new(&lp, &batch, num_chunks, quotient_depth)
                .expect("compact witness layout");
            let mut cursor = 0usize;
            let mut unit_position = 0usize;
            for chunk in 0..num_chunks {
                for &group_index in &group_order {
                    let params = lp.group_params(&batch, group_index).expect("group params");
                    let dims = lp
                        .group_role_dims(&batch, group_index)
                        .expect("group dimensions");
                    let claims = batch
                        .group_layout(group_index)
                        .expect("group layout")
                        .num_polynomials();
                    let blocks = WitnessLayout::resolve_chunk_block_ranges(
                        params.num_live_blocks(),
                        num_chunks,
                    )
                    .expect("chunk ranges")[chunk]
                        .clone();
                    let unit = &layout.units()[unit_position];
                    unit_position += 1;
                    assert_eq!(
                        (unit.group_index(), unit.chunk_index()),
                        (group_index, chunk)
                    );
                    assert_eq!(unit.global_block_range(), blocks.clone());

                    let d_a = dims.d_a();
                    let d_b = dims.d_b();
                    let d_d = dims.d_d();
                    let q_b = d_a / d_b;
                    let q_d = d_a / d_d;
                    let delta_z = params.num_digits_inner();
                    let delta_f = params.num_digits_fold();
                    let delta_d = params.num_digits_open();
                    let delta_b = params.num_digits_outer();
                    let n_a = params.a_rows_len();
                    let z_len = params.num_positions_per_block() * delta_z * delta_f * d_a;
                    let e_len = claims * blocks.len() * delta_d * d_a;
                    let t_len = claims * blocks.len() * n_a * delta_b * d_a;
                    assert_eq!(unit.z_range(), cursor..cursor + z_len);
                    cursor += z_len;
                    assert_eq!(unit.e_range(), cursor..cursor + e_len);
                    cursor += e_len;
                    assert_eq!(unit.t_range(), cursor..cursor + t_len);
                    cursor += t_len;

                    let z_base = unit.z_range().start;
                    for position in 0..params.num_positions_per_block() {
                        for witness_digit in 0..delta_z {
                            for fold_digit in 0..delta_f {
                                for coefficient in 0..d_a {
                                    let expected = z_base
                                        + (((position * delta_z + witness_digit) * delta_f
                                            + fold_digit)
                                            * d_a
                                            + coefficient);
                                    assert_eq!(
                                        unit.z_coefficient_index(
                                            d_a,
                                            params.num_positions_per_block(),
                                            delta_z,
                                            delta_f,
                                            position,
                                            witness_digit,
                                            fold_digit,
                                            coefficient,
                                        )
                                        .expect("Z address"),
                                        expected
                                    );
                                }
                            }
                        }
                    }
                    let e_base = unit.e_range().start;
                    let t_base = unit.t_range().start;
                    for claim in 0..claims {
                        for global_block in blocks.clone() {
                            let local_block = global_block - blocks.start;
                            for subcolumn in 0..q_d {
                                for digit in 0..delta_d {
                                    for coefficient in 0..d_d {
                                        let expected = e_base
                                            + ((((claim * blocks.len() + local_block) * q_d
                                                + subcolumn)
                                                * delta_d
                                                + digit)
                                                * d_d
                                                + coefficient);
                                        assert_eq!(
                                            unit.e_coefficient_index(
                                                d_a,
                                                d_d,
                                                claims,
                                                delta_d,
                                                claim,
                                                global_block,
                                                subcolumn,
                                                digit,
                                                coefficient,
                                            )
                                            .expect("E address"),
                                            expected
                                        );
                                    }
                                }
                            }
                            for a_row in 0..n_a {
                                for subcolumn in 0..q_b {
                                    for digit in 0..delta_b {
                                        for coefficient in 0..d_b {
                                            let expected = t_base
                                                + (((((claim * blocks.len() + local_block)
                                                    * n_a
                                                    + a_row)
                                                    * q_b
                                                    + subcolumn)
                                                    * delta_b
                                                    + digit)
                                                    * d_b
                                                    + coefficient);
                                            assert_eq!(
                                                unit.t_coefficient_index(
                                                    d_a,
                                                    d_b,
                                                    claims,
                                                    n_a,
                                                    delta_b,
                                                    claim,
                                                    global_block,
                                                    a_row,
                                                    subcolumn,
                                                    digit,
                                                    coefficient,
                                                )
                                                .expect("T address"),
                                                expected
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(layout.r_range().start, cursor);
            let mut expected_r_dims = Vec::new();
            for &group_index in &group_order {
                let params = lp.group_params(&batch, group_index).expect("group params");
                let dims = lp
                    .group_role_dims(&batch, group_index)
                    .expect("group dimensions");
                expected_r_dims.extend(std::iter::repeat_n(dims.d_a(), 1 + params.a_rows_len()));
                expected_r_dims.extend(std::iter::repeat_n(dims.d_b(), params.b_rows_len()));
            }
            expected_r_dims.extend(std::iter::repeat_n(
                lp.role_dims().d_d(),
                lp.open_commit_matrix.output_rank(),
            ));
            assert_eq!(layout.r_rows().len(), expected_r_dims.len());
            for (row_index, (&ring_dim, row)) in
                expected_r_dims.iter().zip(layout.r_rows()).enumerate()
            {
                assert_eq!(row.ring_dim(), ring_dim);
                assert_eq!(row.range(), cursor..cursor + quotient_depth * ring_dim);
                for digit in 0..quotient_depth {
                    for coefficient in 0..ring_dim {
                        assert_eq!(
                            layout
                                .r_coefficient_index(row_index, digit, coefficient)
                                .expect("R address"),
                            cursor + digit * ring_dim + coefficient
                        );
                    }
                }
                cursor += quotient_depth * ring_dim;
            }
            assert_eq!(layout.live_coeff_len(), cursor);
            assert_eq!(
                lp.output_witness_len::<Prime128OffsetA7F7>(&batch)
                    .expect("canonical witness length"),
                cursor
            );
        }
    }
}
