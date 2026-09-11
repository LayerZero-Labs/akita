use super::*;
use crate::schedule::GroupCommitPhaseParams;
use crate::{dyadic_block_ranges, WitnessLayout};

#[test]
fn multi_group_m_row_count_matches_canonical_layout() {
    let (lp, _) = sample_multi_group_root_params();
    let n_a_final = lp.inner().matrix.output_rank();
    let n_b_final = lp.outer().matrix.output_rank();
    let n_a_pre = lp.precommitted_groups()[0]
        .profile
        .inner
        .matrix
        .output_rank();
    let n_b_pre = lp.precommitted_groups()[0]
        .profile
        .outer
        .matrix
        .output_rank();
    let n_d = lp.open().matrix.output_rank();

    assert_eq!(
        lp.relation_matrix_row_count(2).unwrap(),
        1 + n_a_final + n_b_final + 1 + n_a_pre + n_b_pre + n_d + 6
    );
}

#[test]
fn multi_group_row_offsets_match_a_before_b_layout() {
    let (lp, batch) = sample_multi_group_root_params();
    let n_a_final = lp.inner().matrix.output_rank();
    let n_b_final = lp.outer().matrix.output_rank();
    let n_a_pre = lp.precommitted_groups()[0]
        .profile
        .inner
        .matrix
        .output_rank();
    let n_b_pre = lp.precommitted_groups()[0]
        .profile
        .outer
        .matrix
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
    let precommitted = lp.preceding_group_mut_for_test(0).unwrap();
    let outer = &precommitted.profile.outer.matrix;
    precommitted.profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width(),
        outer.coeff_linf_bound(),
        64,
    );
    let dims = lp
        .group_role_dims(&batch, 0)
        .expect("precommitted group role dimensions");
    assert_eq!(
        dims,
        CommitmentRingDims {
            inner: 64,
            outer: 64,
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
    let precommitted = lp.preceding_group_mut_for_test(0).unwrap();
    precommitted
        .profile
        .outer
        .matrix
        .sis_table_key
        .ring_dimension /= 2;
    let err = precommitted
        .validate()
        .expect_err("frozen B dimension must match the serialized B matrix");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

fn precommit_admission_fixture() -> (
    GroupCommitPhaseParams,
    PrecommittedGroupAdmissionPolicy,
    SparseChallengeConfig,
    usize,
) {
    let challenge = SparseChallengeConfig::production_for_ring_dim(64).expect("D64 challenge");
    let policy = PrecommittedGroupAdmissionPolicy {
        decomposition: crate::DecompositionParams {
            log_basis: 3,
            log_commit_bound: 128,
            log_open_bound: Some(128),
        },
        sis_security_policy: crate::sis::DEFAULT_SIS_SECURITY_POLICY,
        sis_table_digest: crate::SisTableDigest::CURRENT,
        sis_modulus_profile: SisModulusProfileId::Q128OffsetA7F7,
        num_response_chunks: 1,
    };
    let num_digits_fold = 2;
    let a_bound = crate::sis::rounded_up_role_a_inf_norm(
        policy.sis_security_policy,
        policy.sis_table_digest,
        policy.sis_modulus_profile,
        64,
        3,
        &challenge,
        num_digits_fold,
        policy.num_response_chunks,
    )
    .expect("A admission bound");
    let b_bound = crate::sis::rounded_up_collision_inf_norm(
        policy.sis_security_policy,
        policy.sis_modulus_profile,
        crate::SisMatrixRole::Outer,
        64,
        3,
    )
    .expect("B admission bound");
    let inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: policy.sis_modulus_profile,
            role: crate::SisMatrixRole::Inner,
            ring_dimension: 64,
            coeff_linf_bound: a_bound,
        },
        32 * 16,
    )
    .expect("audited A matrix");
    let outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: policy.sis_security_policy,
            table_digest: policy.sis_table_digest,
            modulus_profile: policy.sis_modulus_profile,
            role: crate::SisMatrixRole::Outer,
            ring_dimension: 64,
            coeff_linf_bound: b_bound,
        },
        inner_commit_matrix.output_rank() * 43 * 8,
    )
    .expect("audited B matrix");
    let layout = GroupCommitPhaseParams {
        version: GroupCommitPhaseParams::VERSION,
        group: PolynomialGroupLayout::new(14, 1),
        blocks: crate::BlockGeometry::new(256, 32, 8),
        outer_slice_count: crate::CommitmentSliceCount::ONE,
        inner: crate::RoleParams::new(crate::GadgetDigits::new(8, 16), inner_commit_matrix),
        outer: crate::RoleParams::new(crate::GadgetDigits::new(3, 43), outer_commit_matrix),
    };
    (layout, policy, challenge, num_digits_fold)
}

#[test]
fn opening_d_segment_width_uses_the_method_physical_width() {
    let evaluation_trace =
        crate::opening_d_segment_width(crate::OpeningMethod::EvaluationTrace, 4, 512, 64, 3, 5, 2)
            .expect("full A-ring D segment");
    let coefficient_packing = crate::opening_d_segment_width(
        crate::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
        4,
        512,
        64,
        3,
        5,
        2,
    )
    .expect("reduced packing D segment");

    assert_eq!(evaluation_trace, 3 * 5 * 2 * (512 / 64));
    assert_eq!(coefficient_packing, 3 * 5 * 2 * (4 * 64 / 64));
    assert_eq!(evaluation_trace, 2 * coefficient_packing);
    assert!(crate::opening_d_segment_width(
        crate::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension: 64,
        },
        4,
        512,
        96,
        3,
        5,
        2,
    )
    .is_err());
}

#[test]
fn precommit_admission_rejects_policy_and_basis_mismatches() {
    let (layout, policy, challenge, num_digits_fold) = precommit_admission_fixture();
    GroupOpenPhaseParams::admit(
        layout,
        num_digits_fold,
        policy,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis,
    )
    .expect("valid precommit admission");

    let mismatched_modulus = PrecommittedGroupAdmissionPolicy {
        sis_modulus_profile: SisModulusProfileId::Q64Offset59,
        ..policy
    };
    let error = GroupOpenPhaseParams::admit(
        layout,
        num_digits_fold,
        mismatched_modulus,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis,
    )
    .expect_err("mismatched modulus must be rejected");
    assert!(error.to_string().contains("modulus profile does not match"));
    let error = GroupOpenPhaseParams::admit(
        layout,
        num_digits_fold,
        policy,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis - 1,
    )
    .expect_err("opening below frozen outer basis must be rejected");
    assert!(error.to_string().contains("must dominate"));

    let mut wrong_outer_depth = layout;
    wrong_outer_depth.outer.digits.num_digits += 1;
    let outer = wrong_outer_depth.outer.matrix;
    wrong_outer_depth.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        outer.sis_table_key(),
        outer.input_width()
            + wrong_outer_depth.inner.matrix.output_rank() * wrong_outer_depth.blocks.live_blocks,
    )
    .expect("canonical wrong-depth B matrix");
    let error = GroupOpenPhaseParams::admit(
        wrong_outer_depth,
        num_digits_fold,
        policy,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis,
    )
    .expect_err("wrong frozen outer digit depth must be rejected");
    assert!(error.to_string().contains("outer digit depth"), "{error}");
}

#[test]
fn precommit_admission_rejects_insufficient_a_and_b_bounds() {
    let (layout, policy, challenge, num_digits_fold) = precommit_admission_fixture();
    let mut low_a = layout;
    let inner = low_a.inner.matrix;
    let inner_key = inner.sis_table_key().expect("L infinity test matrix");
    let lower_a_bound =
        crate::sis::inner_coeff_linf_bounds(inner_key.modulus_profile, inner_key.ring_dimension)
            .into_iter()
            .rfind(|&bound| bound < inner.coeff_linf_bound().expect("L infinity test matrix"))
            .expect("lower supported A bound");
    low_a.inner.matrix = InnerCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            coeff_linf_bound: lower_a_bound,
            ..inner_key
        },
        inner.input_width(),
    )
    .expect("canonical low-bound A matrix");
    let error = GroupOpenPhaseParams::admit(
        low_a,
        num_digits_fold,
        policy,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis,
    )
    .expect_err("insufficient A bound must be rejected");
    assert!(error.to_string().contains("A bound"), "{error}");

    let mut low_b = layout;
    let outer = low_b.outer.matrix;
    let lower_b_bound = crate::sis::GADGET_COEFF_LINF_ANCHORS
        .iter()
        .copied()
        .rfind(|&bound| bound < outer.coeff_linf_bound())
        .expect("lower supported B bound");
    low_b.outer.matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            coeff_linf_bound: lower_b_bound,
            ..outer.sis_table_key()
        },
        outer.input_width(),
    )
    .expect("canonical low-bound B matrix");
    let error = GroupOpenPhaseParams::admit(
        low_b,
        num_digits_fold,
        policy,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis,
    )
    .expect_err("insufficient B bound must be rejected");
    assert!(error.to_string().contains("B bound"), "{error}");
}

#[test]
fn precommit_admission_prices_the_consuming_response_chunk_count() {
    let (layout, policy, challenge, num_digits_fold) = precommit_admission_fixture();
    let chunked_policy = PrecommittedGroupAdmissionPolicy {
        num_response_chunks: 2,
        ..policy
    };
    let error = GroupOpenPhaseParams::admit(
        layout,
        num_digits_fold,
        chunked_policy,
        OpeningMethod::EvaluationTrace,
        challenge,
        layout.outer.digits.log_basis,
    )
    .expect_err("an A row sized for one response must not admit two response chunks");
    assert!(error.to_string().contains("A bound"), "{error}");
}

#[test]
fn native_group_dimensions_are_independent_of_final_group_order() {
    use jolt_field::Prime128OffsetA7F7;

    let (mut lp, batch) = sample_multi_group_root_params();
    let precommitted = lp.preceding_group_mut_for_test(0).unwrap();
    precommitted.opening.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(128).expect("D128 challenge");
    let inner = &precommitted.profile.inner.matrix;
    let inner_bound = crate::sis::rounded_up_role_a_inf_norm(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        128,
        precommitted.opening.log_basis_open,
        &precommitted.opening.fold_challenge_config,
        precommitted.opening.num_digits_fold,
        1,
    )
    .expect("D128 exact A bound");
    precommitted.profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner_bound,
        128,
    );
    let outer = &precommitted.profile.outer.matrix;
    precommitted.profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * 2,
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );
    assert_eq!(lp.d_a(), 64, "the final group remains native at A=64");
    assert_eq!(
        lp.group_role_dims(&batch, 0)
            .expect("precommitted group dimensions")
            .d_a(),
        128
    );
    let relation_geometry =
        crate::RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &batch)
            .expect("relation geometry");
    let witness_layout = WitnessLayout::new(
        &lp,
        &batch,
        &relation_geometry,
        lp.witness_chunk.num_chunks,
        crate::RelationQuotientPlan::quotient_lift(crate::r_decomp_levels::<Prime128OffsetA7F7>(
            lp.open().digits.log_basis,
        ))
        .unwrap(),
    )
    .expect("witness layout");
    assert_eq!(
        lp.output_witness_len::<Prime128OffsetA7F7>(&batch, 1)
            .expect("output witness length"),
        witness_layout.live_coeff_len()
    );
    assert_eq!(
        lp.output_witness_len_for_field_bits(128, 1, &batch)
            .expect("policy-bound output witness length"),
        witness_layout.live_coeff_len()
    );
    assert!(
        witness_layout.live_coeff_len().is_multiple_of(128),
        "the grouped witness must include padding for the widest successor A carrier"
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
    let outer = &lp.outer().matrix;
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * (d_a / d_b),
        outer.coeff_linf_bound(),
        d_b,
    );
    let open = &lp.open().matrix;
    lp.open_matrix = OpenCommitMatrixParams::new_unchecked(
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
        3,
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
) -> GroupOpenPhaseParams {
    let mut lp = address_oracle_group_params(d_a, d_b, d_d, blocks);
    certify_test_sis_bounds(&mut lp);
    let outer = &lp.outer().matrix;
    lp.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        outer.input_width() * claims,
        outer.coeff_linf_bound(),
        d_b,
    );
    let layout = GroupCommitPhaseParams::from_params_unchecked_for_test(
        PolynomialGroupLayout::new(4, claims),
        &lp,
    );
    GroupOpenPhaseParams {
        setup_natural_len: None,
        profile: layout,
        opening: crate::GroupOpeningPlan::evaluation_trace(
            lp.fold_challenge_config(),
            lp.open().digits.log_basis,
            lp.open().digits.num_digits,
            lp.num_digits_fold(),
        ),
    }
}

fn address_oracle_fixture(group_count: usize) -> (CommittedGroupParams, OpeningClaimsLayout) {
    let (final_dims, precommitted) = match group_count {
        1 => ((64, 64, 64, 8, 2), Vec::new()),
        2 => ((64, 64, 64, 8, 2), vec![(128, 64, 64, 16, 1)]),
        3 => (
            (64, 64, 64, 8, 2),
            vec![(128, 64, 64, 16, 1), (64, 64, 64, 8, 3)],
        ),
        _ => panic!("address-oracle fixture supports one to three groups"),
    };
    let (d_a, d_b, d_d, blocks, final_claims) = final_dims;
    let mut lp = address_oracle_group_params(d_a, d_b, d_d, blocks);
    lp.set_precommitted_groups(
        precommitted
            .iter()
            .map(|&(a, b, d, blocks, claims)| address_oracle_precommit(a, b, d, blocks, claims))
            .collect(),
    )
    .unwrap();
    let precommitted_layouts = lp
        .precommitted_groups()
        .iter()
        .map(|group| group.profile.group)
        .collect::<Vec<_>>();
    let batch = OpeningClaimsLayout::from_root_groups(
        &precommitted_layouts,
        PolynomialGroupLayout::new(4, final_claims),
    )
    .expect("address-oracle opening layout");
    (lp, batch)
}

#[test]
fn relation_geometry_supports_mixed_root_opening_methods() {
    let (mut lp, batch) = address_oracle_fixture(2);
    lp.preceding_group_mut_for_test(0)
        .unwrap()
        .opening
        .opening_method = crate::OpeningMethod::SubringCoefficientPacking {
        challenge_subring_dimension: 64,
    };
    lp.preceding_group_mut_for_test(0)
        .unwrap()
        .opening
        .fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(64).expect("D64 challenge");

    let geometry =
        crate::RelationWitnessGeometry::for_level(&lp, &batch, 2).expect("mixed opening geometry");
    let final_group = batch.root_final_group_index().expect("final group");
    assert_eq!(
        geometry.group_opening_geometry(final_group).unwrap(),
        crate::RelationRowGeometry::native(64).unwrap()
    );
    let precommitted = geometry.group_opening_geometry(0).unwrap();
    assert_eq!(precommitted.polynomial_modulus_dimension(), 64);
    assert_eq!(precommitted.coordinate_plane_count(), 2);
    assert_eq!(precommitted.physical_coefficient_width(), 128);
    assert_eq!(geometry.relation_coefficient_block_len().unwrap(), 64);

    let layout = WitnessLayout::new(
        &lp,
        &batch,
        &geometry,
        2,
        crate::RelationQuotientPlan::quotient_lift(2).unwrap(),
    )
    .expect("mixed-method witness layout");
    assert_eq!(
        layout
            .unit(final_group, 0)
            .unwrap()
            .e_geometry()
            .coordinate_plane_count(),
        1
    );
    assert_eq!(layout.unit(0, 0).unwrap().e_geometry(), precommitted);
    assert!(crate::RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &batch).is_err());
    assert!(lp.validate_opening_batch(&batch).is_ok());
}

#[test]
fn relation_geometry_revalidates_frozen_precommitted_profiles() {
    let (mut lp, batch) = address_oracle_fixture(2);
    lp.preceding_group_mut_for_test(0)
        .unwrap()
        .profile
        .outer
        .matrix
        .sis_table_key
        .ring_dimension /= 2;
    assert!(crate::RelationWitnessGeometry::for_level(&lp, &batch, 2).is_err());
}

#[test]
fn compact_witness_addresses_match_independent_formula_matrix() {
    use jolt_field::Prime128OffsetA7F7;

    for group_count in [1usize, 2, 3] {
        let (base_lp, batch) = address_oracle_fixture(group_count);
        let group_order = batch.root_group_order().expect("authenticated group order");
        for num_chunks in [1usize, 2, 4, 8] {
            let mut lp = base_lp.clone();
            lp.witness_chunk = crate::witness::ChunkedWitnessCfg {
                num_chunks,
                num_activated_levels: usize::from(num_chunks > 1),
            };
            let quotient_depth =
                crate::r_decomp_levels::<Prime128OffsetA7F7>(lp.open().digits.log_basis);
            let relation_geometry =
                crate::RelationWitnessGeometry::for_evaluation_trace_execution(&lp, &batch)
                    .expect("relation geometry");
            let layout = WitnessLayout::new(
                &lp,
                &batch,
                &relation_geometry,
                num_chunks,
                crate::RelationQuotientPlan::quotient_lift(quotient_depth).unwrap(),
            )
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
                    let blocks = dyadic_block_ranges(params.num_live_blocks(), num_chunks)
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
            let relation_layout = relation_geometry.rhs_layout();
            let row_families = relation_layout
                .row_families()
                .expect("canonical row families");
            assert_eq!(layout.tail_range().start, cursor);
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
                lp.open().matrix.output_rank(),
            ));
            for map_index in 0..crate::COMPRESSION_MAP_COUNT {
                for relation_group_index in 0..group_order.len() {
                    expected_r_dims.push(
                        relation_layout
                            .group_compression_plan(relation_group_index)
                            .expect("F plan")
                            .1
                            .maps()[map_index]
                            .ring_dimension(),
                    );
                }
                expected_r_dims.push(
                    relation_layout
                        .opening_compression_plan()
                        .expect("H plan")
                        .maps()[map_index]
                        .ring_dimension(),
                );
            }
            assert_eq!(layout.r_rows().len(), expected_r_dims.len());
            let compression_row_count = crate::COMPRESSION_MAP_COUNT * (group_order.len() + 1);
            let ordinary_row_count = expected_r_dims.len() - compression_row_count;
            for (row_index, (&ring_dim, row)) in expected_r_dims
                .iter()
                .zip(layout.r_rows())
                .take(ordinary_row_count)
                .enumerate()
            {
                assert_eq!(row.geometry().physical_coefficient_width(), ring_dim);
                assert_eq!(row.range(), cursor..cursor + quotient_depth * ring_dim);
                for digit in 0..quotient_depth {
                    for coefficient in 0..ring_dim {
                        assert_eq!(
                            layout
                                .r_coefficient_index(row_index, digit, 0, coefficient)
                                .expect("R address"),
                            cursor + digit * ring_dim + coefficient
                        );
                    }
                }
                cursor += quotient_depth * ring_dim;
            }
            let support = layout.negative_binary_support_intervals();
            assert_eq!(support.len(), crate::COMPRESSION_MAP_COUNT);
            let prefix_alignment = cursor..support[0].start;
            if !prefix_alignment.is_empty() {
                assert!(layout
                    .compression_alignment_ranges()
                    .contains(&prefix_alignment));
            }
            assert_eq!(
                layout.compression_layers().len(),
                crate::COMPRESSION_MAP_COUNT
            );
            for (map_index, support_interval) in support.iter().enumerate() {
                let layer_alignment = cursor..support_interval.start;
                if !layer_alignment.is_empty() {
                    assert!(layout
                        .compression_alignment_ranges()
                        .contains(&layer_alignment));
                }
                cursor = support_interval.start;
                let layer = &layout.compression_layers()[map_index];
                assert_eq!(layer.map_index(), map_index);
                assert_eq!(layer.f_spans().len(), group_order.len());
                let canonical_f_quotient_rows = row_families
                    .iter()
                    .enumerate()
                    .filter_map(|(row_index, row)| match row {
                        crate::RelationRowFamily::CompressionF {
                            group_index,
                            map_index: row_map_index,
                            ..
                        } if *row_map_index == map_index => Some((*group_index, row_index)),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    layer.f_quotient_rows(),
                    Some(canonical_f_quotient_rows.as_slice())
                );
                for (relation_group_index, &group_index) in group_order.iter().enumerate() {
                    let (planned_group_index, plan) = relation_layout
                        .group_compression_plan(relation_group_index)
                        .expect("group compression plan");
                    assert_eq!(planned_group_index, group_index);
                    let (span_group_index, span) = &layer.f_spans()[relation_group_index];
                    assert_eq!(*span_group_index, group_index);
                    assert_eq!(span.map(), plan.maps()[map_index]);
                    assert_eq!(
                        span.range(),
                        cursor..cursor + span.map().padded_digit_count()
                    );
                    cursor += span.map().padded_digit_count();
                }
                let h_map = relation_layout
                    .opening_compression_plan()
                    .expect("H plan")
                    .maps()[map_index];
                assert_eq!(layer.h_span().map(), h_map);
                assert_eq!(
                    layer.h_span().range(),
                    cursor..cursor + h_map.padded_digit_count()
                );
                cursor += h_map.padded_digit_count();
                assert_eq!(support_interval.end, cursor);
                for &(group_index, row_index) in layer
                    .f_quotient_rows()
                    .expect("quotient-lift compression rows")
                {
                    assert!(group_order.contains(&group_index));
                    let row = &layout.r_rows()[row_index];
                    assert_eq!(
                        row.range(),
                        cursor
                            ..cursor + quotient_depth * row.geometry().physical_coefficient_width()
                    );
                    cursor = row.range().end;
                }
                let h_row = &layout.r_rows()[layer.h_quotient_row().expect("quotient-lift H row")];
                assert_eq!(
                    h_row.range(),
                    cursor..cursor + quotient_depth * h_row.geometry().physical_coefficient_width()
                );
                cursor = h_row.range().end;
            }
            let suffix_alignment = cursor..layout.live_coeff_len();
            if !suffix_alignment.is_empty() {
                assert!(layout
                    .compression_alignment_ranges()
                    .contains(&suffix_alignment));
            }
            cursor = layout.live_coeff_len();
            assert_eq!(layout.tail_range().end, cursor);
            assert_eq!(layout.live_coeff_len(), cursor);
            assert_eq!(
                lp.output_witness_len::<Prime128OffsetA7F7>(&batch, 1)
                    .expect("canonical witness length"),
                cursor
            );
        }
    }
}
