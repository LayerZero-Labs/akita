use super::*;
use crate::{
    CommittedGroupParams, CommittedGroupProfile, OpeningClaimsLayout, OuterCommitMatrixParams,
    PolynomialGroupLayout, PrecommittedLevelParams, SisModulusProfileId,
};
use akita_challenges::SparseChallengeConfig;

fn sample_level_params() -> CommittedGroupParams {
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 3, 2, 2, 2)
    .expect("sample level params")
}

fn prefix_eligible_level_params() -> CommittedGroupParams {
    let field_element_digits = crate::sis::compute_num_digits_field_width(
        SisModulusProfileId::Q32Offset99.field_bits(),
        3,
    );
    CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        2,
        2,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 3, field_element_digits, 2, 2)
    .expect("prefix eligible level params")
}

#[test]
fn active_setup_field_len_matches_packed_role_maximum() {
    let lp = sample_level_params();
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("opening batch");
    let w_a = lp.num_positions_per_block * lp.num_digits_inner;
    let w_b = lp.outer_commit_matrix.input_width();
    let w_d = opening_batch.num_total_polynomials() * lp.num_live_blocks * lp.num_digits_open;
    let expected_ring_slots = lp
        .inner_commit_matrix
        .output_rank()
        .checked_mul(w_a)
        .unwrap()
        .max(
            lp.outer_commit_matrix
                .output_rank()
                .checked_mul(w_b)
                .unwrap(),
        )
        .max(
            lp.open_commit_matrix
                .output_rank()
                .checked_mul(w_d)
                .unwrap(),
        );
    let geometry =
        active_setup_projection_geometry(&lp, &opening_batch).expect("projection geometry");
    assert_eq!(geometry.required(), expected_ring_slots);
    let dims = lp.role_dims();
    let base_d = dims.d_a().min(dims.d_b()).min(dims.d_d());
    assert_eq!(
        active_setup_field_len(&lp, &opening_batch).expect("field len"),
        expected_ring_slots * base_d
    );
}

#[test]
fn active_setup_field_len_prices_one_physical_sliced_b_matrix() {
    let mut lp = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(4, 32, 2, 2, 2)
    .expect("sliced level params");
    let opening_batch = OpeningClaimsLayout::new(7, 3).expect("opening batch");
    lp.outer_slice_count = crate::CommitmentSliceCount::FOUR;
    let slice_geometry = crate::CommitmentSliceGeometry::try_new(
        lp.outer_slice_count,
        lp.num_live_blocks,
        opening_batch.num_total_polynomials(),
        lp.inner_commit_matrix.output_rank(),
        lp.num_digits_outer,
        lp.inner_commit_matrix.ring_dimension(),
        lp.outer_commit_matrix.ring_dimension(),
    )
    .expect("slice geometry");
    let outer = lp.outer_commit_matrix;
    lp.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        slice_geometry.physical_input_width(),
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );

    let geometry =
        active_setup_projection_geometry(&lp, &opening_batch).expect("projection geometry");
    let base_d = geometry.base_ring_dim();
    let expected_b_projection = lp.outer_commit_matrix.output_rank()
        * slice_geometry.physical_input_width()
        * (lp.outer_commit_matrix.ring_dimension() / base_d);
    assert_eq!(geometry.b_projection_width(), expected_b_projection);

    let logical_unsliced_projection = lp
        .outer_slice_count
        .logical_output_rows(lp.outer_commit_matrix.output_rank())
        .expect("logical B rows")
        * opening_batch.num_total_polynomials()
        * lp.inner_commit_matrix.output_rank()
        * lp.num_live_blocks
        * lp.num_digits_outer
        * (lp.outer_commit_matrix.ring_dimension() / base_d);
    assert!(geometry.b_projection_width() < logical_unsliced_projection);
}

#[test]
fn setup_prefix_slot_identity_binds_outer_slice_count() {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        3,
        3,
        2,
        SparseChallengeConfig::pm1_only(3),
    )
    .with_decomp(2, 16, 2, 2, 2)
    .expect("prefix commitment params");
    let unsliced = setup_prefix_precommitted_params(&params, 1024).expect("unsliced prefix");

    let mut sliced_params = params;
    sliced_params.outer_slice_count = crate::CommitmentSliceCount::TWO;
    let sliced = setup_prefix_precommitted_params(&sliced_params, 1024).expect("sliced prefix");

    let unsliced_id = setup_prefix_slot_id(777, unsliced);
    let sliced_id = setup_prefix_slot_id(777, sliced);
    assert_ne!(unsliced_id, sliced_id);
    let mut unsliced_bytes = Vec::new();
    unsliced_id.append_descriptor_bytes(&mut unsliced_bytes);
    let mut sliced_bytes = Vec::new();
    sliced_id.append_descriptor_bytes(&mut sliced_bytes);
    assert_ne!(unsliced_bytes, sliced_bytes);
}

#[test]
fn active_setup_field_len_includes_mixed_role_subcolumns() {
    let mut lp = sample_level_params();
    let inner = &lp.inner_commit_matrix;
    lp.inner_commit_matrix = crate::InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner
            .sis_table_key()
            .expect("L infinity test matrix")
            .table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner
            .coeff_linf_bound()
            .expect("L infinity test matrix")
            .max(1),
        128,
    );
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("opening batch");
    let a_slots =
        lp.inner_commit_matrix.output_rank() * lp.num_positions_per_block * lp.num_digits_inner * 2;
    let b_slots = lp.outer_commit_matrix.output_rank() * lp.outer_commit_matrix.input_width() * 2;
    let d_slots = lp.open_commit_matrix.output_rank()
        * opening_batch.num_total_polynomials()
        * lp.num_live_blocks
        * lp.num_digits_open
        * 2;
    let expected_field_len = a_slots.max(b_slots).max(d_slots) * 64;

    assert_eq!(
        active_setup_field_len(&lp, &opening_batch).expect("mixed-D field len"),
        expected_field_len
    );
}

fn retarget_group_role_dims(
    params: &mut CommittedGroupParams,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
) {
    params.fold_challenge_config =
        SparseChallengeConfig::production_for_ring_dim(inner_ring_dimension)
            .expect("production challenge");
    let inner = params.inner_commit_matrix;
    params.inner_commit_matrix = crate::InnerCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: inner.security_policy(),
            table_digest: inner
                .sis_table_key()
                .expect("L infinity test matrix")
                .table_digest,
            modulus_profile: inner.sis_modulus_profile(),
            role: crate::sis::SisMatrixRole::Inner,
            ring_dimension: u32::try_from(inner_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 131_071,
        },
        inner.input_width(),
    )
    .expect("audited retargeted A matrix");
    let outer = params.outer_commit_matrix;
    params.outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: outer.security_policy(),
            table_digest: outer.sis_table_key().table_digest,
            modulus_profile: outer.sis_modulus_profile(),
            role: crate::sis::SisMatrixRole::Outer,
            ring_dimension: u32::try_from(outer_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 3,
        },
        outer.input_width() * (inner_ring_dimension / outer_ring_dimension),
    )
    .expect("audited retargeted B matrix");
}

fn retarget_group_role_dims_wide(
    params: &mut CommittedGroupParams,
    inner_ring_dimension: usize,
    outer_ring_dimension: usize,
    min_input_width: usize,
) {
    retarget_group_role_dims(params, inner_ring_dimension, outer_ring_dimension);
    let inner = params.inner_commit_matrix;
    params.inner_commit_matrix = crate::InnerCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: inner.security_policy(),
            table_digest: inner
                .sis_table_key()
                .expect("L infinity test matrix")
                .table_digest,
            modulus_profile: inner.sis_modulus_profile(),
            role: crate::sis::SisMatrixRole::Inner,
            ring_dimension: u32::try_from(inner_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 131_071,
        },
        inner.input_width().max(min_input_width),
    )
    .expect("wide audited A matrix");
    let outer = params.outer_commit_matrix;
    params.outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        crate::SisTableKey {
            policy: outer.security_policy(),
            table_digest: outer.sis_table_key().table_digest,
            modulus_profile: outer.sis_modulus_profile(),
            role: crate::sis::SisMatrixRole::Outer,
            ring_dimension: u32::try_from(outer_ring_dimension).expect("test ring dimension"),
            coeff_linf_bound: 3,
        },
        outer.input_width().max(min_input_width),
    )
    .expect("wide audited B matrix");
}

fn precommitted_group(
    params: &CommittedGroupParams,
    group: PolynomialGroupLayout,
) -> PrecommittedLevelParams {
    PrecommittedLevelParams {
        layout: CommittedGroupProfile::from_params(group, params),
        log_basis_open: params.log_basis_open,
        fold_challenge_config: params.fold_challenge_config,
        num_digits_open: params.num_digits_open,
        num_digits_fold: params.num_digits_fold,
    }
}

fn verifier_slot_for_id<F: FieldCore>(id: SetupPrefixSlotId) -> SetupPrefixVerifierSlot<F> {
    let payload_coefficients = setup_prefix_compression_plan(&id.commitment_params)
        .expect("setup-prefix compression plan")
        .terminal_coefficients();
    SetupPrefixVerifierSlot {
        id,
        commitment: SetupPrefixPublicCommitment {
            rows: vec![RingVec::from_coeffs(vec![F::zero(); payload_coefficients])],
        },
    }
}

#[test]
fn active_setup_field_len_projects_each_group_at_its_native_dimensions() {
    let mut final_params = sample_level_params();
    retarget_group_role_dims(&mut final_params, 128, 64);

    let mut precommitted_params = sample_level_params();
    retarget_group_role_dims(&mut precommitted_params, 256, 128);
    let precommitted_layout = PolynomialGroupLayout::new(5, 1);
    final_params.precommitted_groups = vec![precommitted_group(
        &precommitted_params,
        precommitted_layout,
    )];
    let opening_batch = OpeningClaimsLayout::from_root_groups(
        &[precommitted_layout],
        PolynomialGroupLayout::new(5, 3),
    )
    .expect("heterogeneous opening batch");

    let base_ring_dimension = 64usize;
    let mut expected_a_projection = 0usize;
    let mut expected_b_projection = 0usize;
    let mut expected_d_physical_cols = 0usize;
    for group_index in 0..opening_batch.num_groups() {
        let group_layout = opening_batch
            .group_layout(group_index)
            .expect("group layout");
        let group_params = final_params
            .group_params(&opening_batch, group_index)
            .expect("group params");
        let dims = final_params
            .group_role_dims(&opening_batch, group_index)
            .expect("group role dimensions");
        let a_cols = group_params.num_positions_per_block() * group_params.num_digits_inner();
        let b_cols = group_params.b_col_len();
        let d_cols = group_layout.num_polynomials()
            * group_params.num_live_blocks()
            * group_params.num_digits_open()
            * (dims.d_a() / dims.d_d());
        expected_a_projection = expected_a_projection
            .max(group_params.a_rows_len() * a_cols * (dims.d_a() / base_ring_dimension));
        expected_b_projection = expected_b_projection
            .max(group_params.b_rows_len() * b_cols * (dims.d_b() / base_ring_dimension));
        expected_d_physical_cols += d_cols;
    }
    let expected_d_projection = final_params.open_commit_matrix.output_rank()
        * expected_d_physical_cols
        * (final_params.role_dims().d_d() / base_ring_dimension);
    let expected_ring_slots = expected_a_projection
        .max(expected_b_projection)
        .max(expected_d_projection);
    let geometry = active_setup_projection_geometry(&final_params, &opening_batch)
        .expect("heterogeneous projection geometry");

    assert_eq!(geometry.base_ring_dim(), base_ring_dimension);
    assert_eq!(geometry.a_projection_width(), expected_a_projection);
    assert_eq!(geometry.b_projection_width(), expected_b_projection);
    assert_eq!(geometry.d_projection_width(), expected_d_projection);
    assert_eq!(geometry.required(), expected_ring_slots);
    assert_eq!(
        active_setup_field_len(&final_params, &opening_batch).expect("active setup field length"),
        expected_ring_slots * base_ring_dimension
    );
}

#[test]
fn setup_prefix_params_project_b_width_for_smaller_outer_dimension() {
    let mut prefix_params = prefix_eligible_level_params();
    retarget_group_role_dims(&mut prefix_params, 128, 64);
    let outer = prefix_params.outer_commit_matrix;
    prefix_params.outer_commit_matrix = OuterCommitMatrixParams::new_unchecked(
        outer.security_policy(),
        outer.sis_table_key().table_digest,
        outer.sis_modulus_profile(),
        outer.output_rank(),
        32,
        outer.coeff_linf_bound(),
        outer.ring_dimension(),
    );
    let params = setup_prefix_precommitted_params(&prefix_params, 128)
        .expect("mixed-dimension setup-prefix params");

    params
        .validate()
        .expect("projected B width must satisfy the precommitted contract");
    let ratio = params.layout.inner_commit_matrix.ring_dimension()
        / params.layout.outer_commit_matrix.ring_dimension();
    let expected_b_width = params.layout.num_live_blocks
        * params.layout.inner_commit_matrix.output_rank()
        * params.layout.num_digits_outer
        * ratio;
    assert_eq!(
        params.layout.outer_commit_matrix.input_width(),
        expected_b_width
    );
}

#[test]
fn setup_prefix_coverage_eval_len_uses_exact_registry_match() {
    use akita_field::Prime32Offset99 as F;

    let mut level_params = prefix_eligible_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let source_ring_dimension = 32;
    let natural_len = 129usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
    let id = setup_prefix_slot_id(natural_len, commitment_params);
    level_params.setup_prefix = Some(id.clone());
    let slot = verifier_slot_for_id(id.clone());
    let mut registry = SetupPrefixVerifierRegistry::<F>::new([0; 32].into());
    registry.insert(slot).expect("insert slot");

    let setup_eval_len = setup_prefix_coverage_eval_len(
        Some(n_prefix),
        &registry.get(&id).expect("registered slot").id,
        &level_params,
        natural_len,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect("selection succeeds");
    assert_eq!(id.d_setup(), 64);
    assert_eq!(setup_eval_len, 8);

    let external_setup_eval_len = setup_prefix_coverage_eval_len(
        None,
        &registry.get(&id).expect("registered slot").id,
        &level_params,
        natural_len,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect("external committed source selection succeeds");
    assert_eq!(external_setup_eval_len, 8);

    let err = setup_prefix_coverage_eval_len(
        Some(n_prefix),
        &registry.get(&id).expect("registered slot").id,
        &level_params,
        natural_len,
        512,
        "slot does not cover request",
    )
    .expect_err("producer dimension must divide the full prefix");
    assert!(err
        .to_string()
        .contains("setup prefix full length must be divisible"));

    let err = setup_prefix_coverage_eval_len(
        Some(n_prefix),
        &registry.get(&id).expect("registered slot").id,
        &level_params,
        natural_len + 1,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect_err("different natural_len must fail");
    assert!(err.to_string().contains("slot does not cover request"));

    let err = setup_prefix_coverage_eval_len(
        Some(5 * source_ring_dimension),
        &registry.get(&id).expect("registered slot").id,
        &level_params,
        193,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect_err("natural prefix beyond shared setup must fail");
    assert!(err
        .to_string()
        .contains("setup prefix request exceeds shared matrix capacity"));
}

#[test]
fn setup_prefix_coverage_eval_len_rejects_unplanned_level_params() {
    let mut level_params = prefix_eligible_level_params();
    let d_setup = 64;
    let natural_len = 65usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    let id = setup_prefix_slot_id(
        natural_len,
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params"),
    );
    level_params.setup_prefix = None;

    let err = setup_prefix_coverage_eval_len(
        Some(2 * d_setup),
        &id,
        &level_params,
        natural_len,
        d_setup,
        "slot does not cover request",
    )
    .expect_err("unplanned Stage 3 prefix must fail");
    assert!(err
        .to_string()
        .contains("Stage 3 requires a selected setup-prefix slot"));
}

#[test]
fn prover_registry_duplicate_insert_does_not_replace_existing_slot() {
    use akita_field::Prime32Offset99 as F;

    let mut level_params = sample_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, 64).expect("prefix params");
    let id = setup_prefix_slot_id(1, commitment_params);
    let slot = || {
        let inner_rows =
            RingVec::from_coeffs_with_ring_dim(vec![F::zero(); 64], 64).expect("inner rows");
        let matrix = &id.commitment_params.layout.outer_commit_matrix;
        let plan = crate::CompressionChainPlan::for_complete_source(
            matrix.sis_modulus_profile(),
            matrix.output_rank() * matrix.ring_dimension(),
        )
        .expect("compression plan");
        let stages = plan
            .maps()
            .iter()
            .map(|map| {
                crate::PackedNegativeBinary::from_bytes(*map, vec![0; map.packed_digit_bytes()])
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("packed stages");
        let witness =
            crate::CompressionChainWitness::new(plan.clone(), stages).expect("compression witness");
        let quotients = plan
            .maps()
            .iter()
            .map(|map| {
                RingVec::from_coeffs_with_ring_dim(
                    vec![F::zero(); map.output_coefficients()],
                    map.ring_dimension(),
                )
                .expect("quotient")
            })
            .collect::<Vec<_>>();
        let hint = AkitaCommitmentHint::<F>::singleton_with_outer_compression(
            inner_rows, &witness, &quotients,
        )
        .expect("hint");
        SetupPrefixSlot {
            id: id.clone(),
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![
                    F::zero();
                    plan.terminal_coefficients()
                ])],
            },
            hint,
        }
    };

    let mut registry = SetupPrefixProverRegistry::<F>::new([0; 32].into());
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);

    let mut missing_stages = slot();
    missing_stages.hint = AkitaCommitmentHint::singleton(
        RingVec::from_coeffs_with_ring_dim(vec![F::zero(); 64], 64).expect("inner rows"),
    )
    .expect("uncompressed hint");
    let mut missing_registry = SetupPrefixProverRegistry::<F>::new([0; 32].into());
    missing_registry
        .insert(missing_stages)
        .expect_err("setup-prefix hints must retain both compression stages");
}

#[test]
fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
    use akita_field::Prime32Offset99 as F;

    let mut level_params = sample_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, 64).expect("prefix params");
    let id = setup_prefix_slot_id(1, commitment_params);
    let slot = || verifier_slot_for_id(id.clone());

    let mut registry = SetupPrefixVerifierRegistry::<F>::new([0; 32].into());
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);
}
