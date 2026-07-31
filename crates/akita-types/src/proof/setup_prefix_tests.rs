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
    let field_element_digits = crate::sis::compute_num_digits_field_width(128, 3);
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
    let w_b = opening_batch.num_total_polynomials()
        * lp.inner_commit_matrix.output_rank()
        * lp.num_live_blocks
        * lp.num_digits_open;
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
fn active_setup_field_len_includes_mixed_role_subcolumns() {
    let mut lp = sample_level_params();
    let inner = &lp.inner_commit_matrix;
    lp.inner_commit_matrix = crate::InnerCommitMatrixParams::new_unchecked(
        inner.security_policy(),
        inner.sis_table_key().table_digest,
        inner.sis_modulus_profile(),
        inner.output_rank(),
        inner.input_width(),
        inner.coeff_linf_bound().max(1),
        128,
    );
    let opening_batch = OpeningClaimsLayout::new(5, 3).expect("opening batch");
    let a_slots =
        lp.inner_commit_matrix.output_rank() * lp.num_positions_per_block * lp.num_digits_inner * 2;
    let b_slots = lp.outer_commit_matrix.output_rank()
        * opening_batch.num_total_polynomials()
        * lp.inner_commit_matrix.output_rank()
        * lp.num_live_blocks
        * lp.num_digits_outer
        * 2;
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
            table_digest: inner.sis_table_key().table_digest,
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
        let b_cols = group_layout.num_polynomials()
            * group_params.a_rows_len()
            * group_params.num_live_blocks()
            * group_params.num_digits_outer()
            * (dims.d_a() / dims.d_b());
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
fn select_setup_prefix_slot_uses_exact_registry_match() {
    use akita_field::Prime32Offset99 as F;

    let level_params = prefix_eligible_level_params();
    let source_ring_dimension = 32;
    let natural_len = 129usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
    let id = setup_prefix_slot_id(natural_len, commitment_params);
    let mut level_params = level_params;
    level_params.setup_prefix = Some(id.clone());
    let slot = SetupPrefixVerifierSlot {
        id: id.clone(),
        natural_len,
        padded_len: n_prefix,
        commitment: SetupPrefixPublicCommitment {
            rows: vec![RingVec::from_coeffs(vec![F::zero()])],
        },
    };
    let mut registry = SetupPrefixVerifierRegistry::<F>::new([0; 32].into());
    registry.insert(slot).expect("insert slot");

    let selection = select_setup_prefix_slot(
        5,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len))
        },
        &level_params,
        natural_len,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect("selection succeeds")
    .expect("slot selected");
    assert_eq!(&selection.0.id, &id);
    assert_eq!(selection.0.id.d_setup(), 64);
    assert_eq!(selection.1, 8);

    let err = select_setup_prefix_slot(
        5,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len))
        },
        &level_params,
        natural_len + 1,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect_err("different natural_len must fail");
    assert!(err.to_string().contains("slot does not cover request"));

    let err = select_setup_prefix_slot(
        5,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len / 2))
        },
        &level_params,
        natural_len,
        source_ring_dimension,
        "slot does not cover request",
    )
    .expect_err("insufficient padded slot capacity must fail");
    assert!(err.to_string().contains(
        "slot does not cover request: slot natural/padded lengths are 129/128, active lengths are 129/256"
    ));

    let err = select_setup_prefix_slot(
        5,
        |candidate| {
            registry
                .get(candidate)
                .map(|slot| (slot, slot.natural_len, slot.padded_len))
        },
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
fn select_setup_prefix_slot_rejects_missing_registry_entry() {
    use akita_field::Prime32Offset99 as F;

    let mut level_params = prefix_eligible_level_params();
    let d_setup = 64;
    let natural_len = 65usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    level_params.setup_prefix = Some(setup_prefix_slot_id(
        natural_len,
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params"),
    ));

    let err = select_setup_prefix_slot::<SetupPrefixVerifierSlot<F>, _>(
        2,
        |_: &SetupPrefixSlotId| None,
        &level_params,
        natural_len,
        d_setup,
        "slot does not cover request",
    )
    .expect_err("missing registry entry must fail");
    assert!(err
        .to_string()
        .contains("required setup prefix slot is missing from registry"));
    let _ = F::zero();
}

#[test]
fn prover_registry_duplicate_insert_does_not_replace_existing_slot() {
    use crate::proof::DigitBlocks;
    use akita_field::Prime32Offset99 as F;

    let commitment_params =
        setup_prefix_precommitted_params(&sample_level_params(), 64).expect("prefix params");
    let id = setup_prefix_slot_id(1, commitment_params);
    let slot = || {
        // D-free hint: one empty digit block at stride 32 (the former D).
        let decomposed = DigitBlocks::from_blocks(vec![Vec::new()], 64).expect("digit blocks");
        let hint = AkitaCommitmentHint::<F>::singleton(decomposed);
        SetupPrefixSlot {
            id: id.clone(),
            natural_len: id.natural_len,
            padded_len: id.n_prefix().expect("padded len"),
            // One commitment row of d_setup = 32 coefficients.
            commitment: SetupPrefixPublicCommitment {
                rows: vec![RingVec::from_coeffs(vec![F::zero(); 64])],
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
}

#[test]
fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
    use akita_field::Prime32Offset99 as F;

    let commitment_params =
        setup_prefix_precommitted_params(&sample_level_params(), 64).expect("prefix params");
    let id = setup_prefix_slot_id(1, commitment_params);
    let slot = || SetupPrefixVerifierSlot {
        id: id.clone(),
        natural_len: id.natural_len,
        padded_len: id.n_prefix().expect("padded len"),
        commitment: SetupPrefixPublicCommitment {
            rows: vec![RingVec::from_coeffs(vec![F::zero()])],
        },
    };

    let mut registry = SetupPrefixVerifierRegistry::<F>::new([0; 32].into());
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);
}
