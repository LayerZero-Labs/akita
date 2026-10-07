use super::*;
use akita_challenges::SparseChallengeConfig;
use akita_params::layout::setup_prefix_slots::{
    active_setup_field_len, scheduled_setup_prefix, setup_prefix_precommitted_params,
};
use akita_params::test_fixtures::{
    prefix_eligible_level_params, retarget_group_role_dims, retarget_group_role_dims_wide,
    sample_level_params,
};
use akita_params::{
    CommittedGroupParams, OpeningClaimsLayout, OuterCommitMatrixParams, SisModulusProfileId,
};

#[test]
fn setup_prefix_profile_must_commit_every_ring_in_the_full_domain() {
    let natural_len = 129;
    let n_prefix = padded_setup_prefix_len(natural_len);
    let mut level_params = prefix_eligible_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let mut prefix = setup_prefix_precommitted_params(&level_params, n_prefix)
        .expect("full setup-prefix profile");
    prefix
        .profile
        .validate_setup_prefix_geometry(natural_len)
        .expect("full setup-prefix geometry");

    let blocks = prefix.profile.blocks;
    let omitted_tail_rings = blocks.live_ring_elements_per_claim - 1;
    prefix.profile.blocks = akita_params::BlockGeometry::new(
        omitted_tail_rings,
        blocks.positions_per_block,
        omitted_tail_rings.div_ceil(blocks.positions_per_block),
    );
    let error = prefix
        .profile
        .validate_setup_prefix_geometry(natural_len)
        .expect_err("a setup-prefix profile cannot omit real tail rings");
    assert!(error.to_string().contains("must commit all"));
}

#[test]
fn setup_prefix_uses_full_field_digits_for_a_tight_recursive_consumer() {
    let params = CommittedGroupParams::params_only(
        SisModulusProfileId::Q32Offset99,
        64,
        3,
        2,
        3,
        2,
        SparseChallengeConfig::production_for_ring_dim(64).unwrap(),
    )
    .with_decomp(32, 3, 1, 2, 2)
    .expect("tight recursive consumer with setup-prefix capacity");
    assert_eq!(
        params.inner().matrix.input_width(),
        params.blocks().positions_per_block * params.inner().digits.num_digits
    );
    let prefix = setup_prefix_precommitted_params(&params, 128).expect("setup prefix params");
    assert_eq!(
        prefix.profile.inner.digits.num_digits,
        akita_params::sis::compute_num_digits_field_width(
            SisModulusProfileId::Q32Offset99.field_bits(),
            params.inner().digits.log_basis,
        )
    );
    assert_ne!(
        prefix.profile.inner.digits.num_digits,
        params.inner().digits.num_digits
    );
}

#[test]
fn active_setup_field_len_includes_mixed_role_subcolumns() {
    let mut lp = sample_level_params();
    let inner = &lp.inner().matrix;
    lp.own_group_mut().profile.inner.matrix = akita_params::InnerCommitMatrixParams::new_unchecked(
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
    let a_slots = lp.inner().matrix.output_rank()
        * lp.blocks().positions_per_block
        * lp.inner().digits.num_digits
        * 2;
    let b_slots = lp.outer().matrix.output_rank() * lp.outer().matrix.input_width() * 2;
    let d_slots = lp.open().matrix.output_rank()
        * opening_batch.num_total_polynomials()
        * lp.blocks().live_blocks
        * lp.open().digits.num_digits
        * 2;
    let expected_field_len = a_slots.max(b_slots).max(d_slots) * 64;

    assert_eq!(
        active_setup_field_len(&lp, &opening_batch).expect("mixed-D field len"),
        expected_field_len
    );
}

fn verifier_slot_for_id<F: Field>(id: SetupPrefixSlotId) -> SetupPrefixVerifierSlot<F> {
    let payload_coefficients = setup_prefix_compression_plan(&id.commitment_profile)
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
fn setup_prefix_params_project_b_width_for_smaller_outer_dimension() {
    let mut prefix_params = prefix_eligible_level_params();
    retarget_group_role_dims(&mut prefix_params, 128, 64);
    let outer = prefix_params.outer().matrix;
    prefix_params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
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
    let ratio =
        params.profile.inner.matrix.ring_dimension() / params.profile.outer.matrix.ring_dimension();
    let expected_b_width = params.profile.blocks.live_blocks
        * params.profile.inner.matrix.output_rank()
        * params.profile.outer.digits.num_digits
        * ratio;
    assert_eq!(params.profile.outer.matrix.input_width(), expected_b_width);
}

#[test]
fn setup_prefix_coverage_eval_len_uses_exact_registry_match() {
    use jolt_field::Prime32Offset99 as F;

    let mut level_params = prefix_eligible_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let source_ring_dimension = 32;
    let natural_len = 129usize;
    let n_prefix = padded_setup_prefix_len(natural_len);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params");
    let scheduled = scheduled_setup_prefix(natural_len, commitment_params);
    level_params.set_setup_prefix(Some(scheduled)).unwrap();
    let id = scheduled.slot_id().expect("setup prefix group");
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
    let id = scheduled_setup_prefix(
        natural_len,
        setup_prefix_precommitted_params(&level_params, n_prefix).expect("prefix params"),
    )
    .slot_id()
    .expect("setup prefix group");
    level_params.set_setup_prefix(None).unwrap();

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
fn verifier_registry_duplicate_insert_does_not_replace_existing_slot() {
    use jolt_field::Prime32Offset99 as F;

    let natural_len = 64;
    let mut level_params = sample_level_params();
    retarget_group_role_dims_wide(&mut level_params, 64, 64, 1024);
    let commitment_params =
        setup_prefix_precommitted_params(&level_params, natural_len).expect("prefix params");
    let id = scheduled_setup_prefix(natural_len, commitment_params)
        .slot_id()
        .expect("setup prefix group");
    let slot = || verifier_slot_for_id(id.clone());

    let mut registry = SetupPrefixVerifierRegistry::<F>::new([0; 32].into());
    registry.insert(slot()).expect("first insert");
    registry
        .insert(slot())
        .expect_err("duplicate insert must fail");

    assert_eq!(registry.len(), 1);
}
