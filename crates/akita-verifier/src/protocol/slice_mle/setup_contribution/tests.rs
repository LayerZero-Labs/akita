use super::fixtures::{SetupContributionFixture, SetupContributionShape, TEST_RING_DIM};
use akita_types::{ensure_setup_envelope, setup_required_for_shape, SetupRelationShape};

#[test]
fn setup_contribution_inputs_match_challenge_free_geometry() {
    for shape in [
        SetupContributionShape::root_single_point(),
        SetupContributionShape::recursive_multigroup(),
        SetupContributionShape::tiered_root_single_point(),
        SetupContributionShape::terminal_relation_only(),
        SetupContributionShape::dense_non_pow2_z(),
        SetupContributionShape::batched_root(),
        SetupContributionShape::e_t_offset_carry(),
        SetupContributionShape::pow2_z_offset_carry(),
    ] {
        let fixture = SetupContributionFixture::from_shape(&shape);
        let inputs = fixture.prepared.create_setup_contribution_inputs();
        let relation_shape = SetupRelationShape::from(&inputs);
        let required = setup_required_for_shape(&relation_shape).expect("required");
        ensure_setup_envelope(&fixture.setup, required, TEST_RING_DIM).expect("envelope");
    }
}

#[test]
fn ensure_setup_envelope_rejects_undersized_matrix_for_fixture_shape() {
    use akita_algebra::CyclotomicRing;
    use akita_field::Prime128OffsetA7F7;
    use akita_types::{AkitaExpandedSetup, AkitaSetupSeed, FlatMatrix};

    type F = Prime128OffsetA7F7;
    let fixture =
        SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point());
    let inputs = fixture.prepared.create_setup_contribution_inputs();
    let relation_shape = SetupRelationShape::from(&inputs);
    let required = setup_required_for_shape(&relation_shape).expect("required");

    let tiny_setup = AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        AkitaSetupSeed {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            gen_ring_dim: TEST_RING_DIM,
            max_setup_len: 1,
            public_matrix_seed: [9u8; 32],
        },
        FlatMatrix::from_ring_slice::<TEST_RING_DIM>(
            &[CyclotomicRing::<F, TEST_RING_DIM>::zero(); 1],
        ),
    );
    let err = ensure_setup_envelope(&tiny_setup, required, TEST_RING_DIM).expect_err("tiny");
    assert!(matches!(err, akita_field::AkitaError::InvalidSetup(_)));
}

#[test]
fn setup_contribution_matches_recursive_on_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_direct_matches_recursive();
}

#[test]
fn setup_contribution_matches_recursive_on_recursive_multigroup_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multigroup())
        .assert_direct_matches_recursive();
}

/// Tiered plan: the `bar_omega` (recursive) path must include the first-tier
/// `B'` and second-tier `F` bindings so it agrees with the direct scan. Guards
/// against the recursive setup mode silently omitting the `F` binding.
#[test]
fn setup_contribution_matches_recursive_on_tiered_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::tiered_root_single_point())
        .assert_direct_matches_recursive();
}

/// The two `bar_omega` implementations — the const-generic
/// `bar_omega_segment_eval` (via `evaluate_bar_omega_with_eq`) and the generic
/// `weight_at` (via `materialize_bar_omega`) — must agree, single-tier and
/// tiered alike.
#[test]
fn eq_eval_matches_materialized_bar_omega_single_and_tiered() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_eq_eval_matches_materialized();
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multigroup())
        .assert_eq_eval_matches_materialized();
    SetupContributionFixture::from_shape(&SetupContributionShape::tiered_root_single_point())
        .assert_eq_eval_matches_materialized();
}

#[test]
fn setup_contribution_matches_recursive_on_terminal_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::terminal_relation_only())
        .assert_direct_matches_recursive();
}

#[test]
fn setup_contribution_matches_recursive_on_dense_non_pow2_z_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::dense_non_pow2_z())
        .assert_direct_matches_recursive();
}

#[test]
fn setup_contribution_matches_recursive_on_batched_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::batched_root())
        .assert_direct_matches_recursive();
}

#[test]
fn setup_contribution_matches_recursive_with_e_t_offset_carries() {
    SetupContributionFixture::from_shape(&SetupContributionShape::e_t_offset_carry())
        .assert_direct_matches_recursive();
}

#[test]
fn setup_contribution_matches_recursive_with_pow2_z_offset_carry() {
    SetupContributionFixture::from_shape(&SetupContributionShape::pow2_z_offset_carry())
        .assert_direct_matches_recursive();
}

#[test]
fn stage3_geometry_matches_setup_contribution_plan_on_fixtures() {
    for shape in [
        SetupContributionShape::root_single_point(),
        SetupContributionShape::recursive_multigroup(),
        SetupContributionShape::tiered_root_single_point(),
        SetupContributionShape::terminal_relation_only(),
        SetupContributionShape::dense_non_pow2_z(),
        SetupContributionShape::batched_root(),
        SetupContributionShape::e_t_offset_carry(),
        SetupContributionShape::pow2_z_offset_carry(),
    ] {
        SetupContributionFixture::from_shape(&shape)
            .assert_geometry_matches_setup_contribution_plan();
    }
}
