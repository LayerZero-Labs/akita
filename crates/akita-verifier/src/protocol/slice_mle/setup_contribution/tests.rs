use super::fixtures::{SetupContributionFixture, SetupContributionShape, TestField, TEST_RING_DIM};
use crate::protocol::slice_mle::setup_inner_product_oracle::materialize_setup_omega;
use akita_algebra::ring::scalar_powers;

#[test]
fn setup_contribution_matches_oracle_on_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_contribution_matches_oracle_on_recursive_multigroup_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multigroup())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_contribution_matches_oracle_on_terminal_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::terminal_relation_only())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_contribution_matches_oracle_on_dense_non_pow2_z_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::dense_non_pow2_z())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_contribution_matches_oracle_on_batched_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::batched_root())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_contribution_matches_oracle_with_w_t_offset_carries() {
    SetupContributionFixture::from_shape(&SetupContributionShape::z_first_w_t_offset_carry())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_contribution_matches_oracle_with_pow2_z_offset_carry() {
    SetupContributionFixture::from_shape(&SetupContributionShape::pow2_z_offset_carry())
        .assert_matches_materialized_oracle();
}

#[test]
fn setup_oracle_keeps_alpha_on_weight_side() {
    let fixture =
        SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point());
    let alpha = fixture.alpha_pows[1];
    let omega = materialize_setup_omega::<TestField, TestField, TEST_RING_DIM>(
        &fixture.prepared,
        &fixture.full_vec_randomness,
        &fixture.alpha_pows,
        &fixture.fold_gadget,
        fixture.offset_w,
        fixture.offset_t,
        fixture.offset_z,
    );

    for (lambda, &bar_weight) in omega.bar_omega.iter().enumerate() {
        if bar_weight.is_zero() {
            continue;
        }
        for y in 1..TEST_RING_DIM {
            let expected = bar_weight * fixture.alpha_pows[y];
            assert_eq!(
                omega.coefficient_weight(lambda, y, TEST_RING_DIM),
                expected,
                "omega_S({lambda}, {y}) must equal bar_omega({lambda}) * alpha^{y}"
            );
        }
    }

    let shifted_alpha = alpha + fixture.full_vec_randomness[0];
    let shifted_alpha_pows = scalar_powers(shifted_alpha, TEST_RING_DIM);
    let shifted_omega = materialize_setup_omega::<TestField, TestField, TEST_RING_DIM>(
        &fixture.prepared,
        &fixture.full_vec_randomness,
        &shifted_alpha_pows,
        &fixture.fold_gadget,
        fixture.offset_w,
        fixture.offset_t,
        fixture.offset_z,
    );
    assert_ne!(
        omega.omega_s, shifted_omega.omega_s,
        "changing alpha must change omega_S while bar_omega stays fixed"
    );
    assert_eq!(
        omega.bar_omega, shifted_omega.bar_omega,
        "bar_omega must not depend on alpha"
    );
}
