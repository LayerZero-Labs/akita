//! Setup-contribution characterization test cases: the packed
//! `setup_contribution_plan` path must match the naive oracle and an
//! eq-weighted materialized reference across the fixture shapes.

use super::fixtures::{SetupContributionFixture, SetupContributionShape};

#[test]
fn setup_contribution_matches_materialized_on_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_recursive_multi_group_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multi_group())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_index_weight_mle_matches_materialized() {
    SetupContributionFixture::from_shape(&SetupContributionShape::root_single_point())
        .assert_setup_index_weight_mle_matches_materialized();
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multi_group())
        .assert_setup_index_weight_mle_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_terminal_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::terminal_relation_only())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_dense_non_pow2_z_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::dense_non_pow2_z())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_on_batched_root_fixture() {
    SetupContributionFixture::from_shape(&SetupContributionShape::batched_root())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_with_e_t_offset_carries() {
    SetupContributionFixture::from_shape(&SetupContributionShape::e_t_offset_carry())
        .assert_direct_matches_materialized();
}

#[test]
fn setup_contribution_matches_materialized_with_pow2_z_offset_carry() {
    SetupContributionFixture::from_shape(&SetupContributionShape::pow2_z_offset_carry())
        .assert_direct_matches_materialized();
}
