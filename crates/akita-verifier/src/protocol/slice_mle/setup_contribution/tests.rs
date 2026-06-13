use super::fixtures::{SetupContributionFixture, SetupContributionShape};

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
    SetupContributionFixture::from_shape(&SetupContributionShape::z_first_e_t_offset_carry())
        .assert_direct_matches_recursive();
}

#[test]
fn setup_contribution_matches_recursive_with_pow2_z_offset_carry() {
    SetupContributionFixture::from_shape(&SetupContributionShape::pow2_z_offset_carry())
        .assert_direct_matches_recursive();
}
