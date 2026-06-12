use super::fixtures::{SetupContributionFixture, SetupContributionShape};

#[test]
fn setup_contribution_matches_recursive_on_supported_shapes() {
    for shape in [
        SetupContributionShape::root_single_point(),
        SetupContributionShape::recursive_multigroup(),
        SetupContributionShape::tiered_root_single_point(),
        SetupContributionShape::terminal_relation_only(),
        SetupContributionShape::dense_non_pow2_z(),
        SetupContributionShape::batched_root(),
        SetupContributionShape::z_first_e_t_offset_carry(),
        SetupContributionShape::pow2_z_offset_carry(),
    ] {
        SetupContributionFixture::from_shape(&shape).assert_direct_matches_recursive();
    }
}

/// The two `bar_omega` implementations — the const-generic
/// `bar_omega_segment_eval` (via `evaluate_bar_omega_with_eq`) and the generic
/// `weight_at` (via `materialize_bar_omega`) — must agree, single-tier and
/// tiered alike.
#[test]
fn eq_eval_matches_materialized_bar_omega_single_and_tiered() {
    SetupContributionFixture::from_shape(&SetupContributionShape::recursive_multigroup())
        .assert_eq_eval_matches_materialized();
    SetupContributionFixture::from_shape(&SetupContributionShape::tiered_root_single_point())
        .assert_eq_eval_matches_materialized();
}
