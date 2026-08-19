use super::*;

#[test]
fn precommit_profile_accepts_a_partial_physical_ring() {
    let params = committed_params_with_geometry(64, 1, 1);
    let profile =
        CommittedGroupProfile::try_from_params(PolynomialGroupLayout::singleton(4), &params)
            .expect("sixteen logical cells fit in one partial D64 ring");

    assert_eq!(profile.group, PolynomialGroupLayout::singleton(4));
    assert_eq!(profile.num_live_ring_elements_per_claim, 1);
    profile
        .validate_frozen_precommit(128)
        .expect("partial physical ring profile remains valid");
}
