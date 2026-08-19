use super::*;

#[test]
fn restricted_precommit_profile_preserves_small_logical_domain() {
    let producer = precommitted_descriptor(12);
    let restricted = producer
        .try_restrict_to_group(PolynomialGroupLayout::singleton(4))
        .expect("restrict profile to a sixteen-cell source");

    assert_eq!(restricted.group, PolynomialGroupLayout::singleton(4));
    assert_eq!(restricted.num_live_ring_elements_per_claim, 1);
    assert_eq!(restricted.num_positions_per_block, 1);
    assert_eq!(restricted.num_live_blocks, 1);
    assert_eq!(
        restricted.outer_slice_count,
        crate::CommitmentSliceCount::ONE
    );
    restricted
        .validate_frozen_precommit(128)
        .expect("one partially live physical ring is valid");
}

#[test]
fn restricted_precommit_profile_supports_constant_polynomial() {
    let restricted = precommitted_descriptor(12)
        .try_restrict_to_group(PolynomialGroupLayout::singleton(0))
        .expect("restrict profile to a one-cell source");

    assert_eq!(restricted.num_live_ring_elements_per_claim, 1);
    restricted
        .validate_frozen_precommit(128)
        .expect("constant source keeps one partially live physical ring");
}
