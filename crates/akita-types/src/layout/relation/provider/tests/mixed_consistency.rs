use super::*;

#[test]
fn consistency_accepts_canonical_groups_with_unequal_native_role_dimensions() {
    let mut lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .unwrap();
    let group = PolynomialGroupLayout::new(2, 1);
    lp.precommitted_groups.push(PrecommittedLevelParams {
        layout: PrecommittedGroupParams::from_params(group, &lp),
        a_key: key(32, 63, 1),
        b_key: lp.b_key.clone(),
        num_blocks: 1,
        block_len: 1,
        num_digits_commit: 1,
        num_digits_open: 1,
        num_digits_fold_one: 1,
    });
    let opening =
        OpeningClaimsLayout::from_root_groups(&[group], PolynomialGroupLayout::new(2, 1)).unwrap();
    let layout = RelationLayout::from_authenticated_statement(
        &lp,
        &opening,
        crate::RelationMatrixRowLayout::WithDBlock,
        128,
    )
    .unwrap();

    let pre_group = RelationGroupId::Precommitted { index: 0 };
    let pre_a = layout
        .family_provider(RelationRowId::A { group: pre_group })
        .unwrap();
    assert_eq!(pre_a.native_ring_dim(), 32);
    let pre_z = layout
        .segment(RelationSegmentId::Z { group: pre_group })
        .unwrap()
        .span();
    assert!(!pre_z.len().is_multiple_of(64));

    let consistency = layout.family_provider(RelationRowId::Consistency).unwrap();
    assert_eq!(consistency.native_ring_dim(), 64);
}
