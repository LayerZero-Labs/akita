use super::groups::FoldGroups;
use super::*;
use crate::test_fixtures::{
    address_oracle_fixture, sample_layout_lp, sample_multi_group_root_params,
};

#[test]
fn fold_groups_reject_empty_and_misordered_topologies_without_unwinding() {
    let empty = std::panic::catch_unwind(|| FoldGroups::try_from_vec(Vec::new()));
    assert!(matches!(empty, Ok(Err(AkitaError::InvalidSetup(_)))));

    let ordinary = *sample_layout_lp().own_group();
    let mut prefix = ordinary;
    prefix.setup_natural_len = Some(64);

    let sole_prefix = std::panic::catch_unwind(|| FoldGroups::try_from_vec(vec![prefix]));
    assert!(matches!(sole_prefix, Ok(Err(AkitaError::InvalidSetup(_)))));

    let late_prefix =
        std::panic::catch_unwind(|| FoldGroups::try_from_vec(vec![ordinary, prefix, ordinary]));
    assert!(matches!(late_prefix, Ok(Err(AkitaError::InvalidSetup(_)))));
}

#[test]
fn precommitted_params_reject_frozen_matrix_dimension_mismatch() {
    let (mut lp, _) = sample_multi_group_root_params();
    let precommitted = lp.preceding_group_mut_for_test(0).unwrap();
    precommitted
        .profile
        .outer
        .matrix
        .sis_table_key
        .ring_dimension /= 2;
    let err = precommitted
        .validate()
        .expect_err("frozen B dimension must match the serialized B matrix");
    assert!(matches!(err, AkitaError::InvalidSetup(_)));
}

#[test]
fn relation_geometry_revalidates_frozen_precommitted_profiles() {
    let (mut lp, batch) = address_oracle_fixture(2);
    lp.preceding_group_mut_for_test(0)
        .unwrap()
        .profile
        .outer
        .matrix
        .sis_table_key
        .ring_dimension /= 2;
    assert!(crate::RelationWitnessGeometry::for_level(&lp, &batch, 2).is_err());
}
