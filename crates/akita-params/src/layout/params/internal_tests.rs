use super::groups::FoldGroups;
use super::*;
use crate::test_fixtures::{
    address_oracle_fixture, address_oracle_group_params, address_oracle_precommit,
    sample_layout_lp, sample_multi_group_root_params,
};
use crate::PolynomialGroupLayout;

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

#[test]
fn opening_batch_validation_checks_every_distinct_group_in_a_repeated_run() {
    let repeated_fixture = || {
        let mut lp = address_oracle_group_params(64, 64, 64, 8);
        lp.set_precommitted_groups(vec![address_oracle_precommit(128, 64, 64, 16, 1); 3])
            .unwrap();
        let layouts = lp
            .precommitted_groups()
            .iter()
            .map(|group| group.profile.group)
            .collect::<Vec<_>>();
        let batch =
            OpeningClaimsLayout::from_root_groups(&layouts, PolynomialGroupLayout::new(4, 2))
                .expect("repeated opening layout");
        (lp, batch)
    };
    let corrupt = |lp: &mut CommittedGroupParams, group_index| {
        lp.preceding_group_mut_for_test(group_index)
            .unwrap()
            .profile
            .outer
            .matrix
            .sis_table_key
            .ring_dimension /= 2;
    };

    let (lp, batch) = repeated_fixture();
    assert_eq!(
        lp.validated_groups(&batch).expect("repeated groups").len(),
        batch.num_groups()
    );
    for corrupted in [vec![0], vec![2], vec![1, 2]] {
        let (mut lp, batch) = repeated_fixture();
        for &group_index in &corrupted {
            corrupt(&mut lp, group_index);
        }
        assert!(
            lp.validate_opening_batch(&batch).is_err(),
            "corrupting groups {corrupted:?} must fail validation"
        );
    }
}
