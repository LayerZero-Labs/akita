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
fn opening_validation_rejects_changed_policy_after_valid_input() {
    let (lp, _) = sample_multi_group_root_params();
    let params = *lp.preceding_group_params(0).unwrap();
    params.validate().unwrap();
    // The repeat is answered from the stored success.
    params.validate().unwrap();

    let mut changed = params;
    changed.profile.version = 0;
    assert!(matches!(
        changed.validate(),
        Err(AkitaError::InvalidSetup(message))
            if message == "unsupported committed-group profile version 0"
    ));

    let mut changed = params;
    changed.setup_natural_len = Some(0);
    assert!(matches!(
        changed.validate(),
        Err(AkitaError::InvalidSetup(message))
            if message == "setup-prefix inner digit depth 2 does not cover the full field width (43 digits)"
    ));

    let mut changed = params;
    changed.opening.log_basis_open = 0;
    assert!(matches!(
        changed.validate(),
        Err(AkitaError::InvalidSetup(message))
            if message == "precommitted exact fold plan is missing or inconsistent"
    ));

    changed.profile.version = 0;
    assert!(matches!(
        changed.validate(),
        Err(AkitaError::InvalidSetup(message))
            if message == "unsupported committed-group profile version 0"
    ));

    params.validate().unwrap();
}

#[test]
fn opening_validation_is_unchanged_once_the_table_is_full() {
    let (lp, _) = sample_multi_group_root_params();
    let params = *lp.preceding_group_params(0).unwrap();
    // `validate` requires only a nonzero fold digit depth, so these are twelve
    // distinct valid values: more than the table of validated values can hold.
    let values: Vec<_> = (1..=12)
        .map(|num_digits_fold| {
            let mut value = params;
            value.opening.num_digits_fold = num_digits_fold;
            value
        })
        .collect();
    for _ in 0..2 {
        for value in &values {
            value.validate().unwrap();
        }
    }

    // Changed copies of the first and of the last value, whichever of them the
    // table holds, are rejected with the errors of the full checks.
    for value in [values[0], values[11]] {
        let mut changed = value;
        changed.profile.version = 0;
        assert!(matches!(
            changed.validate(),
            Err(AkitaError::InvalidSetup(message))
                if message == "unsupported committed-group profile version 0"
        ));

        let mut changed = value;
        changed.opening.log_basis_open = 0;
        assert!(matches!(
            changed.validate(),
            Err(AkitaError::InvalidSetup(message))
                if message == "precommitted exact fold plan is missing or inconsistent"
        ));

        value.validate().unwrap();
    }
}

#[test]
fn opening_validation_is_consistent_across_threads() {
    use std::sync::Barrier;

    let (lp, _) = sample_multi_group_root_params();
    let values = [
        *lp.preceding_group_params(0).unwrap(),
        address_oracle_precommit(128, 64, 64, 16, 1),
    ];
    assert_ne!(values[0], values[1]);
    let mut invalid = values[0];
    invalid.profile.version = 0;
    let start = Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                start.wait();
                for _ in 0..32 {
                    for params in &values {
                        params.validate().unwrap();
                    }
                    assert!(matches!(
                        invalid.validate(),
                        Err(AkitaError::InvalidSetup(message))
                            if message == "unsupported committed-group profile version 0"
                    ));
                }
            });
        }
    });
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
