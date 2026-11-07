use super::*;
use akita_params::GrindingQueryKind;

#[test]
fn group_prices_follow_exact_work_thresholds_and_nonce_byte_bounds() {
    for (owners, columns, count, bits, bytes) in [
        (1, 256, 256, 0, 0),
        (257, 1, 257, 1, 2),
        (4, 256, 1024, 2, 2),
        (4096, 256, 1_048_576, 12, 3),
    ] {
        let group = BinaryColumnGroupPlan::new(0, owners, columns).unwrap();
        assert_eq!(group.coordinate_count(), count);
        assert_eq!(group.work_run().grind_bits(), bits);
        assert_eq!(group.work_run().nonce_max_bytes().unwrap(), bytes);
        assert_eq!(group.work_run().challenge_set_bits(), 136);
        assert_eq!(group.work_run().loss_factor(), count as u64);
        assert_eq!(group.work_run().binary_group(), Some(0));
        assert_eq!(group.response_run().binary_group(), Some(0));
        assert_eq!(group.response_run().nonce_bits(), 12);
    }
}

#[test]
fn owner_major_coordinates_are_unique_and_checked() {
    let group = BinaryColumnGroupPlan::new(2, 3, 5).unwrap();
    let mut offsets = Vec::new();
    for owner in 0..group.owners() {
        for column in 0..group.columns() {
            offsets.push(group.coordinate_index(owner, column).unwrap());
        }
    }
    assert_eq!(offsets, (0..15).collect::<Vec<_>>());
    for (owner, column) in [(3, 0), (0, 5), (usize::MAX, usize::MAX)] {
        assert!(matches!(
            group.coordinate_index(owner, column),
            Err(AkitaError::InvalidInput(_))
        ));
    }
}

#[test]
fn malformed_counts_and_aggregate_caps_return_typed_errors() {
    for (group, owners, columns) in [
        (64, 1, 1),
        (u32::MAX, 1, 1),
        (0, 0, 1),
        (0, 4097, 1),
        (0, 1, 0),
        (0, 1, 257),
        (0, usize::MAX, usize::MAX),
    ] {
        assert!(matches!(
            BinaryColumnGroupPlan::new(group, owners, columns),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
    let first = BinaryColumnGroupPlan::new(0, 4096, 256).unwrap();
    let extra = BinaryColumnGroupPlan::new(1, 1, 1).unwrap();
    assert!(BinaryColumnPlan::new(vec![first]).is_ok());
    for groups in [vec![], vec![extra], vec![extra, first], vec![first, extra]] {
        assert!(matches!(
            BinaryColumnPlan::new(groups),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
    let groups = (0..64)
        .map(|group| BinaryColumnGroupPlan::new(group, 1, 1).unwrap())
        .collect::<Vec<_>>();
    assert!(BinaryColumnPlan::new(groups.clone()).is_ok());
    let mut oversized = groups;
    oversized.push(extra);
    assert!(matches!(
        BinaryColumnPlan::new(oversized),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn plan_replays_response_work_and_coordinates_once_per_group() {
    let groups = vec![
        BinaryColumnGroupPlan::new(0, 1, 256).unwrap(),
        BinaryColumnGroupPlan::new(1, 257, 1).unwrap(),
        BinaryColumnGroupPlan::new(2, 2, 3).unwrap(),
    ];
    let plan = BinaryColumnPlan::new(groups.clone()).unwrap();
    assert_eq!(plan.coordinate_count(), 519);
    assert_eq!(plan.grinding_plan().expanded_query_count(), 528);
    for (&group, runs) in groups
        .iter()
        .zip(plan.grinding_plan().runs().chunks_exact(3))
    {
        assert_eq!(
            runs,
            &[group.response_run(), group.work_run(), group.draw_run()]
        );
        assert_eq!(runs[0].kind(), GrindingQueryKind::FoldResponse);
        assert_eq!(runs[1].kind(), GrindingQueryKind::FoldChallengeWork);
        assert_eq!(
            runs[2].fold_coordinate_count(),
            Some(group.coordinate_count() as u64)
        );
    }
    // The shape belongs to the statement even when its coordinate count and
    // therefore its work price are equal.
    let a = BinaryColumnPlan::new(vec![BinaryColumnGroupPlan::new(0, 2, 3).unwrap()]).unwrap();
    let b = BinaryColumnPlan::new(vec![BinaryColumnGroupPlan::new(0, 3, 2).unwrap()]).unwrap();
    assert_eq!(
        a.grinding_plan().digest().unwrap(),
        b.grinding_plan().digest().unwrap()
    );
    assert_ne!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
}
