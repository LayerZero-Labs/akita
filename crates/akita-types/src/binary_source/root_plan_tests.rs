use super::*;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, CANONICAL_IMAGE_WINDOW};
use akita_challenges::SparseChallenge;
use jolt_field::{One, Prime128Offset275, Ring, Zero};

type F = Prime128Offset275;

fn descriptor() -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32]],
        BinaryCommitmentProfiles {
            a: [2; 32],
            b: [3; 32],
        },
    )
    .unwrap()
}

#[test]
fn five_segments_share_addresses_when_scheduled_domain_grows() {
    let descriptor = descriptor();
    let plan = BinaryRootWitnessPlan::new(&descriptor, 2, 256).unwrap();
    assert_eq!(plan.repacking().domain_len(), 65536);
    assert_eq!(
        plan.repacking()
            .segments()
            .iter()
            .map(|shape| (shape.elements, shape.digits))
            .collect::<Vec<_>>(),
        vec![(1, 4), (4, 8), (2, 1), (1, 9), (1, 6)]
    );
    let larger = plan.clone().with_domain_len(131072).unwrap();
    for segment in 0..5 {
        assert_eq!(
            plan.repacking().segment_range(segment).unwrap(),
            larger.repacking().segment_range(segment).unwrap()
        );
    }
    assert!(larger.repacking().is_padding_address(131071).unwrap());
    assert!(plan.clone().with_domain_len(32768).is_err());
    assert!(plan.with_domain_len(65537).is_err());
    assert!(BinaryRootWitnessPlan::new(&descriptor, 0, 256).is_err());
}

#[test]
fn direct_weights_join_a_and_terminal_carries_but_not_extended_padding() {
    let descriptor = descriptor();
    let plan = BinaryRootWitnessPlan::new(&descriptor, 2, 256)
        .unwrap()
        .with_domain_len(131072)
        .unwrap();
    let claim = BinaryTerminalClaim::new(
        &descriptor,
        [1; 32],
        vec![[0; 21]; 2],
        vec![[0; 21]],
        [0; 21],
    )
    .unwrap();
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: Default::default(),
                coeffs: Default::default()
            };
            2
        ],
        2,
        1,
    )
    .unwrap();
    let matrix = vec![vec![[0i128; 648]]; 2];
    let weights = plan
        .relation_weights(
            Some(&claim),
            Some(&matrix),
            &challenges,
            F::from_u64(17),
            &[[F::one(); 648]; 2],
            [F::from_u64(19), F::from_u64(23)],
        )
        .unwrap();
    let table = vec![F::zero(); plan.repacking().domain_len()];
    weights.check_table(&table).unwrap();
    for address in [
        plan.repacking().coefficient_address(2, 0, 0, 0).unwrap(),
        plan.repacking().coefficient_address(3, 0, 0, 0).unwrap(),
        plan.repacking().coefficient_address(4, 0, 0, 0).unwrap(),
    ] {
        let mut altered = table.clone();
        altered[address] = F::one();
        assert!(weights.check_table(&altered).is_err());
    }
    // No root row reads the enlarged domain's padding.
    assert!(plan.repacking().is_padding_address(131071).unwrap());
    let mut altered = table.clone();
    altered[131071] = F::one();
    weights.check_table(&altered).unwrap();
    let mut invalid_matrix = matrix.clone();
    invalid_matrix[0][0][0] = i128::from(CANONICAL_IMAGE_WINDOW.1) + 1;
    assert!(binary_direct_a_weights(
        &plan,
        Some(&invalid_matrix),
        &challenges,
        &[[F::one(); 648]; 2],
    )
    .is_err());
}
