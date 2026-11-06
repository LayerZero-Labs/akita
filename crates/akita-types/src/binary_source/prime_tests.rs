use super::*;
use crate::binary_source::prime_weights::prime_plane_relation_weights;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinaryRootWitnessPlan};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use akita_challenges::{Challenges, SparseChallenge};
use akita_params::{RelationPolynomial, RelationRowGeometry};
use jolt_field::{One, Prime128Offset275, Ring};

type F = Prime128Offset275;

fn descriptor() -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32], [2; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap()
}

fn claim() -> PrimeGroupClaim {
    PrimeGroupClaim {
        point: vec![0; 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: 0,
        }],
    }
}

#[test]
fn prime_manifest_rejects_repeated_owners_empty_values_and_noncanonical_fields() {
    let source = descriptor();
    for change in 0..7 {
        let mut bad = claim();
        match change {
            0 => bad.evaluations[0].value = BINARY_SETUP_P,
            1 => bad.point[0] = BINARY_SETUP_P,
            2 => bad.evaluations[0].polynomial = 2,
            3 => bad.evaluations.clear(),
            4 => bad.evaluations.push(bad.evaluations[0].clone()),
            5 => {
                bad.point.pop();
            }
            _ => bad.evaluations.push(PrimePlaneClaim {
                polynomial: 0,
                value: 1,
            }),
        }
        assert!(
            matches!(
                PrimePlaneManifest::new(&source, Some(bad)),
                Err(AkitaError::InvalidInput(_))
            ),
            "mutation {change}"
        );
    }
    assert!(PrimePlaneManifest::new(&source, None)
        .unwrap()
        .planes()
        .is_empty());
}

#[test]
fn prime_manifest_orders_one_plane_per_polynomial_without_a_separate_cap() {
    let source = BinarySourceDescriptor::new(
        descriptor().index_map().clone(),
        (0..6).map(|i| [i; 32]).collect(),
        descriptor().profiles().clone(),
    )
    .unwrap();
    let claim = PrimeGroupClaim {
        point: vec![2; 11],
        evaluations: (0..6)
            .rev()
            .map(|polynomial| PrimePlaneClaim {
                polynomial,
                value: polynomial as u128,
            })
            .collect(),
    };
    let checked = PrimePlaneManifest::new(&source, Some(claim.clone())).unwrap();
    assert_eq!(checked.planes().len(), 6);
    assert_eq!(
        checked
            .planes()
            .iter()
            .map(|p| p.polynomial)
            .collect::<Vec<_>>(),
        (0..6).collect::<Vec<_>>()
    );
    let mut sorted = claim;
    sorted.evaluations.reverse();
    assert_eq!(
        checked,
        PrimePlaneManifest::new(&source, Some(sorted)).unwrap()
    );
    let bytes = checked.canonical_bytes();
    let mut expected = b"AkitaPrimePlanes\x02".to_vec();
    expected.extend_from_slice(&11u64.to_le_bytes());
    for _ in 0..11 {
        expected.extend_from_slice(&2u128.to_le_bytes());
    }
    expected.extend_from_slice(&6u64.to_le_bytes());
    for i in 0..6u64 {
        expected.extend_from_slice(&i.to_le_bytes());
        expected.extend_from_slice(&u128::from(i).to_le_bytes());
    }
    assert_eq!(bytes, expected);
    let plan = BinaryRootWitnessPlan::new(&source, 3, 64).unwrap();
    assert!(plan.clone().with_prime_planes(false, 6).is_ok());
    assert!(plan.with_prime_planes(false, 7).is_err());
}

#[test]
fn prime_point_tensor_order_is_explicit() {
    let source = descriptor();
    let mut c = claim();
    c.point = vec![2; 11];
    let (rows, columns, omega) = c.weights::<F>(&source).unwrap();
    assert_eq!(columns, vec![-F::one(), F::from_u64(2)]);
    assert_eq!(omega.len(), 256);
    assert_eq!(omega[0], F::one());
    assert_eq!(omega[1], -F::from_u64(2));
    assert_eq!(
        rows,
        vec![F::one(), -F::from_u64(2), -F::from_u64(2), F::from_u64(4)]
    );
}

#[test]
fn prime_plane_relation_uses_every_response_digit_and_reduced_scalar_product() {
    let source = descriptor();
    let mut c = claim();
    // EP_0 = 9 Z, c_0 = Z^161, so V_0 = -9 - 9 Z^81 in Phi_243.
    // Coefficient equality at point (2,...,2) reads EP_0 as -18.
    c.point[..8].fill(2);
    c.evaluations[0].value = BINARY_SETUP_P - 18;
    let manifest = PrimePlaneManifest::new(&source, Some(c)).unwrap();
    let repacking = RootCoefficientRepacking::new(
        vec![
            RootRepackingSegment {
                geometry: RelationRowGeometry::for_polynomial(
                    RelationPolynomial::minus_trinomial(648).unwrap(),
                    1,
                    1024,
                )
                .unwrap(),
                elements: 2,
                digits: 4,
            },
            RootRepackingSegment {
                geometry: RelationRowGeometry::for_polynomial(
                    RelationPolynomial::plus_trinomial(162).unwrap(),
                    1,
                    256,
                )
                .unwrap(),
                elements: 2,
                digits: PRIME_PLANE_DIGITS,
            },
        ],
        64,
    )
    .unwrap();
    let mut response = vec![0i8; 2 * 4 * 648];
    for digit in 0..2 {
        response[digit * 648] = -1;
        response[digit * 648 + 324] = 1; // undo signed packing at scalar coefficient 81.
    }
    let mut planes = vec![0i8; 2 * PRIME_PLANE_DIGITS * 162];
    for digit in 0..2 {
        planes[digit * 162 + 1] = 1;
    }
    let table = repacking
        .repack(&[&response, &planes])
        .unwrap()
        .into_iter()
        .map(|x| F::from_i128(i128::from(x)))
        .collect::<Vec<_>>();
    let zero = SparseChallenge {
        positions: Default::default(),
        coeffs: Default::default(),
    };
    let challenges = Challenges::from_sparse(
        vec![
            SparseChallenge {
                positions: vec![161].into(),
                coeffs: vec![1].into(),
            },
            zero.clone(),
            zero.clone(),
            zero,
        ],
        2,
        2,
    )
    .unwrap();
    let weights = prime_plane_relation_weights(
        &source,
        &manifest,
        &repacking,
        0,
        1,
        &challenges,
        F::from_u64(17),
        &[F::from_u64(19), F::from_u64(23)],
    )
    .unwrap();
    weights.check_table(&table).unwrap();
    for address in [
        repacking.coefficient_address(0, 0, 1, 0).unwrap(),
        repacking.coefficient_address(1, 0, 1, 1).unwrap(),
    ] {
        let mut bad = table.clone();
        bad[address] += F::one();
        assert!(weights.check_table(&bad).is_err());
    }
    let mut false_claim = manifest.claim().unwrap().clone();
    false_claim.evaluations[0].value += 1;
    let false_manifest = PrimePlaneManifest::new(&source, Some(false_claim)).unwrap();
    assert!(prime_plane_relation_weights(
        &source,
        &false_manifest,
        &repacking,
        0,
        1,
        &challenges,
        F::from_u64(17),
        &[F::from_u64(19), F::from_u64(23)]
    )
    .unwrap()
    .check_table(&table)
    .is_err());
}
