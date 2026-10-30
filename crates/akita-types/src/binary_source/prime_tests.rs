use super::*;
use crate::binary_source::prime_weights::prime_plane_relation_weights;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap};
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

fn claim() -> PrimePlaneClaim {
    PrimePlaneClaim {
        polynomial: 0,
        point: vec![0; 11],
        tensor_component: 0,
        value: 0,
    }
}

#[test]
fn prime_plane_manifest_rejects_caps_conflicts_and_noncanonical_fields() {
    let source = descriptor();
    let base = claim();
    let one = PrimePlaneManifest::new(&source, std::slice::from_ref(&base)).unwrap();
    let duplicates = PrimePlaneManifest::new(&source, &[base.clone(), base.clone()]).unwrap();
    assert_eq!(duplicates.planes(), one.planes());
    assert_eq!(duplicates.canonical_bytes(), one.canonical_bytes());
    let mut other = base.clone();
    other.value = 1;
    assert!(matches!(
        PrimePlaneManifest::new(&source, &[base.clone(), other]),
        Err(AkitaError::InvalidInput(_))
    ));
    for change in 0..4 {
        let mut bad = base.clone();
        match change {
            0 => bad.value = BINARY_SETUP_P,
            1 => bad.point[0] = BINARY_SETUP_P,
            2 => bad.tensor_component = 4,
            _ => bad.polynomial = 2,
        }
        assert!(matches!(
            PrimePlaneManifest::new(&source, &[bad]),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    let six_points = (0..6)
        .map(|x| {
            let mut c = base.clone();
            c.point[0] = x;
            c
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        PrimePlaneManifest::new(&source, &six_points),
        Err(AkitaError::InvalidInput(_))
    ));
    let six_components = (0..6)
        .map(|x| {
            let mut c = base.clone();
            c.polynomial = x / 4;
            c.tensor_component = (x % 4) as u8;
            c
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        PrimePlaneManifest::new(&source, &six_components),
        Err(AkitaError::InvalidInput(_))
    ));
    let mut second_owner = base.clone();
    second_owner.polynomial = 1;
    assert_eq!(
        PrimePlaneManifest::new(&source, &[base, second_owner])
            .unwrap()
            .planes()
            .len(),
        2
    );
}

#[test]
fn prime_plane_component_and_point_tensor_order_are_explicit() {
    let source = descriptor();
    for component in 0..4 {
        let mut c = claim();
        c.point = vec![2; 11];
        c.tensor_component = component;
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
}

#[test]
fn prime_plane_relation_uses_every_response_digit_and_reduced_scalar_product() {
    let source = descriptor();
    let mut c = claim();
    // EP_0 = 9 Z, c_0 = Z^161, so V_0 = -9 - 9 Z^81 in Phi_243.
    // Coefficient equality at point (2,...,2) reads EP_0 as -18.
    c.point[..8].fill(2);
    c.value = BINARY_SETUP_P - 18;
    let manifest = PrimePlaneManifest::new(&source, &[c]).unwrap();
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
    let mut false_claim = manifest.planes()[0].clone();
    false_claim.value += 1;
    let false_manifest = PrimePlaneManifest::new(&source, &[false_claim]).unwrap();
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
