use super::*;
use crate::binary_source::prime_weights::prime_plane_relation_weights;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinaryRootWitnessPlan};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use akita_challenges::{Challenges, SparseChallenge};
use akita_params::binary_setup::BINARY_SETUP_P;
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

#[test]
fn p64_prime_manifest_uses_base_residues_and_eight_byte_coordinates() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base, Zero};
    let field = BinaryFieldProfile::P64;
    let source = descriptor();
    let mut c = claim();
    c.point.fill(field.modulus() - 1);
    c.evaluations[0].value = field.modulus() - 1;
    let manifest = PrimePlaneManifest::with_field_profile(&source, Some(c.clone()), field).unwrap();
    let mut expected = b"AkitaPrimePlanes/p64\x03".to_vec();
    expected.extend_from_slice(&11u64.to_le_bytes());
    for _ in 0..11 {
        expected.extend_from_slice(&((field.modulus() - 1) as u64).to_le_bytes());
    }
    expected.extend_from_slice(&1u64.to_le_bytes());
    expected.extend_from_slice(&0u64.to_le_bytes());
    expected.extend_from_slice(&((field.modulus() - 1) as u64).to_le_bytes());
    assert_eq!(manifest.canonical_bytes(), expected);
    assert!(c.weights::<Ext2<Base>>(&source).is_ok());
    for raw in [field.modulus(), u128::MAX, 1u128 << 64] {
        for point in [false, true] {
            let mut bad = c.clone();
            if point {
                bad.point[0] = raw;
            } else {
                bad.evaluations[0].value = raw;
            }
            assert!(matches!(
                PrimePlaneManifest::with_field_profile(&source, Some(bad), field),
                Err(AkitaError::InvalidInput(_))
            ));
        }
    }
    // An E64 point or value cannot be converted to this base-residue API.
    let extension = Ext2::<Base>::from_base_fn(|i| if i == 0 { Base::zero() } else { Base::one() });
    let base = crate::field_reduction::FpExtEncoding::<Base>::degree_one_base(&extension)
        .ok_or_else(|| AkitaError::InvalidInput("prime claims require base residues".into()));
    assert!(matches!(base, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn p64_prime_reach_plane_sizes_and_prime_only_d_rank() {
    use crate::binary_source::{binary_ordinary_profile_fixture, BinaryCommittedGroupManifest};
    use akita_params::binary_setup::{BinaryFieldProfile, BINARY_SETUP_P64};
    let reach = |digits: u32| 3 * (8u128.pow(digits) - 1) / 7;
    assert_eq!(reach(21), 3_952_873_730_080_618_203);
    assert_eq!(reach(22), 31_622_989_840_644_945_627);
    assert_eq!(4 * (8u128.pow(22) - 1) / 7, 42_163_986_454_193_260_836);
    assert_eq!(BINARY_SETUP_P64 / 2, 9_223_372_036_854_775_778);
    assert!(reach(21) < u128::from(BINARY_SETUP_P64 / 2));
    assert!(reach(22) > u128::from(BINARY_SETUP_P64));
    for planes in [1, 5] {
        let field = BinaryFieldProfile::P64;
        let index = BinaryIndexMap::new(128, 1024, 1024, 256).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::profiles_for_field(&index, planes, 0, field).unwrap();
        let source = BinarySourceDescriptor::new(
            index,
            (0..planes).map(|i| [i as u8; 32]).collect(),
            profiles,
        )
        .unwrap();
        let manifest = BinaryCommittedGroupManifest::with_field_profile(
            source,
            0,
            &binary_ordinary_profile_fixture(field),
            field,
        )
        .unwrap();
        let plan =
            BinaryRootWitnessPlan::for_profile(manifest.descriptor(), 3, 64, field, false, planes)
                .unwrap();
        let segment = plan.prime_segment().unwrap();
        let shape = &plan.repacking().segments()[segment];
        assert_eq!(shape.digits, 22);
        assert_eq!(shape.elements * shape.digits * 162, planes * 912_384);
        assert_eq!(
            plan.repacking().segment_range(segment).unwrap().len(),
            planes * 1_441_792
        );
        let columns = (planes * 22 * 162 * 256).div_ceil(64);
        assert_eq!(
            manifest.d_label(false, planes).unwrap().rank as usize,
            field
                .digit_commitment_rank(
                    akita_params::binary_setup::SetupStreamRole::WitnessCommitment,
                    columns
                )
                .unwrap()
        );
    }
}

#[test]
fn p64_prime_units_and_agreement_lift_are_exact() {
    use akita_params::binary_setup::BINARY_SETUP_P64;
    let pow = |mut base: u64, mut exponent: u32| {
        let mut value = 1u64;
        while exponent != 0 {
            if exponent & 1 != 0 {
                value = value * base % 243;
            }
            base = base * base % 243;
            exponent >>= 1;
        }
        value
    };
    let residue = BINARY_SETUP_P64 % 243;
    assert_eq!(pow(residue, 162), 1);
    // 2 and 3 are the only prime divisors of 162.
    assert_ne!(pow(residue, 81), 1);
    assert_ne!(pow(residue, 54), 1);
    assert_eq!(2 * 92 + 4095, 4279);
    const { assert!(4279 < BINARY_SETUP_P64) };
}

#[test]
fn p64_prime_structured_and_reference_weights_match_over_e64() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base, Zero};
    type E = Ext2<Base>;
    let field = BinaryFieldProfile::P64;
    let source = descriptor();
    let mut c = claim();
    c.point[..8].fill(2);
    c.evaluations[0].value = field.modulus() - 18;
    let manifest = PrimePlaneManifest::with_field_profile(&source, Some(c), field).unwrap();
    let plan = BinaryRootWitnessPlan::for_profile(&source, 3, 64, field, false, 1).unwrap();
    let repacking = plan.repacking();
    let segment = plan.prime_segment().unwrap();
    let mut natural = repacking
        .segments()
        .iter()
        .map(|shape| {
            vec![0i8; shape.elements * shape.digits * shape.geometry.polynomial_modulus_dimension()]
        })
        .collect::<Vec<_>>();
    for digit in 0..2 {
        natural[0][digit * 648] = -1;
        natural[0][digit * 648 + 324] = 1;
        natural[segment][digit * 162 + 1] = 1;
    }
    let table = repacking
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap()
        .into_iter()
        .map(|v| E::from_i64(i64::from(v)))
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
    let ext = |i: usize| E::from_base_fn(|c| Base::from_u64((17 + i * 31 + c * 43) as u64));
    let weights = prime_plane_relation_weights(
        &source,
        &manifest,
        repacking,
        0,
        segment,
        &challenges,
        ext(0),
        &[ext(1), ext(2)],
    )
    .unwrap();
    weights.check_table(&table).unwrap();
    let reference = crate::binary_source::root_public::reference::prime_plane_weights(
        &source,
        &manifest,
        repacking,
        0,
        segment,
        &challenges,
        ext(0),
        &[ext(1), ext(2)],
    )
    .unwrap();
    let mut dense = vec![E::zero(); repacking.domain_len()];
    for &(address, weight) in reference.terms() {
        dense[address] += weight;
    }
    assert_eq!(weights.claim(), reference.claim());
    assert_eq!(weights.materialize().unwrap(), dense);
    let r = (0..repacking.domain_len().trailing_zeros())
        .map(|i| ext(500 + i as usize))
        .collect::<Vec<_>>();
    assert_eq!(
        weights.evaluate(&r).unwrap(),
        reference.evaluate(&r).unwrap()
    );
    let address = repacking.coefficient_address(segment, 0, 21, 0).unwrap();
    let mut bad = table;
    bad[address] += E::one();
    assert!(matches!(
        weights.check_table(&bad),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn p64_root_rejects_legacy_prime_and_mixed_profile_identities() {
    use crate::binary_source::{
        binary_ordinary_profile_fixture, binary_word_manifest_fixture, BinaryRootPublic,
    };
    use akita_params::binary_setup::BinaryFieldProfile;
    let field = BinaryFieldProfile::P64;
    let manifest = binary_word_manifest_fixture(field, 1);
    let source = manifest.descriptor();
    let c = PrimeGroupClaim {
        point: vec![2; 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: 3,
        }],
    };
    let setup = crate::AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: [41; 32].into(),
    };
    let legacy = PrimePlaneManifest::new(source, Some(c.clone())).unwrap();
    assert!(matches!(
        BinaryRootPublic::new(&manifest, None, &legacy, &setup, 1 << 17, 64, None),
        Err(AkitaError::InvalidInput(_))
    ));
    let prime = PrimePlaneManifest::with_field_profile(source, Some(c), field).unwrap();
    assert!(matches!(
        BinaryRootPublic::new(
            &manifest,
            None,
            &prime,
            &setup,
            1 << 17,
            64,
            Some(binary_ordinary_profile_fixture(BinaryFieldProfile::P128))
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}
