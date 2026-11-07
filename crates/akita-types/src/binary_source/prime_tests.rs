use super::*;
use crate::binary_source::prime_weights::prime_plane_relation_weights;
use crate::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinaryRootWitnessPlan};
use crate::root_repacking::{RootCoefficientRepacking, RootRepackingSegment};
use akita_challenges::{Challenges, SparseChallenge};
use akita_params::binary_setup::BINARY_SETUP_P;
use akita_params::{RelationPolynomial, RelationRowGeometry};
use jolt_field::{CanonicalEncoding, One, Prime128Offset275, Ring};

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
        point: vec![0.into(); 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: 0.into(),
        }],
    }
}

#[test]
fn prime_manifest_rejects_repeated_owners_empty_values_and_noncanonical_fields() {
    let source = descriptor();
    for change in 0..7 {
        let mut bad = claim();
        match change {
            0 => bad.evaluations[0].value = BINARY_SETUP_P.into(),
            1 => bad.point[0] = BINARY_SETUP_P.into(),
            2 => bad.evaluations[0].polynomial = 2,
            3 => bad.evaluations.clear(),
            4 => bad.evaluations.push(bad.evaluations[0].clone()),
            5 => {
                bad.point.pop();
            }
            _ => bad.evaluations.push(PrimePlaneClaim {
                polynomial: 0,
                value: 1.into(),
            }),
        }
        assert!(
            matches!(
                PrimePlaneManifest::with_field_profile(
                    &source,
                    Some(bad),
                    BinaryFieldProfile::P128
                ),
                Err(AkitaError::InvalidInput(_))
            ),
            "mutation {change}"
        );
    }
    assert!(
        PrimePlaneManifest::with_field_profile(&source, None, BinaryFieldProfile::P128)
            .unwrap()
            .planes()
            .is_empty()
    );
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
        point: vec![2.into(); 11],
        evaluations: (0..6)
            .rev()
            .map(|polynomial| PrimePlaneClaim {
                polynomial,
                value: (polynomial as u128).into(),
            })
            .collect(),
    };
    let checked = PrimePlaneManifest::with_field_profile(
        &source,
        Some(claim.clone()),
        BinaryFieldProfile::P128,
    )
    .unwrap();
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
        PrimePlaneManifest::with_field_profile(&source, Some(sorted), BinaryFieldProfile::P128)
            .unwrap()
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
    assert!(
        BinaryRootWitnessPlan::for_profile(&source, 3, 64, BinaryFieldProfile::P128, false, 6)
            .is_ok()
    );
    assert!(
        BinaryRootWitnessPlan::for_profile(&source, 3, 64, BinaryFieldProfile::P128, false, 7)
            .is_err()
    );
}

#[test]
fn prime_point_tensor_order_is_explicit() {
    let source = descriptor();
    let mut c = claim();
    c.point = vec![2.into(); 11];
    let (rows, columns, omega) = c.weights::<F>(&source).unwrap().remove(0);
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
    c.point[..8].fill(2.into());
    c.evaluations[0].value = (BINARY_SETUP_P - 18).into();
    let manifest =
        PrimePlaneManifest::with_field_profile(&source, Some(c), BinaryFieldProfile::P128).unwrap();
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
                digits: manifest.field_profile().prime_plane_digits(),
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
    let mut planes = vec![0i8; 2 * manifest.field_profile().prime_plane_digits() * 162];
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
    let mut false_claim = manifest.claims().first().unwrap().clone();
    false_claim.evaluations[0].value =
        (false_claim.evaluations[0].value.coordinates()[0] + 1).into();
    let false_manifest = PrimePlaneManifest::with_field_profile(
        &source,
        Some(false_claim),
        BinaryFieldProfile::P128,
    )
    .unwrap();
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
fn p64_prime_manifest_binds_both_canonical_coordinates_and_rejects_other_versions() {
    let field = BinaryFieldProfile::P64;
    let source = descriptor();
    let mut c = claim();
    c.point.fill([field.modulus() - 1, 17].into());
    c.evaluations[0].value = [19, field.modulus() - 1].into();
    let manifest = PrimePlaneManifest::with_field_profile(&source, Some(c.clone()), field).unwrap();
    assert_eq!(manifest.planes().len(), 2);
    let mut expected = b"AkitaPrimePlanes/p64\x04".to_vec();
    expected.extend_from_slice(&11u64.to_le_bytes());
    for _ in 0..11 {
        expected.extend_from_slice(&((field.modulus() - 1) as u64).to_le_bytes());
        expected.extend_from_slice(&17u64.to_le_bytes());
    }
    expected.extend_from_slice(&1u64.to_le_bytes());
    expected.extend_from_slice(&0u64.to_le_bytes());
    expected.extend_from_slice(&19u64.to_le_bytes());
    expected.extend_from_slice(&((field.modulus() - 1) as u64).to_le_bytes());
    assert_eq!(manifest.canonical_bytes(), expected);
    assert_eq!(
        PrimePlaneManifest::from_bytes(&source, field, &expected).unwrap(),
        manifest
    );
    assert!(PrimePlaneManifest::from_bytes(&source, BinaryFieldProfile::P128, &expected).is_err());
    let p128 =
        PrimePlaneManifest::with_field_profile(&source, Some(claim()), BinaryFieldProfile::P128)
            .unwrap();
    assert_eq!(
        PrimePlaneManifest::from_bytes(&source, BinaryFieldProfile::P128, &p128.canonical_bytes())
            .unwrap(),
        p128
    );
    assert!(PrimePlaneManifest::from_bytes(&source, field, &p128.canonical_bytes()).is_err());
    for end in 0..expected.len() {
        assert!(PrimePlaneManifest::from_bytes(&source, field, &expected[..end]).is_err());
    }
    let mut trailing = expected.clone();
    trailing.push(0);
    assert!(matches!(
        PrimePlaneManifest::from_bytes(&source, field, &trailing),
        Err(akita_serialization::SerializationError::InvalidData(_))
    ));
    for offset in [
        b"AkitaPrimePlanes/p64\x04".len(),
        b"AkitaPrimePlanes/p64\x04".len() + 8 + 11 * 16,
    ] {
        let mut huge = expected.clone();
        huge[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            PrimePlaneManifest::from_bytes(&source, field, &huge),
            Err(akita_serialization::SerializationError::InvalidData(_))
        ));
    }
    for offset in [
        b"AkitaPrimePlanes/p64\x04".len() + 8,
        b"AkitaPrimePlanes/p64\x04".len() + 8 + 11 * 16 + 16,
    ] {
        for coordinate in 0..2 {
            let mut bad = manifest.canonical_bytes();
            let start = offset + 8 * coordinate;
            bad[start..start + 8].copy_from_slice(&(field.modulus() as u64).to_le_bytes());
            assert!(matches!(
                PrimePlaneManifest::from_bytes(&source, field, &bad),
                Err(akita_serialization::SerializationError::InvalidData(_))
            ));
        }
    }
    expected[b"AkitaPrimePlanes/p64".len()] = 3;
    assert!(PrimePlaneManifest::from_bytes(&source, field, &expected).is_err());
    for raw in [field.modulus(), u128::MAX, 1u128 << 64] {
        for coordinate in 0..2 {
            for point in [false, true] {
                let mut bad = c.clone();
                let mut value = [1, 2];
                value[coordinate] = raw;
                if point {
                    bad.point[0] = value.into();
                } else {
                    bad.evaluations[0].value = value.into();
                }
                assert!(matches!(
                    PrimePlaneManifest::with_field_profile(&source, Some(bad), field),
                    Err(AkitaError::InvalidInput(_))
                ));
            }
        }
    }
    assert!(
        PrimePlaneManifest::with_field_profile(&source, Some(c), BinaryFieldProfile::P128).is_err()
    );
    assert!(PrimePlaneManifest::with_field_profile(&source, Some(claim()), field).is_err());
}

#[test]
fn p64_prime_reach_plane_sizes_and_prime_only_d_rank() {
    use crate::binary_source::{binary_ordinary_profile_fixture, BinaryCommittedGroupManifest};
    use akita_params::binary_setup::BinaryFieldProfile;
    let reach = |digits: usize, digit: u128| {
        // The 43-digit negative reach exceeds u128; saturation preserves these
        // comparisons with the centered base-coordinate interval.
        (0..digits).fold(0u128, |sum, i| {
            sum.saturating_add(digit.saturating_mul(8u128.pow(i as u32)))
        })
    };
    for field in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
        let digits = field.prime_plane_digits();
        let centered = (field.modulus() - 1) / 2;
        assert!(reach(digits, 3) >= centered);
        assert!(reach(digits, 4) >= centered);
        assert!(reach(digits - 1, 3) < centered || reach(digits - 1, 4) < centered);
    }
    for planes in [2, 10] {
        let field = BinaryFieldProfile::P64;
        let index = BinaryIndexMap::new(128, 1024, 1024, 256).unwrap();
        let profiles =
            BinaryCommittedGroupManifest::profiles_for_field(&index, planes / 2, 0, field).unwrap();
        let source = BinarySourceDescriptor::new(
            index,
            (0..planes / 2).map(|i| [i as u8; 32]).collect(),
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
        assert!(BinaryRootWitnessPlan::for_profile(
            manifest.descriptor(),
            3,
            64,
            field,
            false,
            2 * planes + 2
        )
        .is_err());
        assert!(manifest.d_label(false, 2 * planes + 2).is_err());
        assert!(manifest.d_label(false, planes - 1).is_err());
        let segment = plan.prime_segment().unwrap();
        let shape = &plan.repacking().segments()[segment];
        let depth = field.prime_plane_digits();
        let columns = manifest.descriptor().index_map().columns();
        assert_eq!(shape.digits, depth);
        assert_eq!(shape.elements, planes * columns);
        assert_eq!(
            shape.elements * shape.digits * shape.geometry.polynomial_modulus_dimension(),
            planes * columns * depth * 162
        );
        assert_eq!(
            plan.repacking().segment_range(segment).unwrap().len(),
            planes * columns * depth * shape.geometry.physical_coefficient_width()
        );
        let width = (planes * depth * 162 * columns).div_ceil(64);
        crate::binary_source::root_public::reference::assert_digit_rank::<akita_params::sis::Open>(
            manifest.d_label(false, planes).unwrap(),
            width,
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
    use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing, FoldChallengeDistribution};
    let challenge = BinaryChallengeProfile::bounded_weight(
        BinaryScalarRing::Cyclotomic243,
        FoldChallengeDistribution::BinaryRoot.weight(),
    )
    .unwrap();
    let gamma = u128::from(challenge.multiplication_linf_operator_bound());
    let (low, high) = crate::binary_source::BINARY_RESPONSE_INTERVAL;
    let delta = (i128::from(high) - i128::from(low)) as u128;
    assert!(2 * gamma + delta < u128::from(BINARY_SETUP_P64));
}

#[test]
fn p64_prime_structured_and_reference_weights_match_over_e64() {
    use akita_params::binary_setup::BinaryFieldProfile;
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base, Zero};
    type E = Ext2<Base>;
    let field = BinaryFieldProfile::P64;
    let source = descriptor();
    let mut c = claim();
    c.point.fill([0, 0].into());
    c.point[..8].fill([2, 1].into());
    let x = E::from_base_fn(|i| Base::from_u64(if i == 0 { 2 } else { 1 }));
    let value = E::from_u64(9) * x * (0..7).fold(E::one(), |v, _| v * (E::one() - x));
    c.evaluations[0].value = [
        value.base_coefficient(0).to_u128_checked().unwrap(),
        value.base_coefficient(1).to_u128_checked().unwrap(),
    ]
    .into();
    let manifest = PrimePlaneManifest::with_field_profile(&source, Some(c), field).unwrap();
    let plan = BinaryRootWitnessPlan::for_profile(&source, 3, 64, field, false, 2).unwrap();
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
        point: vec![2.into(); 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: 3.into(),
        }],
    };
    let setup = crate::AkitaSetupDescriptor {
        max_num_vars: 0,
        max_num_batched_polys: 0,
        num_field_elements: crate::MAX_GENERIC_SETUP_DECODE_FIELD_ELEMENTS,
        setup_seed: [41; 32].into(),
    };
    let legacy =
        PrimePlaneManifest::with_field_profile(source, Some(c.clone()), BinaryFieldProfile::P128)
            .unwrap();
    assert!(matches!(
        BinaryRootPublic::new(&manifest, None, &legacy, &setup, 1 << 17, 64, None),
        Err(AkitaError::InvalidInput(_))
    ));
    let mut c = c;
    c.point.fill([2, 1].into());
    c.evaluations[0].value = [3, 1].into();
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

#[test]
fn p64_native_tensors_and_readings_match_naive_evaluation() {
    use jolt_field::{Ext2, ExtField, Prime64Offset59 as Base, Zero};
    type E = Ext2<Base>;
    let source = descriptor();
    let point = (0..11)
        .map(|i| E::from_base_fn(|c| Base::from_u64((19 + 7 * i + 13 * c) as u64)))
        .collect::<Vec<_>>();
    // Direct products of equality factors, without the tensor constructor.
    let naive = |point: &[E], index: usize| {
        point
            .iter()
            .enumerate()
            .fold(E::one(), |weight, (bit, &x)| {
                weight
                    * if index >> bit & 1 == 0 {
                        E::one() - x
                    } else {
                        x
                    }
            })
    };
    let coefficient =
        |row: usize, column: usize, k: usize| ((row * 17 + column * 11 + k * 3) % 5) as i128 - 2;
    let mut value = E::zero();
    for row in 0..4 {
        for column in 0..2 {
            for k in 0..162 {
                value += naive(&point, k + 256 * (column + 2 * row))
                    * E::from_i128(coefficient(row, column, k));
            }
        }
    }
    assert_ne!(value.base_coefficient(1), Base::zero());
    let coordinates =
        |v: E| std::array::from_fn::<_, 2, _>(|c| v.base_coefficient(c).to_u128_checked().unwrap());
    let claim = PrimeGroupClaim {
        point: point
            .iter()
            .copied()
            .map(|v| coordinates(v).into())
            .collect(),
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: coordinates(value).into(),
        }],
    };
    let tensors = claim.weights::<Base>(&source).unwrap();
    for (c, (rows, columns, omega)) in tensors.iter().enumerate() {
        for (weights, axes) in [
            (rows, &point[9..]),
            (columns, &point[8..9]),
            (omega, &point[..8]),
        ] {
            for (index, &weight) in weights.iter().enumerate() {
                assert_eq!(weight, naive(axes, index).base_coefficient(c));
            }
        }
    }
    let manifest =
        PrimePlaneManifest::with_field_profile(&source, Some(claim), BinaryFieldProfile::P64)
            .unwrap();
    let plan =
        BinaryRootWitnessPlan::for_profile(&source, 3, 64, BinaryFieldProfile::P64, false, 2)
            .unwrap();
    let repacking = plan.repacking();
    let segment = plan.prime_segment().unwrap();
    let mut natural = repacking
        .segments()
        .iter()
        .map(|s| vec![0i8; s.elements * s.digits * s.geometry.polynomial_modulus_dimension()])
        .collect::<Vec<_>>();
    let digits = |mut integer: i128, depth: usize| {
        let mut digits = Vec::new();
        for _ in 0..depth {
            let remainder = integer.rem_euclid(8);
            let digit = if remainder >= 4 {
                remainder - 8
            } else {
                remainder
            };
            digits.push(digit as i8);
            integer = (integer - digit) / 8;
        }
        assert_eq!(integer, 0);
        digits
    };
    for row in 0..4 {
        for k in 0..162 {
            let packed =
                (coefficient(row, 0, k) + coefficient(row, 1, k)) * if k % 2 == 0 { 1 } else { -1 };
            for (digit, value) in digits(packed, 4).into_iter().enumerate() {
                natural[0][digit * 648 + 4 * k + row] = value;
            }
        }
    }
    let modulus = BinaryFieldProfile::P64.modulus();
    for column in 0..2 {
        for k in 0..162 {
            let partial = (0..4).fold(E::zero(), |sum, row| {
                sum + naive(&point[9..], row) * E::from_i128(coefficient(row, column, k))
            });
            for (c, canonical) in coordinates(partial).into_iter().enumerate() {
                let centered = if canonical > modulus / 2 {
                    -((modulus - canonical) as i128)
                } else {
                    canonical as i128
                };
                for (digit, value) in digits(centered, 22).into_iter().enumerate() {
                    natural[segment][((c * 2 + column) * 22 + digit) * 162 + k] = value;
                }
            }
        }
    }
    let table = repacking
        .repack(&natural.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap()
        .into_iter()
        .map(|v| E::from_i128(i128::from(v)))
        .collect::<Vec<_>>();
    let challenges = Challenges::from_sparse(
        (0..4)
            .map(|_| SparseChallenge {
                positions: vec![0].into(),
                coeffs: vec![1].into(),
            })
            .collect(),
        2,
        2,
    )
    .unwrap();
    let ext = |i: usize| E::from_base_fn(|c| Base::from_u64((17 + 31 * i + 43 * c) as u64));
    let batches = [ext(1), ext(2)];
    let weights = prime_plane_relation_weights(
        &source,
        &manifest,
        repacking,
        0,
        segment,
        &challenges,
        ext(0),
        &batches,
    )
    .unwrap();
    weights.check_table(&table).unwrap();
    assert_eq!(weights.claim(), batches[1] * value);
    for isolated in [[batches[0], E::zero()], [E::zero(), batches[1]]] {
        prime_plane_relation_weights(
            &source,
            &manifest,
            repacking,
            0,
            segment,
            &challenges,
            ext(0),
            &isolated,
        )
        .unwrap()
        .check_table(&table)
        .unwrap();
    }
    // Physical coordinate planes do not each introduce two independent rows.
    assert!(matches!(
        prime_plane_relation_weights(
            &source,
            &manifest,
            repacking,
            0,
            segment,
            &challenges,
            ext(0),
            &[ext(1), ext(2), ext(3), ext(4)],
        ),
        Err(AkitaError::InvalidProof)
    ));
    let dense = weights.materialize().unwrap();
    for c in 0..2 {
        let basis = E::from_base_fn(|i| Base::from_bool(i == c));
        for column in 0..2 {
            for k in 0..162 {
                let product = basis * naive(&point[8..9], column) * naive(&point[..8], k);
                let reading = product * batches[1];
                let alpha_power = (0..k).fold(E::one(), |power, _| power * ext(0));
                let address = repacking
                    .coefficient_address(segment, c * 2 + column, 0, k)
                    .unwrap();
                assert_eq!(dense[address], reading - basis * batches[0] * alpha_power);
            }
        }
    }
    let reference = crate::binary_source::root_public::reference::prime_plane_weights(
        &source,
        &manifest,
        repacking,
        0,
        segment,
        &challenges,
        ext(0),
        &batches,
    )
    .unwrap();
    assert_eq!(reference.materialize().unwrap(), dense);
    assert_eq!(reference.claim(), weights.claim());
}

#[test]
fn p64_prime_compilers_reject_a_different_quadratic_basis() {
    use jolt_field::{Ext2Config, ExtField, FpExt2, Prime64Offset59 as Base, Zero};

    struct EightNr;
    impl Ext2Config<Base> for EightNr {
        fn non_residue() -> Base {
            Base::from_u64(8)
        }
    }
    type E = FpExt2<Base, EightNr>;

    // Euler's criterion establishes that this is another valid quadratic field.
    let exponent = (BinaryFieldProfile::P64.modulus() - 1) / 2;
    let legendre = (0..64).rev().fold(Base::one(), |power, bit| {
        let square = power.square();
        if exponent >> bit & 1 == 1 {
            square * EightNr::non_residue()
        } else {
            square
        }
    });
    assert_eq!(legendre, -Base::one());
    assert_eq!(E::DEGREE, 2);
    assert_eq!(E::from_u128(BinaryFieldProfile::P64.modulus()), E::zero());
    let basis = E::from_base_fn(|i| Base::from_bool(i == 1));
    assert_eq!(basis.square(), E::from_u64(8));

    let source = descriptor();
    let claim = PrimeGroupClaim {
        point: vec![[2, 1].into(); 11],
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: [3, 5].into(),
        }],
    };
    assert!(claim.field_weights::<Base, super::E64>(&source).is_ok());
    assert!(matches!(
        claim.field_weights::<Base, E>(&source),
        Err(AkitaError::InvalidProof)
    ));
    let manifest =
        PrimePlaneManifest::with_field_profile(&source, Some(claim), BinaryFieldProfile::P64)
            .unwrap();
    let plan =
        BinaryRootWitnessPlan::for_profile(&source, 3, 64, BinaryFieldProfile::P64, false, 2)
            .unwrap();
    let challenges = Challenges::from_sparse(
        (0..4)
            .map(|_| SparseChallenge {
                positions: vec![0].into(),
                coeffs: vec![1].into(),
            })
            .collect(),
        2,
        2,
    )
    .unwrap();
    let alpha = E::one() + basis;
    let batches = [E::one(); 2];
    assert!(matches!(
        prime_plane_relation_weights(
            &source,
            &manifest,
            plan.repacking(),
            0,
            plan.prime_segment().unwrap(),
            &challenges,
            alpha,
            &batches,
        ),
        Err(AkitaError::InvalidProof)
    ));
    assert!(matches!(
        crate::binary_source::root_public::reference::prime_plane_weights(
            &source,
            &manifest,
            plan.repacking(),
            0,
            plan.prime_segment().unwrap(),
            &challenges,
            alpha,
            &batches,
        ),
        Err(AkitaError::InvalidProof)
    ));
}

#[test]
fn p64_two_points_bind_order_and_match_reference_weights() {
    use jolt_field::{ExtField, Prime64Offset59 as Base, Zero};
    type E = Ext2<Base>;
    let source = descriptor();
    let field = BinaryFieldProfile::P64;
    let claims = (0..2).map(|point| PrimeGroupClaim {
        point: (0..11).map(|axis| [2 + axis, 3 + point].into()).collect(),
        evaluations: vec![PrimePlaneClaim {
            polynomial: 0,
            value: [7 + point, 11 + point].into(),
        }],
    }).collect::<Vec<_>>();
    let manifest = PrimePlaneManifest::with_field_profile(&source, claims.clone(), field).unwrap();
    assert_eq!(manifest.planes().len(), 4);
    assert_eq!(manifest.planes().iter().map(|p| (p.point, p.polynomial, p.coordinate))
        .collect::<Vec<_>>(), vec![(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]);
    let bytes = manifest.canonical_bytes();
    assert!(bytes.starts_with(b"AkitaPrimePlanes/p64\x05"));
    assert_eq!(PrimePlaneManifest::from_bytes(&source, field, &bytes).unwrap(), manifest);
    for end in 0..bytes.len() {
        assert!(PrimePlaneManifest::from_bytes(&source, field, &bytes[..end]).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(PrimePlaneManifest::from_bytes(&source, field, &trailing).is_err());
    let mut reversed = claims.clone();
    reversed.reverse();
    assert_ne!(bytes, PrimePlaneManifest::with_field_profile(&source, reversed, field)
        .unwrap().canonical_bytes());
    assert!(PrimePlaneManifest::with_field_profile(
        &source, [claims[0].clone(), claims[0].clone()], field,
    ).is_err());
    assert!(PrimePlaneManifest::with_field_profile(
        &source, claims.iter().cloned().chain([claims[0].clone()]), field,
    ).is_err());
    let base = claim();
    assert!(PrimePlaneManifest::with_field_profile(
        &source, [base.clone(), base], BinaryFieldProfile::P128,
    ).is_err());
    let mut short = claims.clone();
    short[1].point.pop();
    assert!(PrimePlaneManifest::with_field_profile(&source, short, field).is_err());
    let mut duplicate_owner = claims.clone();
    let value = duplicate_owner[1].evaluations[0].clone();
    duplicate_owner[1].evaluations.push(value);
    assert!(PrimePlaneManifest::with_field_profile(&source, duplicate_owner, field).is_err());
    for point in 0..2 {
        for coordinate in 0..2 {
            let mut changed = claims.clone();
            let PrimeClaimValue::Extension(value) = &mut changed[point].evaluations[0].value
                else { unreachable!() };
            value[coordinate] = field.modulus();
            assert!(PrimePlaneManifest::with_field_profile(&source, changed, field).is_err());
        }
    }
    let plan = BinaryRootWitnessPlan::for_profile(&source, 3, 64, field, false, 4).unwrap();
    let repacking = plan.repacking();
    let segment = plan.prime_segment().unwrap();
    let challenges = Challenges::from_sparse(
        (0..4).map(|_| SparseChallenge {
            positions: vec![0, 161].into(), coeffs: vec![1, -1].into(),
        }).collect(), 2, 2,
    ).unwrap();
    let ext = |i: usize| E::from_base_fn(|c| Base::from_u64((17 + i * 31 + c * 43) as u64));
    let batches = [ext(1), ext(2), ext(3), ext(4)];
    let compiled = prime_plane_relation_weights(
        &source, &manifest, repacking, 0, segment, &challenges, ext(0), &batches,
    ).unwrap();
    let reference = crate::binary_source::root_public::reference::prime_plane_weights(
        &source, &manifest, repacking, 0, segment, &challenges, ext(0), &batches,
    ).unwrap();
    assert_eq!(compiled.claim(), reference.claim());
    assert_eq!(compiled.materialize().unwrap(), reference.materialize().unwrap());
    let evaluation_point = (0..repacking.domain_len().trailing_zeros())
        .map(|i| ext(20 + i as usize)).collect::<Vec<_>>();
    assert_eq!(compiled.evaluate(&evaluation_point).unwrap(),
        reference.evaluate(&evaluation_point).unwrap());
    // Each point keeps two E64 identities, independent of physical plane count.
    assert!(prime_plane_relation_weights(
        &source, &manifest, repacking, 0, segment, &challenges, ext(0), &[E::zero(); 8],
    ).is_err());
}

#[test]
fn p64_v5_literal_and_malformed_records() {
    use akita_serialization::SerializationError;
    let source = descriptor();
    let field = BinaryFieldProfile::P64;
    let claims = (0..2).map(|point| PrimeGroupClaim {
        point: vec![[2, 3 + point].into(); 11],
        evaluations: vec![
            PrimePlaneClaim { polynomial: 0, value: [5, 7].into() },
            PrimePlaneClaim { polynomial: 1, value: [11, 13].into() },
        ],
    }).collect::<Vec<_>>();
    let manifest = PrimePlaneManifest::with_field_profile(&source, claims, field).unwrap();
    // Literal layout, independent of canonical_bytes: no leading record count.
    let mut literal = b"AkitaPrimePlanes/p64\x05".to_vec();
    for second_coordinate in [3u64, 4] {
        literal.extend_from_slice(&11u64.to_le_bytes());
        for _ in 0..11 {
            literal.extend_from_slice(&2u64.to_le_bytes());
            literal.extend_from_slice(&second_coordinate.to_le_bytes());
        }
        for word in [2u64, 0, 5, 7, 1, 11, 13] {
            literal.extend_from_slice(&word.to_le_bytes());
        }
    }
    assert_eq!(manifest.canonical_bytes(), literal);
    assert_eq!(PrimePlaneManifest::from_bytes(&source, field, &literal).unwrap(), manifest);
    let reject = |bytes: &[u8], profile, truncated: bool| {
        let result = std::panic::catch_unwind(|| {
            PrimePlaneManifest::from_bytes(&source, profile, bytes)
        }).expect("malformed v5 bytes must not panic");
        match result {
            Err(SerializationError::InvalidData(_)) => (),
            Err(SerializationError::IoError(error)) if truncated => {
                assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
            }
            other => panic!("expected typed v5 rejection, got {other:?}"),
        }
    };
    let tag = b"AkitaPrimePlanes/p64\x05".len();
    let record = 8 + 11 * 16 + 8 + 2 * 24;
    for point in 0..2 {
        let start = tag + point * record;
        let values = start + 8 + 11 * 16;
        // Huge, zero and wrong per-record counts, before allocation.
        for (offset, counts) in [(start, [u64::MAX, 0, 10]), (values, [u64::MAX, 0, 3])] {
            for count in counts {
                let mut bytes = literal.clone();
                bytes[offset..offset + 8].copy_from_slice(&count.to_le_bytes());
                reject(&bytes, field, false);
            }
        }
        // A smaller value count shifts the next record boundary or leaves trailing data.
        let mut bytes = literal.clone();
        bytes[values..values + 8].copy_from_slice(&1u64.to_le_bytes());
        reject(&bytes, field, false);
        for owner in [0u64, 2, u64::MAX] {
            let mut bytes = literal.clone();
            let offset = values + 8 + 24;
            bytes[offset..offset + 8].copy_from_slice(&owner.to_le_bytes());
            reject(&bytes, field, false);
        }
        for offset in [start + 8, start + 16, values + 16, values + 24,
            values + 40, values + 48]
        {
            let mut bytes = literal.clone();
            bytes[offset..offset + 8].copy_from_slice(&(field.modulus() as u64).to_le_bytes());
            reject(&bytes, field, false);
        }
        let mut bytes = literal.clone();
        for byte in 0..24 {
            bytes.swap(values + 8 + byte, values + 32 + byte);
        }
        reject(&bytes, field, false); // valid owners, noncanonical order
    }
    let mut duplicate = literal.clone();
    duplicate.copy_within(tag..tag + record, tag + record);
    reject(&duplicate, field, false);
    reject(&literal, BinaryFieldProfile::P128, false);
    for version in [0, 2, 4, 6] {
        let mut bytes = literal.clone();
        bytes[tag - 1] = version;
        reject(&bytes, field, false);
    }
    let mut wrong_tag = literal.clone();
    wrong_tag[0] ^= 1;
    reject(&wrong_tag, field, false);
    // Includes missing second record, and every partial record/limb boundary.
    for end in 0..literal.len() {
        reject(&literal[..end], field, true);
    }
    let mut extra = literal.clone();
    extra.extend_from_slice(&literal[tag..tag + record]);
    reject(&extra, field, false);
    let mut trailing = literal;
    trailing.push(0);
    reject(&trailing, field, false);
}
