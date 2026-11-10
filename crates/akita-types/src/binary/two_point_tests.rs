use super::*;
use crate::binary::{
    binary_ordinary_profile_fixture, BinaryCommitmentProfiles, BinaryIndexMap, PrimeClaimValue,
    PrimePlaneClaim, PrimePlaneManifest,
};
use akita_params::sis::OuterCommitMatrixParams;
use akita_params::PolynomialGroupLayout;

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

fn ordinary_profile(owners: usize) -> GroupCommitPhaseParams {
    let mut profile = binary_ordinary_profile_fixture(BinaryFieldProfile::P64);
    profile.group = PolynomialGroupLayout::new(10, owners);
    let width = profile
        .derive_slice_geometry()
        .unwrap()
        .physical_input_width();
    profile.outer.matrix =
        OuterCommitMatrixParams::try_new_with_min_rank(profile.outer.matrix.sis_table_key(), width)
            .unwrap();
    profile.validate_frozen_precommit(64).unwrap();
    profile
}

fn points(arity: usize, owners: usize) -> [PrimeGroupClaim; 2] {
    std::array::from_fn(|point| PrimeGroupClaim {
        point: (0..arity)
            .map(|axis| [2 + point as u128 * 17 + axis as u128, 3 + axis as u128].into())
            .collect(),
        evaluations: (0..owners)
            .map(|polynomial| PrimePlaneClaim {
                polynomial,
                value: [41 + point as u128 * 11 + polynomial as u128, 53].into(),
            })
            .collect(),
    })
}

fn group(source: TwoPointClaimSource<'_>, points: [PrimeGroupClaim; 2]) -> TwoPointGroupClaim {
    match source {
        TwoPointClaimSource::Binary(descriptor) => TwoPointGroupClaim::Binary {
            word: BinaryWordClaim::new(descriptor, [1; 32], vec![[7; 16]; 3], [9; 16]).unwrap(),
            points,
        },
        TwoPointClaimSource::Ordinary(_) => TwoPointGroupClaim::Ordinary { points },
    }
}

fn decode_error(source: TwoPointClaimSource<'_>, bytes: &[u8]) {
    assert!(matches!(
        TwoPointClaimManifest::from_bytes(source, bytes),
        Err(SerializationError::InvalidData(_) | SerializationError::IoError(_))
    ));
}

#[test]
fn two_distinct_points_each_have_two_coordinates_and_independent_byte_recipes() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for (source, arity, tag) in [
        (
            TwoPointClaimSource::Binary(&descriptor),
            11,
            b"AkitaBinaryPoints/p64\x01".as_slice(),
        ),
        (
            TwoPointClaimSource::Ordinary(&profile),
            10,
            b"AkitaOrdinaryPoints/p64\x01".as_slice(),
        ),
    ] {
        let input = points(arity, 2);
        let manifest = TwoPointClaimManifest::new(source, group(source, input.clone())).unwrap();
        let planes: Vec<_> = manifest
            .planes()
            .map(|plane| (plane.point, plane.polynomial, plane.coordinate, plane.value))
            .collect();
        assert_eq!(
            planes,
            vec![
                (0, 0, 0, 41),
                (0, 0, 1, 53),
                (0, 1, 0, 42),
                (0, 1, 1, 53),
                (1, 0, 0, 52),
                (1, 0, 1, 53),
                (1, 1, 0, 53),
                (1, 1, 1, 53),
            ]
        );
        let mut expected = tag.to_vec();
        if matches!(source, TwoPointClaimSource::Binary(_)) {
            expected.extend_from_slice(b"AkitaBinaryWord\x01");
            expected.extend_from_slice(&[1; 32]);
            expected.extend_from_slice(&3u64.to_le_bytes());
            expected.extend_from_slice(&[7; 48]);
            expected.extend_from_slice(&[9; 16]);
        }
        for point in 0..2u64 {
            let mut words = vec![arity as u64];
            for axis in 0..arity as u64 {
                words.extend([2 + point * 17 + axis, 3 + axis]);
            }
            words.extend([2, 0, 41 + point * 11, 53, 1, 42 + point * 11, 53]);
            expected.extend(words.into_iter().flat_map(u64::to_le_bytes));
        }
        assert_eq!(manifest.canonical_bytes(), expected);
        assert_eq!(manifest.claim().points(), &input);
        assert_eq!(
            TwoPointClaimManifest::from_bytes(source, &expected).unwrap(),
            manifest
        );
    }
}

#[test]
fn point_order_is_bound_while_owner_order_is_canonical() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let input = points(arity, owners);
        let manifest = TwoPointClaimManifest::new(source, group(source, input.clone())).unwrap();
        let [first, second] = input.clone();
        let reversed = TwoPointClaimManifest::new(source, group(source, [second, first])).unwrap();
        assert_ne!(manifest.canonical_bytes(), reversed.canonical_bytes());
        assert_eq!(
            TwoPointClaimManifest::from_bytes(source, &reversed.canonical_bytes()).unwrap(),
            reversed
        );
        let mut shuffled = input;
        for point in &mut shuffled {
            point.evaluations.reverse();
        }
        assert_eq!(
            TwoPointClaimManifest::new(source, group(source, shuffled)).unwrap(),
            manifest
        );
        let record_size = 16 + 16 * arity + 24 * owners;
        let bytes = manifest.canonical_bytes();
        let start = bytes.len() - 2 * record_size;
        for point in 0..2 {
            let mut bad = bytes.clone();
            let values = start + point * record_size + 16 + 16 * arity;
            let (prefix, suffix) = bad.split_at_mut(values + 24);
            prefix[values..].swap_with_slice(&mut suffix[..24]);
            decode_error(source, &bad);
        }
    }
}

#[test]
fn duplicate_points_owner_pairs_and_malformed_in_memory_lengths_reject() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let input = points(arity, owners);
        let [first, _] = input.clone();
        assert!(TwoPointClaimManifest::new(source, group(source, [first.clone(), first])).is_err());
        for point in 0..2 {
            for mutation in 0..8 {
                let mut bad = input.clone();
                match mutation {
                    0 => {
                        bad[point].point.pop();
                    }
                    1 => bad[point].point.push([1, 2].into()),
                    2 => bad[point].evaluations.clear(),
                    3 => bad[point].evaluations[1].polynomial = 0,
                    4 => bad[point].evaluations[0].polynomial = owners,
                    5 => bad[point].point[0] = 0.into(),
                    6 => bad[point].evaluations[0].value = 0.into(),
                    _ => {
                        let duplicate = bad[point].evaluations[0].clone();
                        bad[point].evaluations.push(duplicate);
                    }
                }
                assert!(matches!(
                    TwoPointClaimManifest::new(source, group(source, bad)),
                    Err(AkitaError::InvalidInput(_))
                ));
            }
        }
        let manifest = TwoPointClaimManifest::new(source, group(source, input.clone())).unwrap();
        let bytes = manifest.canonical_bytes();
        let record_size = 16 + 16 * arity + 24 * owners;
        let start = bytes.len() - 2 * record_size;
        let mut duplicate_point = bytes.clone();
        duplicate_point.copy_within(start..start + record_size, start + record_size);
        decode_error(source, &duplicate_point);
        for point in 0..2 {
            let mut duplicate_owner = bytes.clone();
            let owner = start + point * record_size + 16 + arity * 16 + 24;
            duplicate_owner[owner..owner + 8].copy_from_slice(&0u64.to_le_bytes());
            decode_error(source, &duplicate_owner);
        }
        // Points can differ only in c1 and still be distinct E64 points.
        let mut distinct = input;
        distinct[1].point = distinct[0].point.clone();
        distinct[1].point[0] = [2, 4].into();
        assert!(TwoPointClaimManifest::new(source, group(source, distinct)).is_ok());
    }
}

#[test]
fn all_truncations_trailing_records_and_length_words_reject() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let manifest =
            TwoPointClaimManifest::new(source, group(source, points(arity, owners))).unwrap();
        let bytes = manifest.canonical_bytes();
        for end in 0..bytes.len() {
            decode_error(source, &bytes[..end]);
        }
        let record_size = 16 + 16 * arity + 24 * owners;
        let start = bytes.len() - 2 * record_size;
        for suffix in [
            vec![0],
            bytes[start..start + record_size].to_vec(),
            vec![0; 16],
        ] {
            let mut bad = bytes.clone();
            bad.extend(suffix);
            decode_error(source, &bad);
        }
        for point in 0..2 {
            for (offset, valid) in [(0, arity as u64), (8 + arity * 16, owners as u64)] {
                for count in [0, 1, valid - 1, valid + 1, u64::MAX] {
                    if count == valid {
                        continue;
                    }
                    let mut bad = bytes.clone();
                    let offset = start + point * record_size + offset;
                    bad[offset..offset + 8].copy_from_slice(&count.to_le_bytes());
                    decode_error(source, &bad);
                }
            }
            let mut absent = bytes.clone();
            let offset = start + point * record_size;
            absent.splice(offset..offset + record_size, [0; 16]);
            decode_error(source, &absent);
        }
    }
}

#[test]
fn every_residue_in_both_points_and_both_value_coordinates_is_canonical() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    let modulus = BinaryFieldProfile::P64.modulus();
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let input = points(arity, owners);
        let manifest = TwoPointClaimManifest::new(source, group(source, input.clone())).unwrap();
        let bytes = manifest.canonical_bytes();
        let record_size = 16 + 16 * arity + 24 * owners;
        let start = bytes.len() - 2 * record_size;
        for point in 0..2 {
            for element in 0..arity + owners {
                for coordinate in 0..2 {
                    for residue in [0, modulus - 1, modulus, u128::from(u64::MAX), u128::MAX] {
                        let mut changed = input.clone();
                        let value = if element < arity {
                            &mut changed[point].point[element]
                        } else {
                            &mut changed[point].evaluations[element - arity].value
                        };
                        let PrimeClaimValue::Extension(values) = value else {
                            unreachable!()
                        };
                        values[coordinate] = residue;
                        assert_eq!(
                            TwoPointClaimManifest::new(source, group(source, changed)).is_ok(),
                            residue < modulus
                        );
                        if residue > u128::from(u64::MAX) {
                            continue;
                        }
                        let offset = start
                            + point * record_size
                            + coordinate * 8
                            + if element < arity {
                                8 + element * 16
                            } else {
                                16 + arity * 16 + (element - arity) * 24 + 8
                            };
                        let mut changed = bytes.clone();
                        changed[offset..offset + 8]
                            .copy_from_slice(&(residue as u64).to_le_bytes());
                        assert_eq!(
                            TwoPointClaimManifest::from_bytes(source, &changed).is_ok(),
                            residue < modulus
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn binary_word_is_mandatory_bound_and_validated_against_source() {
    let descriptor = descriptor();
    let source = TwoPointClaimSource::Binary(&descriptor);
    let input = points(11, 2);
    let manifest = TwoPointClaimManifest::new(source, group(source, input.clone())).unwrap();
    let bytes = manifest.canonical_bytes();
    // The binary tag cannot introduce point records without a word record.
    let mut missing = b"AkitaBinaryPoints/p64\x01".to_vec();
    missing.extend_from_slice(&bytes[142..]);
    decode_error(source, &missing);
    let mut bad = bytes.clone();
    bad[38..70].fill(99);
    decode_error(source, &bad);
    for count in [0u64, 2, 4, u64::MAX] {
        let mut bad = bytes.clone();
        bad[70..78].copy_from_slice(&count.to_le_bytes());
        decode_error(source, &bad);
    }
    for offset in [38, 78, 126] {
        let mut changed = bytes.clone();
        // A valid alternate owner or word coordinate/value changes the statement.
        if offset == 38 {
            changed[38..70].fill(2);
        } else {
            changed[offset] ^= 1;
        }
        let changed = TwoPointClaimManifest::from_bytes(source, &changed).unwrap();
        assert_ne!(changed, manifest);
    }
    let other = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[8; 32], [9; 32]],
        descriptor.profiles().clone(),
    )
    .unwrap();
    assert!(manifest
        .validate(TwoPointClaimSource::Binary(&other))
        .is_err());
    let wrong_word = BinaryWordClaim::new(&other, [8; 32], vec![[0; 16]; 3], [0; 16]).unwrap();
    assert!(TwoPointClaimManifest::new(
        source,
        TwoPointGroupClaim::Binary {
            word: wrong_word,
            points: input,
        }
    )
    .is_err());
}

#[test]
fn ordinary_requires_all_owners_and_source_kinds_and_versions_do_not_mix() {
    let descriptor = descriptor();
    let binary = TwoPointClaimSource::Binary(&descriptor);
    for owners in [1, 2, 3, 5] {
        let profile = ordinary_profile(owners);
        let source = TwoPointClaimSource::Ordinary(&profile);
        let input = points(10, owners);
        let manifest = TwoPointClaimManifest::new(source, group(source, input.clone())).unwrap();
        assert_eq!(manifest.planes().count(), owners * 4);
        assert!(manifest.validate(binary).is_err());
        decode_error(binary, &manifest.canonical_bytes());
        assert!(TwoPointClaimManifest::new(binary, group(source, input.clone())).is_err());
        for point in 0..2 {
            let mut missing = input.clone();
            missing[point].evaluations.pop();
            assert!(TwoPointClaimManifest::new(source, group(source, missing)).is_err());
        }
    }
    let mut subset = points(11, 2);
    subset[0].evaluations.pop();
    subset[1].evaluations.remove(0);
    assert!(TwoPointClaimManifest::new(binary, group(binary, subset.clone())).is_ok());
    subset[1].point = subset[0].point.clone();
    assert!(TwoPointClaimManifest::new(binary, group(binary, subset)).is_err());
    let p128 = binary_ordinary_profile_fixture(BinaryFieldProfile::P128);
    let source = TwoPointClaimSource::Ordinary(&p128);
    assert!(TwoPointClaimManifest::new(source, group(source, points(10, 1))).is_err());
    let mut malformed = ordinary_profile(1);
    malformed.version = 0;
    let source = TwoPointClaimSource::Ordinary(&malformed);
    assert!(TwoPointClaimManifest::new(source, group(source, points(10, 1))).is_err());
    let single = PrimePlaneManifest::with_field_profile(
        &descriptor,
        Some(points(11, 1)[0].clone()),
        BinaryFieldProfile::P64,
    )
    .unwrap();
    decode_error(binary, &single.canonical_bytes());
    let two = TwoPointClaimManifest::new(binary, group(binary, points(11, 2))).unwrap();
    assert!(PrimePlaneManifest::from_bytes(
        &descriptor,
        BinaryFieldProfile::P64,
        &two.canonical_bytes()
    )
    .is_err());
}

#[test]
fn source_kind_mismatches_reject_with_matching_arity_and_owner_count() {
    let descriptor = descriptor();
    let mut profile = ordinary_profile(2);
    profile.group = PolynomialGroupLayout::new(11, 2);
    profile.blocks.live_ring_elements_per_claim = 16;
    profile.blocks.live_blocks = 8;
    let width = profile
        .derive_slice_geometry()
        .unwrap()
        .physical_input_width();
    profile.outer.matrix =
        OuterCommitMatrixParams::try_new_with_min_rank(profile.outer.matrix.sis_table_key(), width)
            .unwrap();
    profile.validate_frozen_precommit(64).unwrap();
    let binary = TwoPointClaimSource::Binary(&descriptor);
    let ordinary = TwoPointClaimSource::Ordinary(&profile);
    assert_eq!(binary.shape().unwrap(), ordinary.shape().unwrap());
    for (source, opposite) in [(binary, ordinary), (ordinary, binary)] {
        let claim = group(source, points(11, 2));
        let manifest = TwoPointClaimManifest::new(source, claim.clone()).unwrap();
        manifest.validate(source).unwrap();
        assert!(matches!(
            TwoPointClaimManifest::new(opposite, claim),
            Err(AkitaError::InvalidInput(message)) if message == "two-point source kind mismatch"
        ));
        assert!(matches!(
            manifest.validate(opposite),
            Err(AkitaError::InvalidInput(message)) if message == "two-point source kind mismatch"
        ));
    }
}

#[test]
fn prime_record_prechecks_reject_before_consuming_payload() {
    for field in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
        // Each record has enough payload; only the supplied source limits or
        // the pairing of empty points and empty values make it invalid.
        for (count, count_values, arity, owners, consumed) in [
            (1u64, 1u64, 2, 2, 8),
            (3, 1, 2, 2, 8),
            (2, 3, 2, 2, 48),
            (0, 1, 2, 2, 16),
            (2, 0, 2, 2, 48),
        ] {
            let mut bytes = count.to_le_bytes().to_vec();
            bytes.resize(8 + count as usize * 16, 0);
            bytes.extend_from_slice(&count_values.to_le_bytes());
            for owner in 0..count_values {
                bytes.extend_from_slice(&owner.to_le_bytes());
                bytes.extend_from_slice(&[0; 16]);
            }
            let mut reader = bytes.as_slice();
            assert!(matches!(
                read_claim(&mut reader, field, arity, owners, Some(|| {
                    SerializationError::InvalidData("truncated prime record".into())
                })),
                Err(SerializationError::InvalidData(message))
                    if message == "noncanonical prime record"
            ));
            assert_eq!(reader, &bytes[consumed..]);
        }
        // Source-valid counts with one payload byte missing isolate the bounds
        // from read_exact, which would instead return an I/O error.
        let mut short_point = 2u64.to_le_bytes().to_vec();
        short_point.extend_from_slice(&[0; 31]);
        let mut short_values = 2u64.to_le_bytes().to_vec();
        short_values.extend_from_slice(&[0; 32]);
        short_values.extend_from_slice(&2u64.to_le_bytes());
        short_values.extend_from_slice(&[0; 47]);
        for (bytes, consumed) in [(short_point, 8), (short_values, 48)] {
            let mut reader = bytes.as_slice();
            assert!(matches!(
                read_claim(&mut reader, field, 2, 2, Some(|| {
                    SerializationError::InvalidData("truncated prime record".into())
                })),
                Err(SerializationError::InvalidData(message)) if message == "truncated prime record"
            ));
            assert_eq!(reader, &bytes[consumed..]);
        }
    }
}

#[test]
fn word_record_prechecks_reject_before_consuming_payload() {
    let descriptor = descriptor();
    for count in [0u64, 2, 4] {
        let mut bytes = b"AkitaBinaryWord\x01".to_vec();
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&count.to_le_bytes());
        let payload = bytes.len();
        bytes.resize(payload + (count as usize + 1) * 16, 0);
        let mut reader = bytes.as_slice();
        assert!(matches!(
            BinaryWordClaim::read_from(&descriptor, &mut reader),
            Err(SerializationError::InvalidData(_))
        ));
        // Final claim validation would consume this payload before rejecting.
        assert_eq!(reader, &bytes[payload..]);
    }
    for payload_len in [47, 63] {
        let mut bytes = b"AkitaBinaryWord\x01".to_vec();
        bytes.extend_from_slice(&[1; 32]);
        bytes.extend_from_slice(&3u64.to_le_bytes());
        let payload = bytes.len();
        bytes.resize(payload + payload_len, 0);
        let mut reader = bytes.as_slice();
        assert!(matches!(
            BinaryWordClaim::read_from(&descriptor, &mut reader),
            Err(SerializationError::InvalidData(_))
        ));
        assert_eq!(reader, &bytes[payload..]);
    }
}

#[test]
fn trailing_bytes_have_an_explicit_framing_error() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let manifest =
            TwoPointClaimManifest::new(source, group(source, points(arity, owners))).unwrap();
        let bytes = manifest.canonical_bytes();
        let record_size = 16 + 16 * arity + 24 * owners;
        for suffix in [
            vec![0],
            vec![0; 16],
            bytes[bytes.len() - record_size..].to_vec(),
        ] {
            let mut trailing = bytes.clone();
            trailing.extend(suffix);
            assert!(matches!(
                TwoPointClaimManifest::from_bytes(source, &trailing),
                Err(SerializationError::InvalidData(message))
                    if message == "trailing bytes in two-point statement"
            ));
        }
    }
}

#[test]
fn wire_owner_ids_and_version_headers_reject() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let manifest =
            TwoPointClaimManifest::new(source, group(source, points(arity, owners))).unwrap();
        let bytes = manifest.canonical_bytes();
        let record_size = 16 + 16 * arity + 24 * owners;
        let start = bytes.len() - 2 * record_size;
        for point in 0..2 {
            for owner in [owners as u64, 1u64 << 32] {
                let mut bad = bytes.clone();
                let offset = start + point * record_size + 16 + 16 * arity;
                bad[offset..offset + 8].copy_from_slice(&owner.to_le_bytes());
                decode_error(source, &bad);
            }
        }
        for offset in [0, source.tag().len() - 1] {
            let mut bad = bytes.clone();
            bad[offset] ^= 1;
            assert!(matches!(
                TwoPointClaimManifest::from_bytes(source, &bad),
                Err(SerializationError::InvalidData(_))
            ));
        }
    }
    let word = BinaryWordClaim::new(&descriptor, [1; 32], vec![[7; 16]; 3], [9; 16]).unwrap();
    for offset in [0, b"AkitaBinaryWord\x01".len() - 1] {
        let mut bytes = word.canonical_bytes();
        bytes[offset] ^= 1;
        let mut reader = bytes.as_slice();
        assert!(matches!(
            BinaryWordClaim::read_from(&descriptor, &mut reader),
            Err(SerializationError::InvalidData(_))
        ));
        assert_eq!(reader, bytes.as_slice());
    }
}

#[test]
fn word_dimension_is_revalidated_independently_of_prime_dimensions() {
    let descriptor = descriptor();
    let other = BinarySourceDescriptor::new(
        BinaryIndexMap::new(64, 16, 8, 2).unwrap(),
        descriptor.polynomials().to_vec(),
        descriptor.profiles().clone(),
    )
    .unwrap();
    let source = TwoPointClaimSource::Binary(&descriptor);
    let other_source = TwoPointClaimSource::Binary(&other);
    assert_eq!(source.shape().unwrap(), other_source.shape().unwrap());
    let word = BinaryWordClaim::new(&other, [1; 32], vec![[7; 16]; 4], [9; 16]).unwrap();
    let claim = TwoPointGroupClaim::Binary {
        word,
        points: points(11, 2),
    };
    let manifest = TwoPointClaimManifest::new(other_source, claim.clone()).unwrap();
    assert!(TwoPointClaimManifest::new(source, claim).is_err());
    assert!(manifest.validate(source).is_err());
    decode_error(source, &manifest.canonical_bytes());
}

#[test]
fn both_point_records_must_be_present() {
    let mut record = 2u64.to_le_bytes().to_vec();
    record.extend_from_slice(&[0; 32]);
    record.extend_from_slice(&1u64.to_le_bytes());
    record.extend_from_slice(&[0; 24]);
    for absent in 0..2 {
        let mut bytes = Vec::new();
        for point in 0..2 {
            if point == absent {
                bytes.extend_from_slice(&[0; 16]);
            } else {
                bytes.extend_from_slice(&record);
            }
        }
        let mut reader = bytes.as_slice();
        assert!(matches!(
            read_points(&mut reader, 2, 1),
            Err(SerializationError::InvalidData(message))
                if message == "two-point statement requires both points"
        ));
    }
}

#[test]
fn decoding_rejects_invalid_frozen_source_profiles() {
    let profile = ordinary_profile(2);
    let source = TwoPointClaimSource::Ordinary(&profile);
    let bytes = TwoPointClaimManifest::new(source, group(source, points(10, 2)))
        .unwrap()
        .canonical_bytes();
    let p128 = binary_ordinary_profile_fixture(BinaryFieldProfile::P128);
    let mut wrong_inner = profile;
    wrong_inner.inner.matrix = p128.inner.matrix;
    let mut wrong_outer = profile;
    wrong_outer.outer.matrix = p128.outer.matrix;
    let mut wrong_version = profile;
    wrong_version.version = 0;
    let mut wrong_geometry = profile;
    wrong_geometry.group = PolynomialGroupLayout::new(11, 2);
    for invalid in [wrong_inner, wrong_outer, wrong_version, wrong_geometry] {
        assert!(matches!(
            TwoPointClaimManifest::from_bytes(TwoPointClaimSource::Ordinary(&invalid), &bytes),
            Err(SerializationError::InvalidData(_))
        ));
    }
}

#[test]
fn truncated_prime_payloads_return_early_invalid_data() {
    let descriptor = descriptor();
    let profile = ordinary_profile(2);
    for source in [
        TwoPointClaimSource::Binary(&descriptor),
        TwoPointClaimSource::Ordinary(&profile),
    ] {
        let (arity, owners) = source.shape().unwrap();
        let bytes = TwoPointClaimManifest::new(source, group(source, points(arity, owners)))
            .unwrap()
            .canonical_bytes();
        let record_size = 16 + 16 * arity + 24 * owners;
        let start = bytes.len() - 2 * record_size;
        for point in 0..2 {
            let offset = start + point * record_size;
            for end in [offset + 8 + 16 * arity - 1, offset + record_size - 1] {
                assert!(matches!(
                    TwoPointClaimManifest::from_bytes(source, &bytes[..end]),
                    Err(SerializationError::InvalidData(message))
                        if message == "truncated prime record"
                ));
            }
        }
    }
}

#[cfg(target_pointer_width = "32")]
#[test]
fn prime_record_owner_conversion_rejects_nonzero_high_bits() {
    let mut bytes = 2u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0; 32]);
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&(1u64 << 32).to_le_bytes());
    bytes.extend_from_slice(&[0; 16]);
    let mut reader = bytes.as_slice();
    assert!(matches!(
        read_claim(&mut reader, BinaryFieldProfile::P64, 2, 1, Some(|| {
            SerializationError::InvalidData("truncated prime record".into())
        })),
        Err(SerializationError::InvalidData(message)) if message == "noncanonical prime record"
    ));
}
