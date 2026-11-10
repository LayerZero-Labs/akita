use super::*;
use crate::binary::{BinaryCommitmentProfiles, BinaryIndexMap};
use akita_serialization::SerializationError;

#[test]
fn one_point_encodings_match_independent_recipe() {
    let source = BinarySourceDescriptor::new(
        BinaryIndexMap::new(128, 8, 8, 2).unwrap(),
        vec![[1; 32], [2; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap();
    for field in [BinaryFieldProfile::P128, BinaryFieldProfile::P64] {
        for owners in 0..=2 {
            let claim = (owners != 0).then(|| PrimeGroupClaim {
                point: (0..11)
                    .map(|axis| match field {
                        BinaryFieldProfile::P128 => {
                            PrimeClaimValue::Base(0x10000000000000000000000000 + axis as u128)
                        }
                        BinaryFieldProfile::P64 => {
                            [101 + 2 * axis as u128, 102 + 2 * axis as u128].into()
                        }
                    })
                    .collect(),
                evaluations: (0..owners)
                    .rev()
                    .map(|polynomial| PrimePlaneClaim {
                        polynomial,
                        value: match field {
                            BinaryFieldProfile::P128 => {
                                (0x20000000000000000000000000 + polynomial as u128).into()
                            }
                            BinaryFieldProfile::P64 => {
                                [301 + 2 * polynomial as u128, 302 + 2 * polynomial as u128].into()
                            }
                        },
                    })
                    .collect(),
            });
            let manifest = PrimePlaneManifest::with_field_profile(&source, claim, field).unwrap();
            // Independent recipe for the one-point tags, scalar widths and record order.
            let mut expected = match field {
                BinaryFieldProfile::P128 => b"AkitaPrimePlanes\x02".to_vec(),
                BinaryFieldProfile::P64 => b"AkitaPrimePlanes/p64\x04".to_vec(),
            };
            let tag_len = expected.len();
            let arity = if owners == 0 { 0u64 } else { 11 };
            expected.extend_from_slice(&arity.to_le_bytes());
            for axis in 0..arity {
                match field {
                    BinaryFieldProfile::P128 => expected.extend_from_slice(
                        &(0x10000000000000000000000000u128 + u128::from(axis)).to_le_bytes(),
                    ),
                    BinaryFieldProfile::P64 => {
                        expected.extend_from_slice(&(101 + 2 * axis).to_le_bytes());
                        expected.extend_from_slice(&(102 + 2 * axis).to_le_bytes());
                    }
                }
            }
            expected.extend_from_slice(&(owners as u64).to_le_bytes());
            for owner in 0..owners as u64 {
                expected.extend_from_slice(&owner.to_le_bytes());
                match field {
                    BinaryFieldProfile::P128 => expected.extend_from_slice(
                        &(0x20000000000000000000000000u128 + u128::from(owner)).to_le_bytes(),
                    ),
                    BinaryFieldProfile::P64 => {
                        expected.extend_from_slice(&(301 + 2 * owner).to_le_bytes());
                        expected.extend_from_slice(&(302 + 2 * owner).to_le_bytes());
                    }
                }
            }
            assert_eq!(manifest.canonical_bytes(), expected);
            assert_eq!(
                PrimePlaneManifest::from_bytes(&source, field, &expected).unwrap(),
                manifest
            );
            // A complete tag followed by a truncated record is an I/O EOF error.
            for end in 0..expected.len() {
                let error =
                    PrimePlaneManifest::from_bytes(&source, field, &expected[..end]).unwrap_err();
                if end < tag_len {
                    assert!(matches!(error, SerializationError::InvalidData(_)));
                } else {
                    assert!(matches!(
                        error,
                        SerializationError::IoError(error)
                            if error.kind() == std::io::ErrorKind::UnexpectedEof
                                && error.to_string() == "failed to fill whole buffer"
                    ));
                }
            }
            if owners == 2 {
                let mut unsorted = expected.clone();
                let start = tag_len + 16 + 11 * 16;
                let (prefix, suffix) = unsorted.split_at_mut(start + 24);
                prefix[start..].swap_with_slice(&mut suffix[..24]);
                assert!(matches!(
                    PrimePlaneManifest::from_bytes(&source, field, &unsorted),
                    Err(SerializationError::InvalidData(_))
                ));
            }
            // Invalid source counts take precedence over insufficient payload.
            for (offset, count) in [(tag_len, 12u64), (tag_len + 8 + arity as usize * 16, 3)] {
                let mut bad = expected[..offset].to_vec();
                bad.extend_from_slice(&count.to_le_bytes());
                assert!(matches!(
                    PrimePlaneManifest::from_bytes(&source, field, &bad),
                    Err(SerializationError::InvalidData(message))
                        if message == "noncanonical prime manifest"
                ));
            }
            expected.push(0);
            assert!(matches!(
                PrimePlaneManifest::from_bytes(&source, field, &expected),
                Err(SerializationError::InvalidData(_))
            ));
        }
    }
}
