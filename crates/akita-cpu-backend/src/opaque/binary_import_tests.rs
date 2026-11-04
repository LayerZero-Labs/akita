use super::*;
use akita_types::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap};

fn descriptor(width: usize, logical_len: usize) -> BinarySourceDescriptor {
    BinarySourceDescriptor::new(
        BinaryIndexMap::new(width, logical_len, 8, 2).unwrap(),
        vec![[1; 32], [2; 32]],
        BinaryCommitmentProfiles {
            a: [3; 32],
            b: [4; 32],
        },
    )
    .unwrap()
}

#[test]
fn packed_storage_matches_logical_words_including_partial_carriers_and_all_padding() {
    for width in [1, 8, 16, 128] {
        let per_carrier = 128 / width;
        for len in [1, per_carrier + 1, 8 * per_carrier] {
            let descriptor = descriptor(width, len);
            let mask = u128::MAX >> (128 - width);
            let words = (0..2)
                .map(|owner| {
                    (0..len)
                        .map(|i| ((i as u128 + 11) * (owner + 3) * 0x1020_3040_5060_7081) & mask)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            // Pack in logical order independently of the signed ring map.
            let carriers = words
                .iter()
                .map(|words| {
                    words
                        .chunks(per_carrier)
                        .map(|chunk| {
                            chunk.iter().enumerate().fold(0, |carrier, (slot, word)| {
                                carrier | (word << (slot * width))
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let logical = BinarySourceInput::Logical(words)
                .prepare(&descriptor)
                .unwrap();
            let packed = BinarySourceInput::Packed {
                logical_len: len,
                carriers: carriers.clone(),
            }
            .prepare(&descriptor)
            .unwrap();
            assert_eq!(logical, packed);
            // Every padded carrier is materialized once, in kernel order, and
            // committed padding carriers are zero.
            let index = descriptor.index_map();
            assert_eq!(packed.len(), 2 * index.padded_carriers());
            for (owner, carriers) in carriers.iter().enumerate() {
                for carrier in 0..index.padded_carriers() {
                    let kernel = index.kernel_carrier_index(carrier).unwrap();
                    assert_eq!(
                        packed[owner * index.padded_carriers() + kernel],
                        carriers.get(carrier).copied().unwrap_or(0)
                    );
                }
            }
        }
    }
}

#[test]
fn compact_import_rejects_wrong_lengths_owners_and_nonzero_tail_bits() {
    for width in [1, 8, 16, 128] {
        let per_carrier = 128 / width;
        let logical_len = per_carrier + 1;
        let descriptor = descriptor(width, logical_len);
        let carriers = vec![vec![0, 1]; 2];
        let prepare = |logical_len, carriers| {
            BinarySourceInput::Packed {
                logical_len,
                carriers,
            }
            .prepare(&descriptor)
        };
        assert!(prepare(logical_len, carriers.clone()).is_ok());
        assert!(prepare(logical_len - 1, carriers.clone()).is_err());
        assert!(prepare(logical_len + 1, carriers.clone()).is_err());
        assert!(prepare(usize::MAX, carriers.clone()).is_err());
        assert!(prepare(logical_len, vec![vec![0]; 2]).is_err());
        assert!(prepare(logical_len, vec![vec![0; 3]; 2]).is_err());
        assert!(prepare(logical_len, vec![vec![0; 2]]).is_err());
        if width < 128 {
            let mut bad = carriers.clone();
            bad[1][1] |= 1u128 << width;
            assert!(prepare(logical_len, bad).is_err());
        }
    }
}

#[test]
fn compact_import_enforces_carrier_budget_before_allocation() {
    // The budget counts one u128 per padded carrier: 2^23 carriers in total.
    for (owners, padded_carriers, permitted) in [
        (1, 1 << 24, false),
        (2, 1 << 23, false),
        (1, 1 << 23, true),
        (2, 1 << 22, true),
    ] {
        let descriptor = BinarySourceDescriptor::new(
            BinaryIndexMap::new(128, 1, padded_carriers, 2).unwrap(),
            (0..owners).map(|owner| [owner as u8; 32]).collect(),
            BinaryCommitmentProfiles {
                a: [3; 32],
                b: [4; 32],
            },
        )
        .unwrap();
        // Each owner supplies one carrier; committed padding fills the rest.
        let carriers = vec![vec![1]; owners];
        let packed = BinarySourceInput::Packed {
            logical_len: 1,
            carriers: carriers.clone(),
        }
        .prepare(&descriptor);
        if permitted {
            let storage = packed.unwrap();
            let kernel = descriptor.index_map().kernel_carrier_index(0).unwrap();
            assert_eq!(storage.len(), owners * padded_carriers);
            for owner in 0..owners {
                assert_eq!(storage[owner * padded_carriers + kernel], 1);
            }
            assert_eq!(
                storage.iter().filter(|&&carrier| carrier != 0).count(),
                owners
            );
        } else {
            assert!(matches!(
                packed,
                Err(AkitaError::InvalidInput(message))
                    if message == "binary source allocation exceeds its budget"
            ));
            assert!(matches!(
                BinarySourceInput::Logical(carriers).prepare(&descriptor),
                Err(AkitaError::InvalidInput(message))
                    if message == "binary source allocation exceeds its budget"
            ));
        }
    }
}
