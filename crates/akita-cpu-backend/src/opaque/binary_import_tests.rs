use super::*;
use akita_types::binary_source::{BinaryCommitmentProfiles, BinaryIndexMap, BinaryPositionClass};

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
                carriers,
            }
            .prepare(&descriptor)
            .unwrap();
            // Import retains the compact variant, not the transient i128 rings.
            let BinarySourceCoefficients::Packed(storage) = &packed else {
                panic!("compact import retained an expanded source");
            };
            assert_eq!(
                storage.iter().map(Vec::len).sum::<usize>(),
                2 * len.div_ceil(per_carrier)
            );
            assert_eq!(
                logical.expanded(&descriptor).unwrap(),
                packed.expanded(&descriptor).unwrap()
            );
            for owner in 0..2 {
                assert_eq!(
                    logical.canonical_carriers(&descriptor, owner).unwrap(),
                    packed.canonical_carriers(&descriptor, owner).unwrap()
                );
            }
            let expanded = packed.expanded(&descriptor).unwrap();
            let index = descriptor.index_map();
            for owner in 0..2 {
                for carrier in 0..8 {
                    for coordinate in 0..162 {
                        if index.classify_scalar(carrier, coordinate).unwrap()
                            == BinaryPositionClass::CommittedPadding
                        {
                            let position = index.scalar_to_ring(carrier, coordinate).unwrap();
                            let element = owner * 2 + position.column;
                            assert_eq!(expanded[element][position.coefficient], 0);
                        }
                    }
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
