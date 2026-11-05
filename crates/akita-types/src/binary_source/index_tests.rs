use super::*;

#[test]
fn exhaustive_small_shapes_and_every_width_round_trip() {
    for ell in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let words_per_carrier = 128 / ell;
        for columns in [1, 2, 4] {
            for padded_carriers in [4 * columns, 8 * columns] {
                let capacity = padded_carriers * words_per_carrier;
                let lengths = [
                    1,
                    words_per_carrier.saturating_sub(1).max(1),
                    words_per_carrier,
                    words_per_carrier + 1,
                    3 * words_per_carrier + 1,
                    capacity - 1,
                    capacity,
                ];
                for logical_len in lengths {
                    let map =
                        BinaryIndexMap::new(ell, logical_len, padded_carriers, columns).unwrap();
                    for carrier in 0..padded_carriers {
                        for coordinate in 0..256 {
                            let ring = map.scalar_to_ring(carrier, coordinate).unwrap();
                            assert_eq!(ring.column, carrier % columns);
                            assert_eq!(ring.ring_row, carrier / (4 * columns));
                            assert_eq!(ring.coefficient, 4 * coordinate + (carrier / columns) % 4);
                            assert_eq!(
                                ring.sign,
                                if coordinate.is_multiple_of(2) { 1 } else { -1 }
                            );
                            assert_eq!(
                                map.ring_to_scalar(ring.column, ring.ring_row, ring.coefficient),
                                Ok(BinaryScalarPosition {
                                    carrier,
                                    coordinate,
                                })
                            );
                            let linear_payload_bit = carrier * 128 + coordinate;
                            let expected = if coordinate >= 162 {
                                BinaryPositionClass::Virtual
                            } else if coordinate >= 128 || linear_payload_bit >= logical_len * ell {
                                BinaryPositionClass::CommittedPadding
                            } else {
                                BinaryPositionClass::Payload {
                                    word: linear_payload_bit / ell,
                                    bit: linear_payload_bit % ell,
                                }
                            };
                            assert_eq!(map.classify_scalar(carrier, coordinate), Ok(expected));
                            assert_eq!(
                                map.ring_to_logical(ring.column, ring.ring_row, ring.coefficient),
                                Ok(expected)
                            );
                            if let BinaryPositionClass::Payload { word, bit } = expected {
                                assert_eq!(map.logical_to_ring(word, bit), Ok(ring));
                            }
                        }
                    }
                    // Enumerate the other direction independently, including every
                    // virtual ring slot and every partial ring component.
                    for column in 0..columns {
                        for ring_row in 0..padded_carriers / (4 * columns) {
                            for coefficient in 0..1024 {
                                let scalar =
                                    map.ring_to_scalar(column, ring_row, coefficient).unwrap();
                                let ring = map
                                    .scalar_to_ring(scalar.carrier, scalar.coordinate)
                                    .unwrap();
                                assert_eq!(ring.column, column);
                                assert_eq!(ring.ring_row, ring_row);
                                assert_eq!(ring.coefficient, coefficient);
                                assert_eq!(
                                    map.ring_to_logical(column, ring_row, coefficient).unwrap()
                                        == BinaryPositionClass::Virtual,
                                    coefficient >= 648
                                );
                            }
                        }
                    }
                    for word in 0..logical_len {
                        for bit in 0..ell {
                            let ring = map.logical_to_ring(word, bit).unwrap();
                            assert_eq!(
                                map.ring_to_logical(ring.column, ring.ring_row, ring.coefficient),
                                Ok(BinaryPositionClass::Payload { word, bit })
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn partial_last_element_and_all_padding_boundaries_are_distinct() {
    // Seventeen byte words fill one carrier and one byte of the next. With
    // C=2 those carriers are different columns of the same first ring row.
    let map = BinaryIndexMap::new(8, 17, 16, 2).unwrap();
    assert_eq!(
        map.logical_to_ring(16, 7),
        Ok(BinaryRingPosition {
            column: 1,
            ring_row: 0,
            coefficient: 28,
            sign: -1,
        })
    );
    for (carrier, coordinate, class) in [
        (0, 127, BinaryPositionClass::Payload { word: 15, bit: 7 }),
        (0, 128, BinaryPositionClass::CommittedPadding),
        (0, 161, BinaryPositionClass::CommittedPadding),
        (0, 162, BinaryPositionClass::Virtual),
        (0, 255, BinaryPositionClass::Virtual),
        (1, 7, BinaryPositionClass::Payload { word: 16, bit: 7 }),
        (1, 8, BinaryPositionClass::CommittedPadding),
        (2, 0, BinaryPositionClass::CommittedPadding),
        (7, 161, BinaryPositionClass::CommittedPadding),
        (8, 0, BinaryPositionClass::CommittedPadding),
        (15, 161, BinaryPositionClass::CommittedPadding),
        (15, 162, BinaryPositionClass::Virtual),
    ] {
        assert_eq!(map.classify_scalar(carrier, coordinate), Ok(class));
        let ring = map.scalar_to_ring(carrier, coordinate).unwrap();
        assert_eq!(
            map.ring_to_logical(ring.column, ring.ring_row, ring.coefficient),
            Ok(class)
        );
    }
    for coefficient in [647, 648, 1023] {
        assert_eq!(
            map.ring_to_logical(1, 1, coefficient),
            Ok(if coefficient < 648 {
                BinaryPositionClass::CommittedPadding
            } else {
                BinaryPositionClass::Virtual
            })
        );
    }
}

#[test]
fn kernel_permutation_is_column_contiguous_and_bidirectional() {
    let map = BinaryIndexMap::new(128, 8, 8, 2).unwrap();
    let expected = [0, 4, 1, 5, 2, 6, 3, 7];
    for (carrier, kernel) in expected.into_iter().enumerate() {
        assert_eq!(map.kernel_carrier_index(carrier), Ok(kernel));
        assert_eq!(map.carrier_from_kernel_index(kernel), Ok(carrier));
    }
    for columns in [1, 2, 4, 8] {
        let map = BinaryIndexMap::new(1, 1, 8 * columns, columns).unwrap();
        let scalar_rows = 8;
        for column in 0..columns {
            for row in 0..scalar_rows {
                let carrier = row * columns + column;
                assert_eq!(
                    map.kernel_carrier_index(carrier),
                    Ok(column * scalar_rows + row)
                );
                assert_eq!(
                    map.carrier_from_kernel_index(column * scalar_rows + row),
                    Ok(carrier)
                );
            }
        }
    }
}

#[test]
fn claim_variable_split_publishes_dense_words_and_carrier_address() {
    for ell in [1usize, 2, 4, 8, 16, 32, 64, 128] {
        let map = BinaryIndexMap::new(ell, 1, 32, 4).unwrap();
        let split = map.claim_variable_split();
        assert_eq!(1 << split.intra_carrier, 128 / ell);
        assert_eq!(1 << split.word_bit, ell);
        assert_eq!(split.intra_carrier + split.word_bit, 7);
        assert_eq!(1 << split.carrier_address, map.padded_carriers());
        assert_eq!(map.word_width(), ell);
        assert_eq!(map.logical_len(), 1);
        assert_eq!(map.columns(), 4);
    }
}

#[test]
fn rejects_unsupported_widths_shapes_and_out_of_range_addresses() {
    for ell in [0, 3, 6, 127, 129, 256, usize::MAX] {
        assert!(matches!(
            BinaryIndexMap::new(ell, 1, 8, 2),
            Err(AkitaError::InvalidInput(_))
        ));
    }
    for (logical, padded, columns) in [
        (0, 8, 2),
        (1, 0, 2),
        (1, 7, 1),
        (1, 2, 1),
        (1, 8, 0),
        (1, 16, 3),
        (1, 8, usize::MAX),
        (1, 1usize << (usize::BITS - 1), 1),
    ] {
        assert!(BinaryIndexMap::new(1, logical, padded, columns).is_err());
    }
    assert!(matches!(
        BinaryIndexMap::new(128, 9, 8, 2),
        Err(AkitaError::InvalidSize {
            expected: 8,
            actual: 9
        })
    ));
    let map = BinaryIndexMap::new(32, 3, 8, 2).unwrap();
    for (word, bit) in [(3, 0), (0, 32), (usize::MAX, usize::MAX)] {
        assert!(map.logical_to_ring(word, bit).is_err());
    }
    for (carrier, coordinate) in [(8, 0), (0, 256), (usize::MAX, usize::MAX)] {
        assert!(map.scalar_to_ring(carrier, coordinate).is_err());
        assert!(map.classify_scalar(carrier, coordinate).is_err());
    }
    for (column, row, coefficient) in [(2, 0, 0), (0, 1, 0), (0, 0, 1024)] {
        assert!(map.ring_to_scalar(column, row, coefficient).is_err());
        assert!(map.ring_to_logical(column, row, coefficient).is_err());
    }
    assert!(map
        .ring_to_scalar(usize::MAX, usize::MAX, usize::MAX)
        .is_err());
    assert!(map.kernel_carrier_index(8).is_err());
    assert!(map.carrier_from_kernel_index(8).is_err());
}
