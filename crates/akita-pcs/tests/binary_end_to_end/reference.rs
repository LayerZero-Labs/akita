//! Independent logical-word and whole-carrier MLE reference evaluations.

use super::*;

pub(super) fn host(value: u128) -> BinaryField128 {
    BinaryField128::from_words([value as u64, (value >> 64) as u64])
}

pub(super) fn encoding(value: BinaryField128) -> [u8; 16] {
    let [low, high] = value.to_words();
    (u128::from(low) | (u128::from(high) << 64)).to_le_bytes()
}

pub(super) fn logical_words(descriptor: &BinarySourceDescriptor, carriers: &[u128]) -> Vec<u128> {
    let width = descriptor.index_map().word_width();
    let mask = u128::MAX >> (128 - width);
    carriers
        .iter()
        .flat_map(|carrier| (0..128 / width).map(move |slot| (carrier >> (slot * width)) & mask))
        .collect()
}

pub(super) fn logical_value(
    descriptor: &BinarySourceDescriptor,
    carriers: &[u128],
    point: &[BinaryField128],
) -> BinaryField128 {
    let mut values = logical_words(descriptor, carriers)
        .into_iter()
        .map(host)
        .collect::<Vec<_>>();
    for &coordinate in point {
        values = values
            .chunks_exact(2)
            .map(|pair| (BinaryField128::ONE + coordinate) * pair[0] + coordinate * pair[1])
            .collect();
    }
    assert_eq!(values.len(), 1, "reference logical domain mismatch");
    values[0]
}

pub(super) fn claim(
    descriptor: &BinarySourceDescriptor,
    carriers: &[u128],
    point: &[BinaryField128],
) -> BinaryWordClaim {
    BinaryWordClaim::new(
        descriptor,
        descriptor.polynomials()[0],
        point.iter().copied().map(encoding).collect(),
        encoding(logical_value(descriptor, carriers, point)),
    )
    .unwrap()
}

pub(super) fn carrier_value(carriers: &[u128], point: &[BinaryField162]) -> BinaryField162 {
    let mut values = carriers
        .iter()
        .map(|&value| scalar(value as u64, (value >> 64) as u64, 0))
        .collect::<Vec<_>>();
    for &coordinate in point {
        values = values
            .chunks_exact(2)
            .map(|pair| (BinaryField162::ONE + coordinate) * pair[0] + coordinate * pair[1])
            .collect();
    }
    assert_eq!(values.len(), 1, "reference carrier domain mismatch");
    values[0]
}
