//! Independent evaluations of the public dense index map for caller terminals.
use super::*;
use akita_types::binary_source::BinaryPositionClass;
use jolt_field::CanonicalEncoding;

pub(super) fn words(index: &BinaryIndexMap, owner: usize) -> Vec<u128> {
    let mask = u128::MAX >> (128 - index.word_width());
    (0..index.logical_len())
        .map(|i| {
            (0x1234_5678_9abc_def0_1357_2468_9876_5431u128.rotate_left((i % 128) as u32)
                ^ (i as u128 * 17 + owner as u128))
                & mask
        })
        .collect()
}

pub(super) fn word_claim(
    descriptor: &BinarySourceDescriptor,
    owner: usize,
    words: &[u128],
) -> BinaryWordClaim {
    let index = descriptor.index_map();
    let logical_log = akita_error::checked::ceil_log2(index.logical_len()).unwrap();
    let point = (0..logical_log)
        .map(|axis| reference::host(31 + axis as u128 * 17 + owner as u128 * 13))
        .collect::<Vec<_>>();
    let mut values = words
        .iter()
        .copied()
        .map(reference::host)
        .collect::<Vec<_>>();
    values.resize(
        akita_error::checked::pow2(logical_log).unwrap(),
        BinaryField128::ZERO,
    );
    for &r in &point {
        values = values
            .chunks_exact(2)
            .map(|pair| (BinaryField128::ONE + r) * pair[0] + r * pair[1])
            .collect();
    }
    assert_eq!(values.len(), 1);
    BinaryWordClaim::new(
        descriptor,
        descriptor.polynomials()[owner],
        point.into_iter().map(reference::encoding).collect(),
        reference::encoding(values[0]),
    )
    .unwrap()
}

fn equality(point: &[u128], index: usize) -> F {
    point.iter().enumerate().fold(F::one(), |w, (axis, &r)| {
        let r = F::from_u128(r);
        w * if (index >> axis) & 1 == 0 {
            F::one() - r
        } else {
            r
        }
    })
}

pub(super) fn prime_claim(
    descriptor: &BinarySourceDescriptor,
    owner: usize,
    words: &[u128],
    seed: usize,
    tensor_component: u8,
) -> PrimePlaneClaim {
    let index = descriptor.index_map();
    let point = (0..8 + index.claim_variable_split().carrier_address)
        .map(|axis| 19 + axis as u128 * 7 + seed as u128 * 13)
        .collect::<Vec<_>>();
    let mut value = F::zero();
    // Scan the complete MLE domain, including zero padding and virtual slots.
    for carrier in 0..index.padded_carriers() {
        for coordinate in 0..256 {
            let ring = index.scalar_to_ring(carrier, coordinate).unwrap();
            let class = index.classify_scalar(carrier, coordinate).unwrap();
            assert_eq!(
                index
                    .ring_to_logical(ring.column, ring.ring_row, ring.coefficient)
                    .unwrap(),
                class
            );
            match class {
                BinaryPositionClass::Payload { word, bit } if (words[word] >> bit) & 1 == 1 => {
                    value += equality(&point[..8], coordinate) * equality(&point[8..], carrier);
                }
                _ => {}
            }
        }
    }
    PrimePlaneClaim {
        polynomial: owner,
        point,
        tensor_component,
        value: value.to_u128_checked().unwrap(),
    }
}
