//! Independent nonzero word and prime readings for the opening fixture.

use super::*;
use akita_algebra::{CanonicalEncoding, One, Zero};
use akita_config::proof_optimized::fp64::ExtensionField as E;
use akita_types::binary_source::{BinarySourceDescriptor, BinaryWordClaim, PrimeGroupClaim};
use akita_types::FpExtEncoding;

pub(crate) fn words(group: usize) -> Vec<u128> {
    let width = [1, 16, 128, 128][group];
    (0..8).map(|i| {
        let value = (0x91u128 << 120) | (1u128 << (17 + i)) | (3 + 7 * i + group) as u128;
        if width == 128 { value } else { value & ((1 << width) - 1) }
    }).collect()
}

// Polynomial-basis multiplication modulo X^128 + X^7 + X^2 + X + 1.
fn multiply(mut a: u128, mut b: u128) -> u128 {
    let mut result = 0;
    for _ in 0..128 {
        if b & 1 != 0 { result ^= a; }
        let carry = a >> 127;
        a <<= 1;
        if carry != 0 { a ^= 0x87; }
        b >>= 1;
    }
    result
}

pub(crate) fn word(descriptor: &BinarySourceDescriptor, words: &[u128]) -> BinaryWordClaim {
    let variables = (8 * 128 / descriptor.index_map().word_width()).trailing_zeros() as usize;
    let point = (0..variables).map(|i| (7 + 2 * i as u128).to_le_bytes())
        .collect::<Vec<_>>();
    let mut value = 0;
    for (i, &word) in words.iter().enumerate() {
        let weight = point.iter().enumerate().fold(1, |weight, (bit, coordinate)| {
            let r = u128::from_le_bytes(*coordinate);
            multiply(weight, if i >> bit & 1 == 1 { r } else { r ^ 1 })
        });
        value ^= multiply(word, weight);
    }
    assert_ne!(value, 0);
    BinaryWordClaim::new(descriptor, descriptor.polynomials()[0], point, value.to_le_bytes())
        .unwrap()
}

pub(crate) fn prime(width: usize, words: &[u128], claim: &mut PrimeGroupClaim) {
    let point = claim.point.iter().map(|r| {
        native_open_fixture::ext::<E>(r.coordinates().iter().map(|&x| x as u64)
            .collect::<Vec<_>>().try_into().unwrap())
    }).collect::<Vec<_>>();
    let mut value = E::zero();
    for (word, &payload) in words.iter().enumerate() {
        for bit in 0..width {
            if payload >> bit & 1 == 0 { continue; }
            // Scalar source table: coefficient, then carrier, both low-bit first.
            let index = (word * width + bit) % 128 + 256 * (word * width / 128);
            let equality = point.iter().enumerate().fold(E::one(), |weight, (axis, &r)| {
                weight * if index >> axis & 1 == 1 { r } else { E::one() - r }
            });
            value += equality;
        }
    }
    let coordinates = [0, 1].map(|i| value.ext_coords()[i].to_u128_checked().unwrap());
    assert!(coordinates.iter().all(|&x| x != 0));
    claim.evaluations[0].value = coordinates.into();
}
