//! Deterministic benchmark inputs and independent claimed evaluations.
use super::{E, F, FIELD};
use akita_algebra::binary::BinaryField128;
use akita_error::{checked, AkitaError};
use akita_types::binary::{
    BinaryCommittedGroupManifest, BinaryIndexMap, BinarySourceDescriptor, BinaryWordClaim,
    PrimeClaimValue, PrimeGroupClaim, PrimePlaneClaim,
};
use jolt_field::{CanonicalEncoding, ExtField, One, Ring, Zero};
use serde::{Deserialize, Serialize};

/// Claimed prime point and value, as canonical base coordinates of each E element.
type PrimeCoordinates = (Vec<Vec<u128>>, Vec<u128>);

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Claims {
    pub word_point: Vec<[u8; 16]>,
    pub word_value: [u8; 16],
    pub prime: Option<PrimeCoordinates>,
}

pub(super) fn descriptor(log_carriers: usize) -> Result<BinarySourceDescriptor, AkitaError> {
    let count = checked::pow2(log_carriers).ok_or(AkitaError::InvalidProof)?;
    // Smoke exercises the same kernels with a narrow matrix; benchmark rows
    // retain the specified C=256 and rank three.
    let columns = if log_carriers == 10 { 2 } else { 256 };
    let index = BinaryIndexMap::new(128, count, count, columns)?;
    let profiles = BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, FIELD)?;
    BinarySourceDescriptor::new(index, vec![[29; 32]], profiles)
}

pub(super) fn words(count: usize) -> Vec<u128> {
    (0..count)
        .map(|i| {
            // A fixed integer mixer makes the source reproducible across modes.
            let x = (i as u128).wrapping_add(0x0123_4567_89ab_cdef_fedc_ba98_7654_3210);
            let x = (x ^ (x >> 31)).wrapping_mul(0xda94_2042_e4dd_58b5_9e37_79b9_7f4a_7c15);
            x ^ (x >> 47)
        })
        .collect()
}

pub(super) fn evaluate(log: usize, shape: &str, words: &[u128]) -> Result<Claims, AkitaError> {
    let point = (0..log)
        .map(|axis| (31 + axis as u128 * 17).to_le_bytes())
        .collect::<Vec<_>>();
    let mut values = words.iter().map(|&w| host(w)).collect::<Vec<_>>();
    for coordinate in &point {
        let r = host(u128::from_le_bytes(*coordinate));
        let half = values.len() / 2;
        for i in 0..half {
            values[i] = values[2 * i] + r * (values[2 * i] + values[2 * i + 1]);
        }
        values.truncate(half);
    }
    let [low, high] = values.first().ok_or(AkitaError::InvalidProof)?.to_words();
    let mut result = Claims {
        word_point: point,
        word_value: (u128::from(low) | (u128::from(high) << 64)).to_le_bytes(),
        prime: None,
    };
    if shape != "binary-claim" {
        let arity = checked::sum([8, log]).ok_or(AkitaError::InvalidProof)?;
        // P64 coordinates are E64 elements with a nonzero u coordinate.
        let point = (0..arity)
            .map(|axis| {
                E::from_base_fn(|i| F::from_u128([19 + axis as u128 * 7, 5 + axis as u128 * 3][i]))
            })
            .collect::<Vec<_>>();
        let mut bit_weights = vec![E::one()];
        for &r in point.get(..8).ok_or(AkitaError::InvalidProof)? {
            let size = bit_weights.len();
            bit_weights.resize(
                checked::product([size, 2]).ok_or(AkitaError::InvalidProof)?,
                E::zero(),
            );
            for i in 0..size {
                let old = bit_weights[i];
                bit_weights[i] = old * (E::one() - r);
                bit_weights[i + size] = old * r;
            }
        }
        let mut values = words
            .iter()
            .map(|&word| {
                bit_weights
                    .iter()
                    .take(128)
                    .enumerate()
                    .fold(E::zero(), |sum, (bit, weight)| {
                        if (word >> bit) & 1 == 0 {
                            sum
                        } else {
                            sum + *weight
                        }
                    })
            })
            .collect::<Vec<_>>();
        for &r in point.get(8..).ok_or(AkitaError::InvalidProof)? {
            let half = values.len() / 2;
            for i in 0..half {
                values[i] = values[2 * i] + r * (values[2 * i + 1] - values[2 * i]);
            }
            values.truncate(half);
        }
        let value = coordinates(values.first().ok_or(AkitaError::InvalidProof)?)?;
        let point = point.iter().map(coordinates).collect::<Result<_, _>>()?;
        result.prime = Some((point, value));
    }
    Ok(result)
}

fn coordinates(value: &E) -> Result<Vec<u128>, AkitaError> {
    (0..<E as ExtField<F>>::DEGREE)
        .map(|i| {
            value
                .base_coefficient(i)
                .to_u128_checked()
                .ok_or(AkitaError::InvalidProof)
        })
        .collect()
}

fn claim_value(coordinates: &[u128]) -> Result<PrimeClaimValue, AkitaError> {
    match *coordinates {
        [base] => Ok(base.into()),
        [c0, c1] => Ok([c0, c1].into()),
        _ => Err(AkitaError::InvalidProof),
    }
}

fn host(value: u128) -> BinaryField128 {
    BinaryField128::from_words([value as u64, (value >> 64) as u64])
}

impl Claims {
    pub(super) fn word(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<BinaryWordClaim, AkitaError> {
        BinaryWordClaim::new(
            descriptor,
            [29; 32],
            self.word_point.clone(),
            self.word_value,
        )
    }
    pub(super) fn prime(&self) -> Result<Option<PrimeGroupClaim>, AkitaError> {
        self.prime
            .as_ref()
            .map(|(point, value)| {
                Ok(PrimeGroupClaim {
                    point: point
                        .iter()
                        .map(|coordinate| claim_value(coordinate))
                        .collect::<Result<_, AkitaError>>()?,
                    evaluations: vec![PrimePlaneClaim {
                        polynomial: 0,
                        value: claim_value(value)?,
                    }],
                })
            })
            .transpose()
    }
}
