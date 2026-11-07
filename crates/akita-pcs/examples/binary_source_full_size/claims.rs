//! Deterministic benchmark inputs and independent claimed evaluations.
use super::config::MeasurementConfig;
use akita_algebra::{binary::BinaryField128, eq_poly::EqPolynomial};
use akita_error::{checked, AkitaError};
use akita_params::binary_setup::BinaryFieldProfile;
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryIndexMap, BinarySourceDescriptor, BinaryWordClaim,
    PrimeClaimValue, PrimeGroupClaim, PrimePlaneClaim,
};
use jolt_field::{CanonicalEncoding, Ext2, ExtField, Field, Prime64Offset59, Ring};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Claims {
    pub word_point: Option<Vec<[u8; 16]>>,
    pub word_value: [u8; 16],
    pub prime: Option<(Vec<u128>, u128)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extension: Option<Vec<ExtensionReading>>,
    pub ordinary: Option<(Vec<u128>, u128)>,
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct ExtensionReading {
    pub point: Vec<[u64; 2]>,
    pub value: [u64; 2],
}

pub(super) fn descriptor<C: MeasurementConfig>(
    log_carriers: usize,
) -> Result<BinarySourceDescriptor, AkitaError> {
    let count = checked::pow2(log_carriers).ok_or(AkitaError::InvalidProof)?;
    // Smoke exercises the same kernels with a narrow matrix; benchmark rows
    // retain the specified C=256 and rank three.
    let columns = if log_carriers <= 10 { 2 } else { 256 };
    let index = BinaryIndexMap::new(128, count, count, columns)?;
    let profiles = BinaryCommittedGroupManifest::profiles_for_field(&index, 1, 0, C::PROFILE)?;
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

pub(super) fn evaluate<C: MeasurementConfig>(
    log: usize,
    shape: &str,
    words: &[u128],
) -> Result<Claims, AkitaError> {
    let mut result = Claims::new::<C>(log, shape)?;
    if !shape.starts_with("prime-") {
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
        let [low, high] = values
            .first()
            .ok_or_else(|| AkitaError::Internal("empty word evaluation".into()))?
            .to_words();
        result.word_value = (u128::from(low) | (u128::from(high) << 64)).to_le_bytes();
        result.word_point = Some(point);
    }
    if let Some((point, value)) = &mut result.prime {
        let point = point
            .iter()
            .map(|&x| C::Field::from_u128(x))
            .collect::<Vec<_>>();
        *value = prime_value(words, &point)?
            .to_u128_checked()
            .ok_or_else(|| AkitaError::Internal("prime evaluation exceeds u128".into()))?;
    }
    if let Some(readings) = &mut result.extension {
        for reading in readings {
            let point = reading
                .point
                .iter()
                .map(|x| Ext2::<Prime64Offset59>::from_base_fn(|i| Prime64Offset59::from_u64(x[i])))
                .collect::<Vec<_>>();
            let value = prime_value(words, &point)?;
            for (coordinate, output) in reading.value.iter_mut().enumerate() {
                *output = value
                    .base_coefficient(coordinate)
                    .to_u128_checked()
                    .and_then(|x| u64::try_from(x).ok())
                    .ok_or(AkitaError::InvalidProof)?;
            }
        }
    }
    Ok(result)
}

// Fold the source bits independently of the EP relation compiler. Base and
// extension fixtures use the same multilinear evaluation and axis order.
pub(super) fn prime_value<T: Field>(words: &[u128], point: &[T]) -> Result<T, AkitaError> {
    let bit_weights = EqPolynomial::evals(point.get(..8).ok_or(AkitaError::InvalidProof)?)?;
    let mut values = words
        .iter()
        .map(|&word| {
            bit_weights
                .iter()
                .take(128)
                .enumerate()
                .fold(T::zero(), |sum, (bit, weight)| {
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
    let value = values
        .first()
        .copied()
        .ok_or_else(|| AkitaError::Internal("empty prime evaluation".into()))?;
    if values.len() != 1 {
        return Err(AkitaError::InvalidInput(
            "prime fixture size mismatch".into(),
        ));
    }
    Ok(value)
}

fn host(value: u128) -> BinaryField128 {
    BinaryField128::from_words([value as u64, (value >> 64) as u64])
}

impl Claims {
    /// Deterministic claim geometry shared by catalog pricing and source evaluation.
    /// Values start at zero; evaluation fills them from the committed words.
    pub(super) fn new<C: MeasurementConfig>(log: usize, shape: &str) -> Result<Self, AkitaError> {
        let arity = checked::sum([8, log]).ok_or(AkitaError::InvalidProof)?;
        let points = match shape {
            "prime-extension-one" => 1,
            "prime-extension-two"
            | "prime-extension-two-plus-ordinary"
            | "binary-and-prime-extension-two"
            | "binary-and-prime-extension-two-plus-ordinary" => 2,
            _ => 0,
        };
        if points > 0 && C::PROFILE != BinaryFieldProfile::P64 {
            return Err(AkitaError::InvalidInput(
                "extension prime fixtures require p64".into(),
            ));
        }
        Ok(Self {
            word_point: None,
            word_value: [0; 16],
            prime: (shape != "binary-claim" && points == 0)
                .then(|| ((0..arity).map(|axis| 19 + axis as u128 * 7).collect(), 0)),
            extension: (points > 0).then(|| {
                (0..points)
                    .map(|copy| ExtensionReading {
                        point: (0..arity)
                            .map(|axis| [19 + axis as u64 * 7, 5 + axis as u64 * 3 + copy * 11])
                            .collect(),
                        value: [0; 2],
                    })
                    .collect()
            }),
            ordinary: shape.ends_with("-plus-ordinary").then(|| {
                let point = (0..log.max(18)).map(|axis| 7 + axis as u128 * 11).collect();
                (point, 7)
            }),
        })
    }

    pub(super) fn word(
        &self,
        descriptor: &BinarySourceDescriptor,
    ) -> Result<Option<BinaryWordClaim>, AkitaError> {
        self.word_point
            .as_ref()
            .map(|point| BinaryWordClaim::new(descriptor, [29; 32], point.clone(), self.word_value))
            .transpose()
    }
    pub(super) fn prime<C: MeasurementConfig>(&self) -> Result<Vec<PrimeGroupClaim>, AkitaError> {
        if let Some(readings) = &self.extension {
            if self.prime.is_some() || C::PROFILE != BinaryFieldProfile::P64 {
                return Err(AkitaError::InvalidInput(
                    "invalid extension fixture profile".into(),
                ));
            }
            return Ok(readings
                .iter()
                .map(|reading| PrimeGroupClaim {
                    point: reading
                        .point
                        .iter()
                        .map(|value| PrimeClaimValue::Extension(value.map(u128::from)))
                        .collect(),
                    evaluations: vec![PrimePlaneClaim {
                        polynomial: 0,
                        value: PrimeClaimValue::Extension(reading.value.map(u128::from)),
                    }],
                })
                .collect());
        }
        let encode = |value| match C::PROFILE {
            BinaryFieldProfile::P128 => PrimeClaimValue::Base(value),
            BinaryFieldProfile::P64 => PrimeClaimValue::Extension([value, 0]),
        };
        Ok(self
            .prime
            .iter()
            .map(|(point, value)| PrimeGroupClaim {
                point: point.iter().copied().map(encode).collect(),
                evaluations: vec![PrimePlaneClaim {
                    polynomial: 0,
                    value: encode(*value),
                }],
            })
            .collect())
    }
}
