//! Scalar representations and low-index-first source addressing.

use akita_algebra::binary::{field_switch::SwitchField, BinaryField162};
use akita_algebra::fft::SmoothFftField;
use akita_algebra::ring::{pack_scalar_components, PlusTrinomial, TrinomialModulus, TrinomialRing};
use akita_challenges::{BinaryChallenge, BinaryChallengeProfile};
use akita_error::{checked, AkitaError};
use jolt_field::Field;

use crate::profile::BinaryClearSetup;

/// Expand binary equality weights, with the first coordinate on the low index bit.
///
/// This duplicates the private `row_weights` helper in
/// `akita-algebra/src/binary/field_switch.rs` for arbitrary endpoint geometry.
/// Replace it when `akita-algebra` exposes a checked arbitrary-dimension F162
/// equality expansion (tracking issue LayerZero-Labs/akita#45).
pub fn equality_weights(point: &[BinaryField162]) -> Result<Vec<BinaryField162>, AkitaError> {
    let size = checked::pow2(point.len())
        .ok_or_else(|| AkitaError::InvalidInput("binary equality table size overflow".into()))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| AkitaError::InvalidInput("binary equality table allocation failed".into()))?;
    values.push(BinaryField162::ONE);
    for &coordinate in point {
        let length = values.len();
        for index in 0..length {
            let old = *values.get(index).ok_or(AkitaError::InvalidProof)?;
            values.push(old * coordinate);
            *values.get_mut(index).ok_or(AkitaError::InvalidProof)? =
                old * (BinaryField162::ONE + coordinate);
        }
    }
    Ok(values)
}

/// Lift binary polynomial coordinates to zero or one prime coefficients.
pub fn scalar_from_binary<F: Field>(
    value: BinaryField162,
) -> Result<TrinomialRing<F, 162, PlusTrinomial>, AkitaError> {
    let bytes = value.to_bytes();
    let mut coefficients = [F::zero(); 162];
    for (index, coefficient) in coefficients.iter_mut().enumerate() {
        let byte = bytes.get(index / 8).ok_or(AkitaError::InvalidProof)?;
        *coefficient = F::from_u64(u64::from((byte >> (index % 8)) & 1));
    }
    TrinomialRing::from_coefficients(coefficients)
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))
}

/// Map signed integer coordinates into the prime scalar ring.
pub fn scalar_from_signed<F: Field>(
    value: &[i64; 162],
) -> Result<TrinomialRing<F, 162, PlusTrinomial>, AkitaError> {
    let coefficients = value.map(|coefficient| {
        let magnitude = F::from_u64(coefficient.unsigned_abs());
        if coefficient < 0 {
            -magnitude
        } else {
            magnitude
        }
    });
    TrinomialRing::from_coefficients(coefficients)
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))
}

/// Lift the signed terms of one sampled challenge into the prime scalar ring.
pub fn challenge_scalar<F: Field>(
    challenge: &BinaryChallenge,
) -> Result<TrinomialRing<F, 162, PlusTrinomial>, AkitaError> {
    let mut coefficients = [0i64; 162];
    for term in challenge.terms() {
        if !matches!(term.coefficient, -1 | 1) {
            return Err(AkitaError::InvalidInput(
                "invalid binary challenge sign".into(),
            ));
        }
        let coefficient = coefficients
            .get_mut(usize::from(term.position))
            .ok_or_else(|| AkitaError::InvalidInput("challenge exceeds scalar degree".into()))?;
        if *coefficient != 0 {
            return Err(AkitaError::InvalidInput(
                "duplicate binary challenge position".into(),
            ));
        }
        *coefficient = i64::from(term.coefficient);
    }
    scalar_from_signed(&coefficients)
}

/// Project the admitted challenge's canonical support encoding modulo two.
/// Only the degree-162 scalar ring has the F162 byte representation.
pub fn challenge_binary(
    challenge: &BinaryChallenge,
    profile: &BinaryChallengeProfile,
) -> Result<BinaryField162, AkitaError> {
    if profile.scalar_ring().degree() != BinaryField162::DEGREE {
        return Err(AkitaError::InvalidInput(
            "binary challenge parity requires scalar degree 162".into(),
        ));
    }
    let bytes = challenge.canonical_support_encoding(profile)?;
    BinaryField162::from_bytes(&bytes)
        .ok_or_else(|| AkitaError::InvalidInput("invalid F162 challenge support encoding".into()))
}

/// Pack one column, using scalar row `element * k + component`.
pub fn pack_source_column<H, F, const D: usize, M>(
    setup: &BinaryClearSetup<F, D, M>,
    source: &[H::Source],
    column: usize,
) -> Result<Vec<TrinomialRing<F, D, M>>, AkitaError>
where
    H: SwitchField,
    F: SmoothFftField,
    M: TrinomialModulus,
{
    if source.len() != setup.source_len() || column >= setup.columns() {
        return Err(AkitaError::InvalidInput(
            "source column geometry mismatch".into(),
        ));
    }
    let start = checked::product([column, setup.scalar_rows()]).ok_or(AkitaError::InvalidProof)?;
    let range = checked::range(start, setup.scalar_rows()).ok_or(AkitaError::InvalidProof)?;
    let words = source.get(range).ok_or(AkitaError::InvalidProof)?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(setup.m())
        .map_err(|_| AkitaError::InvalidInput("packed source allocation failed".into()))?;
    for chunk in words.chunks_exact(setup.k()) {
        let mut components = Vec::with_capacity(setup.k());
        for &word in chunk {
            components.push(scalar_from_binary::<F>(
                akita_algebra::binary::field_switch::embed_source::<H>(word),
            )?);
        }
        result.push(
            pack_scalar_components(&components)
                .map_err(|error| AkitaError::InvalidInput(error.to_string()))?,
        );
    }
    Ok(result)
}
