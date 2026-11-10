//! Canonical point records shared by one-point and two-point statements.

use super::{PrimeClaimValue, PrimeGroupClaim, PrimePlaneClaim};
use akita_error::checked;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_serialization::SerializationError;
use std::io::Read;

/// Reads one point record. Counts are bounded by the source shape before any
/// allocation. With `truncated` set, a payload shorter than its declared counts
/// is rejected with that error before anything is read; without it, records
/// are read in order and a short payload surfaces as the reader's own error.
pub(super) fn read_claim(
    reader: &mut &[u8],
    field: BinaryFieldProfile,
    arity: usize,
    owners: usize,
    truncated: Option<fn() -> SerializationError>,
) -> Result<Option<PrimeGroupClaim>, SerializationError> {
    let bad = || SerializationError::InvalidData("noncanonical prime record".into());
    let count = usize::try_from(read_u64(reader)?).map_err(|_| bad())?;
    if count != 0 && count != arity {
        return Err(bad());
    }
    // Each field element occupies 16 bytes.
    if let Some(truncated) = truncated {
        if checked::product([count, 16]).ok_or_else(bad)? > reader.len() {
            return Err(truncated());
        }
    }
    let mut point = Vec::new();
    point.try_reserve_exact(count).map_err(|_| bad())?;
    for _ in 0..count {
        point.push(read_element(reader, field)?);
    }
    let count_values = usize::try_from(read_u64(reader)?).map_err(|_| bad())?;
    if count_values > owners || ((count_values == 0) != (count == 0)) {
        return Err(bad());
    }
    if let Some(truncated) = truncated {
        if checked::product([count_values, 24]).ok_or_else(bad)? > reader.len() {
            return Err(truncated());
        }
    }
    let mut evaluations = Vec::new();
    evaluations
        .try_reserve_exact(count_values)
        .map_err(|_| bad())?;
    for _ in 0..count_values {
        evaluations.push(PrimePlaneClaim {
            polynomial: usize::try_from(read_u64(reader)?).map_err(|_| bad())?,
            value: read_element(reader, field)?,
        });
    }
    Ok((count_values != 0).then_some(PrimeGroupClaim { point, evaluations }))
}

fn read_u64(reader: &mut &[u8]) -> Result<u64, SerializationError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_element(
    reader: &mut &[u8],
    field: BinaryFieldProfile,
) -> Result<PrimeClaimValue, SerializationError> {
    match field {
        BinaryFieldProfile::P128 => {
            let mut bytes = [0; 16];
            reader.read_exact(&mut bytes)?;
            Ok(u128::from_le_bytes(bytes).into())
        }
        BinaryFieldProfile::P64 => {
            Ok([u128::from(read_u64(reader)?), u128::from(read_u64(reader)?)].into())
        }
    }
}

pub(super) fn append_claim(
    bytes: &mut Vec<u8>,
    claim: Option<&PrimeGroupClaim>,
    field: BinaryFieldProfile,
) {
    let coordinate_bytes = match field {
        BinaryFieldProfile::P128 => 16,
        BinaryFieldProfile::P64 => 8,
    };
    let point = claim.map_or(&[][..], |claim| claim.point.as_slice());
    bytes.extend_from_slice(&(point.len() as u64).to_le_bytes());
    for element in point {
        for coordinate in element.coordinates() {
            bytes.extend_from_slice(&coordinate.to_le_bytes()[..coordinate_bytes]);
        }
    }
    let claims = claim.map_or(&[][..], |claim| claim.evaluations.as_slice());
    bytes.extend_from_slice(&(claims.len() as u64).to_le_bytes());
    for claim in claims {
        bytes.extend_from_slice(&(claim.polynomial as u64).to_le_bytes());
        for value in claim.value.coordinates() {
            bytes.extend_from_slice(&value.to_le_bytes()[..coordinate_bytes]);
        }
    }
}
