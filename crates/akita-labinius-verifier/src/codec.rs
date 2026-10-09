//! Fixed-width canonical proof atoms. No proof-controlled length prefixes.
use crate::{channel::ClearChannel, profile::BinaryClearSetup};
use akita_algebra::{
    binary::BinaryField162, fft::SmoothFftField, ring::trinomial::TrinomialModulus,
};
use akita_error::{checked, AkitaError};

/// Append a variable public identity with a u64 little-endian byte length.
pub fn length_prefixed(output: &mut Vec<u8>, value: &[u8]) -> Result<(), AkitaError> {
    let len = u64::try_from(value.len())
        .map_err(|_| AkitaError::InvalidInput("identity length exceeds u64".into()))?;
    let extra = checked::sum([8, value.len()])
        .ok_or_else(|| AkitaError::InvalidInput("identity length overflow".into()))?;
    output
        .try_reserve(extra)
        .map_err(|_| AkitaError::InvalidInput("identity allocation failed".into()))?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(value);
    Ok(())
}

pub fn exchange_binary<S: ClearChannel>(
    channel: &mut S,
    value: &mut BinaryField162,
) -> Result<(), AkitaError> {
    let mut bytes = value.to_bytes();
    channel.message(&mut bytes)?;
    *value = BinaryField162::from_bytes(&bytes).ok_or(AkitaError::InvalidProof)?;
    Ok(())
}

/// Number of bytes in the minimal unsigned offset encoding (at least one).
pub fn response_width(lower: i64, upper: i64) -> Result<usize, AkitaError> {
    if lower > upper {
        return Err(AkitaError::InvalidInput(
            "reversed response interval".into(),
        ));
    }
    let diameter = (i128::from(upper) - i128::from(lower)) as u64;
    Ok(((64 - diameter.leading_zeros()).div_ceil(8) as usize).max(1))
}

/// Encode one signed integer as its offset from the admitted lower endpoint.
pub fn encode_response_coefficient(
    value: i64,
    lower: i64,
    upper: i64,
) -> Result<Vec<u8>, AkitaError> {
    let width = response_width(lower, upper)?;
    if value < lower || value > upper {
        return Err(AkitaError::InvalidInput(
            "integer response outside interval".into(),
        ));
    }
    let offset = (i128::from(value) - i128::from(lower)) as u64;
    Ok(offset.to_le_bytes().into_iter().take(width).collect())
}

/// Decode exactly the minimal width and reject unused offsets above the diameter.
pub fn decode_response_coefficient(
    bytes: &[u8],
    lower: i64,
    upper: i64,
) -> Result<i64, AkitaError> {
    if bytes.len() != response_width(lower, upper)? {
        return Err(AkitaError::InvalidProof);
    }
    let mut offset = [0u8; 8];
    offset
        .get_mut(..bytes.len())
        .ok_or(AkitaError::InvalidProof)?
        .copy_from_slice(bytes);
    let offset = u64::from_le_bytes(offset);
    let diameter = i128::from(upper) - i128::from(lower);
    if i128::from(offset) > diameter {
        return Err(AkitaError::InvalidProof);
    }
    i64::try_from(i128::from(lower) + i128::from(offset)).map_err(|_| AkitaError::InvalidProof)
}

/// Exchange the setup-sized response, enforcing its canonical range encoding.
pub fn exchange_response<
    S: ClearChannel,
    F: SmoothFftField,
    const D: usize,
    M: TrinomialModulus,
>(
    channel: &mut S,
    setup: &BinaryClearSetup<F, D, M>,
    response: &mut Vec<[i64; 162]>,
) -> Result<(), AkitaError> {
    if response.len() != setup.scalar_rows() {
        return Err(AkitaError::InvalidInput(
            "response row count mismatch".into(),
        ));
    }
    let width = response_width(setup.lower(), setup.upper())?;
    // Exchange one fixed-width coefficient at a time. This bounds transient
    // parsing scratch independently of the response's admitted row count.
    for row in response {
        for value in row {
            let mut bytes = encode_response_coefficient(*value, setup.lower(), setup.upper())?;
            if bytes.len() != width {
                return Err(AkitaError::InvalidInput("response width mismatch".into()));
            }
            channel.message(&mut bytes)?;
            *value = decode_response_coefficient(&bytes, setup.lower(), setup.upper())?;
        }
    }
    Ok(())
}

/// Exchange one canonical fixed-width little-endian coefficient field element.
pub fn exchange_field<F: SmoothFftField, S: ClearChannel>(
    channel: &mut S,
    value: &mut F,
) -> Result<(), AkitaError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(F::NUM_BYTES)
        .map_err(|_| AkitaError::InvalidProof)?;
    bytes.resize(F::NUM_BYTES, 0);
    value.to_bytes_le(&mut bytes);
    channel.message(&mut bytes)?;
    *value = F::from_bytes_le_checked(&bytes).ok_or(AkitaError::InvalidProof)?;
    Ok(())
}

/// Exchange a centered e-bit signed integer in unsigned offset form.
/// This encoding enforces exactly the interval used by `LoweredPublic::new`.
pub fn exchange_bounded_integer<S: ClearChannel>(
    channel: &mut S,
    value: &mut i128,
    bits: u32,
) -> Result<(), AkitaError> {
    if bits == 0 || bits >= 128 {
        return Err(AkitaError::InvalidInput(
            "unsupported root integer width".into(),
        ));
    }
    let offset = 1i128
        .checked_shl(bits - 1)
        .ok_or(AkitaError::InvalidProof)?;
    let limit = 1u128.checked_shl(bits).ok_or(AkitaError::InvalidProof)?;
    let encoded = value
        .checked_add(offset)
        .and_then(|x| u128::try_from(x).ok())
        .filter(|&x| x < limit)
        .ok_or(AkitaError::InvalidProof)?;
    let width = checked::div_ceil(bits as usize, 8).ok_or(AkitaError::InvalidProof)?;
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&encoded.to_le_bytes());
    channel.message(bytes.get_mut(..width).ok_or(AkitaError::InvalidProof)?)?;
    // The initialized bytes beyond the wire width are zero for a valid input.
    let decoded = u128::from_le_bytes(bytes);
    if decoded >= limit {
        return Err(AkitaError::InvalidProof);
    }
    *value = i128::try_from(decoded).map_err(|_| AkitaError::InvalidProof)? - offset;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::panic)]
mod tests {
    use super::*;
    #[test]
    fn canonical_interval_offsets() {
        for (lo, hi) in [(-1000, 1000), (i64::MIN, i64::MAX), (0, 0), (-256, 0)] {
            for v in [lo, hi, 0] {
                let bytes = encode_response_coefficient(v, lo, hi).unwrap();
                assert_eq!(decode_response_coefficient(&bytes, lo, hi), Ok(v));
                let mut excess = bytes;
                excess.push(0);
                assert_eq!(
                    decode_response_coefficient(&excess, lo, hi),
                    Err(AkitaError::InvalidProof)
                );
            }
        }
        assert_eq!(
            decode_response_coefficient(&[21], -10, 10),
            Err(AkitaError::InvalidProof)
        );
        assert!(encode_response_coefficient(11, -10, 10).is_err());
        assert_eq!(response_width(-128, 127), Ok(1));
        assert_eq!(response_width(-128, 128), Ok(2));
    }
}
