//! Integer module coordinates for `Z[Y]/(Y^648 - Y^324 + 1)`.
//!
//! Over `S = Z[Z]/(Z^162 + Z^81 + 1)`, the basis is `1,Y,Y²,Y³`
//! and `Z` embeds as `-Y^4`. Coefficients are in increasing degree order.
//! These maps are coefficient maps, not a switch to a negacyclic ring.

use akita_error::AkitaError;

/// Pack four canonical integer S components into R.
///
/// Coefficient `4*s+a` is `(-1)^s * components[a][s]`. Negating
/// `i128::MIN` at an odd position is rejected rather than wrapping.
pub fn pack_signed_components(components: &[[i128; 162]]) -> Result<[i128; 648], AkitaError> {
    let components: &[[i128; 162]; 4] =
        components.try_into().map_err(|_| AkitaError::InvalidSize {
            expected: 4,
            actual: components.len(),
        })?;
    let mut result = [0; 648];
    for (s, block) in result.chunks_exact_mut(4).enumerate() {
        for (destination, component) in block.iter_mut().zip(components) {
            let value = component.get(s).ok_or(AkitaError::InvalidProof)?;
            *destination = if s % 2 == 0 {
                *value
            } else {
                value.checked_neg().ok_or_else(|| {
                    AkitaError::InvalidInput("signed module coefficient overflows i128".into())
                })?
            };
        }
    }
    Ok(result)
}

/// Undo the signed interleaving of a canonical degree-648 R element.
///
/// Rejects a wrong coefficient count and unrepresentable sign changes.
pub fn unpack_signed_components(coefficients: &[i128]) -> Result<[[i128; 162]; 4], AkitaError> {
    let coefficients: &[i128; 648] =
        coefficients
            .try_into()
            .map_err(|_| AkitaError::InvalidSize {
                expected: 648,
                actual: coefficients.len(),
            })?;
    let mut result = [[0; 162]; 4];
    for (s, block) in coefficients.chunks_exact(4).enumerate() {
        for (component, value) in result.iter_mut().zip(block) {
            let destination = component.get_mut(s).ok_or(AkitaError::InvalidProof)?;
            *destination = if s % 2 == 0 {
                *value
            } else {
                value.checked_neg().ok_or_else(|| {
                    AkitaError::InvalidInput("signed module coefficient overflows i128".into())
                })?
            };
        }
    }
    Ok(result)
}

/// Reduce a canonical S element coefficient-wise modulo two into packed F162.
///
/// This is the quotient homomorphism `S -> GF(2^162)`. Negative odd
/// coefficients have parity one. A noncanonical coefficient count is rejected.
#[cfg(feature = "labinius-binary")]
pub fn scalar_to_binary(
    coefficients: &[i128],
) -> Result<crate::binary::BinaryField162, AkitaError> {
    let coefficients: &[i128; 162] =
        coefficients
            .try_into()
            .map_err(|_| AkitaError::InvalidSize {
                expected: 162,
                actual: coefficients.len(),
            })?;
    let mut bytes = [0u8; 21];
    for (chunk, byte) in coefficients.chunks(8).zip(bytes.iter_mut()) {
        for (bit, coefficient) in chunk.iter().enumerate() {
            *byte |= ((coefficient & 1) as u8) << bit;
        }
    }
    crate::binary::BinaryField162::from_bytes(&bytes).ok_or(AkitaError::InvalidProof)
}

#[cfg(test)]
mod tests;
