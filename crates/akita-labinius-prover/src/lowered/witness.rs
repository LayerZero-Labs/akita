use akita_algebra::{SmoothFftField, TrinomialModulus};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{commitment::BinaryClearCommitment, lowered::LoweredRootLayout};

/// Encode the exact accepted scalar interval in the canonical response table.
///
/// Sign-flipped packed coefficients use offset `off - 1`; hence their digits
/// complement the ordinary signed encoding. Honest coefficient tails are zero.
pub fn encode_witness(
    layout: &LoweredRootLayout,
    response: &[[i64; 162]],
) -> Result<Vec<u8>, AkitaError> {
    if response.len() != layout.scalar_rows() {
        return Err(AkitaError::InvalidProof);
    }
    let encoding = layout.encoding();
    let range = encoding.response();
    let (lower, upper) = range.interval();
    let digit_bits = encoding.base().bits();
    let mask = 1u128
        .checked_shl(digit_bits)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    let limit = 1u128
        .checked_shl(range.bits())
        .ok_or(AkitaError::InvalidProof)?;
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.witness_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    table.resize(layout.witness_len(), 0);
    for element in 0..layout.m() {
        for coefficient in 0..layout.degree() {
            let scalar_coefficient = coefficient / layout.k();
            let component = coefficient % layout.k();
            let row =
                checked::mul_add(element, layout.k(), component).ok_or(AkitaError::InvalidProof)?;
            let value = i128::from(
                *response
                    .get(row)
                    .and_then(|values| values.get(scalar_coefficient))
                    .ok_or(AkitaError::InvalidProof)?,
            );
            if value < lower || value > upper {
                return Err(AkitaError::InvalidProof);
            }
            let offset = layout.off(coefficient)?;
            let packed = value
                .checked_mul(layout.sigma(coefficient)?)
                .and_then(|value| value.checked_add(offset))
                .ok_or(AkitaError::InvalidProof)?;
            let mut unsigned = u128::try_from(packed).map_err(|_| AkitaError::InvalidProof)?;
            if unsigned >= limit {
                return Err(AkitaError::InvalidProof);
            }
            for digit in 0..range.digit_count() {
                let address = layout
                    .response_layout()
                    .address(element, digit, coefficient)?;
                *table.get_mut(address).ok_or(AkitaError::InvalidProof)? =
                    u8::try_from(unsigned & mask).map_err(|_| AkitaError::InvalidProof)?;
                unsigned = unsigned
                    .checked_shr(digit_bits)
                    .ok_or(AkitaError::InvalidProof)?;
            }
            if unsigned != 0 {
                return Err(AkitaError::InvalidProof);
            }
        }
    }
    Ok(table)
}

/// Flatten commitment coefficients into the admitted padded image domain.
pub fn flatten_image<F: SmoothFftField, const D: usize, M: TrinomialModulus>(
    layout: &LoweredRootLayout,
    commitment: &BinaryClearCommitment<F, D, M>,
) -> Result<Vec<F>, AkitaError> {
    if D != layout.degree() || commitment.images.len() != layout.image_count() {
        return Err(AkitaError::InvalidProof);
    }
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.image_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    table.resize(layout.image_len(), F::zero());
    for (element, image) in commitment.images.iter().enumerate() {
        for (coefficient, &value) in image.coefficients().iter().enumerate() {
            let address = layout.image_address(element, coefficient)?;
            *table.get_mut(address).ok_or(AkitaError::InvalidProof)? = value;
        }
    }
    Ok(table)
}
