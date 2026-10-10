use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{commitment::BinaryClearCommitment, lowered::LoweredRootLayout};
use akita_params::sis::labinius::LABINIUS_BALANCED_LOG_BASIS;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Encode a response inside the accepted interval `[lower, upper]` in the
/// canonical response table, as stored base-16 digits in `[0, 15]`.
///
/// A scalar coefficient `v` packed with sign `+1` is stored as `v - lower`,
/// that is, each balanced digit plus 8. Where the packing negates it the
/// stored value is `upper - v`, the digitwise complement. Padding slots of the
/// digit axis and coefficient tails stay zero.
pub fn encode_witness(
    layout: &LoweredRootLayout,
    response: &[[i64; 162]],
) -> Result<Vec<u8>, AkitaError> {
    if response.len() != layout.scalar_rows() {
        return Err(AkitaError::InvalidProof);
    }
    let encoding = layout.encoding();
    let (lower, upper) = encoding.response_interval();
    let lower = i64::try_from(lower).map_err(|_| AkitaError::InvalidProof)?;
    let upper = i64::try_from(upper).map_err(|_| AkitaError::InvalidProof)?;
    let digit_count = encoding.response_digit_count();
    let digit_slots = encoding.response_digit_slots();
    const DIGIT_MASK: u64 = (1 << LABINIUS_BALANCED_LOG_BASIS) - 1;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(layout.degree())
        .map_err(|_| AkitaError::InvalidProof)?;
    for coefficient in 0..layout.degree() {
        let sign =
            i64::try_from(layout.sigma(coefficient)?).map_err(|_| AkitaError::InvalidProof)?;
        let offset =
            i64::try_from(layout.off(coefficient)?).map_err(|_| AkitaError::InvalidProof)?;
        // Both affine endpoints fit, so the arithmetic below cannot overflow.
        for endpoint in [lower, upper] {
            endpoint
                .checked_mul(sign)
                .and_then(|value| value.checked_add(offset))
                .ok_or(AkitaError::InvalidProof)?;
        }
        coefficients.push((
            coefficient / layout.k(),
            coefficient % layout.k(),
            sign,
            offset,
        ));
    }
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.witness_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    table.resize(layout.witness_len(), 0);
    let response_layout = layout.response_layout();
    let block_len = response_layout.polynomial_stride();
    let active_len = checked::product([layout.m(), block_len]).ok_or(AkitaError::InvalidProof)?;
    let active_range =
        checked::range(response_layout.offset(), active_len).ok_or(AkitaError::InvalidProof)?;
    let active = table
        .get_mut(active_range)
        .ok_or(AkitaError::InvalidProof)?;
    // ResponseLayout::address puts digits innermost, then coefficients, then
    // ring elements. Each worker therefore owns one complete element block.
    let encode_element = |(block, rows): (&mut [u8], &[[i64; 162]])| {
        for (digits, &(scalar_coefficient, component, sign, offset)) in
            block.chunks_exact_mut(digit_slots).zip(&coefficients)
        {
            let value = *rows
                .get(component)
                .and_then(|values| values.get(scalar_coefficient))
                .ok_or(AkitaError::InvalidProof)?;
            if value < lower || value > upper {
                return Err(AkitaError::InvalidProof);
            }
            let packed = value * sign + offset;
            let mut unsigned = u64::try_from(packed).map_err(|_| AkitaError::InvalidProof)?;
            for digit in digits.iter_mut().take(digit_count) {
                *digit = (unsigned & DIGIT_MASK) as u8;
                unsigned >>= LABINIUS_BALANCED_LOG_BASIS;
            }
            // The interval's diameter is `16^digit_count - 1`.
            if unsigned != 0 {
                return Err(AkitaError::InvalidProof);
            }
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    active
        .par_chunks_exact_mut(block_len)
        .zip(response.par_chunks_exact(layout.k()))
        .try_for_each(encode_element)?;
    #[cfg(not(feature = "parallel"))]
    active
        .chunks_exact_mut(block_len)
        .zip(response.chunks_exact(layout.k()))
        .try_for_each(encode_element)?;
    Ok(table)
}

/// Encode a clear commitment's residues in the canonical image digit table,
/// as stored base-16 digits in `[0, 15]`.
///
/// This is the table the root reduction's image oracle binds, at commit time
/// or at statement binding. An image coefficient `T`, the canonical residue in
/// `[0, q)`, is stored as the digits of `T + image_offset`, lowest first, that
/// is, each balanced digit plus 8. The address is `digit + image_digit_slots *
/// (coefficient + padded_coefficients * (column * n_A + row))`. Padding slots
/// of the digit axis, coefficient tails and entry padding stay zero.
pub fn encode_image(
    layout: &LoweredRootLayout,
    commitment: &BinaryClearCommitment,
) -> Result<Vec<u8>, AkitaError> {
    let encoding = layout.encoding();
    let degree = layout.degree();
    let digit_count = encoding.image_digit_count();
    let digit_slots = encoding.image_digit_slots();
    let offset = u64::try_from(encoding.image_offset()).map_err(|_| AkitaError::InvalidProof)?;
    let coefficient_count =
        checked::product([layout.image_count(), degree]).ok_or(AkitaError::InvalidProof)?;
    if degree == 0 || commitment.images.len() != coefficient_count {
        return Err(AkitaError::InvalidProof);
    }
    const DIGIT_MASK: u64 = (1 << LABINIUS_BALANCED_LOG_BASIS) - 1;
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.image_len())
        .map_err(|_| AkitaError::InvalidProof)?;
    table.resize(layout.image_len(), 0);
    let entry_len = checked::product([layout.padded_coefficients(), digit_slots])
        .ok_or(AkitaError::InvalidProof)?;
    let active_len =
        checked::product([layout.image_count(), entry_len]).ok_or(AkitaError::InvalidProof)?;
    let active = table
        .get_mut(..active_len)
        .ok_or(AkitaError::InvalidProof)?;
    let encode_entry = |(block, residues): (&mut [u8], &[u32])| {
        for (digits, &residue) in block.chunks_exact_mut(digit_slots).zip(residues) {
            let mut unsigned = u64::from(residue)
                .checked_add(offset)
                .ok_or(AkitaError::InvalidProof)?;
            for digit in digits.iter_mut().take(digit_count) {
                *digit = (unsigned & DIGIT_MASK) as u8;
                unsigned >>= LABINIUS_BALANCED_LOG_BASIS;
            }
            // A canonical residue plus the offset fits the weighted digits.
            if unsigned != 0 {
                return Err(AkitaError::InvalidProof);
            }
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    active
        .par_chunks_exact_mut(entry_len)
        .zip(commitment.images.par_chunks_exact(degree))
        .try_for_each(encode_entry)?;
    #[cfg(not(feature = "parallel"))]
    active
        .chunks_exact_mut(entry_len)
        .zip(commitment.images.chunks_exact(degree))
        .try_for_each(encode_entry)?;
    Ok(table)
}
