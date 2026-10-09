use akita_algebra::{SmoothFftField, TrinomialModulus};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::{commitment::BinaryClearCommitment, lowered::LoweredRootLayout};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    let lower = i64::try_from(lower).map_err(|_| AkitaError::InvalidProof)?;
    let upper = i64::try_from(upper).map_err(|_| AkitaError::InvalidProof)?;
    let digit_bits = encoding.base().bits();
    let mask = 1u64
        .checked_shl(digit_bits)
        .and_then(|value| value.checked_sub(1))
        .ok_or(AkitaError::InvalidProof)?;
    if mask > u64::from(u8::MAX) {
        return Err(AkitaError::InvalidProof);
    }
    let limit = 1u64
        .checked_shl(range.bits())
        .ok_or(AkitaError::InvalidProof)?;
    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(layout.degree())
        .map_err(|_| AkitaError::InvalidProof)?;
    for coefficient in 0..layout.degree() {
        let sign =
            i64::try_from(layout.sigma(coefficient)?).map_err(|_| AkitaError::InvalidProof)?;
        let offset =
            i64::try_from(layout.off(coefficient)?).map_err(|_| AkitaError::InvalidProof)?;
        // The closed root profile admits [-2^15, 2^15 - 1]. Its signed
        // packing uses offsets 2^15 or 2^15 - 1, giving [0, 2^16 - 1].
        // Checking both affine endpoints also establishes that i64 arithmetic
        // below cannot overflow if another admitted profile is introduced.
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
        for (digits, &(scalar_coefficient, component, sign, offset)) in block
            .chunks_exact_mut(range.digit_count())
            .zip(&coefficients)
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
            if unsigned >= limit {
                return Err(AkitaError::InvalidProof);
            }
            for digit in digits {
                // The checked mask fits u8 and its construction proves that
                // digit_bits is below 64, so neither operation can reject.
                *digit = (unsigned & mask) as u8;
                unsigned >>= digit_bits;
            }
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
    let active_len = checked::product([layout.image_count(), layout.padded_coefficients()])
        .ok_or(AkitaError::InvalidProof)?;
    let active = table
        .get_mut(..active_len)
        .ok_or(AkitaError::InvalidProof)?;
    let copy_image = |(block, coefficients): (&mut [F], &[F])| {
        block
            .get_mut(..D)
            .ok_or(AkitaError::InvalidProof)?
            .copy_from_slice(coefficients);
        Ok::<_, AkitaError>(())
    };
    #[cfg(feature = "parallel")]
    {
        // Project only field coefficients into the parallel traversal; the
        // public API does not require the modulus marker type to be Sync.
        let mut images: Vec<&[F]> = Vec::new();
        images
            .try_reserve_exact(commitment.images.len())
            .map_err(|_| AkitaError::InvalidProof)?;
        for image in &commitment.images {
            images.push(image.coefficients());
        }
        active
            .par_chunks_exact_mut(layout.padded_coefficients())
            .zip(images.into_par_iter())
            .try_for_each(copy_image)?;
    }
    #[cfg(not(feature = "parallel"))]
    active
        .chunks_exact_mut(layout.padded_coefficients())
        .zip(
            commitment
                .images
                .iter()
                .map(|image| image.coefficients().as_slice()),
        )
        .try_for_each(copy_image)?;
    Ok(table)
}
