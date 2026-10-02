//! Canonical compression digit emission shared with portable witness patches.

use akita_algebra::{
    balanced_decompose_coefficients_pow2_i8_into, ring::cyclotomic::BalancedDecomposePow2Params,
};
use akita_error::{checked, AkitaError};
use akita_params::{CompressionWitnessSpan, PackedNegativeBinary, WitnessQuotientRowLayout};
use jolt_field::{CanonicalEncoding, Field};

pub(crate) fn emit_packed_negative_binary(
    mut write: impl FnMut(usize, &[i8]) -> Result<(), AkitaError>,
    span: &CompressionWitnessSpan,
    packed: &PackedNegativeBinary,
) -> Result<(), AkitaError> {
    if packed.map() != span.map() || span.range().len() != packed.map().padded_digit_count() {
        return Err(AkitaError::Internal(
            "packed compression witness map or extent differs from its span".into(),
        ));
    }
    let range = span.range();
    const CHUNK: usize = 4096;
    let mut scratch = [0i8; CHUNK];
    let mut written = 0usize;
    while written < range.len() {
        let count = CHUNK.min(range.len() - written);
        scratch[..count].fill(0);
        for (offset, coefficient) in scratch[..count].iter_mut().enumerate() {
            let linear = written + offset;
            if linear < packed.map().real_digit_count()
                && packed.bytes()[linear / 8] >> (linear % 8) & 1 == 1
            {
                *coefficient = -1;
            }
        }
        write(range.start + written, &scratch[..count])?;
        written += count;
    }
    Ok(())
}

pub(crate) fn quotient_digits<F: Field + CanonicalEncoding>(
    coefficients: &[F],
    row: &WitnessQuotientRowLayout,
    params: &BalancedDecomposePow2Params<F>,
) -> Result<Vec<i8>, AkitaError> {
    let expected_len =
        checked::product([params.levels(), row.geometry().physical_coefficient_width()])
            .ok_or_else(|| AkitaError::Internal("R witness row length overflow".into()))?;
    if row.range().len() != expected_len {
        return Err(AkitaError::Internal(format!(
            "quotient tail row extent mismatch: expected {expected_len}, actual {}",
            row.range().len(),
        )));
    }
    if coefficients.len() != row.geometry().physical_coefficient_width() {
        return Err(AkitaError::Internal(format!(
            "quotient tail row width mismatch: expected {}, actual {}",
            row.geometry().physical_coefficient_width(),
            coefficients.len(),
        )));
    }
    let mut digits = vec![0i8; expected_len];
    balanced_decompose_coefficients_pow2_i8_into(coefficients, &mut digits, params);
    Ok(digits)
}
