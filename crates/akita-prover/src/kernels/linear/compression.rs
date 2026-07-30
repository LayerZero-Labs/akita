use super::*;

/// Batched negative-binary compression mat-vec over one shared rank-one matrix prefix.
///
/// Digits must lie in `{-1, 0}`. Arithmetic is the existing digit mat-vec at
/// `log_basis = 1`; this wrapper only enforces the compression shape contract.
#[tracing::instrument(skip_all, name = "compression_rows")]
pub(crate) fn compression_rows<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    digit_vectors: &[&[[i8; D]]],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let column_count = validate_compression_rows(digit_vectors)?;
    mat_vec_mul_ntt_digits_i8(slot, 1, column_count, digit_vectors, 1)
}

pub(crate) fn validate_compression_rows<const D: usize>(
    digit_vectors: &[&[[i8; D]]],
) -> Result<usize, AkitaError> {
    let column_count = digit_vectors.first().map_or(0, |digits| digits.len());
    if digit_vectors.is_empty() || column_count == 0 {
        return Err(AkitaError::InvalidInput(
            "compression batch must contain nonempty digit vectors".to_string(),
        ));
    }
    if digit_vectors
        .iter()
        .any(|digits| digits.len() != column_count)
    {
        return Err(AkitaError::InvalidInput(
            "compression batch digit vectors must have one exact shape".to_string(),
        ));
    }
    if digit_vectors
        .iter()
        .any(|digits| !digit_rows_within_digit_bound(digits, column_count, 1))
    {
        return Err(AkitaError::InvalidInput(
            "compression batch contains a digit outside {-1,0}".to_string(),
        ));
    }
    Ok(column_count)
}
