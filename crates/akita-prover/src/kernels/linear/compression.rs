use akita_error::AkitaError;

pub(crate) fn validate_compression_batch_shape<const D: usize>(
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
    Ok(column_count)
}
