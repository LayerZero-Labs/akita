use super::*;

/// Batched negative-binary compression mat-vec over one shared matrix prefix.
#[tracing::instrument(skip_all, name = "compression_rows")]
pub(crate) fn compression_rows<F: FieldCore + CanonicalField, const D: usize>(
    slot: &PreparedNttCache<D>,
    output_rank: usize,
    digit_vectors: &[&[[i8; D]]],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let column_count = validate_compression_rows(output_rank, digit_vectors)?;

    match slot {
        PreparedNttCache::Q32 { neg, params, .. } => {
            compression_rows_with_params(neg, output_rank, column_count, digit_vectors, params)
        }
        PreparedNttCache::Q64 { neg, params, .. } => {
            compression_rows_with_params(neg, output_rank, column_count, digit_vectors, params)
        }
        PreparedNttCache::Q128 { neg, params, .. } => {
            compression_rows_with_params(neg, output_rank, column_count, digit_vectors, params)
        }
    }
}

pub(crate) fn validate_compression_rows<const D: usize>(
    output_rank: usize,
    digit_vectors: &[&[[i8; D]]],
) -> Result<usize, AkitaError> {
    let column_count = digit_vectors.first().map_or(0, |digits| digits.len());
    if output_rank == 0 {
        return Err(AkitaError::InvalidInput(
            "compression output rank must be nonzero".to_string(),
        ));
    }
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

fn compression_rows_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    matrix: &[CyclotomicCrtNtt<W, K, D>],
    output_rank: usize,
    column_count: usize,
    digit_vectors: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let required_matrix_len = output_rank.checked_mul(column_count).ok_or_else(|| {
        AkitaError::InvalidInput("compression matrix prefix length overflow".to_string())
    })?;
    let matrix = matrix.get(..required_matrix_len).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "compression matrix requires {required_matrix_len} ring elements, but setup has {}",
            matrix.len()
        ))
    })?;
    let rows = matrix
        .chunks_exact(column_count)
        .take(output_rank)
        .collect::<Vec<_>>();
    let safe_width =
        safe_crt_chunk_width::<F, W, K, D>(params, column_count, 1).ok_or_else(|| {
            AkitaError::InvalidSetup(
                "compression CRT profile cannot accumulate one signed digit".to_string(),
            )
        })?;
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, 1);

    Ok(drive_block_chunked_matvec(
        digit_vectors.len(),
        output_rank,
        column_count,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accumulators, start, end| {
            for column in start..end {
                for (vector_accumulators, digits) in accumulators.iter_mut().zip(digit_vectors) {
                    let digit = &digits[column];
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let digit_ntt = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (accumulator, row) in vector_accumulators.iter_mut().zip(&rows) {
                        accumulate_pointwise_product_into(
                            accumulator,
                            &row[column],
                            &digit_ntt,
                            params,
                        );
                    }
                }
            }
        },
    ))
}
