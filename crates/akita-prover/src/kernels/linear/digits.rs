use super::*;

pub(super) fn mat_vec_mul_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, true>(ntt_mat, blocks, log_basis, params)
}

pub(super) fn mat_vec_mul_dense_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, false>(ntt_mat, blocks, log_basis, params)
}

pub(super) fn mat_vec_mul_digits_i8_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let live_block_count = blocks.len();
    if live_block_count == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; live_block_count];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    debug_assert!(
        blocks
            .iter()
            .all(|block| digit_rows_within_digit_bound::<D>(
                block,
                inner_width.min(block.len()),
                digit_bound
            )),
        "predecomposed digit block contains digits outside its log_basis range"
    );
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && live_block_count >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width == max_data_width
    {
        if inner_width <= safe_width {
            return mat_vec_mul_digits_i8_block_parallel::<F, W, K, D, CHECK_ZERO>(
                ntt_mat,
                blocks,
                digit_bound,
                params,
            );
        }
        return mat_vec_mul_digits_i8_block_parallel_chunked::<F, W, K, D, CHECK_ZERO>(
            ntt_mat,
            blocks,
            inner_width,
            safe_width,
            digit_bound,
            params,
        );
    }

    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    drive_block_chunked_matvec(
        live_block_count,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            if CHECK_ZERO {
                for (block_idx, block) in blocks.iter().enumerate() {
                    if start >= block.len() {
                        continue;
                    }
                    let block_tile_end = end.min(block.len());
                    let tile = &block[start..block_tile_end];
                    for (i, digit) in tile.iter().enumerate() {
                        if is_zero_plane(digit) {
                            continue;
                        }
                        let col = start + i;
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                        }
                    }
                }
            } else {
                for block_idx in 0..live_block_count {
                    let block = blocks[block_idx];
                    if start >= block.len() {
                        continue;
                    }
                    let block_tile_end = end.min(block.len());
                    let tile = &block[start..block_tile_end];
                    for (i, digit) in tile.iter().enumerate() {
                        let col = start + i;
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                        }
                    }
                }
            }
        },
    )
}

pub(super) fn mat_vec_mul_digits_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    live_block_count: usize,
    positions_per_block: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if live_block_count == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(positions_per_block);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; live_block_count];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    debug_assert!(
        digit_rows_within_digit_bound::<D>(
            coeffs,
            inner_width.saturating_mul(live_block_count),
            digit_bound
        ),
        "predecomposed strided digit block contains digits outside its log_basis range"
    );
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && live_block_count >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width <= safe_width
    {
        return mat_vec_mul_digits_i8_strided_block_parallel(
            ntt_mat,
            coeffs,
            live_block_count,
            inner_width,
            digit_bound,
            params,
        );
    }

    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    drive_block_chunked_matvec(
        live_block_count,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            for col in start..end {
                let seq_start = col * live_block_count;
                if seq_start >= coeffs.len() {
                    break;
                }
                let live_blocks = live_block_count.min(coeffs.len() - seq_start);
                let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
                for (block_idx, digit) in coeffs_for_col.iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                }
            }
        },
    )
}
pub(super) fn mat_vec_mul_raw_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    live_block_count: usize,
    positions_per_block: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    if live_block_count == 0 {
        return Ok(vec![]);
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(positions_per_block);
    if inner_width == 0 || n_a == 0 {
        return Ok(vec![
            vec![CyclotomicRing::<F, D>::zero(); n_a];
            live_block_count
        ]);
    }

    // Unlike the balanced-digit paths (bound <= 32, always within capacity),
    // the raw signed-i8 bound is read from the witness and can in principle be
    // large enough that even a single CRT term cannot lift exactly. Reject that
    // at this checked boundary rather than panicking on a `Result` path.
    let rhs_bound = strided_i8_abs_bound(coeffs, live_block_count, inner_width);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, rhs_bound)
        .ok_or_else(|| {
            AkitaError::InvalidInput(
                "raw i8 recursive-witness coefficients exceed the CRT lift range for these parameters"
                    .to_string(),
            )
        })?;
    // Recursive-witness commit shapes are small-row (n_a <= 4). Fan out over
    // blocks whenever that exposes at least as much parallelism as the shared
    // driver's column tiles would: the many-block root gets block fanout, and
    // the deeper few-block levels still beat the 1-2 column tiles their narrow
    // widths produce. Only when blocks are scarce but tiles are plentiful do we
    // fall through to the tiled driver. Requires the full width to fit one CRT
    // lift; over-capacity widths still chunk in the driver.
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS && inner_width <= safe_width {
        let num_tiles = inner_width.div_ceil(base_tile_width::<W, K, D>());
        if live_block_count >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS || live_block_count >= num_tiles
        {
            return Ok(mat_vec_mul_raw_i8_strided_block_parallel(
                ntt_mat,
                coeffs,
                live_block_count,
                inner_width,
                params,
            ));
        }
    }
    Ok(drive_block_chunked_matvec(
        live_block_count,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            accumulate_raw_i8_strided_range(
                accs,
                ntt_mat,
                coeffs,
                live_block_count,
                start,
                end,
                params,
            );
        },
    ))
}

/// Fold-major (block) raw signed-i8 ring mat-vec for `num_digits_commit == 1`.
///
/// Mirrors [`mat_vec_mul_digits_i8_with_params`] exactly in block/column layout
/// and output shape, but treats each `[i8; D]` as a raw signed ring-coefficient
/// vector rather than a balanced gadget digit: it lifts with
/// `from_i8_with_params` (valid for any `i8`) instead of a balanced-digit LUT,
/// and sizes the CRT chunk width from the data-derived coefficient bound. This
/// is the commit path for a recursive witness whose extension-field tensor
/// base-lift packing (`pack_tensor_base_lift_i8_digits`) sums gadget digits and
/// can push coefficients past the balanced range `[-2^(log_basis-1),
/// 2^(log_basis-1))`. Degree-one fields keep the faster balanced-digit kernel.
pub(super) fn mat_vec_mul_raw_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let live_block_count = blocks.len();
    if live_block_count == 0 {
        return Ok(vec![]);
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return Ok(vec![
            vec![CyclotomicRing::<F, D>::zero(); n_a];
            live_block_count
        ]);
    }
    // Read the raw signed-i8 bound directly from the witness. It can in
    // principle be large enough that even a single CRT term cannot lift
    // exactly; reject that at this checked boundary rather than panicking.
    let rhs_bound = blocks
        .iter()
        .flat_map(|block| block.iter().take(inner_width))
        .flat_map(|row| row.iter())
        .map(|&coeff| u64::from(coeff.unsigned_abs()))
        .max()
        .unwrap_or(0);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, rhs_bound)
        .ok_or_else(|| {
            AkitaError::InvalidInput(
                "raw i8 recursive-witness coefficients exceed the CRT lift range for these parameters"
                    .to_string(),
            )
        })?;
    Ok(drive_block_chunked_matvec(
        live_block_count,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(),
        safe_width,
        params,
        |accs, start, end| {
            for block_idx in 0..live_block_count {
                let block = blocks[block_idx];
                if start >= block.len() {
                    continue;
                }
                let block_tile_end = end.min(block.len());
                let tile = &block[start..block_tile_end];
                for (i, coeff) in tile.iter().enumerate() {
                    if is_zero_plane(coeff) {
                        continue;
                    }
                    let col = start + i;
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_params(coeff, params);
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                }
            }
        },
    ))
}

fn strided_i8_abs_bound<const D: usize>(
    coeffs: &[[i8; D]],
    live_block_count: usize,
    inner_width: usize,
) -> u64 {
    let mut bound = 0u64;
    for col in 0..inner_width {
        let Some(seq_start) = col.checked_mul(live_block_count) else {
            break;
        };
        if seq_start >= coeffs.len() {
            break;
        }
        let live_blocks = live_block_count.min(coeffs.len() - seq_start);
        for row in &coeffs[seq_start..seq_start + live_blocks] {
            for &coeff in row {
                bound = bound.max(u64::from(coeff.unsigned_abs()));
            }
        }
    }
    bound
}

#[allow(clippy::too_many_arguments)]
fn accumulate_raw_i8_strided_range<W: PrimeWidth, const K: usize, const D: usize>(
    accs: &mut [Vec<CyclotomicCrtNtt<W, K, D>>],
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    live_block_count: usize,
    tile_start: usize,
    tile_end: usize,
    params: &CrtNttParamSet<W, K, D>,
) {
    for col in tile_start..tile_end {
        let Some(seq_start) = col.checked_mul(live_block_count) else {
            break;
        };
        if seq_start >= coeffs.len() {
            break;
        }
        let live_blocks = live_block_count.min(coeffs.len() - seq_start);
        let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
        for (block_idx, coeff) in coeffs_for_col.iter().enumerate() {
            if is_zero_plane(coeff) {
                continue;
            }
            let ntt_d = CyclotomicCrtNtt::from_i8_with_params(coeff, params);
            for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
            }
        }
    }
}
