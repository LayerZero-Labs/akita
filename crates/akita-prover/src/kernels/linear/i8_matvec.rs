use super::*;

pub(super) fn mat_vec_mul_i8_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = blocks.len();
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks
        .iter()
        .map(|b| b.len() * num_digits)
        .max()
        .unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, inner_width);
    if inner_width <= chunk_width
        && n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
        return if CHECK_ZERO {
            mat_vec_mul_i8_block_parallel_with_params(
                ntt_mat, blocks, num_digits, log_basis, params,
            )
        } else {
            mat_vec_mul_i8_dense_block_parallel_with_params(
                ntt_mat, blocks, num_digits, log_basis, params,
            )
        };
    }

    let lut = DigitMontLut::new(params);
    let raw_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    if inner_width <= chunk_width {
        let tw = aligned_i8_tile_width(raw_tw, inner_width, num_digits);
        let num_tiles = inner_width.div_ceil(tw);

        let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
            |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(inner_width);
                let ring_start = tile_start / num_digits;
                let ring_end = ((tile_end - 1) / num_digits) + 1;
                let digit_offset = tile_start - ring_start * num_digits;
                let tile_len = tile_end - tile_start;

                for block_idx in 0..num_blocks {
                    let block = blocks[block_idx];
                    if ring_start >= block.len() {
                        continue;
                    }
                    let block_ring_end = ring_end.min(block.len());
                    let partial_coeffs = &block[ring_start..block_ring_end];
                    let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                    let available = all_digits.len().saturating_sub(digit_offset);
                    let n = tile_len.min(available);

                    for (j, digit) in all_digits[digit_offset..digit_offset + n]
                        .iter()
                        .enumerate()
                    {
                        if CHECK_ZERO && is_zero_plane(digit) {
                            continue;
                        }
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(
                                acc,
                                &mat_row[tile_start + j],
                                &ntt_d,
                                params,
                            );
                        }
                    }
                }
                accs
            },
            |mut a: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, b| {
                for block_idx in 0..num_blocks {
                    for row in 0..n_a {
                        add_ntt_into(&mut a[block_idx][row], &b[block_idx][row], params);
                    }
                }
                a
            }
        );

        return cfg_into_iter!(final_accs)
            .map(|row_accs| {
                row_accs
                    .into_iter()
                    .map(|acc| acc.to_ring_with_params(params))
                    .collect()
            })
            .collect();
    }

    // Keep full tiles on ring boundaries so on-the-fly decomposition does not
    // re-expand the same ring when adjacent tiles meet mid digit-pack.
    let chunk_width = bounded_i8_tile_width(chunk_width, inner_width, num_digits);
    let cache_tw = bounded_i8_tile_width(raw_tw, chunk_width, num_digits);
    let num_chunks = inner_width.div_ceil(chunk_width);

    let final_accs: Vec<Vec<CyclotomicRing<F, D>>> = cfg_fold_reduce!(
        0..num_chunks,
        || vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks],
        |mut accs: Vec<Vec<CyclotomicRing<F, D>>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(inner_width);
            let mut chunk_accs = vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks];

            for tile_start in (chunk_start..chunk_end).step_by(cache_tw) {
                let tile_end = (tile_start + cache_tw).min(chunk_end);
                let ring_start = tile_start / num_digits;
                let ring_end = ((tile_end - 1) / num_digits) + 1;
                let digit_offset = tile_start - ring_start * num_digits;
                let tile_len = tile_end - tile_start;

                for block_idx in 0..num_blocks {
                    let block = blocks[block_idx];
                    if ring_start >= block.len() {
                        continue;
                    }
                    let block_ring_end = ring_end.min(block.len());
                    let partial_coeffs = &block[ring_start..block_ring_end];
                    let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                    let available = all_digits.len().saturating_sub(digit_offset);
                    let n = tile_len.min(available);

                    for (j, digit) in all_digits[digit_offset..digit_offset + n]
                        .iter()
                        .enumerate()
                    {
                        if CHECK_ZERO && is_zero_plane(digit) {
                            continue;
                        }
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in chunk_accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(
                                acc,
                                &mat_row[tile_start + j],
                                &ntt_d,
                                params,
                            );
                        }
                    }
                }
            }
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    let partial = chunk_accs[block_idx][row].to_ring_with_params(params);
                    accs[block_idx][row] += partial;
                }
            }
            accs
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    a[block_idx][row] += b[block_idx][row];
                }
            }
            a
        }
    );

    final_accs
}

pub(super) fn mat_vec_mul_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params_impl::<F, W, K, D, true>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

pub(super) fn mat_vec_mul_i8_dense_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_i8_with_params_impl::<F, W, K, D, false>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

pub(super) fn mat_vec_mul_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[CyclotomicRing<F, D>],
    num_blocks: usize,
    block_len: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len.saturating_mul(num_digits));
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, inner_width);
    if inner_width <= chunk_width
        && n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
        return mat_vec_mul_i8_strided_block_parallel_with_params(
            ntt_mat, coeffs, num_blocks, block_len, num_digits, log_basis, params,
        );
    }

    let lut = DigitMontLut::new(params);
    let raw_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    if inner_width <= chunk_width {
        let tw = aligned_i8_tile_width(raw_tw, inner_width, num_digits);
        let num_tiles = inner_width.div_ceil(tw);

        let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
            |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(inner_width);
                let ring_start = tile_start / num_digits;
                let ring_end = ((tile_end - 1) / num_digits) + 1;
                let digit_offset = tile_start - ring_start * num_digits;
                let tile_len = tile_end - tile_start;

                for (block_idx, block_accs) in accs.iter_mut().enumerate() {
                    let mut partial_coeffs =
                        Vec::with_capacity(ring_end.saturating_sub(ring_start));
                    for col in ring_start..ring_end {
                        let seq = block_idx + col * num_blocks;
                        let Some(coeff) = coeffs.get(seq) else {
                            break;
                        };
                        partial_coeffs.push(*coeff);
                    }
                    if partial_coeffs.is_empty() {
                        continue;
                    }

                    let all_digits = decompose_block_i8(&partial_coeffs, num_digits, log_basis);
                    let available = all_digits.len().saturating_sub(digit_offset);
                    let n = tile_len.min(available);

                    for (j, digit) in all_digits[digit_offset..digit_offset + n]
                        .iter()
                        .enumerate()
                    {
                        if is_zero_plane(digit) {
                            continue;
                        }
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in block_accs.iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(
                                acc,
                                &mat_row[tile_start + j],
                                &ntt_d,
                                params,
                            );
                        }
                    }
                }
                accs
            },
            |mut a: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, b| {
                for block_idx in 0..num_blocks {
                    for row in 0..n_a {
                        add_ntt_into(&mut a[block_idx][row], &b[block_idx][row], params);
                    }
                }
                a
            }
        );

        return cfg_into_iter!(final_accs)
            .map(|row_accs| {
                row_accs
                    .into_iter()
                    .map(|acc| acc.to_ring_with_params(params))
                    .collect()
            })
            .collect();
    }

    // Keep full tiles on ring boundaries so on-the-fly decomposition does not
    // re-expand the same ring when adjacent tiles meet mid digit-pack.
    let chunk_width = bounded_i8_tile_width(chunk_width, inner_width, num_digits);
    let cache_tw = bounded_i8_tile_width(raw_tw, chunk_width, num_digits);
    let num_chunks = inner_width.div_ceil(chunk_width);

    let final_accs: Vec<Vec<CyclotomicRing<F, D>>> = cfg_fold_reduce!(
        0..num_chunks,
        || vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks],
        |mut accs: Vec<Vec<CyclotomicRing<F, D>>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(inner_width);
            let mut chunk_accs = vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks];

            for tile_start in (chunk_start..chunk_end).step_by(cache_tw) {
                let tile_end = (tile_start + cache_tw).min(chunk_end);
                let ring_start = tile_start / num_digits;
                let ring_end = ((tile_end - 1) / num_digits) + 1;
                let digit_offset = tile_start - ring_start * num_digits;
                let tile_len = tile_end - tile_start;

                for (block_idx, block_chunk_accs) in chunk_accs.iter_mut().enumerate() {
                    let mut partial_coeffs =
                        Vec::with_capacity(ring_end.saturating_sub(ring_start));
                    for col in ring_start..ring_end {
                        let seq = block_idx + col * num_blocks;
                        let Some(coeff) = coeffs.get(seq) else {
                            break;
                        };
                        partial_coeffs.push(*coeff);
                    }
                    if partial_coeffs.is_empty() {
                        continue;
                    }

                    let all_digits = decompose_block_i8(&partial_coeffs, num_digits, log_basis);
                    let available = all_digits.len().saturating_sub(digit_offset);
                    let n = tile_len.min(available);

                    for (j, digit) in all_digits[digit_offset..digit_offset + n]
                        .iter()
                        .enumerate()
                    {
                        if is_zero_plane(digit) {
                            continue;
                        }
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in block_chunk_accs.iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(
                                acc,
                                &mat_row[tile_start + j],
                                &ntt_d,
                                params,
                            );
                        }
                    }
                }
            }
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    let partial = chunk_accs[block_idx][row].to_ring_with_params(params);
                    accs[block_idx][row] += partial;
                }
            }
            accs
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    a[block_idx][row] += b[block_idx][row];
                }
            }
            a
        }
    );

    final_accs
}
