use super::*;

pub(super) fn mat_vec_mul_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const L: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    lut_len: DigitLutLen<L>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, true, L>(
        ntt_mat, blocks, log_basis, lut_len, params,
    )
}

pub(super) fn mat_vec_mul_dense_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const L: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    lut_len: DigitLutLen<L>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, false, L>(
        ntt_mat, blocks, log_basis, lut_len, params,
    )
}

pub(super) fn mat_vec_mul_digits_i8_with_params_impl<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
    const L: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    lut_len: DigitLutLen<L>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = blocks.len();
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    debug_assert!(
        blocks
            .iter()
            .all(|block| digit_rows_within_lut_range::<D, L>(block, inner_width.min(block.len()))),
        "predecomposed digit block contains digits outside its log_basis range"
    );
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width == max_data_width
        && inner_width <= safe_width
    {
        return mat_vec_mul_digits_i8_block_parallel::<F, W, K, D, CHECK_ZERO, L>(
            ntt_mat, blocks, lut_len, params,
        );
    }

    let lut = DigitMontLut::<W, K, L>::new(params);
    if inner_width <= safe_width {
        let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
        let num_tiles = inner_width.div_ceil(tw);

        let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
            |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(inner_width);

                for block_idx in 0..num_blocks {
                    let block = blocks[block_idx];
                    if tile_start >= block.len() {
                        continue;
                    }
                    let block_tile_end = tile_end.min(block.len());
                    for (j, digit) in block[tile_start..block_tile_end].iter().enumerate() {
                        if CHECK_ZERO && is_zero_plane(digit) {
                            continue;
                        }
                        let ntt_d = unsafe {
                            CyclotomicCrtNtt::from_i8_with_lut_unchecked(digit, params, &lut)
                        };
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

    let num_chunks = inner_width.div_ceil(safe_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks],
        |mut out: Vec<Vec<CyclotomicRing<F, D>>>, chunk_idx| {
            let tile_start = chunk_idx * safe_width;
            let tile_end = (tile_start + safe_width).min(inner_width);
            let mut accs = vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks];

            for block_idx in 0..num_blocks {
                let block = blocks[block_idx];
                if tile_start >= block.len() {
                    continue;
                }
                let block_tile_end = tile_end.min(block.len());
                for (j, digit) in block[tile_start..block_tile_end].iter().enumerate() {
                    if CHECK_ZERO && is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = unsafe {
                        CyclotomicCrtNtt::from_i8_with_lut_unchecked(digit, params, &lut)
                    };
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

            for (out_block, acc_block) in out.iter_mut().zip(accs) {
                for (dst, acc) in out_block.iter_mut().zip(acc_block) {
                    *dst += acc.to_ring_with_params(params);
                }
            }
            out
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for (a_block, b_block) in a.iter_mut().zip(b) {
                for (dst, src) in a_block.iter_mut().zip(b_block) {
                    *dst += src;
                }
            }
            a
        }
    )
}

pub(super) fn mat_vec_mul_digits_i8_strided_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const L: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    coeffs: &[[i8; D]],
    num_blocks: usize,
    block_len: usize,
    log_basis: u32,
    lut_len: DigitLutLen<L>,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    debug_assert!(
        digit_rows_within_lut_range::<D, L>(coeffs, inner_width.saturating_mul(num_blocks)),
        "predecomposed strided digit block contains digits outside its log_basis range"
    );
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width <= safe_width
    {
        return mat_vec_mul_digits_i8_strided_block_parallel(
            ntt_mat,
            coeffs,
            num_blocks,
            inner_width,
            lut_len,
            params,
        );
    }

    let lut = DigitMontLut::<W, K, L>::new(params);
    if inner_width <= safe_width {
        let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
        let num_tiles = inner_width.div_ceil(tw);

        let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
            |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(inner_width);

                for col in tile_start..tile_end {
                    let seq_start = col * num_blocks;
                    if seq_start >= coeffs.len() {
                        break;
                    }
                    let live_blocks = num_blocks.min(coeffs.len() - seq_start);
                    let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
                    for (block_idx, digit) in coeffs_for_col.iter().enumerate() {
                        if is_zero_plane(digit) {
                            continue;
                        }
                        let ntt_d = unsafe {
                            CyclotomicCrtNtt::from_i8_with_lut_unchecked(digit, params, &lut)
                        };
                        for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
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

    let num_chunks = inner_width.div_ceil(safe_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks],
        |mut out: Vec<Vec<CyclotomicRing<F, D>>>, chunk_idx| {
            let tile_start = chunk_idx * safe_width;
            let tile_end = (tile_start + safe_width).min(inner_width);
            let mut accs = vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks];

            for col in tile_start..tile_end {
                let seq_start = col * num_blocks;
                if seq_start >= coeffs.len() {
                    break;
                }
                let live_blocks = num_blocks.min(coeffs.len() - seq_start);
                let coeffs_for_col = &coeffs[seq_start..seq_start + live_blocks];
                for (block_idx, digit) in coeffs_for_col.iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = unsafe {
                        CyclotomicCrtNtt::from_i8_with_lut_unchecked(digit, params, &lut)
                    };
                    for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                }
            }

            for (out_block, acc_block) in out.iter_mut().zip(accs) {
                for (dst, acc) in out_block.iter_mut().zip(acc_block) {
                    *dst += acc.to_ring_with_params(params);
                }
            }
            out
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for (a_block, b_block) in a.iter_mut().zip(b) {
                for (dst, src) in a_block.iter_mut().zip(b_block) {
                    *dst += src;
                }
            }
            a
        }
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
    num_blocks: usize,
    block_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    if num_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(block_len);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks];
    }

    let rhs_bound = strided_i8_abs_bound(coeffs, num_blocks, inner_width);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, rhs_bound)
        .expect("single raw i8 CRT term must fit supported parameters");
    if inner_width <= safe_width {
        let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
        let num_tiles = inner_width.div_ceil(tw);

        let final_accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks],
            |mut accs: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(inner_width);

                accumulate_raw_i8_strided_range(
                    &mut accs, ntt_mat, coeffs, num_blocks, tile_start, tile_end, params,
                );
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

    let num_chunks = inner_width.div_ceil(safe_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_blocks],
        |mut out: Vec<Vec<CyclotomicRing<F, D>>>, chunk_idx| {
            let tile_start = chunk_idx * safe_width;
            let tile_end = (tile_start + safe_width).min(inner_width);
            let mut accs = vec![vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a]; num_blocks];

            accumulate_raw_i8_strided_range(
                &mut accs, ntt_mat, coeffs, num_blocks, tile_start, tile_end, params,
            );

            for (out_block, acc_block) in out.iter_mut().zip(accs) {
                for (dst, acc) in out_block.iter_mut().zip(acc_block) {
                    *dst += acc.to_ring_with_params(params);
                }
            }
            out
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for (a_block, b_block) in a.iter_mut().zip(b) {
                for (dst, src) in a_block.iter_mut().zip(b_block) {
                    *dst += src;
                }
            }
            a
        }
    )
}

fn strided_i8_abs_bound<const D: usize>(
    coeffs: &[[i8; D]],
    num_blocks: usize,
    inner_width: usize,
) -> u64 {
    let mut bound = 0u64;
    for col in 0..inner_width {
        let Some(seq_start) = col.checked_mul(num_blocks) else {
            break;
        };
        if seq_start >= coeffs.len() {
            break;
        }
        let live_blocks = num_blocks.min(coeffs.len() - seq_start);
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
    num_blocks: usize,
    tile_start: usize,
    tile_end: usize,
    params: &CrtNttParamSet<W, K, D>,
) {
    for col in tile_start..tile_end {
        let Some(seq_start) = col.checked_mul(num_blocks) else {
            break;
        };
        if seq_start >= coeffs.len() {
            break;
        }
        let live_blocks = num_blocks.min(coeffs.len() - seq_start);
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
