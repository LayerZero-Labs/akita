use super::*;

pub(super) fn mat_vec_mul_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, true>(ntt_mat, blocks, params)
}

pub(super) fn mat_vec_mul_dense_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, false>(ntt_mat, blocks, params)
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

    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, inner_width);
    if inner_width <= chunk_width
        && n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
        return mat_vec_mul_digits_i8_block_parallel::<F, W, K, D, CHECK_ZERO>(
            ntt_mat, blocks, params,
        );
    }

    let lut = DigitMontLut::new(params);
    let cache_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>()))
        .max(1)
        .min(chunk_width);
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
                    add_ring_into(&mut accs[block_idx][row], partial);
                }
            }
            accs
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    add_ring_into(&mut a[block_idx][row], b[block_idx][row]);
                }
            }
            a
        }
    );

    final_accs
}

pub(super) fn mat_vec_mul_digits_i8_strided_with_params<
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

    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, inner_width);
    if inner_width <= chunk_width
        && n_a <= SMALL_ROW_BLOCK_PARALLEL_MAX_ROWS
        && num_blocks >= SMALL_ROW_BLOCK_PARALLEL_MIN_BLOCKS
    {
        return mat_vec_mul_digits_i8_strided_block_parallel(
            ntt_mat,
            coeffs,
            num_blocks,
            inner_width,
            params,
        );
    }

    let lut = DigitMontLut::new(params);
    let cache_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>()))
        .max(1)
        .min(chunk_width);
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
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in chunk_accs[block_idx].iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                        }
                    }
                }
            }
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    let partial = chunk_accs[block_idx][row].to_ring_with_params(params);
                    add_ring_into(&mut accs[block_idx][row], partial);
                }
            }
            accs
        },
        |mut a: Vec<Vec<CyclotomicRing<F, D>>>, b| {
            for block_idx in 0..num_blocks {
                for row in 0..n_a {
                    add_ring_into(&mut a[block_idx][row], b[block_idx][row]);
                }
            }
            a
        }
    );

    final_accs
}
