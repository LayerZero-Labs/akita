use super::*;

pub(super) struct I8ColumnScratch<W: PrimeWidth, const K: usize, const D: usize> {
    rhs: [[MontCoeff<W>; D]; K],
    lazy_dot: [[MontCoeff<W>; D]; I32_LAZY_DOT_BATCH],
}

impl<W: PrimeWidth, const K: usize, const D: usize> I8ColumnScratch<W, K, D> {
    pub(super) fn new() -> Self {
        Self {
            rhs: [[MontCoeff::from_raw(W::default()); D]; K],
            lazy_dot: [[MontCoeff::from_raw(W::default()); D]; I32_LAZY_DOT_BATCH],
        }
    }
}

pub(super) fn accumulate_i8_columns<
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    accs: &mut [CyclotomicCrtNtt<W, K, D>],
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    column_start: usize,
    digits: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
    lut: &DigitMontLut<W, K>,
    scratch: &mut I8ColumnScratch<W, K, D>,
) {
    let batch_size = params.pointwise_dot_batch_size();
    if batch_size == 1 {
        for (offset, digit) in digits.iter().enumerate() {
            if CHECK_ZERO && is_zero_plane(digit) {
                continue;
            }
            CyclotomicCrtNtt::add_assign_col_pointwise_mul_i8_multi_with_lut_scratch(
                accs,
                ntt_mat,
                column_start + offset,
                digit,
                params,
                lut,
                &mut scratch.rhs,
            );
        }
        return;
    }

    for batch_start in (0..digits.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(digits.len());
        let batch = &digits[batch_start..batch_end];
        if CHECK_ZERO && batch.iter().any(is_zero_plane) {
            for (offset, digit) in batch.iter().enumerate() {
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_col_pointwise_mul_i8_multi_with_lut_scratch(
                    accs,
                    ntt_mat,
                    column_start + batch_start + offset,
                    digit,
                    params,
                    lut,
                    &mut scratch.rhs,
                );
            }
            continue;
        }

        if batch.len() == 1 {
            CyclotomicCrtNtt::add_assign_col_pointwise_mul_i8_multi_with_lut_scratch(
                accs,
                ntt_mat,
                column_start + batch_start,
                &batch[0],
                params,
                lut,
                &mut scratch.rhs,
            );
            continue;
        }

        CyclotomicCrtNtt::add_assign_col_pointwise_dot_i8_multi_with_lut_scratch(
            accs,
            ntt_mat,
            column_start + batch_start,
            batch,
            params,
            lut,
            &mut scratch.lazy_dot,
        );
    }
}

/// Block-parallel fast path for small `n_a` and many blocks.
///
/// Parallelizes over blocks (high fanout) instead of column tiles (low fanout).
/// With many blocks but few matrix rows, the old tile-based approach had limited
/// parallelism (few tiles) while this path gives num_live_blocks-way parallelism.
pub(super) fn mat_vec_mul_digits_i8_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    digit_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            let mut scratch = I8ColumnScratch::new();

            accumulate_i8_columns::<W, K, D, CHECK_ZERO>(
                &mut accs,
                ntt_mat,
                0,
                block,
                params,
                &lut,
                &mut scratch,
            );

            accs.into_iter().map(|acc| acc.to_ring(params)).collect()
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_block_parallel_chunked<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    inner_width: usize,
    chunk_width: usize,
    digit_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert!(chunk_width > 0);
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);

    cfg_into_iter!(blocks)
        .map(|block| {
            let live_width = block.len().min(inner_width);
            let mut out = vec![CyclotomicRing::<F, D>::zero(); n_a];
            let mut scratch = I8ColumnScratch::new();
            for chunk_start in (0..live_width).step_by(chunk_width) {
                let chunk_end = (chunk_start + chunk_width).min(live_width);
                let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
                accumulate_i8_columns::<W, K, D, CHECK_ZERO>(
                    &mut accs,
                    ntt_mat,
                    chunk_start,
                    &block[chunk_start..chunk_end],
                    params,
                    &lut,
                    &mut scratch,
                );
                for (dst, acc) in out.iter_mut().zip(accs) {
                    *dst += acc.to_ring(params);
                }
            }
            out
        })
        .collect()
}

pub(super) fn mat_vec_mul_i8_block_parallel_with_params_impl<
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
    let n_a = ntt_mat.len();
    let digit_bound = balanced_digit_abs_bound(log_basis);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut scratch = I8ColumnScratch::new();
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                accumulate_i8_columns::<W, K, D, CHECK_ZERO>(
                    &mut accs,
                    ntt_mat,
                    col,
                    &digit_buf,
                    params,
                    &lut,
                    &mut scratch,
                );
                col += digit_buf.len();
            }

            accs.into_iter().map(|acc| acc.to_ring(params)).collect()
        })
        .collect()
}

pub(super) fn mat_vec_mul_i8_block_parallel_with_params<
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
    mat_vec_mul_i8_block_parallel_with_params_impl::<F, W, K, D, true>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mat_vec_mul_i8_block_parallel_chunked_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[CyclotomicRing<F, D>]],
    inner_width: usize,
    chunk_width: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    debug_assert!(chunk_width > 0);
    debug_assert!(num_digits > 0);
    let n_a = ntt_mat.len();
    let digit_bound = balanced_digit_abs_bound(log_basis);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut out = vec![CyclotomicRing::<F, D>::zero(); n_a];
            let mut scratch = I8ColumnScratch::new();
            for chunk_start in (0..inner_width).step_by(chunk_width) {
                let chunk_end = (chunk_start + chunk_width).min(inner_width);
                let ring_start = chunk_start / num_digits;
                if ring_start >= block.len() {
                    break;
                }
                let ring_end = ((chunk_end - 1) / num_digits) + 1;
                let digit_offset = chunk_start - ring_start * num_digits;
                let tile_len = chunk_end - chunk_start;
                let block_ring_end = ring_end.min(block.len());
                let partial_coeffs = &block[ring_start..block_ring_end];
                let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                let available = all_digits.len().saturating_sub(digit_offset);
                let n = tile_len.min(available);
                let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];

                accumulate_i8_columns::<W, K, D, CHECK_ZERO>(
                    &mut accs,
                    ntt_mat,
                    chunk_start,
                    &all_digits[digit_offset..digit_offset + n],
                    params,
                    &lut,
                    &mut scratch,
                );

                for (dst, acc) in out.iter_mut().zip(accs) {
                    *dst += acc.to_ring(params);
                }
            }
            out
        })
        .collect()
}

pub(super) fn mat_vec_mul_i8_dense_block_parallel_with_params<
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
    if ntt_mat.len() == 1 {
        return mat_vec_mul_i8_dense_single_row_with_params(
            ntt_mat, blocks, num_digits, log_basis, params,
        )
        .into_iter()
        .map(|ring| vec![ring])
        .collect();
    }

    mat_vec_mul_i8_block_parallel_with_params_impl::<F, W, K, D, false>(
        ntt_mat, blocks, num_digits, log_basis, params,
    )
}

pub(super) fn mat_vec_mul_i8_dense_single_row_with_params<
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
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert_eq!(ntt_mat.len(), 1);
    let num_live_blocks = blocks.len();
    if num_live_blocks == 0 {
        return vec![];
    }
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks
        .iter()
        .map(|block| block.len().saturating_mul(num_digits))
        .max()
        .unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_live_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    let mat_row = &ntt_mat[0];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2Params::new(num_digits, log_basis, q);

    if inner_width <= safe_width && inner_width == max_data_width {
        return cfg_into_iter!(blocks)
            .map(|block| {
                let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                let mut digit_buf = vec![[0i8; D]; num_digits];
                let mut scratch = I8ColumnScratch::new();
                let mut col = 0usize;

                for coeff_vec in block.iter() {
                    coeff_vec.balanced_decompose_pow2_i8_into_with_params(
                        &mut digit_buf,
                        &decompose_params,
                    );
                    accumulate_i8_columns::<W, K, D, false>(
                        std::slice::from_mut(&mut acc),
                        ntt_mat,
                        col,
                        &digit_buf,
                        params,
                        &lut,
                        &mut scratch,
                    );
                    col += digit_buf.len();
                }

                acc.to_ring(params)
            })
            .collect();
    }

    // Over-capacity fallback chooses the available fanout: many commitment
    // blocks use block-parallel work, while narrow callers with few blocks use
    // chunk-parallel work so long CRT splits do not serialize.
    let chunk_width = capacity_safe_i8_chunk_width(safe_width, inner_width, num_digits);
    let num_chunks = inner_width.div_ceil(chunk_width);
    if num_live_blocks < DENSE_I8_BLOCK_PARALLEL_MIN_BLOCKS {
        return mat_vec_mul_i8_dense_single_row_chunk_parallel_with_params(
            mat_row,
            blocks,
            inner_width,
            chunk_width,
            num_digits,
            log_basis,
            params,
            &lut,
        );
    }

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut out = CyclotomicRing::<F, D>::zero();
            let mut scratch = I8ColumnScratch::new();

            for chunk_idx in 0..num_chunks {
                let tile_start = chunk_idx * chunk_width;
                let tile_end = (tile_start + chunk_width).min(inner_width);
                let ring_start = tile_start / num_digits;
                let ring_end = ((tile_end - 1) / num_digits) + 1;
                let digit_offset = tile_start - ring_start * num_digits;
                let tile_len = tile_end - tile_start;
                if ring_start >= block.len() {
                    break;
                }

                let block_ring_end = ring_end.min(block.len());
                let partial_coeffs = &block[ring_start..block_ring_end];
                let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                let available = all_digits.len().saturating_sub(digit_offset);
                let n = tile_len.min(available);
                let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();

                accumulate_i8_columns::<W, K, D, false>(
                    std::slice::from_mut(&mut acc),
                    ntt_mat,
                    tile_start,
                    &all_digits[digit_offset..digit_offset + n],
                    params,
                    &lut,
                    &mut scratch,
                );

                out += acc.to_ring(params);
            }

            out
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn mat_vec_mul_i8_dense_single_row_chunk_parallel_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat_row: &[CyclotomicCrtNtt<W, K, D>],
    blocks: &[&[CyclotomicRing<F, D>]],
    inner_width: usize,
    chunk_width: usize,
    num_digits: usize,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
    lut: &DigitMontLut<W, K>,
) -> Vec<CyclotomicRing<F, D>> {
    blocks
        .iter()
        .map(|block| {
            let live_width = inner_width.min(block.len().saturating_mul(num_digits));
            if live_width == 0 {
                return CyclotomicRing::<F, D>::zero();
            }
            let num_chunks = live_width.div_ceil(chunk_width);
            cfg_fold_reduce!(
                0..num_chunks,
                || CyclotomicRing::<F, D>::zero(),
                |mut out: CyclotomicRing<F, D>, chunk_idx| {
                    let tile_start = chunk_idx * chunk_width;
                    let tile_end = (tile_start + chunk_width).min(live_width);
                    let ring_start = tile_start / num_digits;
                    let ring_end = ((tile_end - 1) / num_digits) + 1;
                    let digit_offset = tile_start - ring_start * num_digits;
                    let tile_len = tile_end - tile_start;
                    let partial_coeffs = &block[ring_start..ring_end.min(block.len())];
                    let all_digits = decompose_block_i8(partial_coeffs, num_digits, log_basis);
                    let available = all_digits.len().saturating_sub(digit_offset);
                    let n = tile_len.min(available);
                    let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                    let mut scratch = I8ColumnScratch::new();

                    accumulate_i8_columns::<W, K, D, false>(
                        std::slice::from_mut(&mut acc),
                        &[mat_row],
                        tile_start,
                        &all_digits[digit_offset..digit_offset + n],
                        params,
                        lut,
                        &mut scratch,
                    );

                    out += acc.to_ring(params);
                    out
                },
                |mut a: CyclotomicRing<F, D>, b| {
                    a += b;
                    a
                }
            )
        })
        .collect()
}
