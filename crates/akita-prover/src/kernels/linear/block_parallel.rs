use super::*;

/// Block-parallel fast path for small `n_a` and many blocks.
///
/// Parallelizes over blocks (high fanout) instead of column tiles (low fanout).
/// With many blocks but few matrix rows, the old tile-based approach had limited
/// parallelism (few tiles) while this path gives num_blocks-way parallelism.
pub(super) fn mat_vec_mul_digits_i8_block_parallel<
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
    if ntt_mat.len() == 1 {
        return mat_vec_mul_digits_i8_single_row_block_parallel::<F, W, K, D, CHECK_ZERO>(
            ntt_mat, blocks, params,
        )
        .into_iter()
        .map(|ring| vec![ring])
        .collect();
    }
    if ntt_mat.len() == 2 {
        return mat_vec_mul_digits_i8_two_row_block_parallel::<F, W, K, D, CHECK_ZERO>(
            ntt_mat, blocks, params,
        );
    }
    if ntt_mat.len() == 3 {
        return mat_vec_mul_digits_i8_three_row_block_parallel::<F, W, K, D, CHECK_ZERO>(
            ntt_mat, blocks, params,
        );
    }

    let n_a = ntt_mat.len();
    let lut = DigitMontLut::<W, K>::new(params);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];

            for (j, digit) in block.iter().enumerate() {
                if CHECK_ZERO && is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(acc, &mat_row[j], &ntt_d, params);
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_single_row_block_parallel<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert_eq!(ntt_mat.len(), 1);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row = &ntt_mat[0];

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (j, digit) in block.iter().enumerate() {
                if CHECK_ZERO && is_zero_plane(digit) {
                    continue;
                }
                acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                    &mat_row[j],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            acc.to_ring_with_params(params)
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_two_row_block_parallel<
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
    debug_assert_eq!(ntt_mat.len(), 2);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (j, digit) in block.iter().enumerate() {
                if CHECK_ZERO && is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_pair_with_lut_scratch(
                    [&mut acc0, &mut acc1],
                    [&mat_row0[j], &mat_row1[j]],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
            ]
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_three_row_block_parallel<
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
    debug_assert_eq!(ntt_mat.len(), 3);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let mat_row2 = &ntt_mat[2];

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc2 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (j, digit) in block.iter().enumerate() {
                if CHECK_ZERO && is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_triple_with_lut_scratch(
                    [&mut acc0, &mut acc1, &mut acc2],
                    [&mat_row0[j], &mat_row1[j], &mat_row2[j]],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
                acc2.to_ring_with_params(params),
            ]
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_strided_block_parallel<
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
    if ntt_mat.len() == 1 {
        return mat_vec_mul_digits_i8_single_row_strided_block_parallel(
            ntt_mat, coeffs, num_blocks, block_len, params,
        )
        .into_iter()
        .map(|ring| vec![ring])
        .collect();
    }
    if ntt_mat.len() == 2 {
        return mat_vec_mul_digits_i8_two_row_strided_block_parallel(
            ntt_mat, coeffs, num_blocks, block_len, params,
        );
    }
    if ntt_mat.len() == 3 {
        return mat_vec_mul_digits_i8_three_row_strided_block_parallel(
            ntt_mat, coeffs, num_blocks, block_len, params,
        );
    }

    let n_a = ntt_mat.len();
    let lut = DigitMontLut::<W, K>::new(params);

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];

            for col in 0..block_len {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                    accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_single_row_strided_block_parallel<
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
) -> Vec<CyclotomicRing<F, D>> {
    debug_assert_eq!(ntt_mat.len(), 1);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row = &ntt_mat[0];

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (col, mat_coeff) in mat_row.iter().take(block_len).enumerate() {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                    mat_coeff,
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            acc.to_ring_with_params(params)
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_two_row_strided_block_parallel<
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
    debug_assert_eq!(ntt_mat.len(), 2);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (col, (mat_coeff0, mat_coeff1)) in mat_row0
                .iter()
                .zip(mat_row1.iter())
                .take(block_len)
                .enumerate()
            {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_pair_with_lut_scratch(
                    [&mut acc0, &mut acc1],
                    [mat_coeff0, mat_coeff1],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
            ]
        })
        .collect()
}

pub(super) fn mat_vec_mul_digits_i8_three_row_strided_block_parallel<
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
    debug_assert_eq!(ntt_mat.len(), 3);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let mat_row2 = &ntt_mat[2];

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc2 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

            for (col, ((mat_coeff0, mat_coeff1), mat_coeff2)) in mat_row0
                .iter()
                .zip(mat_row1.iter())
                .zip(mat_row2.iter())
                .take(block_len)
                .enumerate()
            {
                let seq = block_idx + col * num_blocks;
                let Some(digit) = coeffs.get(seq) else {
                    break;
                };
                if is_zero_plane(digit) {
                    continue;
                }
                CyclotomicCrtNtt::add_assign_pointwise_mul_i8_triple_with_lut_scratch(
                    [&mut acc0, &mut acc1, &mut acc2],
                    [mat_coeff0, mat_coeff1, mat_coeff2],
                    digit,
                    params,
                    &lut,
                    &mut rhs_scratch,
                );
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
                acc2.to_ring_with_params(params),
            ]
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
    let lut = DigitMontLut::<W, K>::new(params);
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    if CHECK_ZERO && is_zero_plane(digit) {
                        col += 1;
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(acc, &mat_row[col], &ntt_d, params);
                    }
                    col += 1;
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
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
    if ntt_mat.len() == 2 {
        return mat_vec_mul_i8_dense_two_row_fused_with_params(
            ntt_mat, blocks, num_digits, log_basis, params,
        );
    }
    if ntt_mat.len() == 3 {
        return mat_vec_mul_i8_dense_three_row_fused_with_params(
            ntt_mat, blocks, num_digits, log_basis, params,
        );
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
    let num_blocks = blocks.len();
    if num_blocks == 0 {
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
        return vec![CyclotomicRing::<F, D>::zero(); num_blocks];
    }

    let digit_bound = balanced_digit_abs_bound(log_basis);
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .expect("single i8 CRT term must fit supported parameters");
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row = &ntt_mat[0];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    if inner_width <= safe_width && inner_width == max_data_width {
        return cfg_into_iter!(blocks)
            .map(|block| {
                let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                let mut digit_buf = vec![[0i8; D]; num_digits];
                let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
                let mut col = 0usize;

                for coeff_vec in block.iter() {
                    coeff_vec.balanced_decompose_pow2_i8_into_with_params(
                        &mut digit_buf,
                        &decompose_params,
                    );
                    for digit in &digit_buf {
                        acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                            &mat_row[col],
                            digit,
                            params,
                            &lut,
                            &mut rhs_scratch,
                        );
                        col += 1;
                    }
                }

                acc.to_ring_with_params(params)
            })
            .collect();
    }

    let chunk_width = capacity_safe_i8_chunk_width(safe_width, inner_width, num_digits);
    let num_chunks = inner_width.div_ceil(chunk_width);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut out = CyclotomicRing::<F, D>::zero();
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];

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

                for (j, digit) in all_digits[digit_offset..digit_offset + n]
                    .iter()
                    .enumerate()
                {
                    acc.add_assign_pointwise_mul_i8_with_lut_scratch(
                        &mat_row[tile_start + j],
                        digit,
                        params,
                        &lut,
                        &mut rhs_scratch,
                    );
                }

                out += acc.to_ring_with_params(params);
            }

            out
        })
        .collect()
}

pub(super) fn mat_vec_mul_i8_dense_two_row_fused_with_params<
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
    debug_assert_eq!(ntt_mat.len(), 2);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    CyclotomicCrtNtt::add_assign_pointwise_mul_i8_pair_with_lut_scratch(
                        [&mut acc0, &mut acc1],
                        [&mat_row0[col], &mat_row1[col]],
                        digit,
                        params,
                        &lut,
                        &mut rhs_scratch,
                    );
                    col += 1;
                }
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
            ]
        })
        .collect()
}

pub(super) fn mat_vec_mul_i8_dense_three_row_fused_with_params<
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
    debug_assert_eq!(ntt_mat.len(), 3);
    let lut = DigitMontLut::<W, K>::new(params);
    let mat_row0 = &ntt_mat[0];
    let mat_row1 = &ntt_mat[1];
    let mat_row2 = &ntt_mat[2];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc0 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc1 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut acc2 = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    CyclotomicCrtNtt::add_assign_pointwise_mul_i8_triple_with_lut_scratch(
                        [&mut acc0, &mut acc1, &mut acc2],
                        [&mat_row0[col], &mat_row1[col], &mat_row2[col]],
                        digit,
                        params,
                        &lut,
                        &mut rhs_scratch,
                    );
                    col += 1;
                }
            }

            vec![
                acc0.to_ring_with_params(params),
                acc1.to_ring_with_params(params),
                acc2.to_ring_with_params(params),
            ]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mat_vec_mul_i8_strided_block_parallel_with_params<
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
    let n_a = ntt_mat.len();
    let lut = DigitMontLut::<W, K>::new(params);
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(0..num_blocks)
        .map(|block_idx| {
            let mut accs: Vec<CyclotomicCrtNtt<W, K, D>> =
                vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut mat_col = 0usize;

            for col in 0..block_len {
                let seq = block_idx + col * num_blocks;
                let Some(coeff) = coeffs.get(seq) else {
                    break;
                };
                coeff
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
                for digit in &digit_buf {
                    if !is_zero_plane(digit) {
                        let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                        for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                            accumulate_pointwise_product_into(
                                acc,
                                &mat_row[mat_col],
                                &ntt_d,
                                params,
                            );
                        }
                    }
                    mat_col += 1;
                }
            }

            accs.into_iter()
                .map(|acc| acc.to_ring_with_params(params))
                .collect()
        })
        .collect()
}
