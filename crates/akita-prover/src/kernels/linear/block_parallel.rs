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
    let lut = DigitMontLut::new(params);

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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);

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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
    let mat_row = &ntt_mat[0];
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

    cfg_into_iter!(blocks)
        .map(|block| {
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            let mut digit_buf = vec![[0i8; D]; num_digits];
            let mut rhs_scratch = [[MontCoeff::from_raw(W::default()); D]; K];
            let mut col = 0usize;

            for coeff_vec in block.iter() {
                coeff_vec
                    .balanced_decompose_pow2_i8_into_with_params(&mut digit_buf, &decompose_params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
    let lut = DigitMontLut::new(params);
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
