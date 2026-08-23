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
    let num_live_blocks = blocks.len();
    if num_live_blocks == 0 {
        return vec![];
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return vec![vec![CyclotomicRing::<F, D>::zero(); n_a]; num_live_blocks];
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
    if n_a <= DENSE_I8_BLOCK_PARALLEL_MAX_ROWS
        && num_live_blocks >= DENSE_I8_BLOCK_PARALLEL_MIN_BLOCKS
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
    let pointwise_dot_batch_size = params.pointwise_dot_batch_size();
    drive_block_chunked_matvec(
        num_live_blocks,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(n_a),
        safe_width,
        params,
        |accs, start, end| {
            if pointwise_dot_batch_size > 1 {
                let mut transformed = (0..num_live_blocks)
                    .map(|_| Vec::with_capacity(pointwise_dot_batch_size))
                    .collect::<Vec<_>>();
                for batch_start in (start..end).step_by(pointwise_dot_batch_size) {
                    let batch_end = (batch_start + pointwise_dot_batch_size).min(end);
                    // Keep one matrix sub-tile hot while applying it to every
                    // right-hand side. Fall back to the sparse order below if
                    // any input can skip an all-zero plane.
                    let has_zero_plane = CHECK_ZERO
                        && blocks.iter().any(|block| {
                            batch_start < block.len()
                                && block[batch_start..batch_end.min(block.len())]
                                    .iter()
                                    .any(is_zero_plane)
                        });
                    if has_zero_plane {
                        for (block_idx, block) in blocks.iter().enumerate() {
                            let block_batch_end = batch_end.min(block.len());
                            if batch_start >= block_batch_end {
                                continue;
                            }
                            for (offset, digit) in
                                block[batch_start..block_batch_end].iter().enumerate()
                            {
                                if is_zero_plane(digit) {
                                    continue;
                                }
                                let col = batch_start + offset;
                                let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                                for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat.iter())
                                {
                                    accumulate_pointwise_product_into(
                                        acc,
                                        &mat_row[col],
                                        &ntt_d,
                                        params,
                                    );
                                }
                            }
                        }
                        continue;
                    }

                    for (digits_ntt, block) in transformed.iter_mut().zip(blocks) {
                        digits_ntt.clear();
                        let block_batch_end = batch_end.min(block.len());
                        if batch_start < block_batch_end {
                            digits_ntt.extend(block[batch_start..block_batch_end].iter().map(
                                |digit| CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut),
                            ));
                        }
                    }
                    for (row_idx, mat_row) in ntt_mat.iter().enumerate() {
                        for (block_idx, digits_ntt) in transformed.iter().enumerate() {
                            if digits_ntt.is_empty() {
                                continue;
                            }
                            accs[block_idx][row_idx].add_assign_pointwise_dot(
                                &mat_row[batch_start..batch_start + digits_ntt.len()],
                                digits_ntt,
                                params,
                            );
                        }
                    }
                }
            } else if CHECK_ZERO {
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
                for block_idx in 0..num_live_blocks {
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

enum PackedI8Lift<W: PrimeWidth, const K: usize> {
    Balanced {
        log_basis: u32,
        digit_bound: u64,
        lut: DigitMontLut<W, K>,
        check_zero_planes: bool,
    },
    Raw {
        rhs_bound: u64,
    },
}

impl<W: PrimeWidth, const K: usize> PackedI8Lift<W, K> {
    fn rhs_bound(&self) -> u64 {
        match self {
            Self::Balanced { digit_bound, .. } => *digit_bound,
            Self::Raw { rhs_bound } => *rhs_bound,
        }
    }

    fn range_error(&self) -> &'static str {
        match self {
            Self::Balanced { .. } => {
                "packed recursive digits exceed the CRT lift range for these parameters"
            }
            Self::Raw { .. } => {
                "raw packed recursive digits exceed the CRT lift range for these parameters"
            }
        }
    }

    fn width_error(&self) -> &'static str {
        match self {
            Self::Balanced { .. } => "packed recursive commitment block exceeds its row width",
            Self::Raw { .. } => "raw packed recursive commitment block exceeds its row width",
        }
    }
}

fn mat_vec_mul_packed_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    Decode,
    Block,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_live_blocks: usize,
    row_width: usize,
    decode_block: &Decode,
    params: &CrtNttParamSet<W, K, D>,
    lift: PackedI8Lift<W, K>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    Decode: Fn(usize) -> Result<Block, AkitaError> + Sync,
    Block: AsRef<[[i8; D]]>,
{
    if num_live_blocks < DENSE_I8_BLOCK_PARALLEL_MIN_BLOCKS {
        return cfg_into_iter!(0..num_live_blocks)
            .map(|block_index| {
                let block = decode_block(block_index)?;
                let block = block.as_ref();
                match &lift {
                    PackedI8Lift::Balanced {
                        log_basis,
                        digit_bound,
                        check_zero_planes,
                        ..
                    } => {
                        debug_assert!(digit_rows_within_digit_bound(
                            block,
                            row_width.min(block.len()),
                            *digit_bound,
                        ));
                        let mut rows = if *check_zero_planes {
                            mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, true>(
                                ntt_mat,
                                &[block],
                                *log_basis,
                                params,
                            )
                        } else {
                            mat_vec_mul_digits_i8_with_params_impl::<F, W, K, D, false>(
                                ntt_mat,
                                &[block],
                                *log_basis,
                                params,
                            )
                        };
                        rows.pop().ok_or(AkitaError::InvalidProof)
                    }
                    PackedI8Lift::Raw { .. } => {
                        let mut rows =
                            mat_vec_mul_raw_digits_i8_with_params(ntt_mat, &[block], params)?;
                        rows.pop().ok_or(AkitaError::InvalidProof)
                    }
                }
            })
            .collect();
    }

    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let inner_width = mat_width.min(row_width);
    if inner_width == 0 || n_a == 0 {
        return Ok(vec![
            vec![CyclotomicRing::<F, D>::zero(); n_a];
            num_live_blocks
        ]);
    }
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, lift.rhs_bound())
        .ok_or_else(|| AkitaError::InvalidInput(lift.range_error().into()))?;

    cfg_into_iter!(0..num_live_blocks)
        .map(|block_index| {
            let block = decode_block(block_index)?;
            let block = block.as_ref();
            if block.len() > row_width {
                return Err(AkitaError::InvalidSetup(lift.width_error().into()));
            }
            let mut out = vec![CyclotomicRing::<F, D>::zero(); n_a];
            match &lift {
                PackedI8Lift::Balanced {
                    digit_bound,
                    lut,
                    check_zero_planes,
                    ..
                } => {
                    debug_assert!(digit_rows_within_digit_bound(
                        block,
                        inner_width.min(block.len()),
                        *digit_bound,
                    ));
                    let mut scratch = I8ColumnScratch::new();
                    for chunk_start in (0..block.len().min(inner_width)).step_by(safe_width) {
                        let chunk_end =
                            (chunk_start + safe_width).min(block.len().min(inner_width));
                        let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
                        if *check_zero_planes {
                            accumulate_i8_columns::<W, K, D, true>(
                                &mut accs,
                                ntt_mat,
                                chunk_start,
                                &block[chunk_start..chunk_end],
                                params,
                                lut,
                                &mut scratch,
                            );
                        } else {
                            accumulate_i8_columns::<W, K, D, false>(
                                &mut accs,
                                ntt_mat,
                                chunk_start,
                                &block[chunk_start..chunk_end],
                                params,
                                lut,
                                &mut scratch,
                            );
                        }
                        for (dst, acc) in out.iter_mut().zip(accs) {
                            *dst += acc.to_ring(params);
                        }
                    }
                }
                PackedI8Lift::Raw { .. } => {
                    for chunk_start in (0..block.len().min(inner_width)).step_by(safe_width) {
                        let chunk_end =
                            (chunk_start + safe_width).min(block.len().min(inner_width));
                        let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
                        for (offset, coefficients) in
                            block[chunk_start..chunk_end].iter().enumerate()
                        {
                            if is_zero_plane(coefficients) {
                                continue;
                            }
                            let column = chunk_start + offset;
                            let rhs = CyclotomicCrtNtt::from_i8_with_params(coefficients, params);
                            for (acc, matrix_row) in accs.iter_mut().zip(ntt_mat) {
                                accumulate_pointwise_product_into(
                                    acc,
                                    &matrix_row[column],
                                    &rhs,
                                    params,
                                );
                            }
                        }
                        for (dst, acc) in out.iter_mut().zip(accs) {
                            *dst += acc.to_ring(params);
                        }
                    }
                }
            }
            Ok(out)
        })
        .collect()
}

/// Block-streamed predecomposed mat-vec for a compact physical digit source.
///
/// `decode_block` materializes at most one commitment block per worker. The
/// shared digit LUT and matrix stay outside that decode boundary, preserving
/// the block-parallel kernel without retaining the full byte witness.
pub(super) fn mat_vec_mul_packed_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    Decode,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_live_blocks: usize,
    row_width: usize,
    decode_block: &Decode,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    Decode: Fn(usize) -> Result<Vec<[i8; D]>, AkitaError> + Sync,
{
    let digit_bound = balanced_digit_abs_bound(log_basis);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    mat_vec_mul_packed_i8_with_params(
        ntt_mat,
        num_live_blocks,
        row_width,
        decode_block,
        params,
        PackedI8Lift::Balanced {
            log_basis,
            digit_bound,
            lut,
            check_zero_planes: true,
        },
    )
}

/// Dense counterpart of [`mat_vec_mul_packed_digits_i8_with_params`].
///
/// Dense decompositions overwhelmingly contain live planes, so this preserves
/// the dense kernel policy of skipping the otherwise redundant all-zero scan.
pub(super) fn mat_vec_mul_packed_dense_digits_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    Decode,
    Block,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_live_blocks: usize,
    row_width: usize,
    decode_block: &Decode,
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    Decode: Fn(usize) -> Result<Block, AkitaError> + Sync,
    Block: AsRef<[[i8; D]]>,
{
    let digit_bound = balanced_digit_abs_bound(log_basis);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound);
    mat_vec_mul_packed_i8_with_params(
        ntt_mat,
        num_live_blocks,
        row_width,
        decode_block,
        params,
        PackedI8Lift::Balanced {
            log_basis,
            digit_bound,
            lut,
            check_zero_planes: false,
        },
    )
}

/// Fold-major (block) raw signed-i8 ring mat-vec for `num_digits_inner == 1`.
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
    let num_live_blocks = blocks.len();
    if num_live_blocks == 0 {
        return Ok(vec![]);
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return Ok(vec![
            vec![CyclotomicRing::<F, D>::zero(); n_a];
            num_live_blocks
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
        num_live_blocks,
        n_a,
        inner_width,
        safe_width,
        base_tile_width::<W, K, D>(n_a),
        safe_width,
        params,
        |accs, start, end| {
            for block_idx in 0..num_live_blocks {
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

/// Raw signed-i8 counterpart of
/// [`mat_vec_mul_packed_digits_i8_with_params`].
pub(super) fn mat_vec_mul_packed_raw_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    Decode,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_live_blocks: usize,
    row_width: usize,
    rhs_bound: u64,
    decode_block: &Decode,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    Decode: Fn(usize) -> Result<Vec<[i8; D]>, AkitaError> + Sync,
{
    mat_vec_mul_packed_i8_with_params(
        ntt_mat,
        num_live_blocks,
        row_width,
        decode_block,
        params,
        PackedI8Lift::Raw { rhs_bound },
    )
}
