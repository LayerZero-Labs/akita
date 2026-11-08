use super::*;

#[derive(Clone, Copy)]
pub(crate) enum I8CommitPolicy {
    Dense,
    Recursive {
        raw: bool,
    },
    Tiled {
        digit_bound: u64,
        raw: bool,
        block_index: usize,
    },
    /// [`Self::Dense`] in time and with memory addresses independent of the
    /// digits: no zero-plane skip and a constant-time CRT reconstruction
    /// ([`CyclotomicCrtNtt::to_ring_ct`]). For digits of secret images, the
    /// compression maps of a Σ attempt's `t_D` (`specs/whiteout-akita.md` §6.6).
    ConstantTime,
}

pub(super) fn mat_vec_mul_digits_i8_with_params<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    digit_rows_matvec::<F, W, K, D, true>(ntt_mat, blocks, log_basis, I8CommitPolicy::Dense, params)
}

/// [`mat_vec_mul_digits_i8_with_params`] under [`I8CommitPolicy::ConstantTime`].
pub(super) fn mat_vec_mul_digits_i8_ct_with_params<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    digit_rows_matvec::<F, W, K, D, false>(
        ntt_mat,
        blocks,
        log_basis,
        I8CommitPolicy::ConstantTime,
        params,
    )
}

fn digit_rows_matvec<
    F: Field + CanonicalEncoding,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    blocks: &[&[[i8; D]]],
    log_basis: u32,
    policy: I8CommitPolicy,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let source = DigitRows {
        blocks,
        digit_abs_bound: balanced_digit_abs_bound(log_basis),
    };
    let mut start = 0;
    let ranges = blocks
        .iter()
        .map(|block| {
            let range = start..start + block.len();
            start = range.end;
            range
        })
        .collect::<Vec<_>>();
    mat_vec_mul_blocks_i8_with_params_impl::<F, _, W, K, D, CHECK_ZERO>(
        ntt_mat,
        &source,
        &ranges,
        DigitShape {
            num_digits: 1,
            log_basis,
        },
        policy,
        "digit matvec CRT capacity cannot fit a single term",
        params,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mat_vec_mul_blocks_i8_with_params_impl<
    F: Field + CanonicalEncoding,
    S: I8Rows<D>,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const CHECK_ZERO: bool,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    source: &S,
    blocks: &[Range<usize>],
    shape: DigitShape,
    policy: I8CommitPolicy,
    capacity_error: &str,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let num_live_blocks = blocks.len();
    if num_live_blocks == 0 {
        return Ok(vec![]);
    }
    let n_a = ntt_mat.len();
    let mat_width = ntt_mat.first().map_or(0, |row| row.len());
    let max_data_width = blocks
        .iter()
        .map(|b| b.len().saturating_mul(shape.num_digits))
        .max()
        .unwrap_or(0);
    let inner_width = mat_width.min(max_data_width);
    if inner_width == 0 || n_a == 0 {
        return Ok(vec![
            vec![CyclotomicRing::<F, D>::zero(); n_a];
            num_live_blocks
        ]);
    }
    if let I8CommitPolicy::Recursive { raw } = policy {
        let digit_bound = balanced_digit_abs_bound(shape.log_basis);
        if num_live_blocks >= DENSE_I8_BLOCK_PARALLEL_MIN_BLOCKS {
            if max_data_width > mat_width {
                return Err(AkitaError::InvalidSetup(
                    "packed recursive commitment block exceeds its row width".into(),
                ));
            }
            let chunk_width = if raw {
                mat_width
            } else {
                safe_crt_chunk_width::<F, W, K, D>(params, mat_width, digit_bound)
                    .ok_or_else(|| AkitaError::InvalidInput(capacity_error.into()))?
            };
            return mat_vec_mul_digits_i8_block_parallel_chunked::<F, S, W, K, D, CHECK_ZERO>(
                ntt_mat,
                source,
                blocks,
                shape,
                mat_width,
                chunk_width,
                digit_bound,
                raw,
                params,
            );
        }
        let decode_rings = params.pointwise_dot_batch_size();
        let init = || source.i8_scratch(shape, decode_rings);
        let apply = |scratch: &mut S::I8Scratch, (block_index, block): (usize, &Range<usize>)| {
            let digit_bound = if raw {
                i8_block_abs_bound(
                    source,
                    block_index,
                    block,
                    shape,
                    inner_width,
                    decode_rings,
                    scratch,
                )?
            } else {
                digit_bound
            };
            let mut rows = mat_vec_mul_blocks_i8_with_params_impl::<F, S, W, K, D, CHECK_ZERO>(
                ntt_mat,
                source,
                std::slice::from_ref(block),
                shape,
                I8CommitPolicy::Tiled {
                    digit_bound,
                    raw,
                    block_index,
                },
                capacity_error,
                params,
            )?;
            rows.pop().ok_or_else(|| {
                AkitaError::Internal("single-block commitment matvec returned no batch".into())
            })
        };
        #[cfg(feature = "parallel")]
        {
            return blocks
                .par_iter()
                .enumerate()
                .map_init(init, apply)
                .collect();
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = init();
            return blocks
                .iter()
                .enumerate()
                .map(|block| apply(&mut scratch, block))
                .collect();
        }
    }
    let digit_bound = match policy {
        I8CommitPolicy::Tiled { digit_bound, .. } => digit_bound,
        _ => source.digit_abs_bound(shape),
    };
    let safe_width = safe_crt_chunk_width::<F, W, K, D>(params, inner_width, digit_bound)
        .ok_or_else(|| match policy {
            I8CommitPolicy::Tiled { raw: true, .. } => AkitaError::InvalidInput(
                "raw i8 recursive-witness coefficients exceed the CRT lift range for these parameters".into(),
            ),
            I8CommitPolicy::Tiled { .. } => AkitaError::InvalidInput(capacity_error.into()),
            _ => AkitaError::InvalidSetup(capacity_error.into()),
        })?;
    let chunk_width = capacity_safe_i8_chunk_width(safe_width, inner_width, shape.num_digits);
    if matches!(policy, I8CommitPolicy::Dense)
        && n_a <= DENSE_I8_BLOCK_PARALLEL_MAX_ROWS
        && num_live_blocks >= DENSE_I8_BLOCK_PARALLEL_MIN_BLOCKS
        && inner_width == max_data_width
    {
        if inner_width <= safe_width {
            return mat_vec_mul_digits_i8_block_parallel::<F, S, W, K, D, CHECK_ZERO>(
                ntt_mat,
                source,
                blocks,
                shape,
                digit_bound,
                params,
            );
        }
        return mat_vec_mul_digits_i8_block_parallel_chunked::<F, S, W, K, D, CHECK_ZERO>(
            ntt_mat,
            source,
            blocks,
            shape,
            inner_width,
            chunk_width,
            digit_bound,
            false,
            params,
        );
    }
    let lut_bound = if matches!(policy, I8CommitPolicy::Tiled { raw: true, .. }) {
        (digit_bound + 1).next_power_of_two().min(128)
    } else {
        digit_bound
    };
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, lut_bound);
    let batch = params.pointwise_dot_batch_size();
    let tile_width = aligned_i8_tile_width(
        base_tile_width::<W, K, D>(n_a),
        inner_width,
        shape.num_digits,
    );
    let max_rings = batch.max(shape.num_digits).div_ceil(shape.num_digits) + 1;
    let constant_time = matches!(policy, I8CommitPolicy::ConstantTime);
    drive_block_chunked_matvec(
        num_live_blocks,
        n_a,
        inner_width,
        safe_width,
        tile_width,
        chunk_width,
        params,
        || {
            (
                source.i8_scratch(shape, max_rings),
                vec![
                    vec![CyclotomicCrtNtt::<W, K, D>::zero(); batch.max(shape.num_digits)];
                    num_live_blocks
                ],
                vec![0usize; num_live_blocks],
            )
        },
        |accs, (scratch, transformed, live), start, end| {
            // Decode a ring-aligned batch once for each right-hand side, then
            // reuse its transforms across every matrix row.
            let batch_width = aligned_i8_tile_width(batch, end - start, shape.num_digits);
            for batch_start in (start..end).step_by(batch_width) {
                let batch_end = (batch_start + batch_width).min(end);
                for (block_idx, block) in blocks.iter().enumerate() {
                    let ring_start = batch_start / shape.num_digits;
                    if ring_start >= block.len() {
                        live[block_idx] = 0;
                        continue;
                    }
                    let ring_end = batch_end.div_ceil(shape.num_digits).min(block.len());
                    let offset = batch_start % shape.num_digits;
                    let source_block_index = match policy {
                        I8CommitPolicy::Tiled { block_index, .. } => block_index,
                        _ => block_idx,
                    };
                    let rows = source.i8_rows(
                        source_block_index,
                        block.start,
                        ring_start..ring_end,
                        shape,
                        scratch,
                    )?;
                    let n = (batch_end - batch_start).min(rows.len() - offset);
                    let digits = &rows[offset..offset + n];
                    if CHECK_ZERO && !constant_time && digits.iter().any(is_zero_plane) {
                        for (i, digit) in digits.iter().enumerate() {
                            if is_zero_plane(digit) {
                                continue;
                            }
                            transformed[block_idx][i].assign_i8_with_lut(digit, params, &lut);
                            for (acc, mat_row) in accs[block_idx].iter_mut().zip(ntt_mat) {
                                accumulate_pointwise_product_into(
                                    acc,
                                    &mat_row[batch_start + i],
                                    &transformed[block_idx][i],
                                    params,
                                );
                            }
                        }
                        live[block_idx] = 0;
                    } else {
                        live[block_idx] = n;
                        for (slot, digit) in transformed[block_idx].iter_mut().zip(digits) {
                            slot.assign_i8_with_lut(digit, params, &lut);
                        }
                    }
                }
                for (row_idx, mat_row) in ntt_mat.iter().enumerate() {
                    for (block_idx, (&n, digits_ntt)) in
                        live.iter().zip(transformed.iter()).enumerate()
                    {
                        for offset in (0..n).step_by(batch) {
                            let len = batch.min(n - offset);
                            accs[block_idx][row_idx].add_assign_pointwise_dot(
                                &mat_row[batch_start + offset..batch_start + offset + len],
                                &digits_ntt[offset..offset + len],
                                params,
                            );
                        }
                    }
                }
            }
            Ok(())
        },
        |acc, params| {
            if constant_time {
                acc.to_ring_ct(params)
            } else {
                acc.to_ring(params)
            }
        },
    )
}
