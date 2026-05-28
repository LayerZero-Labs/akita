use super::*;

/// Column-tiled mat-vec for a single pre-decomposed i8 digit vector.
///
/// Same tiling strategy as [`mat_vec_mul_ntt_i8`] but for a single
/// input vector of i8 digit planes (already decomposed). Tiles the matrix
/// columns to keep each tile in L2, eliminating the full `ntt_vec`
/// materialization of the non-tiled path.
/// Tile width is auto-computed from ring parameters and target L2 cache size.
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8")]
pub fn mat_vec_mul_ntt_single_i8<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, p)
        }
        NttSlotCache::Q64 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, p)
        }
        NttSlotCache::Q128 { neg, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_with_params(&rows, vec, p)
        }
    }
}

/// Cyclic-domain variant of [`mat_vec_mul_ntt_single_i8`].
#[tracing::instrument(skip_all, name = "mat_vec_mul_ntt_single_i8_cyclic")]
pub fn mat_vec_mul_ntt_single_i8_cyclic<F: FieldCore + CanonicalField, const D: usize>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    vec: &[[i8; D]],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, p)
        }
        NttSlotCache::Q64 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, p)
        }
        NttSlotCache::Q128 { cyc, params: p, .. } => {
            let rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            mat_vec_mul_single_i8_cyclic_with_params(&rows, vec, p)
        }
    }
}

pub(super) fn mat_vec_mul_single_i8_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let lut = DigitMontLut::new(params);
    let vec_len = vec.len().min(inner_width);
    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, vec_len);
    if vec_len <= chunk_width {
        let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
        let num_tiles = vec_len.div_ceil(tw);

        let final_accs: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a],
            |mut accs: Vec<CyclotomicCrtNtt<W, K, D>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(vec_len);
                for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(
                            acc,
                            &mat_row[tile_start + j],
                            &ntt_d,
                            params,
                        );
                    }
                }
                accs
            },
            |mut a: Vec<CyclotomicCrtNtt<W, K, D>>, b| {
                for row in 0..n_a {
                    add_ntt_into(&mut a[row], &b[row], params);
                }
                a
            }
        );

        return final_accs
            .into_iter()
            .map(|acc| acc.to_ring_with_params(params))
            .collect();
    }

    let cache_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>()))
        .max(1)
        .min(chunk_width);
    let num_chunks = vec_len.div_ceil(chunk_width);

    let final_accs: Vec<CyclotomicRing<F, D>> = cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); n_a],
        |mut accs: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(vec_len);
            let mut chunk_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            for tile_start in (chunk_start..chunk_end).step_by(cache_tw) {
                let tile_end = (tile_start + cache_tw).min(chunk_end);
                for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                    for (acc, mat_row) in chunk_accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(
                            acc,
                            &mat_row[tile_start + j],
                            &ntt_d,
                            params,
                        );
                    }
                }
            }
            for row in 0..n_a {
                let partial = chunk_accs[row].to_ring_with_params(params);
                accs[row] += partial;
            }
            accs
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for row in 0..n_a {
                a[row] += b[row];
            }
            a
        }
    );

    final_accs
}

pub(super) fn mat_vec_mul_single_i8_cyclic_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    ntt_mat: &[&[CyclotomicCrtNtt<W, K, D>]],
    vec: &[[i8; D]],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let n_a = ntt_mat.len();
    let inner_width = ntt_mat.first().map_or(0, |row| row.len());
    if inner_width == 0 || n_a == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); n_a];
    }

    let lut = DigitMontLut::new(params);
    let vec_len = vec.len().min(inner_width);
    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, vec_len);
    if vec_len <= chunk_width {
        let tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
        let num_tiles = vec_len.div_ceil(tw);

        let final_accs: Vec<CyclotomicCrtNtt<W, K, D>> = cfg_fold_reduce!(
            0..num_tiles,
            || vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a],
            |mut accs: Vec<CyclotomicCrtNtt<W, K, D>>, tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(vec_len);
                for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
                    for (acc, mat_row) in accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(
                            acc,
                            &mat_row[tile_start + j],
                            &ntt_d,
                            params,
                        );
                    }
                }
                accs
            },
            |mut a: Vec<CyclotomicCrtNtt<W, K, D>>, b| {
                for row in 0..n_a {
                    add_ntt_into(&mut a[row], &b[row], params);
                }
                a
            }
        );

        return final_accs
            .into_iter()
            .map(|acc| acc.to_ring_cyclic(params))
            .collect();
    }

    let cache_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>()))
        .max(1)
        .min(chunk_width);
    let num_chunks = vec_len.div_ceil(chunk_width);

    let final_accs: Vec<CyclotomicRing<F, D>> = cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); n_a],
        |mut accs: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(vec_len);
            let mut chunk_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); n_a];
            for tile_start in (chunk_start..chunk_end).step_by(cache_tw) {
                let tile_end = (tile_start + cache_tw).min(chunk_end);
                for (j, digit) in vec[tile_start..tile_end].iter().enumerate() {
                    if is_zero_plane(digit) {
                        continue;
                    }
                    let ntt_d = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
                    for (acc, mat_row) in chunk_accs.iter_mut().zip(ntt_mat.iter()) {
                        accumulate_pointwise_product_into(
                            acc,
                            &mat_row[tile_start + j],
                            &ntt_d,
                            params,
                        );
                    }
                }
            }
            for row in 0..n_a {
                let partial = chunk_accs[row].to_ring_cyclic(params);
                accs[row] += partial;
            }
            accs
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for row in 0..n_a {
                a[row] += b[row];
            }
            a
        }
    );

    final_accs
}
