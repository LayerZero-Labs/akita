use super::*;

/// Minimum number of Rayon work-units for the fused kernel.
///
/// The fused kernel replaces three separate `cfg_fold_reduce` calls
/// (each creating ~N tiles) with a single call. To preserve the ~3N total
/// work-units that rayon::join previously provided, we enforce at least
/// this many tiles so Rayon's work-stealing keeps all cores busy.
const MIN_FUSED_TILES: usize = 30;

/// Fused column-tiled kernel for the three split-eq mat-vec products.
///
/// Replaces three separate NTT-cached mat-vec calls (D-cyclic, B-cyclic,
/// A-quotient) with a single pass over the shared NTT cache. Within each
/// column tile, cache entries are loaded once and reused across all three
/// products with their exact row bounds, eliminating redundant DRAM reads.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn fused_split_eq_quotients_with_params<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    n_d: usize,
    n_b: usize,
    n_a: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let mat_width = cyc_rows.first().map_or(0, |r| r.len());
    let w_len = w_hat.len().min(mat_width);
    let t_len = t_hat.len().min(mat_width);
    let z_len = z_pre.len().min(mat_width);
    let max_col = w_len.max(t_len).max(z_len);

    if max_col == 0 {
        return (
            vec![CyclotomicRing::<F, D>::zero(); n_d],
            vec![CyclotomicRing::<F, D>::zero(); n_b],
            vec![CyclotomicRing::<F, D>::zero(); n_a],
        );
    }

    let lut = DigitMontLut::new(params);
    let centered_lut = (z_pre_max_abs <= CENTERED_LUT_MAX_ABS)
        .then(|| CenteredMontLut::<W, K>::new(params, z_pre_max_abs as i32));

    let base_tw = (TARGET_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let i8_chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(I8_RHS_MAX_ABS, max_col);
    let z_chunk_width =
        crt_accumulation_chunk_width::<F, W, K, D>(u128::from(z_pre_max_abs.max(1)), max_col);
    let chunk_width = i8_chunk_width.min(z_chunk_width).max(1);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();
    if max_col <= chunk_width {
        let tw = base_tw.min(max_col.div_ceil(MIN_FUSED_TILES).max(1));
        let num_tiles = max_col.div_ceil(tw);

        let (d_accs, b_accs, a_neg_accs, a_cyc_accs) = cfg_fold_reduce!(
            0..num_tiles,
            || (
                vec![zero.clone(); n_d],
                vec![zero.clone(); n_b],
                vec![zero.clone(); n_a],
                vec![zero.clone(); n_a],
            ),
            |mut accs: (
                Vec<CyclotomicCrtNtt<W, K, D>>,
                Vec<CyclotomicCrtNtt<W, K, D>>,
                Vec<CyclotomicCrtNtt<W, K, D>>,
                Vec<CyclotomicCrtNtt<W, K, D>>,
            ),
             tile_idx| {
                let tile_start = tile_idx * tw;
                let tile_end = (tile_start + tw).min(max_col);

                for j in tile_start..tile_end {
                    if j < w_len && !is_zero_plane(&w_hat[j]) {
                        let ntt_w =
                            CyclotomicCrtNtt::from_i8_cyclic_with_lut(&w_hat[j], params, &lut);
                        for (acc_d, cyc_row) in accs.0.iter_mut().zip(cyc_rows.iter()) {
                            accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                        }
                    }

                    if j < t_len && !is_zero_plane(&t_hat[j]) {
                        let ntt_t =
                            CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, &lut);
                        for (acc_b, cyc_row) in accs.1.iter_mut().zip(cyc_rows.iter()) {
                            accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                        }
                    }

                    if j < z_len && !is_zero_centered_row(&z_pre[j]) {
                        let (ntt_z_neg, ntt_z_cyc) = if let Some(ref clut) = centered_lut {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut(
                                &z_pre[j], params, clut,
                            )
                        } else {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_pre[j], params)
                        };
                        for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in accs
                            .2
                            .iter_mut()
                            .zip(accs.3.iter_mut())
                            .zip(neg_rows.iter().zip(cyc_rows.iter()))
                        {
                            accumulate_pointwise_product_into(
                                acc_neg,
                                &neg_row[j],
                                &ntt_z_neg,
                                params,
                            );
                            accumulate_pointwise_product_into(
                                acc_cyc,
                                &cyc_row[j],
                                &ntt_z_cyc,
                                params,
                            );
                        }
                    }
                }
                accs
            },
            |mut a: (
                Vec<CyclotomicCrtNtt<W, K, D>>,
                Vec<CyclotomicCrtNtt<W, K, D>>,
                Vec<CyclotomicCrtNtt<W, K, D>>,
                Vec<CyclotomicCrtNtt<W, K, D>>,
            ),
             b| {
                for r in 0..n_d {
                    add_ntt_into(&mut a.0[r], &b.0[r], params);
                }
                for r in 0..n_b {
                    add_ntt_into(&mut a.1[r], &b.1[r], params);
                }
                for r in 0..n_a {
                    add_ntt_into(&mut a.2[r], &b.2[r], params);
                    add_ntt_into(&mut a.3[r], &b.3[r], params);
                }
                a
            }
        );

        let d_result = d_accs
            .into_iter()
            .map(|acc| acc.to_ring_cyclic(params))
            .collect();
        let b_result = b_accs
            .into_iter()
            .map(|acc| acc.to_ring_cyclic(params))
            .collect();
        let a_result = a_neg_accs
            .into_iter()
            .zip(a_cyc_accs)
            .map(|(neg_acc, cyc_acc)| {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                let neg_c = neg_ring.coefficients();
                let cyc_c = cyc_ring.coefficients();
                let q: [F; D] = from_fn(|k| (cyc_c[k] - neg_c[k]).half());
                CyclotomicRing::from_coefficients(q)
            })
            .collect();

        return (d_result, b_result, a_result);
    }

    let cache_tw = base_tw
        .min(max_col.div_ceil(MIN_FUSED_TILES).max(1))
        .min(chunk_width)
        .max(1);
    let num_chunks = max_col.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || (
            vec![CyclotomicRing::<F, D>::zero(); n_d],
            vec![CyclotomicRing::<F, D>::zero(); n_b],
            vec![CyclotomicRing::<F, D>::zero(); n_a],
        ),
        |mut accs: (
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
        ),
         chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(max_col);
            let mut d_accs = vec![zero.clone(); n_d];
            let mut b_accs = vec![zero.clone(); n_b];
            let mut a_neg_accs = vec![zero.clone(); n_a];
            let mut a_cyc_accs = vec![zero.clone(); n_a];

            for tile_start in (chunk_start..chunk_end).step_by(cache_tw) {
                let tile_end = (tile_start + cache_tw).min(chunk_end);
                for j in tile_start..tile_end {
                    if j < w_len && !is_zero_plane(&w_hat[j]) {
                        let ntt_w =
                            CyclotomicCrtNtt::from_i8_cyclic_with_lut(&w_hat[j], params, &lut);
                        for (acc_d, cyc_row) in d_accs.iter_mut().zip(cyc_rows.iter()) {
                            accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                        }
                    }

                    if j < t_len && !is_zero_plane(&t_hat[j]) {
                        let ntt_t =
                            CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, &lut);
                        for (acc_b, cyc_row) in b_accs.iter_mut().zip(cyc_rows.iter()) {
                            accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                        }
                    }

                    if j < z_len && !is_zero_centered_row(&z_pre[j]) {
                        let (ntt_z_neg, ntt_z_cyc) = if let Some(ref clut) = centered_lut {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut(
                                &z_pre[j], params, clut,
                            )
                        } else {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_pre[j], params)
                        };
                        for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in a_neg_accs
                            .iter_mut()
                            .zip(a_cyc_accs.iter_mut())
                            .zip(neg_rows.iter().zip(cyc_rows.iter()))
                        {
                            accumulate_pointwise_product_into(
                                acc_neg,
                                &neg_row[j],
                                &ntt_z_neg,
                                params,
                            );
                            accumulate_pointwise_product_into(
                                acc_cyc,
                                &cyc_row[j],
                                &ntt_z_cyc,
                                params,
                            );
                        }
                    }
                }
            }
            for (dst, acc) in accs.0.iter_mut().zip(d_accs.into_iter()) {
                *dst += acc.to_ring_cyclic(params);
            }
            for (dst, acc) in accs.1.iter_mut().zip(b_accs.into_iter()) {
                *dst += acc.to_ring_cyclic(params);
            }
            for ((dst, neg_acc), cyc_acc) in accs
                .2
                .iter_mut()
                .zip(a_neg_accs.into_iter())
                .zip(a_cyc_accs.into_iter())
            {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                let neg_c = neg_ring.coefficients();
                let cyc_c = cyc_ring.coefficients();
                let q: [F; D] = from_fn(|k| (cyc_c[k] - neg_c[k]).half());
                *dst += CyclotomicRing::from_coefficients(q);
            }
            accs
        },
        |mut a: (
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
            Vec<CyclotomicRing<F, D>>,
        ),
         b| {
            for (dst, rhs) in a.0.iter_mut().zip(b.0.into_iter()) {
                *dst += rhs;
            }
            for (dst, rhs) in a.1.iter_mut().zip(b.1.into_iter()) {
                *dst += rhs;
            }
            for (dst, rhs) in a.2.iter_mut().zip(b.2.into_iter()) {
                *dst += rhs;
            }
            a
        }
    )
}

/// Fused split-eq quotient kernel dispatching over [`NttSlotCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · w_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix and must use the
/// same row `stride` so that logical position `(i, j)` maps to the same
/// physical flat-cache element regardless of role.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
pub fn fused_split_eq_quotients<F: FieldCore + CanonicalField + HalvingField, const D: usize>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    stride: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let n_cyc = n_d.max(n_b).max(n_a);
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * stride..(i + 1) * stride])
                .collect();
            let cyc_rows: Vec<&[_]> = (0..n_cyc)
                .map(|i| &cyc[i * stride..(i + 1) * stride])
                .collect();
            fused_split_eq_quotients_with_params(
                &cyc_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                w_hat,
                t_hat,
                z_pre,
                z_pre_max_abs,
                p,
            )
        }
        NttSlotCache::Q64 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * stride..(i + 1) * stride])
                .collect();
            let cyc_rows: Vec<&[_]> = (0..n_cyc)
                .map(|i| &cyc[i * stride..(i + 1) * stride])
                .collect();
            fused_split_eq_quotients_with_params(
                &cyc_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                w_hat,
                t_hat,
                z_pre,
                z_pre_max_abs,
                p,
            )
        }
        NttSlotCache::Q128 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * stride..(i + 1) * stride])
                .collect();
            let cyc_rows: Vec<&[_]> = (0..n_cyc)
                .map(|i| &cyc[i * stride..(i + 1) * stride])
                .collect();
            fused_split_eq_quotients_with_params(
                &cyc_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                w_hat,
                t_hat,
                z_pre,
                z_pre_max_abs,
                p,
            )
        }
    }
}
