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
    d_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    b_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    a_cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
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
    let d_width = d_cyc_rows.first().map_or(0, |r| r.len());
    let b_width = b_cyc_rows.first().map_or(0, |r| r.len());
    let a_width = a_cyc_rows.first().map_or(0, |r| r.len());
    let w_len = w_hat.len().min(d_width);
    let t_len = t_hat.len().min(b_width);
    let z_len = z_pre.len().min(a_width);
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
    let tw = base_tw.min(max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

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
                    let ntt_w = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&w_hat[j], params, &lut);
                    for (acc_d, cyc_row) in accs.0.iter_mut().zip(d_cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                    }
                }

                if j < t_len && !is_zero_plane(&t_hat[j]) {
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, &lut);
                    for (acc_b, cyc_row) in accs.1.iter_mut().zip(b_cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                    }
                }

                if j < z_len && !is_zero_centered_row(&z_pre[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref clut) = centered_lut {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut(&z_pre[j], params, clut)
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_pre[j], params)
                    };
                    for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in accs
                        .2
                        .iter_mut()
                        .zip(accs.3.iter_mut())
                        .zip(neg_rows.iter().zip(a_cyc_rows.iter()))
                    {
                        accumulate_pointwise_product_into(acc_neg, &neg_row[j], &ntt_z_neg, params);
                        accumulate_pointwise_product_into(acc_cyc, &cyc_row[j], &ntt_z_cyc, params);
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

    (d_result, b_result, a_result)
}

/// Fused split-eq quotient kernel dispatching over [`NttSlotCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · w_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_split_eq_quotients")]
pub fn fused_split_eq_quotients<F: FieldCore + CanonicalField + HalvingField, const D: usize>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let d_width = w_hat.len();
    let b_width = t_hat.len();
    let a_width = z_pre.len();
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &neg[i * a_width..(i + 1) * a_width])
                .collect();
            let d_rows: Vec<&[_]> = (0..n_d)
                .map(|i| &cyc[i * d_width..(i + 1) * d_width])
                .collect();
            let b_rows: Vec<&[_]> = (0..n_b)
                .map(|i| &cyc[i * b_width..(i + 1) * b_width])
                .collect();
            let a_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &cyc[i * a_width..(i + 1) * a_width])
                .collect();
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
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
                .map(|i| &neg[i * a_width..(i + 1) * a_width])
                .collect();
            let d_rows: Vec<&[_]> = (0..n_d)
                .map(|i| &cyc[i * d_width..(i + 1) * d_width])
                .collect();
            let b_rows: Vec<&[_]> = (0..n_b)
                .map(|i| &cyc[i * b_width..(i + 1) * b_width])
                .collect();
            let a_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &cyc[i * a_width..(i + 1) * a_width])
                .collect();
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
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
                .map(|i| &neg[i * a_width..(i + 1) * a_width])
                .collect();
            let d_rows: Vec<&[_]> = (0..n_d)
                .map(|i| &cyc[i * d_width..(i + 1) * d_width])
                .collect();
            let b_rows: Vec<&[_]> = (0..n_b)
                .map(|i| &cyc[i * b_width..(i + 1) * b_width])
                .collect();
            let a_rows: Vec<&[_]> = (0..n_a)
                .map(|i| &cyc[i * a_width..(i + 1) * a_width])
                .collect();
            fused_split_eq_quotients_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
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
