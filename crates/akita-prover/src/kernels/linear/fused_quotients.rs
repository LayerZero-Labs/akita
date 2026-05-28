use super::*;
use std::mem::size_of;

/// Minimum number of Rayon work-units for the fused one-shot kernel.
const MIN_FUSED_TILES: usize = 30;
#[cfg(target_arch = "aarch64")]
const FUSED_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const FUSED_L2_CACHE_BYTES: usize = 1024 * 1024;

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
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    let mat_width = cyc_rows.first().map_or(0, |r| r.len());
    let w_len = w_hat.len().min(mat_width);
    let t_len = t_hat.len().min(mat_width);
    let z_len = z_pre.len().min(mat_width);
    let max_col = w_len.max(t_len).max(z_len);

    if max_col == 0 {
        return Ok((
            vec![CyclotomicRing::<F, D>::zero(); n_d],
            vec![CyclotomicRing::<F, D>::zero(); n_b],
            vec![CyclotomicRing::<F, D>::zero(); n_a],
        ));
    }

    let z_abs_bound = u64::from(z_pre_max_abs);
    debug_assert!(
        centered_rows_within_bound(z_pre, z_len, z_abs_bound),
        "fused quotient centered RHS bound is smaller than the actual max"
    );
    let w_safe = w_len == 0
        || safe_crt_chunk_width::<F, W, K, D>(params, w_len, w_digit_abs_bound) == Some(w_len);
    let t_safe = t_len == 0
        || safe_crt_chunk_width::<F, W, K, D>(params, t_len, t_digit_abs_bound) == Some(t_len);
    let z_safe = z_len == 0
        || z_abs_bound == 0
        || safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_abs_bound) == Some(z_len);
    if w_safe && t_safe && z_safe {
        return Ok(fused_split_eq_quotients_one_shot(
            cyc_rows,
            neg_rows,
            n_d,
            n_b,
            n_a,
            w_hat,
            t_hat,
            z_pre,
            z_abs_bound,
            max_col,
            w_len,
            t_len,
            z_len,
            params,
        ));
    }

    let d_result =
        accumulate_cyclic_i8_rows(cyc_rows, n_d, w_hat, w_len, w_digit_abs_bound, params);
    let b_result =
        accumulate_cyclic_i8_rows(cyc_rows, n_b, t_hat, t_len, t_digit_abs_bound, params);
    let a_result = accumulate_centered_quotient_rows(
        neg_rows,
        cyc_rows,
        n_a,
        z_pre,
        z_len,
        z_abs_bound,
        params,
    );

    Ok((d_result, b_result, a_result))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_one_shot<
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
    z_abs_bound: u64,
    max_col: usize,
    w_len: usize,
    t_len: usize,
    z_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let lut = DigitMontLut::new(params);
    let centered_lut = (z_abs_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, z_abs_bound as i32));
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
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
                    for (acc_d, cyc_row) in accs.0.iter_mut().zip(cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_d, &cyc_row[j], &ntt_w, params);
                    }
                }

                if j < t_len && !is_zero_plane(&t_hat[j]) {
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, &lut);
                    for (acc_b, cyc_row) in accs.1.iter_mut().zip(cyc_rows.iter()) {
                        accumulate_pointwise_product_into(acc_b, &cyc_row[j], &ntt_t, params);
                    }
                }

                if j < z_len && !is_zero_centered_row(&z_pre[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut(&z_pre[j], params, lut)
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_pre[j], params)
                    };
                    for ((acc_neg, acc_cyc), (neg_row, cyc_row)) in accs
                        .2
                        .iter_mut()
                        .zip(accs.3.iter_mut())
                        .zip(neg_rows.iter().zip(cyc_rows.iter()))
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
            quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring)
        })
        .collect();

    (d_result, b_result, a_result)
}

fn accumulate_cyclic_i8_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    rhs: &[[i8; D]],
    rhs_len: usize,
    rhs_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if rhs_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let chunk_width = safe_crt_chunk_width::<F, W, K, D>(params, rhs_len, rhs_abs_bound)
        .expect("single i8 CRT term must fit supported parameters");
    if rhs_len <= chunk_width {
        let (rows, _, _) = fused_split_eq_quotients_one_shot(
            cyc_rows,
            &[],
            num_rows,
            0,
            0,
            rhs,
            &[],
            &[],
            0,
            rhs_len,
            rhs_len,
            0,
            0,
            params,
        );
        return rows;
    }

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut = DigitMontLut::new(params);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(rhs_len);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk_start..chunk_end {
                if is_zero_plane(&rhs[j]) {
                    continue;
                }
                let ntt_rhs = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&rhs[j], params, &lut);
                for (acc, row) in accs.iter_mut().zip(cyc_rows.iter()) {
                    accumulate_pointwise_product_into(acc, &row[j], &ntt_rhs, params);
                }
            }

            for (dst, acc) in out.iter_mut().zip(accs) {
                *dst += acc.to_ring_cyclic(params);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn centered_rows_within_bound<const D: usize>(rows: &[[i32; D]], len: usize, bound: u64) -> bool {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .all(|&coeff| {
            if coeff == i32::MIN {
                bound >= (1u64 << 31)
            } else {
                u64::from(coeff.unsigned_abs()) <= bound
            }
        })
}

fn centered_i32_ring<F: CanonicalField, const D: usize>(coeffs: &[i32; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|k| F::from_i64(coeffs[k] as i64)))
}

fn accumulate_centered_quotient_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_pre: &[[i32; D]],
    z_len: usize,
    z_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if z_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    if z_abs_bound == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let Some(chunk_width) = safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_abs_bound) else {
        return accumulate_centered_quotient_rows_field(
            neg_rows, cyc_rows, num_rows, z_pre, z_len, params,
        );
    };
    if z_len <= chunk_width {
        let (_, _, rows) = fused_split_eq_quotients_one_shot(
            cyc_rows,
            neg_rows,
            0,
            0,
            num_rows,
            &[],
            &[],
            z_pre,
            z_abs_bound,
            z_len,
            0,
            0,
            z_len,
            params,
        );
        return rows;
    }

    let centered_lut = (z_abs_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, z_abs_bound as i32));
    let num_chunks = z_len.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk_start = chunk_idx * chunk_width;
            let chunk_end = (chunk_start + chunk_width).min(z_len);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk_start..chunk_end {
                if is_zero_centered_row(&z_pre[j]) {
                    continue;
                }
                let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_lut(&z_pre[j], params, lut)
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_pre[j], params)
                };
                for ((neg_acc, cyc_acc), (neg_row, cyc_row)) in neg_accs
                    .iter_mut()
                    .zip(cyc_accs.iter_mut())
                    .zip(neg_rows.iter().zip(cyc_rows.iter()))
                {
                    accumulate_pointwise_product_into(neg_acc, &neg_row[j], &ntt_z_neg, params);
                    accumulate_pointwise_product_into(cyc_acc, &cyc_row[j], &ntt_z_cyc, params);
                }
            }

            for ((dst, neg_acc), cyc_acc) in out.iter_mut().zip(neg_accs).zip(cyc_accs) {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                *dst += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut a: Vec<CyclotomicRing<F, D>>, b| {
            for (dst, src) in a.iter_mut().zip(b) {
                *dst += src;
            }
            a
        }
    )
}

fn accumulate_centered_quotient_rows_field<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_pre: &[[i32; D]],
    z_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            let mut out = CyclotomicRing::<F, D>::zero();
            for j in 0..z_len {
                if is_zero_centered_row(&z_pre[j]) {
                    continue;
                }
                let z = centered_i32_ring::<F, D>(&z_pre[j]);
                let neg_lhs: CyclotomicRing<F, D> =
                    neg_rows[row_idx][j].to_ring_with_params(params);
                let cyc_lhs: CyclotomicRing<F, D> = cyc_rows[row_idx][j].to_ring_cyclic(params);
                let neg_product = neg_lhs * z;
                let mut cyc_product = CyclotomicRing::<F, D>::zero();
                add_cyclic_product_into(&mut cyc_product, &cyc_lhs, &z);
                out += quotient_from_cyclic_and_negacyclic(&cyc_product, &neg_product);
            }
            out
        })
        .collect()
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
#[cfg(test)]
pub(crate) fn fused_split_eq_quotients<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    stride: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    fused_split_eq_quotients_with_digit_bound(
        slot,
        n_d,
        n_b,
        n_a,
        stride,
        w_hat,
        t_hat,
        z_pre,
        z_pre_max_abs,
        I8_RHS_MAX_ABS,
        I8_RHS_MAX_ABS,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fused_split_eq_quotients_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    stride: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    fused_split_eq_quotients_with_digit_bound(
        slot,
        n_d,
        n_b,
        n_a,
        stride,
        w_hat,
        t_hat,
        z_pre,
        z_pre_max_abs,
        BALANCED_DIGIT_RHS_MAX_ABS,
        I8_RHS_MAX_ABS,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fused_split_eq_quotients_with_digit_bound<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    n_d: usize,
    n_b: usize,
    n_a: usize,
    stride: usize,
    w_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_pre: &[[i32; D]],
    z_pre_max_abs: u32,
    w_digit_abs_bound: u64,
    t_digit_abs_bound: u64,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
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
                w_digit_abs_bound,
                t_digit_abs_bound,
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
                w_digit_abs_bound,
                t_digit_abs_bound,
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
                w_digit_abs_bound,
                t_digit_abs_bound,
                p,
            )
        }
    }
}
