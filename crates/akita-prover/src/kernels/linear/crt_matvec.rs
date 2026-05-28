use super::*;

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn mat_vec_mul_unchecked<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = Vec::with_capacity(mat.len());
    for row in mat {
        debug_assert_eq!(row.len(), vec.len());
        let mut acc = CyclotomicRing::<F, D>::zero();
        for (a, x) in row.iter().zip(vec.iter()) {
            acc += *a * *x;
        }
        out.push(acc);
    }
    out
}

#[cfg(all(test, not(feature = "zk")))]
pub(super) fn precompute_dense_mat_ntt_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicCrtNtt<W, K, D>>> {
    cfg_iter!(mat)
        .map(|row| {
            row.iter()
                .map(|a| CyclotomicCrtNtt::from_ring_with_params(a, params))
                .collect()
        })
        .collect()
}

#[cfg(all(test, not(feature = "zk")))]
fn mat_vec_mul_dense_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
        .iter()
        .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
        .collect();

    mat.iter()
        .map(|row| {
            debug_assert_eq!(row.len(), ntt_vec.len());
            let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
            for (a, x_ntt) in row.iter().zip(ntt_vec.iter()) {
                let a_ntt = CyclotomicCrtNtt::from_ring_with_params(a, params);
                accumulate_pointwise_product_into(&mut acc, &a_ntt, x_ntt, params);
            }
            acc.to_ring_with_params(params)
        })
        .collect()
}

#[cfg(all(test, not(feature = "zk")))]
fn mat_vec_mul_dense_many_with_params<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    let ntt_mat = precompute_dense_mat_ntt_with_params(mat, params);
    vecs.iter()
        .map(|vec| {
            let ntt_vec: Vec<CyclotomicCrtNtt<W, K, D>> = vec
                .iter()
                .map(|v| CyclotomicCrtNtt::from_ring_with_params(v, params))
                .collect();

            ntt_mat
                .iter()
                .map(|row_ntt| {
                    debug_assert_eq!(row_ntt.len(), ntt_vec.len());
                    let mut acc = CyclotomicCrtNtt::<W, K, D>::zero();
                    for (a_ntt, x_ntt) in row_ntt.iter().zip(ntt_vec.iter()) {
                        accumulate_pointwise_product_into(&mut acc, a_ntt, x_ntt, params);
                    }
                    acc.to_ring_with_params(params)
                })
                .collect()
        })
        .collect()
}

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn mat_vec_mul_crt_ntt<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_with_params(mat, vec, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_with_params(mat, vec, p),
    };
    Ok(out)
}

#[cfg(all(test, not(feature = "zk")))]
pub(crate) fn mat_vec_mul_crt_ntt_many<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vecs: &[Vec<CyclotomicRing<F, D>>],
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
    let params = select_crt_ntt_params::<F, D>()?;
    let out = match &params {
        ProtocolCrtNttParams::Q32(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q64(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
        ProtocolCrtNttParams::Q128(p) => mat_vec_mul_dense_many_with_params(mat, vecs, p),
    };
    Ok(out)
}

fn unreduced_quotient_ntt<F, W, const K: usize, const D: usize>(
    ntt_row: &[CyclotomicCrtNtt<W, K, D>],
    cyc_row: &[CyclotomicCrtNtt<W, K, D>],
    vec_neg: &[CyclotomicCrtNtt<W, K, D>],
    vec_cyc: &[CyclotomicCrtNtt<W, K, D>],
    rhs_max_abs: u32,
    params: &CrtNttParamSet<W, K, D>,
) -> CyclotomicRing<F, D>
where
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
{
    let n = ntt_row.len().min(vec_neg.len());
    let chunk_width = crt_accumulation_chunk_width::<F, W, K, D>(rhs_max_abs, n);
    let mut out = CyclotomicRing::<F, D>::zero();

    for chunk_start in (0..n).step_by(chunk_width) {
        let chunk_end = (chunk_start + chunk_width).min(n);
        let mut acc_neg = CyclotomicCrtNtt::<W, K, D>::zero();
        let mut acc_cyc = CyclotomicCrtNtt::<W, K, D>::zero();

        for j in chunk_start..chunk_end {
            accumulate_pointwise_product_into(&mut acc_neg, &ntt_row[j], &vec_neg[j], params);
            accumulate_pointwise_product_into(&mut acc_cyc, &cyc_row[j], &vec_cyc[j], params);
        }

        let neg_ring: CyclotomicRing<F, D> = acc_neg.to_ring_with_params(params);
        let cyc_ring: CyclotomicRing<F, D> = acc_cyc.to_ring_cyclic(params);

        let neg_coeffs = neg_ring.coefficients();
        let cyc_coeffs = cyc_ring.coefficients();
        let quotient: [F; D] = from_fn(|k| (cyc_coeffs[k] - neg_coeffs[k]).half());
        add_ring_into(&mut out, CyclotomicRing::from_coefficients(quotient));
    }

    out
}

/// Compute unreduced quotients for matrix rows against a witness vector.
///
/// For each row: `r_i = high_part(sum_j row_ij * vec_j) = (cyc - neg) / 2`.
/// Vec NTT conversions and matrix cyclic NTT are precomputed once (not per-row).
pub fn unreduced_quotient_rows_ntt_cached<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    num_rows: usize,
    num_cols: usize,
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    match slot {
        NttSlotCache::Q32 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            let cyc_rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            let n = num_cols.min(vec.len());
            let v_neg: Vec<_> = cfg_iter!(vec[..n])
                .map(|x| CyclotomicCrtNtt::from_ring_with_params(x, p))
                .collect();
            let v_cyc: Vec<_> = cfg_iter!(vec[..n])
                .map(|x| CyclotomicCrtNtt::from_ring_cyclic(x, p))
                .collect();
            let rhs_max_abs = max_centered_abs_u32(&vec[..n]).unwrap_or(u32::MAX);
            cfg_into_iter!(0..num_rows)
                .map(|i| {
                    unreduced_quotient_ntt(neg_rows[i], cyc_rows[i], &v_neg, &v_cyc, rhs_max_abs, p)
                })
                .collect()
        }
        NttSlotCache::Q64 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            let cyc_rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            let n = num_cols.min(vec.len());
            let v_neg: Vec<_> = cfg_iter!(vec[..n])
                .map(|x| CyclotomicCrtNtt::from_ring_with_params(x, p))
                .collect();
            let v_cyc: Vec<_> = cfg_iter!(vec[..n])
                .map(|x| CyclotomicCrtNtt::from_ring_cyclic(x, p))
                .collect();
            let rhs_max_abs = max_centered_abs_u32(&vec[..n]).unwrap_or(u32::MAX);
            cfg_into_iter!(0..num_rows)
                .map(|i| {
                    unreduced_quotient_ntt(neg_rows[i], cyc_rows[i], &v_neg, &v_cyc, rhs_max_abs, p)
                })
                .collect()
        }
        NttSlotCache::Q128 {
            neg,
            cyc,
            params: p,
        } => {
            let neg_rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &neg[i * num_cols..(i + 1) * num_cols])
                .collect();
            let cyc_rows: Vec<&[_]> = (0..num_rows)
                .map(|i| &cyc[i * num_cols..(i + 1) * num_cols])
                .collect();
            let n = num_cols.min(vec.len());
            let v_neg: Vec<_> = cfg_iter!(vec[..n])
                .map(|x| CyclotomicCrtNtt::from_ring_with_params(x, p))
                .collect();
            let v_cyc: Vec<_> = cfg_iter!(vec[..n])
                .map(|x| CyclotomicCrtNtt::from_ring_cyclic(x, p))
                .collect();
            let rhs_max_abs = max_centered_abs_u32(&vec[..n]).unwrap_or(u32::MAX);
            cfg_into_iter!(0..num_rows)
                .map(|i| {
                    unreduced_quotient_ntt(neg_rows[i], cyc_rows[i], &v_neg, &v_cyc, rhs_max_abs, p)
                })
                .collect()
        }
    }
}
