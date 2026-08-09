use super::*;

/// Streamed variant of [`fused_split_eq_quotients_one_shot`]: instead of
/// reading pre-transformed entries from a prepared NTT cache, it transforms
/// each needed element of A's field form on the fly inside the column-tile
/// loop (cyclic-only for the D/B products, both domains for the A quotient).
///
/// The root-level ring-switch relation views nearly the whole matrix but
/// reads every element exactly once per prove, so caching its transform is
/// pure standing memory (~30 GiB at the jolt 2^26 shape); streaming trades
/// that for transform work inside an already-parallel pass. Entry values are
/// produced by the same `from_ring_*_with_params` conversions that populate
/// the cache, so results are bit-identical to the cached path.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn fused_split_eq_quotients_one_shot_streamed<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    params: &CrtNttParamSet<W, K, D>,
) -> (
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
    Vec<CyclotomicRing<F, D>>,
) {
    let digit_bound = plan.w_digit_abs_bound.max(plan.t_digit_abs_bound);
    let digit_lut = (plan.witness_len != 0 || plan.t_len != 0)
        .then(|| DigitMontLut::<W, K>::new_with_digit_bound(params, digit_bound));
    let centered_lut = (plan.z_len != 0 && plan.z_bounds.lut <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, plan.z_bounds.lut as i32));
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(plan.max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = plan.max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let (d_accs, b_accs, a_neg_accs, a_cyc_accs) = cfg_fold_reduce!(
        0..num_tiles,
        || (
            vec![zero.clone(); plan.n_d],
            vec![zero.clone(); plan.n_b],
            vec![zero.clone(); plan.n_a],
            vec![zero.clone(); plan.n_a],
        ),
        |mut accs: (
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
            Vec<CyclotomicCrtNtt<W, K, D>>,
        ),
         tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(plan.max_col);

            for j in tile_start..tile_end {
                if j < plan.witness_len && !is_zero_plane(&e_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_w = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&e_hat[j], params, lut);
                    for (i, acc_d) in accs.0.iter_mut().enumerate() {
                        let a_ring = source[plan.d_row(i).start + j];
                        let a_cyc = CyclotomicCrtNtt::from_ring_cyclic(&a_ring, params);
                        accumulate_pointwise_product_into(acc_d, &a_cyc, &ntt_w, params);
                    }
                }

                if j < plan.t_len && !is_zero_plane(&t_hat[j]) {
                    let lut = digit_lut.as_ref().expect("digit LUT exists");
                    let ntt_t = CyclotomicCrtNtt::from_i8_cyclic_with_lut(&t_hat[j], params, lut);
                    for (i, acc_b) in accs.1.iter_mut().enumerate() {
                        let a_ring = source[plan.b_row(i).start + j];
                        let a_cyc = CyclotomicCrtNtt::from_ring_cyclic(&a_ring, params);
                        accumulate_pointwise_product_into(acc_b, &a_cyc, &ntt_t, params);
                    }
                }

                if j < plan.z_len && !is_zero_centered_row(&z_folded_rings[j]) {
                    let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                        // SAFETY: `plan_fused_quotients` computed
                        // `z_bounds.lut` from these `plan.z_len` rows. This
                        // loop keeps `j < plan.z_len`, and `lut` was built for
                        // that inclusive centered coefficient bound.
                        unsafe {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                                &z_folded_rings[j],
                                params,
                                lut,
                            )
                        }
                    } else {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_params(
                            &z_folded_rings[j],
                            params,
                        )
                    };
                    for (i, (acc_neg, acc_cyc)) in
                        accs.2.iter_mut().zip(accs.3.iter_mut()).enumerate()
                    {
                        let a_ring = source[plan.a_row(i).start + j];
                        let (a_neg, a_cyc) =
                            CyclotomicCrtNtt::from_ring_pair_with_params(&a_ring, params);
                        accumulate_pointwise_product_into(acc_neg, &a_neg, &ntt_z_neg, params);
                        accumulate_pointwise_product_into(acc_cyc, &a_cyc, &ntt_z_cyc, params);
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
            for r in 0..plan.n_d {
                add_ntt_into(&mut a.0[r], &b.0[r], params);
            }
            for r in 0..plan.n_b {
                add_ntt_into(&mut a.1[r], &b.1[r], params);
            }
            for r in 0..plan.n_a {
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
            let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
            let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
            quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring)
        })
        .collect();

    (d_result, b_result, a_result)
}

pub(super) fn streamed_cyclic_i8_rows_chunked<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    rhs: &[[i8; D]],
    plan: FusedQuotientPlan,
    role: DigitRole,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let (num_rows, rhs_len, rhs_abs_bound) = plan.digit_shape(role);
    if num_rows == 0 {
        return vec![];
    }
    if rhs_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, rhs_abs_bound);
    let num_chunks = rhs_len.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(rhs_len, chunk_width, chunk_idx);
            let mut accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for (j, rhs_row) in rhs.iter().enumerate().take(chunk.end).skip(chunk.start) {
                if is_zero_plane(rhs_row) {
                    continue;
                }
                let ntt_rhs = CyclotomicCrtNtt::from_i8_cyclic_with_lut(rhs_row, params, &lut);
                for (i, acc) in accs.iter_mut().enumerate() {
                    let a_ring = source[plan.digit_row(role, i).start + j];
                    let a_cyc = CyclotomicCrtNtt::from_ring_cyclic(&a_ring, params);
                    accumulate_pointwise_product_into(acc, &a_cyc, &ntt_rhs, params);
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

/// Streamed counterpart of [`accumulate_centered_quotient_rows`]'s chunked
/// path: per safe-width chunk, transform each touched A element from the
/// field form (both domains) in-loop and reduce the chunk's accumulators to
/// a quotient ring contribution. Chunk bracketing matches the cached path;
/// the arithmetic is exact, so results are identical to it.
#[allow(clippy::needless_range_loop)]
pub(super) fn streamed_centered_quotient_rows_chunked<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    z_folded_rings: &[[i32; D]],
    plan: FusedQuotientPlan,
    chunk_width: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    let num_rows = plan.n_a;
    let z_len = plan.z_len;
    let z_lut_abs_bound = plan.z_bounds.lut;
    if num_rows == 0 {
        return vec![];
    }
    if z_len == 0 || z_lut_abs_bound == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }
    let centered_lut = (z_lut_abs_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, z_lut_abs_bound as i32));
    let num_chunks = z_len.div_ceil(chunk_width);

    cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let chunk = FusedQuotientPlan::chunk_range(z_len, chunk_width, chunk_idx);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for j in chunk {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
                    // SAFETY: `z_lut_abs_bound` is the actual centered bound
                    // for the first `z_len` rows. This loop keeps `j < z_len`,
                    // and `lut` was built for that inclusive bound.
                    unsafe {
                        CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                            &z_folded_rings[j],
                            params,
                            lut,
                        )
                    }
                } else {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_params(&z_folded_rings[j], params)
                };
                for (i, (neg_acc, cyc_acc)) in
                    neg_accs.iter_mut().zip(cyc_accs.iter_mut()).enumerate()
                {
                    let a_ring = source[plan.a_row(i).start + j];
                    let (a_neg, a_cyc) =
                        CyclotomicCrtNtt::from_ring_pair_with_params(&a_ring, params);
                    accumulate_pointwise_product_into(neg_acc, &a_neg, &ntt_z_neg, params);
                    accumulate_pointwise_product_into(cyc_acc, &a_cyc, &ntt_z_cyc, params);
                }
            }

            for ((dst, neg_acc), cyc_acc) in out.iter_mut().zip(neg_accs).zip(cyc_accs) {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring(params);
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
