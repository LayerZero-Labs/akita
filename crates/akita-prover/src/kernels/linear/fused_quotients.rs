use super::*;
use std::mem::size_of;

mod dispatch;

/// Minimum number of Rayon work-units for the fused one-shot kernel.
const MIN_FUSED_TILES: usize = 30;
#[cfg(target_arch = "aarch64")]
const FUSED_L2_CACHE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const FUSED_L2_CACHE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct CenteredRhsBounds {
    capacity: u64,
}

#[derive(Clone, Copy, Default)]
struct ObservedI8Bounds {
    abs: u64,
    lut: u64,
}

struct CyclicI8Request<'a, W: PrimeWidth, const K: usize, const D: usize> {
    cyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    coeffs: &'a [[i8; D]],
    abs_bound: u64,
}

#[allow(dead_code)]
struct PairedI8Request<'a, W: PrimeWidth, const K: usize, const D: usize> {
    cyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    negacyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    coeffs: &'a [[i8; D]],
    abs_bound: u64,
}

struct PairedCenteredI32Request<'a, W: PrimeWidth, const K: usize, const D: usize> {
    cyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    negacyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    coeffs: &'a [[i32; D]],
    bounds: CenteredRhsBounds,
}

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
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
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
    let z_bounds = CenteredRhsBounds {
        capacity: u64::from(z_folded_max_abs),
    };
    let cyclic_requests = [
        CyclicI8Request {
            cyclic_rows: d_cyc_rows,
            num_rows: n_d,
            coeffs: e_hat,
            abs_bound: w_digit_abs_bound,
        },
        CyclicI8Request {
            cyclic_rows: b_cyc_rows,
            num_rows: n_b,
            coeffs: t_hat,
            abs_bound: t_digit_abs_bound,
        },
    ];
    let centered_requests = [PairedCenteredI32Request {
        cyclic_rows: a_cyc_rows,
        negacyclic_rows: neg_rows,
        num_rows: n_a,
        coeffs: z_folded_rings,
        bounds: z_bounds,
    }];
    let ([d_result, b_result], [], [a_result]) = fused_quotient_rhs_batch::<F, W, K, D, 2, 0, 1>(
        &cyclic_requests,
        &[],
        &centered_requests,
        params,
    )?;
    Ok((d_result, b_result, a_result))
}

struct StaticFusedAcc<
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const NC: usize,
    const NP: usize,
    const NI: usize,
> {
    cyclic: [Vec<CyclotomicCrtNtt<W, K, D>>; NC],
    paired_i8_neg: [Vec<CyclotomicCrtNtt<W, K, D>>; NP],
    paired_i8_cyc: [Vec<CyclotomicCrtNtt<W, K, D>>; NP],
    centered_neg: [Vec<CyclotomicCrtNtt<W, K, D>>; NI],
    centered_cyc: [Vec<CyclotomicCrtNtt<W, K, D>>; NI],
}

fn validate_rows<W: PrimeWidth, const K: usize, const D: usize>(
    rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    width: usize,
    label: &str,
) -> Result<(), AkitaError> {
    if rows.len() != num_rows || rows.iter().any(|row| row.len() < width) {
        return Err(AkitaError::InvalidInput(format!(
            "fused quotient {label} matrix shape does not match request"
        )));
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn fused_quotient_rhs_batch<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
    const NC: usize,
    const NP: usize,
    const NI: usize,
>(
    cyclic_requests: &[CyclicI8Request<'_, W, K, D>; NC],
    paired_i8_requests: &[PairedI8Request<'_, W, K, D>; NP],
    centered_requests: &[PairedCenteredI32Request<'_, W, K, D>; NI],
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        [Vec<CyclotomicRing<F, D>>; NC],
        [Vec<CyclotomicRing<F, D>>; NP],
        [Vec<CyclotomicRing<F, D>>; NI],
    ),
    AkitaError,
> {
    let mut cyclic_observed = [ObservedI8Bounds::default(); NC];
    for (request, observed) in cyclic_requests.iter().zip(cyclic_observed.iter_mut()) {
        validate_rows(
            request.cyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "cyclic",
        )?;
        *observed = validate_i8_rows_and_bound(request.coeffs, request.abs_bound)?;
    }
    let mut paired_i8_observed = [ObservedI8Bounds::default(); NP];
    for (request, observed) in paired_i8_requests.iter().zip(paired_i8_observed.iter_mut()) {
        validate_rows(
            request.cyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "paired cyclic",
        )?;
        validate_rows(
            request.negacyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "paired negacyclic",
        )?;
        *observed = validate_i8_rows_and_bound(request.coeffs, request.abs_bound)?;
    }
    let mut centered_observed = [0u64; NI];
    for (request, observed) in centered_requests.iter().zip(centered_observed.iter_mut()) {
        validate_rows(
            request.cyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "centered cyclic",
        )?;
        validate_rows(
            request.negacyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "centered negacyclic",
        )?;
        *observed = centered_rows_abs_bound(request.coeffs, request.coeffs.len());
    }
    let max_col = cyclic_requests
        .iter()
        .map(|request| request.coeffs.len())
        .chain(
            paired_i8_requests
                .iter()
                .map(|request| request.coeffs.len()),
        )
        .chain(centered_requests.iter().map(|request| request.coeffs.len()))
        .max()
        .unwrap_or(0);
    if max_col == 0 {
        return Ok((
            std::array::from_fn(|i| vec![CyclotomicRing::zero(); cyclic_requests[i].num_rows]),
            std::array::from_fn(|i| vec![CyclotomicRing::zero(); paired_i8_requests[i].num_rows]),
            std::array::from_fn(|i| vec![CyclotomicRing::zero(); centered_requests[i].num_rows]),
        ));
    }
    let all_one_shot = cyclic_requests.iter().all(|request| {
        safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
            == Some(request.coeffs.len())
    }) && paired_i8_requests.iter().all(|request| {
        safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
            == Some(request.coeffs.len())
    }) && centered_requests.iter().zip(centered_observed).all(
        |(request, observed)| {
            safe_crt_chunk_width::<F, W, K, D>(
                params,
                request.coeffs.len(),
                request.bounds.capacity.max(observed),
            ) == Some(request.coeffs.len())
        },
    );
    if !all_one_shot {
        let mut cyclic = Vec::with_capacity(NC);
        for (request, observed) in cyclic_requests.iter().zip(cyclic_observed) {
            cyclic.push(accumulate_cyclic_i8_rows(
                request.cyclic_rows,
                request.num_rows,
                request.coeffs,
                request.coeffs.len(),
                request.abs_bound,
                observed.lut,
                params,
            )?);
        }
        let cyclic: [Vec<CyclotomicRing<F, D>>; NC] = cyclic.try_into().map_err(|_| {
            AkitaError::InvalidInput("fused cyclic fallback result count mismatch".into())
        })?;
        let paired_i8 = std::array::from_fn(|i| {
            let request = &paired_i8_requests[i];
            let centered: Vec<[i32; D]> = request
                .coeffs
                .iter()
                .map(|row| from_fn(|j| i32::from(row[j])))
                .collect();
            accumulate_centered_quotient_rows(
                request.negacyclic_rows,
                request.cyclic_rows,
                request.num_rows,
                &centered,
                centered.len(),
                CenteredRhsBounds {
                    capacity: request.abs_bound.max(paired_i8_observed[i].abs),
                },
                paired_i8_observed[i].abs,
                params,
            )
        });
        let centered = std::array::from_fn(|i| {
            let request = &centered_requests[i];
            accumulate_centered_quotient_rows(
                request.negacyclic_rows,
                request.cyclic_rows,
                request.num_rows,
                request.coeffs,
                request.coeffs.len(),
                CenteredRhsBounds {
                    capacity: request.bounds.capacity.max(centered_observed[i]),
                },
                centered_observed[i],
                params,
            )
        });
        return Ok((cyclic, paired_i8, centered));
    }

    let digit_lut_bound = cyclic_observed
        .iter()
        .map(|observed| observed.lut)
        .max()
        .unwrap_or(1);
    let digit_lut = cyclic_requests
        .iter()
        .any(|request| !request.coeffs.is_empty())
        .then(|| DigitMontLut::<W, K>::new_with_digit_bound(params, digit_lut_bound));
    let paired_i8_bound = paired_i8_observed
        .iter()
        .map(|observed| observed.abs)
        .max()
        .unwrap_or(0);
    let paired_i8_lut = paired_i8_requests
        .iter()
        .any(|request| !request.coeffs.is_empty())
        .then(|| CenteredMontLut::<W, K>::new(params, paired_i8_bound as i32));
    let centered_luts: [Option<CenteredMontLut<W, K>>; NI] = std::array::from_fn(|i| {
        (!centered_requests[i].coeffs.is_empty()
            && centered_observed[i] <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, centered_observed[i] as i32))
    });
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let accumulators = cfg_fold_reduce!(
        0..num_tiles,
        || StaticFusedAcc {
            cyclic: std::array::from_fn(|i| vec![zero.clone(); cyclic_requests[i].num_rows]),
            paired_i8_neg: std::array::from_fn(|i| vec![
                zero.clone();
                paired_i8_requests[i].num_rows
            ]),
            paired_i8_cyc: std::array::from_fn(|i| vec![
                zero.clone();
                paired_i8_requests[i].num_rows
            ]),
            centered_neg: std::array::from_fn(|i| vec![
                zero.clone();
                centered_requests[i].num_rows
            ]),
            centered_cyc: std::array::from_fn(|i| vec![
                zero.clone();
                centered_requests[i].num_rows
            ]),
        },
        |mut accs: StaticFusedAcc<W, K, D, NC, NP, NI>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(max_col);
            for j in tile_start..tile_end {
                for (request, lane_acc) in cyclic_requests.iter().zip(accs.cyclic.iter_mut()) {
                    if j < request.coeffs.len() && !is_zero_plane(&request.coeffs[j]) {
                        let ntt = CyclotomicCrtNtt::from_i8_cyclic_with_lut(
                            &request.coeffs[j],
                            params,
                            digit_lut.as_ref().expect("digit LUT exists"),
                        );
                        for (row_acc, row) in lane_acc.iter_mut().zip(request.cyclic_rows) {
                            accumulate_pointwise_product_into(row_acc, &row[j], &ntt, params);
                        }
                    }
                }
                for ((request, neg_accs), cyc_accs) in paired_i8_requests
                    .iter()
                    .zip(accs.paired_i8_neg.iter_mut())
                    .zip(accs.paired_i8_cyc.iter_mut())
                {
                    if j < request.coeffs.len() && !is_zero_plane(&request.coeffs[j]) {
                        let centered = from_fn(|k| i32::from(request.coeffs[j][k]));
                        let (neg_ntt, cyc_ntt) = unsafe {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                                &centered,
                                params,
                                paired_i8_lut.as_ref().expect("paired i8 LUT exists"),
                            )
                        };
                        for (row_acc, row) in cyc_accs.iter_mut().zip(request.cyclic_rows) {
                            accumulate_pointwise_product_into(row_acc, &row[j], &cyc_ntt, params);
                        }
                        for (row_acc, row) in neg_accs.iter_mut().zip(request.negacyclic_rows) {
                            accumulate_pointwise_product_into(row_acc, &row[j], &neg_ntt, params);
                        }
                    }
                }
                for lane in 0..NI {
                    let request = &centered_requests[lane];
                    if j < request.coeffs.len() && !is_zero_centered_row(&request.coeffs[j]) {
                        let (neg_ntt, cyc_ntt) = if let Some(lut) = &centered_luts[lane] {
                            unsafe {
                                CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                                    &request.coeffs[j],
                                    params,
                                    lut,
                                )
                            }
                        } else {
                            CyclotomicCrtNtt::from_centered_i32_pair_with_params(
                                &request.coeffs[j],
                                params,
                            )
                        };
                        for (row_acc, row) in
                            accs.centered_cyc[lane].iter_mut().zip(request.cyclic_rows)
                        {
                            accumulate_pointwise_product_into(row_acc, &row[j], &cyc_ntt, params);
                        }
                        for (row_acc, row) in accs.centered_neg[lane]
                            .iter_mut()
                            .zip(request.negacyclic_rows)
                        {
                            accumulate_pointwise_product_into(row_acc, &row[j], &neg_ntt, params);
                        }
                    }
                }
            }
            accs
        },
        |mut a: StaticFusedAcc<W, K, D, NC, NP, NI>, b| {
            for lane in 0..NC {
                for (dst, src) in a.cyclic[lane].iter_mut().zip(b.cyclic[lane].iter()) {
                    add_ntt_into(dst, src, params);
                }
            }
            for lane in 0..NP {
                for (dst, src) in a.paired_i8_neg[lane]
                    .iter_mut()
                    .zip(b.paired_i8_neg[lane].iter())
                {
                    add_ntt_into(dst, src, params);
                }
                for (dst, src) in a.paired_i8_cyc[lane]
                    .iter_mut()
                    .zip(b.paired_i8_cyc[lane].iter())
                {
                    add_ntt_into(dst, src, params);
                }
            }
            for lane in 0..NI {
                for (dst, src) in a.centered_neg[lane]
                    .iter_mut()
                    .zip(b.centered_neg[lane].iter())
                {
                    add_ntt_into(dst, src, params);
                }
                for (dst, src) in a.centered_cyc[lane]
                    .iter_mut()
                    .zip(b.centered_cyc[lane].iter())
                {
                    add_ntt_into(dst, src, params);
                }
            }
            a
        }
    );
    let cyclic = accumulators.cyclic.map(|rows| {
        rows.into_iter()
            .map(|value| value.to_ring_cyclic(params))
            .collect()
    });
    let paired_i8 = std::array::from_fn(|i| {
        accumulators.paired_i8_neg[i]
            .iter()
            .zip(&accumulators.paired_i8_cyc[i])
            .map(|(neg, cyc)| {
                quotient_from_cyclic_and_negacyclic(
                    &cyc.to_ring_cyclic(params),
                    &neg.to_ring_with_params(params),
                )
            })
            .collect()
    });
    let centered = std::array::from_fn(|i| {
        accumulators.centered_neg[i]
            .iter()
            .zip(&accumulators.centered_cyc[i])
            .map(|(neg, cyc)| {
                quotient_from_cyclic_and_negacyclic(
                    &cyc.to_ring_cyclic(params),
                    &neg.to_ring_with_params(params),
                )
            })
            .collect()
    });
    Ok((cyclic, paired_i8, centered))
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
    rhs_observed_abs: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if num_rows == 0 {
        return Ok(vec![]);
    }
    if rhs_len == 0 {
        return Ok(vec![CyclotomicRing::<F, D>::zero(); num_rows]);
    }

    let chunk_width = safe_crt_chunk_width::<F, W, K, D>(params, rhs_len, rhs_abs_bound)
        .ok_or_else(|| {
            AkitaError::InvalidInput("fused cyclic declared capacity exceeds CRT support".into())
        })?;
    debug_assert!(chunk_width < rhs_len);

    let num_chunks = rhs_len.div_ceil(chunk_width);
    let lut_bound = rhs_observed_abs.max(1).next_power_of_two();
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, lut_bound);

    Ok(cfg_fold_reduce!(
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
    ))
}

fn validate_i8_rows_and_bound<const D: usize>(
    rows: &[[i8; D]],
    declared: u64,
) -> Result<ObservedI8Bounds, AkitaError> {
    let (abs, max_positive) =
        rows.iter()
            .flat_map(|row| row.iter())
            .fold((0u64, 0u64), |(abs, positive), &coeff| {
                (
                    abs.max(u64::from(coeff.unsigned_abs())),
                    positive.max(u64::try_from(coeff).unwrap_or(0)),
                )
            });
    let lut = abs
        .max(max_positive.saturating_add(1))
        .max(1)
        .next_power_of_two();
    if abs > declared || lut > declared {
        return Err(AkitaError::InvalidInput(
            "fused quotient digits exceed their declared balanced range".into(),
        ));
    }
    Ok(ObservedI8Bounds { abs, lut })
}

fn centered_rows_abs_bound<const D: usize>(rows: &[[i32; D]], len: usize) -> u64 {
    rows.iter()
        .take(len)
        .flat_map(|row| row.iter())
        .map(|&coeff| u64::from(coeff.unsigned_abs()))
        .max()
        .unwrap_or(0)
}

fn centered_i32_ring<F: CanonicalField, const D: usize>(coeffs: &[i32; D]) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(from_fn(|k| F::from_i64(coeffs[k] as i64)))
}

#[allow(clippy::too_many_arguments)]
fn accumulate_centered_quotient_rows<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    neg_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    cyc_rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_len: usize,
    z_bounds: CenteredRhsBounds,
    actual_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    if num_rows == 0 {
        return vec![];
    }
    if z_len == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    if actual_abs_bound == 0 {
        return vec![CyclotomicRing::<F, D>::zero(); num_rows];
    }

    let Some(chunk_width) = safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity)
    else {
        return accumulate_centered_quotient_rows_field(
            neg_rows,
            cyc_rows,
            num_rows,
            z_folded_rings,
            z_len,
            params,
        );
    };
    debug_assert!(chunk_width < z_len);

    let centered_lut = (actual_abs_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<W, K>::new(params, actual_abs_bound as i32));
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
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let (ntt_z_neg, ntt_z_cyc) = if let Some(ref lut) = centered_lut {
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
    z_folded_rings: &[[i32; D]],
    z_len: usize,
    params: &CrtNttParamSet<W, K, D>,
) -> Vec<CyclotomicRing<F, D>> {
    cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            let mut out = CyclotomicRing::<F, D>::zero();
            for j in 0..z_len {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let z = centered_i32_ring::<F, D>(&z_folded_rings[j]);
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
/// - D-cyclic: `cyc[0..n_d] · e_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
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
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    dispatch::fused_split_eq_quotients_with_digit_bound(
        slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        balanced_digit_abs_bound(6),
        balanced_digit_abs_bound(6),
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
    e_hat: &[[i8; D]],
    t_hat: &[[i8; D]],
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    log_basis: u32,
) -> Result<
    (
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
        Vec<CyclotomicRing<F, D>>,
    ),
    AkitaError,
> {
    validate_i8_log_basis(log_basis)?;
    let digit_bound = balanced_digit_abs_bound(log_basis);
    dispatch::fused_split_eq_quotients_with_digit_bound(
        slot,
        n_d,
        n_b,
        n_a,
        e_hat,
        t_hat,
        z_folded_rings,
        z_folded_max_abs,
        digit_bound,
        digit_bound,
    )
}

#[cfg(test)]
mod paired_i8_tests {
    use super::*;
    use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES};
    use akita_field::{Fp64, Prime128Offset275};

    #[test]
    fn paired_i8_lane_matches_cyclic_negacyclic_quotient_identity() {
        type F = Fp64<4_294_967_197>;
        const D: usize = 64;
        let ProtocolCrtNttParams::Q32(params) =
            select_crt_ntt_params::<F, D>().expect("Q32 test parameters")
        else {
            panic!("test field must use Q32 parameters");
        };
        let lhs =
            CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64((i % 7) as i64 - 3)));
        let rhs_coeffs = [[-1i8; D]];
        let rhs = CyclotomicRing::<F, D>::from_coefficients([F::from_i64(-1); D]);
        let neg =
            [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_with_params(&lhs, &params)];
        let cyc = [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_cyclic(&lhs, &params)];
        let neg_rows = [neg.as_slice()];
        let cyc_rows = [cyc.as_slice()];
        let requests = [PairedI8Request {
            cyclic_rows: &cyc_rows,
            negacyclic_rows: &neg_rows,
            num_rows: 1,
            coeffs: &rhs_coeffs,
            abs_bound: 1,
        }];

        let ([], [actual], []) = fused_quotient_rhs_batch::<F, i32, Q32_NUM_PRIMES, D, 0, 1, 0>(
            &[],
            &requests,
            &[],
            &params,
        )
        .expect("paired i8 request");
        let mut cyclic = CyclotomicRing::zero();
        add_cyclic_product_into(&mut cyclic, &lhs, &rhs);
        let expected = quotient_from_cyclic_and_negacyclic(&cyclic, &(lhs * rhs));
        assert_eq!(actual, vec![expected]);
    }

    #[test]
    fn centered_lane_derives_capacity_when_hint_is_underreported() {
        type F = Fp64<4_294_967_197>;
        const D: usize = 64;
        let ProtocolCrtNttParams::Q32(params) =
            select_crt_ntt_params::<F, D>().expect("Q32 test parameters")
        else {
            panic!("test field must use Q32 parameters");
        };
        let zero = CyclotomicRing::<F, D>::zero();
        let neg =
            [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_with_params(&zero, &params)];
        let cyc = [CyclotomicCrtNtt::<i32, Q32_NUM_PRIMES, D>::from_ring_cyclic(&zero, &params)];
        let neg_rows = [neg.as_slice()];
        let cyc_rows = [cyc.as_slice()];
        let coeffs = [[1i32; D]];
        let requests = [PairedCenteredI32Request {
            cyclic_rows: &cyc_rows,
            negacyclic_rows: &neg_rows,
            num_rows: 1,
            coeffs: &coeffs,
            bounds: CenteredRhsBounds { capacity: 0 },
        }];
        let result = fused_quotient_rhs_batch::<F, i32, Q32_NUM_PRIMES, D, 0, 0, 1>(
            &[],
            &[],
            &requests,
            &params,
        )
        .expect("observed centered bound augments capacity hint");
        assert_eq!(result.2[0], vec![CyclotomicRing::zero()]);
    }

    #[test]
    fn cyclic_fallback_sizes_lut_from_small_observed_bound() {
        type F = Prime128Offset275;
        const D: usize = 64;
        let ProtocolCrtNttParams::Q128(params) =
            select_crt_ntt_params::<F, D>().expect("Q128 test parameters")
        else {
            panic!("test field must use Q128 parameters");
        };
        let declared = 127;
        let chunk = safe_crt_chunk_width::<F, i32, Q128_NUM_PRIMES, D>(&params, 4096, declared)
            .expect("one declared-capacity term fits");
        let cols = chunk + 1;
        let zero = CyclotomicRing::<F, D>::zero();
        let entry = CyclotomicCrtNtt::from_ring_cyclic(&zero, &params);
        let row = vec![entry; cols];
        let rows = [row.as_slice()];
        let coeffs = vec![[1i8; D]; cols];
        let requests = [CyclicI8Request {
            cyclic_rows: &rows,
            num_rows: 1,
            coeffs: &coeffs,
            abs_bound: declared,
        }];
        let ([result], [], []) = fused_quotient_rhs_batch::<F, i32, Q128_NUM_PRIMES, D, 1, 0, 0>(
            &requests,
            &[],
            &[],
            &params,
        )
        .expect("fallback uses observed LUT bound");
        assert_eq!(result, vec![CyclotomicRing::zero()]);
    }
}
