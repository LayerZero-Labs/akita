use super::*;
use crate::compute::{CompressionRowsMode, CompressionRowsOutput, CompressionRowsPlan};
use std::mem::size_of;

mod dispatch;
mod fallback;

use fallback::*;

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
    centered_digits: bool,
}

struct NegacyclicI8Request<'a, W: PrimeWidth, const K: usize, const D: usize> {
    negacyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    coeffs: &'a [[i8; D]],
    abs_bound: u64,
}

struct PairedI8Request<'a, W: PrimeWidth, const K: usize, const D: usize> {
    cyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    negacyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    coeffs: &'a [[i8; D]],
    abs_bound: u64,
}

struct PairedI8Rows<F: FieldCore, const D: usize> {
    negacyclic: Vec<CyclotomicRing<F, D>>,
    quotient: Vec<CyclotomicRing<F, D>>,
}

struct PairedCenteredI32Request<'a, W: PrimeWidth, const K: usize, const D: usize> {
    cyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    negacyclic_rows: &'a [&'a [CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    coeffs: &'a [[i32; D]],
    bounds: CenteredRhsBounds,
}

/// Fused column-tiled kernel for the three ring-switch relation-row products.
///
/// Replaces three separate NTT-cached mat-vec calls (D-cyclic, B-cyclic,
/// A-quotient) with a single pass over the shared NTT cache. Within each
/// column tile, cache entries are loaded once and reused across all three
/// products with their exact row bounds, eliminating redundant DRAM reads.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn fused_ring_switch_relation_rows_with_params<
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
            centered_digits: false,
        },
        CyclicI8Request {
            cyclic_rows: b_cyc_rows,
            num_rows: n_b,
            coeffs: t_hat,
            abs_bound: t_digit_abs_bound,
            centered_digits: false,
        },
    ];
    let centered_requests = [PairedCenteredI32Request {
        cyclic_rows: a_cyc_rows,
        negacyclic_rows: neg_rows,
        num_rows: n_a,
        coeffs: z_folded_rings,
        bounds: z_bounds,
    }];
    let (mut cyclic, negacyclic, paired, mut centered) = fused_relation_rows_batch::<F, W, K, D>(
        &cyclic_requests,
        &[],
        &[],
        &centered_requests,
        params,
    )?;
    if !negacyclic.is_empty() || !paired.is_empty() || cyclic.len() != 2 || centered.len() != 1 {
        return Err(AkitaError::InvalidInput(
            "fused relation row result shape mismatch".into(),
        ));
    }
    let a_result = centered
        .pop()
        .ok_or_else(|| AkitaError::InvalidInput("fused A relation result is absent".into()))?;
    let b_result = cyclic
        .pop()
        .ok_or_else(|| AkitaError::InvalidInput("fused B relation result is absent".into()))?;
    let d_result = cyclic
        .pop()
        .ok_or_else(|| AkitaError::InvalidInput("fused D relation result is absent".into()))?;
    Ok((d_result, b_result, a_result))
}

struct DynamicFusedAcc<W: PrimeWidth, const K: usize, const D: usize> {
    cyclic: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>,
    negacyclic_i8: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>,
    paired_i8_neg: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>,
    paired_i8_cyc: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>,
    centered_neg: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>,
    centered_cyc: Vec<Vec<CyclotomicCrtNtt<W, K, D>>>,
}

fn validate_rows<W: PrimeWidth, const K: usize, const D: usize>(
    rows: &[&[CyclotomicCrtNtt<W, K, D>]],
    num_rows: usize,
    width: usize,
    label: &str,
) -> Result<(), AkitaError> {
    if rows.len() != num_rows || rows.iter().any(|row| row.len() < width) {
        return Err(AkitaError::InvalidInput(format!(
            "fused relation row {label} matrix shape does not match request"
        )));
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn fused_relation_rows_batch<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    cyclic_requests: &[CyclicI8Request<'_, W, K, D>],
    negacyclic_i8_requests: &[NegacyclicI8Request<'_, W, K, D>],
    paired_i8_requests: &[PairedI8Request<'_, W, K, D>],
    centered_requests: &[PairedCenteredI32Request<'_, W, K, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        Vec<Vec<CyclotomicRing<F, D>>>,
        Vec<Vec<CyclotomicRing<F, D>>>,
        Vec<PairedI8Rows<F, D>>,
        Vec<Vec<CyclotomicRing<F, D>>>,
    ),
    AkitaError,
> {
    let mut cyclic_observed = vec![ObservedI8Bounds::default(); cyclic_requests.len()];
    for (request, observed) in cyclic_requests.iter().zip(cyclic_observed.iter_mut()) {
        validate_rows(
            request.cyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "cyclic",
        )?;
        *observed = validate_i8_rows_and_bound(request.coeffs, request.abs_bound)?;
        if !request.centered_digits && observed.lut > request.abs_bound {
            return Err(AkitaError::InvalidInput(
                "balanced relation row digits exceed the positive endpoint".into(),
            ));
        }
    }
    for request in negacyclic_i8_requests {
        validate_rows(
            request.negacyclic_rows,
            request.num_rows,
            request.coeffs.len(),
            "negacyclic",
        )?;
        validate_i8_rows_and_bound(request.coeffs, request.abs_bound)?;
    }
    let mut paired_i8_observed = vec![ObservedI8Bounds::default(); paired_i8_requests.len()];
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
    let mut centered_observed = vec![0u64; centered_requests.len()];
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
            negacyclic_i8_requests
                .iter()
                .map(|request| request.coeffs.len()),
        )
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
            cyclic_requests
                .iter()
                .map(|request| vec![CyclotomicRing::zero(); request.num_rows])
                .collect(),
            negacyclic_i8_requests
                .iter()
                .map(|request| vec![CyclotomicRing::zero(); request.num_rows])
                .collect(),
            paired_i8_requests
                .iter()
                .map(|request| PairedI8Rows {
                    negacyclic: vec![CyclotomicRing::zero(); request.num_rows],
                    quotient: vec![CyclotomicRing::zero(); request.num_rows],
                })
                .collect(),
            centered_requests
                .iter()
                .map(|request| vec![CyclotomicRing::zero(); request.num_rows])
                .collect(),
        ));
    }
    let i8_one_shot = cyclic_requests.iter().all(|request| {
        safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
            == Some(request.coeffs.len())
    }) && negacyclic_i8_requests.iter().all(|request| {
        safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
            == Some(request.coeffs.len())
    }) && paired_i8_requests.iter().all(|request| {
        safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
            == Some(request.coeffs.len())
    });
    let centered_one_shot = centered_requests
        .iter()
        .zip(centered_observed.iter().copied())
        .all(|(request, observed)| {
            safe_crt_chunk_width::<F, W, K, D>(
                params,
                request.coeffs.len(),
                request.bounds.capacity.max(observed),
            ) == Some(request.coeffs.len())
        });
    if !i8_one_shot || !centered_one_shot {
        let (cyclic, negacyclic_i8, paired_i8) = if i8_one_shot {
            let (cyclic, negacyclic, paired, centered) = fused_relation_rows_batch(
                cyclic_requests,
                negacyclic_i8_requests,
                paired_i8_requests,
                &[],
                params,
            )?;
            debug_assert!(centered.is_empty());
            (cyclic, negacyclic, paired)
        } else {
            accumulate_i8_requests_streaming(
                cyclic_requests,
                negacyclic_i8_requests,
                paired_i8_requests,
                params,
            )?
        };
        let centered = if centered_one_shot {
            let (cyclic, negacyclic, paired, centered) =
                fused_relation_rows_batch(&[], &[], &[], centered_requests, params)?;
            debug_assert!(cyclic.is_empty() && negacyclic.is_empty() && paired.is_empty());
            centered
        } else {
            (0..centered_requests.len())
                .map(|i| {
                    let request = &centered_requests[i];
                    accumulate_centered_paired_rows(
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
                    .1
                })
                .collect()
        };
        return Ok((cyclic, negacyclic_i8, paired_i8, centered));
    }

    let digit_lut_bound = cyclic_observed
        .iter()
        .zip(cyclic_requests)
        .filter_map(|(observed, request)| (!request.centered_digits).then_some(observed.lut))
        .max()
        .unwrap_or(1);
    let digit_lut = cyclic_requests
        .iter()
        .any(|request| !request.centered_digits && !request.coeffs.is_empty())
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
    let centered_luts: Vec<Option<CenteredMontLut<W, K>>> = (0..centered_requests.len())
        .map(|i| {
            (!centered_requests[i].coeffs.is_empty()
                && centered_observed[i] <= u64::from(CENTERED_LUT_MAX_ABS))
            .then(|| CenteredMontLut::<W, K>::new(params, centered_observed[i] as i32))
        })
        .collect();
    let base_tw = (FUSED_L2_CACHE_BYTES / (K * D * size_of::<W>())).max(1);
    let tw = base_tw.min(max_col.div_ceil(MIN_FUSED_TILES).max(1));
    let num_tiles = max_col.div_ceil(tw);
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();

    let accumulators = cfg_fold_reduce!(
        0..num_tiles,
        || DynamicFusedAcc {
            cyclic: cyclic_requests
                .iter()
                .map(|request| vec![zero.clone(); request.num_rows])
                .collect(),
            negacyclic_i8: negacyclic_i8_requests
                .iter()
                .map(|request| vec![zero.clone(); request.num_rows])
                .collect(),
            paired_i8_neg: paired_i8_requests
                .iter()
                .map(|request| vec![zero.clone(); request.num_rows])
                .collect(),
            paired_i8_cyc: paired_i8_requests
                .iter()
                .map(|request| vec![zero.clone(); request.num_rows])
                .collect(),
            centered_neg: centered_requests
                .iter()
                .map(|request| vec![zero.clone(); request.num_rows])
                .collect(),
            centered_cyc: centered_requests
                .iter()
                .map(|request| vec![zero.clone(); request.num_rows])
                .collect(),
        },
        |mut accs: DynamicFusedAcc<W, K, D>, tile_idx| {
            let tile_start = tile_idx * tw;
            let tile_end = (tile_start + tw).min(max_col);
            for j in tile_start..tile_end {
                for (request, lane_acc) in cyclic_requests.iter().zip(accs.cyclic.iter_mut()) {
                    if j < request.coeffs.len() && !is_zero_plane(&request.coeffs[j]) {
                        let ntt = if request.centered_digits {
                            CyclotomicCrtNtt::from_i8_cyclic(&request.coeffs[j], params)
                        } else {
                            CyclotomicCrtNtt::from_i8_cyclic_with_lut(
                                &request.coeffs[j],
                                params,
                                digit_lut.as_ref().expect("digit LUT exists"),
                            )
                        };
                        for (row_acc, row) in lane_acc.iter_mut().zip(request.cyclic_rows) {
                            accumulate_pointwise_product_into(row_acc, &row[j], &ntt, params);
                        }
                    }
                }
                for (request, neg_accs) in negacyclic_i8_requests
                    .iter()
                    .zip(accs.negacyclic_i8.iter_mut())
                {
                    if j < request.coeffs.len() && !is_zero_plane(&request.coeffs[j]) {
                        let neg_ntt =
                            CyclotomicCrtNtt::from_i8_with_params(&request.coeffs[j], params);
                        for (row_acc, row) in neg_accs.iter_mut().zip(request.negacyclic_rows) {
                            accumulate_pointwise_product_into(row_acc, &row[j], &neg_ntt, params);
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
                for lane in 0..centered_requests.len() {
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
        |mut a: DynamicFusedAcc<W, K, D>, b| {
            for lane in 0..cyclic_requests.len() {
                for (dst, src) in a.cyclic[lane].iter_mut().zip(b.cyclic[lane].iter()) {
                    add_ntt_into(dst, src, params);
                }
            }
            for lane in 0..negacyclic_i8_requests.len() {
                for (dst, src) in a.negacyclic_i8[lane]
                    .iter_mut()
                    .zip(b.negacyclic_i8[lane].iter())
                {
                    add_ntt_into(dst, src, params);
                }
            }
            for lane in 0..paired_i8_requests.len() {
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
            for lane in 0..centered_requests.len() {
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
    let cyclic = accumulators
        .cyclic
        .into_iter()
        .map(|rows| {
            rows.into_iter()
                .map(|value| value.to_ring_cyclic(params))
                .collect()
        })
        .collect();
    let negacyclic_i8 = accumulators
        .negacyclic_i8
        .into_iter()
        .map(|rows| {
            rows.into_iter()
                .map(|value| value.to_ring_with_params(params))
                .collect()
        })
        .collect();
    let paired_i8 = accumulators
        .paired_i8_neg
        .into_iter()
        .zip(accumulators.paired_i8_cyc)
        .map(|(neg_rows, cyc_rows)| {
            let negacyclic = neg_rows
                .iter()
                .map(|neg| neg.to_ring_with_params(params))
                .collect::<Vec<_>>();
            let quotient = negacyclic
                .iter()
                .zip(cyc_rows.iter())
                .map(|(neg, cyc)| {
                    quotient_from_cyclic_and_negacyclic(&cyc.to_ring_cyclic(params), neg)
                })
                .collect();
            PairedI8Rows {
                negacyclic,
                quotient,
            }
        })
        .collect();
    let centered = accumulators
        .centered_neg
        .into_iter()
        .zip(accumulators.centered_cyc)
        .map(|(neg_rows, cyc_rows)| {
            neg_rows
                .iter()
                .zip(cyc_rows.iter())
                .map(|(neg, cyc)| {
                    quotient_from_cyclic_and_negacyclic(
                        &cyc.to_ring_cyclic(params),
                        &neg.to_ring_with_params(params),
                    )
                })
                .collect()
        })
        .collect();
    Ok((cyclic, negacyclic_i8, paired_i8, centered))
}

/// Fused ring-switch relation-row kernel dispatching over [`NttSlotCache`] variants.
///
/// Computes three NTT-cached mat-vec products in a single tiled pass:
/// - D-cyclic: `cyc[0..n_d] · e_hat` (cyclic domain)
/// - B-cyclic: `cyc[0..n_b] · t_hat` (cyclic domain)
/// - A-quotient: `(cyc[0..n_a]·z_cyc − neg[0..n_a]·z_neg) / 2`
///
/// All roles share the same underlying coefficient matrix, but each role uses
/// its own packed row width.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[tracing::instrument(skip_all, name = "fused_ring_switch_relation_rows")]
#[cfg(test)]
pub(crate) fn fused_ring_switch_relation_rows<
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
    dispatch::fused_ring_switch_relation_rows_with_digit_bound(
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
pub(crate) fn fused_ring_switch_relation_rows_prover_bounds<
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
    dispatch::fused_ring_switch_relation_rows_with_digit_bound(
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

/// Exact-shape compression batch over one checked prepared NTT prefix.
pub(crate) fn compression_rows_with_slot<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    slot: &NttSlotCache<D>,
    plan: CompressionRowsPlan<'_, F, D>,
) -> Result<Vec<CompressionRowsOutput<F, D>>, AkitaError> {
    dispatch::compression_rows(slot, plan)
}

#[cfg(test)]
mod internal_tests;
