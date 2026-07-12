use super::*;

struct SingleDomainState<F: FieldCore, W: PrimeWidth, const K: usize, const D: usize> {
    ntt: Vec<CyclotomicCrtNtt<W, K, D>>,
    out: Vec<CyclotomicRing<F, D>>,
    since_flush: usize,
    safe_width: usize,
}

struct PairedDomainState<F: FieldCore, W: PrimeWidth, const K: usize, const D: usize> {
    neg_ntt: Vec<CyclotomicCrtNtt<W, K, D>>,
    cyc_ntt: Vec<CyclotomicCrtNtt<W, K, D>>,
    neg_out: Vec<CyclotomicRing<F, D>>,
    quotient: Vec<CyclotomicRing<F, D>>,
    since_flush: usize,
    safe_width: usize,
}

fn flush_cyclic<F: FieldCore + CanonicalField, W: PrimeWidth, const K: usize, const D: usize>(
    state: &mut SingleDomainState<F, W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    for (dst, acc) in state.out.iter_mut().zip(state.ntt.iter_mut()) {
        *dst += acc.to_ring_cyclic(params);
        *acc = CyclotomicCrtNtt::zero();
    }
    state.since_flush = 0;
}

fn flush_negacyclic<
    F: FieldCore + CanonicalField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    state: &mut SingleDomainState<F, W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    for (dst, acc) in state.out.iter_mut().zip(state.ntt.iter_mut()) {
        *dst += acc.to_ring_with_params(params);
        *acc = CyclotomicCrtNtt::zero();
    }
    state.since_flush = 0;
}

fn flush_paired<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    state: &mut PairedDomainState<F, W, K, D>,
    params: &CrtNttParamSet<W, K, D>,
) {
    for row in 0..state.neg_ntt.len() {
        let neg = state.neg_ntt[row].to_ring_with_params(params);
        let cyc = state.cyc_ntt[row].to_ring_cyclic(params);
        state.neg_out[row] += neg;
        state.quotient[row] += quotient_from_cyclic_and_negacyclic(&cyc, &neg);
        state.neg_ntt[row] = CyclotomicCrtNtt::zero();
        state.cyc_ntt[row] = CyclotomicCrtNtt::zero();
    }
    state.since_flush = 0;
}

#[allow(clippy::type_complexity)]
pub(super) fn accumulate_i8_requests_streaming<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    cyclic_requests: &[CyclicI8Request<'_, W, K, D>],
    negacyclic_requests: &[NegacyclicI8Request<'_, W, K, D>],
    paired_requests: &[PairedI8Request<'_, W, K, D>],
    params: &CrtNttParamSet<W, K, D>,
) -> Result<
    (
        Vec<Vec<CyclotomicRing<F, D>>>,
        Vec<Vec<CyclotomicRing<F, D>>>,
        Vec<PairedI8Rows<F, D>>,
    ),
    AkitaError,
> {
    let zero = CyclotomicCrtNtt::<W, K, D>::zero();
    let mut cyclic = cyclic_requests
        .iter()
        .map(|request| {
            let safe_width =
                safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("cyclic i8 term exceeds CRT support".into())
                    })?;
            Ok(SingleDomainState {
                ntt: vec![zero.clone(); request.num_rows],
                out: vec![CyclotomicRing::zero(); request.num_rows],
                since_flush: 0,
                safe_width,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut negacyclic = negacyclic_requests
        .iter()
        .map(|request| {
            let safe_width =
                safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("negacyclic i8 term exceeds CRT support".into())
                    })?;
            Ok(SingleDomainState {
                ntt: vec![zero.clone(); request.num_rows],
                out: vec![CyclotomicRing::zero(); request.num_rows],
                since_flush: 0,
                safe_width,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut paired = paired_requests
        .iter()
        .map(|request| {
            let safe_width =
                safe_crt_chunk_width::<F, W, K, D>(params, request.coeffs.len(), request.abs_bound)
                    .ok_or_else(|| {
                        AkitaError::InvalidInput("paired i8 term exceeds CRT support".into())
                    })?;
            Ok(PairedDomainState {
                neg_ntt: vec![zero.clone(); request.num_rows],
                cyc_ntt: vec![zero.clone(); request.num_rows],
                neg_out: vec![CyclotomicRing::zero(); request.num_rows],
                quotient: vec![CyclotomicRing::zero(); request.num_rows],
                since_flush: 0,
                safe_width,
            })
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let cyclic_luts = cyclic_requests
        .iter()
        .map(|request| {
            (!request.centered_digits).then(|| {
                let observed = request
                    .coeffs
                    .iter()
                    .flat_map(|row| row.iter())
                    .map(|coeff| u64::from(coeff.unsigned_abs()))
                    .max()
                    .unwrap_or(0)
                    .max(1)
                    .next_power_of_two();
                DigitMontLut::<W, K>::new_with_digit_bound(params, observed)
            })
        })
        .collect::<Vec<_>>();
    let paired_luts = paired_requests
        .iter()
        .map(|request| {
            let observed = request
                .coeffs
                .iter()
                .flat_map(|row| row.iter())
                .map(|coeff| coeff.unsigned_abs())
                .max()
                .unwrap_or(0);
            CenteredMontLut::<W, K>::new(params, i32::from(observed))
        })
        .collect::<Vec<_>>();
    let max_columns = cyclic_requests
        .iter()
        .map(|request| request.coeffs.len())
        .chain(
            negacyclic_requests
                .iter()
                .map(|request| request.coeffs.len()),
        )
        .chain(paired_requests.iter().map(|request| request.coeffs.len()))
        .max()
        .unwrap_or(0);

    for j in 0..max_columns {
        for (index, request) in cyclic_requests.iter().enumerate() {
            if j >= request.coeffs.len() {
                continue;
            }
            if !is_zero_plane(&request.coeffs[j]) {
                let rhs = if request.centered_digits {
                    CyclotomicCrtNtt::from_i8_cyclic(&request.coeffs[j], params)
                } else {
                    CyclotomicCrtNtt::from_i8_cyclic_with_lut(
                        &request.coeffs[j],
                        params,
                        cyclic_luts[index]
                            .as_ref()
                            .expect("balanced cyclic LUT exists"),
                    )
                };
                for (acc, row) in cyclic[index].ntt.iter_mut().zip(request.cyclic_rows) {
                    accumulate_pointwise_product_into(acc, &row[j], &rhs, params);
                }
            }
            cyclic[index].since_flush += 1;
            if cyclic[index].since_flush == cyclic[index].safe_width {
                flush_cyclic(&mut cyclic[index], params);
            }
        }
        for (index, request) in negacyclic_requests.iter().enumerate() {
            if j >= request.coeffs.len() {
                continue;
            }
            if !is_zero_plane(&request.coeffs[j]) {
                let rhs = CyclotomicCrtNtt::from_i8_with_params(&request.coeffs[j], params);
                for (acc, row) in negacyclic[index]
                    .ntt
                    .iter_mut()
                    .zip(request.negacyclic_rows)
                {
                    accumulate_pointwise_product_into(acc, &row[j], &rhs, params);
                }
            }
            negacyclic[index].since_flush += 1;
            if negacyclic[index].since_flush == negacyclic[index].safe_width {
                flush_negacyclic(&mut negacyclic[index], params);
            }
        }
        for (index, request) in paired_requests.iter().enumerate() {
            if j >= request.coeffs.len() {
                continue;
            }
            if !is_zero_plane(&request.coeffs[j]) {
                let centered = from_fn(|k| i32::from(request.coeffs[j][k]));
                let (neg_rhs, cyc_rhs) = unsafe {
                    CyclotomicCrtNtt::from_centered_i32_pair_with_lut_unchecked(
                        &centered,
                        params,
                        &paired_luts[index],
                    )
                };
                for row in 0..request.num_rows {
                    accumulate_pointwise_product_into(
                        &mut paired[index].neg_ntt[row],
                        &request.negacyclic_rows[row][j],
                        &neg_rhs,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        &mut paired[index].cyc_ntt[row],
                        &request.cyclic_rows[row][j],
                        &cyc_rhs,
                        params,
                    );
                }
            }
            paired[index].since_flush += 1;
            if paired[index].since_flush == paired[index].safe_width {
                flush_paired(&mut paired[index], params);
            }
        }
    }
    for state in &mut cyclic {
        if state.since_flush != 0 {
            flush_cyclic(state, params);
        }
    }
    for state in &mut negacyclic {
        if state.since_flush != 0 {
            flush_negacyclic(state, params);
        }
    }
    for state in &mut paired {
        if state.since_flush != 0 {
            flush_paired(state, params);
        }
    }
    Ok((
        cyclic.into_iter().map(|state| state.out).collect(),
        negacyclic.into_iter().map(|state| state.out).collect(),
        paired
            .into_iter()
            .map(|state| PairedI8Rows {
                negacyclic: state.neg_out,
                quotient: state.quotient,
            })
            .collect(),
    ))
}

pub(super) fn validate_i8_rows_and_bound<const D: usize>(
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
    if abs > declared {
        return Err(AkitaError::InvalidInput(
            "fused relation row digits exceed their declared absolute bound".into(),
        ));
    }
    Ok(ObservedI8Bounds { abs, lut })
}

pub(super) fn centered_rows_abs_bound<const D: usize>(rows: &[[i32; D]], len: usize) -> u64 {
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
pub(super) fn accumulate_centered_paired_rows<
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
) -> (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>) {
    if num_rows == 0 {
        return (vec![], vec![]);
    }
    if z_len == 0 {
        let zero = vec![CyclotomicRing::<F, D>::zero(); num_rows];
        return (zero.clone(), zero);
    }

    if actual_abs_bound == 0 {
        let zero = vec![CyclotomicRing::<F, D>::zero(); num_rows];
        return (zero.clone(), zero);
    }

    let Some(chunk_width) = safe_crt_chunk_width::<F, W, K, D>(params, z_len, z_bounds.capacity)
    else {
        return accumulate_centered_paired_rows_field(
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
        || (
            vec![CyclotomicRing::<F, D>::zero(); num_rows],
            vec![CyclotomicRing::<F, D>::zero(); num_rows],
        ),
        |mut out: (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), chunk_idx| {
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

            for (row, (neg_acc, cyc_acc)) in neg_accs.into_iter().zip(cyc_accs).enumerate() {
                let neg_ring: CyclotomicRing<F, D> = neg_acc.to_ring_with_params(params);
                let cyc_ring: CyclotomicRing<F, D> = cyc_acc.to_ring_cyclic(params);
                out.0[row] += neg_ring;
                out.1[row] += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut a: (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), b| {
            for (dst, src) in a.0.iter_mut().zip(b.0) {
                *dst += src;
            }
            for (dst, src) in a.1.iter_mut().zip(b.1) {
                *dst += src;
            }
            a
        }
    )
}

fn accumulate_centered_paired_rows_field<
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
) -> (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>) {
    let rows = cfg_into_iter!(0..num_rows)
        .map(|row_idx| {
            let mut neg_out = CyclotomicRing::<F, D>::zero();
            let mut quotient = CyclotomicRing::<F, D>::zero();
            for j in 0..z_len {
                if is_zero_centered_row(&z_folded_rings[j]) {
                    continue;
                }
                let z = centered_i32_ring::<F, D>(&z_folded_rings[j]);
                let neg_lhs: CyclotomicRing<F, D> =
                    neg_rows[row_idx][j].to_ring_with_params(params);
                let cyc_lhs: CyclotomicRing<F, D> = cyc_rows[row_idx][j].to_ring_cyclic(params);
                let neg_product = neg_lhs * z;
                neg_out += neg_product;
                let mut cyc_product = CyclotomicRing::<F, D>::zero();
                add_cyclic_product_into(&mut cyc_product, &cyc_lhs, &z);
                quotient += quotient_from_cyclic_and_negacyclic(&cyc_product, &neg_product);
            }
            (neg_out, quotient)
        })
        .collect::<Vec<_>>();
    rows.into_iter().unzip()
}
