use super::*;

#[allow(clippy::too_many_arguments)]
fn centered_quotient_rows_with_i16_tail_params<
    F: Field + CanonicalEncoding,
    const K: usize,
    const D: usize,
>(
    neg: &[CyclotomicCrtNtt<i32, K, D>],
    cyc: &[CyclotomicCrtNtt<i32, K, D>],
    tail_neg: &[CyclotomicCrtNtt<i16, 1, D>],
    tail_cyc: &[CyclotomicCrtNtt<i16, 1, D>],
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
    params: &CrtNttParamSet<i32, K, D>,
    tail_params: &CrtNttParamSet<i16, 1, D>,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    if num_rows == 0 {
        return Ok(Vec::new());
    }
    let width = z_folded_rings.len();
    let required = num_rows
        .checked_mul(width)
        .ok_or_else(|| AkitaError::InvalidSetup("quotient matrix shape overflows".into()))?;
    if width == 0
        || [neg.len(), cyc.len(), tail_neg.len(), tail_cyc.len()]
            .into_iter()
            .any(|length| length < required)
    {
        return Err(AkitaError::InvalidSetup(
            "base-plus-tail quotient cache is shorter than its matrix shape".into(),
        ));
    }
    let actual_bound = centered_rows_abs_bound(z_folded_rings, width);
    if actual_bound == 0 {
        return Ok(vec![CyclotomicRing::<F, D>::zero(); num_rows]);
    }
    let capacity_bound = u64::from(z_folded_max_abs).max(actual_bound);
    let capacity = params
        .crt_capacity()
        .with_prime_modulus(tail_params.primes[0].p as u128);
    let chunk_width = capacity
        .max_safe_width::<F, D>(capacity_bound)
        .map(|safe| safe.min(width))
        .filter(|&safe| safe > 0)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("centered quotient exceeds base plus i16-tail capacity".into())
        })?;
    let mixed_params = I16TailParams::new(params.clone(), tail_params.clone());
    let base_lut = (actual_bound <= u64::from(CENTERED_LUT_MAX_ABS))
        .then(|| CenteredMontLut::<i32, K>::new(params, actual_bound as i32));
    let num_chunks = width.div_ceil(chunk_width);
    let dot_batch = params.pointwise_dot_batch_size();

    Ok(cfg_fold_reduce!(
        0..num_chunks,
        || vec![CyclotomicRing::<F, D>::zero(); num_rows],
        |mut out: Vec<CyclotomicRing<F, D>>, chunk_idx| {
            let start = chunk_idx * chunk_width;
            let end = (start + chunk_width).min(width);
            let mut base_neg_accs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); num_rows];
            let mut base_cyc_accs = vec![CyclotomicCrtNtt::<i32, K, D>::zero(); num_rows];
            let mut tail_neg_accs = vec![CyclotomicCrtNtt::<i16, 1, D>::zero(); num_rows];
            let mut tail_cyc_accs = vec![CyclotomicCrtNtt::<i16, 1, D>::zero(); num_rows];
            let source: FusedMatrixSource<'_, F, i32, K, D> = FusedMatrixSource::Cached {
                negacyclic: neg,
                cyclic: cyc,
            };
            let mut base_scratch = PairedRunScratch::new(dot_batch);
            let mut tail_scratch = PairedRunScratch::new(dot_batch);
            accumulate_centered_pair_runs(
                &source,
                z_folded_rings,
                num_rows,
                start..end,
                base_lut.as_ref(),
                &mut base_neg_accs,
                &mut base_cyc_accs,
                &mut base_scratch,
                Some(&mut TailPairRuns {
                    neg: tail_neg,
                    cyc: tail_cyc,
                    neg_accs: &mut tail_neg_accs,
                    cyc_accs: &mut tail_cyc_accs,
                    scratch: &mut tail_scratch,
                    params: tail_params,
                }),
                params,
            );

            for row in 0..num_rows {
                let neg_ring = ntt_with_i16_tail_to_ring(
                    &base_neg_accs[row],
                    &tail_neg_accs[row],
                    &mixed_params,
                );
                let cyc_ring = cyclic_ntt_with_i16_tail_to_ring(
                    &base_cyc_accs[row],
                    &tail_cyc_accs[row],
                    &mixed_params,
                );
                out[row] += quotient_from_cyclic_and_negacyclic(&cyc_ring, &neg_ring);
            }
            out
        },
        |mut left: Vec<CyclotomicRing<F, D>>, right| {
            for (dst, src) in left.iter_mut().zip(right) {
                *dst += src;
            }
            left
        }
    ))
}

/// Centered A-quotient rows using the protocol CRT prefix plus its 14-bit tail.
pub fn centered_quotient_rows_with_i16_tail<F: Field + CanonicalEncoding, const D: usize>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    tail_slot: &PreparedNttCache<D>,
    num_rows: usize,
    z_folded_rings: &[[i32; D]],
    z_folded_max_abs: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
    let tail = tail_slot.i16_tail_pair().ok_or_else(|| {
        AkitaError::InvalidSetup("paired i16-tail NTT domain not prepared".into())
    })?;
    macro_rules! dispatch {
        ($neg_base:expr, $cyc_base:expr) => {{
            let (neg_base, cyc_base) = ($neg_base, $cyc_base);
            if neg_base.params() != cyc_base.params() {
                return Err(AkitaError::InvalidSetup(
                    "cyclic and negacyclic NTT profiles do not match".into(),
                ));
            }
            centered_quotient_rows_with_i16_tail_params(
                neg_base.negacyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("negacyclic NTT domain not prepared".into())
                })?,
                cyc_base.cyclic().ok_or_else(|| {
                    AkitaError::InvalidSetup("cyclic NTT domain not prepared".into())
                })?,
                tail.negacyclic(),
                tail.cyclic(),
                num_rows,
                z_folded_rings,
                z_folded_max_abs,
                neg_base.params(),
                tail.params(),
            )
        }};
    }
    if let (Some(neg), Some(cyc)) = (negacyclic_slot.q32_base(), cyclic_slot.q32_base()) {
        dispatch!(neg, cyc)
    } else if let (Some(neg), Some(cyc)) = (negacyclic_slot.q64_base(), cyclic_slot.q64_base()) {
        dispatch!(neg, cyc)
    } else if let (Some(neg), Some(cyc)) = (negacyclic_slot.q128_base(), cyclic_slot.q128_base()) {
        dispatch!(neg, cyc)
    } else {
        Err(AkitaError::InvalidSetup(
            "cyclic and negacyclic NTT profiles do not match".into(),
        ))
    }
}
