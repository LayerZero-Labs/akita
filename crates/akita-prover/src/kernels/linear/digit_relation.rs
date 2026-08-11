use super::*;

/// Both transform-domain products contributed by the ring-switch D role.
pub(crate) struct DigitRelationRows<F: FieldCore, const D: usize> {
    pub(crate) negacyclic: Vec<CyclotomicRing<F, D>>,
    pub(crate) cyclic: Vec<CyclotomicRing<F, D>>,
}

pub(crate) fn digit_relation_matrix_extent(
    num_rows: usize,
    width: usize,
) -> Result<usize, AkitaError> {
    num_rows
        .checked_mul(width)
        .ok_or_else(|| AkitaError::InvalidSetup("D-role matrix extent overflow".into()))
}

/// Evaluate the cached D-role relation in both transform domains.
///
/// Keep the two specialized mat-vec kernels separate: each is faster on the
/// cached route than reconstructing both products through the general fused
/// B/A machinery.
pub(crate) fn digit_relation_rows_cached_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    negacyclic_slot: &PreparedNttCache<D>,
    cyclic_slot: &PreparedNttCache<D>,
    num_rows: usize,
    digits: &[[i8; D]],
    log_basis: u32,
) -> Result<DigitRelationRows<F, D>, AkitaError> {
    let negacyclic =
        mat_vec_mul_ntt_single_i8(negacyclic_slot, num_rows, digits.len(), digits, log_basis)?;
    let cyclic =
        mat_vec_mul_ntt_single_i8_cyclic(cyclic_slot, num_rows, digits.len(), digits, log_basis)?;
    Ok(DigitRelationRows { negacyclic, cyclic })
}

/// Stream the D-role matrix once and evaluate both transform-domain products.
pub(crate) fn digit_relation_rows_streamed_prover_bounds<
    F: FieldCore + CanonicalField + HalvingField,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    num_rows: usize,
    digits: &[[i8; D]],
    log_basis: u32,
) -> Result<DigitRelationRows<F, D>, AkitaError> {
    validate_i8_log_basis(log_basis)?;
    let width = digits.len();
    if num_rows != 0 && width == 0 {
        return Err(AkitaError::InvalidInput(
            "active D role must have a nonzero source width".into(),
        ));
    }
    let matrix_extent = digit_relation_matrix_extent(num_rows, width)?;
    if source.len() < matrix_extent {
        return Err(AkitaError::InvalidSetup(format!(
            "D-role matrix needs {matrix_extent} elements, got {}",
            source.len()
        )));
    }
    let digit_abs_bound = balanced_digit_abs_bound(log_basis);
    if !digit_rows_within_digit_bound::<D>(digits, width, digit_abs_bound) {
        return Err(AkitaError::InvalidInput(
            "D-role digits exceed the configured opening basis".into(),
        ));
    }

    macro_rules! run {
        ($params:expr) => {{
            digit_relation_rows_with_params(source, num_rows, digits, digit_abs_bound, &$params)
        }};
    }
    match select_crt_ntt_params::<F, D>()? {
        ProtocolCrtNttParams::Q32(params) => run!(params),
        ProtocolCrtNttParams::Q64(params) => run!(params),
        ProtocolCrtNttParams::Q128(params) => run!(params),
    }
}

fn digit_relation_rows_with_params<
    F: FieldCore + CanonicalField + HalvingField,
    W: PrimeWidth,
    const K: usize,
    const D: usize,
>(
    source: &[CyclotomicRing<F, D>],
    num_rows: usize,
    digits: &[[i8; D]],
    digit_abs_bound: u64,
    params: &CrtNttParamSet<W, K, D>,
) -> Result<DigitRelationRows<F, D>, AkitaError> {
    if num_rows == 0 {
        return Ok(DigitRelationRows {
            negacyclic: Vec::new(),
            cyclic: Vec::new(),
        });
    }
    let width = digits.len();
    let chunk_width = safe_crt_chunk_width::<F, W, K, D>(params, width, digit_abs_bound)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("CRT parameters cannot represent one D-role digit term".into())
        })?;
    let num_chunks = width.div_ceil(chunk_width);
    let lut = DigitMontLut::<W, K>::new_with_digit_bound(params, digit_abs_bound);

    let (negacyclic, cyclic) = cfg_fold_reduce!(
        0..num_chunks,
        || (
            vec![CyclotomicRing::<F, D>::zero(); num_rows],
            vec![CyclotomicRing::<F, D>::zero(); num_rows],
        ),
        |mut out: (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), chunk_idx| {
            let start = chunk_idx * chunk_width;
            let end = (start + chunk_width).min(width);
            let mut neg_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];
            let mut cyc_accs = vec![CyclotomicCrtNtt::<W, K, D>::zero(); num_rows];

            for (column, digit) in digits.iter().enumerate().take(end).skip(start) {
                if is_zero_plane(digit) {
                    continue;
                }
                let rhs_neg = CyclotomicCrtNtt::from_i8_with_lut(digit, params, &lut);
                let rhs_cyc = CyclotomicCrtNtt::from_i8_cyclic_with_lut(digit, params, &lut);
                for row in 0..num_rows {
                    let index = row * width + column;
                    let (matrix_neg, matrix_cyc) =
                        CyclotomicCrtNtt::from_ring_pair_with_params(&source[index], params);
                    accumulate_pointwise_product_into(
                        &mut neg_accs[row],
                        &matrix_neg,
                        &rhs_neg,
                        params,
                    );
                    accumulate_pointwise_product_into(
                        &mut cyc_accs[row],
                        &matrix_cyc,
                        &rhs_cyc,
                        params,
                    );
                }
            }

            for (dst, acc) in out.0.iter_mut().zip(neg_accs) {
                *dst += acc.to_ring(params);
            }
            for (dst, acc) in out.1.iter_mut().zip(cyc_accs) {
                *dst += acc.to_ring_cyclic(params);
            }
            out
        },
        |mut left: (Vec<CyclotomicRing<F, D>>, Vec<CyclotomicRing<F, D>>), right| {
            for (dst, src) in left.0.iter_mut().zip(right.0) {
                *dst += src;
            }
            for (dst, src) in left.1.iter_mut().zip(right.1) {
                *dst += src;
            }
            left
        }
    );
    Ok(DigitRelationRows { negacyclic, cyclic })
}
