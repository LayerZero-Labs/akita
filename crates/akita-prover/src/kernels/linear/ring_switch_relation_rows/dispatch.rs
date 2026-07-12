use super::*;

fn checked_packed_rows<'a, T>(
    flat: &'a [T],
    num_rows: usize,
    width: usize,
    label: &str,
) -> Result<Vec<&'a [T]>, AkitaError> {
    let required = num_rows.checked_mul(width).ok_or_else(|| {
        AkitaError::InvalidInput(format!("fused relation row {label} matrix shape overflow"))
    })?;
    if flat.len() < required {
        return Err(AkitaError::InvalidInput(format!(
            "fused relation row {label} cache is too short: actual={} required={required}",
            flat.len()
        )));
    }
    (0..num_rows)
        .map(|row| {
            let start = row.checked_mul(width).ok_or_else(|| {
                AkitaError::InvalidInput(format!("fused relation row {label} row offset overflow"))
            })?;
            let end = start.checked_add(width).ok_or_else(|| {
                AkitaError::InvalidInput(format!("fused relation row {label} row end overflow"))
            })?;
            flat.get(start..end).ok_or_else(|| {
                AkitaError::InvalidInput(format!(
                    "fused relation row {label} cache row is out of bounds"
                ))
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn fused_ring_switch_relation_rows_with_digit_bound<
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
    let d_width = e_hat.len();
    let b_width = t_hat.len();
    let a_width = z_folded_rings.len();
    macro_rules! execute {
        ($neg:expr, $cyc:expr, $params:expr) => {{
            let neg_rows = checked_packed_rows($neg, n_a, a_width, "A negacyclic")?;
            let d_rows = checked_packed_rows($cyc, n_d, d_width, "D cyclic")?;
            let b_rows = checked_packed_rows($cyc, n_b, b_width, "B cyclic")?;
            let a_rows = checked_packed_rows($cyc, n_a, a_width, "A cyclic")?;
            fused_ring_switch_relation_rows_with_params(
                &d_rows,
                &b_rows,
                &a_rows,
                &neg_rows,
                n_d,
                n_b,
                n_a,
                e_hat,
                t_hat,
                z_folded_rings,
                z_folded_max_abs,
                w_digit_abs_bound,
                t_digit_abs_bound,
                $params,
            )
        }};
    }
    match slot {
        NttSlotCache::Q32 { neg, cyc, params } => execute!(neg, cyc, params),
        NttSlotCache::Q64 { neg, cyc, params } => execute!(neg, cyc, params),
        NttSlotCache::Q128 { neg, cyc, params } => execute!(neg, cyc, params),
    }
}

pub(super) fn compression_rows<F: FieldCore + CanonicalField + HalvingField, const D: usize>(
    slot: &NttSlotCache<D>,
    plan: CompressionRowsPlan<'_, F, D>,
) -> Result<Vec<CompressionRowsOutput<F, D>>, AkitaError> {
    if plan.row_count == 0 || plan.column_count == 0 || plan.items.is_empty() {
        return Err(AkitaError::InvalidInput(
            "compression rows require nonzero rows, columns, and items".into(),
        ));
    }
    if plan
        .items
        .iter()
        .any(|item| item.digits.len() != plan.column_count || item.digit_abs_bound == 0)
    {
        return Err(AkitaError::InvalidInput(
            "compression rows item shape or digit bound is invalid".into(),
        ));
    }

    // Bound the per-tile accumulator footprint independently of the number of
    // right-hand sides offered by the caller. Overflow batches are split in
    // input order and rescan the same prepared matrix prefix; ordinary planner
    // batches fit in one pass, while an adversarially large batch cannot create
    // an unbounded `items × rows` allocation inside each Rayon tile.
    let stored_elements = match slot {
        NttSlotCache::Q32 { neg, cyc, .. } => neg.len().checked_add(cyc.len()),
        NttSlotCache::Q64 { neg, cyc, .. } => neg.len().checked_add(cyc.len()),
        NttSlotCache::Q128 { neg, cyc, .. } => neg.len().checked_add(cyc.len()),
    }
    .ok_or_else(|| AkitaError::InvalidInput("compression cache element count overflow".into()))?;
    let element_bytes = slot
        .cache_bytes()
        .checked_div(stored_elements)
        .ok_or_else(|| AkitaError::InvalidInput("compression cache has no NTT elements".into()))?;
    let lane_bytes = plan
        .row_count
        .checked_mul(element_bytes)
        .ok_or_else(|| AkitaError::InvalidInput("compression accumulator size overflow".into()))?;
    let lanes_per_item = |item: &crate::compute::CompressionRowsItem<'_, F, D>| match item.mode {
        CompressionRowsMode::EagerPaired => 2usize,
        CompressionRowsMode::NegacyclicOnly | CompressionRowsMode::CyclicWithKnownNeg(_) => 1,
    };
    let max_lanes = FUSED_L2_CACHE_BYTES.checked_div(lane_bytes).unwrap_or(0);
    if max_lanes == 0
        || plan
            .items
            .iter()
            .any(|item| lanes_per_item(item) > max_lanes)
    {
        return Err(AkitaError::InvalidInput(
            "one compression item exceeds the accumulator memory budget".into(),
        ));
    }
    let mut end = 0usize;
    let mut lanes = 0usize;
    while let Some(item) = plan.items.get(end) {
        let next = lanes + lanes_per_item(item);
        if next > max_lanes {
            break;
        }
        lanes = next;
        end += 1;
    }
    if end < plan.items.len() {
        let mut output = Vec::with_capacity(plan.items.len());
        let mut start = 0usize;
        while start < plan.items.len() {
            let mut stop = start;
            let mut chunk_lanes = 0usize;
            while let Some(item) = plan.items.get(stop) {
                let next = chunk_lanes + lanes_per_item(item);
                if next > max_lanes {
                    break;
                }
                chunk_lanes = next;
                stop += 1;
            }
            output.extend(compression_rows(
                slot,
                CompressionRowsPlan {
                    row_count: plan.row_count,
                    column_count: plan.column_count,
                    items: &plan.items[start..stop],
                },
            )?);
            start = stop;
        }
        return Ok(output);
    }
    let needs_negacyclic = plan.items.iter().any(|item| {
        matches!(
            item.mode,
            CompressionRowsMode::NegacyclicOnly | CompressionRowsMode::EagerPaired
        )
    });
    let needs_cyclic = plan.items.iter().any(|item| {
        matches!(
            item.mode,
            CompressionRowsMode::EagerPaired | CompressionRowsMode::CyclicWithKnownNeg(_)
        )
    });
    macro_rules! execute {
        ($neg:expr, $cyc:expr, $params:expr) => {{
            let neg_rows = if needs_negacyclic {
                checked_packed_rows(
                    $neg,
                    plan.row_count,
                    plan.column_count,
                    "compression negacyclic",
                )?
            } else {
                Vec::new()
            };
            let cyc_rows = if needs_cyclic {
                checked_packed_rows(
                    $cyc,
                    plan.row_count,
                    plan.column_count,
                    "compression cyclic",
                )?
            } else {
                Vec::new()
            };
            let mut cyclic_requests = Vec::new();
            let mut negacyclic_requests = Vec::new();
            let mut paired_requests = Vec::new();
            for item in plan.items {
                match item.mode {
                    CompressionRowsMode::NegacyclicOnly => {
                        negacyclic_requests.push(NegacyclicI8Request {
                            negacyclic_rows: &neg_rows,
                            num_rows: plan.row_count,
                            coeffs: item.digits,
                            abs_bound: item.digit_abs_bound,
                        });
                    }
                    CompressionRowsMode::EagerPaired => {
                        paired_requests.push(PairedI8Request {
                            cyclic_rows: &cyc_rows,
                            negacyclic_rows: &neg_rows,
                            num_rows: plan.row_count,
                            coeffs: item.digits,
                            abs_bound: item.digit_abs_bound,
                        });
                    }
                    CompressionRowsMode::CyclicWithKnownNeg(known) => {
                        if known.len() != plan.row_count {
                            return Err(AkitaError::InvalidInput(
                                "known compression negacyclic image has wrong row count".into(),
                            ));
                        }
                        cyclic_requests.push(CyclicI8Request {
                            cyclic_rows: &cyc_rows,
                            num_rows: plan.row_count,
                            coeffs: item.digits,
                            abs_bound: item.digit_abs_bound,
                            centered_digits: true,
                        });
                    }
                }
            }
            let (cyclic, negacyclic, paired, centered) = fused_relation_rows_batch::<F, _, _, D>(
                &cyclic_requests,
                &negacyclic_requests,
                &paired_requests,
                &[],
                $params,
            )?;
            if !centered.is_empty() {
                return Err(AkitaError::InvalidInput(
                    "compression batch produced centered relation rows".into(),
                ));
            }
            let mut cyclic = cyclic.into_iter();
            let mut negacyclic = negacyclic.into_iter();
            let mut paired = paired.into_iter();
            let mut output = Vec::with_capacity(plan.items.len());
            for item in plan.items {
                match item.mode {
                    CompressionRowsMode::NegacyclicOnly => {
                        output.push(CompressionRowsOutput {
                            u_neg: Some(negacyclic.next().ok_or_else(|| {
                                AkitaError::InvalidInput(
                                    "compression negacyclic output is absent".into(),
                                )
                            })?),
                            quotient: None,
                        });
                    }
                    CompressionRowsMode::EagerPaired => {
                        let rows = paired.next().ok_or_else(|| {
                            AkitaError::InvalidInput("compression paired output is absent".into())
                        })?;
                        output.push(CompressionRowsOutput {
                            u_neg: Some(rows.negacyclic),
                            quotient: Some(rows.quotient),
                        });
                    }
                    CompressionRowsMode::CyclicWithKnownNeg(known) => {
                        let cyclic_rows = cyclic.next().ok_or_else(|| {
                            AkitaError::InvalidInput("compression cyclic output is absent".into())
                        })?;
                        let quotient = cyclic_rows
                            .iter()
                            .zip(known)
                            .map(|(cyc, neg)| quotient_from_cyclic_and_negacyclic(cyc, neg))
                            .collect();
                        output.push(CompressionRowsOutput {
                            u_neg: None,
                            quotient: Some(quotient),
                        });
                    }
                }
            }
            if cyclic.next().is_some() || negacyclic.next().is_some() || paired.next().is_some() {
                return Err(AkitaError::InvalidInput(
                    "compression batch output count mismatch".into(),
                ));
            }
            Ok(output)
        }};
    }
    match slot {
        NttSlotCache::Q32 { neg, cyc, params } => execute!(neg, cyc, params),
        NttSlotCache::Q64 { neg, cyc, params } => execute!(neg, cyc, params),
        NttSlotCache::Q128 { neg, cyc, params } => execute!(neg, cyc, params),
    }
}
