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
