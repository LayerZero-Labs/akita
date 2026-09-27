use super::*;
use akita_types::SubfieldMultiplierOpeningPoint;

pub(super) fn fold_onehot_block<F, I, const D: usize>(
    poly: &OneHotPoly<F, I>,
    ring_range: std::ops::Range<usize>,
    scalars: &[F],
) -> CyclotomicRing<F, D>
where
    F: Field,
    I: OneHotIndex,
{
    let mut coeffs_acc = [F::zero(); D];
    let (_, coefficients) = poly
        .ring_range_coefficients(D, ring_range.clone())
        .expect("validated one hot fold range");
    for coefficient in coefficients {
        let coefficient = coefficient.expect("validated one hot field position");
        let position = coefficient.ring_idx(D) - ring_range.start;
        if let Some(&scalar) = scalars.get(position) {
            coeffs_acc[coefficient.coeff_idx(D)] += scalar;
        }
    }
    CyclotomicRing::from_coefficients(coeffs_acc)
}

#[cfg(test)]
pub(super) fn fold_onehot_block_ring<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[CyclotomicRing<F, D>],
    num_positions_per_block: usize,
) -> CyclotomicRing<F, D>
where
    F: Field,
{
    let mut acc = CyclotomicRing::<F, D>::zero();

    for entry in entries {
        let pos = entry.pos_in_block();
        let coeff_idx = entry.coeff_idx();
        if pos < scalars.len() && pos < num_positions_per_block {
            scalars[pos].shift_accumulate_into(&mut acc, coeff_idx);
        }
    }

    acc
}

pub(super) fn fold_onehot_block_subfield<F, I, const D: usize>(
    poly: &OneHotPoly<F, I>,
    ring_range: std::ops::Range<usize>,
    multipliers: &SubfieldMultiplierOpeningPoint<F>,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: Field,
    I: OneHotIndex,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    let (_, coefficients) = poly.ring_range_coefficients(D, ring_range.clone())?;
    for coefficient in coefficients {
        let coefficient = coefficient?;
        multipliers.accumulate_position_monomial(
            coefficient.ring_idx(D) - ring_range.start,
            coefficient.coeff_idx(D),
            F::one(),
            &mut acc,
        )?;
    }
    Ok(acc)
}
