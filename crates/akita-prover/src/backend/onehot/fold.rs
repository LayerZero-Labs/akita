use super::*;
use akita_types::SubfieldMultiplierOpeningPoint;

pub(super) fn fold_onehot_block<F, I, const D: usize>(
    onehot_k: usize,
    indices: &[Option<I>],
    ring_range: std::ops::Range<usize>,
    scalars: &[F],
) -> CyclotomicRing<F, D>
where
    F: FieldCore,
    I: OneHotIndex,
{
    let mut coeffs_acc = [F::zero(); D];
    let entries = OneHotRingRange::new(onehot_k, indices, D, ring_range.clone())
        .expect("validated one hot fold range");
    for entry in entries {
        let entry = entry.expect("validated one hot field position");
        let position = entry.ring_index - ring_range.start;
        if let Some(&scalar) = scalars.get(position) {
            coeffs_acc[entry.coefficient_index] += scalar;
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
    F: FieldCore,
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
    onehot_k: usize,
    indices: &[Option<I>],
    ring_range: std::ops::Range<usize>,
    multipliers: &SubfieldMultiplierOpeningPoint<F>,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
    I: OneHotIndex,
{
    let mut acc = CyclotomicRing::<F, D>::zero();
    let entries = OneHotRingRange::new(onehot_k, indices, D, ring_range.clone())?;
    for entry in entries {
        let entry = entry?;
        multipliers.accumulate_position_monomial(
            entry.ring_index - ring_range.start,
            entry.coefficient_index,
            F::one(),
            &mut acc,
        )?;
    }
    Ok(acc)
}
