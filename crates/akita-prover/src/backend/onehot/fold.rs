use super::*;

pub(super) fn fold_onehot_block<F, const D: usize>(
    entries: &[SparseRingBlockEntry],
    scalars: &[F],
    num_positions_per_block: usize,
) -> CyclotomicRing<F, D>
where
    F: FieldCore,
{
    let mut coeffs_acc = [F::zero(); D];

    for entry in entries {
        let pos = entry.pos_in_block();
        let coeff_idx = entry.coeff_idx();
        if pos < scalars.len() && pos < num_positions_per_block {
            let s = scalars[pos];
            coeffs_acc[coeff_idx] += s;
        }
    }

    CyclotomicRing::from_coefficients(coeffs_acc)
}

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
