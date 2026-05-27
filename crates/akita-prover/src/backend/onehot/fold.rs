use super::*;

pub(super) fn fold_single_chunk_onehot_block<F: FieldCore, const D: usize>(
    entries: &[SingleChunkEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut coeffs_acc = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            coeffs_acc[entry.coeff_idx()] += scalars[pos];
        }
    }
    CyclotomicRing::from_coefficients(coeffs_acc)
}

pub(super) fn fold_multi_chunk_onehot_block<F: FieldCore, const D: usize>(
    entries: &[MultiChunkEntry],
    scalars: &[F],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut coeffs_acc = [F::zero(); D];
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            let s = scalars[pos];
            for &ci in entry.nonzero_coeffs() {
                coeffs_acc[ci as usize] += s;
            }
        }
    }
    CyclotomicRing::from_coefficients(coeffs_acc)
}

pub(super) fn fold_single_chunk_onehot_block_ring<F: FieldCore, const D: usize>(
    entries: &[SingleChunkEntry],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            scalars[pos].shift_accumulate_into(&mut acc, entry.coeff_idx());
        }
    }
    acc
}

pub(super) fn fold_multi_chunk_onehot_block_ring<F: FieldCore, const D: usize>(
    entries: &[MultiChunkEntry],
    scalars: &[CyclotomicRing<F, D>],
    block_len: usize,
) -> CyclotomicRing<F, D> {
    let mut acc = CyclotomicRing::<F, D>::zero();
    for entry in entries {
        let pos = entry.pos_in_block();
        if pos < scalars.len() && pos < block_len {
            for &coeff_idx in entry.nonzero_coeffs() {
                scalars[pos].shift_accumulate_into(&mut acc, coeff_idx as usize);
            }
        }
    }
    acc
}
