use super::*;
use akita_types::RingMultiplierOpeningPoint;

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
    multipliers: &RingMultiplierOpeningPoint<F>,
) -> Result<CyclotomicRing<F, D>, AkitaError>
where
    F: FieldCore,
    I: OneHotIndex,
{
    let field_start = ring_range
        .start
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidInput("one-hot fold range overflow".into()))?;
    let field_end = ring_range
        .end
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidInput("one-hot fold range overflow".into()))?;
    let chunk_start = field_start / onehot_k;
    let chunk_end = field_end.div_ceil(onehot_k).min(indices.len());
    let mut acc = CyclotomicRing::<F, D>::zero();
    for (local_chunk, hot_index) in indices[chunk_start..chunk_end].iter().copied().enumerate() {
        let Some(hot_index) = hot_index else {
            continue;
        };
        let field_position = (chunk_start + local_chunk)
            .checked_mul(onehot_k)
            .and_then(|base| base.checked_add(hot_index.as_usize()))
            .ok_or_else(|| AkitaError::InvalidInput("one-hot field position overflow".into()))?;
        let ring_index = field_position / D;
        if !ring_range.contains(&ring_index) {
            continue;
        }
        multipliers.accumulate_position_monomial(
            ring_index - ring_range.start,
            field_position % D,
            F::one(),
            &mut acc,
        )?;
    }
    Ok(acc)
}
