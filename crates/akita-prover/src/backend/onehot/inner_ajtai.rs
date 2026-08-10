use super::*;

/// Wide-accumulator inner Ajtai: compute `t = A * s` for a one-hot block.
///
/// Instead of materializing the full decomposed vector `s` and doing a dense
/// matvec, we accumulate only the nonzero contributions using fused
/// shift-accumulate into `WideCyclotomicRing<W, D>` (carry-free i32
/// additions), then reduce once at the end:
///
/// ```text
/// t[a] += A[a][entry.commit_col(num_digits)] * (X^{k_1} + X^{k_2} + ...)
/// ```
///
/// Using the wide accumulator avoids per-addition modular reduction versus
/// a direct field-ring accumulator.
#[allow(non_snake_case)]
#[cfg(test)]
pub(crate) fn inner_ajtai_wide_onehot<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    entries: &[SparseRingBlockEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::CommitAccum, D>::zero(); n_a];

    for entry in entries {
        let pos_in_block = entry.pos_in_block();
        let coeff_idx = entry.coeff_idx();
        let col = pos_in_block * num_digits;
        for (a_row, t_w) in a_view.rows().zip(t_wide.iter_mut()) {
            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
            a_wide.shift_accumulate_into(t_w, coeff_idx);
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}

#[cfg(test)]
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_wide_single_chunk_tiled<F, const D: usize>(
    a_view: &RingMatrixView<'_, F, D>,
    entries: &[SparseRingBlockEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasCommitAccum,
    F::CommitAccum: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];

    for tile in entries.chunks(F::MAX_COMMIT_ACCUMULATIONS) {
        let partial = inner_ajtai_wide_onehot::<F, D>(a_view, tile, num_digits);
        for (dst, src) in t.iter_mut().zip(partial.iter()) {
            *dst += *src;
        }
    }

    t
}
