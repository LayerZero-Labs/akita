use super::*;

/// Wide-accumulator multi-chunk inner Ajtai: compute `t = A * s` for a
/// one-hot block.
///
/// Instead of materializing the full decomposed vector `s` and doing a dense
/// matvec, we accumulate only the nonzero contributions using fused
/// shift-accumulate into `WideCyclotomicRing<W, D>` (carry-free i32
/// additions), then reduce once at the end:
///
/// ```text
/// t[a] += A[a][entry.pos * num_digits] * (X^{k_1} + X^{k_2} + ...)
/// ```
///
/// Using the wide accumulator avoids per-addition modular reduction versus
/// a direct field-ring accumulator. Long multi-chunk blocks are internally
/// tiled so no wide accumulator receives more than
/// [`MAX_WIDE_SHIFT_ACCUMULATIONS`] shift-adds before reduction.
#[allow(non_snake_case)]
pub(crate) fn inner_ajtai_wide_multi_chunk<F, const D: usize>(
    A: &RingMatrixView<'_, F, D>,
    multi_chunk_entries: &[MultiChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + ReduceTo<F>,
{
    let n_a = A.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];
    let mut t: Option<Vec<CyclotomicRing<F, D>>> = None;
    let mut shift_accumulations = 0usize;

    for entry in multi_chunk_entries {
        let col = entry.pos_in_block() * num_digits;
        let mut coeffs = entry.nonzero_coeffs();
        while !coeffs.is_empty() {
            if shift_accumulations == MAX_WIDE_SHIFT_ACCUMULATIONS {
                let t = t.get_or_insert_with(|| vec![CyclotomicRing::<F, D>::zero(); n_a]);
                for (dst, src) in t.iter_mut().zip(t_wide.iter_mut()) {
                    *dst += std::mem::replace(src, WideCyclotomicRing::zero()).reduce();
                }
                shift_accumulations = 0;
            }

            let remaining = MAX_WIDE_SHIFT_ACCUMULATIONS - shift_accumulations;
            let take = remaining.min(coeffs.len());
            let (current, rest) = coeffs.split_at(take);
            for (a_row, t_w) in A.rows().zip(t_wide.iter_mut()) {
                let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
                for &ci in current {
                    a_wide.shift_accumulate_into(t_w, ci as usize);
                }
            }
            shift_accumulations += take;
            coeffs = rest;
        }
    }

    if let Some(mut t) = t {
        for (dst, src) in t.iter_mut().zip(t_wide) {
            *dst += src.reduce();
        }
        t
    } else {
        t_wide.into_iter().map(|w| w.reduce()).collect()
    }
}

pub(super) fn inner_ajtai_wide_single_chunk<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    single_chunk_entries: &[SingleChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t_wide = vec![WideCyclotomicRing::<F::Wide, D>::zero(); n_a];

    for entry in single_chunk_entries {
        let col = entry.pos_in_block() * num_digits;
        let coeff_idx = entry.coeff_idx();
        for (a_row, t_w) in a_view.rows().zip(t_wide.iter_mut()) {
            let a_wide = WideCyclotomicRing::from_ring(&a_row[col]);
            a_wide.shift_accumulate_into(t_w, coeff_idx);
        }
    }

    t_wide.into_iter().map(|w| w.reduce()).collect()
}
pub(super) fn inner_ajtai_wide_single_chunk_tiled<F, const D: usize>(
    a_view: &akita_types::RingMatrixView<'_, F, D>,
    single_chunk_entries: &[SingleChunkEntry],
    num_digits: usize,
) -> Vec<CyclotomicRing<F, D>>
where
    F: FieldCore + CanonicalField + HasWide,
    F::Wide: AdditiveGroup + From<F> + akita_field::fields::wide::ReduceTo<F>,
{
    let n_a = a_view.num_rows();
    let mut t = vec![CyclotomicRing::<F, D>::zero(); n_a];

    for tile in single_chunk_entries.chunks(MAX_WIDE_SHIFT_ACCUMULATIONS) {
        let partial = inner_ajtai_wide_single_chunk(a_view, tile, num_digits);
        for (dst, src) in t.iter_mut().zip(partial.iter()) {
            *dst += *src;
        }
    }

    t
}
