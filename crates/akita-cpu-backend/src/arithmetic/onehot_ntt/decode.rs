use crate::opaque::{CommitInnerPlan, OneHotSource};
use crate::OneHotIndex;
use akita_error::{checked, AkitaError};

pub(super) fn validate<const D: usize, I: OneHotIndex>(
    source: OneHotSource<'_, I>,
    plan: CommitInnerPlan,
) -> Result<(), AkitaError> {
    let count = source.num_live_blocks_for(D, plan.num_positions_per_block)?;
    if plan.ring_dimension != D || count != plan.num_live_blocks || plan.num_digits_inner == 0 {
        return Err(AkitaError::InvalidInput(
            "one-hot NTT block layout mismatch".into(),
        ));
    }
    let fields = checked::pow2(source.num_vars)
        .ok_or_else(|| AkitaError::InvalidInput("one-hot extent overflow".into()))?;
    if source.chunk_size == 0 || fields / source.chunk_size != source.indices.len() {
        return Err(AkitaError::InvalidInput(
            "one-hot source length mismatch".into(),
        ));
    }
    Ok(())
}

/// Decode directly into a bounded column tile, without sparse entry buffers.
pub(super) fn fill<T: Copy + Default + From<i8>, I: OneHotIndex, const D: usize>(
    source: OneHotSource<'_, I>,
    plan: CommitInnerPlan,
    block: usize,
    column_start: usize,
    output: &mut [[T; D]],
) -> Result<(), AkitaError> {
    output.fill([T::default(); D]);
    let total = checked::pow2(source.num_vars)
        .ok_or_else(|| AkitaError::InvalidInput("one-hot extent overflow".into()))?;
    for (offset, row) in output.iter_mut().enumerate() {
        let column = column_start + offset;
        if !column.is_multiple_of(plan.num_digits_inner) {
            continue;
        }
        let ring = block
            .checked_mul(plan.num_positions_per_block)
            .and_then(|v| v.checked_add(column / plan.num_digits_inner))
            .ok_or_else(|| AkitaError::InvalidInput("one-hot ring offset overflow".into()))?;
        let start = ring
            .checked_mul(D)
            .ok_or_else(|| AkitaError::InvalidInput("one-hot field offset overflow".into()))?;
        if start >= total {
            continue;
        }
        let end = (start + D).min(total);
        let first = start / source.chunk_size;
        let last = end.div_ceil(source.chunk_size).min(source.indices.len());
        for (offset, hot) in source.indices[first..last].iter().enumerate() {
            if let Some(hot) = hot {
                let hot = hot.as_usize();
                if hot >= source.chunk_size {
                    return Err(AkitaError::InvalidInput(
                        "one-hot index outside chunk".into(),
                    ));
                }
                let position = (first + offset)
                    .checked_mul(source.chunk_size)
                    .and_then(|v| v.checked_add(hot))
                    .ok_or_else(|| AkitaError::InvalidInput("one-hot position overflow".into()))?;
                if (start..end).contains(&position) {
                    row[position - start] = T::from(1i8);
                }
            }
        }
    }
    Ok(())
}
