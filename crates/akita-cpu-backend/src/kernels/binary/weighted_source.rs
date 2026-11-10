//! Weighted integer bits lifted into a coefficient field.

use akita_algebra::binary::field_switch::SwitchField;
use akita_error::{checked, AkitaError};
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Weighted packed-source bits, with shared eight-row subset sums.
///
/// Columns store `weights.len() * components` packed carriers contiguously.
/// Each output column has `padded_coefficients` entries, with bit `s` of
/// component `a` at `s * components + a`; all remaining entries are zero.
/// `negated_bits` selects bit coordinates whose integer packing sign is -1.
/// This arithmetic boundary accepts explicit weights, independently of any
/// proof layout, transcript, or application claim.
pub fn weighted_source_bits<H: SwitchField, E: Field>(
    source: &[H::Source],
    weights: &[E],
    components: usize,
    columns: usize,
    padded_coefficients: usize,
    negated_bits: u128,
) -> Result<Vec<E>, AkitaError> {
    let invalid = || AkitaError::InvalidInput("weighted source bit geometry mismatch".into());
    let k = components;
    let m = weights.len();
    let rows = checked::product([m, k]).ok_or_else(invalid)?;
    let padded = padded_coefficients;
    let degree = checked::product([128, k]).ok_or_else(invalid)?;
    if !m.is_power_of_two()
        || k == 0
        || columns == 0
        || padded < degree
        || checked::product([rows, columns]) != Some(source.len())
    {
        return Err(invalid());
    }
    let group_size = m.min(8);
    let sums_per_group = checked::pow2(group_size).ok_or_else(invalid)?;
    let groups = checked::exact_div(m, group_size).ok_or_else(invalid)?;
    let subset_len = checked::product([groups, sums_per_group]).ok_or_else(invalid)?;
    let group_rows = checked::product([group_size, k]).ok_or_else(invalid)?;
    let byte_coefficients = checked::product([8, k]).ok_or_else(invalid)?;
    let source_coefficients = checked::product([u128::BITS as usize, k]).ok_or_else(invalid)?;
    let allocation = || AkitaError::InvalidInput("prime left opening allocation failed".into());
    let mut subsets = Vec::new();
    subsets
        .try_reserve_exact(subset_len)
        .map_err(|_| allocation())?;
    subsets.resize(subset_len, E::zero());
    for (sums, group) in subsets
        .chunks_exact_mut(sums_per_group)
        .zip(weights.chunks_exact(group_size))
    {
        for (i, &weight) in group.iter().enumerate() {
            let half = checked::pow2(i).ok_or_else(invalid)?;
            let (earlier, later) = sums.split_at_mut_checked(half).ok_or_else(invalid)?;
            for (slot, &previous) in later.iter_mut().zip(earlier.iter()) {
                *slot = previous + weight;
            }
        }
    }
    let negated = negated_bits;
    let mut table = Vec::new();
    table
        .try_reserve_exact(checked::product([columns, padded]).ok_or_else(invalid)?)
        .map_err(|_| allocation())?;
    table.resize(
        checked::product([columns, padded]).ok_or_else(invalid)?,
        E::zero(),
    );
    let fill = |(column, words): (&mut [E], &[H::Source])| -> Result<(), AkitaError> {
        let live = column.get_mut(..degree).ok_or_else(invalid)?;
        let live = live.get_mut(..source_coefficients).ok_or_else(invalid)?;
        for (cells, sums) in words
            .chunks_exact(group_rows)
            .zip(subsets.chunks_exact(sums_per_group))
        {
            for component in 0..k {
                // Unused lanes of a short group stay zero.
                let mut lanes = [0u128; 8];
                for (lane, element) in lanes.iter_mut().zip(cells.chunks_exact(k)) {
                    *lane = (*element.get(component).ok_or_else(invalid)?).into();
                }
                for block in live.chunks_exact_mut(byte_coefficients) {
                    let mut matrix = 0u64;
                    for (lane, shift) in lanes.iter_mut().zip((0..64).step_by(8)) {
                        matrix |= (*lane as u64 & 0xff) << shift;
                        *lane >>= 8;
                    }
                    // Transpose eight source bytes: output byte s selects
                    // the equality weights whose source bit s is set.
                    let swap = (matrix ^ (matrix >> 7)) & 0x00aa_00aa_00aa_00aa;
                    matrix ^= swap ^ (swap << 7);
                    let swap = (matrix ^ (matrix >> 14)) & 0x0000_cccc_0000_cccc;
                    matrix ^= swap ^ (swap << 14);
                    let swap = (matrix ^ (matrix >> 28)) & 0x0000_0000_f0f0_f0f0;
                    matrix ^= swap ^ (swap << 28);
                    for (coefficients, selector) in
                        block.chunks_exact_mut(k).zip(matrix.to_le_bytes())
                    {
                        let weight = *sums.get(usize::from(selector)).ok_or_else(invalid)?;
                        *coefficients.get_mut(component).ok_or_else(invalid)? += weight;
                    }
                }
            }
        }
        for (s, coefficients) in live.chunks_exact_mut(k).enumerate() {
            if (negated >> s) & 1 != 0 {
                for slot in coefficients {
                    *slot = -*slot;
                }
            }
        }
        Ok(())
    };
    #[cfg(feature = "parallel")]
    table
        .par_chunks_mut(padded)
        .zip(source.par_chunks_exact(rows))
        .try_for_each(fill)?;
    #[cfg(not(feature = "parallel"))]
    table
        .chunks_mut(padded)
        .zip(source.chunks_exact(rows))
        .try_for_each(fill)?;
    Ok(table)
}
