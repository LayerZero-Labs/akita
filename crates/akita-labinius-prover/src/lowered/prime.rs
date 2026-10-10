use akita_algebra::{binary::field_switch::SwitchField, EqPolynomial};
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::lowered::LoweredRootLayout;
use jolt_field::Field;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The prime left opening `uP[col] = sum_j eq(ring_point, j) * src[j][col]` of
/// every column, as one table in the layout's prime order: padded coefficient
/// innermost, column outermost, coefficient tails zero.
///
/// `src[j][col]` is the packed source ring element of `pack_source_column`:
/// coefficient `s * k + c` is bit `s` of source word
/// `col * scalar_rows + j * k + c`, negated where `LoweredRootLayout::sigma`
/// is `-1`. The bits are read as the integers zero and one and lifted into
/// `E`; nothing is reduced modulo two.
///
/// Equality weights are grouped in blocks of up to eight. Their subset sums
/// cost 255 additions in `E` per full group and are shared by all columns and
/// components. Each column and component uses 128 additions per group.
/// Packing signs are applied once per slot. Columns are independent.
pub fn prime_left_opening<H: SwitchField, E: Field>(
    layout: &LoweredRootLayout,
    source: &[H::Source],
    ring_point: &[E],
) -> Result<Vec<E>, AkitaError> {
    let invalid = || AkitaError::InvalidInput("prime left opening geometry mismatch".into());
    let k = layout.k();
    let rows = layout.scalar_rows();
    let padded = layout.padded_coefficients();
    let degree = layout.degree();
    if k == 0
        || rows == 0
        || padded < degree
        || checked::product([rows, layout.columns()]) != Some(source.len())
        || checked::product([layout.m(), k]) != Some(rows)
        || checked::pow2(ring_point.len()) != Some(layout.m())
    {
        return Err(invalid());
    }
    let weights = EqPolynomial::evals_serial(ring_point, None)?;
    let group_size = layout.m().min(8);
    let sums_per_group = checked::pow2(group_size).ok_or_else(invalid)?;
    let groups = checked::exact_div(layout.m(), group_size).ok_or_else(invalid)?;
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
    // Bit `s` of the mask is set where coordinate `s` is packed negated.
    let mut negated = 0u128;
    for s in 0..(u128::BITS as usize).min(degree / k) {
        let coefficient = checked::product([s, k]).ok_or_else(invalid)?;
        if layout.sigma(coefficient)? != 1 {
            negated |= 1 << s;
        }
    }
    let mut table = Vec::new();
    table
        .try_reserve_exact(layout.prime_len())
        .map_err(|_| allocation())?;
    table.resize(layout.prime_len(), E::zero());
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
