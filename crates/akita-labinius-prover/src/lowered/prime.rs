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
/// Each set source bit adds or subtracts one equality weight, so the cost is
/// one addition in `E` per set bit of the source. Columns are independent.
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
        .map_err(|_| AkitaError::InvalidInput("prime left opening allocation failed".into()))?;
    table.resize(layout.prime_len(), E::zero());
    let fill = |(column, words): (&mut [E], &[H::Source])| -> Result<(), AkitaError> {
        let live = column.get_mut(..degree).ok_or_else(invalid)?;
        for (cells, &weight) in words.chunks_exact(k).zip(&weights) {
            for (component, &word) in cells.iter().enumerate() {
                let mut bits: u128 = word.into();
                while bits != 0 {
                    let s = bits.trailing_zeros();
                    let slot = checked::mul_add(s as usize, k, component)
                        .and_then(|coefficient| live.get_mut(coefficient))
                        .ok_or_else(invalid)?;
                    if (negated >> s) & 1 == 0 {
                        *slot += weight;
                    } else {
                        *slot -= weight;
                    }
                    bits &= bits - 1;
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
