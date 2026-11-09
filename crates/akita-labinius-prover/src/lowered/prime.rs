use akita_algebra::{binary::field_switch::SwitchField, EqPolynomial};
use akita_cpu_backend::kernels::binary_source::weighted_source::weighted_source_bits;
use akita_error::{checked, AkitaError};
use akita_labinius_verifier::lowered::LoweredRootLayout;
use jolt_field::Field;

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
    let mut negated = 0u128;
    for s in 0..(u128::BITS as usize).min(degree / k) {
        let coefficient = checked::product([s, k]).ok_or_else(invalid)?;
        if layout.sigma(coefficient)? != 1 {
            negated |= 1 << s;
        }
    }
    weighted_source_bits::<H, E>(source, &weights, k, layout.columns(), padded, negated)
}
