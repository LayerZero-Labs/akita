use super::*;
use akita_types::DigitBlocks;

/// Convert a field element to a centered signed byte when it fits.
#[inline(always)]
pub fn try_centered_i8<F: CanonicalField>(coeff: F, q: u128, half_q: u128) -> Option<i8> {
    let canonical = coeff.to_canonical_u128();
    let centered = if canonical > half_q {
        -((q - canonical) as i128)
    } else {
        canonical as i128
    };
    if (i8::MIN as i128..=i8::MAX as i128).contains(&centered) {
        Some(centered as i8)
    } else {
        None
    }
}

/// Basis-decompose a block of ring elements into `block.len() * num_digits` gadget components.
pub fn decompose_block<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<CyclotomicRing<F, D>> {
    let mut out = vec![CyclotomicRing::<F, D>::zero(); block.len() * num_digits];
    for (i, coeff_vec) in block.iter().enumerate() {
        coeff_vec.balanced_decompose_pow2_into(
            &mut out[i * num_digits..(i + 1) * num_digits],
            log_basis,
        );
    }
    out
}

/// Like [`decompose_block`] but outputs `[i8; D]` digit planes instead of ring elements.
pub fn decompose_block_i8<F: FieldCore + CanonicalField, const D: usize>(
    block: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = vec![[0i8; D]; block.len() * num_digits];
    decompose_rows_i8_into(block, &mut out, num_digits, log_basis);
    out
}

/// Decompose each ring element in `rows` into `[i8; D]` digit planes.
pub fn decompose_rows_i8<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    num_digits: usize,
    log_basis: u32,
) -> Vec<[i8; D]> {
    let mut out = vec![[0i8; D]; rows.len() * num_digits];
    decompose_rows_i8_into(rows, &mut out, num_digits, log_basis);
    out
}

/// Decompose each ring element in `rows` into a preallocated flat digit buffer.
///
/// # Panics
///
/// Panics if `out.len() != rows.len() * num_digits`.
pub fn decompose_rows_i8_into<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    out: &mut [[i8; D]],
    num_digits: usize,
    log_basis: u32,
) {
    assert_eq!(
        out.len(),
        rows.len() * num_digits,
        "flat digit output length must match rows * num_digits",
    );
    if num_digits == 0 {
        return;
    }
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2Params::new(num_digits, log_basis, q);

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(num_digits)
        .zip(rows.par_iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_i8_into_with_params(dst_chunk, &decompose_params)
        });

    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(num_digits)
        .zip(rows.iter())
        .for_each(|(dst_chunk, row)| {
            row.balanced_decompose_pow2_i8_into_with_params(dst_chunk, &decompose_params)
        });
}

/// Project recomposed A-role rows into native role rings and decompose them.
///
/// The output order within each block is
/// `[A row][role subcolumn][digit][role coefficient]`. Skips decomposition for
/// all-zero blocks and leaves their digit buffers zeroed.
pub fn decompose_commit_blocks_into<F, const D_A: usize, const D_ROLE: usize>(
    rows: &[&[CyclotomicRing<F, D_A>]],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<DigitBlocks, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if !D_A.is_multiple_of(D_ROLE) {
        return Err(AkitaError::InvalidSetup(format!(
            "source ring dimension {D_A} is not divisible by role ring dimension {D_ROLE}"
        )));
    }
    let role_subcolumns = D_A / D_ROLE;
    let block_sizes: Vec<usize> = rows
        .iter()
        .map(|&block_rows| {
            block_rows
                .len()
                .checked_mul(role_subcolumns)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "commit witness role subcolumn count overflow".to_string(),
                    )
                })?
                .checked_mul(num_digits_open)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "commit witness digit block length overflow".to_string(),
                    )
                })
        })
        .collect::<Result<_, _>>()?;
    let mut out = DigitBlocks::zeroed(block_sizes, D_ROLE)?;
    let dst_blocks = out.split_typed_blocks_mut::<D_ROLE>()?;
    let q = (-F::one()).to_canonical_u128() + 1;
    let params = BalancedDecomposePow2Params::new(num_digits_open, log_basis, q);
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(rows))
        .for_each(|(dst, &block_rows)| {
            decompose_commit_block_rows_into(block_rows, dst, &params);
        });
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(rows.iter())
        .for_each(|(dst, &block_rows)| {
            decompose_commit_block_rows_into(block_rows, dst, &params);
        });
    Ok(out)
}

fn decompose_commit_block_rows_into<F, const D_A: usize, const D_ROLE: usize>(
    block_rows: &[CyclotomicRing<F, D_A>],
    dst: &mut [[i8; D_ROLE]],
    params: &BalancedDecomposePow2Params,
) where
    F: FieldCore + CanonicalField,
{
    if block_rows.iter().all(CyclotomicRing::is_zero) {
        debug_assert!(dst.iter().all(|plane| plane.iter().all(|&d| d == 0)));
        return;
    }
    let role_subcolumns = D_A / D_ROLE;
    let num_digits = dst.len() / (block_rows.len() * role_subcolumns);
    for (row, row_dst) in block_rows
        .iter()
        .zip(dst.chunks_mut(role_subcolumns * num_digits))
    {
        let (subcolumns, remainder) = row.coefficients().as_chunks::<D_ROLE>();
        debug_assert!(remainder.is_empty());
        for (coefficients, subcolumn_dst) in subcolumns.iter().zip(row_dst.chunks_mut(num_digits)) {
            akita_algebra::balanced_decompose_coefficients_pow2_i8_into(
                coefficients,
                subcolumn_dst.as_flattened_mut(),
                params,
            );
        }
    }
}

/// Like [`decompose_rows_i8_into`] for inner-commitment digit staging only.
///
/// Debug builds round-trip check digits against `rows`; other callers should use
/// [`decompose_rows_i8_into`] directly.
pub fn decompose_commit_rows_i8_into<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    out: &mut [[i8; D]],
    num_digits: usize,
    log_basis: u32,
) {
    decompose_rows_i8_into(rows, out, num_digits, log_basis);
    #[cfg(debug_assertions)]
    {
        if let Err(err) = check_rows_i8_digit_planes(rows, out, num_digits, log_basis) {
            debug_assert!(false, "{err}");
        }
    }
}

#[cfg(debug_assertions)]
fn check_rows_i8_digit_planes<F: FieldCore + CanonicalField, const D: usize>(
    rows: &[CyclotomicRing<F, D>],
    digits: &[[i8; D]],
    num_digits: usize,
    log_basis: u32,
) -> Result<(), AkitaError> {
    if digits.len() != rows.len() * num_digits {
        return Err(AkitaError::InvalidSetup(format!(
            "expected {} digit planes for {} rows with {num_digits} digits each, got {}",
            rows.len() * num_digits,
            rows.len(),
            digits.len()
        )));
    }
    for (row_idx, row) in rows.iter().enumerate() {
        let row_digits = &digits[row_idx * num_digits..(row_idx + 1) * num_digits];
        if row.is_zero() {
            if row_digits.iter().any(|plane| plane.iter().any(|&d| d != 0)) {
                return Err(AkitaError::InvalidSetup(format!(
                    "nonzero decomposed digits for zero inner commitment row {row_idx}"
                )));
            }
        } else {
            let recomposed = CyclotomicRing::gadget_recompose_pow2_i8(row_digits, log_basis);
            if *row != recomposed {
                return Err(AkitaError::InvalidSetup(format!(
                    "recomposed row {row_idx} does not match decomposed digits"
                )));
            }
        }
    }
    Ok(())
}
