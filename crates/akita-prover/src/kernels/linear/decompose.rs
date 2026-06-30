use super::*;
use crate::compute::FlatDigitBlocks;

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
    let decompose_params = BalancedDecomposePow2I8Params::new(num_digits, log_basis, q);

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

/// Stage flat i8 digit blocks for inner commitment from recomposed Ajtai rows.
///
/// Skips i8 decomposition for all-zero blocks and leaves their digit buffers zeroed.
pub fn decompose_commit_blocks_into<F, const D: usize>(
    rows: &[Vec<CyclotomicRing<F, D>>],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<FlatDigitBlocks<D>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let block_sizes: Vec<usize> = rows
        .iter()
        .map(|block_rows| {
            block_rows
                .len()
                .checked_mul(num_digits_open)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "commit witness digit block length overflow".to_string(),
                    )
                })
        })
        .collect::<Result<_, _>>()?;
    let mut out = FlatDigitBlocks::zeroed(block_sizes)?;
    let dst_blocks = out.split_blocks_mut();
    #[cfg(feature = "parallel")]
    cfg_into_iter!(dst_blocks)
        .zip(cfg_iter!(rows))
        .for_each(|(dst, block_rows)| {
            decompose_commit_block_rows_into(block_rows, dst, num_digits_open, log_basis);
        });
    #[cfg(not(feature = "parallel"))]
    dst_blocks
        .into_iter()
        .zip(rows.iter())
        .for_each(|(dst, block_rows)| {
            decompose_commit_block_rows_into(block_rows, dst, num_digits_open, log_basis);
        });
    Ok(out)
}

fn decompose_commit_block_rows_into<F, const D: usize>(
    block_rows: &[CyclotomicRing<F, D>],
    dst: &mut [[i8; D]],
    num_digits_open: usize,
    log_basis: u32,
) where
    F: FieldCore + CanonicalField,
{
    if block_rows.iter().all(CyclotomicRing::is_zero) {
        debug_assert!(dst.iter().all(|plane| plane.iter().all(|&d| d == 0)));
        return;
    }
    decompose_commit_rows_i8_into(block_rows, dst, num_digits_open, log_basis);
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

#[cfg(any(test, debug_assertions))]
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

/// Test helper for inner-commitment digit round-trip checks.
#[cfg(test)]
pub fn check_decomposed_rows_i8_match<F: FieldCore + CanonicalField, const D: usize>(
    inner: &crate::CommitInnerWitness<F>,
    n_a: usize,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<(), AkitaError> {
    use crate::api::commitment::commit_inner_block_digit_count;

    let decomposed = inner.decomposed_inner_rows_trusted::<D>()?;
    let expected_block_digits = commit_inner_block_digit_count(n_a, num_digits_open)?;
    for (block_idx, block_digits) in decomposed.iter_blocks().enumerate() {
        let recomposed_block = inner.recomposed_block_trusted::<D>(block_idx)?;
        if block_digits.len() != expected_block_digits {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {actual} decomposed digits for inner commitment block {block_idx}, expected {expected_block_digits}",
                actual = block_digits.len(),
                block_idx = block_idx,
                expected_block_digits = expected_block_digits
            )));
        }
        if recomposed_block.len() != n_a {
            return Err(AkitaError::InvalidSetup(format!(
                "backend returned {actual} rows for inner commitment block {block_idx}, expected {n_a} A rows",
                actual = recomposed_block.len(),
                block_idx = block_idx,
                n_a = n_a
            )));
        }
        check_rows_i8_digit_planes(recomposed_block, block_digits, num_digits_open, log_basis)
            .map_err(|err| {
                AkitaError::InvalidSetup(format!("inner commitment block {block_idx}: {err}"))
            })?;
    }
    Ok(())
}
