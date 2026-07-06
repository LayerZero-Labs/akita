//! Prover-side re-home of the dropped `recomposed_inner_rows` hint material
//! (runtime-ring cutover, Slice A).
//!
//! # Background
//!
//! Before S4, `AkitaCommitmentHint<F, D>` carried two fields: the serialized
//! `decomposed_inner_rows` (digit planes) and a prover-only
//! `recomposed_inner_rows: Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>`. S4 made
//! the protocol hint D-free ([`akita_types::AkitaCommitmentHint<F>`]), keeping
//! only the D-free decomposed digit stream ([`akita_types::DigitBlocks`]). The
//! D-typed recomposed rows cannot live in a D-free struct.
//!
//! The recomposed rows are **derivable** from the decomposed digit stream: each
//! consecutive run of `num_digits_open` digit planes recomposes (via
//! `CyclotomicRing::gadget_recompose_pow2_i8`) into one inner `A·s_i` ring
//! element. This module re-homes that computation on the prover side: given a
//! D-free hint plus `(num_digits_open, log_basis)`, it reconstructs the typed
//! per-poly → per-block recomposed rows. Callers that previously read the cached
//! `recomposed_inner_rows` now recompute them here.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{AkitaCommitmentHint, DigitBlocks};

/// Recompose one D-free [`DigitBlocks`] digit stream into typed inner rows,
/// grouped by block (`Vec<block> of Vec<CyclotomicRing<F, D>>`).
///
/// Each block's plane count must be a multiple of `num_digits_open`; every run
/// of `num_digits_open` planes recomposes into one ring element.
///
/// # Errors
///
/// Returns an error if `num_digits_open` is zero, if the digit stride does not
/// match `D`, or if a block plane count is not a multiple of `num_digits_open`.
pub fn recompose_inner_rows<F, const D: usize>(
    digits: &DigitBlocks,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if num_digits_open == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_digits_open must be nonzero when recomposing inner rows".to_string(),
        ));
    }
    if digits.digit_stride() != D {
        return Err(AkitaError::InvalidInput(format!(
            "hint digit stride {} does not match prover ring dimension D={D}",
            digits.digit_stride()
        )));
    }
    digits
        .iter_blocks()
        .map(|block| recompose_block::<F, D>(block, num_digits_open, log_basis))
        .collect()
}

/// Recompose every per-polynomial digit stream in a D-free hint into typed
/// inner rows, grouped by polynomial then block.
///
/// This mirrors the former `AkitaCommitmentHint::ensure_recomposed_inner_rows`
/// followed by reading `recomposed_inner_rows`, but recomputes from the D-free
/// decomposed stream rather than caching a D-typed field.
///
/// # Errors
///
/// Propagates [`recompose_inner_rows`] failures.
pub fn recompose_hint_inner_rows<F, const D: usize>(
    hint: &AkitaCommitmentHint<F>,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    hint.decomposed_inner_rows()
        .iter()
        .map(|digits| recompose_inner_rows::<F, D>(digits, num_digits_open, log_basis))
        .collect()
}

/// Recompose the *flattened* (all-polynomials concatenated) view of a hint into
/// typed inner rows grouped by block.
///
/// Equivalent to flattening the hint and recomposing; used by the ring-switch /
/// terminal paths that consume one flat block stream.
///
/// # Errors
///
/// Propagates flattening and [`recompose_inner_rows`] failures.
pub fn recompose_flat_hint_inner_rows<F, const D: usize>(
    hint: &AkitaCommitmentHint<F>,
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    let flat = hint.clone().into_flat_parts()?;
    recompose_inner_rows::<F, D>(&flat, num_digits_open, log_basis)
}

fn recompose_block<F, const D: usize>(
    block: &[i8],
    num_digits_open: usize,
    log_basis: u32,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: FieldCore + CanonicalField,
{
    if !block.len().is_multiple_of(D) {
        return Err(AkitaError::InvalidSize {
            expected: D,
            actual: block.len(),
        });
    }
    let (planes, remainder) = block.as_chunks::<D>();
    debug_assert!(remainder.is_empty(), "checked multiple of D above");
    if !planes.len().is_multiple_of(num_digits_open) {
        return Err(AkitaError::InvalidSetup(format!(
            "decomposed inner row block has {} planes, expected a multiple of \
             num_digits_open={num_digits_open}",
            planes.len()
        )));
    }
    Ok(planes
        .chunks(num_digits_open)
        .map(|digits| CyclotomicRing::gadget_recompose_pow2_i8(digits, log_basis))
        .collect())
}
