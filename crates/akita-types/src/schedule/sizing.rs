//! Recursive witness sizing shared by planning and runtime validation.

use crate::CommittedGroupParams;
use akita_field::{AkitaError, CanonicalField};

/// Number of gadget decomposition levels needed for `r` over field `F`.
pub fn r_decomp_levels<F: CanonicalField>(log_basis: u32) -> usize {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    crate::sis::compute_num_digits_field_width(field_bits, log_basis)
}

/// Detect the field modulus from the canonical representation.
///
/// Uses the identity: the canonical form of `-1` in `Z_q` is `q - 1`.
#[inline]
pub fn detect_field_modulus<F: CanonicalField>() -> u128 {
    crate::dispatch::field_modulus::<F>()
}

/// Total ring elements in an intermediate recursive witness polynomial.
/// Terminal witnesses are quotient-free and must be sized from their
/// [`crate::TerminalResponseShape`] instead.
pub fn intermediate_w_ring_element_count_with_counts<F: CanonicalField>(
    lp: &CommittedGroupParams,
    num_polynomials: usize,
    num_z_segments: usize,
) -> Result<usize, AkitaError> {
    let modulus = detect_field_modulus::<F>();
    let field_bits = 128 - (modulus.saturating_sub(1)).leading_zeros();
    intermediate_w_ring_element_count_with_counts_bits(
        field_bits,
        lp,
        num_polynomials,
        num_z_segments,
    )
}

/// Non-generic variant of [`intermediate_w_ring_element_count_with_counts`] for
/// callers that already know the effective field bit width. The planner
/// search uses this to keep its API free of a base-field type parameter.
pub fn intermediate_w_ring_element_count_with_counts_bits(
    field_bits: u32,
    lp: &CommittedGroupParams,
    num_polynomials: usize,
    num_z_segments: usize,
) -> Result<usize, AkitaError> {
    lp.require_scalar_level("intermediate_w_ring_element_count_with_counts_bits")?;
    let e_hat_count = num_polynomials
        .checked_mul(lp.num_live_blocks)
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness W width overflow".to_string()))?;
    let t_hat_count = num_polynomials
        .checked_mul(lp.num_live_blocks)
        .and_then(|n| n.checked_mul(lp.inner_commit_matrix.output_rank()))
        .and_then(|n| n.checked_mul(lp.num_digits_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T width overflow".to_string()))?;
    let num_digits_fold = lp.num_digits_fold();
    let z_pre_count = num_z_segments
        .checked_mul(lp.inner_width())
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z width overflow".to_string()))?;
    let r_rows = lp.relation_matrix_row_count(1)?;
    let r_count = r_rows
        .checked_mul(crate::sis::compute_num_digits_field_width(
            field_bits,
            lp.log_basis_open,
        ))
        .ok_or_else(|| AkitaError::InvalidSetup("witness r-tail width overflow".to_string()))?;

    e_hat_count
        .checked_add(t_hat_count)
        .and_then(|n| n.checked_add(z_pre_count))
        .and_then(|n| n.checked_add(r_count))
        .ok_or_else(|| AkitaError::InvalidSetup("witness width overflow".to_string()))
}

/// Witness ring-element count for a chunked (multi-chunk) or single-chunk layout.
///
/// `num_chunks == 1` delegates to
/// [`intermediate_w_ring_element_count_with_counts_bits`] with `num_public_rows = 1`,
/// so it is byte-identical to the historical single-chunk pricing.
///
/// `num_chunks > 1` prices the multi-chunk witness layout used by the distributed
/// prover: `num_chunks` chunks each holding a partitioned slice of `ê`/`t̂` plus a
/// **replicated full-width** `ẑ`, followed by a single shared `r`-tail. The
/// per-node relations stack *horizontally* (`M = [M_0 | … | M_{num_chunks-1}]`),
/// sharing the same row blocks (concatenation adds columns, not rows) and summing
/// the partial commitments `u_j` into one `u`, so the quotient `r = Σ_j r_j` keeps
/// the **single-machine shape** — its row count is priced with `num_commitments =
/// 1`, unchanged from the single-chunk layout. The **only** extra cost over the
/// single-chunk layout is `(num_chunks - 1) · z_chunk` ring elements (the
/// replicated `ẑ`).
///
/// The exact `ê`/`t̂` live-block prefix is partitioned without padding. Its
/// total width and the shared `r` tail therefore stay unchanged.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `num_chunks == 0`, `num_chunks > 1`
/// is not a power of two, or any width product overflows. Empty chunk ranges do
/// not change the partitioned E/T total, and every chunk retains its Z copy.
/// Never panics because the runtime DP fallback is verifier reachable.
pub fn intermediate_w_ring_element_count_for_chunks(
    field_bits: u32,
    lp: &CommittedGroupParams,
    num_polynomials: usize,
    num_chunks: usize,
) -> Result<usize, AkitaError> {
    if num_chunks == 0 {
        return Err(AkitaError::InvalidSetup(
            "intermediate_w_ring_element_count_for_chunks: num_chunks must be >= 1".to_string(),
        ));
    }
    if num_chunks == 1 {
        return intermediate_w_ring_element_count_with_counts_bits(
            field_bits,
            lp,
            num_polynomials,
            1,
        );
    }
    if !num_chunks.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "intermediate_w_ring_element_count_for_chunks: num_chunks must be a power of two"
                .to_string(),
        ));
    }
    let overflow = || AkitaError::InvalidSetup("chunked witness width overflow".to_string());
    let single =
        intermediate_w_ring_element_count_with_counts_bits(field_bits, lp, num_polynomials, 1)?;
    let num_digits_fold = lp.num_digits_fold();
    let z_chunk = lp
        .inner_width()
        .checked_mul(num_digits_fold)
        .ok_or_else(overflow)?;
    num_chunks
        .checked_sub(1)
        .and_then(|copies| copies.checked_mul(z_chunk))
        .and_then(|extra| single.checked_add(extra))
        .ok_or_else(overflow)
}
