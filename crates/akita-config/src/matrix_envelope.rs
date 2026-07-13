//! Shared setup-matrix envelope accumulation for certification paths.

use akita_field::AkitaError;
use akita_types::LevelParams;

/// Extend `max_setup_len` with the per-level A/B/D key footprint.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on overflow.
pub(crate) fn accumulate_matrix_envelope_for_level(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    let a_len = lp
        .a_key
        .row_len()
        .checked_mul(lp.inner_width())
        .ok_or_else(|| AkitaError::InvalidSetup("A setup envelope overflow".to_string()))?;
    let b_len = lp
        .b_key
        .row_len()
        .checked_mul(lp.outer_width())
        .ok_or_else(|| AkitaError::InvalidSetup("B setup envelope overflow".to_string()))?;
    let d_len = lp
        .d_key
        .row_len()
        .checked_mul(lp.d_matrix_width())
        .ok_or_else(|| AkitaError::InvalidSetup("D setup envelope overflow".to_string()))?;
    *max_setup_len = (*max_setup_len).max(a_len).max(b_len).max(d_len);
    Ok(())
}
