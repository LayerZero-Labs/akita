//! Shared setup-matrix envelope accumulation helpers.

use akita_field::AkitaError;
use akita_types::{LevelParams, SetupMatrixEnvelope, SetupPrefixSlotId};

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

fn include_matrix(
    envelope: &mut SetupMatrixEnvelope,
    rows: usize,
    columns: usize,
    role: &'static str,
) -> Result<(), AkitaError> {
    let len = rows
        .checked_mul(columns)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} setup envelope overflow")))?;
    envelope.max_setup_len = envelope.max_setup_len.max(len);
    Ok(())
}

/// Include the rounded prefix storage and A/B footprints for one setup-prefix slot.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the slot shape overflows `usize` or
/// has an invalid padded length.
pub(crate) fn inflate_envelope_for_setup_prefix_slot(
    envelope: &mut SetupMatrixEnvelope,
    slot: &SetupPrefixSlotId,
) -> Result<(), AkitaError> {
    let n_prefix = slot.n_prefix()?;
    let prefix_ring_len = n_prefix.checked_div(slot.d_setup).ok_or_else(|| {
        AkitaError::InvalidSetup("setup-prefix slot has invalid padded length".to_string())
    })?;
    let params = &slot.commitment_params;
    envelope.max_setup_len = envelope.max_setup_len.max(prefix_ring_len);
    include_matrix(
        envelope,
        params.a_key.row_len(),
        params.inner_width(),
        "setup-prefix A",
    )?;
    include_matrix(
        envelope,
        params.b_key.row_len(),
        params.outer_width(),
        "setup-prefix B",
    )
}
