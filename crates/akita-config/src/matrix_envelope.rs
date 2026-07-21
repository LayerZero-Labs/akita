//! Shared setup-matrix envelope accumulation helpers.

use akita_field::AkitaError;
use akita_types::{LevelParams, SetupMatrixEnvelope, SetupPrefixSlotId};

/// Extend `max_setup_len` with the full per-level setup footprint.
///
/// Includes the level's own A/B/D matrices, precommitted group-local A/B
/// matrices, and setup-prefix materialization when the level consumes one. The
/// shared D matrix is accounted through `LevelParams::d_matrix_width()`: planner
/// materialization includes every precommitted/setup-prefix D segment in that
/// single width.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] on overflow.
pub(crate) fn accumulate_matrix_envelope_for_level(
    lp: &LevelParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_len(
        max_setup_len,
        lp.a_key.row_len(),
        lp.inner_width(),
        lp.a_key.sis_table_key().ring_dimension as usize,
        lp.d_a(),
        "A setup",
    )?;
    include_matrix_len(
        max_setup_len,
        lp.b_key.row_len(),
        lp.outer_width(),
        lp.b_key.sis_table_key().ring_dimension as usize,
        lp.d_a(),
        "B setup",
    )?;
    include_matrix_len(
        max_setup_len,
        lp.d_key.row_len(),
        lp.d_matrix_width(),
        lp.d_key.sis_table_key().ring_dimension as usize,
        lp.d_a(),
        "D setup",
    )?;
    for group in &lp.precommitted_groups {
        include_matrix_len(
            max_setup_len,
            group.a_key.row_len(),
            group.inner_width(),
            group.a_key.sis_table_key().ring_dimension as usize,
            lp.d_a(),
            "precommitted A setup",
        )?;
        include_matrix_len(
            max_setup_len,
            group.b_key.row_len(),
            group.outer_width(),
            group.b_key.sis_table_key().ring_dimension as usize,
            lp.d_a(),
            "precommitted B setup",
        )?;
    }
    if let Some(slot) = &lp.setup_prefix {
        let mut envelope = SetupMatrixEnvelope {
            max_setup_len: *max_setup_len,
        };
        inflate_envelope_for_setup_prefix_slot(&mut envelope, slot, lp.d_a())?;
        *max_setup_len = envelope.max_setup_len;
    }
    Ok(())
}

fn include_matrix_len(
    max_setup_len: &mut usize,
    rows: usize,
    columns: usize,
    matrix_ring_dim: usize,
    envelope_ring_dim: usize,
    role: &'static str,
) -> Result<(), AkitaError> {
    if envelope_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(format!(
            "{role} envelope ring dimension is zero"
        )));
    }
    let coeff_len = rows
        .checked_mul(columns)
        .and_then(|len| len.checked_mul(matrix_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} envelope overflow")))?;
    let len = coeff_len.div_ceil(envelope_ring_dim);
    *max_setup_len = (*max_setup_len).max(len);
    Ok(())
}

/// Include the padded prefix storage and A/B footprints for one setup-prefix slot.
///
/// The D footprint is not slot-local: it uses the consuming fold's shared
/// `d_key` over the main group plus every precommitted/setup-prefix `e_hat`
/// segment, and is accounted for by `accumulate_matrix_envelope_for_level`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the slot shape overflows `usize` or
/// has an invalid setup dimension.
pub(crate) fn inflate_envelope_for_setup_prefix_slot(
    envelope: &mut SetupMatrixEnvelope,
    slot: &SetupPrefixSlotId,
    envelope_ring_dim: usize,
) -> Result<(), AkitaError> {
    let n_prefix = slot.n_prefix()?;
    if slot.d_setup == 0 || !n_prefix.is_multiple_of(slot.d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot has invalid setup dimension".to_string(),
        ));
    }
    if envelope_ring_dim == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix envelope ring dimension is zero".to_string(),
        ));
    }
    let params = &slot.commitment_params;
    let prefix_ring_len = n_prefix.div_ceil(envelope_ring_dim);
    envelope.max_setup_len = envelope.max_setup_len.max(prefix_ring_len);
    include_matrix_len(
        &mut envelope.max_setup_len,
        params.a_key.row_len(),
        params.inner_width(),
        params.a_key.sis_table_key().ring_dimension as usize,
        envelope_ring_dim,
        "setup-prefix A",
    )?;
    include_matrix_len(
        &mut envelope.max_setup_len,
        params.b_key.row_len(),
        params.outer_width(),
        params.b_key.sis_table_key().ring_dimension as usize,
        envelope_ring_dim,
        "setup-prefix B",
    )
}
