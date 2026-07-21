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
        lp.inner_commit_matrix.output_rank(),
        lp.inner_width(),
        "A setup",
    )?;
    include_matrix_len(
        max_setup_len,
        lp.outer_commit_matrix.output_rank(),
        lp.outer_width(),
        "B setup",
    )?;
    include_matrix_len(
        max_setup_len,
        lp.open_commit_matrix.output_rank(),
        lp.d_matrix_width(),
        "D setup",
    )?;
    for group in &lp.precommitted_groups {
        include_matrix_len(
            max_setup_len,
            group.inner_commit_matrix.output_rank(),
            group.inner_width(),
            "precommitted A setup",
        )?;
        include_matrix_len(
            max_setup_len,
            group.outer_commit_matrix.output_rank(),
            group.outer_width(),
            "precommitted B setup",
        )?;
    }
    if let Some(slot) = &lp.setup_prefix {
        let mut envelope = SetupMatrixEnvelope {
            max_setup_len: *max_setup_len,
        };
        inflate_envelope_for_setup_prefix_slot(&mut envelope, slot)?;
        *max_setup_len = envelope.max_setup_len;
    }
    Ok(())
}

fn include_matrix_len(
    max_setup_len: &mut usize,
    rows: usize,
    columns: usize,
    role: &'static str,
) -> Result<(), AkitaError> {
    let len = rows
        .checked_mul(columns)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} envelope overflow")))?;
    *max_setup_len = (*max_setup_len).max(len);
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

/// Include the padded prefix storage and A/B footprints for one setup-prefix slot.
///
/// The D footprint is not slot-local: it uses the consuming fold's shared
/// `open_commit_matrix` over the main group plus every precommitted/setup-prefix `e_hat`
/// segment, and is accounted for by `accumulate_matrix_envelope_for_level`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the slot shape overflows `usize` or
/// has an invalid setup dimension.
pub(crate) fn inflate_envelope_for_setup_prefix_slot(
    envelope: &mut SetupMatrixEnvelope,
    slot: &SetupPrefixSlotId,
) -> Result<(), AkitaError> {
    let n_prefix = slot.n_prefix()?;
    if slot.d_setup == 0 || !n_prefix.is_multiple_of(slot.d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot has invalid setup dimension".to_string(),
        ));
    }
    let prefix_ring_len = n_prefix / slot.d_setup;
    let params = &slot.commitment_params;
    envelope.max_setup_len = envelope.max_setup_len.max(prefix_ring_len);
    include_matrix(
        envelope,
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        "setup-prefix A",
    )?;
    include_matrix(
        envelope,
        params.outer_commit_matrix.output_rank(),
        params.outer_width(),
        "setup-prefix B",
    )
}
