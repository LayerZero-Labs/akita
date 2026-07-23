//! Canonical setup-matrix envelope accounting.

use akita_field::AkitaError;

use crate::{
    CommittedGroupParams, FoldSchedule, SetupMatrixEnvelope, SetupPrefixSlotId,
    TerminalCommittedGroupParams,
};

/// Compute the maximum reusable setup-matrix length required by `schedule`.
///
/// Lengths are ring elements at each level's inner ring dimension. Mixed-ring
/// setup offloading is rejected by schedule validation, so one maximum is a
/// canonical storage envelope for every currently supported schedule.
pub fn setup_matrix_envelope_for_schedule(
    schedule: &FoldSchedule,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    let mut envelope = SetupMatrixEnvelope::minimum();
    accumulate_matrix_envelope_for_level(
        &schedule.root.params.final_group.commitment,
        &mut envelope.max_setup_len,
    )?;
    for fold in &schedule.recursive_folds {
        accumulate_matrix_envelope_for_level(&fold.params.witness, &mut envelope.max_setup_len)?;
    }
    accumulate_terminal_matrix_envelope(
        &schedule.terminal.params.witness,
        &mut envelope.max_setup_len,
    )?;
    Ok(envelope)
}

/// Extend `max_setup_len` with one non-terminal level's complete setup footprint.
pub fn accumulate_matrix_envelope_for_level(
    params: &CommittedGroupParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_len(
        max_setup_len,
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
        params.d_a(),
        "inner setup",
    )?;
    include_matrix_len(
        max_setup_len,
        params.outer_commit_matrix.output_rank(),
        params.outer_width(),
        params.outer_commit_matrix.ring_dimension(),
        params.d_a(),
        "outer setup",
    )?;
    include_matrix_len(
        max_setup_len,
        params.open_commit_matrix.output_rank(),
        params.d_matrix_width(),
        params.open_commit_matrix.ring_dimension(),
        params.d_a(),
        "opening setup",
    )?;
    for group in &params.precommitted_groups {
        include_matrix_len(
            max_setup_len,
            group.inner_commit_matrix.output_rank(),
            group.inner_width(),
            group.inner_commit_matrix.ring_dimension(),
            params.d_a(),
            "precommitted inner setup",
        )?;
        include_matrix_len(
            max_setup_len,
            group.outer_commit_matrix.output_rank(),
            group.outer_width(),
            group.outer_commit_matrix.ring_dimension(),
            params.d_a(),
            "precommitted outer setup",
        )?;
    }
    if let Some(slot) = &params.setup_prefix {
        let mut envelope = SetupMatrixEnvelope {
            max_setup_len: *max_setup_len,
        };
        inflate_envelope_for_setup_prefix_slot(&mut envelope, slot, params.d_a())?;
        *max_setup_len = envelope.max_setup_len;
    }
    Ok(())
}

/// Extend `max_setup_len` with the terminal inner matrix footprint.
pub fn accumulate_terminal_matrix_envelope(
    params: &TerminalCommittedGroupParams,
    max_setup_len: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_len(
        max_setup_len,
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
        params.d_a(),
        "terminal inner setup",
    )
}

/// Include the padded prefix and its inner/outer commitment matrices.
pub fn inflate_envelope_for_setup_prefix_slot(
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
    envelope.max_setup_len = envelope
        .max_setup_len
        .max(n_prefix.div_ceil(envelope_ring_dim));
    let params = &slot.commitment_params;
    include_matrix_len(
        &mut envelope.max_setup_len,
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
        envelope_ring_dim,
        "setup-prefix inner setup",
    )?;
    include_matrix_len(
        &mut envelope.max_setup_len,
        params.outer_commit_matrix.output_rank(),
        params.outer_width(),
        params.outer_commit_matrix.ring_dimension(),
        envelope_ring_dim,
        "setup-prefix outer setup",
    )
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
    *max_setup_len = (*max_setup_len).max(coeff_len.div_ceil(envelope_ring_dim));
    Ok(())
}
