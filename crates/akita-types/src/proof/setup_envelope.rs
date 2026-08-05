//! Canonical setup-matrix envelope accounting.

use akita_field::AkitaError;

use crate::{
    CommittedGroupParams, FoldSchedule, SetupMatrixEnvelope, SetupPrefixSlotId,
    TerminalCommittedGroupParams,
};

/// Compute the maximum reusable setup-matrix length required by `schedule` at
/// one explicit setup-generation dimension.
pub fn setup_matrix_envelope_for_schedule(
    schedule: &FoldSchedule,
    generation_ring_dimension: usize,
) -> Result<SetupMatrixEnvelope, AkitaError> {
    if generation_ring_dimension == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup generation dimension must be nonzero".into(),
        ));
    }
    let field_elements = setup_matrix_field_elements_for_schedule(schedule)?;
    Ok(SetupMatrixEnvelope {
        max_setup_len: field_elements.div_ceil(generation_ring_dimension),
    })
}

/// Compute the largest physical base-field footprint of any setup matrix or
/// padded setup prefix used by `schedule`.
///
/// Unlike [`setup_matrix_envelope_for_schedule`], this quantity is independent
/// of a level-local A dimension and is therefore comparable across mixed-ring
/// levels. Setup generation converts it to ring elements exactly once with
/// `field_elements.div_ceil(generation_ring_dimension)`.
pub fn setup_matrix_field_elements_for_schedule(
    schedule: &FoldSchedule,
) -> Result<usize, AkitaError> {
    let mut max_field_elements = 1;
    accumulate_matrix_field_elements_for_level(
        &schedule.root.params.final_group.commitment,
        &mut max_field_elements,
    )?;
    for fold in &schedule.recursive_folds {
        accumulate_matrix_field_elements_for_level(&fold.params.witness, &mut max_field_elements)?;
    }
    accumulate_terminal_matrix_field_elements(
        &schedule.terminal.params.witness,
        &mut max_field_elements,
    )?;
    Ok(max_field_elements)
}

/// Extend a physical setup footprint with one non-terminal level.
pub fn accumulate_matrix_field_elements_for_level(
    params: &CommittedGroupParams,
    max_field_elements: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_field_elements(
        max_field_elements,
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
        "inner setup",
    )?;
    include_matrix_field_elements(
        max_field_elements,
        params.outer_commit_matrix.output_rank(),
        params.outer_width(),
        params.outer_commit_matrix.ring_dimension(),
        "outer setup",
    )?;
    include_matrix_field_elements(
        max_field_elements,
        params.open_commit_matrix.output_rank(),
        params.d_matrix_width(),
        params.open_commit_matrix.ring_dimension(),
        "opening setup",
    )?;
    for group in &params.precommitted_groups {
        include_matrix_field_elements(
            max_field_elements,
            group.layout.inner_commit_matrix.output_rank(),
            group.inner_width(),
            group.layout.inner_commit_matrix.ring_dimension(),
            "precommitted inner setup",
        )?;
        include_matrix_field_elements(
            max_field_elements,
            group.layout.outer_commit_matrix.output_rank(),
            group.outer_width(),
            group.layout.outer_commit_matrix.ring_dimension(),
            "precommitted outer setup",
        )?;
    }
    if let Some(slot) = &params.setup_prefix {
        *max_field_elements = (*max_field_elements).max(setup_prefix_slot_field_elements(slot)?);
    }
    Ok(())
}

/// Extend a physical setup footprint with the terminal inner matrix.
pub fn accumulate_terminal_matrix_field_elements(
    params: &TerminalCommittedGroupParams,
    max_field_elements: &mut usize,
) -> Result<(), AkitaError> {
    include_matrix_field_elements(
        max_field_elements,
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
        "terminal inner setup",
    )
}

/// Largest physical base-field footprint of a padded setup prefix or either
/// matrix used to commit it.
pub fn setup_prefix_slot_field_elements(slot: &SetupPrefixSlotId) -> Result<usize, AkitaError> {
    let n_prefix = slot.n_prefix()?;
    if slot.d_setup == 0 || !n_prefix.is_multiple_of(slot.d_setup) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot has invalid setup dimension".to_string(),
        ));
    }
    let mut max_field_elements = n_prefix;
    let params = &slot.commitment_params;
    include_matrix_field_elements(
        &mut max_field_elements,
        params.layout.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.layout.inner_commit_matrix.ring_dimension(),
        "setup-prefix inner setup",
    )?;
    include_matrix_field_elements(
        &mut max_field_elements,
        params.layout.outer_commit_matrix.output_rank(),
        params.outer_width(),
        params.layout.outer_commit_matrix.ring_dimension(),
        "setup-prefix outer setup",
    )?;
    Ok(max_field_elements)
}

fn include_matrix_field_elements(
    max_field_elements: &mut usize,
    rows: usize,
    columns: usize,
    matrix_ring_dim: usize,
    role: &'static str,
) -> Result<(), AkitaError> {
    let field_elements = rows
        .checked_mul(columns)
        .and_then(|len| len.checked_mul(matrix_ring_dim))
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} envelope overflow")))?;
    *max_field_elements = (*max_field_elements).max(field_elements);
    Ok(())
}
