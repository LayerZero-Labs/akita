//! Canonical setup-matrix field-capacity accounting.

use akita_field::AkitaError;

use super::setup_prefix::{active_setup_field_len, suffix_opening_layout};
use crate::{
    CommittedGroupParams, FoldSchedule, OpeningClaimsLayout, SetupMatrixCapacity,
    SetupPrefixSlotId, TerminalCommittedGroupParams,
};

/// Compute the exact maximum reusable setup-matrix field prefix required by
/// `schedule`.
pub fn setup_matrix_capacity_for_schedule(
    schedule: &FoldSchedule,
) -> Result<SetupMatrixCapacity, AkitaError> {
    let num_field_elements = setup_matrix_field_elements_for_schedule(schedule)?;
    Ok(SetupMatrixCapacity { num_field_elements })
}

/// Compute the largest physical base-field footprint of any setup matrix or
/// natural public-matrix prefix used by `schedule`.
///
/// This quantity is independent of a level-local ring dimension and is
/// therefore comparable across mixed-ring levels.
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

/// Compute the exact public-matrix prefix required by a verifier for one
/// resolved schedule.
///
/// A producer whose successor carries an incoming setup-prefix commitment does
/// not require a direct public-matrix scan. The first producer after the
/// offloaded chain does. Terminal commitment verification always requires its
/// exact inner matrix. The returned capacity is the maximum of those direct
/// uses.
pub fn verifier_setup_matrix_capacity_for_schedule(
    schedule: &FoldSchedule,
    root_layout: &OpeningClaimsLayout,
) -> Result<SetupMatrixCapacity, AkitaError> {
    schedule.validate_structure()?;

    let mut num_field_elements = 1usize;
    accumulate_terminal_matrix_field_elements(
        &schedule.terminal.params.witness,
        &mut num_field_elements,
    )?;

    for producer_index in 0..=schedule.recursive_folds.len() {
        let producer_is_offloaded = schedule
            .recursive_folds
            .get(producer_index)
            .is_some_and(|successor| successor.params.incoming_setup_prefix.is_some());
        if producer_is_offloaded {
            continue;
        }

        let direct_fields = if producer_index == 0 {
            active_setup_field_len(&schedule.root.params.final_group.commitment, root_layout)?
        } else {
            let producer = &schedule.recursive_folds[producer_index - 1];
            let incoming_prefix_len = producer
                .params
                .incoming_setup_prefix
                .as_ref()
                .map(|slot| slot.natural_len);
            let layout = suffix_opening_layout(producer.input_witness_len, incoming_prefix_len)?;
            active_setup_field_len(&producer.params.witness, &layout)?
        };
        num_field_elements = num_field_elements.max(direct_fields);
    }

    Ok(SetupMatrixCapacity { num_field_elements })
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

/// Largest physical base-field footprint of a natural setup source prefix or
/// either matrix used to commit its padded protocol object.
pub fn setup_prefix_slot_field_elements(slot: &SetupPrefixSlotId) -> Result<usize, AkitaError> {
    let n_prefix = slot.n_prefix()?;
    if slot.d_setup() == 0 || !n_prefix.is_multiple_of(slot.d_setup()) {
        return Err(AkitaError::InvalidSetup(
            "setup-prefix slot has invalid setup dimension".to_string(),
        ));
    }
    let mut max_field_elements = slot.natural_len;
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
