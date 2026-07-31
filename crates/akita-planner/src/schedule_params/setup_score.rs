use akita_field::AkitaError;
use akita_types::{
    CommittedGroupParams, FoldSchedule, SetupMatrixEnvelope, TerminalCommittedGroupParams,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MixedScore {
    pub setup_field_elements: usize,
    pub matrix_field_elements: usize,
    pub proof_bytes: usize,
}

fn matrix_field_elements(rank: usize, width: usize, dimension: usize) -> Result<usize, AkitaError> {
    rank.checked_mul(width)
        .and_then(|value| value.checked_mul(dimension))
        .ok_or_else(|| AkitaError::InvalidSetup("matrix field-element count overflow".into()))
}

pub(crate) fn level_matrix_field_elements(
    params: &CommittedGroupParams,
) -> Result<usize, AkitaError> {
    let inner = matrix_field_elements(
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
    )?;
    let outer = matrix_field_elements(
        params.outer_commit_matrix.output_rank(),
        params.outer_width(),
        params.outer_commit_matrix.ring_dimension(),
    )?;
    let opening = matrix_field_elements(
        params.open_commit_matrix.output_rank(),
        params.d_matrix_width(),
        params.open_commit_matrix.ring_dimension(),
    )?;
    inner
        .checked_add(outer)
        .and_then(|value| value.checked_add(opening))
        .ok_or_else(|| AkitaError::InvalidSetup("level matrix work overflow".into()))
}

pub(crate) fn terminal_matrix_field_elements(
    params: &TerminalCommittedGroupParams,
) -> Result<usize, AkitaError> {
    matrix_field_elements(
        params.inner_commit_matrix.output_rank(),
        params.inner_width(),
        params.inner_commit_matrix.ring_dimension(),
    )
}

/// Sum the physical field coefficients in every commitment matrix of a schedule.
///
/// For the direct schedules admitted by mixed-dimension search, this is the
/// planner's setup-matrix scan proxy for verifier work. It is additive across
/// levels, unlike the reusable setup envelope, which is a maximum.
pub fn schedule_matrix_field_elements(schedule: &FoldSchedule) -> Result<usize, AkitaError> {
    let mut total = level_matrix_field_elements(&schedule.root.params.final_group.commitment)?;
    for fold in &schedule.recursive_folds {
        total = total
            .checked_add(level_matrix_field_elements(&fold.params.witness)?)
            .ok_or_else(|| AkitaError::InvalidSetup("schedule matrix work overflow".into()))?;
    }
    total
        .checked_add(terminal_matrix_field_elements(
            &schedule.terminal.params.witness,
        )?)
        .ok_or_else(|| AkitaError::InvalidSetup("schedule matrix work overflow".into()))
}

pub(crate) fn level_setup_field_elements(
    params: &CommittedGroupParams,
) -> Result<usize, AkitaError> {
    let mut field_elements = SetupMatrixEnvelope::minimum().max_setup_len;
    akita_types::accumulate_matrix_field_elements_for_level(params, &mut field_elements)?;
    Ok(field_elements)
}

pub(crate) fn terminal_setup_field_elements(
    params: &TerminalCommittedGroupParams,
) -> Result<usize, AkitaError> {
    let mut field_elements = SetupMatrixEnvelope::minimum().max_setup_len;
    akita_types::accumulate_terminal_matrix_field_elements(params, &mut field_elements)?;
    Ok(field_elements)
}

#[cfg(test)]
mod tests {
    use super::MixedScore;

    #[test]
    fn exact_setup_fields_precede_proof_bytes() {
        let generation_dimension = 256;
        let smaller_setup = MixedScore {
            setup_field_elements: generation_dimension + 1,
            matrix_field_elements: 20_000,
            proof_bytes: 10_000,
        };
        let larger_setup = MixedScore {
            setup_field_elements: 2 * generation_dimension - 1,
            matrix_field_elements: 10_000,
            proof_bytes: 1,
        };

        assert_eq!(
            smaller_setup
                .setup_field_elements
                .div_ceil(generation_dimension),
            larger_setup
                .setup_field_elements
                .div_ceil(generation_dimension)
        );
        assert!(smaller_setup < larger_setup);
    }
}
