use akita_error::AkitaError;
use akita_types::{CommittedGroupParams, SetupMatrixCapacity, TerminalCommittedGroupParams};

pub(crate) fn level_setup_field_elements(
    params: &CommittedGroupParams,
) -> Result<usize, AkitaError> {
    let mut field_elements = SetupMatrixCapacity::minimum().num_field_elements;
    akita_types::accumulate_matrix_field_elements_for_level(params, &mut field_elements)?;
    Ok(field_elements)
}

pub(crate) fn terminal_setup_field_elements(
    params: &TerminalCommittedGroupParams,
) -> Result<usize, AkitaError> {
    let mut field_elements = SetupMatrixCapacity::minimum().num_field_elements;
    akita_types::accumulate_terminal_matrix_field_elements(params, &mut field_elements)?;
    Ok(field_elements)
}
