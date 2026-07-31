use akita_field::AkitaError;
use akita_types::{CommittedGroupParams, SetupMatrixCapacity, TerminalCommittedGroupParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MixedScore {
    pub setup_field_elements: usize,
    pub proof_bytes: usize,
}

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

#[cfg(test)]
mod tests {
    use super::MixedScore;

    #[test]
    fn exact_setup_fields_precede_proof_bytes() {
        let generation_dimension = 256;
        let smaller_setup = MixedScore {
            setup_field_elements: generation_dimension + 1,
            proof_bytes: 10_000,
        };
        let larger_setup = MixedScore {
            setup_field_elements: 2 * generation_dimension - 1,
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
