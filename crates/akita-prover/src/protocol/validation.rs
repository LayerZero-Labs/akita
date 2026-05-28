use akita_field::AkitaError;

pub(crate) fn validate_i8_log_basis(log_basis: u32) -> Result<(), AkitaError> {
    if (1..=6).contains(&log_basis) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(
            "log_basis must be in 1..=6 for i8 prover decomposition".to_string(),
        ))
    }
}
