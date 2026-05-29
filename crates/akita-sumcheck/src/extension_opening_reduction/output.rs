use super::*;

/// Check the final extension-opening reduction equality.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if the final sumcheck claim does not
/// match the product of the ordinary witness opening and transparent factor
/// evaluation at the sumcheck challenge point.
pub fn check_extension_opening_reduction_output<E: FieldCore>(
    final_claim: E,
    witness_eval: E,
    factor_eval: E,
) -> Result<(), AkitaError> {
    if final_claim != witness_eval * factor_eval {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}

pub(crate) fn validate_reduction_tables<E: FieldCore>(
    witness_evals: &[E],
    factor_evals: &[E],
) -> Result<(), AkitaError> {
    if witness_evals.len() != factor_evals.len() {
        return Err(AkitaError::InvalidSize {
            expected: witness_evals.len(),
            actual: factor_evals.len(),
        });
    }
    num_rounds_from_table_len(witness_evals.len()).map(|_| ())
}

pub(crate) fn checked_table_len(num_vars: usize) -> Result<usize, AkitaError> {
    if num_vars >= usize::BITS as usize {
        return Err(AkitaError::InvalidInput(format!(
            "extension-opening reduction table has too many variables: {num_vars}"
        )));
    }
    Ok(1usize << num_vars)
}

pub(crate) fn num_rounds_from_table_len(len: usize) -> Result<usize, AkitaError> {
    if len == 0 || !len.is_power_of_two() {
        return Err(AkitaError::InvalidSize {
            expected: len.max(1).next_power_of_two(),
            actual: len,
        });
    }
    Ok(len.trailing_zeros() as usize)
}
