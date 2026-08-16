use akita_field::AkitaError;
use akita_types::{SignedDigitKernel, MAX_I16_LOG_BASIS, MAX_I8_LOG_BASIS};

#[inline]
pub(crate) fn is_i8_log_basis(log_basis: u32) -> bool {
    SignedDigitKernel::for_log_basis(log_basis) == Some(SignedDigitKernel::I8)
}

#[inline]
pub(crate) fn validate_i8_setup_log_basis(log_basis: u32, context: &str) -> Result<(), AkitaError> {
    if is_i8_log_basis(log_basis) {
        Ok(())
    } else {
        Err(AkitaError::InvalidSetup(format!(
            "log_basis must be in 1..={MAX_I8_LOG_BASIS} {context}"
        )))
    }
}

#[inline]
pub(crate) fn signed_digit_kernel_for_setup(
    log_basis: u32,
    context: &str,
) -> Result<SignedDigitKernel, AkitaError> {
    SignedDigitKernel::for_log_basis(log_basis).ok_or_else(|| {
        AkitaError::InvalidSetup(format!(
            "log_basis must be in 1..={MAX_I16_LOG_BASIS} {context}"
        ))
    })
}

#[inline]
pub(crate) fn validate_i8_input_log_basis(log_basis: u32, context: &str) -> Result<(), AkitaError> {
    if is_i8_log_basis(log_basis) {
        Ok(())
    } else {
        Err(AkitaError::InvalidInput(format!(
            "log_basis must be in 1..={MAX_I8_LOG_BASIS} {context}"
        )))
    }
}
