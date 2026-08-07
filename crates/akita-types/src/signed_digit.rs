//! Canonical signed-digit storage selection.

/// Smallest supported signed-digit decomposition basis exponent.
pub const MIN_SIGNED_DIGIT_LOG_BASIS: u32 = 1;

/// Largest basis exponent whose balanced digits fit in signed `i8` storage.
pub const MAX_I8_LOG_BASIS: u32 = i8::BITS;

/// Largest basis exponent whose balanced digits fit in signed `i16` storage.
pub const MAX_I16_LOG_BASIS: u32 = i16::BITS;

/// Integer storage used by a supported balanced signed-digit kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignedDigitKernel {
    /// Balanced digits stored in `i8` coefficients.
    I8,
    /// Balanced digits stored in `i16` coefficients.
    I16,
}

impl SignedDigitKernel {
    /// Select the canonical storage kernel for `log_basis`.
    pub const fn for_log_basis(log_basis: u32) -> Option<Self> {
        if log_basis < MIN_SIGNED_DIGIT_LOG_BASIS {
            None
        } else if log_basis <= MAX_I8_LOG_BASIS {
            Some(Self::I8)
        } else if log_basis <= MAX_I16_LOG_BASIS {
            Some(Self::I16)
        } else {
            None
        }
    }

    /// Largest supported basis exponent for this storage kernel.
    pub const fn max_log_basis(self) -> u32 {
        match self {
            Self::I8 => MAX_I8_LOG_BASIS,
            Self::I16 => MAX_I16_LOG_BASIS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_selection_covers_exact_storage_boundaries() {
        assert_eq!(SignedDigitKernel::for_log_basis(0), None);
        assert_eq!(
            SignedDigitKernel::for_log_basis(1),
            Some(SignedDigitKernel::I8)
        );
        assert_eq!(
            SignedDigitKernel::for_log_basis(8),
            Some(SignedDigitKernel::I8)
        );
        assert_eq!(
            SignedDigitKernel::for_log_basis(9),
            Some(SignedDigitKernel::I16)
        );
        assert_eq!(
            SignedDigitKernel::for_log_basis(16),
            Some(SignedDigitKernel::I16)
        );
        assert_eq!(SignedDigitKernel::for_log_basis(17), None);
    }
}
