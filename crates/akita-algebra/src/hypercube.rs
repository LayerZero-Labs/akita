//! Padded boolean-hypercube geometry shared by multilinear evaluators.

use akita_field::AkitaError;

/// Live length and power-of-two padding for one multilinear axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedHypercube {
    /// Number of live entries along this axis.
    pub live_len: usize,
    /// Padded hypercube side length (`2^log_len`).
    pub padded_len: usize,
    /// `log2(padded_len)`.
    pub log_len: usize,
}

impl PaddedHypercube {
    /// Build padding metadata for a non-zero live axis length.
    ///
    /// # Errors
    ///
    /// Returns an error if `live_len` is zero or the padded length overflows.
    pub fn from_live_len(live_len: usize) -> Result<Self, AkitaError> {
        if live_len == 0 {
            return Err(AkitaError::InvalidInput(
                "hypercube axis requires a non-zero live length".to_string(),
            ));
        }
        let padded_len = live_len
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidInput("hypercube padding overflows".to_string()))?;
        Ok(Self {
            live_len,
            padded_len,
            log_len: padded_len.trailing_zeros() as usize,
        })
    }
}
