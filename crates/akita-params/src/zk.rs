//! Statistical-hiding randomness sizing for commitment images.

use crate::SisModulusProfileId;
use akita_error::{checked, AkitaError};

/// Statistical security parameter λ for the leftover hash lemma.
pub const HIDING_SECURITY_BITS: usize = 128;

/// Slack bits added on top of `2λ` when sizing hiding randomness.
const HIDING_SLACK_BITS: usize = 8;

/// Further outputs hidden by the same randomness as the commitment image.
///
/// Every caller must select a variant explicitly. There is no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HiddenRows {
    /// Only the published commitment image is hidden.
    None,
    /// One additional row on the randomness columns, reduced to stride
    /// component zero, with 64 output coefficients.
    StrideZero,
}

impl HiddenRows {
    /// Additional matrix rows that must be available in the public prefix.
    #[must_use]
    pub const fn rows(self) -> usize {
        match self {
            Self::None => 0,
            Self::StrideZero => 1,
        }
    }

    /// Additional output coefficients the randomness must hide.
    #[must_use]
    pub const fn coefficients(self) -> usize {
        match self {
            Self::None => 0,
            Self::StrideZero => 64,
        }
    }
}

/// Published image and further outputs hidden by one randomness vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HiddenImage {
    /// Published image rank (output ring elements).
    pub rows: usize,
    /// Ring dimension of the image and of each randomness column.
    pub ring_dim: usize,
    /// Field modulus profile of the image coefficients.
    pub modulus: SisModulusProfileId,
    /// Further outputs the same randomness must hide.
    pub hidden: HiddenRows,
}

/// Uniform balanced alphabet `[-2^(log_basis-1), 2^(log_basis-1) - 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DigitAlphabet {
    /// Base-two logarithm of the digit base.
    pub log_basis: u32,
}

/// Fewest whole randomness ring columns whose entropy is at least
/// `(image.rows * image.ring_dim + image.hidden.coefficients()) * ceil(log₂ q)
/// + 2λ + 8` bits.
///
/// Each column contains `image.ring_dim` independent uniform digits from
/// `alphabet`. Each commitment slice draws its own randomness of this size.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] for a zero ring dimension, an alphabet
/// outside `1..=8`, or an overflowing size.
pub fn hiding_randomness_columns(
    image: HiddenImage,
    alphabet: DigitAlphabet,
) -> Result<usize, AkitaError> {
    if image.ring_dim == 0 || !(1..=8).contains(&alphabet.log_basis) {
        return Err(AkitaError::InvalidSetup(
            "hiding randomness needs ring_dim > 0 and log_basis in 1..=8".into(),
        ));
    }
    let overflow = || AkitaError::InvalidSetup("hiding randomness size overflows usize".into());
    let coefficients = checked::mul_add(image.rows, image.ring_dim, image.hidden.coefficients())
        .ok_or_else(overflow)?;
    let needed = checked::mul_add(
        coefficients,
        image.modulus.field_bits() as usize,
        2 * HIDING_SECURITY_BITS + HIDING_SLACK_BITS,
    )
    .ok_or_else(overflow)?;
    let per_column =
        checked::product([image.ring_dim, alphabet.log_basis as usize]).ok_or_else(overflow)?;
    checked::div_ceil(needed, per_column).ok_or_else(overflow)
}

#[cfg(test)]
mod tests;
