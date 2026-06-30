//! Map Akita ring table coordinates to scalar SIS parameters.

use num_bigint::BigUint;

use crate::{
    error::{EstimatorError, Result},
    params::{akita_q128, akita_q32, akita_q64, Bound, SisNorm, SisParameters},
};

/// Supported Akita modulus families for golden and table generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AkitaModulusFamily {
    /// `2^32 - 99`.
    Q32,
    /// `2^64 - 59`.
    Q64,
    /// `2^128 - (2^32 - 22537)`.
    Q128,
}

impl AkitaModulusFamily {
    /// Parse a family label such as `"q32"`.
    ///
    /// # Errors
    ///
    /// Returns an error when the label is unknown.
    pub fn parse(label: &str) -> Result<Self> {
        match label {
            "q32" => Ok(Self::Q32),
            "q64" => Ok(Self::Q64),
            "q128" => Ok(Self::Q128),
            _ => Err(EstimatorError::InvalidParameter {
                field: "family",
                reason: format!("unknown Akita modulus family {label:?}"),
            }),
        }
    }

    /// Return the representative modulus for this family.
    #[must_use]
    pub fn modulus(self) -> BigUint {
        match self {
            Self::Q32 => akita_q32(),
            Self::Q64 => akita_q64(),
            Self::Q128 => akita_q128(),
        }
    }

    /// Return the stable lowercase table label for this family.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Q32 => "q32",
            Self::Q64 => "q64",
            Self::Q128 => "q128",
        }
    }
}

/// Build scalar SIS parameters from Akita ring coordinates.
///
/// Uses the same mapping as `scripts/sis_golden/infinity_core.py`:
/// `n = rank * d`, `m = width * d`, and `length_bound = coeff_linf_bound`.
///
/// # Errors
///
/// Returns validation errors when any dimension is zero.
pub fn scalar_sis_from_ring(
    family: AkitaModulusFamily,
    ring_dimension: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
) -> Result<SisParameters> {
    scalar_sis_from_ring_wide(
        family,
        ring_dimension,
        rank,
        u64::from(width),
        coeff_linf_bound,
    )
}

/// Build scalar SIS parameters from Akita ring coordinates with a wide width.
///
/// Uses the same mapping as [`scalar_sis_from_ring`], but accepts planner-scale
/// ring widths whose scalar column count exceeds `u32::MAX`.
///
/// # Errors
///
/// Returns validation errors when any dimension is zero or the scalar row count
/// overflows `u32`.
pub fn scalar_sis_from_ring_wide(
    family: AkitaModulusFamily,
    ring_dimension: u32,
    rank: u32,
    width: u64,
    coeff_linf_bound: u64,
) -> Result<SisParameters> {
    if ring_dimension == 0 || rank == 0 || width == 0 || coeff_linf_bound == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "ring",
            reason: "ring dimension, rank, width, and coeff_linf_bound must be positive"
                .to_string(),
        });
    }
    let n = rank
        .checked_mul(ring_dimension)
        .ok_or(EstimatorError::InvalidParameter {
            field: "n",
            reason: "rank * ring_dimension overflowed u32".to_string(),
        })?;
    let m =
        width
            .checked_mul(u64::from(ring_dimension))
            .ok_or(EstimatorError::InvalidParameter {
                field: "m",
                reason: "width * ring_dimension overflowed u64".to_string(),
            })?;
    SisParameters::try_new(
        n,
        family.modulus(),
        Some(m),
        Bound::from_u64(coeff_linf_bound),
        SisNorm::Infinity,
    )
}

/// Build Euclidean scalar SIS parameters from Akita ring coordinates.
///
/// This is the mapping used by the shipped L2 table:
/// `n = rank * d`, `m = width * d`, and
/// `length_bound = sqrt(width * collision_l2_sq)`.
///
/// # Errors
///
/// Returns validation errors when any dimension is zero or a scalar dimension
/// overflows.
pub fn scalar_sis_from_ring_euclidean(
    family: AkitaModulusFamily,
    ring_dimension: u32,
    rank: u32,
    width: u64,
    collision_l2_sq: u128,
) -> Result<SisParameters> {
    if ring_dimension == 0 || rank == 0 || width == 0 || collision_l2_sq == 0 {
        return Err(EstimatorError::InvalidParameter {
            field: "ring",
            reason: "ring dimension, rank, width, and collision_l2_sq must be positive".to_string(),
        });
    }
    let n = rank
        .checked_mul(ring_dimension)
        .ok_or(EstimatorError::InvalidParameter {
            field: "n",
            reason: "rank * ring_dimension overflowed u32".to_string(),
        })?;
    let m =
        width
            .checked_mul(u64::from(ring_dimension))
            .ok_or(EstimatorError::InvalidParameter {
                field: "m",
                reason: "width * ring_dimension overflowed u64".to_string(),
            })?;
    let radicand = BigUint::from(width) * BigUint::from(collision_l2_sq);
    SisParameters::try_new(
        n,
        family.modulus(),
        Some(m),
        Bound::sqrt_integer(radicand),
        SisNorm::Euclidean,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_mapping_matches_golden_convention() {
        let params = scalar_sis_from_ring(AkitaModulusFamily::Q32, 32, 1, 2, 15).unwrap();
        assert_eq!(params.n, 32);
        assert_eq!(params.m, Some(64));
        assert_eq!(params.length_bound, Bound::from_u64(15));
    }

    #[test]
    fn euclidean_scalar_mapping_uses_exact_l2_bound() {
        let params =
            scalar_sis_from_ring_euclidean(AkitaModulusFamily::Q32, 32, 1, 2, 128).unwrap();
        assert_eq!(params.n, 32);
        assert_eq!(params.m, Some(64));
        assert_eq!(
            params.length_bound,
            Bound::sqrt_integer(BigUint::from(256u32))
        );
        assert_eq!(params.norm, SisNorm::Euclidean);
    }
}
