//! Coefficient fields of the trinomial transform and their smooth roots.

use jolt_algebra::fft::{field_pow, SmoothFftField};
use jolt_algebra::{
    CanonicalEncoding, Ext2Config, Field, Fp64, FpExt2, Prime128OffsetA7F7, PseudoMersenne, Ring,
};

use super::TrinomialError;

/// A coefficient field with a declared smooth multiplicative subgroup.
///
/// Akita-local stand-in for `jolt_algebra::fft::SmoothFftField`: Akita cannot
/// implement that trait for [`Prime64Offset23703`] (both are foreign), so the
/// trinomial transform reads its roots through this trait until
/// `jolt-algebra` carries that field's impl.
pub trait TrinomialField: Field + CanonicalEncoding + PseudoMersenne {
    /// Order of the supported smooth multiplicative subgroup.
    const SMOOTH_SUBGROUP_ORDER: usize;

    /// Canonical representation of a generator of that subgroup.
    const SMOOTH_OMEGA: u128;

    /// The primitive `n`-th root `SMOOTH_OMEGA^(SMOOTH_SUBGROUP_ORDER / n)`.
    ///
    /// # Errors
    ///
    /// [`TrinomialError::UnsupportedRootOrder`] if `n` is zero or does not
    /// divide [`Self::SMOOTH_SUBGROUP_ORDER`], or if `SMOOTH_OMEGA` is not
    /// canonical.
    fn primitive_root(n: usize) -> Result<Self, TrinomialError> {
        let unsupported = TrinomialError::UnsupportedRootOrder {
            required: n,
            available: Self::SMOOTH_SUBGROUP_ORDER,
        };
        if n == 0 || !Self::SMOOTH_SUBGROUP_ORDER.is_multiple_of(n) {
            return Err(unsupported);
        }
        let omega = Self::from_u128_checked(Self::SMOOTH_OMEGA).ok_or(unsupported)?;
        Ok(field_pow(omega, (Self::SMOOTH_SUBGROUP_ORDER / n) as u64))
    }
}

impl TrinomialField for Prime128OffsetA7F7 {
    const SMOOTH_SUBGROUP_ORDER: usize = <Self as SmoothFftField>::SMOOTH_SUBGROUP_ORDER;
    const SMOOTH_OMEGA: u128 = <Self as SmoothFftField>::SMOOTH_OMEGA;
}

/// Modulus `2^64 - 23_703`, whose multiplicative group contains the
/// `2^3 * 3^5 = 1_944` subgroup needed by the initial LaBinius tower.
pub const PRIME64_OFFSET_23703_MODULUS: u64 = 18_446_744_073_709_527_913;

/// The 64-bit coefficient field `2^64 - 23_703`.
///
/// This local name selects an existing generic `jolt-algebra` implementation;
/// it does not add a parallel field representation.
pub type Prime64Offset23703 = Fp64<PRIME64_OFFSET_23703_MODULUS>;

/// Quadratic-extension configuration `u^2 = 5` for
/// [`Prime64Offset23703`].
///
/// Five is a quadratic non-residue for this `9 mod 16` prime. The default
/// `jolt-algebra` choice `u^2 = 2` is reducible here and must not be used.
pub struct Prime64Offset23703Nr5;

impl Ext2Config<Prime64Offset23703> for Prime64Offset23703Nr5 {
    fn non_residue() -> Prime64Offset23703 {
        Prime64Offset23703::from_u64(5)
    }
}

/// Valid quadratic extension of [`Prime64Offset23703`] with basis `[1, u]`
/// and `u^2 = 5`.
pub type Prime64Offset23703Ext2 = FpExt2<Prime64Offset23703, Prime64Offset23703Nr5>;

impl TrinomialField for Prime64Offset23703 {
    const SMOOTH_SUBGROUP_ORDER: usize = 1_944;
    /// Independently checked to have exact order `1_944` modulo
    /// `2^64 - 23_703` by this module's tests.
    const SMOOTH_OMEGA: u128 = 15_113_553_820_101_191_969;
}

#[cfg(test)]
mod tests;
