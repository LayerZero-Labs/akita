//! Field profiles and checked roots used by the mixed-radix FFT.

#[cfg(feature = "labinius-trinomial")]
use jolt_field::{solinas::Fp64, Ext2Config, FpExt2, Ring};

use super::SmoothFftField;

/// Modulus `2^64 - 23_703`, whose multiplicative group contains the
/// `2^3 * 3^5 = 1_944` subgroup needed by the initial LaBinius tower.
#[cfg(feature = "labinius-trinomial")]
pub const PRIME64_OFFSET_23703_MODULUS: u64 = 18_446_744_073_709_527_913;

/// The 64-bit coefficient field `2^64 - 23_703`.
///
/// This local name selects an existing generic `jolt-field` implementation; it
/// does not add a parallel field representation.
#[cfg(feature = "labinius-trinomial")]
pub type Prime64Offset23703 = Fp64<PRIME64_OFFSET_23703_MODULUS>;

/// Quadratic-extension configuration `u^2 = 5` for
/// [`Prime64Offset23703`].
///
/// Five is a quadratic non-residue for this `9 mod 16` prime. The default
/// `jolt-field` choice `u^2 = 2` is reducible here and must not be used.
#[cfg(feature = "labinius-trinomial")]
pub struct Prime64Offset23703Nr5;

#[cfg(feature = "labinius-trinomial")]
impl Ext2Config<Prime64Offset23703> for Prime64Offset23703Nr5 {
    fn non_residue() -> Prime64Offset23703 {
        Prime64Offset23703::from_u64(5)
    }
}

/// Valid quadratic extension of [`Prime64Offset23703`] with basis `[1, u]`
/// and `u^2 = 5`.
#[cfg(feature = "labinius-trinomial")]
pub type Prime64Offset23703Ext2 = FpExt2<Prime64Offset23703, Prime64Offset23703Nr5>;

#[cfg(feature = "labinius-trinomial")]
impl SmoothFftField for Prime64Offset23703 {
    const SMOOTH_SUBGROUP_ORDER: usize = 1_944;
    /// Independently checked to have exact order `1_944` modulo
    /// `2^64 - 23_703` by this module's tests.
    const SMOOTH_OMEGA: u128 = 15_113_553_820_101_191_969;
}

#[cfg(all(test, feature = "labinius-trinomial"))]
mod tests;
