//! Coefficient fields of the trinomial transform and their smooth roots.

use jolt_algebra::fft::field_pow;
use jolt_algebra::{CanonicalEncoding, Ext2Config, Field, Fp64, FpExt2, PseudoMersenne, Ring};

use super::TrinomialError;

/// A coefficient field with a declared smooth multiplicative subgroup.
///
/// Akita-local bridge for coefficient fields not covered by Jolt’s
/// `SmoothFftField`. The shared trait and field types are foreign, so the
/// trinomial transform supplies their root configuration through this trait
/// while using Jolt’s generic FFT implementation.
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

// Main #387's subgroup configuration, ported onto preview's local root trait.
// Jolt owns SmoothFftField, so its implementation for this foreign field cannot
// be supplied here. The existing generic SmoothDomain accepts these roots.
impl TrinomialField for jolt_algebra::Prime128Offset6A23801F {
    const SMOOTH_SUBGROUP_ORDER: usize = 337_500_000;
    const SMOOTH_OMEGA: u128 = 0x42e2_bddd_f9d0_2211_91f7_d7dc_1549_8d96;
}

/// Smooth auxiliary field for exact integer convolution and trinomial transforms.
///
/// This is not Akita's protocol field. Its 3^7 subgroup is needed for the
/// degree-162/486 scalar and degree-648 commitment transforms; the protocol
/// field's 3^3 subgroup cannot replace it. Exact centered lifts below enforce
/// the integer-convolution bound before mapping results into the proof field.
pub type TrinomialAuxiliary = jolt_algebra::Fp128<{ u128::MAX - 4_294_944_758 }>;

// Jolt dropped its smooth-FFT configuration for this modulus when it retired
// the field, so the subgroup is declared here. `2^3 * 3^7 = 17_496`; the
// generator is `g^((p - 1) / 17_496)` for the smallest primitive root `g`,
// as Jolt's former `SmoothFftField` impl recorded it.
impl TrinomialField for TrinomialAuxiliary {
    const SMOOTH_SUBGROUP_ORDER: usize = 17_496;
    const SMOOTH_OMEGA: u128 = 0x4e9f_650b_7003_d201_9945_e1da_c47c_8b18;
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
