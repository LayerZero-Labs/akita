//! Ordinary bounded dense Akita configurations for response digits.

use crate::F;
use akita_config::{proof_optimized::fp128::DenseBounded, CommitmentConfig};
use akita_params::sis::labinius::LabiniusDigitBase;

/// A response configuration tied to the reduction's digit base.
///
/// The signed commitment bound contains honest digits. The reduction's
/// sumcheck, rather than the commitment, enforces the unsigned alphabet.
pub trait DigitConfig: CommitmentConfig<Field = F, ExtField = F> {
    const BASE: LabiniusDigitBase;
}

macro_rules! digit_config {
    ($name:ident, $base:ident, $bits:literal, $family:literal) => {
        #[doc = concat!("Dense response digits with a signed ", stringify!($bits), "-bit source bound.")]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl DigitConfig for $name {
            const BASE: LabiniusDigitBase = LabiniusDigitBase::$base;
        }
        impl CommitmentConfig for $name {
            type Field = F;
            type ExtField = F;
            const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
                DenseBounded::RING_DIMENSION_SCHEDULE;
            fn decomposition() -> akita_params::DecompositionParams {
                let mut decomposition = DenseBounded::decomposition();
                decomposition.log_commit_bound = $bits;
                decomposition.log_open_bound = Some(128);
                decomposition
            }
            fn ring_challenge_config(d: usize) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
                DenseBounded::ring_challenge_config(d)
            }
            fn sis_modulus_profile() -> akita_params::SisModulusProfileId {
                DenseBounded::sis_modulus_profile()
            }
            fn opening_basis_range() -> (u32, u32) {
                DenseBounded::opening_basis_range()
            }
            fn inner_basis_range() -> (u32, u32) {
                DenseBounded::inner_basis_range()
            }
            fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
                DenseBounded::committed_source_class()
            }
            fn schedule_family_name() -> &'static str { $family }
        }
    };
}
digit_config!(Digits1, Bits1, 2, "labinius_digits_b1");
digit_config!(Digits2, Bits2, 3, "labinius_digits_b2");
digit_config!(Digits4, Bits4, 5, "labinius_digits_b4");
