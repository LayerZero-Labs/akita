//! Proof-field families of the nested Akita opening.
//!
//! A family fixes the base field that holds the committed base-16 digit tables,
//! the challenge field of evaluation points and claimed values, and the Akita
//! configurations that commit digit tables and full-width element tables over
//! that pair. Image and response tables share the digit configuration; the
//! prime left opening uses the element configuration.

use akita_config::{
    proof_optimized::{fp128, fp64},
    CommitmentConfig,
};
use akita_params::SisModulusProfileId;
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Valid};
use akita_types::FpExtEncoding;
use jolt_field::{
    AdditiveGroup, CanonicalEncoding, ExtField, Field, Fold, MulBaseUnreduced, Prime128Offset275,
    PseudoMersenne, Ring, Unreduced, WithCommitAccumulator,
};

pub mod prime;
pub mod tables;

/// One admissible proof-field pair and its table configurations.
///
/// The bounds on the two fields are the union of what Akita's setup, CPU
/// commitment, grouped prover and grouped verifier ask of a base field and its
/// challenge field. They are stated here, on the associated types, so that
/// code generic over a family calls those entry points without restating them.
pub trait FieldFamily: 'static {
    /// Field of the committed digit and element tables.
    type Base: Field
        + CanonicalEncoding
        + Ring
        + Unreduced<Wide: From<Self::Base> + AdditiveGroup>
        + PseudoMersenne
        + WithCommitAccumulator
        + Valid
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + 'static;
    /// Field of Fiat-Shamir challenges, evaluation points and claimed values.
    type Challenge: ExtField<Self::Base>
        + FpExtEncoding<Self::Base>
        + MulBaseUnreduced<Self::Base>
        + Ring
        + Unreduced
        + Fold
        + Valid
        + AkitaSerialize
        + 'static;
    /// Bounded dense Akita configuration for one base-16 digit table.
    type Digits: CommitmentConfig<Field = Self::Base, ExtField = Self::Challenge> + 'static;
    /// Dense Akita configuration for one full-width base-field element table.
    type Elements: CommitmentConfig<Field = Self::Base, ExtField = Self::Challenge> + 'static;
}

/// Signed bit width of the committed-source bound of a digit table.
///
/// A width of `k` declares the centered range `[-2^(k-1), 2^(k-1) - 1]`, so `5`
/// is the smallest declaration that contains the stored alphabet `[0, 15]`.
/// The commitment enforces this range only. The unsigned alphabet is enforced
/// by the reduction's sumcheck.
pub const LOG_DIGIT_COMMIT_BOUND: u32 = 5;

const _: () = assert!(
    (1u32 << (LOG_DIGIT_COMMIT_BOUND - 1)) > 15 && (1u32 << (LOG_DIGIT_COMMIT_BOUND - 2)) <= 15,
    "the digit commit bound must be the smallest signed width containing 15"
);

/// Declares a family and its digit-table and element-table configurations.
///
/// The configuration is the named dense preset with three changes: the source
/// bound is [`LOG_DIGIT_COMMIT_BOUND`], the SIS modulus profile is the exact
/// profile of the family's base field, and the catalog has its own identity.
/// The preset supplies the ring-dimension schedule, the decomposition basis,
/// the field width, both basis ranges, the challenge policy and the source
/// class. The element configuration mirrors the family's full-width dense
/// preset with its own catalog identity and the same SIS modulus profile.
macro_rules! digit_family {
    (
        $(#[$family_doc:meta])* $family:ident,
        $(#[$digits_doc:meta])* $digits:ident,
        $(#[$elements_doc:meta])* $elements:ident,
        base = $base:ty,
        challenge = $challenge:ty,
        preset = $preset:ty,
        element_preset = $element_preset:ty,
        profile = $profile:ident,
        catalog = $catalog:literal,
        element_catalog = $element_catalog:literal
    ) => {
        $(#[$family_doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $family;

        impl FieldFamily for $family {
            type Base = $base;
            type Challenge = $challenge;
            type Digits = $digits;
            type Elements = $elements;
        }

        $(#[$digits_doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $digits;

        impl CommitmentConfig for $digits {
            type Field = $base;
            type ExtField = $challenge;
            const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
                <$preset>::RING_DIMENSION_SCHEDULE;
            fn decomposition() -> akita_params::DecompositionParams {
                let preset = <$preset>::decomposition();
                akita_params::DecompositionParams {
                    log_basis: preset.log_basis,
                    log_commit_bound: LOG_DIGIT_COMMIT_BOUND,
                    // Opening witnesses carry genuine field elements, so a source
                    // bounded below the field width names that width explicitly.
                    log_open_bound: Some(preset.field_bits()),
                }
            }
            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
                <$preset>::ring_challenge_config(d)
            }
            fn sis_modulus_profile() -> SisModulusProfileId {
                SisModulusProfileId::$profile
            }
            fn opening_basis_range() -> (u32, u32) {
                <$preset>::opening_basis_range()
            }
            fn inner_basis_range() -> (u32, u32) {
                <$preset>::inner_basis_range()
            }
            fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
                <$preset>::committed_source_class()
            }
            fn schedule_family_name() -> &'static str {
                $catalog
            }
        }

        $(#[$elements_doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $elements;

        impl CommitmentConfig for $elements {
            type Field = $base;
            type ExtField = $challenge;
            const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
                <$element_preset>::RING_DIMENSION_SCHEDULE;
            fn decomposition() -> akita_params::DecompositionParams {
                let preset = <$element_preset>::decomposition();
                akita_params::DecompositionParams {
                    log_basis: preset.log_basis,
                    log_commit_bound: preset.field_bits(),
                    log_open_bound: None,
                }
            }
            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, akita_error::AkitaError> {
                <$element_preset>::ring_challenge_config(d)
            }
            fn sis_modulus_profile() -> SisModulusProfileId {
                SisModulusProfileId::$profile
            }
            fn opening_basis_range() -> (u32, u32) {
                <$element_preset>::opening_basis_range()
            }
            fn inner_basis_range() -> (u32, u32) {
                <$element_preset>::inner_basis_range()
            }
            fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
                <$element_preset>::committed_source_class()
            }
            fn schedule_family_name() -> &'static str {
                $element_catalog
            }
        }
    };
}

digit_family!(
    /// The 64-bit family: `Prime64Offset59` digits, `Ext2` challenges.
    Family64,
    /// Base-16 digit tables over `Prime64Offset59` with `Ext2` openings,
    /// mirroring `proof_optimized::fp64::Dense`.
    Digits64,
    /// Full-width base-field tables over `Prime64Offset59`, mirroring
    /// `proof_optimized::fp64::Dense`.
    Elements64,
    base = fp64::Field,
    challenge = fp64::ExtensionField,
    preset = fp64::Dense,
    element_preset = fp64::Dense,
    profile = Q64Offset59,
    catalog = "labinius_fp64_digits_b4",
    element_catalog = "labinius_fp64_elements"
);

digit_family!(
    /// The 128-bit family: `Prime128Offset275` digits and challenges.
    Family128,
    /// Base-16 digit tables over `Prime128Offset275`, mirroring
    /// `proof_optimized::fp128::DenseBounded` under the field's own SIS
    /// modulus profile.
    Digits128,
    /// Full-width base-field tables over `Prime128Offset275`, mirroring
    /// `proof_optimized::fp128::Dense` with this family's SIS modulus profile.
    Elements128,
    base = Prime128Offset275,
    challenge = Prime128Offset275,
    preset = fp128::DenseBounded,
    element_preset = fp128::Dense,
    profile = Q128Offset275,
    catalog = "labinius_fp128_digits_b4",
    element_catalog = "labinius_fp128_elements"
);
