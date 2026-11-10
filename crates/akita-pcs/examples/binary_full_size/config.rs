//! Ordinary suffix configurations for the two binary root fields.
//!
//! The binary root's outgoing witness is a table of balanced base-eight
//! digits, so both suffixes commit five-bit signed sources and open full
//! field elements; every other setting comes from the field's dense preset.

/// The 128-bit root: base field P = 2^128 - 275 with base-field challenges.
pub(super) mod p128 {
    use akita_challenges::FoldChallengeDistribution;
    use akita_config::{proof_optimized::fp128::DenseBounded, CommitmentConfig};
    use akita_error::AkitaError;
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_params::{sis::CommittedSourceClass, DecompositionParams, SisModulusProfileId};

    pub(crate) type F = jolt_field::Prime128Offset275;
    pub(crate) type E = F;
    pub(crate) const FIELD: BinaryFieldProfile = BinaryFieldProfile::P128;
    pub(crate) const NAME: &str = "p128";

    /// Ordinary suffix for an explicitly selected experimental binary fixture.
    #[derive(Clone)]
    pub(crate) struct FullSize;

    impl CommitmentConfig for FullSize {
        type Field = F;
        type ExtField = E;
        const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
            DenseBounded::RING_DIMENSION_SCHEDULE;

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: DenseBounded::decomposition().log_basis,
                log_commit_bound: 5,
                log_open_bound: Some(128),
            }
        }
        fn ring_challenge_config(
            dimension: usize,
        ) -> Result<FoldChallengeDistribution, AkitaError> {
            DenseBounded::ring_challenge_config(dimension)
        }
        fn sis_modulus_profile() -> SisModulusProfileId {
            SisModulusProfileId::Q128Offset275
        }
        fn opening_basis_range() -> (u32, u32) {
            DenseBounded::opening_basis_range()
        }
        fn inner_basis_range() -> (u32, u32) {
            DenseBounded::inner_basis_range()
        }
        fn committed_source_class() -> CommittedSourceClass {
            DenseBounded::committed_source_class()
        }
        fn recursive_setup_planning() -> bool {
            true
        }
        fn schedule_family_name() -> &'static str {
            "binary_source_full_size_fixture"
        }
    }
}

/// The 64-bit root: base field p = 2^64 - 59 with E64 = F_p[u]/(u^2 - 2)
/// challenges (specs/binary-p64-root.md).
pub(super) mod p64 {
    use akita_challenges::FoldChallengeDistribution;
    use akita_config::{proof_optimized::fp64::Dense, CommitmentConfig};
    use akita_error::AkitaError;
    use akita_params::binary_setup::BinaryFieldProfile;
    use akita_params::{sis::CommittedSourceClass, DecompositionParams, SisModulusProfileId};

    pub(crate) type F = jolt_field::Prime64Offset59;
    pub(crate) type E = jolt_field::Ext2<F>;
    pub(crate) const FIELD: BinaryFieldProfile = BinaryFieldProfile::P64;
    pub(crate) const NAME: &str = "p64";

    /// Ordinary suffix for an explicitly selected experimental binary fixture.
    #[derive(Clone)]
    pub(crate) struct FullSize;

    impl CommitmentConfig for FullSize {
        type Field = F;
        type ExtField = E;
        // The mixed root admits native degrees up to 648 and a 64-dimensional
        // ordinary B image, which the suffix dimensions must also fit.
        const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
            akita_schedules::RingDimensionSchedule {
                potential_a_dimensions: &[64, 128, 256, 512],
                potential_b_dimensions: &[64],
                suffix_dimensions: &[64],
                ..Dense::RING_DIMENSION_SCHEDULE
            };

        fn decomposition() -> DecompositionParams {
            DecompositionParams {
                log_basis: Dense::decomposition().log_basis,
                // The outgoing root witness holds base-eight digits in [-4,3].
                log_commit_bound: 5,
                // Openings carry full base-field elements below p < 2^64.
                log_open_bound: Some(64),
            }
        }
        fn ring_challenge_config(
            dimension: usize,
        ) -> Result<FoldChallengeDistribution, AkitaError> {
            Dense::ring_challenge_config(dimension)
        }
        fn sis_modulus_profile() -> SisModulusProfileId {
            // The ordinary p64 profile of the mixed root (spec, "Mixed ordinary profile").
            SisModulusProfileId::Q64Offset59
        }
        fn opening_basis_range() -> (u32, u32) {
            Dense::opening_basis_range()
        }
        fn inner_basis_range() -> (u32, u32) {
            Dense::inner_basis_range()
        }
        fn committed_source_class() -> CommittedSourceClass {
            Dense::committed_source_class()
        }
        fn recursive_setup_planning() -> bool {
            true
        }
        fn schedule_family_name() -> &'static str {
            "binary_source_full_size_p64_fixture"
        }
    }
}
