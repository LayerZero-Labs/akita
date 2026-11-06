use akita_challenges::FoldChallengeDistribution;
use akita_config::{proof_optimized::fp128::DenseBounded, CommitmentConfig};
use akita_error::AkitaError;
use akita_params::binary_setup::BinaryFieldProfile;
use akita_params::{sis::CommittedSourceClass, DecompositionParams, SisModulusProfileId};

pub(super) type F = super::super::F;
pub(super) type E = super::super::E;
pub(super) const PROFILE: BinaryFieldProfile = super::super::PROFILE;

/// Ordinary suffix for an explicitly selected experimental binary fixture.
#[derive(Clone)]
pub(super) struct FullSize;

impl CommitmentConfig for FullSize {
    type Field = F;
    type ExtField = E;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        DenseBounded::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: DenseBounded::decomposition().log_basis,
            log_commit_bound: 5,
            log_open_bound: Some(match PROFILE {
                BinaryFieldProfile::P128 => 128,
                BinaryFieldProfile::P64 => 64,
            }),
        }
    }
    fn ring_challenge_config(dimension: usize) -> Result<FoldChallengeDistribution, AkitaError> {
        DenseBounded::ring_challenge_config(dimension)
    }
    fn sis_modulus_profile() -> SisModulusProfileId {
        match PROFILE {
            BinaryFieldProfile::P128 => SisModulusProfileId::Q128Offset275,
            BinaryFieldProfile::P64 => SisModulusProfileId::Q64Offset59,
        }
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
        match PROFILE {
            BinaryFieldProfile::P128 => "binary_source_full_size_fixture",
            BinaryFieldProfile::P64 => "binary_source_p64_research",
        }
    }
}
