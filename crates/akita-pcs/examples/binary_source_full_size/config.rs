use akita_challenges::FoldChallengeFamily;
use akita_config::{proof_optimized::fp128::DenseBounded, CommitmentConfig};
use akita_error::AkitaError;
use akita_params::{sis::CommittedSourceClass, DecompositionParams, SisModulusProfileId};

pub(super) type F = jolt_field::Prime128Offset275;

/// Ordinary suffix for an explicitly selected experimental binary fixture.
#[derive(Clone)]
pub(super) struct FullSize;

impl CommitmentConfig for FullSize {
    type Field = F;
    type ExtField = F;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        DenseBounded::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: DenseBounded::decomposition().log_basis,
            log_commit_bound: 5,
            log_open_bound: Some(128),
        }
    }
    fn ring_challenge_config(dimension: usize) -> Result<FoldChallengeFamily, AkitaError> {
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
