//! Recursive setup-offloading config adapter.

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::{DecompositionParams, SisModulusProfileId};
use jolt_crypto::challenge::SparseChallengeConfig;
use std::marker::PhantomData;

/// Config adapter that enables recursion-aware setup offloading schedules.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecursiveCommitmentConfig<Cfg>(PhantomData<Cfg>);

/// A direct config with a separately generated recursive schedule family.
pub trait RecursiveScheduleConfig: CommitmentConfig {
    /// Stable family identity for the recursive companion artifact.
    const RECURSIVE_SCHEDULE_FAMILY_NAME: &'static str;

    /// Family of the recursive companion's zero-knowledge schedules, if any
    /// (the recursive adapter's [`CommitmentConfig::ZK_SCHEDULE_FAMILY_NAME`]).
    const RECURSIVE_ZK_SCHEDULE_FAMILY_NAME: Option<&'static str> = None;
}

impl<Cfg: RecursiveScheduleConfig> CommitmentConfig for RecursiveCommitmentConfig<Cfg> {
    type Field = Cfg::Field;
    type ExtField = Cfg::ExtField;

    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        Cfg::RING_DIMENSION_SCHEDULE;
    const ZK_SCHEDULE_FAMILY_NAME: Option<&'static str> = Cfg::RECURSIVE_ZK_SCHEDULE_FAMILY_NAME;

    fn schedule_family_name() -> &'static str {
        Cfg::RECURSIVE_SCHEDULE_FAMILY_NAME
    }
    fn decomposition() -> DecompositionParams {
        Cfg::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Cfg::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Cfg::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        Cfg::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Cfg::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Cfg::committed_source_class()
    }

    fn recursive_setup_planning() -> bool {
        true
    }
}
