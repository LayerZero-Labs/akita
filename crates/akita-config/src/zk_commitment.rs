//! The zero-knowledge configuration adapter (`specs/whiteout-akita.md` §1).

use crate::CommitmentConfig;
use akita_error::AkitaError;
use akita_params::{DecompositionParams, SisModulusProfileId};
use jolt_crypto::challenge::SparseChallengeConfig;
use std::marker::PhantomData;

/// `Cfg` proving in zero knowledge: its own field, decomposition and planning
/// policy, under its zero-knowledge schedule family
/// ([`CommitmentConfig::ZK_SCHEDULE_FAMILY_NAME`]). The family is resolved at
/// compile time, so the adapter exists only for a `Cfg` that declares one.
/// Its catalogs admit only zero-knowledge rows, and every descriptor it binds
/// carries `zk = true`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ZkCommitmentConfig<Cfg>(PhantomData<Cfg>);

impl<Cfg: CommitmentConfig> ZkCommitmentConfig<Cfg> {
    const FAMILY: &'static str = match Cfg::ZK_SCHEDULE_FAMILY_NAME {
        Some(family) => family,
        None => panic!("this configuration has no zero-knowledge schedule family"),
    };
}

impl<Cfg: CommitmentConfig> CommitmentConfig for ZkCommitmentConfig<Cfg> {
    type Field = Cfg::Field;
    type ExtField = Cfg::ExtField;

    const EXT_DEGREE: usize = Cfg::EXT_DEGREE;
    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        Cfg::RING_DIMENSION_SCHEDULE;
    const IS_ZK: bool = true;

    fn schedule_family_name() -> &'static str {
        Self::FAMILY
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

    fn committed_source_contract() -> Result<akita_params::sis::CommittedSourceContract, AkitaError>
    {
        Cfg::committed_source_contract()
    }

    fn validate_sis_modulus_profile() -> Result<(), AkitaError> {
        Cfg::validate_sis_modulus_profile()
    }

    fn recursive_setup_planning() -> bool {
        Cfg::recursive_setup_planning()
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Cfg::selection_policy()
    }
}
