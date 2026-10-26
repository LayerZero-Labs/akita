//! Test-local PCS config adapters.
//!
//! Crate unit tests include this module under `cfg(test)`. Production builds
//! never compile it.

#![allow(dead_code)]

use akita_challenges::SparseChallengeConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_error::AkitaError;
use akita_params::{
    CommittedGroupBatchProfile, DecompositionParams, GroupCommitPhaseParams, ScheduleLookupKey,
    SisModulusProfileId,
};
use std::marker::PhantomData;

mod cross_mode;
pub(crate) use cross_mode::cross_mode_catalogs;
mod packing;
pub(crate) use packing::{RootCoefficientPackingConfig, UnsupportedEvaluationTraceConfig};

/// Test-only commitment config that combines an envelope config with a final
/// group config.
///
/// Exact grouped runtime keys select schedules under `Final`, retaining each
/// preceding group's frozen native descriptor. Public setup storage remains
/// flat and dimension-free.
#[derive(Debug)]
pub(crate) struct EnvelopeFinalGroupConfig<Envelope, Final>(PhantomData<fn() -> (Envelope, Final)>);

impl<Envelope, Final> Clone for EnvelopeFinalGroupConfig<Envelope, Final> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Envelope, Final> Copy for EnvelopeFinalGroupConfig<Envelope, Final> {}

impl<Envelope, Final> Default for EnvelopeFinalGroupConfig<Envelope, Final> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Envelope, Final> CommitmentConfig for EnvelopeFinalGroupConfig<Envelope, Final>
where
    Envelope: CommitmentConfig + 'static,
    Final: CommitmentConfig<Field = Envelope::Field, ExtField = Envelope::ExtField> + 'static,
{
    type Field = Envelope::Field;
    type ExtField = Envelope::ExtField;

    const RING_DIMENSION_SCHEDULE: akita_schedules::RingDimensionSchedule =
        Envelope::RING_DIMENSION_SCHEDULE;

    fn decomposition() -> DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d).or_else(|_| Final::ring_challenge_config(d))
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Envelope::selection_policy()
    }

    fn schedule_family_name() -> &'static str {
        Envelope::schedule_family_name()
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Envelope::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        Envelope::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Envelope::inner_basis_range()
    }

    fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
        Envelope::committed_source_class()
    }
}

impl<Envelope, Final> EnvelopeFinalGroupConfig<Envelope, Final>
where
    Envelope: CommitmentConfig + 'static,
    Final: CommitmentConfig<Field = Envelope::Field, ExtField = Envelope::ExtField> + 'static,
{
    pub(crate) fn derive_catalog_row(
        key: &ScheduleLookupKey,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let (policy, ring_challenge_config) = if key.precommitteds.is_empty() {
            (
                policy_of::<Envelope>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
                Envelope::ring_challenge_config
                    as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
            )
        } else {
            (
                policy_of::<Final>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
                Final::ring_challenge_config
                    as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
            )
        };
        let precommitted_source_contracts =
            vec![Envelope::committed_source_contract()?; key.precommitteds.len()];
        let schedule = akita_planner::find_schedule(
            key,
            Envelope::committed_source_contract()?,
            &precommitted_source_contracts,
            &policy,
            ring_challenge_config,
        )?
        .schedule;
        let profiles = CommittedGroupBatchProfile {
            final_group: GroupCommitPhaseParams::try_from_params(
                key.final_group,
                &schedule.root.params,
            )?,
            precommitteds: key.precommitteds.clone(),
        };
        akita_config::ResolvedScheduleRow::try_new(
            profiles,
            schedule,
            &policy_of::<Self>(akita_params::ChunkedWitnessCfg::default_non_chunked()),
        )
    }
}
