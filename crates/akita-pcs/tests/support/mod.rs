//! Test-local PCS config adapters.
//!
//! Crate unit tests include this module under `cfg(test)`. Production builds
//! never compile it.

#![allow(dead_code)]

use akita_challenges::SparseChallengeConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, CommittedGroupProfile,
    DecompositionParams, OpeningScheduleSelection, PolynomialGroupLayout, SetupMatrixCapacity,
    SisModulusProfileId,
};
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{Mutex, OnceLock},
};

#[derive(Clone)]
struct SyntheticResolvedRow {
    config: TypeId,
    row: akita_config::ResolvedScheduleRow,
}

fn synthetic_resolved_rows() -> &'static Mutex<Vec<SyntheticResolvedRow>> {
    static ROWS: OnceLock<Mutex<Vec<SyntheticResolvedRow>>> = OnceLock::new();
    ROWS.get_or_init(|| Mutex::new(Vec::new()))
}

fn select_synthetic_schedule_row<C>(
    profiles: &CommittedGroupBatchProfile,
    key: AkitaScheduleLookupKey,
) -> Result<akita_config::ResolvedScheduleRow, AkitaError>
where
    C: CommitmentConfig + 'static,
{
    let row = C::resolve_catalog_row_for_key(&key)?;
    if row.profiles() != profiles {
        return Err(AkitaError::InvalidSetup(
            "synthetic selected row does not match exact committed profiles".into(),
        ));
    }
    let selection = row.selection();
    let mut rows = synthetic_resolved_rows()
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("synthetic row cache is poisoned".into()))?;
    if let Some(existing) = rows.iter_mut().find(|existing| {
        existing.config == TypeId::of::<C>() && existing.row.selection() == selection
    }) {
        existing.row = row.clone();
        return Ok(row);
    }
    if rows.len() >= 1024 {
        return Err(AkitaError::InvalidSetup(
            "synthetic row cache capacity exceeded".into(),
        ));
    }
    rows.push(SyntheticResolvedRow {
        config: TypeId::of::<C>(),
        row: row.clone(),
    });
    Ok(row)
}

fn resolve_synthetic_schedule_row<C>(
    selection: OpeningScheduleSelection,
) -> Result<akita_config::ResolvedScheduleRow, AkitaError>
where
    C: CommitmentConfig + 'static,
{
    synthetic_resolved_rows()
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("synthetic row cache is poisoned".into()))?
        .iter()
        .find(|entry| entry.config == TypeId::of::<C>() && entry.row.selection() == selection)
        .map(|entry| entry.row.clone())
        .ok_or_else(|| {
            AkitaError::UnsupportedSchedule(
                "synthetic schedule selection is not present in the test catalog".into(),
            )
        })
}

fn synthetic_schedule_key(profiles: &CommittedGroupBatchProfile) -> AkitaScheduleLookupKey {
    AkitaScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
}

/// Test-only commitment config that combines an envelope config with a final
/// group config.
///
/// The precommitted groups use `Envelope::D` while the final group and
/// recursive suffix use `Final::D`.
///
/// `Self::D` remains the uniform planner default. Exact grouped runtime keys
/// select schedules under `Final`, retaining each preceding group's frozen
/// native descriptor. Public setup storage remains flat and dimension-free.
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

    const D: usize = Envelope::D;
    const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
        Envelope::RING_DIMENSION_SCHEDULE_MODE;

    fn decomposition() -> DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d).or_else(|_| Final::ring_challenge_config(d))
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Envelope::selection_policy()
    }
    fn sis_modulus_profile() -> SisModulusProfileId {
        Envelope::sis_modulus_profile()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        let mut num_field_elements =
            Envelope::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?
                .num_field_elements
                .max(
                    Final::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?
                        .num_field_elements,
                );
        for final_polys in 1..max_num_batched_polys {
            let pre_polys = max_num_batched_polys - final_polys;
            for pre_num_vars in [14usize, 15, 16].into_iter().filter(|&n| n <= max_num_vars) {
                let Ok(precommitted) = Self::profile_without_precommitted_groups(
                    PolynomialGroupLayout::new(pre_num_vars, pre_polys),
                ) else {
                    continue;
                };
                let Ok(schedule) = Self::resolve_catalog_row_for_key(&AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(max_num_vars, final_polys),
                    precommitteds: vec![precommitted],
                }) else {
                    continue;
                };
                num_field_elements = num_field_elements.max(
                    akita_types::setup_matrix_capacity_for_schedule(schedule.schedule())?
                        .num_field_elements,
                );
            }
        }
        Ok(SetupMatrixCapacity { num_field_elements })
    }

    fn opening_basis_range() -> (u32, u32) {
        Envelope::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Envelope::inner_basis_range()
    }

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        Envelope::committed_source_class()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Envelope::root_honest_fold_policy()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Envelope::schedule_catalog()
    }

    fn resolve_catalog_row_for_key(
        key: &AkitaScheduleLookupKey,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let (policy, ring_challenge_config) = if key.precommitteds.is_empty() {
            (
                policy_of::<Envelope>(),
                Envelope::ring_challenge_config
                    as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
            )
        } else {
            (
                policy_of::<Final>(),
                Final::ring_challenge_config
                    as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
            )
        };
        let precommitted_honest_fold_policies =
            vec![Envelope::root_honest_fold_policy(); key.precommitteds.len()];
        let schedule = akita_planner::find_schedule(
            key,
            Envelope::root_honest_fold_policy(),
            &precommitted_honest_fold_policies,
            &policy,
            ring_challenge_config,
        )?
        .schedule;
        let profiles = CommittedGroupBatchProfile {
            final_group: CommittedGroupProfile::try_from_params(
                key.final_group,
                &schedule.root.params.final_group.commitment,
            )?,
            precommitteds: key.precommitteds.clone(),
        };
        let selection = OpeningScheduleSelection {
            row_digest: schedule_row_digest(&profiles, &schedule)?,
        };
        akita_config::ResolvedScheduleRow::try_new(
            selection,
            profiles,
            schedule,
            &policy_of::<Self>(),
        )
    }

    fn resolve_catalog_row_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        select_synthetic_schedule_row::<Self>(profiles, synthetic_schedule_key(profiles))
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        resolve_synthetic_schedule_row::<Self>(selection)
    }
}
