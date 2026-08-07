//! Test-local PCS config adapters.
//!
//! Crate unit tests include this module under `cfg(test)`. Production builds
//! never compile it.

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_config::{committed_group_profile, policy_of, CommitmentConfig};
use akita_error::AkitaError;
use akita_types::{
    schedule_row_digest, AkitaScheduleInputs, AkitaScheduleLookupKey, CommittedGroupBatchProfile,
    DecompositionParams, FoldSchedule, OpeningClaimsLayout, OpeningScheduleSelection,
    PolynomialGroupLayout, SetupMatrixCapacity, SisModulusProfileId,
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
    let schedule = C::runtime_schedule(key)?;
    let row_digest = schedule_row_digest(profiles, &schedule)?;
    let selection = OpeningScheduleSelection { row_digest };
    let row = akita_config::ResolvedScheduleRow::try_new(
        selection,
        profiles.clone(),
        schedule,
        &akita_config::policy_of::<C>(),
    )?;
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

    fn decomposition() -> DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d).or_else(|_| Final::ring_challenge_config(d))
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Envelope::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Envelope::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Envelope::ring_subfield_embedding_norm_bound()
            .max(Final::ring_subfield_embedding_norm_bound())
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
                let Ok(precommitted) = committed_group_profile::<Self>(
                    &PolynomialGroupLayout::new(pre_num_vars, pre_polys),
                ) else {
                    continue;
                };
                let Ok(schedule) = Self::runtime_schedule(AkitaScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(max_num_vars, final_polys),
                    precommitteds: vec![precommitted],
                }) else {
                    continue;
                };
                num_field_elements = num_field_elements.max(
                    akita_types::setup_matrix_capacity_for_schedule(&schedule)?.num_field_elements,
                );
            }
        }
        Ok(SetupMatrixCapacity { num_field_elements })
    }

    fn basis_range() -> (u32, u32) {
        Envelope::basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Envelope::root_honest_fold_policy()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Envelope::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        let (policy, ring_challenge_config, fold_challenge_shape_at_level) =
            if key.precommitteds.is_empty() {
                (
                    policy_of::<Envelope>(),
                    Envelope::ring_challenge_config
                        as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
                    Envelope::fold_challenge_shape_at_level
                        as fn(AkitaScheduleInputs) -> TensorChallengeShape,
                )
            } else {
                (
                    policy_of::<Final>(),
                    Final::ring_challenge_config
                        as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
                    Final::fold_challenge_shape_at_level
                        as fn(AkitaScheduleInputs) -> TensorChallengeShape,
                )
            };
        let precommitted_honest_fold_policies =
            vec![Envelope::root_honest_fold_policy(); key.precommitteds.len()];
        akita_planner::find_schedule(
            &key,
            Envelope::root_honest_fold_policy(),
            &precommitted_honest_fold_policies,
            &policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )
        .map(|planned| planned.schedule)
    }

    fn select_schedule_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        select_synthetic_schedule_row::<Self>(profiles, synthetic_schedule_key(profiles))
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        resolve_synthetic_schedule_row::<Self>(selection)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        opening_batch.check()?;
        if opening_batch.num_groups() != 1 {
            return Err(AkitaError::InvalidInput(
                "grouped schedule selection requires exact committed-group descriptors".into(),
            ));
        }
        Self::runtime_schedule(AkitaScheduleLookupKey::single(
            opening_batch.root_final_group_layout()?,
        ))
    }
}
