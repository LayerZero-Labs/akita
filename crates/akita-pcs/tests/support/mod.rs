//! Test-local PCS config adapters.
//!
//! Crate unit tests include this module under `cfg(test)`. Production builds
//! never compile it.

use akita_challenges::SparseChallengeConfig;
use akita_config::{committed_group_profile, policy_of, CommitmentConfig};
use akita_field::AkitaError;
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, DecompositionParams,
    FoldSchedule, OpeningClaimsLayout, OpeningScheduleSelection, PolynomialGroupLayout,
    SetupMatrixCapacity, SisModulusProfileId,
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

/// Test-only config that replaces the exact L3 scalar fold with a Q32
/// multi-block physical-L2 route of the same A rank.
///
/// Keeping the rank unchanged preserves the base catalog's successor geometry;
/// the synthetic row still passes the ordinary schedule audit against this
/// config's exact measured cap.
#[derive(Debug)]
pub(crate) struct ForcedSmallFieldL2Config<Base>(PhantomData<Base>);

impl<Base> Clone for ForcedSmallFieldL2Config<Base> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Base> Copy for ForcedSmallFieldL2Config<Base> {}

impl<Base> Default for ForcedSmallFieldL2Config<Base> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Base: CommitmentConfig> CommitmentConfig for ForcedSmallFieldL2Config<Base> {
    type Field = Base::Field;
    type ExtField = Base::ExtField;

    const D: usize = Base::D;
    const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
        Base::RING_DIMENSION_SCHEDULE_MODE;
    const SELECTIVE_L2_FOLD_CAPS: &'static [akita_schedules::SelectiveL2FoldCap] =
        &[akita_schedules::SelectiveL2FoldCap {
            fold_level: 3,
            input_witness_len: 130_816,
            physical_response_len: 16_384,
            fold_basis: 64,
            fold_digit_count: 2,
            response_l2_sq_cap: 1u128 << 35,
        }];

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Base::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Base::sis_modulus_profile()
    }

    fn setup_matrix_capacity(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixCapacity, AkitaError> {
        Base::setup_matrix_capacity(max_num_vars, max_num_batched_polys)
    }

    fn setup_prefix_inner_ring_dimension() -> usize {
        Base::setup_prefix_inner_ring_dimension()
    }

    fn basis_range() -> (u32, u32) {
        Base::basis_range()
    }

    fn root_honest_fold_policy() -> akita_types::sis::HonestFoldPolicySpec {
        Base::root_honest_fold_policy()
    }

    fn chunked_witness_cfg() -> akita_types::ChunkedWitnessCfg {
        Base::chunked_witness_cfg()
    }

    fn recursive_setup_planning() -> bool {
        Base::recursive_setup_planning()
    }

    fn selection_policy() -> akita_schedules::SelectionPolicyId {
        Base::selection_policy()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        let mut schedule = Base::runtime_schedule(key)?;
        let step = schedule.recursive_folds.get_mut(2).ok_or_else(|| {
            AkitaError::UnsupportedSchedule("small-field L2 fixture requires an L3 fold".into())
        })?;
        let params = &mut step.params.witness;
        let fold_basis = 1usize
            .checked_shl(params.log_basis_open)
            .ok_or_else(|| AkitaError::InvalidSetup("small-field L2 basis overflow".into()))?;
        let matrix = akita_schedules::planner_support::selective_l2_inner_matrix(
            &policy_of::<Self>(),
            akita_schedules::planner_support::SelectiveL2CandidateGeometry {
                fold_level: 3,
                input_witness_len: step.input_witness_len,
                num_claims: 1,
                num_chunks: params.witness_chunk.num_chunks,
                inner_width: params.inner_commit_matrix.input_width(),
                ring_dimension: params.d_a(),
                fold_basis,
                fold_digit_count: params.num_digits_fold,
                fold_challenge_config: &params.fold_challenge_config,
            },
        )?
        .ok_or_else(|| {
            AkitaError::InvalidSetup("small-field L2 fixture geometry lost its exact cap".into())
        })?;
        if matrix.output_rank() != params.inner_commit_matrix.output_rank() {
            return Err(AkitaError::InvalidSetup(
                "small-field L2 fixture must preserve the catalog A rank".into(),
            ));
        }
        params.inner_commit_matrix = matrix;
        schedule.validate_structure()?;
        Ok(schedule)
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

    fn get_params_for_prove(layout: &OpeningClaimsLayout) -> Result<FoldSchedule, AkitaError> {
        layout.check()?;
        Self::runtime_schedule(AkitaScheduleLookupKey::single(
            layout.root_final_group_layout()?,
        ))
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

    const SELECTIVE_L2_FOLD_CAPS: &'static [akita_schedules::SelectiveL2FoldCap] =
        Envelope::SELECTIVE_L2_FOLD_CAPS;

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
        akita_planner::find_schedule(
            &key,
            Envelope::root_honest_fold_policy(),
            &precommitted_honest_fold_policies,
            &policy,
            ring_challenge_config,
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
