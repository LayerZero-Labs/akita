use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{Mutex, OnceLock},
};

use akita_challenges::SparseChallengeConfig;
use akita_config::{policy_of, CommitmentConfig};
use akita_error::AkitaError;
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, DecompositionParams,
    GroupCommitPhaseParams, OpeningScheduleSelection, SetupMatrixCapacity, SisModulusProfileId,
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct QuotientMode;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ReducedMode;

pub(crate) trait RelationModeChoice: Clone + Copy + Send + Sync + 'static {
    const FILTER: akita_planner::TestRelationModeFilter;
}

impl RelationModeChoice for QuotientMode {
    const FILTER: akita_planner::TestRelationModeFilter =
        akita_planner::TestRelationModeFilter::QuotientOnly;
}

impl RelationModeChoice for ReducedMode {
    const FILTER: akita_planner::TestRelationModeFilter =
        akita_planner::TestRelationModeFilter::All;
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CrossModeConfig<Base, Mode, const NUM_VARS: usize, const NUM_POLYS: usize>(
    PhantomData<fn() -> (Base, Mode)>,
);

#[derive(Clone)]
struct CachedRows {
    base: TypeId,
    key: AkitaScheduleLookupKey,
    quotient: akita_config::ResolvedScheduleRow,
    reduced: akita_config::ResolvedScheduleRow,
}

fn cached_rows() -> &'static Mutex<Vec<CachedRows>> {
    static ROWS: OnceLock<Mutex<Vec<CachedRows>>> = OnceLock::new();
    ROWS.get_or_init(|| Mutex::new(Vec::new()))
}

fn planned_row<Base>(
    key: &AkitaScheduleLookupKey,
    filter: akita_planner::TestRelationModeFilter,
) -> Result<akita_config::ResolvedScheduleRow, AkitaError>
where
    Base: CommitmentConfig,
{
    let policy = policy_of::<Base>();
    let schedule = akita_planner::find_schedule_for_test_relation_mode(
        key,
        akita_config::honest_fold_policy_of::<Base>(),
        &[],
        &policy,
        Base::ring_challenge_config,
        filter,
    )?
    .schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )?,
        precommitteds: key.precommitteds.clone(),
    };
    let selection = OpeningScheduleSelection {
        row_digest: schedule_row_digest(&profiles, &schedule)?,
    };
    akita_config::ResolvedScheduleRow::try_new(selection, profiles, schedule, &policy)
}

fn rows_for<Base>(key: &AkitaScheduleLookupKey) -> Result<CachedRows, AkitaError>
where
    Base: CommitmentConfig + 'static,
{
    let base = TypeId::of::<Base>();
    if let Some(rows) = cached_rows()
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("cross-mode row cache is poisoned".into()))?
        .iter()
        .find(|rows| rows.base == base && rows.key == *key)
        .cloned()
    {
        return Ok(rows);
    }

    let quotient = planned_row::<Base>(key, akita_planner::TestRelationModeFilter::QuotientOnly)?;
    let reduced = planned_row::<Base>(key, akita_planner::TestRelationModeFilter::All)?;
    if quotient.profiles() != reduced.profiles() {
        return Err(AkitaError::InvalidSetup(
            "cross-mode schedules must have identical committed profiles".into(),
        ));
    }
    if quotient.selection() == reduced.selection()
        || !reduced
            .schedule()
            .recursive_folds
            .iter()
            .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation())
        || quotient
            .schedule()
            .recursive_folds
            .iter()
            .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation())
    {
        return Err(AkitaError::InvalidSetup(
            "cross-mode rows do not realize distinct quotient and reduced suffixes".into(),
        ));
    }
    let rows = CachedRows {
        base,
        key: key.clone(),
        quotient,
        reduced,
    };
    let mut cache = cached_rows()
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("cross-mode row cache is poisoned".into()))?;
    if let Some(existing) = cache
        .iter()
        .find(|existing| existing.base == base && existing.key == *key)
        .cloned()
    {
        return Ok(existing);
    }
    if cache.len() >= 32 {
        return Err(AkitaError::InvalidSetup(
            "cross-mode row cache capacity exceeded".into(),
        ));
    }
    cache.push(rows.clone());
    Ok(rows)
}

impl<Base, Mode, const NUM_VARS: usize, const NUM_POLYS: usize> CommitmentConfig
    for CrossModeConfig<Base, Mode, NUM_VARS, NUM_POLYS>
where
    Base: CommitmentConfig + 'static,
    Mode: RelationModeChoice,
{
    type Field = Base::Field;
    type ExtField = Base::ExtField;

    const EXT_DEGREE: usize = Base::EXT_DEGREE;
    const RING_DIMENSION_SCHEDULE_MODE: akita_schedules::RingDimensionScheduleMode =
        Base::RING_DIMENSION_SCHEDULE_MODE;

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
        if (max_num_vars, max_num_batched_polys) != (NUM_VARS, NUM_POLYS) {
            return Err(AkitaError::UnsupportedSchedule(
                "cross-mode config admits only its exact test layout".into(),
            ));
        }
        let rows = rows_for::<Base>(&AkitaScheduleLookupKey::single(
            akita_types::PolynomialGroupLayout::new(NUM_VARS, NUM_POLYS),
        ))?;
        let base = Base::setup_matrix_capacity(max_num_vars, max_num_batched_polys)?;
        let quotient = akita_types::setup_matrix_capacity_for_schedule(rows.quotient.schedule())?;
        let reduced = akita_types::setup_matrix_capacity_for_schedule(rows.reduced.schedule())?;
        Ok(SetupMatrixCapacity {
            num_field_elements: base
                .num_field_elements
                .max(quotient.num_field_elements)
                .max(reduced.num_field_elements),
        })
    }

    fn opening_basis_range() -> (u32, u32) {
        Base::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        Base::inner_basis_range()
    }

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        Base::committed_source_class()
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

    fn resolve_catalog_row_for_key(
        key: &AkitaScheduleLookupKey,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        if !key.precommitteds.is_empty()
            || key.final_group != akita_types::PolynomialGroupLayout::new(NUM_VARS, NUM_POLYS)
        {
            return Err(AkitaError::UnsupportedSchedule(
                "cross-mode config admits only its exact test layout".into(),
            ));
        }
        let rows = rows_for::<Base>(key)?;
        Ok(match Mode::FILTER {
            akita_planner::TestRelationModeFilter::All => rows.reduced,
            akita_planner::TestRelationModeFilter::QuotientOnly => rows.quotient,
        })
    }

    fn resolve_catalog_row_for_profiles(
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let key = AkitaScheduleLookupKey {
            final_group: profiles.final_group.group,
            precommitteds: profiles.precommitteds.clone(),
        };
        let row = Self::resolve_catalog_row_for_key(&key)?;
        if row.profiles() != profiles {
            return Err(AkitaError::InvalidSetup(
                "cross-mode row does not match exact committed profiles".into(),
            ));
        }
        Ok(row)
    }

    fn resolve_schedule_selection(
        selection: OpeningScheduleSelection,
    ) -> Result<akita_config::ResolvedScheduleRow, AkitaError> {
        let exact_key = AkitaScheduleLookupKey::single(akita_types::PolynomialGroupLayout::new(
            NUM_VARS, NUM_POLYS,
        ));
        cached_rows()
            .lock()
            .map_err(|_| AkitaError::InvalidSetup("cross-mode row cache is poisoned".into()))?
            .iter()
            .filter(|rows| rows.base == TypeId::of::<Base>() && rows.key == exact_key)
            .flat_map(|rows| [&rows.quotient, &rows.reduced])
            .find(|row| row.selection() == selection)
            .cloned()
            .ok_or_else(|| {
                AkitaError::UnsupportedSchedule(
                    "selection is absent from the cross-mode test catalog".into(),
                )
            })
    }
}
