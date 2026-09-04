//! Explicit one-row catalogs for quotient/reduced relation-mode tests.

use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_planner::{find_schedule_for_test_relation_mode, TestRelationModeFilter};
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupBatchProfile, GroupCommitPhaseParams,
    OpeningScheduleSelection,
};

/// Two valid catalogs for the same opening key, differing only in the
/// planner's permitted ring-relation modes.
pub(crate) struct CrossModeCatalogs {
    pub(crate) quotient: TrustedScheduleCatalog,
    pub(crate) reduced: TrustedScheduleCatalog,
    pub(crate) quotient_selection: OpeningScheduleSelection,
    pub(crate) reduced_selection: OpeningScheduleSelection,
}

fn planned_row<Base: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    relation_modes: TestRelationModeFilter,
) -> Result<(CommittedGroupBatchProfile, akita_types::FoldSchedule), AkitaError> {
    if !key.precommitteds.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "cross-mode fixture supports one final group only".into(),
        ));
    }
    let policy = policy_of::<Base>();
    let final_honest_fold_policy = honest_fold_policy_of::<Base>();
    let planned = find_schedule_for_test_relation_mode(
        key,
        final_honest_fold_policy,
        &[],
        &policy,
        Base::ring_challenge_config,
        relation_modes,
    )?;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &planned.schedule.root.params,
        )?,
        precommitteds: key.precommitteds.clone(),
    };
    Ok((profiles, planned.schedule))
}

/// Plan and admit quotient-only and adaptive rows without ambient registries.
pub(crate) fn cross_mode_catalogs<Base: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<CrossModeCatalogs, AkitaError> {
    let (quotient_profiles, quotient_schedule) =
        planned_row::<Base>(key, TestRelationModeFilter::QuotientOnly)?;
    let (reduced_profiles, reduced_schedule) =
        planned_row::<Base>(key, TestRelationModeFilter::All)?;

    if quotient_profiles != reduced_profiles {
        return Err(AkitaError::InvalidSetup(
            "cross-mode rows must retain one committed-group profile".into(),
        ));
    }
    if quotient_schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation())
    {
        return Err(AkitaError::InvalidSetup(
            "quotient-only planning selected a reduced relation".into(),
        ));
    }
    if !reduced_schedule
        .recursive_folds
        .iter()
        .any(|fold| fold.params.ring_relation_mode.is_reduced_evaluation())
    {
        return Err(AkitaError::InvalidSetup(
            "adaptive planning did not select a reduced relation".into(),
        ));
    }

    let policy = policy_of::<Base>();
    let quotient = TrustedScheduleCatalog::try_new(
        Base::schedule_family_name(),
        [(quotient_profiles.clone(), quotient_schedule)],
        &policy,
        Base::ring_challenge_config,
    )?;
    let reduced = TrustedScheduleCatalog::try_new(
        Base::schedule_family_name(),
        [(reduced_profiles, reduced_schedule)],
        &policy,
        Base::ring_challenge_config,
    )?;
    let quotient_selection = quotient.resolve_key(key)?.selection();
    let reduced_selection = reduced.resolve_key(key)?.selection();
    if quotient_selection == reduced_selection {
        return Err(AkitaError::InvalidSetup(
            "relation-mode rows must have distinct identities".into(),
        ));
    }

    Ok(CrossModeCatalogs {
        quotient,
        reduced,
        quotient_selection,
        reduced_selection,
    })
}
