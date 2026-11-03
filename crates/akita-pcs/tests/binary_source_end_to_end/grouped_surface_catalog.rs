//! Explicit ordinary suffix selection for the grouped interface tests.
use super::*;

pub(super) fn catalog(owners: usize) -> TrustedScheduleCatalog<BinaryTestSuffix> {
    if owners == 1 {
        return test_catalog();
    }
    static CATALOGS: std::sync::LazyLock<
        std::sync::Mutex<
            std::collections::BTreeMap<usize, TrustedScheduleCatalog<BinaryTestSuffix>>,
        >,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));
    let mut catalogs = CATALOGS.lock().unwrap();
    catalogs
        .entry(owners)
        .or_insert_with(|| build(owners))
        .clone()
}

fn build(owners: usize) -> TrustedScheduleCatalog<BinaryTestSuffix> {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = akita_config::policy_of::<BinaryTestSuffix>(chunking);
    let layout = PolynomialGroupLayout::new(18, owners);
    let schedule = akita_planner::find_schedule_with_root_setup_mode(
        &ScheduleLookupKey::single(layout),
        BinaryTestSuffix::committed_source_contract().unwrap(),
        &[],
        &policy,
        BinaryTestSuffix::ring_challenge_config,
        akita_params::SetupContributionMode::Recursive,
        None,
    )
    .unwrap()
    .schedule;
    let profiles = akita_params::CommittedGroupBatchProfile {
        final_group: akita_params::GroupCommitPhaseParams::try_from_params(
            layout,
            &schedule.root.params,
        )
        .unwrap(),
        precommitteds: Vec::new(),
    };
    TrustedScheduleCatalog::new(
        akita_schedules::ValidatedScheduleCatalog::try_new(
            BinaryTestSuffix::schedule_family_name(),
            [(profiles, schedule)],
            &policy,
            BinaryTestSuffix::ring_challenge_config,
        )
        .unwrap(),
        chunking,
    )
    .unwrap()
}
