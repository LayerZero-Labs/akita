use super::*;
use akita_config::{
    policy_of, proof_optimized::fp128::Dense, CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_params::{ChunkedWitnessCfg, CommittedGroupBatchProfile};

#[test]
fn root_accumulation_reuses_the_existing_offloaded_successor_search() {
    type Recursive = RecursiveCommitmentConfig<Dense>;
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = policy_of::<Recursive>(chunking);
    let catalog =
        akita_config::test_support::workspace_schedule_catalog::<Recursive>(chunking).unwrap();
    let mut key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(24));
    let baseline = catalog.resolve_key(&key).unwrap();
    let schedule = baseline.schedule();
    let local =
        akita_params::active_setup_field_len(&schedule.root.params, &key.opening_layout().unwrap())
            .unwrap();
    let prefix_len = schedule.recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .n_prefix()
        .unwrap();
    let larger = catalog
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(28),
        ))
        .unwrap();
    let mut larger_guide = larger.schedule().clone();
    larger_guide.root = schedule.root.clone();
    let inherited_len = larger_guide.recursive_folds[0]
        .params
        .setup_prefix()
        .unwrap()
        .n_prefix()
        .unwrap();
    let mut accumulated = None;
    for (inherited, guide) in [
        (0, schedule),
        (prefix_len / 2, schedule),
        (inherited_len, &larger_guide),
    ] {
        key.root_accumulated_setup_prefix_len = inherited;
        let planned = find_schedule_in_relation_order(
            &key,
            Recursive::committed_source_contract().unwrap(),
            &[],
            &policy,
            Recursive::ring_challenge_config,
            ScheduleSearchOptions {
                root_main_constraint: Some(&schedule.root.params),
                adaptation_guide: Some(guide),
                ..ScheduleSearchOptions::canonical()
            },
        )
        .unwrap_or_else(|error| panic!("inherited={inherited}: {error}"));
        let prefix = planned.schedule.recursive_folds[0]
            .params
            .setup_prefix()
            .unwrap();
        assert_eq!(prefix.setup_natural_len, Some(local.max(inherited)));
        assert_eq!(
            prefix.n_prefix().unwrap(),
            akita_params::padded_setup_prefix_len(local.max(inherited))
        );
        assert_eq!(planned.schedule.root.params, schedule.root.params);
        if inherited == 0 {
            assert_eq!(planned.schedule, *schedule);
        }
        let ids = akita_config::required_setup_prefix_slot_ids_for_schedule(
            &planned.schedule,
            &key.opening_layout().unwrap(),
        )
        .unwrap();
        assert!(ids.contains(&prefix.slot_id().unwrap()));
        assert_eq!(
            planned.estimate.estimated_proof_payload_bytes().unwrap(),
            akita_schedules::expanded_schedule_proof_estimate_bytes(
                &key,
                &planned.schedule,
                &policy
            )
            .unwrap()
        );
        accumulated = Some(planned.schedule);
    }
    let accumulated = accumulated.unwrap();
    let profiles = CommittedGroupBatchProfile {
        final_group: accumulated.root.params.own_group().profile,
        precommitteds: vec![],
    };
    let catalog = akita_schedules::ValidatedScheduleCatalog::try_new(
        Recursive::schedule_family_name(),
        [
            (baseline.profiles().clone(), schedule.clone()),
            (profiles.clone(), accumulated.clone()),
        ],
        &policy,
        Recursive::ring_challenge_config,
    )
    .unwrap();
    let bytes = catalog.to_artifact_bytes().unwrap();
    let replay = akita_schedules::ValidatedScheduleCatalog::from_artifact_bytes(
        &bytes,
        Recursive::schedule_family_name(),
        &policy,
        Recursive::ring_challenge_config,
    )
    .unwrap();
    let selected = replay.resolve_key(&key).unwrap();
    assert_eq!(selected.schedule(), &accumulated);
    let view = replay.to_verifier_view(&[selected.selection()]).unwrap();
    let verifier = akita_schedules::ValidatedScheduleCatalog::from_verifier_view(
        &view,
        Recursive::schedule_family_name(),
        &policy,
        Recursive::ring_challenge_config,
    )
    .unwrap();
    assert_eq!(
        verifier
            .resolve_selection(selected.selection())
            .unwrap()
            .schedule(),
        &accumulated
    );
    let mut wrong = accumulated;
    wrong.root_accumulated_setup_prefix_len *= 2;
    assert!(ResolvedScheduleRow::try_new(profiles, wrong, &policy).is_err());
    key.root_accumulated_setup_prefix_len = 3;
    assert!(find_schedule(
        &key,
        Recursive::committed_source_contract().unwrap(),
        &[],
        &policy,
        Recursive::ring_challenge_config
    )
    .is_err());
}
