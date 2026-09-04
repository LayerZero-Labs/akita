use super::*;

use akita_config::{
    honest_fold_policy_of, policy_of,
    proof_optimized::fp128::{Dense, OneHot},
    CommitmentConfig,
};
use akita_types::CommittedGroupBatchProfile;

fn producer<Cfg: CommitmentConfig>(
    profile: akita_types::GroupCommitPhaseParams,
) -> crate::emit::PrecommittedProducer {
    crate::emit::PrecommittedProducer::try_new(
        profile,
        Cfg::committed_source_contract().expect("producer source contract"),
        honest_fold_policy_of::<Cfg>(),
    )
    .expect("valid precommitted producer")
}

fn grouped_request<Cfg: CommitmentConfig>(
    final_group: PolynomialGroupLayout,
    profiles: &[akita_types::GroupCommitPhaseParams],
) -> crate::emit::GroupedGenerationRequest {
    crate::emit::GroupedGenerationRequest::new(
        final_group,
        profiles.iter().copied().map(producer::<Cfg>).collect(),
    )
}

fn assert_frozen_skeleton(main: &akita_types::FoldSchedule, adapted: &akita_types::FoldSchedule) {
    assert_eq!(adapted.recursive_folds.len(), main.recursive_folds.len());
    for (adapted, main) in adapted.recursive_folds.iter().zip(&main.recursive_folds) {
        assert_eq!(adapted.params.role_dims(), main.params.role_dims());
        assert_eq!(
            adapted.params.blocks().positions_per_block,
            main.params.blocks().positions_per_block
        );
        assert_eq!(
            adapted.params.outer_slice_count(),
            main.params.outer_slice_count()
        );
        assert_eq!(
            adapted.params.inner().digits.log_basis,
            main.params.inner().digits.log_basis
        );
        assert_eq!(
            adapted.params.open().digits.log_basis,
            main.params.open().digits.log_basis
        );
        assert_eq!(
            adapted.params.opening_method(),
            main.params.opening_method()
        );
        assert_eq!(
            adapted.params.fold_challenge_config(),
            main.params.fold_challenge_config()
        );
        assert_eq!(adapted.params.payload_mode, main.params.payload_mode);
        assert_eq!(
            adapted.params.ring_relation_mode,
            main.params.ring_relation_mode
        );
        assert_eq!(adapted.params.source_encoding, main.params.source_encoding);
        assert_eq!(adapted.params.witness_chunk, main.params.witness_chunk);
        assert_eq!(
            adapted.params.setup_prefix().is_some(),
            main.params.setup_prefix().is_some()
        );
        if let (Some(adapted), Some(main)) =
            (adapted.params.setup_prefix(), main.params.setup_prefix())
        {
            assert_eq!(
                adapted.profile.inner.matrix.ring_dimension(),
                main.profile.inner.matrix.ring_dimension()
            );
            assert_eq!(
                adapted.profile.outer.matrix.ring_dimension(),
                main.profile.outer.matrix.ring_dimension()
            );
            assert_eq!(
                adapted.profile.blocks.positions_per_block,
                main.profile.blocks.positions_per_block
            );
            assert_eq!(
                adapted.profile.outer_slice_count,
                main.profile.outer_slice_count
            );
            assert_eq!(
                adapted.profile.inner.digits.log_basis,
                main.profile.inner.digits.log_basis
            );
            assert_eq!(adapted.opening.opening_method, main.opening.opening_method);
            assert_eq!(
                adapted.opening.fold_challenge_config,
                main.opening.fold_challenge_config
            );
            assert_eq!(adapted.opening.log_basis_open, main.opening.log_basis_open);
        }
    }
    assert_eq!(adapted.terminal.d_a(), main.terminal.d_a());
    assert_eq!(
        adapted.terminal.blocks.positions_per_block,
        main.terminal.blocks.positions_per_block
    );
    assert_eq!(
        adapted.terminal.inner.digits.log_basis,
        main.terminal.inner.digits.log_basis
    );
    assert_eq!(
        adapted.terminal.fold.log_basis,
        main.terminal.fold.log_basis
    );
}

fn scalar_row(group: PolynomialGroupLayout) -> Result<ResolvedScheduleRow, AkitaError> {
    akita_config::test_support::workspace_schedule_catalog::<Dense>()?
        .resolve_key(&AkitaScheduleLookupKey::single(group))
}

#[test]
fn adapted_schedule_freezes_main_root_and_rebuilds_grouped_suffix() {
    let policy = policy_of::<Dense>();
    let main_group = PolynomialGroupLayout::singleton(24);
    let main_row = scalar_row(main_group).expect("scalar main row");
    let pre_group = PolynomialGroupLayout::singleton(14);
    let pre_row = scalar_row(pre_group).expect("scalar precommit row");
    let pre_profile = pre_row.profiles().final_group;
    let key = AkitaScheduleLookupKey {
        final_group: main_group,
        precommitteds: vec![pre_profile],
    };
    let request = grouped_request::<Dense>(main_group, &key.precommitteds);

    let adapted = find_adapted_schedule(
        &main_row,
        &request,
        honest_fold_policy_of::<Dense>(),
        &policy,
        Dense::ring_challenge_config,
    )
    .expect("adapted grouped schedule");

    assert_eq!(
        adapted.schedule.root.params.own_group(),
        main_row.schedule().root.params.own_group(),
        "the main group's complete root A/B and opening plan must stay frozen",
    );
    assert_eq!(adapted.schedule.root.params.precommitted_groups().len(), 1);
    assert!(
        !adapted.schedule.recursive_folds.is_empty(),
        "a grouped root must retain a recursive child"
    );
    assert_ne!(
        adapted.schedule.root.output_witness_len,
        main_row.schedule().root.output_witness_len,
        "the grouped relation must recompute its successor witness"
    );
    assert_frozen_skeleton(main_row.schedule(), &adapted.schedule);

    let profiles = CommittedGroupBatchProfile {
        final_group: main_row.profiles().final_group,
        precommitteds: vec![pre_profile],
    };
    ResolvedScheduleRow::try_new(profiles, adapted.schedule, &policy)
        .expect("adapted row must pass the canonical runtime audit");
}

#[test]
fn adapted_schedule_rejects_non_grouped_or_mismatched_requests() {
    let policy = policy_of::<Dense>();
    let main_group = PolynomialGroupLayout::singleton(14);
    let main_row = scalar_row(main_group).expect("scalar main row");

    let scalar_request = crate::emit::GroupedGenerationRequest::new(main_group, Vec::new());
    let scalar_error = find_adapted_schedule(
        &main_row,
        &scalar_request,
        honest_fold_policy_of::<Dense>(),
        &policy,
        Dense::ring_challenge_config,
    )
    .expect_err("adaptation without precommits must fail");
    assert!(matches!(scalar_error, AkitaError::InvalidInput(_)));

    let mismatch = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(15),
        precommitteds: vec![main_row.profiles().final_group],
    };
    let mismatch_request = grouped_request::<Dense>(mismatch.final_group, &mismatch.precommitteds);
    let mismatch_error = find_adapted_schedule(
        &main_row,
        &mismatch_request,
        honest_fold_policy_of::<Dense>(),
        &policy,
        Dense::ring_challenge_config,
    )
    .expect_err("a different main layout must fail");
    assert!(matches!(mismatch_error, AkitaError::InvalidInput(_)));
}

#[test]
fn adapted_schedule_fails_when_the_frozen_suffix_cannot_absorb_the_change() {
    let policy = policy_of::<Dense>();
    let main_group = PolynomialGroupLayout::singleton(14);
    let main_row = scalar_row(main_group).expect("scalar main row");
    let pre_profile = scalar_row(PolynomialGroupLayout::singleton(32))
        .expect("large scalar precommit row")
        .profiles()
        .final_group;
    let key = AkitaScheduleLookupKey {
        final_group: main_group,
        precommitteds: vec![pre_profile; 4],
    };
    let request = grouped_request::<Dense>(main_group, &key.precommitteds);

    let error = find_adapted_schedule(
        &main_row,
        &request,
        honest_fold_policy_of::<Dense>(),
        &policy,
        Dense::ring_challenge_config,
    )
    .expect_err("four large precommits need a different suffix depth");
    assert!(matches!(error, AkitaError::UnsupportedSchedule(_)));
}

#[test]
fn adapted_schedule_rebuilds_a_checked_in_onehot_group_shape() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
        .expect("one-hot catalog");
    let final_group = PolynomialGroupLayout::singleton(20);
    let reference = catalog
        .rows()
        .find(|row| {
            row.profiles().final_group.group == final_group
                && row.profiles().precommitteds.len() == 1
                && row.profiles().precommitteds[0].group == PolynomialGroupLayout::singleton(14)
        })
        .cloned()
        .expect("checked-in grouped reference row");
    let main_row = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(final_group))
        .expect("standalone main row");
    let key = AkitaScheduleLookupKey {
        final_group,
        precommitteds: reference.profiles().precommitteds.clone(),
    };
    let request = grouped_request::<OneHot>(final_group, &key.precommitteds);
    let policy = policy_of::<OneHot>();

    let adapted = find_adapted_schedule(
        &main_row,
        &request,
        honest_fold_policy_of::<OneHot>(),
        &policy,
        OneHot::ring_challenge_config,
    )
    .expect("guided one-hot adaptation");
    assert_frozen_skeleton(main_row.schedule(), &adapted.schedule);

    ResolvedScheduleRow::try_new(
        CommittedGroupBatchProfile {
            final_group: main_row.profiles().final_group,
            precommitteds: key.precommitteds,
        },
        adapted.schedule,
        &policy,
    )
    .expect("adapted one-hot row audit");
}

#[test]
fn adapted_schedule_preserves_recursive_setup_offload_topology() {
    type RecursiveOneHot = akita_config::RecursiveCommitmentConfig<OneHot>;

    let recursive_catalog =
        akita_config::test_support::workspace_schedule_catalog::<RecursiveOneHot>()
            .expect("recursive one-hot catalog");
    let main_group = PolynomialGroupLayout::singleton(36);
    let main_row = recursive_catalog
        .resolve_key(&AkitaScheduleLookupKey::single(main_group))
        .expect("recursive standalone main row");
    assert!(main_row
        .schedule()
        .recursive_folds
        .iter()
        .any(|fold| fold.params.setup_prefix().is_some()));
    let pre_profile = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
        .expect("one-hot catalog")
        .resolve_key(&AkitaScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(14),
        ))
        .expect("standalone precommit row")
        .profiles()
        .final_group;
    let key = AkitaScheduleLookupKey {
        final_group: main_group,
        precommitteds: vec![pre_profile],
    };
    let request = grouped_request::<OneHot>(main_group, &key.precommitteds);
    let policy = policy_of::<RecursiveOneHot>();

    let adapted = find_adapted_schedule(
        &main_row,
        &request,
        honest_fold_policy_of::<RecursiveOneHot>(),
        &policy,
        RecursiveOneHot::ring_challenge_config,
    )
    .expect("guided recursive adaptation");
    assert_frozen_skeleton(main_row.schedule(), &adapted.schedule);

    assert_eq!(
        adapted
            .schedule
            .recursive_folds
            .iter()
            .map(|fold| fold.params.setup_prefix().is_some())
            .collect::<Vec<_>>(),
        main_row
            .schedule()
            .recursive_folds
            .iter()
            .map(|fold| fold.params.setup_prefix().is_some())
            .collect::<Vec<_>>()
    );
    ResolvedScheduleRow::try_new(
        CommittedGroupBatchProfile {
            final_group: main_row.profiles().final_group,
            precommitteds: key.precommitteds,
        },
        adapted.schedule,
        &policy,
    )
    .expect("adapted recursive row audit");
}

#[test]
#[ignore = "manual cold guided-adaptation benchmark"]
fn benchmark_adapted_schedule_against_checked_in_full_plans() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
        .expect("one-hot catalog");
    let dense_catalog =
        akita_config::test_support::workspace_schedule_catalog::<Dense>().expect("dense catalog");
    let policy = policy_of::<OneHot>();
    let cases = [
        ("20x1+14x1", (20, 1), &[(14, 1, false)][..]),
        ("20x2+14x1", (20, 2), &[(14, 1, false)][..]),
        (
            "20x4+14x1x2",
            (20, 4),
            &[(14, 1, false), (14, 1, false)][..],
        ),
        (
            "20x4+14x1x3",
            (20, 4),
            &[(14, 1, false), (14, 1, false), (14, 1, false)][..],
        ),
        ("20x1+14x2", (20, 1), &[(14, 2, false)][..]),
        ("16x1+14x1", (16, 1), &[(14, 1, false)][..]),
        (
            "16x1+14x1+15x2",
            (16, 1),
            &[(14, 1, false), (15, 2, true)][..],
        ),
    ];

    eprintln!(
        "case\tstatus\tmicros\tadapted_bytes\tfull_bytes\tratio_ppm\tadapted_folds\tfull_folds"
    );
    for (name, (main_vars, main_polys), pre_layouts) in cases {
        let main_group = PolynomialGroupLayout::new(main_vars, main_polys);
        let main_row = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(main_group))
            .expect("standalone main row");
        let precommitteds = pre_layouts
            .iter()
            .map(|&(vars, polys, dense)| {
                let producer_catalog = if dense { &dense_catalog } else { &catalog };
                producer_catalog
                    .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                        vars, polys,
                    )))
                    .expect("standalone producer row")
                    .profiles()
                    .final_group
            })
            .collect::<Vec<_>>();
        let key = AkitaScheduleLookupKey {
            final_group: main_group,
            precommitteds,
        };
        let reference = catalog
            .resolve_key(&key)
            .expect("checked-in full-plan reference row");
        let producers = key
            .precommitteds
            .iter()
            .copied()
            .zip(pre_layouts)
            .map(|(profile, &(_, _, dense))| {
                if dense {
                    producer::<Dense>(profile)
                } else {
                    producer::<OneHot>(profile)
                }
            })
            .collect::<Vec<_>>();
        let request = crate::emit::GroupedGenerationRequest::new(main_group, producers);
        let started = std::time::Instant::now();
        let result = find_adapted_schedule(
            &main_row,
            &request,
            honest_fold_policy_of::<OneHot>(),
            &policy,
            OneHot::ring_challenge_config,
        );
        let micros = started.elapsed().as_micros();
        match result {
            Ok(adapted) => {
                let adapted_bytes = akita_schedules::expanded_schedule_proof_payload_bytes(
                    &key,
                    &adapted.schedule,
                    &policy,
                )
                .expect("adapted payload bytes");
                let full_bytes = akita_schedules::expanded_schedule_proof_payload_bytes(
                    &key,
                    reference.schedule(),
                    &policy,
                )
                .expect("full-plan payload bytes");
                let ratio_ppm = adapted_bytes * 1_000_000 / full_bytes;
                eprintln!(
                    "{name}\tok\t{micros}\t{adapted_bytes}\t{full_bytes}\t{ratio_ppm}\t{}\t{}",
                    adapted.schedule.num_fold_levels(),
                    reference.schedule().num_fold_levels(),
                );
            }
            Err(error) => eprintln!("{name}\tfail:{error}\t{micros}\t-\t-\t-\t-\t-"),
        }
    }
}
