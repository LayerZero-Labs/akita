use super::*;

use akita_challenges::FoldChallengeFamily;
use akita_config::proof_optimized::fp128::OneHotMultiChunk;
use akita_config::{
    policy_of, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog,
};
use akita_params::{
    CommittedGroupBatchProfile, FoldSchedule, GroupOpenPhaseParams, InnerCommitSecurityRoute,
    MultiChunkProfileId, OpeningMethod,
};

fn replace_evaluation_trace_challenge(
    group: &mut GroupOpenPhaseParams,
    ring_dimension: usize,
    replacement: FoldChallengeFamily,
) {
    if group.opening.opening_method == OpeningMethod::EvaluationTrace
        && group.profile.inner.matrix.ring_dimension() == ring_dimension
        && matches!(
            group.profile.inner.matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        )
    {
        group.opening.fold_challenge_config = replacement;
    }
}

fn replace_schedule_evaluation_trace_challenges(
    schedule: &mut FoldSchedule,
    ring_dimension: usize,
    replacement: FoldChallengeFamily,
) {
    let mut root_precommitted = schedule.root.params.precommitted_groups().to_vec();
    for group in &mut root_precommitted {
        replace_evaluation_trace_challenge(group, ring_dimension, replacement);
    }
    schedule
        .root
        .params
        .set_precommitted_groups(root_precommitted)
        .expect("preserve root group topology");
    replace_evaluation_trace_challenge(
        schedule.root.params.own_group_mut(),
        ring_dimension,
        replacement,
    );

    for fold in &mut schedule.recursive_folds {
        if let Some(mut prefix) = fold.params.setup_prefix().copied() {
            replace_evaluation_trace_challenge(&mut prefix, ring_dimension, replacement);
            fold.params
                .set_setup_prefix(Some(prefix))
                .expect("preserve recursive group topology");
        }
        replace_evaluation_trace_challenge(
            fold.params.own_group_mut(),
            ring_dimension,
            replacement,
        );
    }

    if schedule.terminal.d_a() == ring_dimension
        && matches!(
            schedule.terminal.inner.matrix.security_route(),
            InnerCommitSecurityRoute::Linf(_)
        )
    {
        schedule.terminal.fold_challenge_config = replacement;
    }
}

#[test]
fn catalog_binding_revalidates_recursive_setup_prefix_challenge_hook() {
    type RecursiveMultiChunk = RecursiveCommitmentConfig<OneHotMultiChunk>;

    // No checked-in recursive row consumes a setup prefix with an
    // evaluation-trace opening. The W8R2 recursive nv38 schedule offloads
    // setup into folds 1 and 2, and fold 2 opens its prefix by evaluation trace.
    let chunking = MultiChunkProfileId::W8R2.cfg();
    let policy = policy_of::<RecursiveMultiChunk>(chunking);
    let planned = find_schedule(
        &ScheduleLookupKey::single(PolynomialGroupLayout::singleton(38)),
        RecursiveMultiChunk::committed_source_contract().unwrap(),
        &[],
        &policy,
        RecursiveMultiChunk::ring_challenge_config,
    )
    .expect("W8R2 recursive nv38 schedule");
    let (prefix_index, target_prefix) = planned
        .schedule
        .recursive_folds
        .iter()
        .enumerate()
        .find_map(|(index, fold)| {
            fold.params
                .setup_prefix()
                .filter(|prefix| prefix.opening.opening_method == OpeningMethod::EvaluationTrace)
                .map(|prefix| (index, prefix))
        })
        .expect("evaluation-trace setup prefix");
    let ring_dimension = target_prefix.profile.inner.matrix.ring_dimension();
    let actual = target_prefix.opening.fold_challenge_config;
    assert!(actual.signed_sparse().unwrap().count_pm1 >= 2);
    let replacement = FoldChallengeFamily::SignedSparse(akita_challenges::SparseChallengeConfig {
        count_pm1: actual.signed_sparse().unwrap().count_pm1 - 2,
        count_pm2: actual.signed_sparse().unwrap().count_pm2 + 1,
    });
    assert_eq!(replacement.l1_norm(), actual.l1_norm());
    assert_ne!(replacement, actual);
    replacement
        .validate_for_ring_dim(ring_dimension)
        .expect("alternate challenge remains structurally admissible");

    let profiles = CommittedGroupBatchProfile {
        final_group: planned.schedule.root.params.own_group().profile,
        precommitteds: Vec::new(),
    };
    let mut schedule = planned.schedule;
    replace_schedule_evaluation_trace_challenges(&mut schedule, ring_dimension, replacement);
    let catalog = akita_schedules::ValidatedScheduleCatalog::try_new(
        RecursiveMultiChunk::schedule_family_name(),
        [(profiles, schedule)],
        &policy,
        |dimension| {
            if dimension == ring_dimension {
                Ok(replacement)
            } else {
                RecursiveMultiChunk::ring_challenge_config(dimension)
            }
        },
    )
    .expect("catalog admitted under its alternate challenge hook");

    let error = TrustedScheduleCatalog::<RecursiveMultiChunk>::new(catalog, chunking)
        .expect_err("concrete config must reject the alternate prefix challenge");
    assert!(
        error.to_string().contains(&format!(
            "recursive fold {prefix_index} setup prefix challenge config does not match the trusted runtime hook"
        )),
        "unexpected setup-prefix challenge error: {error}"
    );
}
