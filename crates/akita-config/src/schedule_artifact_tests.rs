use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::{FoldSchedule, GroupOpenPhaseParams, InnerCommitSecurityRoute, OpeningMethod};

use crate::proof_optimized::fp128;
use crate::{policy_of, CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog};

pub(crate) fn mutated_row_admission_error<Cfg: CommitmentConfig>(
    row: &akita_schedules::ResolvedScheduleRow,
    mutate: impl FnOnce(&mut FoldSchedule),
) -> AkitaError {
    let profiles = row.profiles().clone();
    let mut schedule = row.schedule().clone();
    mutate(&mut schedule);
    akita_schedules::ResolvedScheduleRow::try_new(profiles, schedule, &policy_of::<Cfg>())
        .expect_err("noncanonical row must fail at admission")
}

fn row_with_setup_prefix<Cfg: CommitmentConfig>(
    predicate: impl Fn(&GroupOpenPhaseParams) -> bool,
) -> akita_schedules::ResolvedScheduleRow {
    crate::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog")
        .rows()
        .find(|row| {
            row.schedule()
                .recursive_folds
                .iter()
                .any(|fold| fold.params.setup_prefix().is_some_and(&predicate))
        })
        .cloned()
        .expect("catalog row with matching setup prefix")
}

fn replace_evaluation_trace_challenge(
    group: &mut GroupOpenPhaseParams,
    ring_dimension: usize,
    replacement: SparseChallengeConfig,
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
    replacement: SparseChallengeConfig,
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
fn row_admission_audits_recursive_setup_prefix_policy() {
    type RecursiveOneHot = RecursiveCommitmentConfig<fp128::OneHot>;

    let row = row_with_setup_prefix::<RecursiveOneHot>(|_| true);
    let prefix_index = row
        .schedule()
        .recursive_folds
        .iter()
        .position(|fold| fold.params.setup_prefix().is_some())
        .expect("recursive setup-prefix fold");
    let error = mutated_row_admission_error::<RecursiveOneHot>(&row, |schedule| {
        let fold = schedule
            .recursive_folds
            .get_mut(prefix_index)
            .expect("recursive setup-prefix fold");
        let mut prefix = *fold.params.setup_prefix().expect("setup-prefix group");
        prefix.opening.log_basis_open = 127;
        fold.params
            .set_setup_prefix(Some(prefix))
            .expect("valid prefix topology");
    });
    assert!(
        error.to_string().contains(&format!(
            "recursive fold {prefix_index} setup prefix: opening digit depth is not canonical"
        )),
        "unexpected setup-prefix policy error: {error}"
    );
}

#[test]
fn catalog_binding_revalidates_recursive_setup_prefix_challenge_hook() {
    type RecursiveMultiChunk = RecursiveCommitmentConfig<fp128::OneHotMultiChunk>;

    let row = row_with_setup_prefix::<RecursiveMultiChunk>(|prefix| {
        prefix.opening.opening_method == OpeningMethod::EvaluationTrace
    });
    let (prefix_index, target_prefix) = row
        .schedule()
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
    assert!(actual.count_pm1 >= 2);
    let replacement = SparseChallengeConfig {
        count_pm1: actual.count_pm1 - 2,
        count_pm2: actual.count_pm2 + 1,
    };
    assert_eq!(replacement.l1_norm(), actual.l1_norm());
    assert_ne!(replacement, actual);
    replacement
        .validate_for_ring_dim(ring_dimension)
        .expect("alternate challenge remains structurally admissible");

    let profiles = row.profiles().clone();
    let mut schedule = row.schedule().clone();
    replace_schedule_evaluation_trace_challenges(&mut schedule, ring_dimension, replacement);
    let policy = policy_of::<RecursiveMultiChunk>();
    let catalog = TrustedScheduleCatalog::try_new(
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

    let error = catalog
        .validate_binding(
            RecursiveMultiChunk::schedule_family_name(),
            &policy,
            RecursiveMultiChunk::ring_challenge_config,
        )
        .expect_err("concrete config must reject the alternate prefix challenge");
    assert!(
        error.to_string().contains(&format!(
            "recursive fold {prefix_index} setup prefix challenge config does not match the trusted runtime hook"
        )),
        "unexpected setup-prefix challenge error: {error}"
    );
}
