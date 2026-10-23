use akita_error::AkitaError;
use akita_params::{FoldSchedule, GroupOpenPhaseParams};

use crate::proof_optimized::fp128;
use crate::{policy_of, CommitmentConfig, RecursiveCommitmentConfig};

pub(crate) fn mutated_row_admission_error<Cfg: CommitmentConfig>(
    row: &akita_schedules::ResolvedScheduleRow,
    mutate: impl FnOnce(&mut FoldSchedule),
    chunking: akita_params::ChunkedWitnessCfg,
) -> AkitaError {
    let profiles = row.profiles().clone();
    let mut schedule = row.schedule().clone();
    mutate(&mut schedule);
    akita_schedules::ResolvedScheduleRow::try_new(profiles, schedule, &policy_of::<Cfg>(chunking))
        .expect_err("noncanonical row must fail at admission")
}

fn row_with_setup_prefix<Cfg: CommitmentConfig>(
    predicate: impl Fn(&GroupOpenPhaseParams) -> bool,
    chunking: akita_params::ChunkedWitnessCfg,
) -> akita_schedules::ResolvedScheduleRow {
    crate::test_support::workspace_schedule_catalog::<Cfg>(chunking)
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

#[test]
fn row_admission_audits_recursive_setup_prefix_policy() {
    type RecursiveOneHot = RecursiveCommitmentConfig<fp128::OneHot>;

    let row = row_with_setup_prefix::<RecursiveOneHot>(
        |_| true,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    );
    let prefix_index = row
        .schedule()
        .recursive_folds
        .iter()
        .position(|fold| fold.params.setup_prefix().is_some())
        .expect("recursive setup-prefix fold");
    let error = mutated_row_admission_error::<RecursiveOneHot>(
        &row,
        |schedule| {
            let fold = schedule
                .recursive_folds
                .get_mut(prefix_index)
                .expect("recursive setup-prefix fold");
            let mut prefix = *fold.params.setup_prefix().expect("setup-prefix group");
            prefix.opening.num_digits_open -= 1;
            fold.params
                .set_setup_prefix(Some(prefix))
                .expect("valid prefix topology");
        },
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    );
    assert!(
        error.to_string().contains(&format!(
            "recursive fold {prefix_index} setup prefix: opening digit depth is not canonical"
        )),
        "unexpected setup-prefix policy error: {error}"
    );
}

#[test]
fn dense_recursive_artifacts_cover_benchmark_sizes_and_offload_setup() {
    use crate::proof_optimized::{fp32, fp64};
    use crate::recursive_commitment::RecursiveScheduleConfig;
    use akita_params::PolynomialGroupLayout;
    use akita_params::ScheduleLookupKey;

    fn check<Cfg: RecursiveScheduleConfig>(sizes: &[usize]) {
        type Recursive<C> = RecursiveCommitmentConfig<C>;
        let catalog = crate::test_support::workspace_schedule_catalog::<Recursive<Cfg>>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .expect("dense recursive artifact admitted under current policy");
        assert_eq!(catalog.rows().count(), sizes.len());
        let mut offloaded = 0;
        for &num_vars in sizes {
            let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(num_vars));
            let row = catalog.resolve_key(&key).expect("benchmark size present");
            if row
                .schedule()
                .recursive_folds
                .iter()
                .any(|fold| fold.params.setup_prefix().is_some())
            {
                offloaded += 1;
            }
        }
        assert!(
            offloaded > 0,
            "dense recursive family must actually offload setup"
        );
    }

    check::<fp32::Dense>(&[20, 22, 24, 26, 28, 30]);
    check::<fp64::Dense>(&[21, 23, 25, 27, 29]);
    check::<fp128::Dense>(&[20, 22, 24, 26, 28]);
}

#[test]
fn row_admission_rejects_a_setup_prefix_without_full_width_inner_digits() {
    type RecursiveOneHot = RecursiveCommitmentConfig<fp128::OneHot>;
    let row = row_with_setup_prefix::<RecursiveOneHot>(
        |_| true,
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    );
    let (fold_index, prefix) = row
        .schedule()
        .recursive_folds
        .iter()
        .enumerate()
        .find_map(|(index, fold)| fold.params.setup_prefix().copied().map(|p| (index, p)))
        .expect("recursive setup prefix");
    prefix
        .profile
        .validate_setup_prefix_geometry(prefix.setup_natural_len.expect("setup prefix support"))
        .expect("generated prefixes use the full field width");

    // Preserve matrix geometry while truncating the setup decomposition.
    let mut truncated = prefix;
    truncated.profile.inner.digits.num_digits = 1;
    truncated.profile.inner.matrix = truncated
        .profile
        .inner
        .matrix
        .try_with_input_width(truncated.profile.blocks.positions_per_block)
        .expect("one-digit A width");
    assert!(truncated.validate().is_err());
    let slot = truncated.slot_id().expect("setup prefix slot");
    assert!(akita_serialization::Valid::check(&slot).is_err());
    assert!(akita_params::setup_prefix_slot_field_elements(&slot).is_err());
    for log_basis in [0, 128, u32::MAX] {
        let mut invalid = prefix.profile;
        invalid.inner.digits.log_basis = log_basis;
        assert!(invalid
            .validate_setup_prefix_geometry(slot.natural_len)
            .is_err());
    }
    assert!(truncated
        .profile
        .validate_setup_prefix_geometry(prefix.setup_natural_len.expect("setup prefix support"))
        .is_err());

    let error = mutated_row_admission_error::<RecursiveOneHot>(
        &row,
        |schedule| {
            schedule.recursive_folds[fold_index]
                .params
                .set_setup_prefix(Some(truncated))
                .expect("preserve recursive group topology");
        },
        akita_params::ChunkedWitnessCfg::default_non_chunked(),
    );
    assert!(
        error.to_string().contains("full field width"),
        "unexpected admission error: {error}"
    );
}
