use akita_config::CommitmentConfig;
use akita_types::{FoldSchedule, PolynomialGroupLayout, SetupContributionMode};

pub(super) fn planned_payload_bytes<Cfg: CommitmentConfig>(
    schedule: &FoldSchedule,
    final_group: PolynomialGroupLayout,
) -> usize {
    let key = akita_types::AkitaScheduleLookupKey {
        final_group,
        precommitteds: schedule
            .root
            .params
            .precommitted_groups()
            .iter()
            .map(|group| group.profile)
            .collect(),
    };
    akita_schedules::expanded_schedule_proof_payload_bytes(
        &key,
        schedule,
        &akita_config::policy_of::<Cfg>(),
    )
    .expect("expanded schedule estimate")
}

pub(super) fn assert_observed_proof_size(label: &str, proof: &[u8]) {
    assert!(
        !proof.is_empty(),
        "[{label}] native proof must not be empty"
    );
}

/// The planner models the native Spongefish argument stream directly. A small
/// overcount remains possible because some stage-2 rounds realize degree two
/// although the static schedule prices the degree-three upper bound.
const ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES: usize = 3072;

pub(super) fn report_proof_size_against_planner(
    label: &str,
    proof: &[u8],
    planned_bytes: usize,
    source: &str,
    mode: SetupContributionMode,
    _schedule: &FoldSchedule,
) {
    let actual_bytes = proof.len();
    assert!(
        actual_bytes <= planned_bytes,
        "[{label}] native proof bytes {actual_bytes} exceed the {source} estimate {planned_bytes}"
    );
    let overcount = planned_bytes - actual_bytes;
    assert!(
        overcount <= ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES,
        "[{label}] {source} estimate overcounts the native proof by {overcount} bytes"
    );
    tracing::info!(
        label,
        actual_bytes,
        planned_bytes,
        overcount,
        ?mode,
        "native proof-size comparison"
    );
    eprintln!(
        "[{label}] proof_size: native={actual_bytes} bytes, planned={planned_bytes} bytes, \
         overcount={overcount} bytes, setup_contribution_mode={mode:?}"
    );
}
