use crate::report::observed_stage3_setup_product_bytes;
use akita_config::CommitmentConfig;
use akita_serialization::AkitaSerialize;
use akita_types::{AkitaBatchedProof, FoldSchedule, PolynomialGroupLayout, SetupContributionMode};
use jolt_field::{CanonicalEncoding, Field};

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

pub(super) fn assert_observed_proof_size<FF, E>(label: &str, proof: &AkitaBatchedProof<FF, E>)
where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let mut encoded = Vec::with_capacity(proof.size());
    proof
        .serialize_uncompressed(&mut encoded)
        .expect("profile proof serialization should succeed");
    assert_eq!(
        encoded.len(),
        proof.size(),
        "[{label}] proof.size() must match actual uncompressed serialization length"
    );
}

/// Maximum number of bytes by which the planner's header-stripped proof-size
/// estimate is allowed to *exceed* the real serialized proof.
///
/// The offline formula (`akita_types::level_proof_bytes`) assumes every stage-2
/// sumcheck round ships a degree-3 compressed univariate (three challenge-field
/// coefficients). The prover, however, emits a handful of stage-2 rounds at
/// degree 2 — a y-/x-prefix micro-optimization that trims one leading
/// coefficient and that the header-stripped formula deliberately does not
/// model. The real proof is therefore a few challenge elements *smaller* than
/// the estimate, so the estimate stays a conservative upper bound. We accept
/// that small overcount here rather than couple the offline planner to the
/// prover's exact per-round degree schedule. This is a pre-existing inaccuracy
/// (it reproduces on `main` for schedules whose terminal sumcheck folds an
/// odd-shaped witness) and is tracked for a proper fix in
/// `specs/archive/2026-Q2/planner-refactor.md`.
///
/// The overcount scales with the number of stage-2 rounds, so it is largest
/// for small-field / many-level schedules: across the profile-bench matrix the
/// current worst case is adaptive `dense_fp32` nv26 (planned vs runtime tail sizing).
/// The
/// bound covers those with margin. The `actual <= planned` upper-bound check
/// above is the primary guard against a runtime proof that *grew*; a dropped
/// level (which would inflate the overcount) is independently caught by the
/// planned/proof level-count guard in `scripts/profile_bench_report.py`, and
/// absolute proof growth is bounded by the CI proof-size regression threshold.
const ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES: usize = 3072;

fn terminal_response_z_planner_slack<FF, E>(
    proof: &AkitaBatchedProof<FF, E>,
    schedule: &FoldSchedule,
) -> usize
where
    FF: Field,
    E: Field,
{
    schedule
        .terminal
        .response_shape
        .layout
        .z_payload_bytes()
        .saturating_sub(
            proof
                .terminal_response()
                .z_payloads
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
        )
}

/// Check the runtime proof size against a planner estimate, tolerating the
/// small, conservative overcount documented on
/// [`ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES`].
fn assert_runtime_matches_planned_proof_size(
    label: &str,
    actual_bytes: usize,
    planned_bytes: usize,
    source: &str,
    extra_slack: usize,
) {
    assert!(
        actual_bytes <= planned_bytes,
        "[{label}] runtime proof bytes {actual_bytes} exceed the {source} proof size \
         {planned_bytes}; the planner estimate must remain an upper bound"
    );
    let overcount = planned_bytes - actual_bytes;
    let accepted = ACCEPTED_PLANNER_PROOF_SIZE_OVERCOUNT_BYTES.saturating_add(extra_slack);
    assert!(
        overcount <= accepted,
        "[{label}] {source} proof size {planned_bytes} overcounts the runtime proof bytes \
         {actual_bytes} by {overcount} bytes, exceeding the accepted \
         {accepted}-byte tolerance (stage-2 degree-2 rounds plus segment-typed z slack)"
    );
    if overcount != 0 {
        tracing::warn!(
            label,
            actual_bytes,
            planned_bytes,
            overcount,
            "planner proof-size estimate overcounts the runtime proof (stage-2 degree-2 rounds; \
             see specs/planner-refactor.md)"
        );
        eprintln!(
            "[{label}] NOTE: {source} estimate {planned_bytes} overcounts runtime proof \
             {actual_bytes} by {overcount} bytes (stage-2 degree-2 round micro-optimization; \
             accepted, see specs/planner-refactor.md)"
        );
    }
}

/// Compare the runtime proof against the planner estimate.
///
/// The planner prices the **direct-mode** payload only. In direct mode the
/// whole proof is checked against it. In recursive mode the stage-3
/// setup-product bytes are pure overhead layered on top, so they are stripped
/// before the comparison and reported as an explicit delta instead of being
/// asserted against `schedule.total_bytes`.
pub(super) fn report_proof_size_against_planner<FF, E>(
    label: &str,
    proof: &AkitaBatchedProof<FF, E>,
    planned_bytes: usize,
    source: &str,
    mode: SetupContributionMode,
    schedule: &FoldSchedule,
) where
    FF: Field + CanonicalEncoding + AkitaSerialize,
    E: Field + AkitaSerialize,
{
    let z_slack = terminal_response_z_planner_slack(proof, schedule);
    match mode {
        SetupContributionMode::Direct => {
            assert_runtime_matches_planned_proof_size(
                label,
                proof.size(),
                planned_bytes,
                source,
                z_slack,
            );
        }
        SetupContributionMode::Recursive => {
            let stage3_bytes = observed_stage3_setup_product_bytes(proof);
            let direct_equivalent = proof
                .size()
                .checked_sub(stage3_bytes)
                .expect("stage-3 setup-product bytes are a subset of the serialized proof size");
            let recursive_source = format!("{source} (recursive; stage-3 setup-product excluded)");
            assert_runtime_matches_planned_proof_size(
                label,
                direct_equivalent,
                planned_bytes,
                &recursive_source,
                z_slack,
            );
            tracing::info!(
                label,
                observed_total_bytes = proof.size(),
                stage3_setup_product_bytes = stage3_bytes,
                direct_mode_planner_bytes = planned_bytes,
                "recursive setup-product proof size"
            );
            eprintln!(
                "[{label}] recursive setup: observed={} bytes = direct-mode payload {} \
                 (+/- planner overcount vs {source} {}) + stage-3 setup-product {} bytes",
                proof.size(),
                direct_equivalent,
                planned_bytes,
                stage3_bytes,
            );
        }
    }
}
