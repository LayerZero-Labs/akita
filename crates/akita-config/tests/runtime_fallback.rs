//! Runtime schedule DP-fallback guards for the `Cfg`-free planner.
//!
//! These cover the behaviors the planner refactor introduces:
//!
//! - **Table-miss fallback:** `Cfg::runtime_schedule` returns `Some` for a
//!   key that no shipped table contains, and the schedule it returns is
//!   exactly what the pure DP `akita_planner::find_group_batch_schedule` produces from
//!   the `Cfg`-derived policy.
//! - **Policy-bridge parity:** `policy_of::<Cfg>()` reproduces the values
//!   the DP reads off the `Cfg` impl (invariant 4, single source of truth).
//! - **No-panic boundary:** adversarial-but-bounded keys through
//!   `runtime_schedule` return `Result`, never panic.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32};
use akita_config::{policy_of, CommitmentConfig, RecursiveCommitmentConfig};
use akita_planner::{find_group_batch_schedule, PlannerPolicy};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

/// A one-point 2-poly key that no shipped table carries (shipped tables only
/// hold singleton / 4-batched keys), so it forces the DP fallback path on both
/// prover and verifier.
fn table_miss_key(num_vars: usize) -> PolynomialGroupLayout {
    PolynomialGroupLayout::new(num_vars, 2)
}

fn assert_schedule_eq(
    label: &str,
    lhs: &akita_types::FoldSchedule,
    rhs: &akita_types::FoldSchedule,
) {
    assert_eq!(
        format!("{:?}", lhs.root),
        format!("{:?}", rhs.root),
        "{label}: root diverges"
    );
    assert_eq!(
        format!("{:?}", lhs.recursive_folds),
        format!("{:?}", rhs.recursive_folds),
        "{label}: recursive folds diverge"
    );
    assert_eq!(
        lhs.terminal.input_witness_len, rhs.terminal.input_witness_len,
        "{label}: terminal witness lengths diverge"
    );
    assert_eq!(
        lhs.terminal.params.response_shape, rhs.terminal.params.response_shape,
        "{label}: terminal witness shapes diverge"
    );
}

fn check_table_miss_fallback<Cfg: CommitmentConfig>(num_vars: usize) {
    let key = table_miss_key(num_vars);

    // The shipped table must NOT carry this key — otherwise the test is not
    // exercising the DP fallback path. (Shipped tables only hold
    // singleton / 4-batched keys; this 2-poly key misses every table.)
    let _policy = policy_of::<Cfg>();
    let table_has_key = Cfg::schedule_catalog()
        .and_then(|table| {
            akita_planner::generated::table_entry(table, &AkitaScheduleLookupKey::single(key))
        })
        .is_some();
    assert!(
        !table_has_key,
        "expected a table miss for the 2-poly key; the table unexpectedly carries it"
    );

    let from_runtime = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(key))
        .expect("runtime_schedule must not error on a valid one-point 2-poly key");

    let from_dp = find_group_batch_schedule(
        &AkitaScheduleLookupKey::single(key),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
    .expect("pure DP must succeed for a valid key");

    assert_schedule_eq("table-miss fallback", &from_runtime, &from_dp.schedule);
}

#[test]
fn dp_fallback_fires_for_non_shipped_keys() {
    check_table_miss_fallback::<fp128::D64OneHot>(14);
    check_table_miss_fallback::<fp128::D64Full>(16);
    check_table_miss_fallback::<fp32::D128OneHot>(16);
}

#[test]
fn recursive_adapter_delegates_scalar_keys_to_the_ordinary_planner() {
    let key = AkitaScheduleLookupKey::single(PolynomialGroupLayout::singleton(18));
    let ordinary = fp128::D64OneHot::runtime_schedule(key.clone())
        .expect("ordinary scalar schedule must resolve");
    let recursive = RecursiveCommitmentConfig::<fp128::D64OneHot>::runtime_schedule(key)
        .expect("recursive adapter scalar schedule must resolve");
    assert_schedule_eq("recursive scalar delegation", &ordinary, &recursive);
}

fn assert_policy_matches_cfg<Cfg: CommitmentConfig>() {
    let policy = policy_of::<Cfg>();
    let expected = PlannerPolicy {
        cost_model: akita_planner::PlannerCostModelId::ExactPayloadAndSetupEnvelope,
        selection_policy: if Cfg::recursive_setup_planning() {
            akita_planner::SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope
        } else {
            akita_planner::SelectionPolicyId::MinEstimatedProofPayload
        },
        max_setup_envelope_field_elements: akita_types::MAX_SETUP_MATRIX_FIELD_ELEMENTS,
        min_offloaded_witness_contraction: 3,
        ring_dimension: Cfg::D,
        decomposition: Cfg::decomposition(),
        sis_modulus_profile: Cfg::sis_modulus_profile(),
        sis_security_policy: akita_types::DEFAULT_SIS_SECURITY_POLICY,
        sis_table_digest: akita_types::SisTableDigest::CURRENT,
        ring_subfield_norm_bound: Cfg::ring_subfield_embedding_norm_bound(),
        claim_ext_degree: Cfg::EXT_DEGREE,
        chal_ext_degree: Cfg::EXT_DEGREE,
        basis_range: Cfg::basis_range(),
        onehot_chunk_size: Cfg::onehot_chunk_size(),
        witness_chunk: Cfg::chunked_witness_cfg(),
        recursive_setup_planning: Cfg::recursive_setup_planning(),
    };
    assert_eq!(
        policy, expected,
        "policy_of must derive every field from the Cfg impl"
    );
}

#[test]
fn policy_bridge_matches_cfg_hooks() {
    assert_policy_matches_cfg::<fp128::D64Full>();
    assert_policy_matches_cfg::<fp128::D128Full>();
    assert_policy_matches_cfg::<fp128::D64OneHot>();
    assert_policy_matches_cfg::<fp32::D64OneHot>();
}

#[test]
fn runtime_schedule_never_panics_on_bounded_adversarial_keys() {
    // Degenerate vector counts must be rejected with `AkitaError`, not by
    // panicking. Large-but-bounded
    // `num_vars` must terminate (no unbounded blow-up) and return a result.
    let adversarial = [
        PolynomialGroupLayout::new(10, 0),
        PolynomialGroupLayout::new(0, 1),
        PolynomialGroupLayout::new(40, 1),
    ];
    for key in adversarial {
        // Must return without panicking; either branch (Ok/Err) is fine.
        let _ = fp128::D64OneHot::runtime_schedule(AkitaScheduleLookupKey::single(key));
    }
}
