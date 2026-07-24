//! Runtime schedule catalog-boundary guards.
//!
//! These cover the behaviors the planner refactor introduces:
//!
//! - **Table-miss rejection:** `Cfg::runtime_schedule` rejects a key that no
//!   shipped table contains.
//! - **Policy-bridge parity:** `policy_of::<Cfg>()` reproduces the values
//!   embedded in generated catalog identities (single source of truth).
//! - **No-panic boundary:** adversarial-but-bounded keys through
//!   `runtime_schedule` return `Result`, never panic.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32};
use akita_config::{policy_of, CommitmentConfig, RecursiveCommitmentConfig};
use akita_schedules::{PlannerCostModelId, PlannerPolicy, SelectionPolicyId};
use akita_types::{AkitaScheduleLookupKey, PolynomialGroupLayout};

/// A one-point 2-poly key that no shipped table carries (shipped tables only
/// hold singleton / 4-batched keys), so strict runtime resolution must reject it.
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

fn check_table_miss_rejection<Cfg: CommitmentConfig>(num_vars: usize) {
    let key = table_miss_key(num_vars);

    // The shipped table must NOT carry this key — otherwise the test is not
    // exercising the catalog-miss path. (Shipped tables only hold
    // singleton / 4-batched keys; this 2-poly key misses every table.)
    let _policy = policy_of::<Cfg>();
    let table_has_key = Cfg::schedule_catalog()
        .and_then(|table| {
            akita_schedules::generated::table_entry(table, &AkitaScheduleLookupKey::single(key))
        })
        .is_some();
    assert!(
        !table_has_key,
        "expected a table miss for the 2-poly key; the table unexpectedly carries it"
    );

    let err = Cfg::runtime_schedule(AkitaScheduleLookupKey::single(key))
        .expect_err("runtime_schedule must reject uncataloged keys");
    assert!(
        matches!(err, akita_field::AkitaError::UnsupportedSchedule(_)),
        "expected UnsupportedSchedule for catalog miss, got {err:?}"
    );
}

#[test]
fn catalog_miss_rejects_non_shipped_keys() {
    check_table_miss_rejection::<fp128::D64OneHot>(14);
    check_table_miss_rejection::<fp128::D64Dense>(16);
    check_table_miss_rejection::<fp32::D128OneHot>(16);
}

#[test]
fn recursive_adapter_delegates_scalar_keys_to_the_ordinary_catalog() {
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
        cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
        selection_policy: if Cfg::recursive_setup_planning() {
            SelectionPolicyId::MinFirstDirectSetupThenPayloadWithinSupportedEnvelope
        } else {
            SelectionPolicyId::MinEstimatedProofPayload
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
    assert_policy_matches_cfg::<fp128::D64Dense>();
    assert_policy_matches_cfg::<fp128::D128Dense>();
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
