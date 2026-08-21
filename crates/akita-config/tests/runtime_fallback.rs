//! Runtime schedule catalog-boundary guards.
//!
//! These cover the behaviors the planner refactor introduces:
//!
//! - **Table-miss rejection:** `Cfg::resolve_catalog_row_for_key` rejects a key that no
//!   generated table contains.
//! - **Policy-bridge parity:** `policy_of::<Cfg>()` reproduces the values
//!   embedded in generated catalog identities (single source of truth).
//! - **No-panic boundary:** adversarial-but-bounded keys through
//!   `resolve_catalog_row_for_key` return `Result`, never panic.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32};
use akita_config::{policy_of, CommitmentConfig};
use akita_planner::find_schedule;
use akita_schedules::{
    resolve_generated_catalog_row_for_key, resolve_generated_schedule_selection,
    PlannerCostModelId, PlannerPolicy, ResolvedScheduleRow,
};
use akita_types::{
    AkitaScheduleLookupKey, CommitmentRingDims, GroupCommitPhaseParams, PolynomialGroupLayout,
};

/// A one-point 3-poly key that no generated table carries (generated tables only
/// hold singleton / 2-batched / 4-batched keys), so strict runtime resolution
/// must reject it.
fn table_miss_key(num_vars: usize) -> PolynomialGroupLayout {
    PolynomialGroupLayout::new(num_vars, 3)
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
        lhs.terminal.response_shape, rhs.terminal.response_shape,
        "{label}: terminal witness shapes diverge"
    );
}

fn check_table_miss_rejection<Cfg: CommitmentConfig>(num_vars: usize) {
    let key = table_miss_key(num_vars);

    // The generated table must NOT carry this key — otherwise the test is not
    // exercising the catalog-miss path. Generated tables only hold singleton /
    // 2-batched / 4-batched scalar keys; this 3-poly key misses every table.
    let _policy = policy_of::<Cfg>();
    let table_has_key = Cfg::schedule_catalog()
        .and_then(|table| {
            akita_schedules::generated::table_entry(table, &AkitaScheduleLookupKey::single(key))
        })
        .is_some();
    assert!(
        !table_has_key,
        "expected a table miss for the 3-poly key; the table unexpectedly carries it"
    );

    let err = Cfg::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(key))
        .expect_err("resolve_catalog_row_for_key must reject uncataloged keys");
    assert!(
        matches!(err, akita_error::AkitaError::UnsupportedSchedule(_)),
        "expected UnsupportedSchedule for catalog miss, got {err:?}"
    );
}

#[test]
fn catalog_miss_rejects_non_shipped_keys() {
    check_table_miss_rejection::<fp128::OneHot>(27);
    check_table_miss_rejection::<fp128::Dense>(27);
    check_table_miss_rejection::<fp32::OneHot>(17);
}

#[test]
fn fixed_width_selection_resolves_the_same_exact_generated_row() {
    type Cfg = fp128::Dense;

    let catalog = Cfg::schedule_catalog().expect("dense schedule catalog");
    let entry = catalog.entries.first().expect("nonempty generated catalog");
    let key = entry.to_runtime_lookup_key();
    let selected = resolve_generated_catalog_row_for_key(
        &key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Some(catalog),
    )
    .expect("prover selects exact generated row");
    let resolved = resolve_generated_schedule_selection(
        selected.selection(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Some(catalog),
    )
    .expect("verifier resolves public row selection");

    assert_eq!(resolved.selection(), selected.selection());
    assert_eq!(resolved.profiles(), selected.profiles());
    assert_eq!(
        resolved.schedule().canonical_descriptor_bytes(),
        selected.schedule().canonical_descriptor_bytes()
    );

    let mut unknown = selected.selection();
    unknown.row_digest = akita_types::ScheduleRowDigest::from_bytes([0xff; 32]);
    assert!(matches!(
        resolve_generated_schedule_selection(
            unknown,
            &policy_of::<Cfg>(),
            Cfg::ring_challenge_config,
            Some(catalog),
        ),
        Err(akita_error::AkitaError::UnsupportedSchedule(_))
    ));
}

#[test]
fn cached_catalog_rows_do_not_bypass_runtime_hook_validation() {
    type Cfg = fp128::Dense;

    let catalog = Cfg::schedule_catalog().expect("dense schedule catalog");
    let entry = catalog.entries.first().expect("nonempty generated catalog");
    let selected = resolve_generated_catalog_row_for_key(
        &entry.to_runtime_lookup_key(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Some(catalog),
    )
    .expect("prime the materialized-catalog cache");

    let error = resolve_generated_schedule_selection(
        selected.selection(),
        &policy_of::<Cfg>(),
        |_| {
            Err(akita_error::AkitaError::InvalidSetup(
                "test runtime-hook drift".to_string(),
            ))
        },
        Some(catalog),
    )
    .expect_err("a cache hit must still execute the supplied runtime hook");
    assert!(
        matches!(error, akita_error::AkitaError::InvalidSetup(_)),
        "expected runtime-hook validation failure, got {error:?}"
    );
}

fn assert_mutated_row_is_rejected<Cfg: CommitmentConfig>(
    profiles: akita_types::CommittedGroupBatchProfile,
    schedule: akita_types::FoldSchedule,
) {
    let row_digest = akita_types::schedule_row_digest(&profiles, &schedule)
        .expect("mutated row has a canonical digest");
    let error = ResolvedScheduleRow::try_new(
        akita_types::OpeningScheduleSelection { row_digest },
        profiles,
        schedule,
        &policy_of::<Cfg>(),
    )
    .expect_err("security-invalid row must fail before digest admission");
    assert!(
        matches!(error, akita_error::AkitaError::InvalidSetup(_)),
        "expected InvalidSetup, got {error:?}"
    );
}

#[test]
fn resolved_row_audit_rejects_low_rank_root_d_and_a() {
    type Cfg = fp128::Dense;

    let catalog = Cfg::schedule_catalog().expect("dense schedule catalog");
    let entry = catalog.entries.first().expect("nonempty generated catalog");
    let selected = resolve_generated_catalog_row_for_key(
        &entry.to_runtime_lookup_key(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Some(catalog),
    )
    .expect("valid generated row");
    let profiles = selected.profiles().clone();

    let mut low_rank_d = selected.schedule().clone();
    let matrix = low_rank_d.root.params.open_matrix;
    low_rank_d.root.params.open_matrix = akita_types::OpenCommitMatrixParams::new_unchecked(
        matrix.security_policy(),
        matrix.sis_table_key().table_digest,
        matrix.sis_modulus_profile(),
        0,
        matrix.input_width(),
        matrix.coeff_linf_bound(),
        matrix.ring_dimension(),
    );
    assert_mutated_row_is_rejected::<Cfg>(profiles.clone(), low_rank_d);

    let mut low_rank_a = selected.schedule().clone();
    let matrix = &low_rank_a.root.params.inner().matrix;
    let table_key = matrix
        .sis_table_key()
        .expect("root A matrix must use the L infinity route");
    low_rank_a.root.params.own_group_mut().profile.inner.matrix =
        akita_types::InnerCommitMatrixParams::new_unchecked(
            table_key.policy,
            table_key.table_digest,
            table_key.modulus_profile,
            0,
            matrix.input_width(),
            table_key.coeff_linf_bound,
            table_key.ring_dimension as usize,
        );
    assert_mutated_row_is_rejected::<Cfg>(profiles, low_rank_a);
}

#[test]
fn resolved_row_audit_rejects_each_noncanonical_terminal_shape_field() {
    type Cfg = fp128::Dense;

    let catalog = Cfg::schedule_catalog().expect("dense schedule catalog");
    let entry = catalog.entries.first().expect("nonempty generated catalog");
    let selected = resolve_generated_catalog_row_for_key(
        &entry.to_runtime_lookup_key(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Some(catalog),
    )
    .expect("valid generated row");
    let profiles = selected.profiles().clone();
    let schedule = selected.schedule();

    let mut mutations = Vec::new();
    let mut mutated = schedule.clone();
    mutated.terminal.response_shape.layout.ring_dimension += 1;
    mutations.push(mutated);
    let mut mutated = schedule.clone();
    mutated.terminal.response_shape.layout.groups[0].z_coords += 1;
    mutations.push(mutated);
    let mut mutated = schedule.clone();
    mutated.terminal.response_shape.layout.groups[0].e_field_elems += 1;
    mutations.push(mutated);
    let mut mutated = schedule.clone();
    mutated.terminal.response_shape.layout.groups[0].t_field_elems += 1;
    mutations.push(mutated);
    let mut mutated = schedule.clone();
    mutated.terminal.response_shape.layout.logical_num_elems += 1;
    mutations.push(mutated);
    let mut mutated = schedule.clone();
    mutated.terminal.response_shape.layout.groups[0].z_payload_bytes = 0;
    mutations.push(mutated);

    for mutated in mutations {
        assert_mutated_row_is_rejected::<Cfg>(profiles.clone(), mutated);
    }
}

#[cfg(all(
    feature = "schedules-fp128-onehot",
    not(feature = "schedules-fp128-onehot-recursive")
))]
#[test]
#[cfg(not(any(
    feature = "schedules-fp128-onehot-recursive",
    feature = "schedules-fp128-onehot-recursive-multi-chunk-w8r2"
)))]
fn grouped_recursive_catalog_rejects_without_recursive_feature() {
    let precommitted_group = PolynomialGroupLayout::new(16, 1);
    let descriptor = fp128::OneHot::profile_without_precommitted_groups(precommitted_group)
        .expect("base OneHot precommit profile");
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(32, 2),
        precommitteds: vec![descriptor, descriptor],
    };
    assert!(matches!(
        akita_config::RecursiveCommitmentConfig::<fp128::OneHot>::resolve_catalog_row_for_key(&key),
        Err(akita_error::AkitaError::UnsupportedSchedule(_))
    ));
}

fn assert_policy_matches_cfg<Cfg: CommitmentConfig>() {
    let policy = policy_of::<Cfg>();
    let expected = PlannerPolicy {
        cost_model: PlannerCostModelId::ExactPayloadAndSetupEnvelope,
        selective_l2_response_model:
            akita_schedules::SelectiveL2ResponseModelId::TypedProtocolMomentsV1,
        selection_policy: Cfg::selection_policy(),
        recursive_split_search_policy:
            akita_schedules::RecursiveSplitSearchPolicy::BoundedBalancedExtremesV1,
        setup_field_budget: None,
        min_offloaded_witness_contraction: 3,
        ring_dimension_schedule_mode: Cfg::RING_DIMENSION_SCHEDULE_MODE,
        decomposition: Cfg::decomposition(),
        sis_modulus_profile: Cfg::sis_modulus_profile(),
        sis_security_policy: akita_types::DEFAULT_SIS_SECURITY_POLICY,
        sis_table_digest: akita_types::SisTableDigest::CURRENT,
        sis_l2_table_digest: akita_types::SisL2TableDigest::CURRENT,
        claim_ext_degree: Cfg::EXT_DEGREE,
        chal_ext_degree: Cfg::EXT_DEGREE,
        inner_basis_range: Cfg::inner_basis_range(),
        opening_basis_range: Cfg::opening_basis_range(),
        witness_chunk: Cfg::chunked_witness_cfg(),
        recursive_setup_planning: Cfg::recursive_setup_planning(),
    };
    assert_eq!(
        policy, expected,
        "policy_of must derive every field from the Cfg impl"
    );
}

#[test]
fn runtime_rejects_malformed_extension_geometry_without_panicking() {
    type Cfg = fp128::OneHot;
    let catalog = Cfg::schedule_catalog();
    let key = PolynomialGroupLayout::new(14, 1);
    let reject = |mutate: fn(&mut PlannerPolicy)| {
        let mut policy = policy_of::<Cfg>();
        mutate(&mut policy);
        resolve_generated_catalog_row_for_key(
            &AkitaScheduleLookupKey::single(key),
            &policy,
            Cfg::ring_challenge_config,
            catalog,
        )
        .expect_err("malformed extension geometry must reject")
        .to_string()
    };

    assert!(reject(|policy| policy.claim_ext_degree = 0).contains("nonzero power of two"));
    assert!(reject(|policy| policy.claim_ext_degree = 3).contains("nonzero power of two"));
    assert!(reject(|policy| policy.chal_ext_degree = 0).contains("nonzero power of two"));
    assert!(reject(|policy| policy.chal_ext_degree = 3).contains("nonzero power of two"));
    assert!(reject(|policy| policy.chal_ext_degree = 1usize << 31)
        .contains("challenge field bit width overflow"));
    if usize::BITS > u32::BITS {
        assert!(
            reject(|policy| policy.chal_ext_degree = (u32::MAX as usize) + 1)
                .contains("exceeds u32")
        );
    }
}

#[test]
fn policy_bridge_matches_cfg_hooks() {
    assert_policy_matches_cfg::<fp128::Dense>();
    assert_policy_matches_cfg::<fp128::OneHot>();
    assert_policy_matches_cfg::<fp32::OneHot>();
}

#[test]
fn adaptive_dense_searches_multi_group_roots_while_preserving_precommits() {
    type Cfg = fp128::Dense;
    const PRE_NV: usize = 16;
    const FINAL_NV: usize = 20;

    let pre_group = PolynomialGroupLayout::singleton(PRE_NV);
    let pre_profile =
        Cfg::profile_without_precommitted_groups(pre_group).expect("dense precommit profile");
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::singleton(FINAL_NV),
        precommitteds: vec![pre_profile],
    };
    let precommitted_honest_fold_policies = vec![akita_config::honest_fold_policy_of::<Cfg>()];
    let planned = find_schedule(
        &key,
        akita_config::honest_fold_policy_of::<Cfg>(),
        &precommitted_honest_fold_policies,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("adaptive dense grouped schedule");
    planned
        .schedule
        .validate_structure()
        .expect("valid grouped schedule");
    assert_eq!(
        planned.schedule.root.params.role_dims(),
        CommitmentRingDims {
            inner: 512,
            outer: 64,
            opening: 64,
        }
    );
    assert_eq!(
        planned.schedule.root.params.precommitted_groups()[0].profile,
        key.precommitteds[0]
    );
}

#[test]
fn root_basis_is_derived_from_existing_policy_inputs() {
    let fp128 = policy_of::<fp128::OneHot>();
    assert_eq!(fp128.opening_basis_range, (3, 6));
    assert_eq!(fp128.decomposition.log_basis, 3);

    let fp32 = policy_of::<fp32::OneHot>();
    assert_eq!(fp32.opening_basis_range, (3, 6));
    assert_eq!(fp32.decomposition.log_basis, 3);
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
        let _ = fp128::OneHot::resolve_catalog_row_for_key(&AkitaScheduleLookupKey::single(key));
    }
}
fn committed_descriptor<Cfg: CommitmentConfig>(
    group: PolynomialGroupLayout,
) -> GroupCommitPhaseParams {
    Cfg::profile_without_precommitted_groups(group).expect("heterogeneous group must resolve")
}

#[test]
fn heterogeneous_group_profiles_match_generated_lookup_and_reject_unlisted_order() {
    type Cfg = fp128::OneHot;
    let onehot_256 = committed_descriptor::<Cfg>(PolynomialGroupLayout::new(14, 1));
    let dense = committed_descriptor::<fp128::Dense>(PolynomialGroupLayout::new(15, 2));
    let key = AkitaScheduleLookupKey {
        final_group: PolynomialGroupLayout::new(16, 1),
        precommitteds: vec![onehot_256, dense],
    };

    let precommitted_honest_fold_policies = vec![
        akita_config::honest_fold_policy_of::<Cfg>(),
        akita_config::honest_fold_policy_of::<fp128::Dense>(),
    ];
    let planned = find_schedule(
        &key,
        akita_config::honest_fold_policy_of::<Cfg>(),
        &precommitted_honest_fold_policies,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("heterogeneous group batch must plan offline");

    let runtime = Cfg::resolve_catalog_row_for_key(&key)
        .expect("curated mixed catalog row")
        .into_schedule();
    assert_schedule_eq("curated mixed row replay", &runtime, &planned.schedule);

    let reordered = AkitaScheduleLookupKey {
        precommitteds: vec![dense, onehot_256],
        ..key
    };
    assert_ne!(
        key.canonical_descriptor_bytes(),
        reordered.canonical_descriptor_bytes(),
        "ordered per-group contracts must be part of catalog identity"
    );
    assert!(
        matches!(
            Cfg::resolve_catalog_row_for_key(&reordered),
            Err(akita_error::AkitaError::UnsupportedSchedule(_))
        ),
        "an unlisted mixed ordering must reject without runtime planner search"
    );
}
