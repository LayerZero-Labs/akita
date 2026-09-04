//! Guard test: for every `(family, key)` covered by the generated schedule
//! tables, the **table-hit** expansion must reproduce exactly the schedule
//! the pure DP regenerates **on this branch**.
//!
//! This compares generated tables against the current planner DP only — it does
//! **not** detect divergence from historical `main` (expected when bundled
//! planner changes such as the K256 one-hot migration regenerate tables).
//!
//! Coverage is metadata-driven: every entry in
//! [`akita_planner::generated_families::ALL_GENERATED_FAMILIES`] is checked,
//! so adding a new family to the generator picks it up here automatically
//! (no per-family handwritten row mirror).
//!
//! For each key the test resolves two schedules and asserts they are
//! identical:
//!
//! - **table-backed** via [`table_backed_expanded`] after one full-catalog audit
//!   for scalar schedules, or direct catalog-entry expansion for multi-group-root
//!   schedules under `all-schedules`;
//! - **regenerated** via `family.regen` / `family.regen_group_batch`, which runs
//!   the pure DP from scratch.
//!
//! The comparison is over the *fully resolved* [`FoldSchedule`] — every step's
//! expanded [`CommittedGroupParams`] (SIS buckets + derived matrix widths),
//! typed root/recursive/terminal topology, and witness lengths. Planner byte
//! estimates are deliberately not protocol schedule state. This is strictly
//! stronger than diffing the compact
//! generated fold tuples: it catches any drift where the table-hit
//! expansion would carry a different `inner_commit_matrix.coeff_linf_bound()` (or width, or
//! rank) than the DP used, not just a different stored tuple.
//!
//! When this test fails the panic message lists per-family mismatch counts,
//! the first few offending schedules, and the regenerate command for the
//! active feature set.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32};
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_planner::emit::{bounded_parallel_filter_map, offline_planning_worker_count};
use akita_planner::generated_families::{
    emitted_scalar_keys, GeneratedFamily, GenerationPreplans, GroupedGenerationRequest,
    ALL_GENERATED_FAMILIES,
};
use akita_types::{
    AkitaScheduleLookupKey, FoldSchedule, GroupCommitPhaseParams, PolynomialGroupLayout,
};
use std::sync::OnceLock;

struct PreparedGroupBatchRequests {
    preplans: GenerationPreplans,
    by_family: Vec<Vec<GroupedGenerationRequest>>,
}

fn prepared_group_batch_requests() -> &'static PreparedGroupBatchRequests {
    static PREPARED: OnceLock<PreparedGroupBatchRequests> = OnceLock::new();
    PREPARED.get_or_init(|| {
        let preplans = GenerationPreplans::default();
        let workers = offline_planning_worker_count(ALL_GENERATED_FAMILIES.len());
        let by_family = bounded_parallel_filter_map(ALL_GENERATED_FAMILIES, workers, |family| {
            (family.grouped_requests)(&preplans)
                .map(Some)
                .map_err(|error| {
                    format!(
                        "family {} multi-group key enumeration failed: {error}",
                        family.module_name
                    )
                })
        })
        .unwrap_or_else(|error| panic!("{error}"));
        PreparedGroupBatchRequests {
            preplans,
            by_family,
        }
    })
}

#[cfg(feature = "all-schedules")]
use akita_config::policy_of;
use akita_schedules::generated::{table_entry, table_entry_range};
#[cfg(feature = "all-schedules")]
use akita_schedules::{
    catalog_entries_sorted_for_lookup, schedule_from_entry, validate_catalog_identity,
    validate_generated_schedule_table,
};

#[cfg(feature = "all-schedules")]
#[test]
fn every_grouped_precommitted_descriptor_has_a_generated_producer() {
    let produced = ALL_GENERATED_FAMILIES
        .iter()
        .flat_map(|family| {
            emitted_scalar_keys(family)
                .unwrap_or_else(|error| {
                    panic!("{} S-key enumeration failed: {error}", family.module_name)
                })
                .into_iter()
                .map(|group| {
                    let schedule =
                        (family.resolve_catalog_row_for_key)(AkitaScheduleLookupKey::single(group))
                            .unwrap_or_else(|error| {
                                panic!("{} S-row lookup failed: {error}", family.module_name)
                            });
                    GroupCommitPhaseParams::try_from_params(group, &schedule.schedule().root.params)
                        .expect("valid generated profile")
                })
        })
        .collect::<Vec<_>>();

    for family in ALL_GENERATED_FAMILIES {
        let catalog = (family.schedule_catalog)()
            .unwrap_or_else(|| panic!("{} generated catalog is unavailable", family.module_name));
        for entry in catalog.entries {
            for group in entry.root.precommitted_groups {
                assert!(
                    produced.contains(&group.group.profile),
                    "family {} embeds a grouped precommitted descriptor without an exact generated S producer: {:?}",
                    family.module_name,
                    group.group.profile.group
                );
            }
        }
    }
}

#[cfg(feature = "all-schedules")]
fn assert_table_hit(
    module_name: &str,
    catalog: &akita_schedules::GeneratedScheduleTable,
    keys: &[PolynomialGroupLayout],
) {
    if keys.is_empty() {
        return;
    }
    for &key in keys {
        assert!(
            table_entry(*catalog, &AkitaScheduleLookupKey::single(key)).is_some(),
            "family {module_name} is missing emitted scalar key {key:?}"
        );
    }
}

#[cfg(feature = "all-schedules")]
fn prepare_family_catalog(
    family: &GeneratedFamily,
    keys: &[PolynomialGroupLayout],
) -> akita_schedules::GeneratedScheduleTable {
    let module_name = family.module_name;
    let catalog = (family.schedule_catalog)().unwrap_or_else(|| {
        panic!("family {module_name} must expose schedule_catalog() under all-schedules")
    });
    validate_generated_schedule_table(&catalog, &(family.policy)(), &family.ring_challenge_config)
        .unwrap_or_else(|e| panic!("catalog validation failed for {module_name}: {e}"));
    assert!(
        catalog_entries_sorted_for_lookup(catalog.entries),
        "family {module_name} catalog entries must be sorted for binary lookup"
    );
    assert_table_hit(module_name, &catalog, keys);
    catalog
}

#[cfg(feature = "all-schedules")]
#[test]
fn catalog_identity_rejects_noncurrent_protocol_epoch() {
    let mut catalog = fp128::Dense::schedule_catalog().expect("generated catalog");
    catalog.identity.protocol_epoch -= 1;
    let error = validate_catalog_identity(
        &catalog,
        &policy_of::<fp128::Dense>(),
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("noncurrent protocol epoch must not validate");
    assert!(error.to_string().contains("catalog identity mismatch"));
}

#[cfg(feature = "all-schedules")]
#[test]
fn generated_catalog_identities_match_runtime_schema() {
    for family in ALL_GENERATED_FAMILIES {
        let catalog = (family.schedule_catalog)()
            .unwrap_or_else(|| panic!("{} generated catalog is unavailable", family.module_name));
        let expected = akita_schedules::expected_catalog_identity(
            catalog.identity.family_name,
            &(family.policy)(),
            catalog.entries,
            family.ring_challenge_config,
        )
        .expect("generated catalog identity");
        assert_eq!(
            catalog.identity, expected,
            "{} generated identity must match the current protocol schema",
            family.module_name
        );
    }
}

#[cfg(feature = "all-schedules")]
#[test]
fn generated_catalogs_pin_dyadic_slice_chunk_interactions() {
    use std::collections::BTreeSet;

    let catalogs = [
        fp128::OneHot::schedule_catalog().expect("W1 catalog"),
        fp128::OneHotMultiChunkW2R2::schedule_catalog().expect("W2 catalog"),
        fp128::OneHotMultiChunkW4R2::schedule_catalog().expect("W4 catalog"),
        fp128::OneHotMultiChunk::schedule_catalog().expect("W8 one-hot catalog"),
        fp128::DenseMultiChunk::schedule_catalog().expect("W8 dense catalog"),
    ];
    let mut observed = BTreeSet::new();
    for catalog in catalogs {
        for entry in catalog.entries {
            observed.insert((
                entry.root.core.group.outer_slice_count,
                entry.root.core.witness_chunks,
            ));
            for fold in entry.recursive_folds.iter().take(2) {
                observed.insert((fold.core.group.outer_slice_count, fold.core.witness_chunks));
            }
        }
    }

    for expected in [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
        (4, 8),
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
    ] {
        assert!(
            observed.contains(&expected),
            "generated schedules must retain S/W={expected:?}; observed {observed:?}"
        );
    }
}

#[cfg(feature = "all-schedules")]
#[test]
fn generated_expansion_rejects_zero_witness_chunks() {
    let catalog = fp128::Dense::schedule_catalog().expect("fp128 dense catalog");
    let policy = policy_of::<fp128::Dense>();
    let original = *catalog
        .entries
        .iter()
        .find(|entry| !entry.recursive_folds.is_empty())
        .expect("recursive generated row");

    let mut root_zero = original;
    root_zero.root.core.witness_chunks = 0;
    let error = schedule_from_entry(
        &root_zero,
        &root_zero.to_runtime_lookup_key(),
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("zero root chunk count must reject");
    assert!(error.to_string().contains("chunk count must be nonzero"));

    let mut recursive_zero = original;
    let folds = Box::leak(recursive_zero.recursive_folds.to_vec().into_boxed_slice());
    folds[0].core.witness_chunks = 0;
    recursive_zero.recursive_folds = folds;
    let error = schedule_from_entry(
        &recursive_zero,
        &recursive_zero.to_runtime_lookup_key(),
        &policy,
        fp128::Dense::ring_challenge_config,
    )
    .expect_err("zero recursive chunk count must reject");
    assert!(error.to_string().contains("chunk count must be nonzero"));
}

#[cfg(feature = "all-schedules")]
#[test]
fn catalog_identity_rejects_planner_policy_changes() {
    let policy = policy_of::<fp128::Dense>();
    let catalog = fp128::Dense::schedule_catalog().expect("generated catalog");
    let assert_rejected = |label: &str, mutated: akita_schedules::GeneratedScheduleTable| {
        let error =
            validate_catalog_identity(&mutated, &policy, fp128::Dense::ring_challenge_config)
                .expect_err("planner-policy mismatch must not validate");
        assert!(
            error.to_string().contains("catalog identity mismatch"),
            "{label} mutation returned the wrong error: {error}"
        );
    };

    let mut mutated = catalog;
    mutated.identity.selection_policy =
        akita_schedules::SelectionPolicyId::MinEstimatedProofPayloadV2;
    assert_rejected("selection policy", mutated);

    let mut mutated = catalog;
    mutated.identity.setup_field_budget = Some(1);
    assert_rejected("setup capacity ceiling", mutated);

    let mut mutated = catalog;
    mutated.identity.min_offloaded_witness_contraction += 1;
    assert_rejected("offloaded witness contraction", mutated);

    let mut mutated = catalog;
    mutated.identity.selective_l2_response_model =
        akita_schedules::SelectiveL2ResponseModelId::Disabled;
    assert_rejected("selective L2 response model", mutated);
}

#[cfg(feature = "all-schedules")]
#[test]
fn catalog_identity_binds_every_role_specific_execution_field() {
    let policy = policy_of::<fp128::Dense>();
    let catalog = fp128::Dense::schedule_catalog().expect("fp128 dense catalog");
    let original = *catalog
        .entries
        .iter()
        .find(|entry| !entry.recursive_folds.is_empty())
        .expect("recursive generated row");
    let identity = akita_schedules::expected_catalog_identity(
        catalog.identity.family_name,
        &policy,
        std::slice::from_ref(&original),
        fp128::Dense::ring_challenge_config,
    )
    .expect("single-row identity");
    let assert_rejected = |entry, label| {
        let mutated = akita_schedules::GeneratedScheduleTable {
            entries: Box::leak(vec![entry].into_boxed_slice()),
            identity,
        };
        let error =
            validate_catalog_identity(&mutated, &policy, fp128::Dense::ring_challenge_config)
                .expect_err("executed generated field mutation must invalidate catalog identity");
        assert!(
            error.to_string().contains("catalog identity mismatch"),
            "{label} mutation returned the wrong error: {error}"
        );
    };

    let mut root_digits = original;
    root_digits.root.num_digits_inner += 1;
    assert_rejected(root_digits, "root inner digits");

    let mut recursive_payload = original;
    let folds = Box::leak(
        recursive_payload
            .recursive_folds
            .to_vec()
            .into_boxed_slice(),
    );
    folds[0].payload_mode = match folds[0].payload_mode {
        akita_types::CommitmentPayloadMode::Compressed => akita_types::CommitmentPayloadMode::Raw,
        akita_types::CommitmentPayloadMode::Raw => akita_types::CommitmentPayloadMode::Compressed,
    };
    recursive_payload.recursive_folds = folds;
    assert_rejected(recursive_payload, "recursive payload mode");

    let mut relation_mode = original;
    relation_mode.root.core.ring_relation_mode = akita_types::RingRelationMode::ReducedEvaluation;
    assert_rejected(relation_mode, "root ring relation mode");

    let mut terminal_payload = original;
    terminal_payload.terminal.z_payload_bytes += 1;
    assert_rejected(terminal_payload, "terminal payload bytes");
}

#[cfg(feature = "all-schedules")]
#[test]
fn equal_lookup_keys_form_one_contiguous_candidate_range() {
    let catalog = fp128::Dense::schedule_catalog().expect("generated catalog");
    let entry = *catalog.entries.first().expect("nonempty generated catalog");
    let entries = Box::leak(vec![entry, entry].into_boxed_slice());
    let duplicate_table = akita_schedules::GeneratedScheduleTable {
        entries,
        identity: catalog.identity,
    };
    let key = entry.to_runtime_lookup_key();

    assert!(catalog_entries_sorted_for_lookup(entries));
    assert_eq!(table_entry_range(duplicate_table, &key), 0..2);
    assert_eq!(table_entry(duplicate_table, &key), entries.first());
}

#[cfg(feature = "all-schedules")]
#[test]
fn mixed_catalog_identity_binds_candidate_dimensions() {
    static WITHOUT_NONWINNER: &[usize] = &[64, 256];

    let policy = policy_of::<fp128::OneHot>();
    let catalog = fp128::OneHot::schedule_catalog().expect("fp128 one-hot catalog");
    let mut mutated = catalog;
    let akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
        num_search_levels,
        suffix_dimensions,
        potential_b_dimensions,
        potential_d_dimensions,
        ..
    } = mutated.identity.ring_dimension_schedule_mode
    else {
        panic!("mixed catalog must use adaptive dimensions");
    };
    mutated.identity.ring_dimension_schedule_mode =
        akita_schedules::RingDimensionScheduleMode::AdaptiveDimension {
            num_search_levels,
            suffix_dimensions,
            potential_a_dimensions: WITHOUT_NONWINNER,
            potential_b_dimensions,
            potential_d_dimensions,
        };
    let error = validate_catalog_identity(&mutated, &policy, fp128::OneHot::ring_challenge_config)
        .expect_err("removing an admitted nonwinner must invalidate the catalog");
    assert!(error.to_string().contains("catalog identity mismatch"));
}

#[cfg(feature = "all-schedules")]
#[test]
fn adaptive_catalog_identity_rejects_terminal_dimension_growth() {
    let policy = policy_of::<fp32::Dense>();
    let catalog = fp32::Dense::schedule_catalog().expect("fp32 dense catalog");
    let mut entry = *catalog.entries.first().expect("nonempty fp32 catalog");

    if !entry.recursive_folds.is_empty() {
        let mut folds = entry.recursive_folds.to_vec();
        let last = folds.last_mut().expect("copied recursive fold");
        last.core.group.inner_commit_matrix.ring_dimension = 64;
        last.core.group.outer_commit_matrix.ring_dimension = 64;
        last.core.open_commit_matrix.ring_dimension = 64;
        entry.recursive_folds = Box::leak(folds.into_boxed_slice());
    } else {
        entry.root.core.group.inner_commit_matrix.ring_dimension = 64;
    }
    entry.terminal.inner_commit_matrix.ring_dimension = 128;

    let mutated = akita_schedules::GeneratedScheduleTable {
        entries: Box::leak(vec![entry].into_boxed_slice()),
        identity: catalog.identity,
    };
    let error = validate_catalog_identity(&mutated, &policy, fp32::Dense::ring_challenge_config)
        .expect_err("terminal dimensions must not grow above the preceding A dimension");
    assert!(
        error
            .to_string()
            .contains("exceeds predecessor A dimension"),
        "unexpected validation error: {error}"
    );
}

#[cfg(feature = "all-schedules")]
#[test]
fn recursive_companion_catalogs_contain_only_offloaded_schedules() {
    for family in ALL_GENERATED_FAMILIES {
        if !(family.policy)().recursive_setup_planning {
            continue;
        }
        let catalog = prepare_family_catalog(family, &[]);
        assert!(
            catalog.entries.iter().all(|entry| entry
                .recursive_folds
                .iter()
                .any(|fold| fold.setup_prefix.is_some())),
            "recursive companion family {} contains a schedule without setup offloading",
            family.module_name
        );
    }
}

#[cfg(feature = "all-schedules")]
#[test]
fn generated_catalogs_cover_emitted_keys() {
    for family in ALL_GENERATED_FAMILIES {
        assert!(
            (family.schedule_catalog)().is_some(),
            "family {} is not linked under all-schedules",
            family.module_name
        );
        let keys = emitted_scalar_keys(family).unwrap_or_else(|error| {
            panic!(
                "family {} key enumeration failed: {error}",
                family.module_name
            )
        });
        let _ = prepare_family_catalog(family, &keys);
    }
}

fn assert_grouped_table_hits(family: &GeneratedFamily, requests: &[GroupedGenerationRequest]) {
    if requests.is_empty() {
        return;
    }
    let catalog = (family.schedule_catalog)().unwrap_or_else(|| {
        panic!(
            "family {} must expose schedule_catalog()",
            family.module_name
        )
    });
    let missing = requests
        .iter()
        .filter(|request| table_entry(catalog, &request.key()).is_none())
        .take(3)
        .map(|request| format!("{:?}", request.key()))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "family {} must have generated grouped-table hits for every enumerated multi-group key; first missing keys: {}",
        family.module_name,
        missing.join("\n  ")
    );
}

#[cfg(feature = "all-schedules")]
fn resolve_family_group_batch_schedule(
    family: &GeneratedFamily,
    catalog: akita_schedules::GeneratedScheduleTable,
    request: &GroupedGenerationRequest,
) -> Result<FoldSchedule, AkitaError> {
    let key = request.key();
    let entry = table_entry(catalog, &key).ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "generated family {} is missing grouped key {:?}",
            family.module_name, key
        ))
    })?;
    schedule_from_entry(
        entry,
        &key,
        &(family.policy)(),
        family.ring_challenge_config,
    )
}

#[cfg(not(feature = "all-schedules"))]
fn resolve_family_group_batch_schedule(
    family: &GeneratedFamily,
    request: &GroupedGenerationRequest,
) -> Result<FoldSchedule, AkitaError> {
    (family.resolve_catalog_row_for_key)(request.key())
}

#[cfg(feature = "all-schedules")]
fn table_backed_expanded(
    family: &GeneratedFamily,
    catalog: akita_schedules::GeneratedScheduleTable,
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    let lookup_key = AkitaScheduleLookupKey::single(key);
    let entry = table_entry(catalog, &lookup_key).ok_or_else(|| {
        AkitaError::UnsupportedSchedule(format!(
            "generated family {} is missing scalar key {key:?}",
            family.module_name
        ))
    })?;
    schedule_from_entry(
        entry,
        &lookup_key,
        &(family.policy)(),
        family.ring_challenge_config,
    )
}

/// One `(family, key)` whose table-hit expansion disagrees with the DP.
struct Mismatch {
    family: &'static str,
    key: String,
    table_backed: String,
    regenerated: String,
}

impl Mismatch {
    fn render(&self) -> String {
        format!(
            "  family={} key={}\n    table-backed: {}\n    regenerated:  {}\n",
            self.family, self.key, self.table_backed, self.regenerated
        )
    }
}

/// Canonical diagnostic form of the fully resolved typed schedule.
fn render_schedule(schedule: &FoldSchedule) -> String {
    format!("{schedule:?}")
}

fn schedules_equal(left: &FoldSchedule, right: &FoldSchedule) -> bool {
    left == right
}

fn compare_schedule_results(
    family: &GeneratedFamily,
    key: PolynomialGroupLayout,
    table_backed: Result<FoldSchedule, AkitaError>,
    regenerated: Result<FoldSchedule, AkitaError>,
) -> Option<Mismatch> {
    match (table_backed, regenerated) {
        (Ok(table_backed), Ok(regenerated)) => {
            if schedules_equal(&table_backed, &regenerated) {
                None
            } else {
                Some(Mismatch {
                    family: family.module_name,
                    key: format!("{key:?}"),
                    table_backed: render_schedule(&table_backed),
                    regenerated: render_schedule(&regenerated),
                })
            }
        }
        (Err(AkitaError::UnsupportedSchedule(_)), Err(AkitaError::UnsupportedSchedule(_))) => None,
        (table_backed, regenerated) => Some(Mismatch {
            family: family.module_name,
            key: format!("{key:?}"),
            table_backed: table_backed
                .map(|schedule| render_schedule(&schedule))
                .unwrap_or_else(|error| format!("error: {error}")),
            regenerated: regenerated
                .map(|schedule| render_schedule(&schedule))
                .unwrap_or_else(|error| format!("error: {error}")),
        }),
    }
}

#[cfg(feature = "all-schedules")]
fn compare_scalar_key(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    catalog: akita_schedules::GeneratedScheduleTable,
    key: PolynomialGroupLayout,
) -> Option<Mismatch> {
    let regenerated = preplans
        .scalar_for_family(family, key)
        .map_or_else(|| (family.regen)(key), Ok);
    compare_schedule_results(
        family,
        key,
        table_backed_expanded(family, catalog, key),
        regenerated,
    )
}

#[cfg(not(feature = "all-schedules"))]
fn compare_scalar_key(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    key: PolynomialGroupLayout,
) -> Option<Mismatch> {
    let regenerated = preplans
        .scalar_for_family(family, key)
        .map_or_else(|| (family.regen)(key), Ok);
    compare_schedule_results(
        family,
        key,
        (family.resolve_catalog_row_for_key)(AkitaScheduleLookupKey::single(key)),
        regenerated,
    )
}

#[cfg(feature = "all-schedules")]
fn check_scalar_keys(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    keys: &[PolynomialGroupLayout],
    catalog: akita_schedules::GeneratedScheduleTable,
    into: &mut Vec<Mismatch>,
) {
    let workers = offline_planning_worker_count(keys.len());

    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for &key in chunk {
                            if let Some(mismatch) =
                                compare_scalar_key(family, preplans, catalog, key)
                            {
                                local.push(mismatch);
                            }
                        }
                        local
                    })
                })
                .collect();
            for handle in handles {
                into.extend(handle.join().expect("worker thread panicked"));
            }
        });
        return;
    }

    for &key in keys {
        if let Some(mismatch) = compare_scalar_key(family, preplans, catalog, key) {
            into.push(mismatch);
        }
    }
}

#[cfg(not(feature = "all-schedules"))]
fn check_scalar_keys(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    keys: &[PolynomialGroupLayout],
    into: &mut Vec<Mismatch>,
) {
    let workers = offline_planning_worker_count(keys.len());

    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for &key in chunk {
                            if let Some(mismatch) = compare_scalar_key(family, preplans, key) {
                                local.push(mismatch);
                            }
                        }
                        local
                    })
                })
                .collect();
            for handle in handles {
                into.extend(handle.join().expect("worker thread panicked"));
            }
        });
        return;
    }

    for &key in keys {
        if let Some(mismatch) = compare_scalar_key(family, preplans, key) {
            into.push(mismatch);
        }
    }
}

#[cfg(feature = "all-schedules")]
fn compare_group_batch_key(
    family: &GeneratedFamily,
    catalog: akita_schedules::GeneratedScheduleTable,
    request: &GroupedGenerationRequest,
) -> Option<Mismatch> {
    let key = request.key();
    let table_backed = resolve_family_group_batch_schedule(family, catalog, request)
        .unwrap_or_else(|e| {
            panic!(
                "table-backed multi-group schedule failed for family {} key={:?}: {e}",
                family.module_name, key
            )
        });
    let regenerated = (family.regen_group_batch)(request.clone()).unwrap_or_else(|e| {
        panic!(
            "multi-group DP regen failed for family {} key={:?}: {e}",
            family.module_name, key
        )
    });

    if schedules_equal(&table_backed, &regenerated) {
        return None;
    }
    Some(Mismatch {
        family: family.module_name,
        key: format!("group-batch {key:?}"),
        table_backed: render_schedule(&table_backed),
        regenerated: render_schedule(&regenerated),
    })
}

#[cfg(not(feature = "all-schedules"))]
fn compare_group_batch_key(
    family: &GeneratedFamily,
    request: &GroupedGenerationRequest,
) -> Option<Mismatch> {
    let key = request.key();
    let table_backed = resolve_family_group_batch_schedule(family, request).unwrap_or_else(|e| {
        panic!(
            "table-backed multi-group schedule failed for family {} key={:?}: {e}",
            family.module_name, key
        )
    });
    let regenerated = (family.regen_group_batch)(request.clone()).unwrap_or_else(|e| {
        panic!(
            "multi-group DP regen failed for family {} key={:?}: {e}",
            family.module_name, key
        )
    });

    if schedules_equal(&table_backed, &regenerated) {
        return None;
    }
    Some(Mismatch {
        family: family.module_name,
        key: format!("group-batch {key:?}"),
        table_backed: render_schedule(&table_backed),
        regenerated: render_schedule(&regenerated),
    })
}

#[cfg(feature = "all-schedules")]
fn check_grouped_requests(
    family: &GeneratedFamily,
    catalog: akita_schedules::GeneratedScheduleTable,
    requests: &[GroupedGenerationRequest],
    into: &mut Vec<Mismatch>,
) {
    if requests.is_empty() {
        return;
    }

    let workers = offline_planning_worker_count(requests.len());
    if workers > 1 && requests.len() >= 2 * workers {
        let chunk_size = requests.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = requests
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for request in chunk {
                            if let Some(mismatch) =
                                compare_group_batch_key(family, catalog, request)
                            {
                                local.push(mismatch);
                            }
                        }
                        local
                    })
                })
                .collect();
            for handle in handles {
                into.extend(handle.join().expect("worker thread panicked"));
            }
        });
        return;
    }

    for request in requests {
        if let Some(mismatch) = compare_group_batch_key(family, catalog, request) {
            into.push(mismatch);
        }
    }
}

#[cfg(not(feature = "all-schedules"))]
fn check_grouped_requests(
    family: &GeneratedFamily,
    requests: &[GroupedGenerationRequest],
    into: &mut Vec<Mismatch>,
) {
    if requests.is_empty() {
        return;
    }

    let workers = offline_planning_worker_count(requests.len());
    if workers > 1 && requests.len() >= 2 * workers {
        let chunk_size = requests.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = requests
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for request in chunk {
                            if let Some(mismatch) = compare_group_batch_key(family, request) {
                                local.push(mismatch);
                            }
                        }
                        local
                    })
                })
                .collect();
            for handle in handles {
                into.extend(handle.join().expect("worker thread panicked"));
            }
        });
        return;
    }

    for request in requests {
        if let Some(mismatch) = compare_group_batch_key(family, request) {
            into.push(mismatch);
        }
    }
}

fn check_family(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    grouped_requests: &[GroupedGenerationRequest],
    into: &mut Vec<Mismatch>,
) {
    if (family.schedule_catalog)().is_none() {
        return;
    }

    let keys: Vec<PolynomialGroupLayout> = emitted_scalar_keys(family)
        .unwrap_or_else(|e| panic!("family {} key enumeration failed: {e}", family.module_name));

    #[cfg(feature = "all-schedules")]
    {
        let catalog = prepare_family_catalog(family, &keys);
        check_scalar_keys(family, preplans, &keys, catalog, into);
        assert_grouped_table_hits(family, grouped_requests);
        check_grouped_requests(family, catalog, grouped_requests, into);
    }
    #[cfg(not(feature = "all-schedules"))]
    {
        assert_grouped_table_hits(family, grouped_requests);
        check_scalar_keys(family, preplans, &keys, into);
        check_grouped_requests(family, grouped_requests, into);
    }
}

fn regen_hint() -> &'static str {
    "scripts/generate-schedule-tables.sh"
}

/// The generated tables must expand to exactly what the key-shaped DP produces.
/// Rolled into one test so the panic message can summarize per-family
/// mismatch counts.
#[test]
#[ignore = "the release generator validates these rows in the shared planning pass"]
fn generated_schedule_tables_match_key_planner() {
    let mut mismatches = Vec::new();
    let prepared = prepared_group_batch_requests();
    for (family, grouped_requests) in ALL_GENERATED_FAMILIES.iter().zip(&prepared.by_family) {
        check_family(
            family,
            &prepared.preplans,
            grouped_requests,
            &mut mismatches,
        );
    }

    if mismatches.is_empty() {
        return;
    }

    let mut buckets: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for m in &mismatches {
        *buckets.entry(m.family).or_default() += 1;
    }
    let summary = buckets
        .iter()
        .map(|(family, count)| format!("{family}: {count} issue(s)"))
        .collect::<Vec<_>>()
        .join("\n  ");
    let preview = mismatches
        .iter()
        .take(3)
        .map(Mismatch::render)
        .collect::<String>();
    panic!(
        "{count} schedule-table issue(s) disagree with key-shaped DP output.\n\
         Per-family counts:\n  {summary}\n\n\
         First issues:\n{preview}\n\
         Regenerate the generated tables with:\n  {hint}",
        count = mismatches.len(),
        hint = regen_hint(),
    );
}
