//! Guard test: for every `(family, key)` covered by the shipped schedule
//! tables, the **table-hit** expansion must reproduce exactly the schedule
//! the pure DP regenerates **on this branch**.
//!
//! This compares shipped tables against the current planner DP only — it does
//! **not** detect divergence from historical `main` (expected when bundled
//! planner changes such as the K256 one-hot migration regenerate tables).
//!
//! Coverage is metadata-driven: every entry in
//! [`akita_config::generated_families::ALL_GENERATED_FAMILIES`] is checked,
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

use akita_config::generated_families::{family_keys, GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::tensor_verifier;
use akita_config::CommitmentConfig;
use akita_field::AkitaError;
use akita_types::{AkitaScheduleLookupKey, FoldSchedule, PolynomialGroupLayout};

#[cfg(feature = "all-schedules")]
use akita_config::policy_of;
use akita_planner::generated::table_entry;
#[cfg(feature = "all-schedules")]
use akita_planner::{
    catalog_entries_sorted_for_lookup, schedule_from_entry, validate_generated_schedule_table,
};

#[test]
fn group_batch_emission_matches_supported_policy_shape() {
    for family in ALL_GENERATED_FAMILIES {
        let policy = (family.policy)();
        assert!(
            !family.emit_group_batch || policy.decomposition.log_commit_bound == 1,
            "family {} must not emit grouped companions for unsupported multi-group-root policies",
            family.module_name
        );
    }
}

fn family_catalog_is_linked(family: &GeneratedFamily) -> bool {
    match family.module_name {
        "fp128_d128_full" => fp128::D128Full::schedule_catalog().is_some(),
        "fp128_d128_onehot" => fp128::D128OneHot::schedule_catalog().is_some(),
        "fp128_d64_onehot" => fp128::D64OneHot::schedule_catalog().is_some(),
        "fp128_d64_onehot_recursive" => {
            <akita_config::RecursiveCommitmentConfig<fp128::D64OneHot> as CommitmentConfig>::schedule_catalog()
                .is_some()
        }
        "fp128_d64_onehot_recursive_multi_chunk_w8r2" => {
            <akita_config::RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk> as CommitmentConfig>::schedule_catalog()
                .is_some()
        }
        "fp128_d64_full" => fp128::D64Full::schedule_catalog().is_some(),
        "fp128_d64_onehot_tensor" => {
            tensor_verifier::fp128::D64OneHotTensor::schedule_catalog().is_some()
        }
        "fp128_d64_onehot_multi_chunk" => fp128::D64OneHotMultiChunk::schedule_catalog().is_some(),
        "fp128_d64_onehot_multi_chunk_w2r2" => {
            fp128::D64OneHotMultiChunkW2R2::schedule_catalog().is_some()
        }
        "fp128_d64_onehot_multi_chunk_w4r2" => {
            fp128::D64OneHotMultiChunkW4R2::schedule_catalog().is_some()
        }
        "fp128_d64_full_multi_chunk" => fp128::D64FullMultiChunk::schedule_catalog().is_some(),
        "fp64_d128" => fp64::D128Full::schedule_catalog().is_some(),
        "fp64_d128_onehot" => fp64::D128OneHot::schedule_catalog().is_some(),
        "fp64_d256_onehot" => fp64::D256OneHot::schedule_catalog().is_some(),
        "fp32_d128_onehot" => fp32::D128OneHot::schedule_catalog().is_some(),
        "fp32_d256_onehot" => fp32::D256OneHot::schedule_catalog().is_some(),
        other => panic!("unknown generated family for catalog guard: {other}"),
    }
}

#[cfg(feature = "all-schedules")]
fn assert_table_hit(
    module_name: &str,
    catalog: &akita_planner::GeneratedScheduleTable,
    keys: &[PolynomialGroupLayout],
) {
    let hit = keys
        .iter()
        .any(|&key| table_entry(*catalog, &AkitaScheduleLookupKey::single(key)).is_some());
    assert!(
        hit,
        "family {module_name} must have at least one shipped-table key hit (non-vacuous catalog guard)"
    );
}

#[cfg(feature = "all-schedules")]
fn prepare_family_catalog<Cfg: CommitmentConfig>(
    module_name: &str,
    keys: &[PolynomialGroupLayout],
) -> akita_planner::GeneratedScheduleTable {
    let catalog = Cfg::schedule_catalog().unwrap_or_else(|| {
        panic!("family {module_name} must expose schedule_catalog() under all-schedules")
    });
    validate_generated_schedule_table(
        &catalog,
        &policy_of::<Cfg>(),
        &Cfg::ring_challenge_config,
        &Cfg::fold_challenge_shape_at_level,
    )
    .unwrap_or_else(|e| panic!("catalog validation failed for {module_name}: {e}"));
    assert!(
        catalog_entries_sorted_for_lookup(catalog.entries),
        "family {module_name} catalog entries must be sorted for binary lookup"
    );
    assert_table_hit(module_name, &catalog, keys);
    catalog
}

#[cfg(feature = "all-schedules")]
fn family_catalog(
    family: &GeneratedFamily,
    keys: &[PolynomialGroupLayout],
) -> akita_planner::GeneratedScheduleTable {
    match family.module_name {
        "fp128_d128_full" => prepare_family_catalog::<fp128::D128Full>(family.module_name, keys),
        "fp128_d128_onehot" => {
            prepare_family_catalog::<fp128::D128OneHot>(family.module_name, keys)
        }
        "fp128_d64_onehot" => prepare_family_catalog::<fp128::D64OneHot>(family.module_name, keys),
        "fp128_d64_onehot_recursive" => prepare_family_catalog::<
            akita_config::RecursiveCommitmentConfig<fp128::D64OneHot>,
        >(family.module_name, keys),
        "fp128_d64_onehot_recursive_multi_chunk_w8r2" => prepare_family_catalog::<
            akita_config::RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>,
        >(family.module_name, keys),
        "fp128_d64_full" => prepare_family_catalog::<fp128::D64Full>(family.module_name, keys),
        "fp128_d64_onehot_tensor" => prepare_family_catalog::<
            tensor_verifier::fp128::D64OneHotTensor,
        >(family.module_name, keys),
        "fp128_d64_onehot_multi_chunk" => {
            prepare_family_catalog::<fp128::D64OneHotMultiChunk>(family.module_name, keys)
        }
        "fp128_d64_onehot_multi_chunk_w2r2" => {
            prepare_family_catalog::<fp128::D64OneHotMultiChunkW2R2>(family.module_name, keys)
        }
        "fp128_d64_onehot_multi_chunk_w4r2" => {
            prepare_family_catalog::<fp128::D64OneHotMultiChunkW4R2>(family.module_name, keys)
        }
        "fp128_d64_full_multi_chunk" => {
            prepare_family_catalog::<fp128::D64FullMultiChunk>(family.module_name, keys)
        }
        "fp64_d128" => prepare_family_catalog::<fp64::D128Full>(family.module_name, keys),
        "fp64_d128_onehot" => prepare_family_catalog::<fp64::D128OneHot>(family.module_name, keys),
        "fp64_d256_onehot" => prepare_family_catalog::<fp64::D256OneHot>(family.module_name, keys),
        "fp32_d128_onehot" => prepare_family_catalog::<fp32::D128OneHot>(family.module_name, keys),
        "fp32_d256_onehot" => prepare_family_catalog::<fp32::D256OneHot>(family.module_name, keys),
        other => panic!("unknown generated family for catalog guard: {other}"),
    }
}

fn assert_group_batch_table_hits<Cfg: CommitmentConfig>(
    module_name: &str,
    keys: &[AkitaScheduleLookupKey],
) {
    if keys.is_empty() {
        return;
    }
    let catalog = Cfg::schedule_catalog()
        .unwrap_or_else(|| panic!("family {module_name} must expose schedule_catalog()"));
    let missing = keys
        .iter()
        .filter(|key| table_entry(catalog, key).is_none())
        .take(3)
        .map(|key| format!("{key:?}"))
        .collect::<Vec<_>>();
    assert!(
        missing.is_empty(),
        "family {module_name} must have shipped grouped-table hits for every enumerated multi-group key; first missing keys: {}",
        missing.join("\n  ")
    );
}

fn assert_family_group_batch_table_hit(family: &GeneratedFamily, keys: &[AkitaScheduleLookupKey]) {
    match family.module_name {
        "fp128_d128_full" => {
            assert_group_batch_table_hits::<fp128::D128Full>(family.module_name, keys)
        }
        "fp128_d128_onehot" => {
            assert_group_batch_table_hits::<fp128::D128OneHot>(family.module_name, keys)
        }
        "fp128_d64_onehot" => {
            assert_group_batch_table_hits::<fp128::D64OneHot>(family.module_name, keys)
        }
        "fp128_d64_onehot_recursive" => assert_group_batch_table_hits::<
            akita_config::RecursiveCommitmentConfig<fp128::D64OneHot>,
        >(family.module_name, keys),
        "fp128_d64_onehot_recursive_multi_chunk_w8r2" => assert_group_batch_table_hits::<
            akita_config::RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>,
        >(family.module_name, keys),
        "fp128_d64_full" => {
            assert_group_batch_table_hits::<fp128::D64Full>(family.module_name, keys)
        }
        "fp128_d64_onehot_tensor" => assert_group_batch_table_hits::<
            tensor_verifier::fp128::D64OneHotTensor,
        >(family.module_name, keys),
        "fp64_d128" => assert_group_batch_table_hits::<fp64::D128Full>(family.module_name, keys),
        "fp64_d128_onehot" => {
            assert_group_batch_table_hits::<fp64::D128OneHot>(family.module_name, keys)
        }
        "fp64_d256_onehot" => {
            assert_group_batch_table_hits::<fp64::D256OneHot>(family.module_name, keys)
        }
        "fp32_d128_onehot" => {
            assert_group_batch_table_hits::<fp32::D128OneHot>(family.module_name, keys)
        }
        "fp32_d256_onehot" => {
            assert_group_batch_table_hits::<fp32::D256OneHot>(family.module_name, keys)
        }
        other => panic!("unknown generated family for grouped catalog guard: {other}"),
    }
}

#[cfg(feature = "all-schedules")]
fn table_backed_group_batch_schedule(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    key: &AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    if let Some(entry) = table_entry(catalog, key) {
        return schedule_from_entry(
            entry,
            key,
            &(family.policy)(),
            family.ring_challenge_config,
            family.fold_challenge_shape_at_level,
        );
    }
    (family.regen_group_batch)(key.clone())
}

#[cfg(not(feature = "all-schedules"))]
fn table_backed_group_batch_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    Cfg::runtime_schedule(key.clone())
}

#[cfg(feature = "all-schedules")]
fn resolve_family_group_batch_schedule(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    key: &AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    table_backed_group_batch_schedule(family, catalog, key)
}

#[cfg(not(feature = "all-schedules"))]
fn resolve_family_group_batch_schedule(
    family: &GeneratedFamily,
    key: &AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    match family.module_name {
        "fp128_d128_full" => table_backed_group_batch_schedule::<fp128::D128Full>(key),
        "fp128_d128_onehot" => table_backed_group_batch_schedule::<fp128::D128OneHot>(key),
        "fp128_d64_onehot" => table_backed_group_batch_schedule::<fp128::D64OneHot>(key),
        "fp128_d64_onehot_recursive" => table_backed_group_batch_schedule::<
            akita_config::RecursiveCommitmentConfig<fp128::D64OneHot>,
        >(key),
        "fp128_d64_onehot_recursive_multi_chunk_w8r2" => table_backed_group_batch_schedule::<
            akita_config::RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>,
        >(key),
        "fp128_d64_full" => table_backed_group_batch_schedule::<fp128::D64Full>(key),
        "fp128_d64_onehot_tensor" => {
            table_backed_group_batch_schedule::<tensor_verifier::fp128::D64OneHotTensor>(key)
        }
        "fp64_d128" => table_backed_group_batch_schedule::<fp64::D128Full>(key),
        "fp64_d128_onehot" => table_backed_group_batch_schedule::<fp64::D128OneHot>(key),
        "fp64_d256_onehot" => table_backed_group_batch_schedule::<fp64::D256OneHot>(key),
        "fp32_d128_onehot" => table_backed_group_batch_schedule::<fp32::D128OneHot>(key),
        "fp32_d256_onehot" => table_backed_group_batch_schedule::<fp32::D256OneHot>(key),
        other => panic!("unknown generated family for multi-group schedule guard: {other}"),
    }
}

#[cfg(feature = "all-schedules")]
fn table_backed_expanded(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_field::AkitaError> {
    let lookup_key = AkitaScheduleLookupKey::single(key);
    if let Some(entry) = table_entry(catalog, &lookup_key) {
        return schedule_from_entry(
            entry,
            &lookup_key,
            &(family.policy)(),
            family.ring_challenge_config,
            family.fold_challenge_shape_at_level,
        );
    }
    (family.regen)(key)
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

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
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
    catalog: akita_planner::GeneratedScheduleTable,
    key: PolynomialGroupLayout,
) -> Option<Mismatch> {
    compare_schedule_results(
        family,
        key,
        table_backed_expanded(family, catalog, key),
        (family.regen)(key),
    )
}

#[cfg(not(feature = "all-schedules"))]
fn compare_scalar_key(family: &GeneratedFamily, key: PolynomialGroupLayout) -> Option<Mismatch> {
    compare_schedule_results(family, key, (family.table_backed)(key), (family.regen)(key))
}

#[cfg(feature = "all-schedules")]
fn check_scalar_keys(
    family: &GeneratedFamily,
    keys: &[PolynomialGroupLayout],
    catalog: akita_planner::GeneratedScheduleTable,
    into: &mut Vec<Mismatch>,
) {
    let workers = worker_count();

    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for &key in chunk {
                            if let Some(mismatch) = compare_scalar_key(family, catalog, key) {
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
        if let Some(mismatch) = compare_scalar_key(family, catalog, key) {
            into.push(mismatch);
        }
    }
}

#[cfg(not(feature = "all-schedules"))]
fn check_scalar_keys(
    family: &GeneratedFamily,
    keys: &[PolynomialGroupLayout],
    into: &mut Vec<Mismatch>,
) {
    let workers = worker_count();

    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for &key in chunk {
                            if let Some(mismatch) = compare_scalar_key(family, key) {
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
        if let Some(mismatch) = compare_scalar_key(family, key) {
            into.push(mismatch);
        }
    }
}

#[cfg(feature = "all-schedules")]
fn compare_group_batch_key(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    key: &AkitaScheduleLookupKey,
) -> Option<Mismatch> {
    let table_backed =
        resolve_family_group_batch_schedule(family, catalog, key).unwrap_or_else(|e| {
            panic!(
                "table-backed multi-group schedule failed for family {} key={key:?}: {e}",
                family.module_name
            )
        });
    let regenerated = (family.regen_group_batch)(key.clone()).unwrap_or_else(|e| {
        panic!(
            "multi-group DP regen failed for family {} key={key:?}: {e}",
            family.module_name
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
    key: &AkitaScheduleLookupKey,
) -> Option<Mismatch> {
    let table_backed = resolve_family_group_batch_schedule(family, key).unwrap_or_else(|e| {
        panic!(
            "table-backed multi-group schedule failed for family {} key={key:?}: {e}",
            family.module_name
        )
    });
    let regenerated = (family.regen_group_batch)(key.clone()).unwrap_or_else(|e| {
        panic!(
            "multi-group DP regen failed for family {} key={key:?}: {e}",
            family.module_name
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
fn check_group_batch_keys(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    keys: &[AkitaScheduleLookupKey],
    into: &mut Vec<Mismatch>,
) {
    if keys.is_empty() {
        return;
    }

    let workers = worker_count();
    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for key in chunk {
                            if let Some(mismatch) = compare_group_batch_key(family, catalog, key) {
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

    for key in keys {
        if let Some(mismatch) = compare_group_batch_key(family, catalog, key) {
            into.push(mismatch);
        }
    }
}

#[cfg(not(feature = "all-schedules"))]
fn check_group_batch_keys(
    family: &GeneratedFamily,
    keys: &[AkitaScheduleLookupKey],
    into: &mut Vec<Mismatch>,
) {
    if keys.is_empty() {
        return;
    }

    let workers = worker_count();
    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for key in chunk {
                            if let Some(mismatch) = compare_group_batch_key(family, key) {
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

    for key in keys {
        if let Some(mismatch) = compare_group_batch_key(family, key) {
            into.push(mismatch);
        }
    }
}

fn check_family(family: &GeneratedFamily, into: &mut Vec<Mismatch>) {
    if !family_catalog_is_linked(family) {
        return;
    }

    let keys: Vec<PolynomialGroupLayout> = family_keys(family)
        .unwrap_or_else(|e| panic!("family {} key enumeration failed: {e}", family.module_name));

    #[cfg(feature = "all-schedules")]
    {
        let catalog = family_catalog(family, &keys);
        let group_batch_keys = (family.group_batch_keys)(family).unwrap_or_else(|e| {
            panic!(
                "family {} multi-group key enumeration failed: {e}",
                family.module_name
            )
        });
        check_scalar_keys(family, &keys, catalog, into);
        if family.emit_group_batch {
            assert_family_group_batch_table_hit(family, &group_batch_keys);
            check_group_batch_keys(family, catalog, &group_batch_keys, into);
        }
    }
    #[cfg(not(feature = "all-schedules"))]
    {
        let group_batch_keys = (family.group_batch_keys)(family).unwrap_or_else(|e| {
            panic!(
                "family {} multi-group key enumeration failed: {e}",
                family.module_name
            )
        });
        if family.emit_group_batch {
            assert_family_group_batch_table_hit(family, &group_batch_keys);
        }
        check_scalar_keys(family, &keys, into);
        if family.emit_group_batch {
            check_group_batch_keys(family, &group_batch_keys, into);
        }
    }
}

fn regen_hint() -> &'static str {
    "cargo run --release -p akita-config --bin gen_schedule_tables -- \
     crates/akita-schedules/src/generated"
}

/// The shipped tables must expand to exactly what the key-shaped DP produces.
/// Rolled into one test so the panic message can summarize per-family
/// mismatch counts.
#[test]
fn generated_schedule_tables_match_key_planner() {
    let mut mismatches = Vec::new();
    for family in ALL_GENERATED_FAMILIES {
        check_family(family, &mut mismatches);
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
         Regenerate the shipped tables with:\n  {hint}",
        count = mismatches.len(),
        hint = regen_hint(),
    );
}
