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
//!   for scalar schedules, or `Cfg::runtime_schedule` for grouped-root
//!   schedules (each expands compact rows through the canonical walker);
//! - **regenerated** via `family.regen` / `family.regen_group_batch`, which runs
//!   the pure DP from scratch.
//!
//! The comparison is over the *fully resolved* [`Schedule`] — every step's
//! expanded [`LevelParams`] (SIS buckets + derived matrix widths,
//! which the compact 7-tuple drops), step kinds / witness shapes, and total
//! proof bytes. This is strictly stronger than diffing the compact
//! `GeneratedStep` tuples: it catches any drift where the table-hit
//! expansion would carry a different `a_key.coeff_linf_bound()` (or width, or
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
use akita_types::{
    AkitaScheduleLookupKey, DirectStep, FoldStep, PolynomialGroupLayout, Schedule, Step,
};

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
            !family.emit_group_batch
                || (policy.decomposition.log_commit_bound == 1 && !policy.tiered),
            "family {} must not emit grouped rows for unsupported grouped-root policies",
            family.module_name
        );
    }
}

fn family_catalog_is_linked(family: &GeneratedFamily) -> bool {
    match family.module_name {
        "fp128_d128_full" => fp128::D128Full::schedule_catalog().is_some(),
        "fp128_d128_onehot" => fp128::D128OneHot::schedule_catalog().is_some(),
        "fp128_d64_onehot" => fp128::D64OneHot::schedule_catalog().is_some(),
        "fp128_d64_full" => fp128::D64Full::schedule_catalog().is_some(),
        "fp128_d64_onehot_tensor" => {
            tensor_verifier::fp128::D64OneHotTensor::schedule_catalog().is_some()
        }
        "fp128_d64_onehot_tiered" => fp128::D64OneHotTiered::schedule_catalog().is_some(),
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
fn grouped_keys_from_catalog(
    catalog: akita_planner::GeneratedScheduleTable,
) -> Vec<AkitaScheduleLookupKey> {
    catalog
        .entries
        .iter()
        .filter(|entry| !entry.precommitteds.is_empty())
        .map(|entry| AkitaScheduleLookupKey {
            final_group: entry.final_group,
            precommitteds: entry.precommitteds.to_vec(),
        })
        .collect()
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
        "fp128_d64_full" => prepare_family_catalog::<fp128::D64Full>(family.module_name, keys),
        "fp128_d64_onehot_tensor" => prepare_family_catalog::<
            tensor_verifier::fp128::D64OneHotTensor,
        >(family.module_name, keys),
        "fp128_d64_onehot_tiered" => {
            prepare_family_catalog::<fp128::D64OneHotTiered>(family.module_name, keys)
        }
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
        "family {module_name} must have shipped grouped-table hits for every enumerated grouped key; first missing keys: {}",
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
        "fp128_d64_full" => {
            assert_group_batch_table_hits::<fp128::D64Full>(family.module_name, keys)
        }
        "fp128_d64_onehot_tensor" => assert_group_batch_table_hits::<
            tensor_verifier::fp128::D64OneHotTensor,
        >(family.module_name, keys),
        "fp128_d64_onehot_tiered" => {
            assert_group_batch_table_hits::<fp128::D64OneHotTiered>(family.module_name, keys)
        }
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

fn table_backed_group_batch_schedule<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
) -> Result<Schedule, AkitaError> {
    Cfg::runtime_schedule(key.clone())
}

fn resolve_family_group_batch_schedule(
    family: &GeneratedFamily,
    key: &AkitaScheduleLookupKey,
) -> Result<Schedule, AkitaError> {
    match family.module_name {
        "fp128_d128_full" => table_backed_group_batch_schedule::<fp128::D128Full>(key),
        "fp128_d128_onehot" => table_backed_group_batch_schedule::<fp128::D128OneHot>(key),
        "fp128_d64_onehot" => table_backed_group_batch_schedule::<fp128::D64OneHot>(key),
        "fp128_d64_full" => table_backed_group_batch_schedule::<fp128::D64Full>(key),
        "fp128_d64_onehot_tensor" => {
            table_backed_group_batch_schedule::<tensor_verifier::fp128::D64OneHotTensor>(key)
        }
        "fp128_d64_onehot_tiered" => {
            table_backed_group_batch_schedule::<fp128::D64OneHotTiered>(key)
        }
        "fp64_d128" => table_backed_group_batch_schedule::<fp64::D128Full>(key),
        "fp64_d128_onehot" => table_backed_group_batch_schedule::<fp64::D128OneHot>(key),
        "fp64_d256_onehot" => table_backed_group_batch_schedule::<fp64::D256OneHot>(key),
        "fp32_d128_onehot" => table_backed_group_batch_schedule::<fp32::D128OneHot>(key),
        "fp32_d256_onehot" => table_backed_group_batch_schedule::<fp32::D256OneHot>(key),
        other => panic!("unknown generated family for grouped schedule guard: {other}"),
    }
}

#[cfg(feature = "all-schedules")]
fn table_backed_expanded(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    key: PolynomialGroupLayout,
) -> Result<Schedule, akita_field::AkitaError> {
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

/// Canonical string form of a fully resolved schedule: total proof bytes
/// plus the `Debug` of every step (which includes each level's expanded
/// `LevelParams` — collision buckets, matrix widths, ranks — and the direct
/// witness shapes).
fn render_schedule(schedule: &Schedule) -> String {
    format!(
        "total_bytes={} steps={:?}",
        schedule.total_bytes, schedule.steps
    )
}

fn fold_steps_equal(left: &FoldStep, right: &FoldStep) -> bool {
    left.current_w_len == right.current_w_len
        && left.next_w_len == right.next_w_len
        && left.level_bytes == right.level_bytes
        && left.params == right.params
}

fn direct_steps_equal(left: &DirectStep, right: &DirectStep) -> bool {
    left.current_w_len == right.current_w_len
        && left.witness_shape == right.witness_shape
        && left.direct_bytes == right.direct_bytes
        && left.params == right.params
}

fn steps_equal(left: &Step, right: &Step) -> bool {
    match (left, right) {
        (Step::Fold(left), Step::Fold(right)) => fold_steps_equal(left, right),
        (Step::Direct(left), Step::Direct(right)) => direct_steps_equal(left, right),
        _ => false,
    }
}

fn schedules_equal(left: &Schedule, right: &Schedule) -> bool {
    if left.total_bytes != right.total_bytes {
        return false;
    }
    if left.steps.len() != right.steps.len() {
        return false;
    }
    for (l, r) in left.steps.iter().zip(right.steps.iter()) {
        if !steps_equal(l, r) {
            return false;
        }
    }
    true
}

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
}

#[cfg(feature = "all-schedules")]
fn compare_scalar_key(
    family: &GeneratedFamily,
    catalog: akita_planner::GeneratedScheduleTable,
    key: PolynomialGroupLayout,
) -> Option<Mismatch> {
    let table_backed = table_backed_expanded(family, catalog, key).unwrap_or_else(|e| {
        panic!(
            "table-backed schedule failed for family {} key={key:?}: {e}",
            family.module_name
        )
    });
    let regenerated = (family.regen)(key).unwrap_or_else(|e| {
        panic!(
            "DP regen failed for family {} key={key:?}: {e}",
            family.module_name
        )
    });

    if schedules_equal(&table_backed, &regenerated) {
        return None;
    }
    Some(Mismatch {
        family: family.module_name,
        key: format!("{key:?}"),
        table_backed: render_schedule(&table_backed),
        regenerated: render_schedule(&regenerated),
    })
}

#[cfg(not(feature = "all-schedules"))]
fn compare_scalar_key(family: &GeneratedFamily, key: PolynomialGroupLayout) -> Option<Mismatch> {
    let table_backed = (family.table_backed)(key).unwrap_or_else(|e| {
        panic!(
            "table-backed schedule failed for family {} key={key:?}: {e}",
            family.module_name
        )
    });
    let regenerated = (family.regen)(key).unwrap_or_else(|e| {
        panic!(
            "DP regen failed for family {} key={key:?}: {e}",
            family.module_name
        )
    });

    if schedules_equal(&table_backed, &regenerated) {
        return None;
    }
    Some(Mismatch {
        family: family.module_name,
        key: format!("{key:?}"),
        table_backed: render_schedule(&table_backed),
        regenerated: render_schedule(&regenerated),
    })
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

fn compare_group_batch_key(
    family: &GeneratedFamily,
    key: &AkitaScheduleLookupKey,
) -> Option<Mismatch> {
    let table_backed = resolve_family_group_batch_schedule(family, key).unwrap_or_else(|e| {
        panic!(
            "table-backed grouped schedule failed for family {} key={key:?}: {e}",
            family.module_name
        )
    });
    let regenerated = (family.regen_group_batch)(key.clone()).unwrap_or_else(|e| {
        panic!(
            "grouped DP regen failed for family {} key={key:?}: {e}",
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
        let group_batch_keys = grouped_keys_from_catalog(catalog);
        check_scalar_keys(family, &keys, catalog, into);
        if family.emit_group_batch {
            check_group_batch_keys(family, &group_batch_keys, into);
        }
    }
    #[cfg(not(feature = "all-schedules"))]
    {
        let group_batch_keys = (family.group_batch_keys)(family).unwrap_or_else(|e| {
            panic!(
                "family {} grouped key enumeration failed: {e}",
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

/// The shipped tables must expand to exactly what `find_schedule` produces.
/// Rolled into one test so the panic message can summarize per-family
/// mismatch counts.
#[test]
fn generated_schedule_tables_match_find_schedule() {
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
        "{count} schedule-table issue(s) disagree with `find_schedule` output.\n\
         Per-family counts:\n  {summary}\n\n\
         First issues:\n{preview}\n\
         Regenerate the shipped tables with:\n  {hint}",
        count = mismatches.len(),
        hint = regen_hint(),
    );
}
