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
//! - **table-backed** via `family.table_backed` (`Cfg::runtime_schedule`),
//!   which serves the shipped table on a hit and expands the compact entry
//!   through the planner's canonical walker;
//! - **regenerated** via `family.regen`, which runs the pure DP from scratch.
//!
//! The comparison is over the *fully resolved* [`Schedule`] — every step's
//! expanded [`LevelParams`] (collision buckets + derived matrix widths,
//! which the compact 7-tuple drops), step kinds / witness shapes, and total
//! proof bytes. This is strictly stronger than diffing the compact
//! `GeneratedStep` tuples: it catches any drift where the table-hit
//! expansion would carry a different `a_key.collision_l2_sq()` (or width, or
//! rank) than the DP used, not just a different stored tuple.
//!
//! When this test fails the panic message lists per-family mismatch counts,
//! the first few offending schedules, and the regenerate command for the
//! active feature set.

#![allow(missing_docs)]

use akita_config::generated_families::{family_keys, GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_types::{AkitaScheduleLookupKey, DirectStep, FoldStep, Schedule, Step};

/// One `(family, key)` whose table-hit expansion disagrees with the DP.
struct Mismatch {
    family: &'static str,
    key: AkitaScheduleLookupKey,
    table_backed: String,
    regenerated: String,
}

impl Mismatch {
    fn render(&self) -> String {
        format!(
            "  family={} key={:?}\n    table-backed: {}\n    regenerated:  {}\n",
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

fn check_family(family: &GeneratedFamily, into: &mut Vec<Mismatch>) {
    let keys: Vec<AkitaScheduleLookupKey> = family_keys(family)
        .unwrap_or_else(|e| panic!("family {} key enumeration failed: {e}", family.module_name));

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4);

    // The table drift guard is a hot CI path. Spread independent keys across a
    // small number of worker threads to reduce wall time on multi-core runners.
    if workers > 1 && keys.len() >= 2 * workers {
        let chunk_size = keys.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles: Vec<_> = keys
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut local = Vec::new();
                        for &key in chunk {
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

                            if !schedules_equal(&table_backed, &regenerated) {
                                local.push(Mismatch {
                                    family: family.module_name,
                                    key,
                                    table_backed: render_schedule(&table_backed),
                                    regenerated: render_schedule(&regenerated),
                                });
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

        if !schedules_equal(&table_backed, &regenerated) {
            into.push(Mismatch {
                family: family.module_name,
                key,
                table_backed: render_schedule(&table_backed),
                regenerated: render_schedule(&regenerated),
            });
        }
    }
}

fn regen_hint() -> &'static str {
    if cfg!(feature = "zk") {
        "cargo run --release -p akita-config --features zk --bin gen_schedule_tables -- \
         crates/akita-schedules/src/generated"
    } else {
        "cargo run --release -p akita-config --bin gen_schedule_tables -- \
         crates/akita-schedules/src/generated"
    }
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
