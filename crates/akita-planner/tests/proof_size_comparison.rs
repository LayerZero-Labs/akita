//! One-off proof-size comparison test for the placeholder-removal refactor.
//!
//! For every `(family, key)` covered by the shipped tables we materialize:
//!
//! - the **table-backed schedule** via `find_schedule(key, true)`, which
//!   serves from `Cfg::schedule_table()` when the entry exists and
//!   re-materializes its `total_bytes` from the audited shape;
//! - the **regenerated schedule** via `find_schedule(key, false)`, which
//!   runs the pure DP (new comparator) from scratch.
//!
//! The test asserts that `new_total <= old_total` for every key — i.e.
//! the refactor never increases the proof size produced by the
//! planner — and prints a summary of the size deltas.

#![allow(missing_docs)]

use akita_planner::generated_families::{family_keys, ALL_GENERATED_FAMILIES};

#[derive(Debug, Default, Clone, Copy)]
struct FamilyStats {
    keys: usize,
    improved: usize,
    unchanged: usize,
    regressed: usize,
    total_old: u128,
    total_new: u128,
}

#[test]
fn refactor_does_not_increase_proof_sizes() {
    let mut regressions: Vec<String> = Vec::new();
    let mut by_family: std::collections::BTreeMap<&str, FamilyStats> =
        std::collections::BTreeMap::new();

    for family in ALL_GENERATED_FAMILIES {
        if (family.schedule_table)().is_none() {
            // No shipped table — skip (matches drift-guard semantics).
            continue;
        }
        let keys = family_keys(family).unwrap_or_else(|e| {
            panic!("family {} key enumeration failed: {e}", family.module_name)
        });

        let stats = by_family.entry(family.module_name).or_default();
        for key in keys {
            let old = (family.regen_with_lookup)(key).unwrap_or_else(|e| {
                panic!(
                    "table-backed schedule failed for family {} key={key:?}: {e}",
                    family.module_name
                )
            });
            let new = (family.regen)(key).unwrap_or_else(|e| {
                panic!(
                    "DP regen failed for family {} key={key:?}: {e}",
                    family.module_name
                )
            });
            stats.keys += 1;
            stats.total_old += old.total_bytes as u128;
            stats.total_new += new.total_bytes as u128;
            match new.total_bytes.cmp(&old.total_bytes) {
                std::cmp::Ordering::Less => stats.improved += 1,
                std::cmp::Ordering::Equal => stats.unchanged += 1,
                std::cmp::Ordering::Greater => {
                    stats.regressed += 1;
                    regressions.push(format!(
                        "{family}: key={key:?} old={old} new={new} delta=+{delta}",
                        family = family.module_name,
                        old = old.total_bytes,
                        new = new.total_bytes,
                        delta = new.total_bytes - old.total_bytes,
                    ));
                }
            }
        }
    }

    eprintln!("\nPer-family proof-size comparison (old=table-backed, new=pure DP):");
    eprintln!(
        "  {:<28}  {:>5}  {:>9}  {:>9}  {:>9}  {:>14}  {:>14}  {:>7}",
        "family", "keys", "improved", "unchanged", "regressed", "total_old", "total_new", "ratio",
    );
    for (family, stats) in &by_family {
        let ratio = if stats.total_old == 0 {
            1.0
        } else {
            stats.total_new as f64 / stats.total_old as f64
        };
        eprintln!(
            "  {:<28}  {:>5}  {:>9}  {:>9}  {:>9}  {:>14}  {:>14}  {:>7.4}",
            family,
            stats.keys,
            stats.improved,
            stats.unchanged,
            stats.regressed,
            stats.total_old,
            stats.total_new,
            ratio,
        );
    }

    if !regressions.is_empty() {
        let preview: String = regressions
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ");
        panic!(
            "{count} key(s) regressed (new total_bytes > old total_bytes).\n\
             First offenders:\n  {preview}",
            count = regressions.len(),
        );
    }
}
