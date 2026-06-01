//! Guard test: the generated schedule tables shipped in `akita-types::generated`
//! must agree exactly with what `gen_schedule_tables` would emit today.
//!
//! Coverage is metadata-driven: every entry in
//! [`akita_config::generated_families::ALL_GENERATED_FAMILIES`] is checked
//! against `Cfg::schedule_table()`, so adding a new family to the
//! generator picks it up here automatically (no per-family handwritten
//! row mirror).
//!
//! For each family the test asserts, in this order:
//!
//! 1. **Table is shipped.** `Cfg::schedule_table()` returns `Some(_)`.
//! 2. **Length match.** The shipped table has exactly the same number of
//!    rows as the regenerated key cross-product (no extras, no missing
//!    entries, no duplicates).
//! 3. **Ordered key sequence.** The `i`-th shipped row carries the
//!    `i`-th regenerated key. Catches reordering, duplicates surviving
//!    behind a length match (a duplicate plus a missing key would still
//!    pass a multiset check), and serialization regressions.
//! 4. **Step content equality.** Every shipped entry's step sequence
//!    matches what the pure DP regen (`family.regen`) produces.
//!
//! When this test fails the panic message lists per-family mismatch
//! counts, the first three offending diffs, and the regenerate command
//! for the active feature set.

#![allow(missing_docs)]

use akita_config::generated_families::{family_keys, GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_types::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleKey, GeneratedScheduleTableEntry,
    GeneratedStep,
};
use akita_types::{generated_schedule_lookup_key, AkitaScheduleLookupKey, Schedule, Step};

/// One observed mismatch between the shipped table and the regen output.
enum Mismatch {
    MissingTable {
        family: &'static str,
    },
    Length {
        family: &'static str,
        expected_len: usize,
        actual_len: usize,
    },
    Key {
        family: &'static str,
        position: usize,
        expected_key: GeneratedScheduleKey,
        actual_key: GeneratedScheduleKey,
    },
    Steps {
        family: &'static str,
        position: usize,
        key: GeneratedScheduleKey,
        expected_steps: Vec<GeneratedStep>,
        actual_steps: Vec<GeneratedStep>,
    },
}

impl Mismatch {
    fn family(&self) -> &'static str {
        match self {
            Mismatch::MissingTable { family }
            | Mismatch::Length { family, .. }
            | Mismatch::Key { family, .. }
            | Mismatch::Steps { family, .. } => family,
        }
    }

    fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        match self {
            Mismatch::MissingTable { family } => {
                let _ = writeln!(
                    s,
                    "  family={family}: Cfg::schedule_table() returned None; \
                     cannot diff against a shipped table"
                );
            }
            Mismatch::Length {
                family,
                expected_len,
                actual_len,
            } => {
                let _ = writeln!(
                    s,
                    "  family={family}: row count mismatch \
                     (regen expects {expected_len}, table has {actual_len})"
                );
            }
            Mismatch::Key {
                family,
                position,
                expected_key,
                actual_key,
            } => {
                let _ = writeln!(s, "  family={family} row {position}: key mismatch");
                let _ = writeln!(s, "    expected key: {expected_key:?}");
                let _ = writeln!(s, "    table key:    {actual_key:?}");
            }
            Mismatch::Steps {
                family,
                position,
                key,
                expected_steps,
                actual_steps,
            } => {
                let _ = writeln!(
                    s,
                    "  family={family} row {position}: step mismatch for key={key:?}"
                );
                let _ = writeln!(s, "    DP regenerated steps ({}):", expected_steps.len());
                for step in expected_steps {
                    let _ = writeln!(s, "      {step:?}");
                }
                let _ = writeln!(s, "    table entry steps     ({}):", actual_steps.len());
                for step in actual_steps {
                    let _ = writeln!(s, "      {step:?}");
                }
            }
        }
        s
    }
}

fn generated_fold_from_params(p: &akita_types::LevelParams) -> GeneratedFoldStep {
    GeneratedFoldStep {
        ring_d: p.ring_dimension as u32,
        log_basis: p.log_basis,
        m_vars: p.log_block_len() as u32,
        r_vars: p.log_num_blocks() as u32,
        n_a: p.a_key.row_len() as u32,
        n_b: p.b_key.row_len() as u32,
        n_d: p.d_key.row_len() as u32,
    }
}

fn schedule_to_generated_steps(schedule: &Schedule) -> Vec<GeneratedStep> {
    schedule
        .steps
        .iter()
        .map(|step| match step {
            Step::Fold(fold) => GeneratedStep::Fold(generated_fold_from_params(&fold.params)),
            Step::Direct(direct) => GeneratedStep::Direct(GeneratedDirectStep {
                commit: direct.params.as_ref().map(generated_fold_from_params),
            }),
        })
        .collect()
}

fn entry_steps(entry: &GeneratedScheduleTableEntry) -> Vec<GeneratedStep> {
    entry.steps.to_vec()
}

fn check_family(family: &GeneratedFamily, into: &mut Vec<Mismatch>) {
    let Some(table) = (family.schedule_table)() else {
        into.push(Mismatch::MissingTable {
            family: family.module_name,
        });
        return;
    };

    let expected_keys: Vec<AkitaScheduleLookupKey> = family_keys(family)
        .unwrap_or_else(|e| panic!("family {} key enumeration failed: {e}", family.module_name));

    if table.entries.len() != expected_keys.len() {
        into.push(Mismatch::Length {
            family: family.module_name,
            expected_len: expected_keys.len(),
            actual_len: table.entries.len(),
        });
        // Continue with the prefix that does line up so we still surface
        // any per-row content mismatches.
    }

    let pair_count = table.entries.len().min(expected_keys.len());
    for (position, (entry, &expected_key)) in table
        .entries
        .iter()
        .zip(expected_keys.iter())
        .take(pair_count)
        .enumerate()
    {
        let expected_generated_key = generated_schedule_lookup_key(expected_key);
        if entry.key != expected_generated_key {
            into.push(Mismatch::Key {
                family: family.module_name,
                position,
                expected_key: expected_generated_key,
                actual_key: entry.key,
            });
            // Don't compare steps when the key is wrong — the regen
            // would solve for a different key.
            continue;
        }

        let schedule = (family.regen)(expected_key).unwrap_or_else(|e| {
            panic!(
                "DP regen failed for family {} key={expected_generated_key:?}: {e}",
                family.module_name
            )
        });
        let expected_steps = schedule_to_generated_steps(&schedule);
        let actual_steps = entry_steps(entry);
        if expected_steps != actual_steps {
            into.push(Mismatch::Steps {
                family: family.module_name,
                position,
                key: expected_generated_key,
                expected_steps,
                actual_steps,
            });
        }
    }
}

fn regen_hint() -> &'static str {
    if cfg!(feature = "zk") {
        "cargo run --release -p akita-config --features zk --bin gen_schedule_tables -- \
         crates/akita-types/src/generated"
    } else {
        "cargo run --release -p akita-config --bin gen_schedule_tables -- \
         crates/akita-types/src/generated"
    }
}

/// All four exactness checks (presence, length, ordered keys, steps)
/// rolled into one test so the panic message can summarize per-family
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
        *buckets.entry(m.family()).or_default() += 1;
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
