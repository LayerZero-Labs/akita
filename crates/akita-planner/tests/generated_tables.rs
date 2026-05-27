//! Guard test: the generated schedule tables shipped in `akita-types::generated`
//! must agree, byte-for-byte at the step level, with the schedules that
//! `find_optimal_schedule` would emit today.
//!
//! When this test fails it almost always means the planner heuristics, layout
//! derivation, or proof-size accounting have moved without re-running
//! `gen_schedule_tables`. The fix is to regenerate the tables with the same
//! command the test prints in its failure message:
//!
//! ```bash
//! cargo run --release -p akita-planner --bin gen_schedule_tables -- \
//!     crates/akita-types/src/generated
//!
//! cargo run --release -p akita-planner --features zk \
//!     --bin gen_schedule_tables -- crates/akita-types/src/generated
//! ```
//!
//! This test mirrors the family/key enumeration in
//! `akita-planner/src/bin/gen_schedule_tables.rs` so that adding a new family
//! there forces the same family to be covered here.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp16, fp32, fp64};
use akita_config::CommitmentConfig;
use akita_planner::find_optimal_schedule;
use akita_types::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleKey, GeneratedScheduleTable,
    GeneratedScheduleTableEntry, GeneratedStep,
};
use akita_types::{
    generated_schedule_lookup_key, AkitaScheduleLookupKey, ClaimIncidenceSummary, Schedule, Step,
};

/// Mirrors `FamilySpec` from `gen_schedule_tables.rs` but keyed at the type
/// level so the test can pull `Cfg::schedule_table()` directly.
struct FamilyCase {
    name: &'static str,
    min_num_vars: usize,
    max_num_vars: usize,
    check: fn(&'static str, usize, usize) -> Vec<Mismatch>,
}

/// A single (family, key) regeneration discrepancy.
struct Mismatch {
    family: &'static str,
    key: GeneratedScheduleKey,
    expected: Vec<GeneratedStep>,
    actual_in_table: Option<Vec<GeneratedStep>>,
}

impl Mismatch {
    fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "  family={} key={:?}", self.family, self.key);
        let _ = writeln!(s, "    DP regenerated steps ({}):", self.expected.len());
        for step in &self.expected {
            let _ = writeln!(s, "      {step:?}");
        }
        match &self.actual_in_table {
            Some(steps) => {
                let _ = writeln!(s, "    table entry steps     ({}):", steps.len());
                for step in steps {
                    let _ = writeln!(s, "      {step:?}");
                }
            }
            None => {
                let _ = writeln!(s, "    table entry: <missing>");
            }
        }
        s
    }
}

fn schedule_to_generated_steps(schedule: &Schedule) -> Vec<GeneratedStep> {
    schedule
        .steps
        .iter()
        .map(|step| match step {
            Step::Fold(fold) => {
                let p = &fold.params;
                GeneratedStep::Fold(GeneratedFoldStep {
                    ring_d: p.ring_dimension as u32,
                    log_basis: p.log_basis,
                    m_vars: p.log_block_len() as u32,
                    r_vars: p.log_num_blocks() as u32,
                    n_a: p.a_key.row_len() as u32,
                    n_b: p.b_key.row_len() as u32,
                    n_d: p.d_key.row_len() as u32,
                })
            }
            Step::Direct(_) => GeneratedStep::Direct(GeneratedDirectStep),
        })
        .collect()
}

fn table_steps_for_key(
    table: GeneratedScheduleTable,
    key: GeneratedScheduleKey,
) -> Option<Vec<GeneratedStep>> {
    let entry: &GeneratedScheduleTableEntry = table.entries.iter().find(|e| e.key == key)?;
    Some(entry.steps.iter().copied().collect())
}

/// Run the DP for `(singleton, batched_4)` keys across `[min, max]`, collect
/// every mismatch against the table shipped by `Cfg::schedule_table()`.
fn check_family<Cfg: CommitmentConfig>(
    name: &'static str,
    min_num_vars: usize,
    max_num_vars: usize,
) -> Vec<Mismatch> {
    let mut mismatches = Vec::new();
    let table = Cfg::schedule_table().unwrap_or_else(|| {
        panic!(
            "family {name} has no shipped generated schedule table; \
             cannot compare against `find_optimal_schedule`"
        )
    });

    let singleton =
        |nv| ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence should be valid");
    let batched_4 =
        |nv| ClaimIncidenceSummary::same_point(nv, 4).expect("batched-4 incidence should be valid");

    for nv in min_num_vars..=max_num_vars {
        for incidence in [singleton(nv), batched_4(nv)] {
            let key = AkitaScheduleLookupKey::new_from_incidence(&incidence)
                .expect("schedule lookup key should be constructible");
            let schedule = find_optimal_schedule::<Cfg>(key, false).unwrap_or_else(|e| {
                panic!(
                    "DP regen failed for family {name} key={:?}: {e}",
                    generated_schedule_lookup_key(key)
                )
            });
            let expected = schedule_to_generated_steps(&schedule);
            let generated_key = generated_schedule_lookup_key(key);
            let actual_in_table = table_steps_for_key(table, generated_key);
            let mismatched = match &actual_in_table {
                Some(table_steps) => table_steps != &expected,
                None => true,
            };
            if mismatched {
                mismatches.push(Mismatch {
                    family: name,
                    key: generated_key,
                    expected,
                    actual_in_table,
                });
            }
        }
    }

    mismatches
}

// The list below mirrors `ALL_FAMILIES` in `gen_schedule_tables.rs`. Add a row
// here whenever a row is added there (the test will fail loudly if the shipped
// table lacks an expected entry, so a missing row here is detected by the
// neighbouring generated-table-shape tests in `akita-config`).
const ALL_FAMILIES: &[FamilyCase] = &[
    FamilyCase {
        name: "fp128_d32_full",
        min_num_vars: 1,
        max_num_vars: 50,
        check: check_family::<fp128::D32Full>,
    },
    FamilyCase {
        name: "fp128_d32_onehot",
        min_num_vars: 1,
        max_num_vars: 50,
        check: check_family::<fp128::D32OneHot>,
    },
    FamilyCase {
        name: "fp128_d64_full",
        min_num_vars: 1,
        max_num_vars: 50,
        check: check_family::<fp128::D64Full>,
    },
    FamilyCase {
        name: "fp128_d64_onehot",
        min_num_vars: 1,
        max_num_vars: 50,
        check: check_family::<fp128::D64OneHot>,
    },
    FamilyCase {
        name: "fp32_d32",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp32::D32Full>,
    },
    FamilyCase {
        name: "fp32_d32_onehot",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp32::D32OneHot>,
    },
    FamilyCase {
        name: "fp32_d64",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp32::D64Full>,
    },
    FamilyCase {
        name: "fp32_d64_onehot",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp32::D64OneHot>,
    },
    FamilyCase {
        name: "fp16_d32_full",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp16::D32Full>,
    },
    FamilyCase {
        name: "fp16_d32_onehot",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp16::D32OneHot>,
    },
    FamilyCase {
        name: "fp16_d64_full",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp16::D64Full>,
    },
    FamilyCase {
        name: "fp16_d64_onehot",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp16::D64OneHot>,
    },
    FamilyCase {
        name: "fp64_d32",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp64::D32Full>,
    },
    FamilyCase {
        name: "fp64_d32_onehot",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp64::D32OneHot>,
    },
    FamilyCase {
        name: "fp64_d64",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp64::D64Full>,
    },
    FamilyCase {
        name: "fp64_d64_onehot",
        min_num_vars: 1,
        max_num_vars: 32,
        check: check_family::<fp64::D64OneHot>,
    },
];

fn regen_hint() -> &'static str {
    if cfg!(feature = "zk") {
        "cargo run --release -p akita-planner --features zk --bin gen_schedule_tables -- \
         crates/akita-types/src/generated"
    } else {
        "cargo run --release -p akita-planner --bin gen_schedule_tables -- \
         crates/akita-types/src/generated"
    }
}

#[test]
fn generated_schedule_tables_match_find_optimal_schedule() {
    let mut mismatches = Vec::new();
    for family in ALL_FAMILIES {
        mismatches.extend((family.check)(
            family.name,
            family.min_num_vars,
            family.max_num_vars,
        ));
    }
    if mismatches.is_empty() {
        return;
    }
    let mut buckets: std::collections::BTreeMap<&str, Vec<&Mismatch>> =
        std::collections::BTreeMap::new();
    for m in &mismatches {
        buckets.entry(m.family).or_default().push(m);
    }
    let summary = buckets
        .iter()
        .map(|(family, ms)| format!("{family}: {} mismatched entries", ms.len()))
        .collect::<Vec<_>>()
        .join("\n  ");
    let preview = mismatches
        .iter()
        .take(3)
        .map(Mismatch::render)
        .collect::<String>();
    panic!(
        "{count} schedule-table entries disagree with `find_optimal_schedule` output.\n\
         Per-family counts:\n  {summary}\n\n\
         First mismatches:\n{preview}\n\
         Regenerate the shipped tables with:\n  {hint}",
        count = mismatches.len(),
        hint = regen_hint(),
    );
}
