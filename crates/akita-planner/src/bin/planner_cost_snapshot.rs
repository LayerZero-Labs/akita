//! Per-row planner search-work snapshot and its regression guard.

use akita_planner::emit::PlannerCost;
use std::collections::BTreeMap;
use std::fmt::Write as _;

const SNAPSHOT_HEADER: &str = "family\tlogical_key\tlookup_key_digest\tsuffix_calls\tcompleted_states\tgenerated_candidates\tdescriptor_builds\n";

/// A row fails the guard when one counter exceeds its baseline by more than
/// `ALLOWED_GROWTH_NUMERATOR / ALLOWED_GROWTH_DENOMINATOR`.
const ALLOWED_GROWTH_NUMERATOR: usize = 5;
const ALLOWED_GROWTH_DENOMINATOR: usize = 4;

pub(super) const COUNTER_NAMES: [&str; 4] = [
    "suffix_calls",
    "completed_states",
    "generated_candidates",
    "descriptor_builds",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PlannerCostRow {
    pub(super) family: String,
    pub(super) logical_key: String,
    pub(super) lookup_key_digest: String,
    pub(super) cost: PlannerCost,
}

impl PlannerCostRow {
    fn map_key(&self) -> (String, String) {
        (self.family.clone(), self.lookup_key_digest.clone())
    }

    const fn counters(&self) -> [usize; 4] {
        [
            self.cost.suffix_calls,
            self.cost.completed_states,
            self.cost.generated_candidates,
            self.cost.descriptor_builds,
        ]
    }
}

fn indexed_rows(
    rows: Vec<PlannerCostRow>,
    context: &str,
) -> Result<BTreeMap<(String, String), PlannerCostRow>, String> {
    let mut indexed = BTreeMap::new();
    for row in rows {
        let key = row.map_key();
        if indexed.insert(key.clone(), row).is_some() {
            return Err(format!(
                "{context}: duplicate planner cost row {} / {}",
                key.0, key.1
            ));
        }
    }
    Ok(indexed)
}

pub(super) fn write_snapshot(rows: Vec<PlannerCostRow>) -> Result<String, String> {
    let mut out = SNAPSHOT_HEADER.to_string();
    for row in indexed_rows(rows, "write planner cost snapshot")?.values() {
        let [suffix_calls, completed_states, generated_candidates, descriptor_builds] =
            row.counters();
        writeln!(
            out,
            "{}\t{}\t{}\t{suffix_calls}\t{completed_states}\t{generated_candidates}\t{descriptor_builds}",
            row.family, row.logical_key, row.lookup_key_digest,
        )
        .map_err(|error| format!("write planner cost snapshot: {error}"))?;
    }
    Ok(out)
}

pub(super) fn parse_snapshot(input: &str) -> Result<Vec<PlannerCostRow>, String> {
    let mut lines = input.lines();
    if lines.next().map(|header| format!("{header}\n")).as_deref() != Some(SNAPSHOT_HEADER) {
        return Err("planner cost snapshot has an unsupported header".to_string());
    }
    lines
        .enumerate()
        .map(|(offset, line)| {
            let line_number = offset + 2;
            let columns = line.split('\t').collect::<Vec<_>>();
            let [family, logical_key, lookup_key_digest, counters @ ..] = columns.as_slice() else {
                return Err(format!(
                    "planner cost snapshot line {line_number}: expected 7 TSV columns, got {}",
                    columns.len()
                ));
            };
            let counters = counters
                .iter()
                .map(|raw| {
                    raw.parse::<usize>().map_err(|error| {
                        format!(
                            "planner cost snapshot line {line_number}: invalid counter `{raw}`: {error}"
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let [suffix_calls, completed_states, generated_candidates, descriptor_builds] =
                counters.as_slice()
            else {
                return Err(format!(
                    "planner cost snapshot line {line_number}: expected 7 TSV columns, got {}",
                    columns.len()
                ));
            };
            Ok(PlannerCostRow {
                family: (*family).to_string(),
                logical_key: (*logical_key).to_string(),
                lookup_key_digest: (*lookup_key_digest).to_string(),
                cost: PlannerCost {
                    suffix_calls: *suffix_calls,
                    completed_states: *completed_states,
                    generated_candidates: *generated_candidates,
                    descriptor_builds: *descriptor_builds,
                },
            })
        })
        .collect()
}

pub(super) struct PlannerCostComparison {
    pub(super) report: String,
    pub(super) failures: usize,
    /// Counter sums in [`COUNTER_NAMES`] order.
    pub(super) baseline_totals: [usize; 4],
    pub(super) current_totals: [usize; 4],
}

fn add_counters(totals: &mut [usize; 4], row: &PlannerCostRow) -> Result<(), String> {
    for (total, value) in totals.iter_mut().zip(row.counters()) {
        *total = total
            .checked_add(value)
            .ok_or_else(|| "planner cost total overflow".to_string())?;
    }
    Ok(())
}

/// Compare the current per-row search work with a checked-in baseline.
///
/// Added and removed rows fail, as does any counter that grows by more than a
/// quarter. Either way the fix is to regenerate the baseline deliberately.
pub(super) fn compare_snapshots(
    baseline: Vec<PlannerCostRow>,
    current: Vec<PlannerCostRow>,
) -> Result<PlannerCostComparison, String> {
    let baseline = indexed_rows(baseline, "planner cost baseline")?;
    let current = indexed_rows(current, "current planner cost")?;
    let mut comparison = PlannerCostComparison {
        report: String::new(),
        failures: 0,
        baseline_totals: [0; 4],
        current_totals: [0; 4],
    };
    for row in baseline.values() {
        add_counters(&mut comparison.baseline_totals, row)?;
        if !current.contains_key(&row.map_key()) {
            comparison.failures += 1;
            writeln!(
                comparison.report,
                "{}\t{}\tremoved",
                row.family, row.logical_key
            )
            .map_err(|error| format!("write planner cost report: {error}"))?;
        }
    }
    for row in current.values() {
        add_counters(&mut comparison.current_totals, row)?;
        let Some(baseline_row) = baseline.get(&row.map_key()) else {
            comparison.failures += 1;
            writeln!(
                comparison.report,
                "{}\t{}\tadded",
                row.family, row.logical_key
            )
            .map_err(|error| format!("write planner cost report: {error}"))?;
            continue;
        };
        for ((name, current_value), baseline_value) in COUNTER_NAMES
            .into_iter()
            .zip(row.counters())
            .zip(baseline_row.counters())
        {
            let scaled_current = current_value
                .checked_mul(ALLOWED_GROWTH_DENOMINATOR)
                .ok_or_else(|| "planner cost comparison overflow".to_string())?;
            let allowed = baseline_value
                .checked_mul(ALLOWED_GROWTH_NUMERATOR)
                .ok_or_else(|| "planner cost comparison overflow".to_string())?;
            if scaled_current > allowed {
                comparison.failures += 1;
                writeln!(
                    comparison.report,
                    "{}\t{}\tgrew\t{name}\t{baseline_value}\t{current_value}",
                    row.family, row.logical_key,
                )
                .map_err(|error| format!("write planner cost report: {error}"))?;
            }
        }
    }
    Ok(comparison)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(digest: &str, suffix_calls: usize) -> PlannerCostRow {
        PlannerCostRow {
            family: "fp64_dense".to_string(),
            logical_key: "final=20:1;precommitted=".to_string(),
            lookup_key_digest: digest.to_string(),
            cost: PlannerCost {
                suffix_calls,
                completed_states: 10,
                generated_candidates: 100,
                descriptor_builds: 7,
            },
        }
    }

    #[test]
    fn snapshot_round_trips() {
        let rows = vec![row("bb", 3), row("aa", 4)];
        let parsed = parse_snapshot(&write_snapshot(rows.clone()).unwrap()).unwrap();
        assert_eq!(parsed, vec![row("aa", 4), row("bb", 3)]);
    }

    #[test]
    fn growth_beyond_a_quarter_fails_and_shrinking_passes() {
        let within = compare_snapshots(vec![row("aa", 100)], vec![row("aa", 125)]).unwrap();
        assert_eq!(within.failures, 0, "{}", within.report);
        let shrink = compare_snapshots(vec![row("aa", 100)], vec![row("aa", 1)]).unwrap();
        assert_eq!(shrink.failures, 0, "{}", shrink.report);
        let grew = compare_snapshots(vec![row("aa", 100)], vec![row("aa", 126)]).unwrap();
        assert_eq!(grew.failures, 1);
        assert!(grew.report.contains("grew\tsuffix_calls\t100\t126"));
    }

    #[test]
    fn row_set_changes_fail() {
        let changed = compare_snapshots(vec![row("aa", 1)], vec![row("bb", 1)]).unwrap();
        assert_eq!(changed.failures, 2);
        assert!(changed.report.contains("removed"));
        assert!(changed.report.contains("added"));
    }
}
