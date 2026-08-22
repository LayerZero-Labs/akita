//! Stable schedule-catalog snapshots and revision comparisons.

use std::collections::BTreeMap;

pub(super) const SNAPSHOT_HEADER: &str = "family\tlogical_key\tlookup_key_digest\tsetup_fields\tfirst_direct_setup_capacity\tproof_bytes\tfold_levels\trow_digest\tpolicy\n";

const LEGACY_SNAPSHOT_HEADER: &str = "family\tlogical_key\tlookup_key_digest\tsetup_fields\tproof_bytes\tfold_levels\trow_digest\tpolicy\n";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SnapshotSchema {
    Legacy,
    Current,
}

pub(super) const COMPARISON_HEADER: &str = "family\tstatus\tlogical_key\tbaseline_lookup_key_digest\tcurrent_lookup_key_digest\tbaseline_setup_fields\tcurrent_setup_fields\tbaseline_first_direct_setup_capacity\tcurrent_first_direct_setup_capacity\tbaseline_proof_bytes\tcurrent_proof_bytes\tbaseline_levels\tcurrent_levels\tbaseline_row_digest\tcurrent_row_digest\tbaseline_policy\tcurrent_policy\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CatalogSnapshotRow {
    pub schema: SnapshotSchema,
    pub family: String,
    pub logical_key: String,
    pub lookup_key_digest: String,
    pub setup_fields: usize,
    pub first_direct_setup_capacity: Option<usize>,
    pub proof_bytes: usize,
    pub fold_levels: usize,
    pub row_digest: String,
    pub policy: String,
}

impl CatalogSnapshotRow {
    fn map_key(&self) -> (String, String) {
        (self.family.clone(), self.logical_key.clone())
    }

    fn validate_text(&self) -> Result<(), String> {
        for (label, value) in [
            ("family", self.family.as_str()),
            ("logical key", self.logical_key.as_str()),
            ("lookup key digest", self.lookup_key_digest.as_str()),
            ("row digest", self.row_digest.as_str()),
            ("policy", self.policy.as_str()),
        ] {
            if value.is_empty() || value.contains(['\t', '\n', '\r']) {
                return Err(format!(
                    "catalog snapshot {label} must be nonempty single-line TSV text"
                ));
            }
        }
        Ok(())
    }

    fn same_catalog_revision(&self, other: &Self) -> bool {
        self.family == other.family
            && self.logical_key == other.logical_key
            && self.lookup_key_digest == other.lookup_key_digest
            && self.setup_fields == other.setup_fields
            && self.proof_bytes == other.proof_bytes
            && self.fold_levels == other.fold_levels
            && self.row_digest == other.row_digest
            && self.policy == other.policy
            && self.first_direct_setup_capacity == other.first_direct_setup_capacity
    }

    fn write_snapshot(&self, out: &mut String) {
        use std::fmt::Write as _;
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.family,
            self.logical_key,
            self.lookup_key_digest,
            self.setup_fields,
            self.first_direct_setup_capacity
                .map_or_else(|| "-".to_string(), |value| value.to_string()),
            self.proof_bytes,
            self.fold_levels,
            self.row_digest,
            self.policy,
        )
        .expect("writing to String cannot fail");
    }
}

fn indexed_rows(
    rows: Vec<CatalogSnapshotRow>,
    context: &str,
) -> Result<BTreeMap<(String, String), CatalogSnapshotRow>, String> {
    let mut indexed = BTreeMap::new();
    for row in rows {
        row.validate_text()?;
        let key = row.map_key();
        if indexed.insert(key.clone(), row).is_some() {
            return Err(format!(
                "{context}: duplicate catalog logical key {} / {}",
                key.0, key.1
            ));
        }
    }
    Ok(indexed)
}

pub(super) fn write_snapshot(rows: Vec<CatalogSnapshotRow>) -> Result<String, String> {
    let indexed = indexed_rows(rows, "write snapshot")?;
    let mut out = SNAPSHOT_HEADER.to_string();
    for row in indexed.values() {
        row.write_snapshot(&mut out);
    }
    Ok(out)
}

fn parse_usize(raw: &str, line: usize, field: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|error| format!("catalog snapshot line {line}: invalid {field} `{raw}`: {error}"))
}

pub(super) fn parse_snapshot(input: &str) -> Result<Vec<CatalogSnapshotRow>, String> {
    let mut lines = input.lines();
    let header = lines
        .next()
        .ok_or_else(|| "catalog snapshot is empty".to_string())?;
    let legacy = format!("{header}\n") == LEGACY_SNAPSHOT_HEADER;
    if !legacy && format!("{header}\n") != SNAPSHOT_HEADER {
        return Err("catalog snapshot has an unsupported header".to_string());
    }
    let mut rows = Vec::new();
    for (offset, line) in lines.enumerate() {
        let line_number = offset + 2;
        if line.is_empty() {
            return Err(format!(
                "catalog snapshot line {line_number}: empty rows are not allowed"
            ));
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        let (
            family,
            logical_key,
            lookup_key_digest,
            setup_fields,
            first_direct,
            proof_bytes,
            fold_levels,
            row_digest,
            policy,
        ) = if legacy {
            let [family, logical_key, lookup_key_digest, setup_fields, proof_bytes, fold_levels, row_digest, policy] =
                columns.as_slice()
            else {
                return Err(format!(
                    "catalog snapshot line {line_number}: expected 8 TSV columns, got {}",
                    columns.len()
                ));
            };
            (
                *family,
                *logical_key,
                *lookup_key_digest,
                *setup_fields,
                None,
                *proof_bytes,
                *fold_levels,
                *row_digest,
                *policy,
            )
        } else {
            let [family, logical_key, lookup_key_digest, setup_fields, first_direct, proof_bytes, fold_levels, row_digest, policy] =
                columns.as_slice()
            else {
                return Err(format!(
                    "catalog snapshot line {line_number}: expected 9 TSV columns, got {}",
                    columns.len()
                ));
            };
            (
                *family,
                *logical_key,
                *lookup_key_digest,
                *setup_fields,
                Some(*first_direct),
                *proof_bytes,
                *fold_levels,
                *row_digest,
                *policy,
            )
        };
        rows.push(CatalogSnapshotRow {
            schema: if legacy {
                SnapshotSchema::Legacy
            } else {
                SnapshotSchema::Current
            },
            family: family.to_string(),
            logical_key: logical_key.to_string(),
            lookup_key_digest: lookup_key_digest.to_string(),
            setup_fields: parse_usize(setup_fields, line_number, "setup_fields")?,
            first_direct_setup_capacity: first_direct
                .filter(|value| *value != "-")
                .map(|value| parse_usize(value, line_number, "first_direct_setup_capacity"))
                .transpose()?,
            proof_bytes: parse_usize(proof_bytes, line_number, "proof_bytes")?,
            fold_levels: parse_usize(fold_levels, line_number, "fold_levels")?,
            row_digest: row_digest.to_string(),
            policy: policy.to_string(),
        });
    }
    let indexed = indexed_rows(rows, "parse snapshot")?;
    Ok(indexed.into_values().collect())
}

fn optional_text<'a>(
    row: Option<&'a CatalogSnapshotRow>,
    field: fn(&'a CatalogSnapshotRow) -> &'a str,
) -> &'a str {
    row.map_or("-", field)
}

fn optional_usize(
    row: Option<&CatalogSnapshotRow>,
    field: fn(&CatalogSnapshotRow) -> usize,
) -> String {
    row.map(field)
        .map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn optional_nested_usize(
    row: Option<&CatalogSnapshotRow>,
    field: fn(&CatalogSnapshotRow) -> Option<usize>,
) -> String {
    row.and_then(field)
        .map_or_else(|| "-".to_string(), |value| value.to_string())
}

pub(super) struct CatalogRevisionComparison {
    pub report: String,
    pub added_rows: usize,
    pub removed_rows: usize,
    pub changed_rows: usize,
    pub equal_rows: usize,
}

pub(super) fn compare_snapshots(
    baseline: Vec<CatalogSnapshotRow>,
    current: Vec<CatalogSnapshotRow>,
) -> Result<CatalogRevisionComparison, String> {
    let mut baseline = indexed_rows(baseline, "baseline snapshot")?;
    let mut current = indexed_rows(current, "current snapshot")?;
    let mut keys = baseline
        .keys()
        .chain(current.keys())
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    keys.dedup();

    let mut report = COMPARISON_HEADER.to_string();
    let mut comparison = CatalogRevisionComparison {
        report: String::new(),
        added_rows: 0,
        removed_rows: 0,
        changed_rows: 0,
        equal_rows: 0,
    };
    for key in keys {
        let old = baseline.remove(&key);
        let new = current.remove(&key);
        let status = match (&old, &new) {
            (None, Some(_)) => {
                comparison.added_rows += 1;
                "added"
            }
            (Some(_), None) => {
                comparison.removed_rows += 1;
                "removed"
            }
            (Some(old), Some(new)) if old.same_catalog_revision(new) => {
                comparison.equal_rows += 1;
                "equal"
            }
            (Some(_), Some(_)) => {
                comparison.changed_rows += 1;
                "changed"
            }
            (None, None) => unreachable!("comparison key came from either snapshot"),
        };
        let old = old.as_ref();
        let new = new.as_ref();
        use std::fmt::Write as _;
        writeln!(
            report,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            key.0,
            status,
            key.1,
            optional_text(old, |row| row.lookup_key_digest.as_str()),
            optional_text(new, |row| row.lookup_key_digest.as_str()),
            optional_usize(old, |row| row.setup_fields),
            optional_usize(new, |row| row.setup_fields),
            optional_nested_usize(old, |row| row.first_direct_setup_capacity),
            optional_nested_usize(new, |row| row.first_direct_setup_capacity),
            optional_usize(old, |row| row.proof_bytes),
            optional_usize(new, |row| row.proof_bytes),
            optional_usize(old, |row| row.fold_levels),
            optional_usize(new, |row| row.fold_levels),
            optional_text(old, |row| row.row_digest.as_str()),
            optional_text(new, |row| row.row_digest.as_str()),
            optional_text(old, |row| row.policy.as_str()),
            optional_text(new, |row| row.policy.as_str()),
        )
        .expect("writing to String cannot fail");
    }
    comparison.report = report;
    Ok(comparison)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(family: &str, key: &str, proof_bytes: usize) -> CatalogSnapshotRow {
        CatalogSnapshotRow {
            schema: SnapshotSchema::Current,
            family: family.into(),
            logical_key: key.into(),
            lookup_key_digest: format!("key-{proof_bytes}"),
            setup_fields: 100,
            first_direct_setup_capacity: Some(64),
            proof_bytes,
            fold_levels: 2,
            row_digest: format!("row-{proof_bytes}"),
            policy: format!("policy-{proof_bytes}"),
        }
    }

    #[test]
    fn snapshot_round_trip_is_sorted_and_checked() {
        let encoded =
            write_snapshot(vec![row("b", "2", 2), row("a", "1", 1)]).expect("write snapshot");
        assert!(encoded.find("a\t1").unwrap() < encoded.find("b\t2").unwrap());
        let decoded = parse_snapshot(&encoded).expect("parse snapshot");
        assert_eq!(decoded, vec![row("a", "1", 1), row("b", "2", 2)]);

        let duplicate =
            format!("{SNAPSHOT_HEADER}a\t1\tk\t1\t1\t1\t1\tr\tp\na\t1\tk\t1\t1\t1\t1\tr\tp\n");
        assert!(parse_snapshot(&duplicate)
            .unwrap_err()
            .contains("duplicate"));

        let legacy = format!("{LEGACY_SNAPSHOT_HEADER}a\t1\tk\t1\t2\t3\tr\tp\n");
        let legacy = parse_snapshot(&legacy).expect("parse legacy snapshot");
        assert_eq!(legacy[0].first_direct_setup_capacity, None);
    }

    #[test]
    fn revision_comparison_reports_complete_logical_key_union() {
        let baseline = vec![
            row("family", "equal", 1),
            row("family", "removed", 2),
            row("family", "changed", 3),
        ];
        let current = vec![
            row("family", "equal", 1),
            row("family", "added", 4),
            row("family", "changed", 5),
        ];
        let comparison = compare_snapshots(baseline, current).expect("compare snapshots");
        assert_eq!(comparison.added_rows, 1);
        assert_eq!(comparison.removed_rows, 1);
        assert_eq!(comparison.changed_rows, 1);
        assert_eq!(comparison.equal_rows, 1);
        assert!(comparison.report.starts_with(COMPARISON_HEADER));
        assert_eq!(comparison.report.lines().count(), 5);
        assert!(comparison
            .report
            .lines()
            .skip(1)
            .all(|line| line.split('\t').count() == 17));
    }

    #[test]
    fn legacy_missing_capacity_is_reported_as_catalog_drift() {
        let current = row("family", "same", 1);
        let mut legacy = current.clone();
        legacy.schema = SnapshotSchema::Legacy;
        legacy.first_direct_setup_capacity = None;
        let comparison = compare_snapshots(vec![legacy], vec![current]).expect("compare snapshots");
        assert_eq!(comparison.equal_rows, 0);
        assert_eq!(comparison.changed_rows, 1);
    }

    #[test]
    fn current_missing_capacity_is_catalog_drift() {
        let baseline = row("family", "same", 1);
        let mut current = baseline.clone();
        current.first_direct_setup_capacity = None;
        let comparison =
            compare_snapshots(vec![baseline], vec![current]).expect("compare current snapshots");
        assert_eq!(comparison.equal_rows, 0);
        assert_eq!(comparison.changed_rows, 1);
    }

    #[test]
    fn checked_catalog_evidence_matches_base_and_head() {
        let base = parse_snapshot(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../specs/evidence/subring-coefficient-packing/base.tsv"
        )))
        .expect("parse checked base snapshot");
        let head = parse_snapshot(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../specs/evidence/subring-coefficient-packing/head.tsv"
        )))
        .expect("parse checked head snapshot");
        let checked_comparison = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../specs/evidence/subring-coefficient-packing/comparison.tsv"
        ));

        assert_eq!(base.len(), 68);
        assert_eq!(head.len(), 81);
        assert!(base
            .iter()
            .all(|row| row.first_direct_setup_capacity.is_some()));
        assert!(head
            .iter()
            .all(|row| row.first_direct_setup_capacity.is_some()));
        let comparison = compare_snapshots(base, head).expect("compare checked snapshots");
        assert_eq!(comparison.report, checked_comparison);
        assert_eq!(comparison.added_rows, 14);
        assert_eq!(comparison.removed_rows, 1);
        assert_eq!(comparison.changed_rows, 67);
        assert_eq!(comparison.equal_rows, 0);
        let removed = comparison
            .report
            .lines()
            .find(|line| line.contains("\tremoved\t"))
            .expect("one removed evidence row");
        assert!(removed.starts_with("fp128_dense\tremoved\tfinal=44:1;precommitted=\t"));
    }
}
