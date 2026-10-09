//! Offline lookup for the staged LaBinius infinity-width audit artifact.
//!
//! This module is not a runtime admission path. It exposes the checked-in rows
//! to benchmarks and follow-on protocol work while binary schedule support is
//! intentionally absent.

use std::sync::LazyLock;

use crate::{
    error::{EstimatorError, Result},
    width_table::{validate_infinity_width_rows, InfinityWidthRow, INFINITY_WIDTH_EVALUATOR_ID},
    AkitaModulusProfileId,
};

const ARTIFACT: &str = include_str!("../data/labinius_infinity_width.csv");
const PREFIX_COLUMNS: usize = 4;

static ROWS: LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>> =
    LazyLock::new(|| parse_artifact(ARTIFACT, false));
static SMALL_MODULUS_ROWS: LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>> =
    LazyLock::new(|| {
        parse_artifact(
            include_str!("../data/labinius_small_modulus_infinity_width.csv"),
            true,
        )
    });

/// Certified width for one staged binary-source SIS cell.
///
/// The shared-prime artifact provides exact cutoffs. The small-modulus artifact
/// also provides search-cap lower bounds, with no claim about the next width.
/// Returns `None` for an uncovered tuple; zero-width candidates are absent.
pub fn certified_max_width(
    modulus_profile: AkitaModulusProfileId,
    ring_dimension: u32,
    rank: u32,
    coeff_linf_bound: u64,
) -> Result<Option<u64>> {
    let rows = if modulus_profile == AkitaModulusProfileId::Q28Offset2103 {
        certified_small_modulus_rows()?
    } else {
        certified_rows()?
    };
    Ok(rows
        .iter()
        .find(|row| {
            row.modulus_profile == modulus_profile
                && row.d == ring_dimension
                && row.rank == rank
                && row.coeff_linf_bound == coeff_linf_bound
        })
        .map(|row| row.max_width))
}

/// All exact staged rows after parsing and certificate validation.
pub fn certified_rows() -> Result<&'static [InfinityWidthRow]> {
    rows(&ROWS)
}

/// Small-commitment-modulus rows with exact cutoffs or certified search-cap lower bounds.
pub fn certified_small_modulus_rows() -> Result<&'static [InfinityWidthRow]> {
    rows(&SMALL_MODULUS_ROWS)
}

fn rows(
    artifact: &'static LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>>,
) -> Result<&'static [InfinityWidthRow]> {
    match artifact.as_ref() {
        Ok(rows) => Ok(rows),
        Err(reason) => Err(EstimatorError::InvalidConfig {
            field: "labinius_width_table",
            reason: reason.clone(),
        }),
    }
}

fn parse_artifact(
    artifact: &str,
    allow_search_cap: bool,
) -> std::result::Result<Vec<InfinityWidthRow>, String> {
    let mut lines = artifact.lines();
    let expected_header = format!(
        "estimator_id,source_profile,scalar_degree,packing_degree,{}",
        InfinityWidthRow::csv_header()
    );
    if lines.next() != Some(expected_header.as_str()) {
        return Err("unexpected artifact header".into());
    }
    let mut rows = Vec::new();
    for (index, line) in lines.enumerate() {
        let mut fields = line.splitn(PREFIX_COLUMNS + 1, ',');
        if fields.next() != Some(INFINITY_WIDTH_EVALUATOR_ID) {
            return Err(format!("row {} has wrong evaluator identity", index + 2));
        }
        for _ in 1..PREFIX_COLUMNS {
            fields
                .next()
                .ok_or_else(|| format!("row {} is missing identity columns", index + 2))?;
        }
        let record = fields
            .next()
            .ok_or_else(|| format!("row {} is missing estimator columns", index + 2))?;
        let row = InfinityWidthRow::from_csv_record(record)
            .map_err(|error| format!("row {} failed to parse: {error}", index + 2))?;
        if allow_search_cap && row.modulus_profile != AkitaModulusProfileId::Q28Offset2103 {
            return Err(format!("row {} has wrong commitment modulus", index + 2));
        }
        let certified_cap = allow_search_cap
            && row.hit_cap
            && row.max_width == row.search_cap
            && row.next_costs.is_none();
        let exact = !row.hit_cap && row.next_costs.is_some();
        if row.max_width == 0 || row.max_costs.is_none() || !(exact || certified_cap) {
            return Err(format!(
                "row {} has no admissible positive-width certificate",
                index + 2
            ));
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("artifact contains no certified rows".into());
    }
    validate_infinity_width_rows(&rows)
        .map_err(|error| format!("artifact certificate validation failed: {error}"))?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact_with(row: &InfinityWidthRow) -> String {
        format!("estimator_id,source_profile,scalar_degree,packing_degree,{}\n{INFINITY_WIDTH_EVALUATOR_ID},phi243-bounded-w46-delta16,162,4,{}\n", InfinityWidthRow::csv_header(), row.to_csv_record())
    }

    #[test]
    fn cap_certificate_is_only_admitted_by_the_small_modulus_path() {
        let row = certified_small_modulus_rows().unwrap()[0].clone();
        let artifact = artifact_with(&row);
        assert!(parse_artifact(&artifact, false).is_err());
        assert!(parse_artifact(&artifact, true).is_ok());

        let mut zero = row.clone();
        zero.max_width = 0;
        zero.hit_cap = false;
        let mut below_cap = row.clone();
        below_cap.max_width -= 1;
        let mut below_target = row;
        below_target.max_costs.as_mut().unwrap().adps16_quantum.rop =
            crate::cost::CostValue::finite_log2(127.0);
        for invalid in [zero, below_cap, below_target] {
            let artifact = artifact_with(&invalid);
            assert!(parse_artifact(&artifact, false).is_err());
            assert!(parse_artifact(&artifact, true).is_err());
        }
    }

    #[test]
    fn exact_lookup_rejects_missing_successors_and_unknown_cells() {
        let rows = certified_rows().unwrap();
        assert!(!rows.is_empty());
        let cell = &rows[0];
        assert_eq!(
            certified_max_width(
                cell.modulus_profile,
                cell.d,
                cell.rank,
                cell.coeff_linf_bound
            )
            .unwrap(),
            Some(cell.max_width)
        );
        assert_eq!(
            certified_max_width(
                AkitaModulusProfileId::Q128OffsetA7F7,
                1_944,
                1,
                858_993_459_000
            )
            .unwrap(),
            None
        );
    }
}
