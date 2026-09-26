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
    LazyLock::new(parse_artifact);

/// Exact certified cutoff for one staged binary-source SIS cell.
///
/// Returns `None` when the checked-in artifact has no exact cell for the tuple.
/// Cap-limited and zero-width candidates are deliberately absent.
pub fn certified_max_width(
    modulus_profile: AkitaModulusProfileId,
    ring_dimension: u32,
    rank: u32,
    coeff_linf_bound: u64,
) -> Result<Option<u64>> {
    let rows = rows()?;
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
    rows()
}

fn rows() -> Result<&'static [InfinityWidthRow]> {
    match ROWS.as_ref() {
        Ok(rows) => Ok(rows),
        Err(reason) => Err(EstimatorError::InvalidConfig {
            field: "labinius_width_table",
            reason: reason.clone(),
        }),
    }
}

fn parse_artifact() -> std::result::Result<Vec<InfinityWidthRow>, String> {
    let mut lines = ARTIFACT.lines();
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
        if row.max_width == 0 || row.hit_cap || row.max_costs.is_none() || row.next_costs.is_none()
        {
            return Err(format!(
                "row {} is not an exact positive cutoff with a rejected successor",
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
