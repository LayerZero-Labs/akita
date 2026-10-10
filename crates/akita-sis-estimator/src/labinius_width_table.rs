//! Offline lookup for the LaBinius infinity-width audit artifact.
//!
//! This module is not a runtime admission path. It exposes the checked-in rows
//! of the commitment prime `2^25 + 14561` to benchmarks and to the generator
//! of the runtime cells in `akita-params`.

use std::sync::LazyLock;

use crate::{
    error::{EstimatorError, Result},
    width_table::{validate_infinity_width_rows, InfinityWidthRow, INFINITY_WIDTH_EVALUATOR_ID},
    AkitaModulusProfileId,
};

const ARTIFACT: &str = include_str!("../data/labinius_commitment_prime_infinity_width.csv");
const PREFIX_COLUMNS: usize = 4;

static ROWS: LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>> =
    LazyLock::new(|| parse_artifact(ARTIFACT));

/// Certified width for one binary-source SIS cell.
///
/// A cell holds either an exact cutoff or a search-cap lower bound, with no
/// claim about the next width. Returns `None` for an uncovered tuple;
/// zero-width candidates are absent.
pub fn certified_max_width(
    modulus_profile: AkitaModulusProfileId,
    ring_dimension: u32,
    rank: u32,
    coeff_linf_bound: u64,
) -> Result<Option<u64>> {
    Ok(certified_rows()?
        .iter()
        .find(|row| {
            row.modulus_profile == modulus_profile
                && row.d == ring_dimension
                && row.rank == rank
                && row.coeff_linf_bound == coeff_linf_bound
        })
        .map(|row| row.max_width))
}

/// All checked-in rows after parsing and certificate validation.
pub fn certified_rows() -> Result<&'static [InfinityWidthRow]> {
    match ROWS.as_ref() {
        Ok(rows) => Ok(rows),
        Err(reason) => Err(EstimatorError::InvalidConfig {
            field: "labinius_width_table",
            reason: reason.clone(),
        }),
    }
}

/// Parse the artifact. Every row is for the commitment prime and certifies a
/// positive width, as an exact cutoff or as a search-cap lower bound.
fn parse_artifact(artifact: &str) -> std::result::Result<Vec<InfinityWidthRow>, String> {
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
        if row.modulus_profile != AkitaModulusProfileId::Q25Plus14561 {
            return Err(format!("row {} has wrong commitment modulus", index + 2));
        }
        let certified_cap =
            row.hit_cap && row.max_width == row.search_cap && row.next_costs.is_none();
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
        format!("estimator_id,source_profile,scalar_degree,packing_degree,{}\n{INFINITY_WIDTH_EVALUATOR_ID},phi243-bounded-w46-delta12,162,4,{}\n", InfinityWidthRow::csv_header(), row.to_csv_record())
    }

    #[test]
    fn only_positive_width_certificates_of_the_commitment_prime_parse() {
        let row = certified_rows().unwrap()[0].clone();
        assert!(parse_artifact(&artifact_with(&row)).is_ok());
        assert_eq!(
            certified_max_width(row.modulus_profile, row.d, row.rank, row.coeff_linf_bound)
                .unwrap(),
            Some(row.max_width)
        );
        assert_eq!(
            certified_max_width(
                row.modulus_profile,
                row.d,
                row.rank,
                row.coeff_linf_bound + 1
            )
            .unwrap(),
            None
        );

        let mut other_modulus = row.clone();
        other_modulus.modulus_profile = AkitaModulusProfileId::Q128OffsetA7F7;
        let mut zero = row.clone();
        zero.max_width = 0;
        zero.hit_cap = false;
        let mut below_cap = row.clone();
        below_cap.max_width -= 1;
        let mut below_target = row;
        below_target.max_costs.as_mut().unwrap().adps16_quantum.rop =
            crate::cost::CostValue::finite_log2(127.0);
        for invalid in [other_modulus, zero, below_cap, below_target] {
            assert!(parse_artifact(&artifact_with(&invalid)).is_err());
        }
    }
}
