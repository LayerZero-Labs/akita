//! Offline lookup for the LaBinius infinity-width audit artifact.
//!
//! This module is not a runtime admission path. It exposes the checked-in rows
//! of the q25 and q24 commitment primes to benchmarks and to the generator
//! of the runtime cells in `akita-params`.

pub mod q24;

use std::sync::LazyLock;

use crate::{
    error::{EstimatorError, Result},
    width_table::{
        validate_infinity_width_rows, InfinityWidthProfile, InfinityWidthRow,
        INFINITY_WIDTH_EVALUATOR_ID,
    },
    AkitaModulusProfileId,
};

const ARTIFACT: &str = include_str!("../data/labinius_commitment_prime_infinity_width.csv");
const PREFIX_COLUMNS: usize = 4;

static ROWS: LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>> =
    LazyLock::new(|| parse_artifact(ARTIFACT, AkitaModulusProfileId::Q25Plus14561));

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
    let rows = if modulus_profile == AkitaModulusProfileId::Q24Minus2523807 {
        q24_certified_rows()?
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

static Q24_ROWS: LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>> =
    LazyLock::new(|| parse_artifact(q24::CSV, q24::PROFILE));

/// Validated q24 certificates under the same 169-bit policy as q25.
/// This offline interface does not select a runtime profile or a catalog.
pub fn q24_certified_rows() -> Result<&'static [InfinityWidthRow]> {
    Q24_ROWS
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|reason| EstimatorError::InvalidConfig {
            field: "q24_width_table",
            reason: reason.clone(),
        })
}

/// Parse the artifact. Every row is for the commitment prime and certifies a
/// positive width, as an exact cutoff or as a search-cap lower bound.
fn parse_artifact(
    artifact: &str,
    profile: AkitaModulusProfileId,
) -> std::result::Result<Vec<InfinityWidthRow>, String> {
    let q24_prefix = format!("{INFINITY_WIDTH_EVALUATOR_ID},binary-source-delta4095,162,4,");
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
        if profile == q24::PROFILE && !line.starts_with(&q24_prefix) {
            return Err(format!("row {} has wrong q24 source identity", index + 2));
        }
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
        if row.modulus_profile != profile || row.profile != InfinityWidthProfile::LocalMinimum {
            return Err(format!(
                "row {} has wrong commitment modulus or certification profile",
                index + 2
            ));
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
    fn q24_parser_rejects_wrong_identity_and_uncertified_profiles() {
        assert!(parse_artifact(q24::CSV, q24::PROFILE).is_ok());
        for (from, to) in [
            ("binary-source-delta4095", "wrong-source"),
            (",162,4,", ",486,4,"),
            (",q24,", ",q25-labinius,"),
            (
                "local-minimum+proven-pruned-certification",
                "lattice-estimator-local-minimum",
            ),
            ("above-target:169.07000000000002", "127.0"),
        ] {
            assert!(parse_artifact(&q24::CSV.replace(from, to), q24::PROFILE).is_err());
        }
    }

    #[test]
    fn only_positive_width_certificates_of_the_commitment_prime_parse() {
        let row = certified_rows().unwrap()[0].clone();
        assert!(parse_artifact(&artifact_with(&row), AkitaModulusProfileId::Q25Plus14561).is_ok());
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
            assert!(parse_artifact(
                &artifact_with(&invalid),
                AkitaModulusProfileId::Q25Plus14561
            )
            .is_err());
        }
    }
}
