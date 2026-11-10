//! Reproducible q24 width certificates and fixed-width quantum audit cells for
//! the binary commitment ring.
//!
//! This module is offline evidence, not a runtime admission path: no catalog
//! or protocol reads it.

use std::sync::LazyLock;

use crate::{
    error::{EstimatorError, Result},
    estimate, scalar_sis_from_ring,
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthOrigin,
        InfinityWidthProfile, InfinityWidthRow, InfinityWidthTableConfig,
        INFINITY_WIDTH_EVALUATOR_ID,
    },
    AkitaModulusProfileId, CostValue, EstimateConfig,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Exact q24 commitment profile, available only with `binary`.
pub const PROFILE: AkitaModulusProfileId = AkitaModulusProfileId::Q24Minus2523807;
/// Source-comparison bounds for weights 46, 64 and 77 at diameter 4095.
pub const BOUNDS: [u64; 3] = [1_506_960, 2_096_640, 2_522_520];
/// Ring-column widths checked independently of the width-prefix certificate.
pub const WIDTHS: [u32; 3] = [4_096, 8_192, 16_384];
/// Positive-width q24 certificates.
pub const CSV: &str = include_str!("../data/q24_commitment_prime_infinity_width.csv");
/// Fixed-width certificate goldens, including rejected rank-two cells.
pub const CELLS_CSV: &str = include_str!("../data/q24_infinity_cells.csv");

/// Generate q24 certificates under the 169-bit quantum policy and search cap.
///
/// Rejected candidates stay out of the width artifact; their actual requested
/// widths are evaluated separately by [`generate_cells_csv`].
pub fn generate_width_csv() -> Result<String> {
    let config = InfinityWidthTableConfig {
        profiles: vec![PROFILE],
        ring_dims: vec![648],
        coeff_linf_bounds: BOUNDS.to_vec(),
        max_rank: 4,
        explicit_origins: Some(
            BOUNDS
                .into_iter()
                .map(|coeff_linf_bound| InfinityWidthOrigin {
                    modulus_profile: PROFILE,
                    d: 648,
                    coeff_linf_bound,
                })
                .collect(),
        ),
        ..InfinityWidthTableConfig::default()
    };
    let rows = generate_infinity_width_rows(&config)?;
    validate_infinity_width_rows(&rows)?;
    let mut output = format!(
        "estimator_id,source_profile,scalar_degree,packing_degree,{}\n",
        InfinityWidthRow::csv_header()
    );
    for row in rows.iter().filter(|row| row.rank >= 2 && row.max_width > 0) {
        output.push_str(&format!(
            "{INFINITY_WIDTH_EVALUATOR_ID},binary-source-delta4095,162,4,{}\n",
            row.to_csv_record()
        ));
    }
    Ok(output)
}

/// Evaluate every requested cell with proven-pruned certification, never with
/// an optimizer probe. Above-target values are lower bounds, not optima.
pub fn generate_cells_csv() -> Result<String> {
    let cells: Vec<_> = (2..=4)
        .flat_map(|rank| {
            BOUNDS
                .into_iter()
                .flat_map(move |eta| WIDTHS.into_iter().map(move |width| (rank, eta, width)))
        })
        .collect();
    #[cfg(feature = "parallel")]
    let records: Result<Vec<_>> = cells
        .par_iter()
        .map(|&(rank, eta, width)| cell_record(rank, eta, width))
        .collect();
    #[cfg(not(feature = "parallel"))]
    let records: Result<Vec<_>> = cells
        .iter()
        .map(|&(rank, eta, width)| cell_record(rank, eta, width))
        .collect();
    let mut output = String::from("estimator_id,profile,d,rank,width,eta,scalar_n,scalar_m,certificate,bits,beta,zeta,clears_128\n");
    for record in records? {
        output.push_str(&record);
    }
    Ok(output)
}

fn cell_record(rank: u32, eta: u64, width: u32) -> Result<String> {
    let params = scalar_sis_from_ring(PROFILE, 648, rank, width, eta)?;
    let cost = estimate(&params, &EstimateConfig::akita_infinity_table())?;
    let (status, bits) = match cost.rop {
        CostValue::Finite(value) if value.log2.is_finite() => ("finite", value.log2),
        CostValue::ProvenAboveTarget(value) if value.log2.is_finite() => {
            ("above-target", value.log2)
        }
        _ => {
            return Err(EstimatorError::Unsupported {
                feature: "non-finite q24 certificate",
            })
        }
    };
    eprintln!(
        "q24 rank={rank} width={width} eta={eta}: {status}:{bits} beta={:?}",
        cost.beta
    );
    Ok(format!("{INFINITY_WIDTH_EVALUATOR_ID},q24,648,{rank},{width},{eta},{},{},{status},{bits},{},{},{}\n",
        params.n, params.m.unwrap_or_default(),
        cost.beta.map_or_else(String::new, |v| v.to_string()),
        cost.zeta.map_or_else(String::new, |v| v.to_string()), bits >= 128.0))
}

const PREFIX_COLUMNS: usize = 4;

static ROWS: LazyLock<std::result::Result<Vec<InfinityWidthRow>, String>> =
    LazyLock::new(|| parse_artifact(CSV));

/// Validated q24 certificates. This offline interface does not select a
/// runtime profile or a catalog.
pub fn certified_rows() -> Result<&'static [InfinityWidthRow]> {
    ROWS.as_ref()
        .map(Vec::as_slice)
        .map_err(|reason| EstimatorError::InvalidConfig {
            field: "q24_width_table",
            reason: reason.clone(),
        })
}

/// Certified q24 width for one cell.
///
/// A cell holds either an exact cutoff or a search-cap lower bound, with no
/// claim about the next width. Returns `None` for an uncovered tuple;
/// zero-width candidates are absent.
pub fn certified_max_width(
    ring_dimension: u32,
    rank: u32,
    coeff_linf_bound: u64,
) -> Result<Option<u64>> {
    Ok(certified_rows()?
        .iter()
        .find(|row| {
            row.d == ring_dimension && row.rank == rank && row.coeff_linf_bound == coeff_linf_bound
        })
        .map(|row| row.max_width))
}

/// Parse the artifact. Every row is for the q24 commitment prime and certifies
/// a positive width, as an exact cutoff or as a search-cap lower bound.
fn parse_artifact(artifact: &str) -> std::result::Result<Vec<InfinityWidthRow>, String> {
    let prefix = format!("{INFINITY_WIDTH_EVALUATOR_ID},binary-source-delta4095,162,4,");
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
        if !line.starts_with(&prefix) {
            return Err(format!("row {} has wrong q24 source identity", index + 2));
        }
        let record = line
            .splitn(PREFIX_COLUMNS + 1, ',')
            .nth(PREFIX_COLUMNS)
            .ok_or_else(|| format!("row {} is missing estimator columns", index + 2))?;
        let row = InfinityWidthRow::from_csv_record(record)
            .map_err(|error| format!("row {} failed to parse: {error}", index + 2))?;
        if row.modulus_profile != PROFILE || row.profile != InfinityWidthProfile::LocalMinimum {
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
        format!(
            "estimator_id,source_profile,scalar_degree,packing_degree,{}\n{INFINITY_WIDTH_EVALUATOR_ID},binary-source-delta4095,162,4,{}\n",
            InfinityWidthRow::csv_header(),
            row.to_csv_record()
        )
    }

    #[test]
    fn parser_rejects_wrong_identity_and_uncertified_profiles() {
        assert!(parse_artifact(CSV).is_ok());
        for (from, to) in [
            ("binary-source-delta4095", "wrong-source"),
            (",162,4,", ",486,4,"),
            (",q24,", ",q128,"),
            (
                "local-minimum+proven-pruned-certification",
                "lattice-estimator-local-minimum",
            ),
            ("above-target:169.07000000000002", "127.0"),
        ] {
            assert!(parse_artifact(&CSV.replace(from, to)).is_err());
        }
    }

    #[test]
    fn only_positive_width_certificates_parse() {
        let row = certified_rows().unwrap()[0].clone();
        assert!(parse_artifact(&artifact_with(&row)).is_ok());
        assert_eq!(
            certified_max_width(row.d, row.rank, row.coeff_linf_bound).unwrap(),
            Some(row.max_width)
        );
        assert_eq!(
            certified_max_width(row.d, row.rank, row.coeff_linf_bound + 1).unwrap(),
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
