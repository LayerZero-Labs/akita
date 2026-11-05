//! Reproducible q24 width certificates and fixed-width quantum audit cells.

use crate::{
    error::{EstimatorError, Result},
    estimate, scalar_sis_from_ring,
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthOrigin,
        InfinityWidthRow, InfinityWidthTableConfig, INFINITY_WIDTH_EVALUATOR_ID,
    },
    AkitaModulusProfileId, CostValue, EstimateConfig,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Exact q24 commitment profile, available only with `labinius-sis`.
pub const PROFILE: AkitaModulusProfileId = AkitaModulusProfileId::Q24Minus2523807;
/// Source-comparison bounds for weights 46, 64 and 77 at diameter 4095.
pub const BOUNDS: [u64; 3] = [1_506_960, 2_096_640, 2_522_520];
/// Ring-column widths checked independently of the width-prefix certificate.
pub const WIDTHS: [u32; 3] = [4_096, 8_192, 16_384];
/// Positive-width q24 certificates. The q25 artifact is independent.
pub const CSV: &str = include_str!("../../data/q24_commitment_prime_infinity_width.csv");
/// Fixed-width certificate goldens, including rejected rank-two cells.
pub const CELLS_CSV: &str = include_str!("../../data/q24_infinity_cells.csv");

/// Generate q24 certificates using the unchanged q25 policy and cap.
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
