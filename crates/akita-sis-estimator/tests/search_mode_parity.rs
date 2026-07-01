//! Compare Python-local-minimum and exhaustive search on a medium golden subset.
//!
//! Full-grid exhaustive search is excluded from CI: for large cells `m = width * d`
//! can reach tens of thousands, making exhaustive zeta search prohibitively slow.
//!
//! Coverage policy:
//! - All trusted cells with `m <= 512` (83 cells: every family, rank, bound).
//! - For `512 < m <= 1024`, one representative bound per geometry (`255`).
//!
//! This yields ~89 cells covering all 3 families, ranks 1/5, ring dims
//! 32–256, and all coeff-bound buckets on the fast tier.

use akita_sis_estimator::{
    estimate, scalar_sis_from_ring, AkitaModulusFamily, CostValue, EstimateConfig, NumericConfig,
};

const GOLDEN_CSV: &str = include_str!("../../../scripts/sis_golden/infinity_golden.csv");

/// Fast tier: include every coeff bound when column count is at most this.
const EXHAUSTIVE_FAST_MAX_M: u32 = 512;

/// Slow tier: include one representative bound per geometry up to this column count.
const EXHAUSTIVE_SLOW_MAX_M: u32 = 1024;

/// Representative coeff bound for slow-tier geometries (middle of the golden ladder).
const EXHAUSTIVE_SLOW_REPRESENTATIVE_BOUND: u64 = 255;

fn local_minimum_config() -> EstimateConfig {
    EstimateConfig::lattice_estimator_parity()
}

fn exhaustive_config() -> EstimateConfig {
    EstimateConfig::akita_infinity_table()
}

#[cfg(feature = "parallel")]
fn parallel_exhaustive_config() -> EstimateConfig {
    EstimateConfig {
        optimizer: akita_sis_estimator::OptimizerConfig::OptimizeZeta {
            beta: akita_sis_estimator::SearchMode::ExhaustiveParallel,
            zeta: akita_sis_estimator::SearchMode::ExhaustiveParallel,
        },
        ..EstimateConfig::default()
    }
}

#[test]
fn exhaustive_search_covers_medium_trusted_grid() {
    let rows = exhaustive_subset_rows();
    assert!(
        rows.len() >= 85,
        "expected at least 85 medium-grid cells, got {}",
        rows.len()
    );
    let families: std::collections::HashSet<_> = rows.iter().map(|row| row.family).collect();
    assert_eq!(families.len(), 3, "expected all three modulus families");
    let ranks: std::collections::HashSet<_> = rows.iter().map(|row| row.rank).collect();
    assert!(ranks.contains(&5), "expected rank-5 coverage");
}

#[test]
fn exhaustive_search_is_at_least_as_good_as_local_minimum_on_medium_subset() {
    let tol = NumericConfig::default().sage_abs_tolerance;
    let mut violations = Vec::new();
    for row in exhaustive_subset_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let local = estimate(&params, &local_minimum_config()).unwrap();
        let exhaustive = estimate(&params, &exhaustive_config()).unwrap();
        if !exhaustive_at_least_as_good(&exhaustive, &local, tol) {
            violations.push(format!(
                "exhaustive worse than local-minimum for {row:?}\n  local={local:?}\n  exhaustive={exhaustive:?}"
            ));
        }
    }
    if !violations.is_empty() {
        panic!("{}", violations.join("\n\n"));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_exhaustive_matches_serial_exhaustive_on_medium_subset() {
    let mut mismatches = Vec::new();
    for row in exhaustive_subset_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let serial = estimate(&params, &exhaustive_config()).unwrap();
        let parallel = estimate(&params, &parallel_exhaustive_config()).unwrap();
        if serial != parallel {
            mismatches.push(format!(
                "parallel exhaustive mismatch for {row:?}\n  serial={serial:?}\n  parallel={parallel:?}"
            ));
        }
    }
    if !mismatches.is_empty() {
        panic!("{}", mismatches.join("\n\n"));
    }
}

/// Rank-20 geometries have `m = 1280` and are too slow for default CI.
/// Run with `cargo test -p akita-sis-estimator --test search_mode_parity -- --ignored`.
#[test]
#[ignore = "rank-20 exhaustive cells are slow; run manually before table generation changes"]
fn exhaustive_search_rank20_geometries_manual() {
    let tol = NumericConfig::default().sage_abs_tolerance;
    let rows: Vec<_> = parse_trusted_rows()
        .into_iter()
        .filter(|row| row.rank == 20 && row.d == 32 && row.coeff_linf_bound == 255)
        .collect();
    assert_eq!(rows.len(), 3, "expected one rank-20 cell per family");
    for row in rows {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let local = estimate(&params, &local_minimum_config()).unwrap();
        let exhaustive = estimate(&params, &exhaustive_config()).unwrap();
        assert!(
            exhaustive_at_least_as_good(&exhaustive, &local, tol),
            "rank-20 exhaustive regression for {row:?}\n  local={local:?}\n  exhaustive={exhaustive:?}"
        );
    }
}

#[derive(Debug)]
struct Row {
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
}

impl Row {
    fn column_count(&self) -> u32 {
        self.width.saturating_mul(self.d)
    }
}

fn exhaustive_subset_rows() -> Vec<Row> {
    parse_trusted_rows()
        .into_iter()
        .filter(exhaustive_subset_includes)
        .collect()
}

fn exhaustive_subset_includes(row: &Row) -> bool {
    let m = row.column_count();
    if m <= EXHAUSTIVE_FAST_MAX_M {
        return true;
    }
    if m > EXHAUSTIVE_SLOW_MAX_M {
        return false;
    }
    row.coeff_linf_bound == EXHAUSTIVE_SLOW_REPRESENTATIVE_BOUND
}

fn parse_trusted_rows() -> Vec<Row> {
    let mut lines = GOLDEN_CSV.lines();
    let header = lines.next().unwrap();
    let columns: Vec<&str> = header.split(',').collect();
    lines
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let get = |name: &str| {
                let index = columns
                    .iter()
                    .position(|column| *column == name)
                    .unwrap_or_else(|| panic!("missing column {name}"));
                fields[index]
            };
            if get("trust") != "trusted" {
                return None;
            }
            Some(Row {
                family: AkitaModulusFamily::parse(get("family")).unwrap(),
                d: get("d").parse().unwrap(),
                rank: get("rank").parse().unwrap(),
                width: get("width").parse().unwrap(),
                coeff_linf_bound: get("coeff_linf_bound").parse().unwrap(),
            })
        })
        .collect()
}

fn exhaustive_at_least_as_good(
    exhaustive: &akita_sis_estimator::LatticeCost,
    reference: &akita_sis_estimator::LatticeCost,
    tol: f64,
) -> bool {
    match (exhaustive.rop, reference.rop) {
        (CostValue::Infinity, CostValue::Infinity) => true,
        (CostValue::Finite(ex), CostValue::Finite(reference)) => ex.log2 <= reference.log2 + tol,
        (CostValue::Finite(_), CostValue::Infinity) => true,
        (CostValue::Infinity, CostValue::Finite(_)) => false,
    }
}
