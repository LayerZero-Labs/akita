//! Compare Python-local-minimum, certified, and full-domain search on golden cells.
//!
//! The same grids are available for benchmarking via
//! `AKITA_SIS_INFINITY_BENCH_SET=exhaustive-ci` in
//! `cargo bench -p akita-sis-estimator --bench infinity_optimizer`.

#[cfg(feature = "parallel")]
use akita_sis_estimator::width_table::InfinityWidthProfile;
use akita_sis_estimator::{
    estimate, scalar_sis_from_ring, AkitaModulusProfileId, CostValue, EstimateConfig, NumericConfig,
};

const GOLDEN_CSV: &str = include_str!("../../../scripts/sis_golden/infinity_golden.csv");

/// Fast tier: include every coeff bound when column count is at most this.
const EXHAUSTIVE_FAST_MAX_M: u32 = 512;

/// Slow tier: include one representative bound per geometry up to this column count.
const EXHAUSTIVE_SLOW_MAX_M: u32 = 1024;

/// Representative coeff bound for slow-tier geometries (middle of the golden ladder).
const EXHAUSTIVE_SLOW_REPRESENTATIVE_BOUND: u64 = 255;

struct SmokeCase {
    family: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
}

/// Representative cells aligned with `benches/infinity_optimizer.rs`.
const SMOKE_CASES: &[SmokeCase] = &[
    SmokeCase {
        family: AkitaModulusProfileId::Q32Offset99,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 15,
    },
    SmokeCase {
        family: AkitaModulusProfileId::Q128OffsetA7F7,
        d: 32,
        rank: 1,
        width: 8,
        coeff_linf_bound: 4095,
    },
    SmokeCase {
        family: AkitaModulusProfileId::Q64Offset59,
        d: 64,
        rank: 1,
        width: 8,
        coeff_linf_bound: 255,
    },
    SmokeCase {
        family: AkitaModulusProfileId::Q64Offset59,
        d: 128,
        rank: 1,
        width: 8,
        coeff_linf_bound: 15,
    },
    SmokeCase {
        family: AkitaModulusProfileId::Q64Offset59,
        d: 32,
        rank: 5,
        width: 10,
        coeff_linf_bound: 255,
    },
    SmokeCase {
        family: AkitaModulusProfileId::Q128OffsetA7F7,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 2,
    },
];

fn local_minimum_config() -> EstimateConfig {
    EstimateConfig::lattice_estimator_parity()
}

fn certified_config() -> EstimateConfig {
    EstimateConfig::akita_infinity_table()
}

#[cfg(feature = "parallel")]
fn full_exhaustive_parallel_config() -> EstimateConfig {
    InfinityWidthProfile::ExhaustiveParallel.config()
}

#[test]
fn certified_search_smoke_grid_covers_families_and_rank5() {
    let families: std::collections::HashSet<_> = SMOKE_CASES.iter().map(|row| row.family).collect();
    assert_eq!(families.len(), 3, "expected all three modulus families");
    assert!(
        SMOKE_CASES.iter().any(|row| row.rank == 5),
        "expected rank-5 coverage"
    );
}

#[test]
fn certified_search_is_at_least_as_good_as_local_minimum_smoke() {
    let tol = NumericConfig::default().sage_abs_tolerance;
    let mut violations = Vec::new();
    for row in smoke_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let local = estimate(&params, &local_minimum_config()).unwrap();
        let certified = estimate(&params, &certified_config()).unwrap();
        if !exhaustive_at_least_as_good(&certified, &local, tol) {
            violations.push(format!(
                "certified search worse than local-minimum for {row:?}\n  local={local:?}\n  certified={certified:?}"
            ));
        }
    }
    if !violations.is_empty() {
        panic!("{}", violations.join("\n\n"));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn full_exhaustive_parallel_matches_certified_search_smoke() {
    let mut mismatches = Vec::new();
    for row in smoke_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let certified = estimate(&params, &certified_config()).unwrap();
        let full = estimate(&params, &full_exhaustive_parallel_config()).unwrap();
        if !same_exact_or_above_target_result(&certified, &full) {
            mismatches.push(format!(
                "full exhaustive mismatch for {row:?}\n  certified={certified:?}\n  full={full:?}"
            ));
        }
    }
    if !mismatches.is_empty() {
        panic!("{}", mismatches.join("\n\n"));
    }
}

#[test]
fn certified_search_covers_medium_trusted_grid() {
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
fn certified_search_is_at_least_as_good_as_local_minimum_on_medium_subset() {
    let tol = NumericConfig::default().sage_abs_tolerance;
    let mut violations = Vec::new();
    for row in exhaustive_subset_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let local = estimate(&params, &local_minimum_config()).unwrap();
        let certified = estimate(&params, &certified_config()).unwrap();
        if !exhaustive_at_least_as_good(&certified, &local, tol) {
            violations.push(format!(
                "certified search worse than local-minimum for {row:?}\n  local={local:?}\n  certified={certified:?}"
            ));
        }
    }
    if !violations.is_empty() {
        panic!("{}", violations.join("\n\n"));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn full_exhaustive_parallel_matches_certified_search_on_medium_subset() {
    let mut mismatches = Vec::new();
    for row in exhaustive_subset_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let certified = estimate(&params, &certified_config()).unwrap();
        let full = estimate(&params, &full_exhaustive_parallel_config()).unwrap();
        if !same_exact_or_above_target_result(&certified, &full) {
            mismatches.push(format!(
                "full exhaustive mismatch for {row:?}\n  certified={certified:?}\n  full={full:?}"
            ));
        }
    }
    if !mismatches.is_empty() {
        panic!("{}", mismatches.join("\n\n"));
    }
}

/// Rank-20 geometries have `m = 1280`.
#[test]
fn certified_search_rank20_geometries() {
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
        let certified = estimate(&params, &certified_config()).unwrap();
        assert!(
            exhaustive_at_least_as_good(&certified, &local, tol),
            "rank-20 certified-search regression for {row:?}\n  local={local:?}\n  certified={certified:?}"
        );
    }
}

#[derive(Debug)]
struct Row {
    family: AkitaModulusProfileId,
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

fn smoke_rows() -> Vec<Row> {
    SMOKE_CASES
        .iter()
        .map(|case| Row {
            family: case.family,
            d: case.d,
            rank: case.rank,
            width: case.width,
            coeff_linf_bound: case.coeff_linf_bound,
        })
        .collect()
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
                family: AkitaModulusProfileId::parse(get("family")).unwrap(),
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
        (CostValue::ProvenAboveTarget(_), CostValue::ProvenAboveTarget(_)) => true,
        (CostValue::ProvenAboveTarget(_), CostValue::Infinity) => true,
        (CostValue::Infinity, CostValue::ProvenAboveTarget(_)) => true,
        (CostValue::Finite(_), CostValue::ProvenAboveTarget(_)) => true,
        (CostValue::ProvenAboveTarget(lower_bound), CostValue::Finite(reference)) => {
            lower_bound.log2 <= reference.log2 + tol
        }
    }
}

#[cfg(feature = "parallel")]
fn same_exact_or_above_target_result(
    lhs: &akita_sis_estimator::LatticeCost,
    rhs: &akita_sis_estimator::LatticeCost,
) -> bool {
    let tol = NumericConfig::default().sage_abs_tolerance;
    match (lhs.rop, rhs.rop) {
        (CostValue::Finite(lhs), CostValue::Finite(rhs)) => (lhs.log2 - rhs.log2).abs() <= tol,
        (CostValue::ProvenAboveTarget(lhs), CostValue::Finite(rhs))
        | (CostValue::Finite(rhs), CostValue::ProvenAboveTarget(lhs)) => rhs.log2 >= lhs.log2,
        (CostValue::ProvenAboveTarget(lhs), CostValue::ProvenAboveTarget(rhs)) => {
            lhs.log2 >= 128.0 && rhs.log2 >= 128.0
        }
        (CostValue::Infinity, CostValue::Infinity) => true,
        (CostValue::ProvenAboveTarget(lower_bound), CostValue::Infinity)
        | (CostValue::Infinity, CostValue::ProvenAboveTarget(lower_bound)) => {
            lower_bound.log2 >= 128.0
        }
        (CostValue::Finite(_), CostValue::Infinity)
        | (CostValue::Infinity, CostValue::Finite(_)) => false,
    }
}
