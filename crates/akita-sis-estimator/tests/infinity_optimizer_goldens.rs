use akita_sis_estimator::{
    estimate, scalar_sis_from_ring, AkitaModulusProfileId, CostValue, EstimateConfig, NumericConfig,
};

const GOLDEN_CSV: &str = include_str!("../../../scripts/sis_golden/infinity_golden.csv");

#[derive(Debug)]
struct OptimizerGoldenRow {
    family: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    rop_log2: ExpectedLog2,
    red_log2: ExpectedLog2,
    sieve_log2: ExpectedLog2,
    repetitions_log2: ExpectedLog2,
    beta: u32,
    eta: u32,
    zeta: u64,
    lattice_dimension: u64,
}

#[test]
fn infinity_optimizer_goldens_match_pr217_trusted_rows() {
    let mut mismatches = Vec::new();
    for row in parse_rows() {
        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let cost = estimate(&params, &EstimateConfig::lattice_estimator_parity()).unwrap();

        record_log2_mismatch(row.rop_log2, cost.rop, "rop", &row, &mut mismatches);
        record_log2_mismatch(
            row.red_log2,
            cost.red.unwrap(),
            "red",
            &row,
            &mut mismatches,
        );
        record_log2_mismatch(
            row.sieve_log2,
            cost.sieve.unwrap(),
            "sieve",
            &row,
            &mut mismatches,
        );
        if row.repetitions_log2 != ExpectedLog2::Infinity {
            record_log2_mismatch(
                row.repetitions_log2,
                cost.repetitions.unwrap(),
                "repetitions",
                &row,
                &mut mismatches,
            );
        }
        record_eq_mismatch("beta", row.beta, cost.beta, &row, &mut mismatches);
        record_eq_mismatch("eta", row.eta, cost.eta, &row, &mut mismatches);
        record_eq_mismatch_u64("zeta", row.zeta, cost.zeta, &row, &mut mismatches);
        if cost.d != row.lattice_dimension {
            mismatches.push(format!(
                "d: expected {}, got {} for {row:?}",
                row.lattice_dimension, cost.d
            ));
        }
    }

    if !mismatches.is_empty() {
        panic!("{}", mismatches.join("\n"));
    }
}

fn parse_rows() -> Vec<OptimizerGoldenRow> {
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
            Some(OptimizerGoldenRow {
                family: AkitaModulusProfileId::parse(get("family")).unwrap(),
                d: parse(get("d")),
                rank: parse(get("rank")),
                width: parse(get("width")),
                coeff_linf_bound: parse(get("coeff_linf_bound")),
                rop_log2: parse_expected_log2(get("rop_log2")),
                red_log2: parse_expected_log2(get("red_log2")),
                sieve_log2: parse_expected_log2(get("sieve_log2")),
                repetitions_log2: parse_expected_log2(get("repetitions_log2")),
                beta: parse(get("beta")),
                eta: parse(get("eta")),
                zeta: parse(get("zeta")),
                lattice_dimension: parse(get("lattice_dimension")),
            })
        })
        .collect()
}

fn parse<T>(value: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    value.parse().unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ExpectedLog2 {
    Finite(f64),
    Infinity,
}

fn parse_expected_log2(value: &str) -> ExpectedLog2 {
    match value {
        "inf" => ExpectedLog2::Infinity,
        _ => ExpectedLog2::Finite(parse(value)),
    }
}

fn record_log2_mismatch(
    expected: ExpectedLog2,
    actual: CostValue,
    field: &str,
    row: &OptimizerGoldenRow,
    mismatches: &mut Vec<String>,
) {
    match expected {
        ExpectedLog2::Finite(expected) => {
            record_log2_close_mismatch(expected, actual, field, row, mismatches);
        }
        ExpectedLog2::Infinity => {
            if !matches!(
                actual,
                CostValue::Infinity | CostValue::ProvenAboveTarget(_)
            ) {
                mismatches.push(format!("{field}: expected inf, got {actual:?} for {row:?}"));
            }
        }
    }
}

fn record_log2_close_mismatch(
    expected: f64,
    actual: CostValue,
    field: &str,
    row: &OptimizerGoldenRow,
    mismatches: &mut Vec<String>,
) {
    match actual {
        CostValue::Finite(actual) => {
            let diff = (expected - actual.log2).abs();
            if diff > NumericConfig::default().sage_abs_tolerance {
                mismatches.push(format!(
                    "{field}: expected {expected}, got {} for {row:?}",
                    actual.log2
                ));
            }
        }
        CostValue::ProvenAboveTarget(_) | CostValue::Infinity => {
            mismatches.push(format!(
                "{field}: expected finite {expected}, got inf for {row:?}"
            ));
        }
    }
}

fn record_eq_mismatch(
    field: &str,
    expected: u32,
    actual: Option<u32>,
    row: &OptimizerGoldenRow,
    mismatches: &mut Vec<String>,
) {
    if actual != Some(expected) {
        mismatches.push(format!(
            "{field}: expected {expected}, got {actual:?} for {row:?}"
        ));
    }
}

fn record_eq_mismatch_u64(
    field: &str,
    expected: u64,
    actual: Option<u64>,
    row: &OptimizerGoldenRow,
    mismatches: &mut Vec<String>,
) {
    if actual != Some(expected) {
        mismatches.push(format!(
            "{field}: expected {expected}, got {actual:?} for {row:?}"
        ));
    }
}
