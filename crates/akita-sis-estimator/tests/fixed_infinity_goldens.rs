use akita_sis_estimator::{
    cost_infinity, scalar_sis_from_ring, AkitaModulusFamily, CostValue, EstimateConfig,
    EstimatorError, NearestNeighborModel, OptimizerConfig, ReductionCostModel, ShapeModel,
};

const FLOAT_TOLERANCE: f64 = 1e-6;

#[derive(Debug)]
struct FixedGoldenRow {
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    beta_input: u32,
    zeta_input: u32,
    rop_log2: f64,
    red_log2: f64,
    sieve_log2: ExpectedLog2,
    repetitions_log2: f64,
    beta: u32,
    eta: u32,
    zeta: u32,
    lattice_dimension: u32,
    trust: String,
}

#[test]
fn fixed_infinity_lgsa_goldens_match_pinned_estimator_trusted_rows() {
    run_fixed_infinity_goldens(
        include_str!("../../../scripts/sis_golden/fixed_infinity_golden.csv"),
        ShapeModel::Lgsa,
        ReductionCostModel::default(),
    );
}

#[test]
fn fixed_infinity_gsa_goldens_match_pinned_estimator_trusted_rows() {
    run_fixed_infinity_goldens(
        include_str!("../../../scripts/sis_golden/fixed_infinity_golden_adps16_gsa_fixed.csv"),
        ShapeModel::Gsa,
        ReductionCostModel::default(),
    );
}

#[test]
fn fixed_infinity_zgsa_goldens_match_pinned_estimator_trusted_rows() {
    run_fixed_infinity_goldens(
        include_str!("../../../scripts/sis_golden/fixed_infinity_golden_adps16_zgsa_fixed.csv"),
        ShapeModel::Zgsa,
        ReductionCostModel::default(),
    );
}

#[test]
fn fixed_infinity_bdgl16_goldens_match_pinned_estimator_trusted_rows() {
    run_fixed_infinity_goldens(
        include_str!("../../../scripts/sis_golden/fixed_infinity_golden_bdgl16_lgsa_fixed.csv"),
        ShapeModel::Lgsa,
        ReductionCostModel::Bdgl16,
    );
}

#[test]
fn fixed_infinity_rejects_unsupported_profiles_without_panicking() {
    let params = scalar_sis_from_ring(AkitaModulusFamily::Q32, 32, 1, 2, 15).unwrap();
    let config = EstimateConfig {
        red_cost_model: ReductionCostModel::Matzov {
            nearest_neighbor: NearestNeighborModel::Classical,
        },
        red_shape_model: ShapeModel::Lgsa,
        ..EstimateConfig::default()
    };

    assert!(matches!(
        cost_infinity(63, &params, 0, &config),
        Err(EstimatorError::Unsupported { .. })
    ));
}

fn run_fixed_infinity_goldens(
    csv: &str,
    shape_model: ShapeModel,
    red_cost_model: ReductionCostModel,
) {
    for row in parse_rows(csv) {
        if row.trust != "trusted" {
            continue;
        }

        let params =
            scalar_sis_from_ring(row.family, row.d, row.rank, row.width, row.coeff_linf_bound)
                .unwrap();
        let config = EstimateConfig {
            red_cost_model,
            red_shape_model: shape_model,
            optimizer: OptimizerConfig::Fixed {
                beta: row.beta_input,
                zeta: row.zeta_input,
            },
            ..EstimateConfig::default()
        };
        let cost = cost_infinity(row.beta_input, &params, row.zeta_input, &config).unwrap();

        assert_log2_close(row.rop_log2, cost.rop, "rop", &row);
        assert_log2_close(row.red_log2, cost.red.unwrap(), "red", &row);
        assert_log2_matches(row.sieve_log2, cost.sieve.unwrap(), "sieve", &row);
        assert_log2_close(
            row.repetitions_log2,
            cost.repetitions.unwrap(),
            "repetitions",
            &row,
        );
        assert_eq!(cost.beta, Some(row.beta), "{row:?}");
        assert_eq!(cost.eta, Some(row.eta), "{row:?}");
        assert_eq!(cost.zeta, Some(row.zeta), "{row:?}");
        assert_eq!(cost.d, row.lattice_dimension, "{row:?}");
    }
}

fn parse_rows(csv: &str) -> Vec<FixedGoldenRow> {
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    let columns: Vec<&str> = header.split(',').collect();
    lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let get = |name: &str| {
                let index = columns
                    .iter()
                    .position(|column| *column == name)
                    .unwrap_or_else(|| panic!("missing column {name}"));
                fields[index]
            };
            FixedGoldenRow {
                family: AkitaModulusFamily::parse(get("family")).unwrap(),
                d: parse(get("d")),
                rank: parse(get("rank")),
                width: parse(get("width")),
                coeff_linf_bound: parse(get("coeff_linf_bound")),
                beta_input: parse(get("beta_input")),
                zeta_input: parse(get("zeta_input")),
                rop_log2: parse(get("rop_log2")),
                red_log2: parse(get("red_log2")),
                sieve_log2: parse_expected_log2(get("sieve_log2")),
                repetitions_log2: parse(get("repetitions_log2")),
                beta: parse(get("beta")),
                eta: parse(get("eta")),
                zeta: parse(get("zeta")),
                lattice_dimension: parse(get("lattice_dimension")),
                trust: get("trust").to_string(),
            }
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

fn assert_log2_matches(
    expected: ExpectedLog2,
    actual: CostValue,
    field: &str,
    row: &FixedGoldenRow,
) {
    match expected {
        ExpectedLog2::Finite(expected) => assert_log2_close(expected, actual, field, row),
        ExpectedLog2::Infinity => assert!(
            matches!(actual, CostValue::Infinity),
            "{field}: expected inf, got {actual:?} for {row:?}",
        ),
    }
}

fn assert_log2_close(expected: f64, actual: CostValue, field: &str, row: &FixedGoldenRow) {
    match actual {
        CostValue::Finite(actual) => {
            let diff = (expected - actual.log2).abs();
            assert!(
                diff <= FLOAT_TOLERANCE,
                "{field}: expected {expected}, got {} for {row:?}",
                actual.log2
            );
        }
        CostValue::Infinity => panic!("{field}: expected finite {expected}, got inf for {row:?}"),
    }
}
