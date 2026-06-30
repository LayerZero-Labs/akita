use akita_sis_estimator::{
    cost_infinity, scalar_sis_from_ring, AkitaModulusFamily, EstimateConfig, OptimizerConfig,
    ReductionCostModel, ShapeModel,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const CASES_CSV_ENV: &str = "AKITA_SIS_FIXED_INFINITY_BENCH_CSV";

#[derive(Clone, Debug)]
struct FixedInfinityCase {
    label: String,
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    beta: u32,
    zeta: u64,
}

fn bench_fixed_infinity(c: &mut Criterion) {
    let cases = load_cases();
    let mut group = c.benchmark_group("sis_fixed_infinity");
    for case in &cases {
        let params = scalar_sis_from_ring(
            case.family,
            case.d,
            case.rank,
            case.width,
            case.coeff_linf_bound,
        )
        .unwrap();
        let config = EstimateConfig {
            red_cost_model: ReductionCostModel::default(),
            red_shape_model: ShapeModel::Lgsa,
            optimizer: OptimizerConfig::Fixed {
                beta: case.beta,
                zeta: case.zeta,
            },
            ..EstimateConfig::default()
        };

        group.bench_function(BenchmarkId::new("cost_infinity", &case.label), |bench| {
            bench.iter(|| {
                black_box(
                    cost_infinity(
                        black_box(case.beta),
                        black_box(&params),
                        black_box(case.zeta),
                        black_box(&config),
                    )
                    .unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn load_cases() -> Vec<FixedInfinityCase> {
    match env::var_os(CASES_CSV_ENV) {
        Some(path) => load_cases_csv(&resolve_csv_path(Path::new(&path))),
        None => default_cases(),
    }
}

fn default_cases() -> Vec<FixedInfinityCase> {
    vec![
        case(AkitaModulusFamily::Q32, 32, 1, 2, 2, 63, 0),
        case(AkitaModulusFamily::Q32, 32, 1, 2, 15, 40, 0),
        case(AkitaModulusFamily::Q32, 32, 5, 10, 15, 50, 0),
        case(AkitaModulusFamily::Q64, 32, 1, 2, 15, 63, 0),
        case(AkitaModulusFamily::Q128, 32, 1, 2, 15, 63, 0),
    ]
}

fn case(
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    beta: u32,
    zeta: u64,
) -> FixedInfinityCase {
    FixedInfinityCase {
        label: format_case_label(family, d, rank, width, coeff_linf_bound, beta, zeta),
        family,
        d,
        rank,
        width,
        coeff_linf_bound,
        beta,
        zeta,
    }
}

fn load_cases_csv(path: &Path) -> Vec<FixedInfinityCase> {
    let contents = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "failed to read fixed infinity bench CSV {}: {error}",
            path.display()
        )
    });
    let mut lines = contents.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("fixed infinity bench CSV {} is empty", path.display()));
    let columns: Vec<&str> = header.split(',').collect();
    let mut cases = Vec::new();
    for (row_index, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        let row = row_index + 2;
        if get_optional(&columns, &fields, "trust") == Some("fragile") {
            continue;
        }
        let family = AkitaModulusFamily::parse(get(&columns, &fields, "family", row)).unwrap();
        let d = parse(get(&columns, &fields, "d", row), "d", row);
        let rank = parse(get(&columns, &fields, "rank", row), "rank", row);
        let width = parse(get(&columns, &fields, "width", row), "width", row);
        let coeff_linf_bound = parse(
            get(&columns, &fields, "coeff_linf_bound", row),
            "coeff_linf_bound",
            row,
        );
        let beta = parse(get(&columns, &fields, "beta_input", row), "beta_input", row);
        let zeta = parse(get(&columns, &fields, "zeta_input", row), "zeta_input", row);
        let label = get_optional(&columns, &fields, "label")
            .filter(|value| !value.is_empty())
            .map_or_else(
                || format_case_label(family, d, rank, width, coeff_linf_bound, beta, zeta),
                str::to_string,
            );
        cases.push(FixedInfinityCase {
            label,
            family,
            d,
            rank,
            width,
            coeff_linf_bound,
            beta,
            zeta,
        });
    }
    assert!(
        !cases.is_empty(),
        "fixed infinity bench CSV {} produced no benchmark cases",
        path.display()
    );
    cases
}

fn resolve_csv_path(path: &Path) -> PathBuf {
    if path.exists() || path.is_absolute() {
        return path.to_path_buf();
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crate_relative = manifest_dir.join(path);
    if crate_relative.exists() {
        return crate_relative;
    }
    manifest_dir.join("../..").join(path)
}

fn get<'a>(columns: &[&str], fields: &'a [&str], name: &str, row: usize) -> &'a str {
    get_optional(columns, fields, name)
        .unwrap_or_else(|| panic!("missing column {name} or field at row {row}"))
}

fn get_optional<'a>(columns: &[&str], fields: &'a [&str], name: &str) -> Option<&'a str> {
    let index = columns.iter().position(|column| *column == name)?;
    fields.get(index).copied()
}

fn parse<T>(value: &str, field: &str, row: usize) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    value
        .parse()
        .unwrap_or_else(|error| panic!("invalid {field} at row {row}: {error:?}"))
}

fn format_case_label(
    family: AkitaModulusFamily,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    beta: u32,
    zeta: u64,
) -> String {
    format!(
        "{}_d{d}_r{rank}_w{width}_linf{coeff_linf_bound}_beta{beta}_zeta{zeta}",
        family_label(family)
    )
}

fn family_label(family: AkitaModulusFamily) -> &'static str {
    match family {
        AkitaModulusFamily::Q32 => "q32",
        AkitaModulusFamily::Q64 => "q64",
        AkitaModulusFamily::Q128 => "q128",
    }
}

criterion_group!(fixed_infinity, bench_fixed_infinity);
criterion_main!(fixed_infinity);
