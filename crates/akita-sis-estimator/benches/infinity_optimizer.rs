use akita_sis_estimator::{
    estimate, scalar_sis_from_ring, AkitaModulusProfileId, EstimateConfig, OptimizerConfig,
    SearchMode,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Duration,
};

const CASES_CSV_ENV: &str = "AKITA_SIS_INFINITY_BENCH_CSV";
const CASE_SET_ENV: &str = "AKITA_SIS_INFINITY_BENCH_SET";
const PROFILES_ENV: &str = "AKITA_SIS_INFINITY_BENCH_PROFILES";
const SAMPLE_SIZE_ENV: &str = "AKITA_SIS_INFINITY_BENCH_SAMPLE_SIZE";
const WARM_UP_MS_ENV: &str = "AKITA_SIS_INFINITY_BENCH_WARM_UP_MS";
const MEASUREMENT_MS_ENV: &str = "AKITA_SIS_INFINITY_BENCH_MEASUREMENT_MS";
const MIN_SAMPLE_SIZE: usize = 10;

#[derive(Clone, Debug)]
struct InfinityCase {
    label: String,
    family: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
}

#[derive(Clone, Copy, Debug)]
struct RepresentativeCase {
    family: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
}

const REPRESENTATIVE_CASES: &[RepresentativeCase] = &[
    RepresentativeCase {
        family: AkitaModulusProfileId::Q32Offset99,
        d: 32,
        rank: 1,
        width: 2,
        coeff_linf_bound: 15,
    },
    RepresentativeCase {
        family: AkitaModulusProfileId::Q128OffsetA7F7,
        d: 32,
        rank: 1,
        width: 8,
        coeff_linf_bound: 4095,
    },
    RepresentativeCase {
        family: AkitaModulusProfileId::Q64Offset59,
        d: 64,
        rank: 1,
        width: 8,
        coeff_linf_bound: 255,
    },
    RepresentativeCase {
        family: AkitaModulusProfileId::Q64Offset59,
        d: 128,
        rank: 1,
        width: 8,
        coeff_linf_bound: 15,
    },
];

fn bench_infinity_optimizer(c: &mut Criterion) {
    let cases = load_cases();
    let profiles = load_profiles();
    let mut group = c.benchmark_group("sis_infinity_optimizer");
    configure_group(&mut group);
    for profile in &profiles {
        let config = profile.config();
        for case in &cases {
            let params = scalar_sis_from_ring(
                case.family,
                case.d,
                case.rank,
                case.width,
                case.coeff_linf_bound,
            )
            .unwrap();
            group.bench_function(BenchmarkId::new(profile.label(), &case.label), |bench| {
                bench.iter(|| black_box(estimate(black_box(&params), black_box(&config)).unwrap()));
            });
        }
    }
    group.finish();
}

fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    if let Some(sample_size) = env_usize(SAMPLE_SIZE_ENV) {
        group.sample_size(sample_size.max(MIN_SAMPLE_SIZE));
    }
    if let Some(warm_up_ms) = env_u64(WARM_UP_MS_ENV) {
        group.warm_up_time(Duration::from_millis(warm_up_ms));
    }
    if let Some(measurement_ms) = env_u64(MEASUREMENT_MS_ENV) {
        group.measurement_time(Duration::from_millis(measurement_ms));
    }
}

fn load_cases() -> Vec<InfinityCase> {
    match env::var_os(CASES_CSV_ENV) {
        Some(path) => load_cases_csv(&resolve_csv_path(Path::new(&path)), CaseSet::from_env()),
        None => default_cases(),
    }
}

fn default_cases() -> Vec<InfinityCase> {
    load_cases_csv(
        &resolve_csv_path(Path::new("scripts/sis_golden/infinity_golden.csv")),
        CaseSet::from_env(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaseSet {
    Representative,
    ExhaustiveCi,
    AllTrusted,
}

impl CaseSet {
    fn from_env() -> Self {
        match env::var(CASE_SET_ENV).as_deref() {
            Ok("representative") | Err(_) => Self::Representative,
            Ok("exhaustive-ci") => Self::ExhaustiveCi,
            Ok("all-trusted") => Self::AllTrusted,
            Ok(value) => panic!(
                "{CASE_SET_ENV} must be one of representative, exhaustive-ci, all-trusted; got {value:?}"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    LocalMinimum,
    ExhaustiveSerial,
    ExhaustiveParallel,
}

impl Profile {
    fn label(self) -> &'static str {
        match self {
            Self::LocalMinimum => "local_minimum",
            Self::ExhaustiveSerial => "exhaustive_serial",
            Self::ExhaustiveParallel => "exhaustive_parallel",
        }
    }

    fn config(self) -> EstimateConfig {
        match self {
            Self::LocalMinimum => EstimateConfig::lattice_estimator_parity(),
            Self::ExhaustiveSerial => EstimateConfig::akita_infinity_table(),
            Self::ExhaustiveParallel => EstimateConfig {
                optimizer: OptimizerConfig::OptimizeZeta {
                    beta: SearchMode::ExhaustiveParallel,
                    zeta: SearchMode::ExhaustiveParallel,
                },
                ..EstimateConfig::default()
            },
        }
    }
}

fn load_profiles() -> Vec<Profile> {
    match env::var(PROFILES_ENV) {
        Ok(value) => {
            let profiles: Vec<_> = value
                .split(',')
                .filter(|value| !value.trim().is_empty())
                .map(parse_profile)
                .collect();
            assert!(!profiles.is_empty(), "{PROFILES_ENV} produced no profiles");
            profiles
        }
        Err(_) => default_profiles(),
    }
}

#[cfg(feature = "parallel")]
fn default_profiles() -> Vec<Profile> {
    vec![
        Profile::LocalMinimum,
        Profile::ExhaustiveSerial,
        Profile::ExhaustiveParallel,
    ]
}

#[cfg(not(feature = "parallel"))]
fn default_profiles() -> Vec<Profile> {
    vec![Profile::LocalMinimum, Profile::ExhaustiveSerial]
}

fn parse_profile(value: &str) -> Profile {
    let profile = match value.trim() {
        "local-minimum" | "local_minimum" => Profile::LocalMinimum,
        "exhaustive-serial" | "exhaustive_serial" => Profile::ExhaustiveSerial,
        "exhaustive-parallel" | "exhaustive_parallel" => Profile::ExhaustiveParallel,
        value => panic!(
            "{PROFILES_ENV} entries must be local-minimum, exhaustive-serial, or exhaustive-parallel; got {value:?}"
        ),
    };
    if profile == Profile::ExhaustiveParallel && !cfg!(feature = "parallel") {
        panic!("{PROFILES_ENV}=exhaustive-parallel requires `--features parallel`");
    }
    profile
}

fn load_cases_csv(path: &Path, case_set: CaseSet) -> Vec<InfinityCase> {
    let contents = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!(
            "failed to read infinity optimizer bench CSV {}: {error}",
            path.display()
        )
    });
    let mut lines = contents.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("infinity optimizer bench CSV {} is empty", path.display()));
    let columns: Vec<&str> = header.split(',').collect();
    let mut cases = Vec::new();
    for (row_index, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        let row = row_index + 2;
        if get_optional(&columns, &fields, "trust") == Some("fragile") {
            continue;
        }
        let family = AkitaModulusProfileId::parse(get(&columns, &fields, "family", row)).unwrap();
        let d = parse(get(&columns, &fields, "d", row), "d", row);
        let rank = parse(get(&columns, &fields, "rank", row), "rank", row);
        let width = parse(get(&columns, &fields, "width", row), "width", row);
        let coeff_linf_bound = parse(
            get(&columns, &fields, "coeff_linf_bound", row),
            "coeff_linf_bound",
            row,
        );
        let candidate = InfinityCase {
            label: get_optional(&columns, &fields, "label")
                .filter(|value| !value.is_empty())
                .map_or_else(
                    || format_case_label(family, d, rank, width, coeff_linf_bound),
                    str::to_string,
                ),
            family,
            d,
            rank,
            width,
            coeff_linf_bound,
        };
        if !case_set_includes(case_set, &candidate) {
            continue;
        }
        cases.push(candidate);
    }
    if case_set == CaseSet::Representative {
        cases = representative_cases(cases);
    }
    assert!(
        !cases.is_empty(),
        "infinity optimizer bench CSV {} produced no benchmark cases",
        path.display()
    );
    cases
}

fn case_set_includes(case_set: CaseSet, case: &InfinityCase) -> bool {
    match case_set {
        CaseSet::Representative | CaseSet::AllTrusted => true,
        CaseSet::ExhaustiveCi => {
            let m = case.column_count();
            m <= 512 || (m <= 1024 && case.coeff_linf_bound == 255)
        }
    }
}

fn representative_cases(cases: Vec<InfinityCase>) -> Vec<InfinityCase> {
    REPRESENTATIVE_CASES
        .iter()
        .map(|spec| {
            cases
                .iter()
                .find(|case| spec.matches(case))
                .unwrap_or_else(|| {
                    panic!(
                        "infinity optimizer bench fixture is missing representative case {}",
                        spec.label()
                    )
                })
                .clone()
        })
        .collect()
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
    family: AkitaModulusProfileId,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
) -> String {
    format!(
        "{}_d{d}_r{rank}_w{width}_linf{coeff_linf_bound}",
        family_label(family)
    )
}

fn family_label(family: AkitaModulusProfileId) -> &'static str {
    match family {
        AkitaModulusProfileId::Q32Offset99 => "q32",
        AkitaModulusProfileId::Q64Offset59 => "q64",
        AkitaModulusProfileId::Q128OffsetA7F7 => "q128",
    }
}

impl InfinityCase {
    fn column_count(&self) -> u32 {
        self.width.saturating_mul(self.d)
    }
}

impl RepresentativeCase {
    fn matches(self, case: &InfinityCase) -> bool {
        case.family == self.family
            && case.d == self.d
            && case.rank == self.rank
            && case.width == self.width
            && case.coeff_linf_bound == self.coeff_linf_bound
    }

    fn label(self) -> String {
        format_case_label(
            self.family,
            self.d,
            self.rank,
            self.width,
            self.coeff_linf_bound,
        )
    }
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().map(|value| parse(&value, name, 0))
}

fn env_u64(name: &str) -> Option<u64> {
    env::var(name).ok().map(|value| parse(&value, name, 0))
}

criterion_group!(infinity_optimizer, bench_infinity_optimizer);
criterion_main!(infinity_optimizer);
