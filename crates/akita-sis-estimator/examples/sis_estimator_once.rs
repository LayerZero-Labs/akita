use akita_sis_estimator::{
    cost_infinity, cost_zeta, estimate, scalar_sis_from_ring, AkitaModulusProfileId, Bound,
    CostValue, EstimateConfig, OptimizerConfig, SearchMode, SisNorm, SisParameters,
};
use std::{
    env,
    hint::black_box,
    process,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Estimate,
    Fixed,
    Zeta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Profile {
    LocalMinimum,
    ExhaustiveSerial,
    ExhaustiveParallel,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    profile: Profile,
    family: AkitaModulusProfileId,
    raw_n: Option<u32>,
    raw_m: Option<u64>,
    d: u32,
    rank: u32,
    width: u32,
    coeff_linf_bound: u64,
    beta: Option<u32>,
    zeta: Option<u64>,
    iterations: u32,
}

fn main() {
    let args = Args::parse_or_exit();
    let params = args
        .params()
        .unwrap_or_else(|error| fatal(&format!("invalid SIS parameters: {error}")));

    let mut total = Duration::ZERO;
    let mut last = None;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let cost = match args.mode {
            Mode::Estimate => estimate(black_box(&params), black_box(&args.profile.config())),
            Mode::Fixed => {
                let beta = args
                    .beta
                    .unwrap_or_else(|| fatal("--beta is required for --mode fixed"));
                let zeta = args
                    .zeta
                    .unwrap_or_else(|| fatal("--zeta is required for --mode fixed"));
                let config = EstimateConfig {
                    optimizer: OptimizerConfig::Fixed { beta, zeta },
                    ..args.profile.config()
                };
                cost_infinity(
                    black_box(beta),
                    black_box(&params),
                    black_box(zeta),
                    black_box(&config),
                )
            }
            Mode::Zeta => {
                let zeta = args
                    .zeta
                    .unwrap_or_else(|| fatal("--zeta is required for --mode zeta"));
                let config = EstimateConfig {
                    optimizer: OptimizerConfig::OptimizeBeta {
                        zeta,
                        beta: args.profile.beta_search_mode(),
                    },
                    ..args.profile.config()
                };
                cost_zeta(black_box(zeta), black_box(&params), black_box(&config))
            }
        }
        .unwrap_or_else(|error| fatal(&format!("estimator failed: {error}")));
        total += start.elapsed();
        last = Some(black_box(cost));
    }

    let cost = last.expect("at least one iteration is required");
    let seconds = total.as_secs_f64();
    let seconds_per_iter = seconds / f64::from(args.iterations);
    println!(
        "mode,family,n,m,d,rank,width,coeff_linf_bound,iterations,total_seconds,seconds_per_iter,rop_log2,beta,zeta,lattice_dimension"
    );
    println!(
        "{},{},{},{},{},{},{},{},{},{:.9},{:.9},{},{},{},{}",
        args.mode.label(),
        args.family.label(),
        params.n,
        params.m.unwrap_or(0),
        args.d,
        args.rank,
        args.width,
        args.coeff_linf_bound,
        args.iterations,
        seconds,
        seconds_per_iter,
        log2_text(cost.rop),
        optional_u32_text(cost.beta),
        optional_u64_text(cost.zeta),
        cost.d
    );
}

impl Args {
    fn parse_or_exit() -> Self {
        let mut args = env::args().skip(1);
        let mut parsed = Self {
            mode: Mode::Estimate,
            profile: Profile::LocalMinimum,
            family: AkitaModulusProfileId::Q32Offset99,
            raw_n: None,
            raw_m: None,
            d: 0,
            rank: 0,
            width: 0,
            coeff_linf_bound: 0,
            beta: None,
            zeta: None,
            iterations: 1,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => usage(0),
                _ => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| fatal(&format!("missing value for {arg}")));
                    match arg.as_str() {
                        "--mode" => parsed.mode = parse_mode(&value),
                        "--profile" => parsed.profile = parse_profile(&value),
                        "--family" => {
                            parsed.family = AkitaModulusProfileId::parse(&value)
                                .unwrap_or_else(|error| fatal(&format!("{error}")));
                        }
                        "--n" => parsed.raw_n = Some(parse(&value, "--n")),
                        "--m" => parsed.raw_m = Some(parse(&value, "--m")),
                        "--d" => parsed.d = parse(&value, "--d"),
                        "--rank" => parsed.rank = parse(&value, "--rank"),
                        "--width" => parsed.width = parse(&value, "--width"),
                        "--coeff-linf-bound" => {
                            parsed.coeff_linf_bound = parse(&value, "--coeff-linf-bound");
                        }
                        "--beta" => parsed.beta = Some(parse(&value, "--beta")),
                        "--zeta" => parsed.zeta = Some(parse(&value, "--zeta")),
                        "--iterations" => parsed.iterations = parse(&value, "--iterations"),
                        _ => fatal(&format!("unknown argument {arg}")),
                    }
                }
            }
        }

        let has_raw_shape = parsed.raw_n.is_some() && parsed.raw_m.is_some();
        let has_ring_shape = parsed.d != 0 && parsed.rank != 0 && parsed.width != 0;
        if (!has_raw_shape && !has_ring_shape)
            || parsed.coeff_linf_bound == 0
            || parsed.iterations == 0
        {
            usage(2);
        }
        parsed
    }

    fn params(&self) -> akita_sis_estimator::Result<SisParameters> {
        match (self.raw_n, self.raw_m) {
            (Some(n), Some(m)) => SisParameters::try_new(
                n,
                self.family.modulus(),
                Some(m),
                Bound::from_u64(self.coeff_linf_bound),
                SisNorm::Infinity,
            ),
            (None, None) => scalar_sis_from_ring(
                self.family,
                self.d,
                self.rank,
                self.width,
                self.coeff_linf_bound,
            ),
            _ => {
                eprintln!("error: --n and --m must be provided together");
                process::exit(2);
            }
        }
    }
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Self::Estimate => "estimate",
            Self::Fixed => "fixed",
            Self::Zeta => "zeta",
        }
    }
}

impl Profile {
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

    const fn beta_search_mode(self) -> SearchMode {
        match self {
            Self::LocalMinimum => SearchMode::PythonLocalMinimum,
            Self::ExhaustiveSerial => SearchMode::Exhaustive,
            Self::ExhaustiveParallel => SearchMode::ExhaustiveParallel,
        }
    }
}

fn parse_mode(value: &str) -> Mode {
    match value {
        "estimate" => Mode::Estimate,
        "fixed" => Mode::Fixed,
        "zeta" => Mode::Zeta,
        _ => fatal("mode must be one of: estimate, fixed, zeta"),
    }
}

fn parse_profile(value: &str) -> Profile {
    match value {
        "local-minimum" | "local_minimum" => Profile::LocalMinimum,
        "exhaustive-serial" | "exhaustive_serial" => Profile::ExhaustiveSerial,
        "exhaustive-parallel" | "exhaustive_parallel" => Profile::ExhaustiveParallel,
        _ => {
            fatal("--profile must be one of: local-minimum, exhaustive-serial, exhaustive-parallel")
        }
    }
}

fn parse<T>(value: &str, field: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    value
        .parse()
        .unwrap_or_else(|error| fatal(&format!("invalid {field}: {error:?}")))
}

fn log2_text(value: CostValue) -> String {
    match value {
        CostValue::Finite(cost) => format!("{:.12}", cost.log2),
        CostValue::ProvenAboveTarget(lower_bound) => {
            format!("above-target:{:.12}", lower_bound.log2)
        }
        CostValue::Infinity => "inf".to_string(),
    }
}

fn optional_u32_text(value: Option<u32>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn optional_u64_text(value: Option<u64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn usage(code: i32) -> ! {
    eprintln!(
        "usage: sis_estimator_once --family q32|q64|q128 (--n N --m N | --d N --rank N --width N) --coeff-linf-bound N [--mode estimate|fixed|zeta] [--profile local-minimum|exhaustive-serial|exhaustive-parallel] [--beta N] [--zeta N] [--iterations N]"
    );
    process::exit(code);
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2);
}
