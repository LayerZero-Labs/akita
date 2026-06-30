use akita_sis_estimator::{
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthProfile,
        InfinityWidthRow, InfinityWidthTableConfig,
    },
    AkitaModulusFamily,
};
use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process,
    time::Instant,
};

#[derive(Debug)]
struct Args {
    config: InfinityWidthTableConfig,
    output: Option<PathBuf>,
}

fn main() {
    let args = Args::parse_or_exit();
    let t0 = Instant::now();
    let rows = generate_infinity_width_rows(&args.config)
        .unwrap_or_else(|error| fatal(&format!("width-table generation failed: {error}")));
    validate_infinity_width_rows(&rows)
        .unwrap_or_else(|error| fatal(&format!("width-table validation failed: {error}")));
    write_rows(&rows, args.output.as_ref())
        .unwrap_or_else(|error| fatal(&format!("failed to write width table: {error}")));
    eprintln!(
        "wrote {} infinity width row(s) in {:.3}s",
        rows.len(),
        t0.elapsed().as_secs_f64()
    );
}

impl Args {
    fn parse_or_exit() -> Self {
        let mut config = InfinityWidthTableConfig::default();
        let mut output = None;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => usage(0),
                _ => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| fatal(&format!("missing value for {arg}")));
                    match arg.as_str() {
                        "--output" => output = Some(PathBuf::from(value)),
                        "--families" => config.families = parse_families(&value),
                        "--dims" => config.ring_dims = parse_csv(&value, "--dims"),
                        "--bounds" => {
                            config.coeff_linf_bounds = parse_csv(&value, "--bounds");
                        }
                        "--max-rank" => config.max_rank = parse(&value, "--max-rank"),
                        "--target-bits" => config.target_bits = parse(&value, "--target-bits"),
                        "--search-cap" => config.search_cap = Some(parse(&value, "--search-cap")),
                        "--profile" => config.profile = parse_profile(&value),
                        _ => fatal(&format!("unknown argument {arg}")),
                    }
                }
            }
        }
        Self { config, output }
    }
}

fn write_rows(rows: &[InfinityWidthRow], output: Option<&PathBuf>) -> io::Result<()> {
    match output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::File::create(path)?;
            write_rows_to(&mut file, rows)
        }
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            write_rows_to(&mut handle, rows)
        }
    }
}

fn write_rows_to(mut writer: impl Write, rows: &[InfinityWidthRow]) -> io::Result<()> {
    writeln!(writer, "{}", InfinityWidthRow::csv_header())?;
    for row in rows {
        writeln!(writer, "{}", row.to_csv_record())?;
    }
    Ok(())
}

fn parse_families(raw: &str) -> Vec<AkitaModulusFamily> {
    raw.split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            AkitaModulusFamily::parse(value.trim())
                .unwrap_or_else(|error| fatal(&format!("invalid --families entry: {error}")))
        })
        .collect()
}

fn parse_csv<T>(raw: &str, field: &str) -> Vec<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    let values: Vec<T> = raw
        .split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| parse(value.trim(), field))
        .collect();
    if values.is_empty() {
        fatal(&format!("{field} must not be empty"));
    }
    values
}

fn parse_profile(value: &str) -> InfinityWidthProfile {
    match value {
        "local-minimum" | "local_minimum" => InfinityWidthProfile::LocalMinimum,
        "exhaustive-serial" | "exhaustive_serial" => InfinityWidthProfile::ExhaustiveSerial,
        "exhaustive-parallel" | "exhaustive_parallel" => {
            #[cfg(not(feature = "parallel"))]
            fatal("profile exhaustive-parallel requires building with --features parallel");
            #[cfg(feature = "parallel")]
            {
                InfinityWidthProfile::ExhaustiveParallel
            }
        }
        _ => fatal("profile must be one of: local-minimum, exhaustive-serial, exhaustive-parallel"),
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

fn usage(code: i32) -> ! {
    eprintln!(
        "usage: infinity_width_table [--output PATH] [--families q32,q64,q128] [--dims 32,64,128,256] [--bounds B1,B2] [--max-rank N] [--target-bits BITS] [--search-cap N] [--profile local-minimum|exhaustive-serial|exhaustive-parallel]"
    );
    process::exit(code);
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2);
}
