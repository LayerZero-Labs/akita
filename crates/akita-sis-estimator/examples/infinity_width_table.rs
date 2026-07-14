use akita_sis_estimator::{
    width_table::{
        generate_infinity_width_rows, is_full_infinity_width_table_config, rust_table_arms,
        validate_infinity_width_rows, InfinityWidthProfile, InfinityWidthRow,
        InfinityWidthTableConfig,
    },
    AkitaModulusProfileId,
};
use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

#[derive(Debug)]
struct Args {
    config: InfinityWidthTableConfig,
    output: Option<PathBuf>,
    format: OutputFormat,
    skip_validation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Csv,
    RustSplit,
}

fn main() {
    let args = Args::parse_or_exit();
    if args.format == OutputFormat::RustSplit && !is_full_infinity_width_table_config(&args.config)
    {
        fatal(
            "rust-split output requires the complete production table config; use CSV for partial comparison jobs",
        );
    }
    let t0 = Instant::now();
    let rows = generate_infinity_width_rows(&args.config)
        .unwrap_or_else(|error| fatal(&format!("width-table generation failed: {error}")));
    if !args.skip_validation {
        validate_infinity_width_rows(&rows)
            .unwrap_or_else(|error| fatal(&format!("width-table validation failed: {error}")));
    }
    match args.format {
        OutputFormat::Csv => write_csv_rows(&rows, args.output.as_ref())
            .unwrap_or_else(|error| fatal(&format!("failed to write CSV: {error}"))),
        OutputFormat::RustSplit => write_rust_split(&rows, &args.config, args.output.as_deref())
            .unwrap_or_else(|error| fatal(&format!("failed to write Rust table: {error}"))),
    }
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
        let mut format = OutputFormat::Csv;
        let mut skip_validation = false;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => usage(0),
                "--skip-validation" => skip_validation = true,
                _ => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| fatal(&format!("missing value for {arg}")));
                    match arg.as_str() {
                        "--output" => output = Some(PathBuf::from(value)),
                        "--format" => format = parse_format(&value),
                        "--profiles" => config.profiles = parse_profiles(&value),
                        "--dims" => config.ring_dims = parse_csv(&value, "--dims"),
                        "--bounds" => {
                            config.coeff_linf_bounds = parse_csv(&value, "--bounds");
                        }
                        "--max-rank" => config.max_rank = parse(&value, "--max-rank"),
                        "--search-cap" => config.search_cap = Some(parse(&value, "--search-cap")),
                        "--progress-every" => {
                            config.progress_every = Some(parse(&value, "--progress-every"));
                        }
                        "--profile" => config.profile = parse_profile(&value),
                        _ => fatal(&format!("unknown argument {arg}")),
                    }
                }
            }
        }
        Self {
            config,
            output,
            format,
            skip_validation,
        }
    }
}

fn write_csv_rows(rows: &[InfinityWidthRow], output: Option<&PathBuf>) -> io::Result<()> {
    match output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = fs::File::create(path)?;
            write_csv_rows_to(&mut file, rows)
        }
        None => {
            let stdout = io::stdout();
            let mut handle = stdout.lock();
            write_csv_rows_to(&mut handle, rows)
        }
    }
}

fn write_csv_rows_to(mut writer: impl Write, rows: &[InfinityWidthRow]) -> io::Result<()> {
    writeln!(writer, "{}", InfinityWidthRow::csv_header())?;
    for row in rows {
        writeln!(writer, "{}", row.to_csv_record())?;
    }
    Ok(())
}

fn write_rust_split(
    rows: &[InfinityWidthRow],
    config: &InfinityWidthTableConfig,
    output: Option<&Path>,
) -> io::Result<()> {
    let default_out_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../akita-types/src/sis/generated_sis_table");
    let out_dir = output.unwrap_or(default_out_dir.as_path());
    fs::create_dir_all(out_dir)?;
    fs::write(
        out_dir.join("mod.rs"),
        rust_mod_source(config.policy, config.profile),
    )?;
    let arms = rust_table_arms(rows, config.max_rank);
    for modulus_profile in [
        AkitaModulusProfileId::Q32Offset99,
        AkitaModulusProfileId::Q64Offset59,
        AkitaModulusProfileId::Q128OffsetA7F7,
    ] {
        fs::write(
            out_dir.join(format!("{}.rs", modulus_profile.label())),
            rust_modulus_profile_source(
                modulus_profile,
                config.policy,
                config.profile,
                arms.get(&modulus_profile).map(Vec::as_slice).unwrap_or(&[]),
            ),
        )?;
    }
    Ok(())
}

fn rust_mod_source(
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
) -> String {
    format!(
        "{}mod q128;\nmod q32;\nmod q64;\n\nuse super::{{ScalarCutoff, SisModulusProfileId, SisSecurityPolicyId}};\n\n/// Generated scalar SIS cutoff table for the named security policy.\n#[rustfmt::skip]\npub(crate) fn sis_scalar_cutoff(\n    policy: SisSecurityPolicyId,\n    modulus_profile: SisModulusProfileId,\n    coeff_linf_bound: u128,\n    n: u64,\n) -> Option<ScalarCutoff> {{\n    if policy != SisSecurityPolicyId::{} {{\n        return None;\n    }}\n    match modulus_profile {{\n        SisModulusProfileId::Q32Offset99 => q32::sis_scalar_cutoff(coeff_linf_bound, n),\n        SisModulusProfileId::Q64Offset59 => q64::sis_scalar_cutoff(coeff_linf_bound, n),\n        SisModulusProfileId::Q128OffsetA7F7 => q128::sis_scalar_cutoff(coeff_linf_bound, n),\n    }}\n}}\n",
        table_header(policy, profile),
        policy.label(),
    )
}

fn rust_modulus_profile_source(
    modulus_profile: AkitaModulusProfileId,
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
    arms: &[String],
) -> String {
    let mut source = format!(
        "{}// Profile: {}\n\n#[rustfmt::skip]\npub(super) fn sis_scalar_cutoff(coeff_linf_bound: u128, n: u64) -> Option<super::ScalarCutoff> {{\n    match (coeff_linf_bound, n) {{\n",
        table_header(policy, profile),
        modulus_profile.label()
    );
    for arm in arms {
        source.push_str("        ");
        source.push_str(arm);
        source.push('\n');
    }
    source.push_str("        _ => None,\n    }\n}\n");
    source
}

fn table_header(
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
) -> String {
    format!(
        "// AUTO-GENERATED by crates/akita-sis-estimator/examples/infinity_width_table.rs -- do not edit by hand.\n//\n// SIS scalar cutoffs for {}.\n// Sole hard gate: ADPS16 quantum LGSA model >= 128 bits.\n// Every accepted cutoff and immediate rejected successor is exhaustively certified.\n// Shape and norm: LGSA, coefficient L-infinity.\n// Rust estimator path: akita-sis-estimator::width_table.\n// Keys are (coefficient-L-infinity bound, scalar n).\n// Optimizer profile: {}.\n\n",
        policy.label(),
        profile.label()
    )
}

fn parse_profiles(raw: &str) -> Vec<AkitaModulusProfileId> {
    raw.split(',')
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            AkitaModulusProfileId::parse(value.trim())
                .unwrap_or_else(|error| fatal(&format!("invalid --profiles entry: {error}")))
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

fn parse_format(value: &str) -> OutputFormat {
    match value {
        "csv" => OutputFormat::Csv,
        "rust-split" | "rust_split" => OutputFormat::RustSplit,
        _ => fatal("--format must be one of: csv, rust-split"),
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
        "usage: infinity_width_table [--output PATH] [--format csv|rust-split] [--profiles q32,q64,q128] [--dims 32,64,128,256] [--bounds B1,B2] [--max-rank N] [--search-cap N] [--profile local-minimum|exhaustive-serial|exhaustive-parallel] [--progress-every N] [--skip-validation]"
    );
    process::exit(code);
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2);
}
