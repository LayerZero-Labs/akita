use akita_sis_estimator::{
    config::ReductionCostModel,
    reduction::{adps16::adps16_exponent, BCSS23_IDEALIZED_EXPONENT},
    width_table::{
        generate_infinity_width_rows, is_full_infinity_width_table_config, rust_table_arms,
        validate_infinity_width_rows, InfinityWidthProfile, InfinityWidthRow,
        InfinityWidthTableConfig,
    },
    AkitaModulusFamily,
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
    if args.format == OutputFormat::RustSplit {
        if !is_full_infinity_width_table_config(&args.config) {
            fatal(
                "rust-split output requires the complete production table config; use CSV for partial comparison jobs",
            );
        }
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
                        "--families" => config.families = parse_families(&value),
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
    for family in [
        AkitaModulusFamily::Q32,
        AkitaModulusFamily::Q64,
        AkitaModulusFamily::Q128,
    ] {
        fs::write(
            out_dir.join(format!("{}.rs", family.label())),
            rust_family_source(
                family,
                config.policy,
                config.profile,
                arms.get(&family).map(Vec::as_slice).unwrap_or(&[]),
            ),
        )?;
    }
    // Keep review-only boundary provenance beside the compact runtime modules.
    // The CSV is not loaded by verifier-facing code, but records the
    // independently optimized hard-model and BCSS scores for every row.
    let audit_path = out_dir.join("policy_audit.csv");
    let mut audit = fs::File::create(audit_path)?;
    write_csv_rows_to(&mut audit, rows)?;
    let review_rows = rows
        .iter()
        .filter(|row| row.idealized_bcss_requires_review())
        .count();
    let review_status = if review_rows > 0 {
        "MANUAL_REVIEW_REQUIRED"
    } else {
        "NO_REVIEW_REQUIRED"
    };
    let review_below_bits = config
        .policy
        .idealized_bcss_diagnostic()
        .review_below_log2_rop;
    let review_disposition = if review_rows > 0 {
        "The BCSS diagnostic is non-gating under this policy; record a written disposition before merging regenerated artifacts."
    } else {
        "No accepted boundary row crossed the BCSS review line."
    };
    let review_path = out_dir.join("policy_review.txt");
    fs::write(
        review_path,
        format!(
            "policy={}\nbcss_review_below_bits={review_below_bits:.1}\naccepted_boundary_rows_requiring_review={}\nstatus={review_status}\n{review_disposition}\n",
            config.policy.label(),
            review_rows,
        ),
    )?;
    if review_rows > 0 {
        eprintln!(
            "BCSS policy review required: {review_rows} accepted boundary row(s) below the 124-bit diagnostic line; see policy_review.txt and policy_audit.csv"
        );
    }
    Ok(())
}

fn rust_mod_source(
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
) -> String {
    format!(
        "{}mod q128;\nmod q32;\nmod q64;\n\nuse super::{{SisModulusFamily, SisSecurityPolicyId}};\n\n/// Generated SIS max-width table for the named security policy.\n#[rustfmt::skip]\npub(crate) fn sis_max_widths(\n    policy: SisSecurityPolicyId,\n    family: SisModulusFamily,\n    d: u32,\n    coeff_linf_bound: u128,\n) -> Option<&'static [u64]> {{\n    if policy != SisSecurityPolicyId::{} {{\n        return None;\n    }}\n    match family {{\n        SisModulusFamily::Q32 => q32::sis_max_widths(d, coeff_linf_bound),\n        SisModulusFamily::Q64 => q64::sis_max_widths(d, coeff_linf_bound),\n        SisModulusFamily::Q128 => q128::sis_max_widths(d, coeff_linf_bound),\n    }}\n}}\n",
        table_header(policy, profile),
        policy.label(),
    )
}

fn rust_family_source(
    family: AkitaModulusFamily,
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
    arms: &[String],
) -> String {
    let mut source = format!(
        "{}// Family: {}\n\n#[rustfmt::skip]\npub(super) fn sis_max_widths(d: u32, coeff_linf_bound: u128) -> Option<&'static [u64]> {{\n    match (d, coeff_linf_bound) {{\n",
        table_header(policy, profile),
        family.label().to_uppercase()
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
    let classical = policy.classical_constraint();
    let quantum = policy.conventional_quantum_constraint();
    let bcss = policy.idealized_bcss_diagnostic();
    let classical_exponent = model_exponent(classical.reduction_model);
    let quantum_exponent = model_exponent(quantum.reduction_model);
    let bcss_exponent = model_exponent(bcss.reduction_model);
    format!(
        "// AUTO-GENERATED by crates/akita-sis-estimator/examples/infinity_width_table.rs -- do not edit by hand.\n//\n// SIS width thresholds for {}.\n// Hard intersection: ADPS16 classical >= {:.1} and ADPS16 quantum >= {:.1}.\n// Non-gating diagnostic: idealized BCSS23 writable-QRAQM, review below {:.1}.\n// Model exponents: classical {:.4}, conventional quantum {:.4}, BCSS23 idealized {:.4}.\n// All values are log2(rop); each model runs an independent optimizer search.\n// Shape and norm: LGSA, coefficient L-infinity.\n// Rust estimator path: akita-sis-estimator::width_table.\n// Keys are coefficient-L-infinity buckets.\n// Optimizer profile: {}.\n// Every model runs an independent full optimizer search.\n// Local-minimum uses Python-compatible local beta/zeta search inside each row;\n// `--features parallel` parallelizes rows, not the local search itself.\n\n",
        policy.label(),
        classical.minimum_log2_rop,
        quantum.minimum_log2_rop,
        bcss.review_below_log2_rop,
        classical_exponent,
        quantum_exponent,
        bcss_exponent,
        profile.label()
    )
}

fn model_exponent(model: ReductionCostModel) -> f64 {
    match model {
        ReductionCostModel::Adps16 { mode } => adps16_exponent(mode),
        ReductionCostModel::Bcss23Idealized => BCSS23_IDEALIZED_EXPONENT,
        _ => f64::NAN,
    }
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
        "usage: infinity_width_table [--output PATH] [--format csv|rust-split] [--families q32,q64,q128] [--dims 32,64,128,256] [--bounds B1,B2] [--max-rank N] [--search-cap N] [--profile local-minimum|exhaustive-serial|exhaustive-parallel] [--progress-every N] [--skip-validation]"
    );
    process::exit(code);
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2);
}
