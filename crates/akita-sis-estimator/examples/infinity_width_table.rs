use akita_sis_estimator::{
    width_table::{
        generate_infinity_width_row, generate_infinity_width_rows, infinity_width_work_items,
        is_production_infinity_width_table_config, runtime_width_rows,
        validate_infinity_width_rows, InfinityWidthProfile, InfinityWidthRow,
        InfinityWidthTableConfig, InfinityWidthWorkItem, RuntimeWidthRow,
        INFINITY_WIDTH_EVALUATOR_ID, PRODUCTION_CERTIFICATE_DOMAIN,
    },
    work_cache::WorkCache,
    AkitaModulusProfileId,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use sha3::{Digest, Sha3_256};
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
    results_dir: Option<PathBuf>,
    plan_only: bool,
    assemble_only: bool,
    shard: Option<Shard>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shard {
    index: u32,
    count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    Csv,
    RustSplit,
}

fn main() {
    let args = Args::parse_or_exit();
    validate_output_request(args.format, args.skip_validation, &args.config)
        .unwrap_or_else(|error| fatal(error));
    let t0 = Instant::now();
    let rows = if args.plan_only {
        write_plan(&args).unwrap_or_else(|error| fatal(&error));
        return;
    } else if let Some(results_dir) = args.results_dir.as_ref() {
        let cache = WorkCache::new(results_dir);
        let work = infinity_width_work_items(&args.config)
            .unwrap_or_else(|error| fatal(&format!("width-table planning failed: {error}")));
        if !args.assemble_only {
            run_missing_work(&work, &args.config, &cache, args.shard)
                .unwrap_or_else(|error| fatal(&error));
        }
        if args.shard.is_some() {
            report_cache_status(&work, &args.config, &cache).unwrap_or_else(|error| fatal(&error));
            return;
        }
        load_complete_rows(&work, &args.config, &cache).unwrap_or_else(|error| fatal(&error))
    } else {
        generate_infinity_width_rows(&args.config)
            .unwrap_or_else(|error| fatal(&format!("width-table generation failed: {error}")))
    };
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

fn validate_output_request(
    format: OutputFormat,
    skip_validation: bool,
    config: &InfinityWidthTableConfig,
) -> Result<(), &'static str> {
    if format != OutputFormat::RustSplit {
        return Ok(());
    }
    if skip_validation {
        return Err("rust-split output requires table validation");
    }
    if !is_production_infinity_width_table_config(config) {
        return Err(
            "rust-split output requires the complete certified production table config; use CSV for comparison profiles or partial jobs",
        );
    }
    Ok(())
}

impl Args {
    fn parse_or_exit() -> Self {
        let mut config = InfinityWidthTableConfig::default();
        let mut output = None;
        let mut format = OutputFormat::Csv;
        let mut skip_validation = false;
        let mut results_dir = None;
        let mut plan_only = false;
        let mut assemble_only = false;
        let mut shard = None;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => usage(0),
                "--skip-validation" => skip_validation = true,
                "--plan-only" => plan_only = true,
                "--assemble-only" => assemble_only = true,
                _ => {
                    let value = args
                        .next()
                        .unwrap_or_else(|| fatal(&format!("missing value for {arg}")));
                    match arg.as_str() {
                        "--output" => output = Some(PathBuf::from(value)),
                        "--results-dir" => results_dir = Some(PathBuf::from(value)),
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
                        "--shard" => shard = Some(parse_shard(&value)),
                        _ => fatal(&format!("unknown argument {arg}")),
                    }
                }
            }
        }
        if plan_only && assemble_only {
            fatal("--plan-only and --assemble-only are mutually exclusive");
        }
        if (assemble_only || shard.is_some() && !plan_only) && results_dir.is_none() {
            fatal("--assemble-only and shard execution require --results-dir");
        }
        if assemble_only && shard.is_some() {
            fatal("--assemble-only and --shard are mutually exclusive");
        }
        Self {
            config,
            output,
            format,
            skip_validation,
            results_dir,
            plan_only,
            assemble_only,
            shard,
        }
    }
}

fn write_plan(args: &Args) -> Result<(), String> {
    let work = infinity_width_work_items(&args.config)
        .map_err(|error| format!("width-table planning failed: {error}"))?;
    let cache = args.results_dir.as_ref().map(WorkCache::new);
    println!("evaluator_id,work_id,status,modulus_profile,d,rank,coeff_linf_bound");
    for item in work {
        let id = item
            .work_id(&args.config)
            .map_err(|error| format!("work identifier failed: {error}"))?;
        if let Some(shard) = args.shard {
            let assigned = id
                .shard(shard.count)
                .map_err(|error| format!("shard selection failed: {error}"))?;
            if assigned != shard.index {
                continue;
            }
        }
        let status = match cache.as_ref() {
            Some(cache) if cached_row(cache, item, &args.config)?.is_some() => "cached",
            _ => "missing",
        };
        println!(
            "{INFINITY_WIDTH_EVALUATOR_ID},{},{status},{},{},{},{}",
            id.hex(),
            item.modulus_profile.label(),
            item.d,
            item.rank,
            item.coeff_linf_bound,
        );
    }
    Ok(())
}

fn run_missing_work(
    work: &[InfinityWidthWorkItem],
    config: &InfinityWidthTableConfig,
    cache: &WorkCache,
    shard: Option<Shard>,
) -> Result<(), String> {
    let mut missing = Vec::new();
    for &item in work {
        let id = item
            .work_id(config)
            .map_err(|error| format!("work identifier failed: {error}"))?;
        let selected = shard
            .map(|shard| {
                id.shard(shard.count)
                    .map(|index| index == shard.index)
                    .map_err(|error| format!("shard selection failed: {error}"))
            })
            .transpose()?
            .unwrap_or(true);
        if selected && cached_row(cache, item, config)?.is_none() {
            missing.push(item);
        }
    }
    let total = missing.len();
    if total == 0 {
        eprintln!("all selected infinity-width work items are already cached");
        return Ok(());
    }
    eprintln!("evaluating {total} missing infinity-width work item(s)");
    evaluate_missing(&missing, config, cache)?;
    Ok(())
}

#[cfg(feature = "parallel")]
fn evaluate_missing(
    missing: &[InfinityWidthWorkItem],
    config: &InfinityWidthTableConfig,
    cache: &WorkCache,
) -> Result<(), String> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let completed = AtomicUsize::new(0);
    let total = missing.len();
    missing.par_iter().try_for_each(|&item| {
        evaluate_and_store(item, config, cache)?;
        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
        report_cached_progress(config.progress_every, done, total);
        Ok(())
    })
}

#[cfg(not(feature = "parallel"))]
fn evaluate_missing(
    missing: &[InfinityWidthWorkItem],
    config: &InfinityWidthTableConfig,
    cache: &WorkCache,
) -> Result<(), String> {
    for (index, &item) in missing.iter().enumerate() {
        evaluate_and_store(item, config, cache)?;
        report_cached_progress(config.progress_every, index + 1, missing.len());
    }
    Ok(())
}

fn evaluate_and_store(
    item: InfinityWidthWorkItem,
    config: &InfinityWidthTableConfig,
    cache: &WorkCache,
) -> Result<(), String> {
    let row = generate_infinity_width_row(item, config).map_err(|error| {
        format!(
            "profile={} d={} rank={} bound={}: {error}",
            item.modulus_profile.label(),
            item.d,
            item.rank,
            item.coeff_linf_bound
        )
    })?;
    row.validate_for_work_item(item, config)
        .map_err(|error| format!("generated work result failed validation: {error}"))?;
    cache
        .store(
            item.work_id(config)
                .map_err(|error| format!("work identifier failed: {error}"))?,
            row.to_work_result().as_bytes(),
        )
        .map_err(|error| format!("store work result failed: {error}"))
}

fn cached_row(
    cache: &WorkCache,
    item: InfinityWidthWorkItem,
    config: &InfinityWidthTableConfig,
) -> Result<Option<InfinityWidthRow>, String> {
    let id = item
        .work_id(config)
        .map_err(|error| format!("work identifier failed: {error}"))?;
    let Some(payload) = cache
        .load(id)
        .map_err(|error| format!("load work result failed: {error}"))?
    else {
        return Ok(None);
    };
    let row = InfinityWidthRow::from_work_result(&payload)
        .map_err(|error| format!("decode cached work result {} failed: {error}", id.hex()))?;
    row.validate_for_work_item(item, config)
        .map_err(|error| format!("cached work result {} failed validation: {error}", id.hex()))?;
    Ok(Some(row))
}

fn load_complete_rows(
    work: &[InfinityWidthWorkItem],
    config: &InfinityWidthTableConfig,
    cache: &WorkCache,
) -> Result<Vec<InfinityWidthRow>, String> {
    let mut rows = Vec::with_capacity(work.len());
    let mut missing = Vec::new();
    for &item in work {
        match cached_row(cache, item, config)? {
            Some(row) => rows.push(row),
            None => missing.push(
                item.work_id(config)
                    .map_err(|error| format!("work identifier failed: {error}"))?
                    .hex(),
            ),
        }
    }
    if !missing.is_empty() {
        let sample = missing
            .iter()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "cannot assemble: {} work result(s) are missing; first missing IDs: {sample}",
            missing.len()
        ));
    }
    rows.sort_by_key(|row| (row.modulus_profile, row.coeff_linf_bound, row.d, row.rank));
    Ok(rows)
}

fn report_cache_status(
    work: &[InfinityWidthWorkItem],
    config: &InfinityWidthTableConfig,
    cache: &WorkCache,
) -> Result<(), String> {
    let mut cached = 0;
    for &item in work {
        if cached_row(cache, item, config)?.is_some() {
            cached += 1;
        }
    }
    eprintln!(
        "infinity width work cache: {cached}/{} complete; run remaining shards, then use --assemble-only",
        work.len()
    );
    Ok(())
}

fn report_cached_progress(progress_every: Option<usize>, completed: usize, total: usize) {
    let Some(every) = progress_every.filter(|value| *value > 0) else {
        return;
    };
    if completed == total || completed.is_multiple_of(every) {
        eprintln!("infinity width work progress: {completed}/{total}");
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
    let runtime_rows = runtime_width_rows(rows, config.max_rank).map_err(io::Error::other)?;
    let mut generated_files = Vec::new();
    for modulus_profile in [
        AkitaModulusProfileId::Q32Offset99,
        AkitaModulusProfileId::Q64Offset59,
        AkitaModulusProfileId::Q128OffsetA7F7,
    ] {
        generated_files.push((
            format!("{}.rs", modulus_profile.label()),
            rust_modulus_profile_source(
                modulus_profile,
                config.policy,
                config.profile,
                runtime_rows
                    .iter()
                    .filter(|row| row.modulus_profile == modulus_profile),
            ),
        ));
    }
    generated_files.push(("policy_audit.csv".to_string(), policy_audit_source(rows)));
    generated_files.push((
        "policy_review.txt".to_string(),
        policy_review_source(rows, config),
    ));
    let base_digest = table_digest(&generated_files)?;
    fs::write(
        out_dir.join("mod.rs"),
        rust_mod_source(config.policy, config.profile, base_digest),
    )?;
    for (filename, contents) in generated_files {
        fs::write(out_dir.join(filename), contents)?;
    }
    Ok(())
}

fn rust_mod_source(
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
    base_digest: [u8; 32],
) -> String {
    format!(
        "{}mod q128;\nmod q32;\nmod q64;\n\nuse super::{{SisModulusProfileId, SisSecurityPolicyId}};\n\n/// SHA3-256 identity of the generated table and provenance files.\npub(super) const SIS_TABLE_DIGEST: [u8; 32] = {};\n\n/// Generated SIS max-width table for the named security policy.\n///\n/// Runtime keys are `(d, coeff_linf_bound) -> widths[rank - 1]`, projected from\n/// scalar cutoffs as `width = cutoff_m(B, n = rank * d) / d`.\n#[rustfmt::skip]\npub(crate) fn sis_max_widths(\n    policy: SisSecurityPolicyId,\n    modulus_profile: SisModulusProfileId,\n    d: u32,\n    coeff_linf_bound: u128,\n) -> Option<&'static [u64]> {{\n    if policy != SisSecurityPolicyId::{} {{\n        return None;\n    }}\n    match modulus_profile {{\n        SisModulusProfileId::Q32Offset99 => q32::sis_max_widths(d, coeff_linf_bound),\n        SisModulusProfileId::Q64Offset59 => q64::sis_max_widths(d, coeff_linf_bound),\n        SisModulusProfileId::Q128OffsetA7F7 => q128::sis_max_widths(d, coeff_linf_bound),\n    }}\n}}\n",
        table_header(policy, profile),
        rust_byte_array(base_digest),
        policy.label(),
    )
}

fn rust_modulus_profile_source<'a>(
    modulus_profile: AkitaModulusProfileId,
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
    rows: impl Iterator<Item = &'a RuntimeWidthRow>,
) -> String {
    let mut source = format!(
        "{}// Profile: {}\n\n#[rustfmt::skip]\npub(super) fn sis_max_widths(d: u32, coeff_linf_bound: u128) -> Option<&'static [u64]> {{\n    match (d, coeff_linf_bound) {{\n",
        table_header(policy, profile),
        modulus_profile.label()
    );
    for row in rows {
        let widths = row
            .widths
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        source.push_str(&format!(
            "        ({}, {}) => Some(&[{}]),\n",
            row.d, row.coeff_linf_bound, widths
        ));
    }
    source.push_str("        _ => None,\n    }\n}\n");
    source
}

fn table_header(
    policy: akita_sis_estimator::SisSecurityPolicy,
    profile: InfinityWidthProfile,
) -> String {
    format!(
        "// AUTO-GENERATED by crates/akita-sis-estimator/examples/infinity_width_table.rs -- do not edit by hand.\n//\n// SIS max-width tables for {}.\n// Sole hard gate: ADPS16 quantum LGSA model >= 128 bits.\n// Each row records the selected optimizer's accepted cutoff and rejected successor.\n// Shape and norm: LGSA, coefficient L-infinity.\n// Rust estimator path: akita-sis-estimator::width_table.\n// Runtime keys are (d, coefficient-L-infinity bound) -> widths by module rank.\n// Values are projected from scalar cutoffs: width[r-1] = cutoff_m(B, n=r*d) / d.\n// Optimizer profile: {}.\n\n",
        policy.label(),
        profile.label()
    )
}

fn policy_audit_source(rows: &[InfinityWidthRow]) -> String {
    let mut source = String::from(InfinityWidthRow::csv_header());
    source.push('\n');
    for row in rows {
        source.push_str(&row.to_csv_record());
        source.push('\n');
    }
    source
}

fn policy_review_source(rows: &[InfinityWidthRow], config: &InfinityWidthTableConfig) -> String {
    let cap_hits = rows.iter().filter(|row| row.hit_cap).count();
    let mut source = format!(
        "Akita SIS generated-table policy review\n\
policy={}\n\
estimator_revision={}\n\
optimizer_profile={}\n\
certificate_domain={}\n\
modulus_profiles=q32:2^32-99,q64:2^64-59,q128:2^128-(2^32-22537)\n\
norm=coefficient-l-infinity\n\
shape=LGSA\n\
adps16_quantum_exponent=0.2650\n\
target_log2_rop={:.1}\n\
max_module_rank={}\n\
search_cap={}\n\
generated_ring_origin_rows={}\n\
exact_boundaries={}\n\
cap_hits={}\n\
monotonicity=validated across rank and coefficient-bound axes\n\
structured_attack_review=The scalarized estimate does not model ring/module structure, CRT splitting, subfield projection, or role-specific matrix structure. No extra numerical adjustment is applied; public claims remain explicitly limited to the scalarized ADPS16 quantum LGSA attack model.\n",
        config.policy.label(),
        INFINITY_WIDTH_EVALUATOR_ID,
        config.profile.label(),
        PRODUCTION_CERTIFICATE_DOMAIN,
        config
            .policy
            .adps16_quantum_constraint()
            .minimum_log2_rop,
        config.max_rank,
        config
            .search_cap
            .unwrap_or(akita_sis_estimator::width_table::DEFAULT_SEARCH_CAP),
        rows.len(),
        rows.len() - cap_hits,
        cap_hits,
    );
    source.push_str("role_coverage_columns=role,modulus_profile,d,coeff_linf_bound,max_module_rank,required_max_width\n");
    for cell in akita_types::sis::sis_role_cells() {
        source.push_str(&format!(
            "role_coverage={},{},{},{},{},{}\n",
            cell.role.name(),
            cell.modulus_profile.name(),
            cell.ring_dimension,
            cell.coeff_linf_bound,
            cell.max_module_rank,
            cell.required_max_width,
        ));
    }
    source
}

fn table_digest(files: &[(String, String)]) -> io::Result<[u8; 32]> {
    const DOMAIN: &[u8] = b"akita-sis-table-digest-adps16-quantum-128bit\0";
    let mut hasher = Sha3_256::new();
    hasher.update(DOMAIN);
    for required_name in [
        "q32.rs",
        "q64.rs",
        "q128.rs",
        "policy_audit.csv",
        "policy_review.txt",
    ] {
        let contents = files
            .iter()
            .find_map(|(name, contents)| (name == required_name).then_some(contents.as_bytes()))
            .ok_or_else(|| io::Error::other(format!("missing generated file {required_name}")))?;
        hasher.update(
            u64::try_from(contents.len())
                .map_err(io::Error::other)?
                .to_le_bytes(),
        );
        hasher.update(required_name.as_bytes());
        hasher.update([0]);
        hasher.update(contents);
    }
    Ok(hasher.finalize().into())
}

fn rust_byte_array(bytes: [u8; 32]) -> String {
    let mut output = String::from("[\n");
    for chunk in bytes.chunks(16) {
        output.push_str("    ");
        output.push_str(
            &chunk
                .iter()
                .map(|byte| format!("0x{byte:02x}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        output.push_str(",\n");
    }
    output.push(']');
    output
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
        "lattice-estimator" | "lattice_estimator" => {
            InfinityWidthProfile::LatticeEstimatorParity
        }
        "exhaustive-serial" | "exhaustive_serial" => InfinityWidthProfile::ExhaustiveSerial,
        "exhaustive-parallel" | "exhaustive_parallel" => {
            #[cfg(not(feature = "parallel"))]
            fatal("profile exhaustive-parallel requires building with --features parallel");
            #[cfg(feature = "parallel")]
            {
                InfinityWidthProfile::ExhaustiveParallel
            }
        }
        _ => fatal(
            "profile must be one of: local-minimum, lattice-estimator, exhaustive-serial, exhaustive-parallel",
        ),
    }
}

fn parse_format(value: &str) -> OutputFormat {
    match value {
        "csv" => OutputFormat::Csv,
        "rust-split" | "rust_split" => OutputFormat::RustSplit,
        _ => fatal("--format must be one of: csv, rust-split"),
    }
}

fn parse_shard(value: &str) -> Shard {
    let Some((index, count)) = value.split_once('/') else {
        fatal("--shard must use one-based INDEX/COUNT syntax, for example 3/16");
    };
    let index: u32 = parse(index, "--shard index");
    let count: u32 = parse(count, "--shard count");
    if count == 0 || index == 0 || index > count {
        fatal("--shard requires 1 <= INDEX <= COUNT");
    }
    Shard {
        index: index - 1,
        count,
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
        "usage: infinity_width_table [--output PATH] [--format csv|rust-split] [--results-dir PATH] [--plan-only|--assemble-only|--shard INDEX/COUNT] [--profiles q32,q64,q128] [--dims 32,64,128,256,512] [--bounds B1,B2] [--max-rank N] [--search-cap N] [--profile local-minimum|lattice-estimator|exhaustive-serial|exhaustive-parallel] [--progress-every N] [--skip-validation]"
    );
    process::exit(code);
}

fn fatal(message: &str) -> ! {
    eprintln!("error: {message}");
    process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

    fn test_results_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "akita-infinity-work-results-{}-{}",
            std::process::id(),
            TEST_NONCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn production_rust_output_requires_the_certified_profile() {
        let mut config = InfinityWidthTableConfig::default();
        config.profile = InfinityWidthProfile::LatticeEstimatorParity;
        assert!(validate_output_request(OutputFormat::RustSplit, false, &config).is_err());
        assert!(validate_output_request(OutputFormat::Csv, false, &config).is_ok());
    }

    #[test]
    fn production_rust_output_requires_validation() {
        let config = InfinityWidthTableConfig::default();
        assert!(validate_output_request(OutputFormat::RustSplit, true, &config).is_err());
        assert!(validate_output_request(OutputFormat::RustSplit, false, &config).is_ok());
    }

    #[test]
    fn policy_review_records_the_canonical_certificate_domain() {
        let source = policy_review_source(&[], &InfinityWidthTableConfig::default());
        assert!(source.contains(&format!(
            "certificate_domain={PRODUCTION_CERTIFICATE_DOMAIN}\n"
        )));
        assert!(source.contains(&format!(
            "estimator_revision={INFINITY_WIDTH_EVALUATOR_ID}\n"
        )));
    }

    #[test]
    fn cached_generation_resumes_and_assembles_exact_rows() {
        let root = test_results_dir();
        let cache = WorkCache::new(&root);
        let config = InfinityWidthTableConfig {
            profiles: vec![AkitaModulusProfileId::Q32Offset99],
            ring_dims: vec![64],
            coeff_linf_bounds: vec![2],
            max_rank: 1,
            search_cap: Some(4),
            profile: InfinityWidthProfile::LatticeEstimatorParity,
            ..InfinityWidthTableConfig::default()
        };
        let work = infinity_width_work_items(&config).unwrap();
        assert!(load_complete_rows(&work, &config, &cache).is_err());
        run_missing_work(&work, &config, &cache, None).unwrap();
        let cached = load_complete_rows(&work, &config, &cache).unwrap();
        assert_eq!(cached, generate_infinity_width_rows(&config).unwrap());
        run_missing_work(&work, &config, &cache, None).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
