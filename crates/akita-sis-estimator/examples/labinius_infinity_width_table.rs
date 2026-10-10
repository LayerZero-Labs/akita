//! Generate the LaBinius binary-source SIS audit rows for the commitment prime.
//!
//! `--from-csv` emits the runtime admission cells from the checked-in certified
//! CSV without running a lattice search. Search mode emits both artifacts.

use akita_challenges::{BinaryChallengeProfile, BinaryScalarRing};
use akita_params::sis::{
    labinius::{LabiniusCommitmentModulus, LabiniusRingDegree, SourceOccurrenceBound},
    source_comparison_inf_norm,
};
use akita_sis_estimator::{
    labinius_width_table::certified_rows,
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthOrigin,
        InfinityWidthRow, InfinityWidthTableConfig, INFINITY_WIDTH_EVALUATOR_ID,
    },
    AkitaModulusProfileId,
};
use sha3::{Digest, Sha3_256};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

const MODULUS_PROFILE: AkitaModulusProfileId = AkitaModulusProfileId::Q25Plus14561;
const CSV_NAME: &str = "labinius_commitment_prime_infinity_width.csv";
const CSV_BYTES: &[u8] = include_bytes!("../data/labinius_commitment_prime_infinity_width.csv");

/// Source of the table: the fold response of the bounded weight-46 challenge
/// whose three balanced base-16 digits span a diameter of `2^12 - 1`.
struct Source;

impl Source {
    const LABEL: &'static str = "phi243-bounded-w46-delta12";
    const SCALAR_RING: BinaryScalarRing = BinaryScalarRing::Cyclotomic243;
    const CHALLENGE_WEIGHT: usize = 46;
    const ACCEPTED_RESPONSE_DIAMETER: u128 = (1 << 12) - 1;

    fn scalar_degree() -> u32 {
        Self::SCALAR_RING.degree() as u32
    }

    fn challenge_profile() -> BinaryChallengeProfile {
        BinaryChallengeProfile::bounded_weight(Self::SCALAR_RING, Self::CHALLENGE_WEIGHT)
            .unwrap_or_else(|error| fatal(&format!("invalid binary source profile: {error}")))
    }

    fn collision_bound() -> u64 {
        let occurrence = SourceOccurrenceBound::binary_extracted(
            u128::from(Self::challenge_profile().multiplication_linf_operator_bound()),
            Self::ACCEPTED_RESPONSE_DIAMETER,
        )
        .unwrap_or_else(|| fatal("source occurrence bound overflow"));
        let eta = source_comparison_inf_norm(
            occurrence.numerator_bound(),
            occurrence.slack_operator_bound(),
            occurrence.numerator_bound(),
            occurrence.slack_operator_bound(),
        )
        .unwrap_or_else(|| fatal("source collision bound overflow"));
        if num_bigint::BigUint::from(eta) >= MODULUS_PROFILE.modulus() {
            fatal("source collision bound does not satisfy eta < q");
        }
        u64::try_from(eta).unwrap_or_else(|_| fatal("source collision bound exceeds u64"))
    }
}

struct Args {
    output: PathBuf,
    rust_output: PathBuf,
    from_csv: bool,
    dims: Vec<u32>,
    max_rank: u32,
    search_cap: Option<u64>,
}

fn main() {
    let args = Args::parse();
    if args.from_csv {
        let rows = certified_rows()
            .unwrap_or_else(|error| fatal(&format!("CSV certificate validation failed: {error}")));
        write_runtime_table(&args.rust_output, rows, CSV_BYTES);
        return;
    }
    let coeff_linf_bound = Source::collision_bound();
    let config = InfinityWidthTableConfig {
        profiles: vec![MODULUS_PROFILE],
        ring_dims: args.dims.clone(),
        coeff_linf_bounds: vec![coeff_linf_bound],
        max_rank: args.max_rank,
        search_cap: args.search_cap,
        explicit_origins: Some(
            args.dims
                .iter()
                .map(|&d| InfinityWidthOrigin {
                    modulus_profile: MODULUS_PROFILE,
                    d,
                    coeff_linf_bound,
                })
                .collect(),
        ),
        ..InfinityWidthTableConfig::default()
    };

    let started = Instant::now();
    let rows = generate_infinity_width_rows(&config)
        .unwrap_or_else(|error| fatal(&format!("generation failed: {error}")));
    validate_infinity_width_rows(&rows)
        .unwrap_or_else(|error| fatal(&format!("certificate validation failed: {error}")));
    for row in &rows {
        eprintln!("candidate {}", row.to_csv_record());
    }
    let certified_rows = rows
        .iter()
        .filter(|row| row.max_width > 0)
        .cloned()
        .collect::<Vec<_>>();
    let mut output = format!(
        "estimator_id,source_profile,scalar_degree,packing_degree,{}\n",
        InfinityWidthRow::csv_header()
    );
    for row in &certified_rows {
        output.push_str(&format!(
            "{INFINITY_WIDTH_EVALUATOR_ID},{},{},{},{}\n",
            Source::LABEL,
            Source::scalar_degree(),
            row.d / Source::scalar_degree(),
            row.to_csv_record()
        ));
    }
    // Build the runtime table first: an empty selection leaves both files alone.
    let runtime =
        runtime_table(&certified_rows, output.as_bytes()).unwrap_or_else(|error| fatal(&error));
    write_file(&args.output, &output);
    eprintln!(
        "wrote {} certified LaBinius row(s) to {} in {:.3}s; omitted {} zero-width candidate(s)",
        certified_rows.len(),
        args.output.display(),
        started.elapsed().as_secs_f64(),
        rows.len() - certified_rows.len()
    );
    write_file(&args.rust_output, &runtime);
    eprintln!(
        "wrote {} certified runtime LaBinius cell(s) to {}",
        certified_rows.len(),
        args.rust_output.display()
    );
}

fn runtime_table(rows: &[InfinityWidthRow], csv: &[u8]) -> Result<String, String> {
    if rows.is_empty() {
        return Err("no row was certified; both outputs were left unchanged".to_owned());
    }
    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| (row.d, row.rank, row.coeff_linf_bound, row.max_width));
    let mut output = format!(
        "// AUTO-GENERATED by labinius_infinity_width_table --from-csv -- do not edit by hand.\n\
         // Source: crates/akita-sis-estimator/data/{CSV_NAME}\n\n\
         use super::width_table::{{LabiniusWidthCell, LabiniusWidthCutoff}};\n\
         use super::{{LabiniusCommitmentModulus, LabiniusRingDegree}};\n\n\
         #[rustfmt::skip]\n\
         pub const LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE: &[LabiniusWidthCell] = &[\n",
    );
    for row in rows {
        if row.modulus_profile != MODULUS_PROFILE {
            return Err(format!(
                "modulus profile {:?} is not the commitment prime",
                row.modulus_profile
            ));
        }
        output.push_str(&format!(
            "    LabiniusWidthCell {{ commitment_modulus: LabiniusCommitmentModulus::{:?}, ring_degree: LabiniusRingDegree::{:?}, rank: {}, coeff_linf_bound: {}, max_width: {}, cutoff: LabiniusWidthCutoff::{} }},\n",
            LabiniusCommitmentModulus::Q25Plus14561,
            ring_degree(row.d)?,
            row.rank,
            row.coeff_linf_bound,
            row.max_width,
            if row.hit_cap { "SearchCap" } else { "Exact" },
        ));
    }
    let digest: [u8; 32] = Sha3_256::digest(csv).into();
    output.push_str(&format!(
        "];\n\n#[rustfmt::skip]\npub const LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE_DIGEST: [u8; 32] = {digest:?};\n"
    ));
    Ok(output)
}

/// Commitment degrees over the source's scalar ring of degree 162.
fn ring_degree(degree: u32) -> Result<LabiniusRingDegree, String> {
    match degree {
        162 => Ok(LabiniusRingDegree::D162),
        324 => Ok(LabiniusRingDegree::D324),
        648 => Ok(LabiniusRingDegree::D648),
        _ => Err(format!(
            "unsupported ring degree {degree}; use 162, 324 or 648"
        )),
    }
}

fn write_runtime_table(path: &Path, rows: &[InfinityWidthRow], csv: &[u8]) {
    let runtime = runtime_table(rows, csv).unwrap_or_else(|error| fatal(&error));
    write_file(path, &runtime);
    eprintln!(
        "wrote {} certified runtime LaBinius cell(s) to {}",
        rows.len(),
        path.display()
    );
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fatal(&format!("create output directory failed: {error}")));
    }
    fs::write(path, contents)
        .unwrap_or_else(|error| fatal(&format!("write {} failed: {error}", path.display())));
}

impl Args {
    fn parse() -> Self {
        let mut parsed = Self {
            output: Path::new("crates/akita-sis-estimator/data").join(CSV_NAME),
            rust_output: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../akita-params/src/sis/labinius/generated_commitment_prime_width_table.rs"),
            from_csv: false,
            dims: vec![648],
            max_rank: 4,
            search_cap: None,
        };
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                usage(0);
            }
            if arg == "--from-csv" {
                parsed.from_csv = true;
                continue;
            }
            let value = args
                .next()
                .unwrap_or_else(|| fatal(&format!("missing value for {arg}")));
            match arg.as_str() {
                "--output" => parsed.output = PathBuf::from(value),
                "--rust-output" => parsed.rust_output = PathBuf::from(value),
                "--dims" => {
                    parsed.dims = value
                        .split(',')
                        .map(str::trim)
                        .filter(|entry| !entry.is_empty())
                        .map(parse)
                        .collect();
                }
                "--max-rank" => parsed.max_rank = parse(&value),
                "--search-cap" => parsed.search_cap = Some(parse(&value)),
                _ => fatal(&format!("unknown argument {arg}")),
            }
        }
        if parsed.dims.is_empty() || parsed.max_rank == 0 {
            fatal("dims and max rank must be nonempty");
        }
        for &d in &parsed.dims {
            ring_degree(d).unwrap_or_else(|error| fatal(&error));
        }
        parsed
    }
}

fn parse<T: std::str::FromStr>(value: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    value
        .parse()
        .unwrap_or_else(|error| fatal(&format!("invalid value {value:?}: {error:?}")))
}

fn usage(code: i32) -> ! {
    eprintln!(
        "usage: labinius_infinity_width_table [--from-csv] [--rust-output PATH] [--output PATH] [--dims 162,324,648] [--max-rank N] [--search-cap N]"
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

    #[test]
    fn checked_in_runtime_table_is_the_csv_generator_output() {
        assert_eq!(
            runtime_table(certified_rows().unwrap(), CSV_BYTES).unwrap(),
            include_str!(
                "../../akita-params/src/sis/labinius/generated_commitment_prime_width_table.rs"
            )
        );
    }

    /// The table's bound is the extracted bound of the sample fold, and every
    /// checked-in row was searched for it.
    #[test]
    fn source_is_the_sample_fold_response() {
        let response = akita_params::sis::labinius::LabiniusFoldResponse::derive(
            &Source::challenge_profile(),
            256,
            4_096,
            LabiniusRingDegree::D648,
        )
        .unwrap();
        assert_eq!(response.diameter(), Source::ACCEPTED_RESPONSE_DIAMETER);
        assert_eq!(u128::from(Source::collision_bound()), response.eta_a());
        for row in certified_rows().unwrap() {
            assert_eq!(row.coeff_linf_bound, Source::collision_bound());
        }
    }
}
