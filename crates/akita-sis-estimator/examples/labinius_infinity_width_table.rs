//! Generate the staged LaBinius binary-source SIS audit rows.
//!
//! `--from-csv` emits the runtime admission cells from the checked-in certified
//! CSV without running a lattice search. Search mode emits both artifacts.

use akita_challenges::{BinaryChallengeFamily, BinaryChallengeProfile, BinaryScalarRing};
use akita_params::sis::{
    labinius::{LabiniusCoefficientPrime, LabiniusRingDegree, SourceOccurrenceBound},
    source_comparison_inf_norm,
};
use akita_sis_estimator::{
    labinius_width_table::{certified_rows, certified_small_modulus_rows},
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

const DELTA_32: u64 = (1u64 << 32) - 1;
const DELTA_16: u64 = (1u64 << 16) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TableKind {
    SharedPrime,
    SmallModulus,
}

impl TableKind {
    fn from_profiles(profiles: &[AkitaModulusProfileId]) -> Self {
        if profiles == [AkitaModulusProfileId::Q28Offset2103] {
            Self::SmallModulus
        } else {
            Self::SharedPrime
        }
    }
}

#[derive(Clone, Copy)]
struct SourceProfile {
    label: &'static str,
    scalar_ring: BinaryScalarRing,
    challenge_family: BinaryChallengeFamily,
    challenge_weight: usize,
    accepted_response_diameter: u128,
}

const SOURCE_PROFILES: &[SourceProfile] = &[
    SourceProfile {
        label: "phi243-bounded-w46-delta16",
        scalar_ring: BinaryScalarRing::Cyclotomic243,
        challenge_family: BinaryChallengeFamily::BoundedWeight,
        challenge_weight: 46,
        accepted_response_diameter: DELTA_16 as u128,
    },
    SourceProfile {
        label: "phi243-fixed-w47-delta16",
        scalar_ring: BinaryScalarRing::Cyclotomic243,
        challenge_family: BinaryChallengeFamily::FixedWeight,
        challenge_weight: 47,
        accepted_response_diameter: DELTA_16 as u128,
    },
    SourceProfile {
        label: "phi243-bounded-w46-delta32",
        scalar_ring: BinaryScalarRing::Cyclotomic243,
        challenge_family: BinaryChallengeFamily::BoundedWeight,
        challenge_weight: 46,
        accepted_response_diameter: DELTA_32 as u128,
    },
    SourceProfile {
        label: "phi243-fixed-w47-delta32",
        scalar_ring: BinaryScalarRing::Cyclotomic243,
        challenge_family: BinaryChallengeFamily::FixedWeight,
        challenge_weight: 47,
        accepted_response_diameter: DELTA_32 as u128,
    },
    SourceProfile {
        label: "phi729-fixed-w25-delta16",
        scalar_ring: BinaryScalarRing::Cyclotomic729,
        challenge_family: BinaryChallengeFamily::FixedWeight,
        challenge_weight: 25,
        accepted_response_diameter: DELTA_16 as u128,
    },
    SourceProfile {
        label: "phi729-fixed-w25-delta32",
        scalar_ring: BinaryScalarRing::Cyclotomic729,
        challenge_family: BinaryChallengeFamily::FixedWeight,
        challenge_weight: 25,
        accepted_response_diameter: DELTA_32 as u128,
    },
];

struct Args {
    output: PathBuf,
    rust_output: PathBuf,
    from_csv: bool,
    profiles: Vec<AkitaModulusProfileId>,
    table_kind: TableKind,
    dims: Vec<u32>,
    source_profiles: Vec<&'static str>,
    max_rank: u32,
    search_cap: Option<u64>,
    use_default_coverage: bool,
}

fn main() {
    let args = Args::parse();
    if args.from_csv {
        let rows = if args.table_kind == TableKind::SmallModulus {
            certified_small_modulus_rows()
        } else {
            certified_rows()
        }
        .unwrap_or_else(|error| fatal(&format!("CSV certificate validation failed: {error}")));
        let csv: &[u8] = if args.table_kind == TableKind::SmallModulus {
            include_bytes!("../data/labinius_small_modulus_infinity_width.csv")
        } else {
            include_bytes!("../data/labinius_infinity_width.csv")
        };
        write_runtime_table(&args.rust_output, args.table_kind, rows, csv);
        return;
    }
    let specs = selected_specs(&args);
    let explicit_origins = specs
        .iter()
        .map(|(modulus_profile, d, source)| InfinityWidthOrigin {
            modulus_profile: *modulus_profile,
            d: *d,
            coeff_linf_bound: source.collision_bound(*modulus_profile),
        })
        .collect::<Vec<_>>();
    let config = InfinityWidthTableConfig {
        profiles: args.profiles.clone(),
        ring_dims: args.dims.clone(),
        coeff_linf_bounds: specs
            .iter()
            .map(|(modulus_profile, _d, source)| source.collision_bound(*modulus_profile))
            .collect(),
        max_rank: args.max_rank,
        search_cap: args.search_cap,
        explicit_origins: Some(explicit_origins),
        ..InfinityWidthTableConfig::default()
    };

    let started = Instant::now();
    let rows = generate_infinity_width_rows(&config)
        .unwrap_or_else(|error| fatal(&format!("generation failed: {error}")));
    validate_infinity_width_rows(&rows)
        .unwrap_or_else(|error| fatal(&format!("certificate validation failed: {error}")));

    if args.table_kind == TableKind::SmallModulus {
        for row in &rows {
            eprintln!("candidate {}", row.to_csv_record());
        }
    }
    let certified_rows = rows
        .iter()
        .filter(|row| {
            row.max_width > 0 && (!row.hit_cap || args.table_kind == TableKind::SmallModulus)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut output = format!(
        "estimator_id,source_profile,scalar_degree,packing_degree,{}\n",
        InfinityWidthRow::csv_header()
    );
    for row in &certified_rows {
        let source = source_for_row(row.modulus_profile, row.d, row.coeff_linf_bound);
        output.push_str(&format!(
            "{INFINITY_WIDTH_EVALUATOR_ID},{},{},{},{}\n",
            source.label,
            source.scalar_degree(),
            row.d / source.scalar_degree(),
            row.to_csv_record()
        ));
    }
    let runtime = runtime_table(args.table_kind, &certified_rows, output.as_bytes())
        .unwrap_or_else(|error| fatal(&error));
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fatal(&format!("create output directory failed: {error}")));
    }
    fs::write(&args.output, &output)
        .unwrap_or_else(|error| fatal(&format!("write {} failed: {error}", args.output.display())));
    eprintln!(
        "wrote {} certified LaBinius row(s) to {} in {:.3}s",
        certified_rows.len(),
        args.output.display(),
        started.elapsed().as_secs_f64()
    );
    eprintln!(
        "omitted {} unpublished candidate row(s)",
        rows.len() - certified_rows.len()
    );
    if let Some(parent) = args.rust_output.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fatal(&format!("create runtime directory failed: {error}")));
    }
    fs::write(&args.rust_output, runtime).unwrap_or_else(|error| {
        fatal(&format!(
            "write {} failed: {error}",
            args.rust_output.display()
        ))
    });
    eprintln!(
        "wrote {} certified runtime LaBinius cell(s) to {}",
        certified_rows.len(),
        args.rust_output.display()
    );
}

fn runtime_table(kind: TableKind, rows: &[InfinityWidthRow], csv: &[u8]) -> Result<String, String> {
    if rows.is_empty() {
        return Err("no row was certified; both outputs were left unchanged".to_owned());
    }
    let small_modulus = kind == TableKind::SmallModulus;
    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| {
        (
            row.modulus_profile,
            row.d,
            row.rank,
            row.coeff_linf_bound,
            row.max_width,
        )
    });
    let (source, column, modulus_type, cell_type, cell_import, table_name) = if small_modulus {
        (
            "labinius_small_modulus_infinity_width.csv",
            "commitment_modulus",
            "LabiniusCommitmentModulus",
            "LabiniusSmallModulusWidthCell",
            "{LabiniusSmallModulusWidthCell, LabiniusWidthCutoff}",
            "LABINIUS_SMALL_MODULUS_WIDTH_TABLE",
        )
    } else {
        (
            "labinius_infinity_width.csv",
            "coefficient_prime",
            "LabiniusCoefficientPrime",
            "LabiniusWidthCell",
            "LabiniusWidthCell",
            "LABINIUS_WIDTH_TABLE",
        )
    };
    let mut output = format!(
        "// AUTO-GENERATED by labinius_infinity_width_table --from-csv -- do not edit by hand.\n\
         // Source: crates/akita-sis-estimator/data/{source}\n\n\
         use super::width_table::{cell_import};\n\
         use super::{{{modulus_type}, LabiniusRingDegree}};\n\n\
         #[rustfmt::skip]\n\
         pub const {table_name}: &[{cell_type}] = &[\n",
    );
    for row in rows {
        let modulus_variant = if small_modulus {
            if row.modulus_profile != AkitaModulusProfileId::Q28Offset2103 {
                return Err(format!(
                    "modulus profile {:?} does not belong to the small-modulus table",
                    row.modulus_profile
                ));
            }
            "Q28Offset2103".to_owned()
        } else {
            format!("{:?}", coefficient_prime(row.modulus_profile)?)
        };
        let cutoff = if small_modulus {
            format!(
                ", cutoff: LabiniusWidthCutoff::{}",
                if row.hit_cap { "SearchCap" } else { "Exact" }
            )
        } else {
            String::new()
        };
        output.push_str(&format!(
            "    {cell_type} {{ {column}: {modulus_type}::{modulus_variant}, ring_degree: LabiniusRingDegree::{:?}, rank: {}, coeff_linf_bound: {}, max_width: {}{cutoff} }},\n",
            ring_degree(row.d),
            row.rank,
            row.coeff_linf_bound,
            row.max_width,
        ));
    }
    let digest: [u8; 32] = Sha3_256::digest(csv).into();
    output.push_str(&format!(
        "];\n\n#[rustfmt::skip]\npub const {table_name}_DIGEST: [u8; 32] = {digest:?};\n"
    ));
    Ok(output)
}

fn ring_degree(degree: u32) -> LabiniusRingDegree {
    match degree {
        162 => LabiniusRingDegree::D162,
        324 => LabiniusRingDegree::D324,
        648 => LabiniusRingDegree::D648,
        486 => LabiniusRingDegree::D486,
        972 => LabiniusRingDegree::D972,
        1_944 => LabiniusRingDegree::D1944,
        _ => fatal("unsupported LaBinius ring degree"),
    }
}

fn write_runtime_table(path: &Path, kind: TableKind, rows: &[InfinityWidthRow], csv: &[u8]) {
    let runtime = runtime_table(kind, rows, csv).unwrap_or_else(|error| fatal(&error));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fatal(&format!("create runtime directory failed: {error}")));
    }
    fs::write(path, runtime)
        .unwrap_or_else(|error| fatal(&format!("write {} failed: {error}", path.display())));
    eprintln!(
        "wrote {} certified runtime LaBinius cell(s) to {}",
        rows.len(),
        path.display()
    );
}

fn selected_specs(args: &Args) -> Vec<(AkitaModulusProfileId, u32, SourceProfile)> {
    let mut specs = Vec::new();
    for &modulus_profile in &args.profiles {
        for &d in &args.dims {
            // P128/D1944 remains secure through the ordinary 6.4e12 search
            // cap even at rank one, so it has no exact rejected successor to
            // publish in this narrow staged artifact.
            if args.use_default_coverage
                && modulus_profile == AkitaModulusProfileId::Q128OffsetA7F7
                && d == 1_944
            {
                continue;
            }
            for &source in SOURCE_PROFILES {
                // These low-bound cells remain secure through the search cap
                // at every selected rank, so they have no exact successor to
                // publish. The delta32 stress cells still cover these degrees.
                if args.use_default_coverage
                    && source.label == "phi729-fixed-w25-delta16"
                    && ((modulus_profile == AkitaModulusProfileId::Q64Offset23703 && d == 1_944)
                        || (modulus_profile == AkitaModulusProfileId::Q128OffsetA7F7 && d == 972))
                {
                    continue;
                }
                if source.scalar_degree() == scalar_degree_for_ring(d)
                    && args.source_profiles.contains(&source.label)
                {
                    specs.push((modulus_profile, d, source));
                }
            }
        }
    }
    if specs.is_empty() {
        fatal("selection produced no LaBinius origins");
    }
    specs
}

fn scalar_degree_for_ring(d: u32) -> u32 {
    match d {
        162 | 324 | 648 => 162,
        486 | 972 | 1_944 => 486,
        _ => fatal("--dims entries must be 162,324,648,486,972,1944"),
    }
}

fn source_for_row(modulus_profile: AkitaModulusProfileId, d: u32, bound: u64) -> SourceProfile {
    SOURCE_PROFILES
        .iter()
        .copied()
        .find(|profile| {
            profile.scalar_degree() == scalar_degree_for_ring(d)
                && profile.collision_bound(modulus_profile) == bound
        })
        .unwrap_or_else(|| fatal("generated row has an unknown source bound"))
}

impl SourceProfile {
    fn scalar_degree(self) -> u32 {
        self.scalar_ring.degree() as u32
    }

    fn challenge_profile(self) -> BinaryChallengeProfile {
        let result = match self.challenge_family {
            BinaryChallengeFamily::FixedWeight => {
                BinaryChallengeProfile::fixed_weight(self.scalar_ring, self.challenge_weight)
            }
            BinaryChallengeFamily::BoundedWeight => {
                BinaryChallengeProfile::bounded_weight(self.scalar_ring, self.challenge_weight)
            }
        };
        result.unwrap_or_else(|error| fatal(&format!("invalid binary source profile: {error}")))
    }

    fn collision_bound(self, modulus_profile: AkitaModulusProfileId) -> u64 {
        let occurrence = SourceOccurrenceBound::binary_extracted(
            u128::from(
                self.challenge_profile()
                    .multiplication_linf_operator_bound(),
            ),
            self.accepted_response_diameter,
        )
        .unwrap_or_else(|| fatal("source occurrence bound overflow"));
        let eta = source_comparison_inf_norm(
            occurrence.numerator_bound(),
            occurrence.slack_operator_bound(),
            occurrence.numerator_bound(),
            occurrence.slack_operator_bound(),
        )
        .unwrap_or_else(|| fatal("source collision bound overflow"));
        if num_bigint::BigUint::from(eta) >= modulus_profile.modulus() {
            fatal("source collision bound does not satisfy eta < P");
        }
        u64::try_from(eta).unwrap_or_else(|_| fatal("source collision bound exceeds u64"))
    }
}

fn coefficient_prime(profile: AkitaModulusProfileId) -> Result<LabiniusCoefficientPrime, String> {
    match profile {
        AkitaModulusProfileId::Q64Offset23703 => Ok(LabiniusCoefficientPrime::P64Offset23703),
        AkitaModulusProfileId::Q128OffsetA7F7 => Ok(LabiniusCoefficientPrime::P128OffsetA7F7),
        _ => Err(format!(
            "modulus profile {profile:?} does not belong to the shared-prime table"
        )),
    }
}

impl Args {
    fn parse() -> Self {
        let mut parsed = Self {
            output: PathBuf::from("crates/akita-sis-estimator/data/labinius_infinity_width.csv"),
            rust_output: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../akita-params/src/sis/labinius/generated_width_table.rs"),
            from_csv: false,
            profiles: vec![
                AkitaModulusProfileId::Q64Offset23703,
                AkitaModulusProfileId::Q128OffsetA7F7,
            ],
            table_kind: TableKind::SharedPrime,
            dims: vec![162, 324, 648, 486, 972, 1_944],
            source_profiles: SOURCE_PROFILES
                .iter()
                .map(|profile| profile.label)
                .collect(),
            max_rank: 3,
            search_cap: None,
            use_default_coverage: true,
        };
        let mut explicit = std::collections::BTreeSet::new();
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
            explicit.insert(arg.clone());
            match arg.as_str() {
                "--output" => parsed.output = PathBuf::from(value),
                "--rust-output" => parsed.rust_output = PathBuf::from(value),
                "--profiles" => {
                    parsed.use_default_coverage = false;
                    parsed.profiles = csv(&value)
                        .into_iter()
                        .map(|label| {
                            AkitaModulusProfileId::parse(label)
                                .unwrap_or_else(|error| fatal(&format!("invalid profile: {error}")))
                        })
                        .collect();
                }
                "--dims" => {
                    parsed.use_default_coverage = false;
                    parsed.dims = csv(&value).into_iter().map(parse).collect();
                }
                "--source-profiles" => {
                    parsed.use_default_coverage = false;
                    parsed.source_profiles = csv(&value)
                        .into_iter()
                        .map(|label| {
                            SOURCE_PROFILES
                                .iter()
                                .find_map(|profile| {
                                    (profile.label == label).then_some(profile.label)
                                })
                                .unwrap_or_else(|| fatal("unknown --source-profiles entry"))
                        })
                        .collect();
                }
                "--max-rank" => parsed.max_rank = parse(&value),
                "--search-cap" => {
                    parsed.use_default_coverage = false;
                    parsed.search_cap = Some(parse(&value));
                }
                _ => fatal(&format!("unknown argument {arg}")),
            }
        }
        parsed.table_kind = TableKind::from_profiles(&parsed.profiles);
        if parsed
            .profiles
            .contains(&AkitaModulusProfileId::Q28Offset2103)
        {
            if parsed.table_kind != TableKind::SmallModulus {
                fatal("q28-labinius must be selected alone");
            }
            if !explicit.contains("--output") {
                parsed.output = PathBuf::from(
                    "crates/akita-sis-estimator/data/labinius_small_modulus_infinity_width.csv",
                );
            }
            if !explicit.contains("--rust-output") {
                parsed.rust_output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
                    "../akita-params/src/sis/labinius/generated_small_modulus_width_table.rs",
                );
            }
            if !explicit.contains("--dims") {
                parsed.dims = vec![648];
            }
            if !explicit.contains("--source-profiles") {
                parsed.source_profiles = vec!["phi243-bounded-w46-delta16"];
            }
            if !explicit.contains("--max-rank") {
                parsed.max_rank = 4;
            }
        }
        if parsed.profiles.is_empty()
            || parsed.dims.is_empty()
            || parsed.source_profiles.is_empty()
            || parsed.max_rank == 0
        {
            fatal("profiles, dims, source profiles, and max rank must be nonempty");
        }
        parsed
    }
}

fn csv(value: &str) -> Vec<&str> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .collect()
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
        "usage: labinius_infinity_width_table [--from-csv] [--rust-output PATH] [--output PATH] [--profiles q64-labinius,q128 | q28-labinius] [--dims 162,324,648,486,972,1944] [--source-profiles phi243-bounded-w46-delta16,phi243-fixed-w47-delta16,phi243-bounded-w46-delta32,phi243-fixed-w47-delta32,phi729-fixed-w25-delta16,phi729-fixed-w25-delta32] [--max-rank N] [--search-cap N]"
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
    use std::collections::BTreeSet;

    #[test]
    fn checked_in_runtime_table_is_the_csv_generator_output() {
        assert_eq!(
            runtime_table(
                TableKind::SharedPrime,
                certified_rows().unwrap(),
                include_bytes!("../data/labinius_infinity_width.csv"),
            )
            .unwrap(),
            include_str!("../../akita-params/src/sis/labinius/generated_width_table.rs"),
        );
    }

    #[test]
    fn small_modulus_runtime_table_is_the_csv_generator_output() {
        assert_eq!(
            runtime_table(
                TableKind::SmallModulus,
                certified_small_modulus_rows().unwrap(),
                include_bytes!("../data/labinius_small_modulus_infinity_width.csv")
            )
            .unwrap(),
            include_str!(
                "../../akita-params/src/sis/labinius/generated_small_modulus_width_table.rs"
            )
        );
    }

    #[test]
    fn empty_shared_prime_rows_are_rejected() {
        assert_eq!(
            runtime_table(TableKind::SharedPrime, &[], b"").unwrap_err(),
            "no row was certified; both outputs were left unchanged"
        );
    }

    #[test]
    fn empty_small_modulus_rows_are_rejected() {
        assert_eq!(
            runtime_table(TableKind::SmallModulus, &[], b"").unwrap_err(),
            "no row was certified; both outputs were left unchanged"
        );
    }

    #[test]
    fn shared_prime_table_rejects_small_modulus_rows() {
        assert!(runtime_table(
            TableKind::SharedPrime,
            certified_small_modulus_rows().unwrap(),
            include_bytes!("../data/labinius_small_modulus_infinity_width.csv"),
        )
        .is_err());
    }

    #[test]
    fn small_modulus_table_rejects_shared_prime_rows() {
        assert!(runtime_table(
            TableKind::SmallModulus,
            certified_rows().unwrap(),
            include_bytes!("../data/labinius_infinity_width.csv"),
        )
        .is_err());
    }

    #[test]
    fn table_kind_is_derived_from_selected_profiles() {
        assert_eq!(
            TableKind::from_profiles(&[AkitaModulusProfileId::Q28Offset2103]),
            TableKind::SmallModulus
        );
        assert_eq!(
            TableKind::from_profiles(&[
                AkitaModulusProfileId::Q64Offset23703,
                AkitaModulusProfileId::Q128OffsetA7F7,
            ]),
            TableKind::SharedPrime
        );
    }

    #[test]
    fn runtime_tables_export_only_the_selected_table_constant() {
        let shared = runtime_table(
            TableKind::SharedPrime,
            certified_rows().unwrap(),
            include_bytes!("../data/labinius_infinity_width.csv"),
        )
        .unwrap();
        let small = runtime_table(
            TableKind::SmallModulus,
            certified_small_modulus_rows().unwrap(),
            include_bytes!("../data/labinius_small_modulus_infinity_width.csv"),
        )
        .unwrap();
        assert!(shared.contains("pub const LABINIUS_WIDTH_TABLE:"));
        assert!(!shared.contains("pub const LABINIUS_SMALL_MODULUS_WIDTH_TABLE:"));
        assert!(small.contains("pub const LABINIUS_SMALL_MODULUS_WIDTH_TABLE:"));
        assert!(!small.contains("pub const LABINIUS_WIDTH_TABLE:"));
    }

    #[test]
    fn canonical_bounds_are_the_exact_binary_diagonal_envelopes() {
        let p64 = AkitaModulusProfileId::Q64Offset23703;
        assert_eq!(SOURCE_PROFILES[0].collision_bound(p64), 24_116_880);
        assert_eq!(SOURCE_PROFILES[1].collision_bound(p64), 24_641_160);
        assert_eq!(SOURCE_PROFILES[2].collision_bound(p64), 1_580_547_964_560);
        assert_eq!(SOURCE_PROFILES[3].collision_bound(p64), 1_614_907_702_920);
        assert_eq!(SOURCE_PROFILES[4].collision_bound(p64), 13_107_000);
        assert_eq!(SOURCE_PROFILES[5].collision_bound(p64), 858_993_459_000);
    }

    #[test]
    fn checked_in_rows_use_the_generator_source_bounds() {
        let mut seen = BTreeSet::new();
        let csv = include_str!("../data/labinius_infinity_width.csv");
        for line in csv.lines().skip(1) {
            let mut fields = line.splitn(5, ',');
            assert_eq!(fields.next(), Some(INFINITY_WIDTH_EVALUATOR_ID));
            let label = fields.next().unwrap();
            let source = SOURCE_PROFILES
                .iter()
                .find(|source| source.label == label)
                .unwrap();
            let scalar_degree: u32 = fields.next().unwrap().parse().unwrap();
            let packing_degree: u32 = fields.next().unwrap().parse().unwrap();
            let row = InfinityWidthRow::from_csv_record(fields.next().unwrap()).unwrap();
            assert_eq!(scalar_degree, source.scalar_degree());
            assert_eq!(row.d, scalar_degree * packing_degree);
            assert_eq!(
                row.coeff_linf_bound,
                source.collision_bound(row.modulus_profile)
            );
            seen.insert(label);
        }
        assert_eq!(seen.len(), SOURCE_PROFILES.len());
    }
}
