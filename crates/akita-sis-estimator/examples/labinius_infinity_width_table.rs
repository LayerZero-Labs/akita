//! Generate the staged LaBinius binary-source SIS audit rows.
//!
//! This is intentionally an offline CSV artifact. It does not generate a
//! runtime schedule table because the binary root protocol is not implemented.

use akita_challenges::{BinaryChallengeFamily, BinaryChallengeProfile, BinaryScalarRing};
use akita_sis_estimator::{
    width_table::{
        generate_infinity_width_rows, validate_infinity_width_rows, InfinityWidthOrigin,
        InfinityWidthRow, InfinityWidthTableConfig, INFINITY_WIDTH_EVALUATOR_ID,
    },
    AkitaModulusProfileId,
};
use akita_types::sis::{
    labinius::{LabiniusCoefficientPrime, SourceOccurrenceBound},
    source_comparison_inf_norm,
};
use std::{env, fs, path::PathBuf, process, time::Instant};

const DELTA_32: u64 = (1u64 << 32) - 1;
const DELTA_16: u64 = (1u64 << 16) - 1;

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
    profiles: Vec<AkitaModulusProfileId>,
    dims: Vec<u32>,
    source_profiles: Vec<&'static str>,
    max_rank: u32,
    search_cap: Option<u64>,
    use_default_coverage: bool,
}

fn main() {
    let args = Args::parse();
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

    let mut output = format!(
        "estimator_id,source_profile,scalar_degree,packing_degree,{}\n",
        InfinityWidthRow::csv_header()
    );
    let certified_rows = rows
        .iter()
        .filter(|row| row.max_width > 0 && !row.hit_cap)
        .collect::<Vec<_>>();
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
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fatal(&format!("create output directory failed: {error}")));
    }
    fs::write(&args.output, output)
        .unwrap_or_else(|error| fatal(&format!("write {} failed: {error}", args.output.display())));
    eprintln!(
        "wrote {} certified LaBinius row(s) to {} in {:.3}s",
        certified_rows.len(),
        args.output.display(),
        started.elapsed().as_secs_f64()
    );
    eprintln!(
        "omitted {} zero-width or cap-limited candidate row(s)",
        rows.len() - certified_rows.len()
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
        if eta >= coefficient_prime(modulus_profile).modulus() {
            fatal("source collision bound does not satisfy eta < P");
        }
        u64::try_from(eta).unwrap_or_else(|_| fatal("source collision bound exceeds u64"))
    }
}

fn coefficient_prime(profile: AkitaModulusProfileId) -> LabiniusCoefficientPrime {
    match profile {
        AkitaModulusProfileId::Q64Offset23703 => LabiniusCoefficientPrime::P64Offset23703,
        AkitaModulusProfileId::Q128OffsetA7F7 => LabiniusCoefficientPrime::P128OffsetA7F7,
        _ => fatal("LaBinius tables support only q64-labinius and q128"),
    }
}

impl Args {
    fn parse() -> Self {
        let mut parsed = Self {
            output: PathBuf::from("crates/akita-sis-estimator/data/labinius_infinity_width.csv"),
            profiles: vec![
                AkitaModulusProfileId::Q64Offset23703,
                AkitaModulusProfileId::Q128OffsetA7F7,
            ],
            dims: vec![162, 324, 648, 486, 972, 1_944],
            source_profiles: SOURCE_PROFILES
                .iter()
                .map(|profile| profile.label)
                .collect(),
            max_rank: 3,
            search_cap: None,
            use_default_coverage: true,
        };
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                usage(0);
            }
            let value = args
                .next()
                .unwrap_or_else(|| fatal(&format!("missing value for {arg}")));
            match arg.as_str() {
                "--output" => parsed.output = PathBuf::from(value),
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
        "usage: labinius_infinity_width_table [--output PATH] [--profiles q64-labinius,q128] [--dims 162,324,648,486,972,1944] [--source-profiles phi243-bounded-w46-delta16,phi243-fixed-w47-delta16,phi243-bounded-w46-delta32,phi243-fixed-w47-delta32,phi729-fixed-w25-delta16,phi729-fixed-w25-delta32] [--max-rank N] [--search-cap N]"
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
