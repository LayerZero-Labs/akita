//! Deterministic proof-size evidence from checked-in dense schedule catalogs.
//!
//! This tool never invokes the planner. It expands and authenticates the
//! compiled catalog rows, then delegates every size calculation to the same
//! canonical helpers used by generated-row replay.

use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use akita_planner::generated_families::{GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_schedules::planner_support::nonterminal_level_payload_bytes;
use akita_types::{
    AkitaScheduleLookupKey, CommitmentPayloadMode, FoldSchedule, InnerCommitSecurityRoute,
    OpeningMethod,
};
#[cfg(feature = "quotient-free-evidence")]
use akita_types::{QuotientCoefficientBreakdown, RingRelationMode};

const DENSE_FAMILIES: [&str; 3] = ["fp32_dense", "fp64_dense", "fp128_dense"];
const STACK_BASE_SHA: &str = "e473df62baa6f3491fa867c25a4b6237451737d4";

const HEADER: [&str; 23] = [
    "revision",
    "sha",
    "family",
    "num_vars",
    "num_polynomials",
    "logical_key",
    "lookup_key_digest",
    "row_digest",
    "schedule_descriptor_digest",
    "cutover_level",
    "fold_level",
    "relation_mode",
    "input_witness_len",
    "output_witness_len",
    "ordinary_quotient_coefficients_removed",
    "compression_quotient_coefficients_removed",
    "payload_mode",
    "opening_method",
    "security_route",
    "incoming_setup_prefix",
    "direct_payload_bytes",
    "stage3_payload_bytes",
    "total_proof_bytes",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReportArgs {
    revision: String,
    sha: String,
    source_tree: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EvidenceRow {
    revision: String,
    sha: String,
    family: &'static str,
    num_vars: usize,
    num_polynomials: usize,
    logical_key: String,
    lookup_key_digest: String,
    row_digest: String,
    schedule_descriptor_digest: String,
    cutover_level: Option<usize>,
    fold_level: usize,
    relation_mode: &'static str,
    input_witness_len: usize,
    output_witness_len: usize,
    ordinary_quotient_coefficients_removed: usize,
    compression_quotient_coefficients_removed: usize,
    payload_mode: &'static str,
    opening_method: String,
    security_route: &'static str,
    incoming_setup_prefix: bool,
    direct_payload_bytes: usize,
    stage3_payload_bytes: usize,
    total_proof_bytes: usize,
}

impl EvidenceRow {
    fn write_tsv(&self, out: &mut String) {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.revision,
            self.sha,
            self.family,
            self.num_vars,
            self.num_polynomials,
            self.logical_key,
            self.lookup_key_digest,
            self.row_digest,
            self.schedule_descriptor_digest,
            self.cutover_level.map_or_else(|| "none".to_string(), |level| level.to_string()),
            self.fold_level,
            self.relation_mode,
            self.input_witness_len,
            self.output_witness_len,
            self.ordinary_quotient_coefficients_removed,
            self.compression_quotient_coefficients_removed,
            self.payload_mode,
            self.opening_method,
            self.security_route,
            self.incoming_setup_prefix,
            self.direct_payload_bytes,
            self.stage3_payload_bytes,
            self.total_proof_bytes,
        )
        .expect("writing to String cannot fail");
    }
}

#[cfg(feature = "quotient-free-evidence")]
fn usage() -> &'static str {
    "usage: cargo run --release -p akita-planner \\
     --features catalog-check,catalog-evidence,quotient-free-evidence \\
     --example quotient_free_catalog_evidence -- \\
     --revision head --sha <40-hex-head-commit> \\
     --source-tree <checked-out-repository>"
}

#[cfg(not(feature = "quotient-free-evidence"))]
fn usage() -> &'static str {
    "usage: cargo run --release -p akita-planner \\
     --features catalog-check,catalog-evidence \\
     --example quotient_free_catalog_evidence -- \\
     --revision base --sha e473df62baa6f3491fa867c25a4b6237451737d4 \\
     --source-tree <checked-out-repository>"
}

fn parse_args(raw: impl IntoIterator<Item = String>) -> Result<ReportArgs, String> {
    let mut revision = None;
    let mut sha = None;
    let mut source_tree = None;
    let mut args = raw.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--revision" => {
                if revision.is_some() {
                    return Err("--revision may be supplied only once".to_string());
                }
                revision = Some(
                    args.next()
                        .ok_or_else(|| format!("--revision requires a value\n{}", usage()))?,
                );
            }
            "--sha" => {
                if sha.is_some() {
                    return Err("--sha may be supplied only once".to_string());
                }
                sha = Some(
                    args.next()
                        .ok_or_else(|| format!("--sha requires a value\n{}", usage()))?,
                );
            }
            "--source-tree" => {
                if source_tree.is_some() {
                    return Err("--source-tree may be supplied only once".to_string());
                }
                source_tree =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        format!("--source-tree requires a value\n{}", usage())
                    })?));
            }
            "--help" | "-h" => return Err(usage().to_string()),
            _ => return Err(format!("unknown argument {arg:?}\n{}", usage())),
        }
    }
    let revision = revision.ok_or_else(|| format!("--revision is required\n{}", usage()))?;
    if !matches!(revision.as_str(), "base" | "head") {
        return Err("--revision must be `base` or `head`".to_string());
    }
    let sha = sha.ok_or_else(|| format!("--sha is required\n{}", usage()))?;
    if sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--sha must be a full 40-character hexadecimal commit SHA".to_string());
    }
    Ok(ReportArgs {
        revision,
        sha: sha.to_ascii_lowercase(),
        source_tree: source_tree
            .ok_or_else(|| format!("--source-tree is required\n{}", usage()))?,
    })
}

fn validate_source_provenance(
    revision: &str,
    expected_sha: &str,
    compiled_sha: &str,
    compiled_state: &str,
    observed_sha: &str,
    status: &str,
    baseline_backport: bool,
) -> Result<(), String> {
    if baseline_backport {
        if revision != "base" || expected_sha != STACK_BASE_SHA {
            return Err(format!(
                "the compatibility reporter requires revision `base` at {STACK_BASE_SHA}"
            ));
        }
        if compiled_state != "approved-base-backport" {
            return Err(format!(
                "base catalogs were compiled from an unauthenticated source state: {compiled_state}"
            ));
        }
    } else {
        if revision != "head" {
            return Err("the quotient-free reporter requires revision `head`".to_string());
        }
        if compiled_state != "clean" {
            return Err(format!(
                "head catalogs were compiled from an unauthenticated source state: {compiled_state}"
            ));
        }
    }
    if compiled_sha != expected_sha {
        return Err(format!(
            "catalogs were compiled from {compiled_sha}, not requested SHA {expected_sha}"
        ));
    }
    if observed_sha != expected_sha {
        return Err(format!(
            "catalog checkout is {observed_sha}, not requested SHA {expected_sha}"
        ));
    }
    for line in status.lines().filter(|line| !line.trim().is_empty()) {
        let path = line.get(3..).unwrap_or_default();
        let is_reporter_backport = path.ends_with("crates/akita-planner/Cargo.toml")
            || path.ends_with("crates/akita-planner/build.rs")
            || path.ends_with("crates/akita-planner/examples/quotient_free_catalog_evidence.rs");
        if !baseline_backport || !is_reporter_backport {
            return Err(format!(
                "catalog evidence requires authenticated sources; unexpected change: {line}"
            ));
        }
    }
    Ok(())
}

fn verify_source_provenance(args: &ReportArgs) -> Result<(), String> {
    let rev_parse = Command::new("git")
        .arg("-C")
        .arg(&args.source_tree)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| format!("run git rev-parse for evidence provenance: {error}"))?;
    if !rev_parse.status.success() {
        return Err("git rev-parse failed for evidence provenance".to_string());
    }
    let observed_sha = String::from_utf8(rev_parse.stdout)
        .map_err(|error| format!("decode git revision: {error}"))?;
    let status = Command::new("git")
        .arg("-C")
        .arg(&args.source_tree)
        .args(["status", "--porcelain", "--untracked-files=all"])
        .output()
        .map_err(|error| format!("run git status for evidence provenance: {error}"))?;
    if !status.status.success() {
        return Err("git status failed for evidence provenance".to_string());
    }
    let status =
        String::from_utf8(status.stdout).map_err(|error| format!("decode git status: {error}"))?;
    let compiled_sha = env!("AKITA_CATALOG_SOURCE_SHA");
    let compiled_state = env!("AKITA_CATALOG_SOURCE_STATE");
    validate_source_provenance(
        &args.revision,
        &args.sha,
        compiled_sha,
        compiled_state,
        observed_sha.trim(),
        &status,
        !cfg!(feature = "quotient-free-evidence"),
    )
}

fn hex(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn logical_key(key: &AkitaScheduleLookupKey) -> String {
    let mut logical = format!(
        "final={}:{};precommitted=",
        key.final_group.num_vars(),
        key.final_group.num_polynomials(),
    );
    for (index, profile) in key.precommitteds.iter().enumerate() {
        if index != 0 {
            logical.push(',');
        }
        write!(
            &mut logical,
            "{}:{}",
            profile.group.num_vars(),
            profile.group.num_polynomials(),
        )
        .expect("writing to String cannot fail");
    }
    logical
}

#[cfg(feature = "quotient-free-evidence")]
fn relation_mode(mode: RingRelationMode) -> &'static str {
    match mode {
        RingRelationMode::QuotientLift => "quotient",
        RingRelationMode::ReducedEvaluation => "reduced-evaluation",
    }
}

fn payload_mode(mode: CommitmentPayloadMode) -> &'static str {
    match mode {
        CommitmentPayloadMode::Compressed => "compressed",
        CommitmentPayloadMode::Raw => "raw",
    }
}

fn opening_method(method: OpeningMethod) -> String {
    match method {
        OpeningMethod::EvaluationTrace => "evaluation-trace".to_string(),
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => format!("subring-coefficient-packing-d{challenge_subring_dimension}"),
    }
}

fn security_route(route: InnerCommitSecurityRoute) -> &'static str {
    match route {
        InnerCommitSecurityRoute::Linf(_) => "Linf",
        InnerCommitSecurityRoute::L2 { .. } => "L2",
    }
}

#[cfg(feature = "quotient-free-evidence")]
fn removed_quotient_coefficients(
    params: &akita_types::CommittedGroupParams,
    input_witness_len: usize,
    claim_ext_degree: usize,
    field_bits: u32,
) -> Result<(usize, usize), String> {
    if !params.ring_relation_mode.is_reduced_evaluation() {
        return Ok((0, 0));
    }
    let breakdown = QuotientCoefficientBreakdown::for_reduced_input_witness(
        params,
        input_witness_len,
        claim_ext_degree,
        field_bits,
    )
    .map_err(|error| format!("derive quotient counterfactual: {error}"))?;
    Ok((breakdown.ordinary, breakdown.compression))
}

#[cfg(not(feature = "quotient-free-evidence"))]
fn removed_quotient_coefficients(
    _params: &akita_types::CommittedGroupParams,
    _input_witness_len: usize,
    _claim_ext_degree: usize,
    _field_bits: u32,
) -> Result<(usize, usize), String> {
    Ok((0, 0))
}

#[cfg(feature = "quotient-free-evidence")]
fn schedule_cutover(schedule: &FoldSchedule) -> Option<usize> {
    nonterminal_levels(schedule).find_map(|(level, fold)| {
        fold.params
            .ring_relation_mode
            .is_reduced_evaluation()
            .then_some(level)
    })
}

#[cfg(not(feature = "quotient-free-evidence"))]
fn schedule_cutover(_schedule: &FoldSchedule) -> Option<usize> {
    None
}

#[cfg(feature = "quotient-free-evidence")]
fn fold_relation_mode(params: &akita_types::CommittedGroupParams) -> &'static str {
    relation_mode(params.ring_relation_mode)
}

#[cfg(not(feature = "quotient-free-evidence"))]
fn fold_relation_mode(_params: &akita_types::CommittedGroupParams) -> &'static str {
    "quotient"
}

fn nonterminal_levels(
    schedule: &FoldSchedule,
) -> impl Iterator<Item = (usize, &akita_types::FoldParams)> {
    std::iter::once((0, &schedule.root)).chain(
        schedule
            .recursive_folds
            .iter()
            .enumerate()
            .map(|(index, fold)| (index + 1, fold)),
    )
}

fn rows_for_family(
    args: &ReportArgs,
    family: &'static GeneratedFamily,
) -> Result<Vec<EvidenceRow>, String> {
    let policy = (family.policy)();
    let catalog = (family.schedule_catalog)()
        .ok_or_else(|| format!("{} catalog is not linked", family.module_name))?;
    let mut rows = Vec::new();
    for entry in catalog.entries {
        let key = entry.to_runtime_lookup_key();
        let resolved = (family.resolve_catalog_row_for_key)(key.clone())
            .map_err(|error| format!("{} {key:?}: {error}", family.module_name))?;
        let schedule = resolved.schedule();
        let total_proof_bytes =
            akita_schedules::expanded_schedule_proof_payload_bytes(&key, schedule, &policy)
                .map_err(|error| format!("{} {key:?}: price proof: {error}", family.module_name))?;
        let cutover_level = schedule_cutover(schedule);
        let lookup_key_digest = hex(&akita_types::instance_descriptor::digest_descriptor_bytes(
            &key.canonical_descriptor_bytes(),
        ));
        let row_digest = hex(resolved.selection().row_digest.as_bytes());
        let schedule_descriptor_digest = hex(&akita_types::digest_effective_schedule(schedule));
        let logical_key = logical_key(&key);
        for (level, fold) in nonterminal_levels(schedule) {
            let successor = schedule
                .recursive_folds
                .get(level)
                .map(|successor| &successor.params);
            let (direct_payload_bytes, stage3_payload_bytes) = nonterminal_level_payload_bytes(
                &policy,
                &fold.params,
                successor,
                fold.input_witness_len,
                fold.output_witness_len,
            )
            .map_err(|error| {
                format!(
                    "{} {key:?} L{level}: price level: {error}",
                    family.module_name
                )
            })?;
            let (ordinary_quotient_coefficients_removed, compression_quotient_coefficients_removed) =
                removed_quotient_coefficients(
                    &fold.params,
                    fold.input_witness_len,
                    policy.claim_ext_degree,
                    policy.decomposition.field_bits(),
                )?;
            rows.push(EvidenceRow {
                revision: args.revision.clone(),
                sha: args.sha.clone(),
                family: family.module_name,
                num_vars: key.final_group.num_vars(),
                num_polynomials: key.final_group.num_polynomials(),
                logical_key: logical_key.clone(),
                lookup_key_digest: lookup_key_digest.clone(),
                row_digest: row_digest.clone(),
                schedule_descriptor_digest: schedule_descriptor_digest.clone(),
                cutover_level,
                fold_level: level,
                relation_mode: fold_relation_mode(&fold.params),
                input_witness_len: fold.input_witness_len,
                output_witness_len: fold.output_witness_len,
                ordinary_quotient_coefficients_removed,
                compression_quotient_coefficients_removed,
                payload_mode: payload_mode(fold.params.payload_mode),
                opening_method: opening_method(fold.params.opening_method()),
                security_route: security_route(fold.params.inner().matrix.security_route()),
                incoming_setup_prefix: fold.params.setup_prefix().is_some(),
                direct_payload_bytes,
                stage3_payload_bytes,
                total_proof_bytes,
            });
        }
    }
    Ok(rows)
}

fn build_report(args: &ReportArgs) -> Result<String, String> {
    let mut families = DENSE_FAMILIES
        .iter()
        .map(|name| {
            ALL_GENERATED_FAMILIES
                .iter()
                .find(|family| family.module_name == *name)
                .ok_or_else(|| format!("missing generated family {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    families.sort_by_key(|family| family.module_name);

    let mut rows = Vec::new();
    for family in families {
        rows.extend(rows_for_family(args, family)?);
    }
    rows.sort();

    let mut out = String::new();
    writeln!(&mut out, "{}", HEADER.join("\t")).expect("writing to String cannot fail");
    for row in rows {
        row.write_tsv(&mut out);
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    let args = parse_args(env::args().skip(1))?;
    verify_source_provenance(&args)?;
    print!("{}", build_report(&args)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn test_args() -> ReportArgs {
        ReportArgs {
            revision: "head".to_string(),
            sha: "0123456789abcdef0123456789abcdef01234567".to_string(),
            source_tree: PathBuf::from("."),
        }
    }

    #[cfg(feature = "quotient-free-evidence")]
    #[test]
    fn compiled_dense_report_is_complete_and_relation_modes_are_monotone() {
        let report = build_report(&test_args()).expect("compiled catalogs must report");
        let mut lines = report.lines();
        assert_eq!(lines.next(), Some(HEADER.join("\t").as_str()));

        let mut families = BTreeSet::new();
        let mut rows: BTreeMap<(&str, &str), Vec<Vec<&str>>> = BTreeMap::new();
        for line in lines {
            let columns = line.split('\t').collect::<Vec<_>>();
            assert_eq!(columns.len(), HEADER.len(), "malformed TSV row: {line}");
            families.insert(columns[2]);
            rows.entry((columns[2], columns[7]))
                .or_default()
                .push(columns);
        }
        assert_eq!(families, DENSE_FAMILIES.into_iter().collect());
        assert!(!rows.is_empty());
        let row_counts = DENSE_FAMILIES
            .into_iter()
            .map(|family| {
                (
                    family,
                    rows.keys()
                        .filter(|(reported_family, _)| *reported_family == family)
                        .count(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            row_counts,
            BTreeMap::from([("fp32_dense", 5), ("fp64_dense", 8), ("fp128_dense", 11)])
        );

        for ((family, row_digest), levels) in rows {
            let declared_cutover = levels[0][9];
            let first_reduced = levels
                .iter()
                .position(|columns| columns[11] == "reduced-evaluation");
            assert!(
                first_reduced.is_some(),
                "{family} {row_digest} has no reduced-evaluation suffix"
            );
            assert_eq!(
                declared_cutover,
                first_reduced.map_or_else(|| "none".to_string(), |level| level.to_string()),
                "{family} {row_digest} reports the wrong cutover"
            );
            assert!(levels.windows(2).all(|pair| {
                pair[0][10].parse::<usize>().expect("numeric fold level") + 1
                    == pair[1][10].parse::<usize>().expect("numeric fold level")
            }));
            assert!(levels
                .iter()
                .all(|columns| columns[8] == levels[0][8] && columns[22] == levels[0][22]));
            let mut reduced_suffix_started = false;
            for (level, columns) in levels.iter().enumerate() {
                assert_eq!(columns[10].parse::<usize>(), Ok(level));
                let ordinary = columns[14]
                    .parse::<usize>()
                    .expect("ordinary quotient count");
                let compression = columns[15]
                    .parse::<usize>()
                    .expect("compression quotient count");
                if columns[11] == "quotient" {
                    assert!(
                        !reduced_suffix_started,
                        "{family} {row_digest} returned to quotient mode at L{level}"
                    );
                    assert_eq!((ordinary, compression), (0, 0));
                } else {
                    reduced_suffix_started = true;
                    assert!(
                        ordinary > 0 || compression > 0,
                        "{family} {row_digest} L{level} removed no quotient coefficients"
                    );
                }
            }
        }
    }

    #[test]
    fn provenance_rejects_sha_mismatch_and_unrelated_changes() {
        let expected = "0123456789abcdef0123456789abcdef01234567";
        let other = "1123456789abcdef0123456789abcdef01234567";
        assert!(
            validate_source_provenance("head", expected, other, "clean", expected, "", false)
                .is_err()
        );
        assert!(
            validate_source_provenance("head", expected, expected, "clean", other, "", false)
                .is_err()
        );
        assert!(validate_source_provenance(
            "head",
            expected,
            expected,
            "clean",
            expected,
            " M crates/akita-schedules/src/generated/fp32_dense.rs",
            false,
        )
        .is_err());
        assert!(validate_source_provenance(
            "head", expected, expected, "clean", expected, "", false,
        )
        .is_ok());
        assert!(validate_source_provenance(
            "base",
            STACK_BASE_SHA,
            STACK_BASE_SHA,
            "approved-base-backport",
            STACK_BASE_SHA,
            " M crates/akita-planner/Cargo.toml\n?? crates/akita-planner/build.rs\n?? crates/akita-planner/examples/quotient_free_catalog_evidence.rs",
            true,
        )
        .is_ok());
        assert!(validate_source_provenance(
            "head",
            expected,
            expected,
            "clean",
            expected,
            " M crates/akita-planner/Cargo.toml",
            true,
        )
        .is_err());
        assert!(validate_source_provenance(
            "base", expected, expected, "clean", expected, "", false,
        )
        .is_err());
        assert!(validate_source_provenance(
            "head",
            STACK_BASE_SHA,
            STACK_BASE_SHA,
            "approved-base-backport",
            STACK_BASE_SHA,
            " M crates/akita-planner/Cargo.toml",
            true,
        )
        .is_err());
    }
}
