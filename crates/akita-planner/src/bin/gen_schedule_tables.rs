//! Generate schedule tables using the offline DP planner.

use akita_planner::emit::{
    bounded_parallel_filter_map, offline_planning_worker_count, MaterializationDiagnostics,
};
use akita_planner::generated_families::{
    emit_spec_for_family, wiring_emit_spec, GeneratedFamily, GenerationPreplans,
    ALL_GENERATED_FAMILIES,
};
use akita_planner::{
    publish_generated_outputs, render_generated_outputs_with_validation, EmitSpec,
};
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, CommittedGroupProfile,
    FoldSchedule, PolynomialGroupLayout,
};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

#[derive(Default)]
struct ExplicitRows {
    final_group: Option<ExplicitGroup>,
    precommitted_groups: Vec<ExplicitGroup>,
}

struct ParsedArgs {
    base_dir: PathBuf,
    wiring_only: bool,
    check_catalog: bool,
    catalog_report: Option<PathBuf>,
    row_progress: bool,
    family_filter: Option<Vec<String>>,
    explicit_rows: ExplicitRows,
}

#[derive(Clone)]
struct ExplicitGroup {
    family: String,
    num_vars: ExplicitRange,
    num_polys: ExplicitRange,
}

#[derive(Clone)]
struct ExplicitRange {
    start: usize,
    end: usize,
}

fn generator_command() -> &'static str {
    "cargo run --release -p akita-planner --features catalog-gen --bin gen_schedule_tables -- <output-dir>"
}

fn usage() -> &'static str {
    "usage: cargo run --release -p akita-planner --features catalog-gen \
     --bin gen_schedule_tables -- <output-dir> [--wiring-only] [--check-catalog] \
     [--catalog-report <path>] [--row-progress] \
     [family_module_name ...]\n\
     positional family names select only those generated families; omit them \
     to generate every family \
     [--final-group family:num_vars_or_range:num_polys_or_range] \
     [--precommitted-group family:num_vars_or_range:num_polys_or_range ...]"
}

fn sorted_unique_specs(specs: &[EmitSpec]) -> Vec<EmitSpec> {
    let mut out: Vec<EmitSpec> = specs.to_vec();
    out.sort_by_key(|spec| spec.module_name);
    out.dedup_by_key(|spec| spec.module_name);
    out
}

fn known_family(name: &str) -> bool {
    ALL_GENERATED_FAMILIES
        .iter()
        .any(|family| family.module_name == name)
}

fn family_by_name(name: &str) -> Option<&'static GeneratedFamily> {
    ALL_GENERATED_FAMILIES
        .iter()
        .find(|family| family.module_name == name)
}

fn explicit_family_is_d64(name: &str) -> bool {
    family_by_name(name).is_some_and(|family| (family.policy)().uniform_ring_dimension == 64)
}

fn parse_usize(raw: &str, context: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|e| format!("{context}: expected unsigned integer, got `{raw}`: {e}"))
}

fn parse_range(raw: &str, context: &str) -> Result<ExplicitRange, String> {
    let bounds = raw
        .split_once("..=")
        .or_else(|| raw.split_once(".."))
        .or_else(|| raw.split_once('-'));
    let (start, end) = if let Some((start, end)) = bounds {
        (parse_usize(start, context)?, parse_usize(end, context)?)
    } else {
        let value = parse_usize(raw, context)?;
        (value, value)
    };
    if start > end {
        return Err(format!(
            "{context}: range start {start} is greater than end {end}"
        ));
    }
    Ok(ExplicitRange { start, end })
}

fn parse_explicit_group(raw: &str) -> Result<ExplicitGroup, String> {
    let parts = raw.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        return Err(format!("expected `family:nv:num_polys`, got `{raw}`"));
    }
    if !known_family(parts[0]) {
        return Err(format!("unknown schedule family: {}", parts[0]));
    }
    Ok(ExplicitGroup {
        family: parts[0].to_string(),
        num_vars: parse_range(parts[1], "num_vars")?,
        num_polys: parse_range(parts[2], "num_polys")?,
    })
}

fn parse_args() -> Result<ParsedArgs, String> {
    parse_args_from(env::args().skip(1).collect())
}

fn parse_args_from(raw_args: Vec<String>) -> Result<ParsedArgs, String> {
    if raw_args.is_empty() {
        return Err(usage().to_string());
    }
    let base_dir = PathBuf::from(&raw_args[0]);
    let mut wiring_only = false;
    let mut check_catalog = false;
    let mut catalog_report = None;
    let mut row_progress = false;
    let mut family_args = Vec::new();
    let mut explicit_rows = ExplicitRows::default();
    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--wiring-only" => {
                wiring_only = true;
                i += 1;
            }
            "--check-catalog" => {
                check_catalog = true;
                i += 1;
            }
            "--row-progress" => {
                row_progress = true;
                i += 1;
            }
            "--catalog-report" => {
                let value = raw_args
                    .get(i + 1)
                    .ok_or_else(|| "--catalog-report requires a path".to_string())?;
                if catalog_report.is_some() {
                    return Err("--catalog-report may be supplied only once".to_string());
                }
                catalog_report = Some(PathBuf::from(value));
                i += 2;
            }
            "--final-group" => {
                let value = raw_args
                    .get(i + 1)
                    .ok_or_else(|| "--final-group requires a value".to_string())?;
                if explicit_rows.final_group.is_some() {
                    return Err("--final-group may be supplied only once".to_string());
                }
                explicit_rows.final_group = Some(parse_explicit_group(value)?);
                i += 2;
            }
            "--precommitted-group" => {
                let value = raw_args
                    .get(i + 1)
                    .ok_or_else(|| "--precommitted-group requires a value".to_string())?;
                explicit_rows
                    .precommitted_groups
                    .push(parse_explicit_group(value)?);
                i += 2;
            }
            flag if flag.starts_with("--") => {
                return Err(format!("unknown option `{flag}`\n{}", usage()));
            }
            family => {
                if !known_family(family) {
                    return Err(format!("unknown schedule family: {family}"));
                }
                family_args.push(family.to_string());
                i += 1;
            }
        }
    }
    if !explicit_rows.precommitted_groups.is_empty() && explicit_rows.final_group.is_none() {
        return Err("--precommitted-group requires --final-group".to_string());
    }
    let explicit_families = explicit_rows.family_names();
    let mut non_d64_families = explicit_families
        .iter()
        .filter(|family| !explicit_family_is_d64(family.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    non_d64_families.sort();
    if !non_d64_families.is_empty() {
        return Err(format!(
            "explicit rows require D64 schedule families; got {}",
            non_d64_families.join(", ")
        ));
    }
    if let Some(final_group) = &explicit_rows.final_group {
        if !family_args.is_empty()
            && (family_args.len() != 1 || family_args[0] != final_group.family)
        {
            return Err(format!(
                "--final-group writes only `{}`; omit positional families or pass that family only",
                final_group.family
            ));
        }
    }
    let family_filter = if let Some(final_group) = &explicit_rows.final_group {
        Some(vec![final_group.family.clone()])
    } else if family_args.is_empty() {
        if explicit_families.is_empty() {
            None
        } else {
            Some(explicit_families.into_iter().collect())
        }
    } else {
        Some(family_args)
    };
    if wiring_only && family_filter.is_some() {
        return Err("--wiring-only does not accept family filters or explicit rows".to_string());
    }
    if check_catalog && (wiring_only || explicit_rows.final_group.is_some()) {
        return Err("--check-catalog requires ordinary generated rows".to_string());
    }
    if check_catalog && !cfg!(feature = "catalog-check") {
        return Err("--check-catalog requires the `catalog-check` feature".to_string());
    }
    if catalog_report.is_some() && !check_catalog {
        return Err("--catalog-report requires --check-catalog".to_string());
    }
    Ok(ParsedArgs {
        base_dir,
        wiring_only,
        check_catalog,
        catalog_report,
        row_progress,
        family_filter,
        explicit_rows,
    })
}

fn selected_families(family_filter: Option<&[String]>) -> Vec<&'static GeneratedFamily> {
    ALL_GENERATED_FAMILIES
        .iter()
        .filter(|family| {
            family_filter.is_none_or(|names| names.iter().any(|name| name == family.module_name))
        })
        .collect()
}

fn validate_materialized_catalog(
    spec: &EmitSpec,
    entries: &[akita_planner::emit::MaterializedEntry],
) -> Result<CatalogComparison, String> {
    let family = family_by_name(spec.module_name)
        .ok_or_else(|| format!("unknown generated family: {}", spec.module_name))?;
    let table = (family.schedule_catalog)().ok_or_else(|| {
        format!(
            "{}: compiled catalog is unavailable; build with all schedule features",
            spec.module_name
        )
    })?;
    compare_materialized_catalog(spec, table, entries)
}

struct CatalogComparison {
    report: String,
    changed_rows: usize,
}

struct CatalogRowMetrics {
    setup_fields: usize,
    proof_bytes: usize,
    fold_levels: usize,
    row_digest: String,
    policy_signature: String,
}

const CATALOG_REPORT_HEADER: &str = "family\tstatus\tkey\told_setup_fields\tnew_setup_fields\told_proof_bytes\tnew_proof_bytes\told_levels\tnew_levels\told_row_digest\tnew_row_digest\told_policy\tnew_policy\n";

fn source_encoding_signature(value: akita_types::CommittedSourceEncoding) -> String {
    match value {
        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable => "canonical".into(),
        akita_types::CommittedSourceEncoding::TensorSubfieldProjection { extension_degree } => {
            format!("tensor-k{extension_degree}")
        }
    }
}

fn security_route_signature(value: akita_types::InnerCommitSecurityRoute) -> &'static str {
    match value {
        akita_types::InnerCommitSecurityRoute::Linf(_) => "Linf",
        akita_types::InnerCommitSecurityRoute::L2 { .. } => "L2",
    }
}

fn opening_policy_signature(
    opening_method: akita_types::OpeningMethod,
    source_encoding: akita_types::CommittedSourceEncoding,
    extension_degree: usize,
    d_a: usize,
    security_route: akita_types::InnerCommitSecurityRoute,
) -> Result<String, String> {
    let opening = match opening_method {
        akita_types::OpeningMethod::EvaluationTrace => "ET,s=-,h=-,w=-".to_string(),
        akita_types::OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => {
            let geometry = akita_types::SubringCoefficientPackingGeometry::try_new(
                extension_degree,
                d_a,
                challenge_subring_dimension,
            )
            .map_err(|error| format!("derive catalog packing geometry: {error}"))?;
            format!(
                "PACK,s={},h={},w={}",
                geometry.challenge_subring_dimension(),
                geometry.packing_factor(),
                geometry.partial_base_field_width(),
            )
        }
    };
    Ok(format!(
        "{opening},src={},dA={d_a},sec={}",
        source_encoding_signature(source_encoding),
        security_route_signature(security_route),
    ))
}

fn catalog_policy_signature(spec: &EmitSpec, schedule: &FoldSchedule) -> Result<String, String> {
    use std::fmt::Write as _;

    let mut signature = String::new();
    let nonterminal = std::iter::once((
        0usize,
        &schedule.root.params.final_group.commitment,
        schedule.root.input_witness_len,
        schedule.root.output_witness_len,
    ))
    .chain(
        schedule
            .recursive_folds
            .iter()
            .enumerate()
            .map(|(index, fold)| {
                (
                    index + 1,
                    &fold.params.witness,
                    fold.input_witness_len,
                    fold.output_witness_len,
                )
            }),
    );
    for (level, params, input_witness_len, output_witness_len) in nonterminal {
        let eor = if matches!(
            params.opening_method,
            akita_types::OpeningMethod::EvaluationTrace
        ) {
            let final_group = akita_types::PolynomialGroupLayout::singleton(
                akita_types::padded_boolean_opening_vars(input_witness_len)
                    .map_err(|error| format!("derive opening arity: {error}"))?,
            );
            let opening_shape = params
                .opening_layout_for_final_group(final_group)
                .and_then(|layout| layout.aggregate_polynomial_group_layout())
                .map_err(|error| format!("derive level opening shape: {error}"))?;
            akita_types::extension_opening_reduction_level_bytes(
                spec.policy
                    .challenge_field_bits()
                    .map_err(|error| format!("derive challenge width: {error}"))?,
                spec.policy.claim_ext_degree,
                opening_shape,
            )
            .map_err(|error| format!("derive level EOR bytes: {error}"))?
        } else {
            0
        };
        if level != 0 {
            signature.push('/');
        }
        write!(
            signature,
            "L{level}[chunks={}@{},eor={eor},in={input_witness_len},out={output_witness_len};witness={}",
            params.witness_chunk.num_chunks,
            params.witness_chunk.num_activated_levels,
            opening_policy_signature(
                params.opening_method,
                params.source_encoding,
                spec.policy.claim_ext_degree,
                params.d_a(),
                params.inner_commit_matrix.security_route(),
            )?,
        )
        .map_err(|error| format!("write catalog policy signature: {error}"))?;
        if level == 0 {
            for (index, group) in schedule.root.params.precommitted_groups.iter().enumerate() {
                write!(
                    signature,
                    ";pre{index}={}",
                    opening_policy_signature(
                        group.commitment.opening.opening_method,
                        akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
                        spec.policy.claim_ext_degree,
                        group.commitment.layout.inner_commit_matrix.ring_dimension(),
                        group.commitment.layout.inner_commit_matrix.security_route(),
                    )?,
                )
                .map_err(|error| format!("write catalog policy signature: {error}"))?;
            }
        } else if let Some(prefix) = schedule.recursive_folds[level - 1]
            .params
            .incoming_setup_prefix
            .as_ref()
        {
            write!(
                signature,
                ";prefix={}",
                opening_policy_signature(
                    prefix.commitment_params.opening.opening_method,
                    akita_types::CommittedSourceEncoding::CanonicalCoefficientTable,
                    spec.policy.claim_ext_degree,
                    prefix
                        .commitment_params
                        .layout
                        .inner_commit_matrix
                        .ring_dimension(),
                    prefix
                        .commitment_params
                        .layout
                        .inner_commit_matrix
                        .security_route(),
                )?,
            )
            .map_err(|error| format!("write catalog policy signature: {error}"))?;
        }
        signature.push(']');
    }
    let terminal_eor = akita_types::extension_opening_reduction_level_bytes(
        spec.policy
            .challenge_field_bits()
            .map_err(|error| format!("derive challenge width: {error}"))?,
        spec.policy.claim_ext_degree,
        akita_types::PolynomialGroupLayout::singleton(
            akita_types::padded_boolean_opening_vars(schedule.terminal.input_witness_len)
                .map_err(|error| format!("derive terminal opening arity: {error}"))?,
        ),
    )
    .map_err(|error| format!("derive terminal EOR bytes: {error}"))?;
    let terminal_source = akita_types::CommittedSourceEncoding::for_producer(
        akita_types::OpeningMethod::EvaluationTrace,
        spec.policy.claim_ext_degree,
        schedule.terminal.params.witness.d_a(),
        akita_types::padded_boolean_opening_vars(schedule.terminal.input_witness_len)
            .map_err(|error| format!("derive terminal source arity: {error}"))?,
        false,
    );
    write!(
        signature,
        "/T[method=ET,src={},eor={terminal_eor},input={},dA={},sec={}]",
        source_encoding_signature(terminal_source),
        schedule.terminal.input_witness_len,
        schedule.terminal.params.witness.d_a(),
        security_route_signature(
            schedule
                .terminal
                .params
                .witness
                .inner_commit_matrix
                .security_route(),
        ),
    )
    .map_err(|error| format!("write catalog policy signature: {error}"))?;
    Ok(signature)
}

fn row_digest_hex(key: &AkitaScheduleLookupKey, schedule: &FoldSchedule) -> Result<String, String> {
    let final_group = CommittedGroupProfile::try_from_params(
        key.final_group,
        &schedule.root.params.final_group.commitment,
    )
    .map_err(|error| format!("derive final committed profile: {error}"))?;
    let profiles = CommittedGroupBatchProfile {
        final_group,
        precommitteds: key.precommitteds.clone(),
    };
    let digest = schedule_row_digest(&profiles, schedule)
        .map_err(|error| format!("derive schedule row digest: {error}"))?;
    Ok(digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn catalog_row_metrics(
    spec: &EmitSpec,
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<CatalogRowMetrics, String> {
    let proof_bytes =
        akita_schedules::expanded_schedule_proof_payload_bytes(key, schedule, &spec.policy)
            .map_err(|error| format!("estimate proof payload: {error}"))?;
    let setup_fields = akita_types::setup_matrix_capacity_for_schedule(schedule)
        .map_err(|error| format!("estimate setup capacity: {error}"))?
        .num_field_elements;
    Ok(CatalogRowMetrics {
        setup_fields,
        proof_bytes,
        fold_levels: schedule.num_fold_levels(),
        row_digest: row_digest_hex(key, schedule)?,
        policy_signature: catalog_policy_signature(spec, schedule)?,
    })
}

fn compact_catalog_key(key: &AkitaScheduleLookupKey) -> String {
    let digest = akita_types::instance_descriptor::digest_descriptor_bytes(
        &key.canonical_descriptor_bytes(),
    );
    let id = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(
        "nv={};polys={};precommits={};key={id}",
        key.final_group.num_vars(),
        key.final_group.num_polynomials(),
        key.precommitteds.len(),
    )
}

fn optional_metric(
    value: Option<&CatalogRowMetrics>,
    field: fn(&CatalogRowMetrics) -> usize,
) -> String {
    value
        .map(field)
        .map_or_else(|| "-".to_string(), |value| value.to_string())
}

fn optional_digest(value: Option<&CatalogRowMetrics>) -> &str {
    value.map_or("-", |metrics| metrics.row_digest.as_str())
}

fn optional_policy(value: Option<&CatalogRowMetrics>) -> &str {
    value.map_or("-", |metrics| metrics.policy_signature.as_str())
}

fn compare_materialized_catalog(
    spec: &EmitSpec,
    table: akita_schedules::GeneratedScheduleTable,
    entries: &[akita_planner::emit::MaterializedEntry],
) -> Result<CatalogComparison, String> {
    let mut old_rows = table
        .entries
        .iter()
        .copied()
        .map(|entry| {
            let key = entry.to_runtime_lookup_key();
            let schedule = akita_schedules::schedule_from_entry(
                &entry,
                &key,
                &spec.policy,
                spec.ring_challenge_config,
            )
            .map_err(|error| format!("{}: expand compiled row: {error}", spec.module_name))?;
            Ok((key, schedule))
        })
        .collect::<Result<Vec<_>, String>>()?;
    old_rows
        .sort_by(|(left, _), (right, _)| akita_schedules::runtime_schedule_key_cmp(left, right));

    let mut report = String::new();
    let mut old_index = 0;
    let mut new_index = 0;
    let mut changed_rows = 0;
    while old_index < old_rows.len() || new_index < entries.len() {
        let ordering = match (old_rows.get(old_index), entries.get(new_index)) {
            (Some((old, _)), Some((new, _))) => akita_schedules::runtime_schedule_key_cmp(old, new),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => break,
        };
        let (status, key, old_schedule, new_schedule) = match ordering {
            std::cmp::Ordering::Less => {
                let (key, schedule) = &old_rows[old_index];
                old_index += 1;
                ("removed", key, Some(schedule), None)
            }
            std::cmp::Ordering::Greater => {
                let (key, schedule) = &entries[new_index];
                new_index += 1;
                ("added", key, None, Some(schedule))
            }
            std::cmp::Ordering::Equal => {
                let (key, old_schedule) = &old_rows[old_index];
                let (_, new_schedule) = &entries[new_index];
                old_index += 1;
                new_index += 1;
                let status = if old_schedule == new_schedule {
                    "equal"
                } else {
                    "changed"
                };
                (status, key, Some(old_schedule), Some(new_schedule))
            }
        };
        if status != "equal" {
            changed_rows += 1;
        }
        let old_metrics = old_schedule
            .map(|schedule| catalog_row_metrics(spec, key, schedule))
            .transpose()?;
        let new_metrics = new_schedule
            .map(|schedule| catalog_row_metrics(spec, key, schedule))
            .transpose()?;
        use std::fmt::Write as _;
        writeln!(
            report,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            spec.module_name,
            status,
            compact_catalog_key(key),
            optional_metric(old_metrics.as_ref(), |metrics| metrics.setup_fields),
            optional_metric(new_metrics.as_ref(), |metrics| metrics.setup_fields),
            optional_metric(old_metrics.as_ref(), |metrics| metrics.proof_bytes),
            optional_metric(new_metrics.as_ref(), |metrics| metrics.proof_bytes),
            optional_metric(old_metrics.as_ref(), |metrics| metrics.fold_levels),
            optional_metric(new_metrics.as_ref(), |metrics| metrics.fold_levels),
            optional_digest(old_metrics.as_ref()),
            optional_digest(new_metrics.as_ref()),
            optional_policy(old_metrics.as_ref()),
            optional_policy(new_metrics.as_ref()),
        )
        .map_err(|error| format!("write catalog comparison: {error}"))?;
    }
    Ok(CatalogComparison {
        report,
        changed_rows,
    })
}

fn resolved_output_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| format!("read current directory: {error}"))?
            .join(path)
    };
    let mut resolved = PathBuf::new();
    let mut missing = Vec::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let removed = if missing.is_empty() {
                    resolved.pop()
                } else {
                    missing.pop();
                    true
                };
                if !removed {
                    return Err(format!(
                        "output path escapes the filesystem root: {}",
                        path.display()
                    ));
                }
            }
            Component::Normal(name) if missing.is_empty() => {
                let candidate = resolved.join(name);
                if candidate.exists() {
                    resolved = fs::canonicalize(&candidate)
                        .map_err(|error| format!("resolve {}: {error}", candidate.display()))?;
                } else {
                    missing.push(name.to_os_string());
                }
            }
            Component::Normal(name) => missing.push(name.to_os_string()),
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
        }
    }
    for component in missing {
        resolved.push(component);
    }
    Ok(resolved)
}

fn validate_explicit_output_isolation(
    base_dir: &Path,
    explicit_rows: &ExplicitRows,
) -> Result<(), String> {
    if explicit_rows.final_group.is_none() {
        return Ok(());
    }
    let checked_in_generated_dir = resolved_output_path(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../akita-schedules/src/generated"),
    )?;
    let requested_dir = resolved_output_path(base_dir)?;
    if requested_dir.starts_with(&checked_in_generated_dir) {
        return Err(format!(
            "explicit schedule sweeps must use an isolated output directory outside {}",
            checked_in_generated_dir.display()
        ));
    }
    Ok(())
}

impl ExplicitRows {
    fn family_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(final_group) = &self.final_group {
            push_unique_name(&mut names, &final_group.family);
        }
        for group in &self.precommitted_groups {
            push_unique_name(&mut names, &group.family);
        }
        names
    }

    fn has_family(&self, family: &GeneratedFamily) -> bool {
        self.final_group
            .as_ref()
            .is_some_and(|group| group.family == family.module_name)
    }
}

impl ExplicitGroup {
    fn layouts(&self) -> Vec<PolynomialGroupLayout> {
        let mut layouts = Vec::new();
        for num_vars in self.num_vars.values() {
            for num_polys in self.num_polys.values() {
                layouts.push(PolynomialGroupLayout::new(num_vars, num_polys));
            }
        }
        layouts
    }
}

impl ExplicitRange {
    fn values(&self) -> impl Iterator<Item = usize> {
        self.start..=self.end
    }
}

fn push_unique_name(names: &mut Vec<String>, name: &str) {
    if !names.iter().any(|existing| existing == name) {
        names.push(name.to_string());
    }
}

fn push_unique_layout(layouts: &mut Vec<PolynomialGroupLayout>, layout: PolynomialGroupLayout) {
    if !layouts.contains(&layout) {
        layouts.push(layout);
    }
}

fn push_unique_group_batch_key(
    keys: &mut Vec<(
        AkitaScheduleLookupKey,
        Vec<akita_types::sis::HonestFoldPolicySpec>,
    )>,
    candidate: (
        AkitaScheduleLookupKey,
        Vec<akita_types::sis::HonestFoldPolicySpec>,
    ),
) {
    if !keys.contains(&candidate) {
        keys.push(candidate);
    }
}

fn expand_precommitted_choices(
    preplans: &GenerationPreplans,
    groups: &[ExplicitGroup],
) -> Result<
    Vec<
        Vec<(
            CommittedGroupProfile,
            akita_types::sis::HonestFoldPolicySpec,
        )>,
    >,
    String,
> {
    groups
        .iter()
        .map(|group| {
            let precommitted_family = family_by_name(&group.family)
                .ok_or_else(|| format!("unknown schedule family: {}", group.family))?;
            group
                .layouts()
                .into_iter()
                .map(|layout| {
                    (precommitted_family.explicit_precommitted_group)(preplans, layout)
                        .map_err(|e| format!("{}: explicit precommitted group: {e}", group.family))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn push_precommitted_combinations(
    choices: &[Vec<(
        CommittedGroupProfile,
        akita_types::sis::HonestFoldPolicySpec,
    )>],
    index: usize,
    profiles: &mut Vec<CommittedGroupProfile>,
    policies: &mut Vec<akita_types::sis::HonestFoldPolicySpec>,
    out: &mut Vec<(
        Vec<CommittedGroupProfile>,
        Vec<akita_types::sis::HonestFoldPolicySpec>,
    )>,
) {
    if index == choices.len() {
        out.push((profiles.clone(), policies.clone()));
        return;
    }
    for (profile, policy) in &choices[index] {
        profiles.push(*profile);
        policies.push(*policy);
        push_precommitted_combinations(choices, index + 1, profiles, policies, out);
        policies.pop();
        profiles.pop();
    }
}

fn emit_spec_with_overrides(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    base_dir: PathBuf,
    explicit_rows: &ExplicitRows,
    generator_command: &'static str,
) -> Result<EmitSpec, String> {
    if !explicit_rows.has_family(family) {
        return emit_spec_for_family(family, preplans, base_dir, generator_command)
            .map_err(|e| format!("{}: emit spec: {e}", family.module_name));
    }

    // Explicit sweeps replace the catalog key set. Start from the cheap wiring
    // shape so a one-key diagnostic does not first plan every default grouped
    // root merely to discard those rows below.
    let mut spec = wiring_emit_spec(family, base_dir);
    spec.generator_command = generator_command;

    let final_group = explicit_rows
        .final_group
        .as_ref()
        .ok_or_else(|| format!("{}: missing --final-group", family.module_name))?;
    spec.keys.clear();
    spec.group_batch_keys.clear();
    let final_layouts = final_group.layouts();

    if explicit_rows.precommitted_groups.is_empty() {
        for layout in final_layouts {
            push_unique_layout(&mut spec.keys, layout);
        }
        spec.keys
            .sort_by_key(|key| (key.num_vars(), key.num_polynomials()));
        return Ok(spec);
    }

    let precommitted_choices =
        expand_precommitted_choices(preplans, &explicit_rows.precommitted_groups)?;
    let mut precommitted_combinations = Vec::new();
    push_precommitted_combinations(
        &precommitted_choices,
        0,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut precommitted_combinations,
    );

    for (precommitteds, precommitted_honest_fold_policies) in precommitted_combinations {
        for final_layout in &final_layouts {
            push_unique_group_batch_key(
                &mut spec.group_batch_keys,
                (
                    AkitaScheduleLookupKey {
                        final_group: *final_layout,
                        precommitteds: precommitteds.clone(),
                    },
                    precommitted_honest_fold_policies.clone(),
                ),
            );
        }
    }
    Ok(spec)
}

fn main() -> Result<(), String> {
    let generation_started = Instant::now();
    let args = parse_args()?;
    validate_explicit_output_isolation(&args.base_dir, &args.explicit_rows)?;
    fs::create_dir_all(&args.base_dir)
        .map_err(|e| format!("create {}: {e}", args.base_dir.display()))?;
    let families_to_write = selected_families(args.family_filter.as_deref());

    let specs = if args.wiring_only {
        Vec::new()
    } else {
        let generator_command = generator_command();
        let preplans = GenerationPreplans::default();
        let indexed_families = families_to_write
            .iter()
            .enumerate()
            .map(|(index, family)| (index, *family))
            .collect::<Vec<_>>();
        let family_count = indexed_families.len();
        let workers = offline_planning_worker_count(family_count);
        let mut specs = bounded_parallel_filter_map(&indexed_families, workers, |item| {
            let (index, family) = *item;
            let family_started = Instant::now();
            eprintln!(
                "planning schedule family {}/{}: {}",
                index + 1,
                family_count,
                family.module_name
            );
            let spec = emit_spec_with_overrides(
                family,
                &preplans,
                args.base_dir.clone(),
                &args.explicit_rows,
                generator_command,
            )?;
            eprintln!(
                "planned schedule family {}/{}: {} ({} scalar keys, {} grouped keys) in {:.2?}",
                index + 1,
                family_count,
                family.module_name,
                spec.keys.len(),
                spec.group_batch_keys.len(),
                family_started.elapsed(),
            );
            Ok(Some(spec))
        })?;
        for (family, spec) in families_to_write.iter().zip(&mut specs) {
            preplans.attach_to_spec(family, spec);
        }
        drop(preplans);
        specs
    };

    let mod_path = args.base_dir.join("mod.rs");
    let wiring_specs = ALL_GENERATED_FAMILIES
        .iter()
        .map(|family| wiring_emit_spec(family, args.base_dir.clone()))
        .collect::<Vec<_>>();
    let mod_path = if mod_path.exists() {
        Some(mod_path)
    } else if args.wiring_only {
        return Err(format!("missing {}", mod_path.display()));
    } else {
        println!("skipped missing {}", mod_path.display());
        None
    };
    let check_catalog = args.check_catalog;
    let mut catalog_report = if check_catalog {
        CATALOG_REPORT_HEADER.to_string()
    } else {
        String::new()
    };
    let mut changed_catalog_rows = 0usize;
    let outputs = render_generated_outputs_with_validation(
        &specs,
        &sorted_unique_specs(&wiring_specs),
        mod_path.as_deref(),
        MaterializationDiagnostics {
            row_progress: args.row_progress,
        },
        |spec, entries| {
            if check_catalog {
                let comparison = validate_materialized_catalog(spec, entries)?;
                catalog_report.push_str(&comparison.report);
                changed_catalog_rows = changed_catalog_rows
                    .checked_add(comparison.changed_rows)
                    .ok_or_else(|| "catalog comparison row count overflow".to_string())?;
            }
            Ok(())
        },
    )?;
    if check_catalog {
        if let Some(path) = &args.catalog_report {
            fs::write(path, &catalog_report)
                .map_err(|error| format!("write {}: {error}", path.display()))?;
            eprintln!("wrote catalog comparison {}", path.display());
        } else {
            eprint!("{catalog_report}");
        }
        if changed_catalog_rows != 0 {
            return Err(format!(
                "compiled catalog differs from the planner in {changed_catalog_rows} rows"
            ));
        }
    }
    let destinations = publish_generated_outputs(outputs)?;
    for destination in &destinations {
        println!("wrote {}", destination.display());
    }
    if args.wiring_only {
        eprintln!(
            "finished schedule module wiring and published {} files in {:.2?}",
            destinations.len(),
            generation_started.elapsed(),
        );
    } else {
        eprintln!(
            "finished {} schedule {} and published {} files in {:.2?}",
            specs.len(),
            if specs.len() == 1 {
                "family"
            } else {
                "families"
            },
            destinations.len(),
            generation_started.elapsed(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positional_family_filters_are_checked_and_ordered() {
        let one =
            parse_args_from(vec!["generated".into(), "fp32_dense".into()]).expect("known family");
        assert_eq!(
            one.family_filter.as_deref(),
            Some(&["fp32_dense".into()][..])
        );
        assert_eq!(
            selected_families(one.family_filter.as_deref())
                .iter()
                .map(|family| family.module_name)
                .collect::<Vec<_>>(),
            vec!["fp32_dense"],
        );

        let multiple = parse_args_from(vec![
            "generated".into(),
            "fp64_dense".into(),
            "fp32_dense".into(),
        ])
        .expect("known families");
        let selected = selected_families(multiple.family_filter.as_deref())
            .iter()
            .map(|family| family.module_name)
            .collect::<Vec<_>>();
        assert_eq!(selected, vec!["fp64_dense", "fp32_dense"]);

        let all = parse_args_from(vec!["generated".into()]).expect("all families");
        assert!(all.family_filter.is_none());
        assert_eq!(selected_families(None).len(), ALL_GENERATED_FAMILIES.len());

        let progress = parse_args_from(vec![
            "generated".into(),
            "--row-progress".into(),
            "fp32_dense".into(),
        ])
        .expect("row progress");
        assert!(progress.row_progress);

        let report = parse_args_from(vec![
            "generated".into(),
            "--check-catalog".into(),
            "--catalog-report".into(),
            "report.tsv".into(),
        ]);
        if cfg!(feature = "catalog-check") {
            assert_eq!(
                report.expect("catalog report").catalog_report,
                Some(PathBuf::from("report.tsv"))
            );
        } else {
            assert!(report
                .err()
                .expect("catalog check feature")
                .contains("catalog-check"));
        }
        assert!(parse_args_from(vec![
            "generated".into(),
            "--catalog-report".into(),
            "report.tsv".into(),
        ])
        .err()
        .expect("report requires comparison")
        .contains("requires --check-catalog"));

        let unknown = parse_args_from(vec!["generated".into(), "not_a_family".into()])
            .err()
            .expect("unknown family must reject");
        assert!(unknown.contains("unknown schedule family"));
    }

    #[test]
    fn explicit_scalar_sweep_replaces_default_catalog_work() {
        let family = family_by_name("fp128_onehot").expect("known family");
        let explicit_rows = ExplicitRows {
            final_group: Some(parse_explicit_group("fp128_onehot:14:1").expect("explicit group")),
            precommitted_groups: Vec::new(),
        };

        let spec = emit_spec_with_overrides(
            family,
            &GenerationPreplans::default(),
            PathBuf::from("generated"),
            &explicit_rows,
            "generator command",
        )
        .expect("explicit emit spec");

        assert_eq!(spec.keys, vec![PolynomialGroupLayout::new(14, 1)]);
        assert!(spec.group_batch_keys.is_empty());
        assert_eq!(spec.generator_command, "generator command");
    }

    #[test]
    fn explicit_group_rejects_source_metadata() {
        assert!(parse_explicit_group("fp128_onehot:14:1:256").is_err());
    }

    #[cfg(feature = "catalog-check")]
    #[test]
    fn catalog_comparison_reports_complete_key_union() {
        let family = family_by_name("fp32_dense").expect("known family");
        let table = (family.schedule_catalog)().expect("compiled fp32 dense table");
        let spec = wiring_emit_spec(family, PathBuf::from("generated"));
        let entries = table
            .entries
            .iter()
            .copied()
            .map(|entry| {
                let key = entry.to_runtime_lookup_key();
                let schedule = akita_schedules::schedule_from_entry(
                    &entry,
                    &key,
                    &spec.policy,
                    spec.ring_challenge_config,
                )
                .expect("expand compiled row");
                assert_eq!(
                    akita_schedules::expanded_schedule_proof_payload_bytes(
                        &key,
                        &schedule,
                        &spec.policy,
                    )
                    .expect("expanded proof payload"),
                    akita_schedules::estimate_proof_bytes(
                        &entry,
                        &key,
                        &spec.policy,
                        spec.ring_challenge_config,
                    )
                    .expect("generated proof payload"),
                );
                (key, schedule)
            })
            .collect::<Vec<_>>();

        let equal = compare_materialized_catalog(&spec, table, &entries).expect("equal report");
        assert_eq!(equal.changed_rows, 0);
        assert_eq!(equal.report.matches("\tequal\t").count(), entries.len());
        assert!(CATALOG_REPORT_HEADER.ends_with("old_policy\tnew_policy\n"));
        assert!(equal
            .report
            .lines()
            .all(|line| line.split('\t').count() == 13));

        let removed = compare_materialized_catalog(&spec, table, &entries[..entries.len() - 1])
            .expect("removed report");
        assert_eq!(removed.changed_rows, 1);
        assert!(removed.report.contains("\tremoved\t"));

        let empty_table = akita_schedules::GeneratedScheduleTable {
            entries: &[],
            identity: table.identity,
        };
        let added =
            compare_materialized_catalog(&spec, empty_table, &entries[..1]).expect("added report");
        assert_eq!(added.changed_rows, 1);
        assert!(added.report.contains("\tadded\t"));

        let mut changed_entries = entries.clone();
        changed_entries[0].1.root.input_witness_len += 1;
        let changed =
            compare_materialized_catalog(&spec, table, &changed_entries).expect("changed report");
        assert_eq!(changed.changed_rows, 1);
        assert!(changed.report.contains("\tchanged\t"));
    }

    #[cfg(feature = "catalog-check")]
    #[test]
    fn generated_w8r2_row_preserves_the_two_level_packing_boundary() {
        use akita_types::OpeningMethod;

        let family =
            family_by_name("fp128_onehot_recursive_multi_chunk_w8r2").expect("known W8R2 family");
        let table = (family.schedule_catalog)().expect("compiled W8R2 table");
        assert_eq!(table.entries.len(), 1);
        let entry = table.entries[0];
        let key = entry.to_runtime_lookup_key();
        let spec = wiring_emit_spec(family, PathBuf::from("generated"));
        let expand = || {
            akita_schedules::schedule_from_entry(
                &entry,
                &key,
                &spec.policy,
                spec.ring_challenge_config,
            )
            .expect("expand W8R2 row")
        };
        let schedule = expand();
        assert_eq!(schedule, expand(), "generated replay must be deterministic");
        schedule.validate_structure().expect("valid W8R2 schedule");

        assert_eq!(
            schedule.root.params.final_group.commitment.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64,
            },
        );
        assert_eq!(schedule.root.params.precommitted_groups.len(), 2);
        assert!(schedule
            .root
            .params
            .precommitted_groups
            .iter()
            .all(|group| {
                group.commitment.opening.opening_method
                    == OpeningMethod::SubringCoefficientPacking {
                        challenge_subring_dimension: 128,
                    }
            }));

        let first_recursive = schedule
            .recursive_folds
            .first()
            .expect("W8R2 row has a recursive packing fold");
        assert_eq!(
            first_recursive.params.witness.opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64,
            },
        );
        assert_eq!(
            first_recursive
                .params
                .incoming_setup_prefix
                .as_ref()
                .expect("first recursive fold consumes the setup prefix")
                .commitment_params
                .opening
                .opening_method,
            OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension: 64,
            },
        );
        assert!(schedule.recursive_folds[1..].iter().all(|fold| fold
            .params
            .witness
            .opening_method
            == OpeningMethod::EvaluationTrace));

        let policy_signature =
            catalog_policy_signature(&spec, &schedule).expect("W8R2 policy signature");
        assert!(policy_signature.contains("L0[chunks=8@2,eor=0,in="));
        assert!(
            policy_signature.contains("witness=PACK,s=64,h=4,w=64,src=canonical,dA=256,sec=Linf")
        );
        assert!(
            policy_signature.contains("pre0=PACK,s=128,h=2,w=128,src=canonical,dA=256,sec=Linf")
        );
        assert!(policy_signature.contains("L1[chunks=8@2,eor=0,in="));
        assert!(policy_signature.contains("prefix=PACK,s=64"));
        assert!(policy_signature.contains("L2[chunks=1@0,eor=0,in="));
        assert!(policy_signature.contains("witness=ET,s=-,h=-,w=-"));
        assert!(policy_signature.contains("/T[method=ET,src=canonical,eor=0,input="));
        assert!(!policy_signature.contains(['\t', '\n']));

        let terminal_eor = akita_types::extension_opening_reduction_level_bytes(
            spec.policy
                .challenge_field_bits()
                .expect("challenge field bits"),
            spec.policy.claim_ext_degree,
            akita_types::PolynomialGroupLayout::singleton(
                akita_types::padded_boolean_opening_vars(schedule.terminal.input_witness_len)
                    .expect("terminal opening vars"),
            ),
        )
        .expect("terminal EOR price");
        assert_eq!(
            terminal_eor, 0,
            "the fp128 base-field terminal follows the ET/EOR pricing path, whose width-one reduction is empty",
        );
        assert_eq!(
            akita_schedules::expanded_schedule_proof_payload_bytes(&key, &schedule, &spec.policy,)
                .expect("expanded proof payload"),
            akita_schedules::estimate_proof_bytes(
                &entry,
                &key,
                &spec.policy,
                spec.ring_challenge_config,
            )
            .expect("generated proof payload"),
        );

        assert_eq!(
            source_encoding_signature(
                akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                    extension_degree: 2,
                }
            ),
            "tensor-k2",
        );
        assert_ne!(
            source_encoding_signature(
                akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                    extension_degree: 2,
                }
            ),
            source_encoding_signature(
                akita_types::CommittedSourceEncoding::TensorSubfieldProjection {
                    extension_degree: 4,
                }
            ),
        );

        let mut activation_changed = schedule.clone();
        activation_changed
            .root
            .params
            .final_group
            .commitment
            .witness_chunk
            .num_activated_levels = 1;
        assert_ne!(
            policy_signature,
            catalog_policy_signature(&spec, &activation_changed)
                .expect("activation policy signature"),
        );

        let mut input_changed = schedule.clone();
        input_changed.root.input_witness_len += 1;
        assert_ne!(
            policy_signature,
            catalog_policy_signature(&spec, &input_changed).expect("input-length policy signature"),
        );
    }

    #[test]
    fn explicit_sweeps_reject_the_checked_in_generated_tree() {
        let explicit_rows = ExplicitRows {
            final_group: Some(parse_explicit_group("fp128_onehot:14:1").expect("explicit group")),
            precommitted_groups: Vec::new(),
        };
        let checked_in_generated_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../akita-schedules/src/generated");

        let error = validate_explicit_output_isolation(
            &checked_in_generated_dir.join("diagnostic"),
            &explicit_rows,
        )
        .expect_err("checked-in generated tree must be protected");
        assert!(error.contains("isolated output directory"));

        let isolated = env::temp_dir().join(format!(
            "akita-explicit-schedule-test-{}",
            std::process::id()
        ));
        validate_explicit_output_isolation(&isolated, &explicit_rows)
            .expect("isolated explicit output");
        validate_explicit_output_isolation(&checked_in_generated_dir, &ExplicitRows::default())
            .expect("ordinary full regeneration may target the checked-in catalog");
    }

    #[cfg(unix)]
    #[test]
    fn output_resolution_applies_parent_after_resolving_symlink() {
        use std::os::unix::fs::symlink;

        let root = env::temp_dir().join(format!(
            "akita-schedule-path-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let target = root.join("real/deep");
        fs::create_dir_all(&target).expect("create symlink target");
        symlink(&target, root.join("link")).expect("create test symlink");

        let resolved = resolved_output_path(&root.join("link/../isolated"))
            .expect("resolve output through symlink");
        let canonical_root = fs::canonicalize(&root).expect("canonical test root");
        assert_eq!(resolved, canonical_root.join("real/isolated"));

        fs::remove_dir_all(&root).expect("remove test directory");
    }
}
