//! Generate external schedule artifacts using the offline DP planner.

mod catalog_policy_report;
mod catalog_snapshot;
mod generation_output_path;

use catalog_policy_report::catalog_policy_signature;
use generation_output_path::validate_explicit_output_isolation;

use akita_planner::emit::{
    bounded_parallel_filter_map, offline_planning_worker_count, GroupedGenerationRequest,
    MaterializationDiagnostics, PrecommittedProducer,
};
use akita_planner::generated_families::{
    emit_spec_for_family, empty_emit_spec, GeneratedFamily, GenerationPreplans,
    ALL_GENERATED_FAMILIES,
};
use akita_planner::{
    publish_artifact_outputs, render_schedule_artifact_outputs_with_validation, EmitSpec,
};
use akita_types::{
    schedule_row_digest, AkitaScheduleLookupKey, CommittedGroupBatchProfile, FoldSchedule,
    GroupCommitPhaseParams, PolynomialGroupLayout,
};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Default)]
struct ExplicitRows {
    final_group: Option<ExplicitGroup>,
    precommitted_groups: Vec<ExplicitGroup>,
}

struct ParsedArgs {
    base_dir: PathBuf,
    check_catalog: bool,
    catalog_report: Option<PathBuf>,
    catalog_snapshot: Option<PathBuf>,
    catalog_baseline: Option<PathBuf>,
    require_catalog_baseline_match: bool,
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

fn usage() -> &'static str {
    "usage: cargo run --release -p akita-planner --features catalog-gen \
     --bin gen_schedule_artifacts -- <output-dir> [--check-catalog] \
     [--catalog-report <path>] [--catalog-snapshot <path>] \
     [--catalog-baseline <snapshot>] [--require-catalog-baseline-match] \
     [--row-progress] \
     [family_name ...]\n\
     positional family names select only those generated families; omit them \
     to generate every family \
     [--final-group family:num_vars_or_range:num_polys_or_range] \
     [--precommitted-group family:num_vars_or_range:num_polys_or_range ...]"
}

fn known_family(name: &str) -> bool {
    ALL_GENERATED_FAMILIES
        .iter()
        .any(|family| family.family_name() == name)
}

fn family_by_name(name: &str) -> Option<&'static GeneratedFamily> {
    ALL_GENERATED_FAMILIES
        .iter()
        .find(|family| family.family_name() == name)
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
    let mut check_catalog = false;
    let mut catalog_report = None;
    let mut catalog_snapshot = None;
    let mut catalog_baseline = None;
    let mut require_catalog_baseline_match = false;
    let mut row_progress = false;
    let mut family_args = Vec::new();
    let mut explicit_rows = ExplicitRows::default();
    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
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
            "--catalog-snapshot" => {
                let value = raw_args
                    .get(i + 1)
                    .ok_or_else(|| "--catalog-snapshot requires a path".to_string())?;
                if catalog_snapshot.is_some() {
                    return Err("--catalog-snapshot may be supplied only once".to_string());
                }
                catalog_snapshot = Some(PathBuf::from(value));
                i += 2;
            }
            "--catalog-baseline" => {
                let value = raw_args
                    .get(i + 1)
                    .ok_or_else(|| "--catalog-baseline requires a path".to_string())?;
                if catalog_baseline.is_some() {
                    return Err("--catalog-baseline may be supplied only once".to_string());
                }
                catalog_baseline = Some(PathBuf::from(value));
                i += 2;
            }
            "--require-catalog-baseline-match" => {
                require_catalog_baseline_match = true;
                i += 1;
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
        None
    } else {
        Some(family_args)
    };
    let catalog_rows_requested =
        check_catalog || catalog_snapshot.is_some() || catalog_baseline.is_some();
    if catalog_rows_requested && explicit_rows.final_group.is_some() {
        return Err("catalog checks and snapshots require the standard artifact rows".to_string());
    }
    if check_catalog && !cfg!(feature = "catalog-check") {
        return Err("--check-catalog requires the `catalog-check` feature".to_string());
    }
    if catalog_report.is_some() && !check_catalog && catalog_baseline.is_none() {
        return Err("--catalog-report requires --check-catalog or --catalog-baseline".to_string());
    }
    if catalog_baseline.is_some() && family_filter.is_some() {
        return Err("--catalog-baseline requires the complete generated family set".to_string());
    }
    if require_catalog_baseline_match && catalog_baseline.is_none() {
        return Err("--require-catalog-baseline-match requires --catalog-baseline".to_string());
    }
    Ok(ParsedArgs {
        base_dir,
        check_catalog,
        catalog_report,
        catalog_snapshot,
        catalog_baseline,
        require_catalog_baseline_match,
        row_progress,
        family_filter,
        explicit_rows,
    })
}

fn selected_families(family_filter: Option<&[String]>) -> Vec<&'static GeneratedFamily> {
    ALL_GENERATED_FAMILIES
        .iter()
        .filter(|family| {
            family_filter.is_none_or(|names| names.iter().any(|name| name == family.family_name()))
        })
        .collect()
}

fn validate_materialized_catalog(
    spec: &EmitSpec,
    entries: &[akita_planner::emit::MaterializedEntry],
) -> Result<CatalogComparison, String> {
    let rows = entries
        .iter()
        .map(|entry| {
            let key = entry.key();
            let schedule = entry.schedule().clone();
            let final_group =
                GroupCommitPhaseParams::try_from_params(key.final_group, &schedule.root.params)
                    .map_err(|error| format!("derive final committed profile: {error}"))?;
            Ok((
                CommittedGroupBatchProfile {
                    final_group,
                    precommitteds: key.precommitteds,
                },
                schedule,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let regenerated = akita_schedules::TrustedScheduleCatalog::try_new(
        spec.family_name,
        rows,
        &spec.policy,
        spec.ring_challenge_config,
    )
    .map_err(|error| format!("build regenerated catalog: {error}"))?
    .to_artifact_bytes()
    .map_err(|error| format!("encode regenerated catalog: {error}"))?;
    let path = spec.output_dir.join(format!("{}.aks", spec.family_name));
    let existing = fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let changed_rows = usize::from(existing != regenerated);
    Ok(CatalogComparison {
        report: if changed_rows == 0 {
            String::new()
        } else {
            format!("{}\tchanged\tartifact-bytes\n", spec.family_name)
        },
        changed_rows,
    })
}

struct CatalogComparison {
    report: String,
    changed_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogRowMetrics {
    setup_fields: usize,
    first_direct_setup_capacity: Option<usize>,
    proof_bytes: usize,
    fold_levels: usize,
    row_digest: String,
    policy_signature: String,
}

const CATALOG_DRIFT_REPORT_HEADER: &str = "family\tstatus\tdetail\n";

fn row_digest_hex(key: &AkitaScheduleLookupKey, schedule: &FoldSchedule) -> Result<String, String> {
    let final_group =
        GroupCommitPhaseParams::try_from_params(key.final_group, &schedule.root.params)
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
    let first_direct_setup_capacity = (matches!(
        spec.policy.selection_policy,
        akita_schedules::SelectionPolicyId::MinFirstDirectSetupThenPayloadV2
            | akita_schedules::SelectionPolicyId::MinPaddedSetupEnvelopeThenFirstDirectThenPayloadV3
    ))
    .then(|| {
        akita_schedules::planner_support::first_direct_setup_capacity_for_schedule(
            schedule,
            &key.opening_layout()?,
        )
    })
    .transpose()
    .map_err(|error| format!("estimate first direct setup capacity: {error}"))?;
    Ok(CatalogRowMetrics {
        setup_fields,
        first_direct_setup_capacity,
        proof_bytes,
        fold_levels: schedule.num_fold_levels(),
        row_digest: row_digest_hex(key, schedule)?,
        policy_signature: catalog_policy_signature(spec, schedule)?,
    })
}

fn catalog_logical_key(key: &AkitaScheduleLookupKey) -> String {
    use std::fmt::Write as _;

    let mut logical = format!(
        "final={}:{};precommitted=",
        key.final_group.num_vars(),
        key.final_group.num_polynomials(),
    );
    for (index, precommitted) in key.precommitteds.iter().enumerate() {
        if index != 0 {
            logical.push(',');
        }
        write!(
            logical,
            "{}:{}",
            precommitted.group.num_vars(),
            precommitted.group.num_polynomials(),
        )
        .expect("writing to String cannot fail");
    }
    logical
}

fn catalog_lookup_key_digest(key: &AkitaScheduleLookupKey) -> String {
    akita_types::instance_descriptor::digest_descriptor_bytes(&key.canonical_descriptor_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn catalog_snapshot_row(
    spec: &EmitSpec,
    key: &AkitaScheduleLookupKey,
    schedule: &FoldSchedule,
    logical_key: String,
) -> Result<catalog_snapshot::CatalogSnapshotRow, String> {
    let metrics = catalog_row_metrics(spec, key, schedule)?;
    Ok(catalog_snapshot::CatalogSnapshotRow {
        schema: catalog_snapshot::SnapshotSchema::Current,
        family: spec.family_name.to_string(),
        logical_key,
        lookup_key_digest: catalog_lookup_key_digest(key),
        setup_fields: metrics.setup_fields,
        first_direct_setup_capacity: metrics.first_direct_setup_capacity,
        proof_bytes: metrics.proof_bytes,
        fold_levels: metrics.fold_levels,
        row_digest: metrics.row_digest,
        policy: metrics.policy_signature,
    })
}

fn materialized_snapshot_rows(
    spec: &EmitSpec,
    entries: &[akita_planner::emit::MaterializedEntry],
) -> Result<Vec<catalog_snapshot::CatalogSnapshotRow>, String> {
    let mut logical_key_counts = std::collections::BTreeMap::new();
    for entry in entries {
        *logical_key_counts
            .entry(catalog_logical_key(&entry.key()))
            .or_insert(0usize) += 1;
    }
    entries
        .iter()
        .map(|entry| {
            let key = entry.key();
            let mut logical_key = catalog_logical_key(&key);
            let is_ambiguous = logical_key_counts
                .get(&logical_key)
                .copied()
                .unwrap_or_default()
                > 1;
            let producers_match_family = entry
                .precommitted_producers()
                .iter()
                .all(|producer| producer.source_contract() == spec.source_contract);
            if is_ambiguous && !producers_match_family {
                use std::fmt::Write as _;
                logical_key.push_str(";producer_contracts=");
                for (index, producer) in entry.precommitted_producers().iter().enumerate() {
                    if index != 0 {
                        logical_key.push(',');
                    }
                    let contract = producer.source_contract();
                    match contract.class() {
                        akita_types::sis::CommittedSourceClass::UnitOneHot {
                            source_chunk_size,
                        } => write!(
                            logical_key,
                            "onehot(chunk={source_chunk_size},bound={})",
                            contract.decomposition().log_commit_bound,
                        ),
                        akita_types::sis::CommittedSourceClass::BalancedSignedDigit => write!(
                            logical_key,
                            "balanced(bound={})",
                            contract.decomposition().log_commit_bound,
                        ),
                    }
                    .map_err(|error| format!("write producer contract key: {error}"))?;
                }
            }
            catalog_snapshot_row(spec, &key, entry.schedule(), logical_key)
        })
        .collect()
}

impl ExplicitRows {
    fn has_family(&self, family: &GeneratedFamily) -> bool {
        self.final_group
            .as_ref()
            .is_some_and(|group| group.family == family.family_name())
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

fn push_unique_layout(layouts: &mut Vec<PolynomialGroupLayout>, layout: PolynomialGroupLayout) {
    if !layouts.contains(&layout) {
        layouts.push(layout);
    }
}

fn push_unique_group_batch_key(
    keys: &mut Vec<GroupedGenerationRequest>,
    candidate: GroupedGenerationRequest,
) {
    if !keys.contains(&candidate) {
        keys.push(candidate);
    }
}

fn expand_precommitted_choices(
    preplans: &GenerationPreplans,
    groups: &[ExplicitGroup],
) -> Result<Vec<Vec<PrecommittedProducer>>, String> {
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
    choices: &[Vec<PrecommittedProducer>],
    index: usize,
    producers: &mut Vec<PrecommittedProducer>,
    out: &mut Vec<Vec<PrecommittedProducer>>,
) {
    if index == choices.len() {
        out.push(producers.clone());
        return;
    }
    for producer in &choices[index] {
        producers.push(*producer);
        push_precommitted_combinations(choices, index + 1, producers, out);
        producers.pop();
    }
}

fn emit_spec_with_overrides(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    base_dir: PathBuf,
    explicit_rows: &ExplicitRows,
) -> Result<EmitSpec, String> {
    if !explicit_rows.has_family(family) {
        return emit_spec_for_family(family, preplans, base_dir)
            .map_err(|e| format!("{}: emit spec: {e}", family.family_name()));
    }

    // Explicit sweeps replace the catalog key set. Start from an empty request
    // shape so a one-key diagnostic does not first plan every default grouped
    // root merely to discard those rows below.
    let mut spec = empty_emit_spec(family, base_dir)
        .map_err(|e| format!("{}: producer contract: {e}", family.family_name()))?;

    let final_group = explicit_rows
        .final_group
        .as_ref()
        .ok_or_else(|| format!("{}: missing --final-group", family.family_name()))?;
    spec.keys.clear();
    spec.grouped_requests.clear();
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
        &mut precommitted_combinations,
    );

    for producers in precommitted_combinations {
        for final_layout in &final_layouts {
            push_unique_group_batch_key(
                &mut spec.grouped_requests,
                GroupedGenerationRequest::new(*final_layout, producers.clone()),
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
            "preparing schedule family requests and dependency schedules {}/{}: {}",
            index + 1,
            family_count,
            family.family_name()
        );
        let spec = emit_spec_with_overrides(
            family,
            &preplans,
            args.base_dir.clone(),
            &args.explicit_rows,
        )?;
        eprintln!(
            "prepared schedule family requests and dependency schedules {}/{}: {} ({} scalar keys, {} grouped keys) in {:.2?}",
            index + 1,
            family_count,
            family.family_name(),
            spec.keys.len(),
            spec.grouped_requests.len(),
            family_started.elapsed(),
        );
        Ok(Some(spec))
    })?;
    for (family, spec) in families_to_write.iter().zip(&mut specs) {
        preplans.attach_to_spec(family, spec);
    }
    drop(preplans);
    let check_catalog = args.check_catalog;
    let collect_catalog_snapshot =
        args.catalog_snapshot.is_some() || args.catalog_baseline.is_some();
    let mut catalog_drift_report = if check_catalog {
        CATALOG_DRIFT_REPORT_HEADER.to_string()
    } else {
        String::new()
    };
    let mut changed_catalog_rows = 0usize;
    let mut current_catalog_rows = Vec::new();
    let outputs = render_schedule_artifact_outputs_with_validation(
        &specs,
        MaterializationDiagnostics {
            row_progress: args.row_progress,
        },
        |spec, entries| {
            if check_catalog {
                let comparison = validate_materialized_catalog(spec, entries)?;
                catalog_drift_report.push_str(&comparison.report);
                changed_catalog_rows = changed_catalog_rows
                    .checked_add(comparison.changed_rows)
                    .ok_or_else(|| "catalog comparison row count overflow".to_string())?;
            }
            if collect_catalog_snapshot {
                current_catalog_rows.extend(materialized_snapshot_rows(spec, entries)?);
            }
            Ok(())
        },
    )?;
    if check_catalog {
        if should_emit_catalog_drift_report(args.catalog_baseline.is_some(), changed_catalog_rows) {
            if let Some(path) = &args.catalog_report {
                fs::write(path, &catalog_drift_report)
                    .map_err(|error| format!("write {}: {error}", path.display()))?;
                eprintln!("wrote catalog drift comparison {}", path.display());
            } else {
                eprint!("{catalog_drift_report}");
            }
        }
        if changed_catalog_rows != 0 {
            return Err(format!(
                "checked-in artifact differs from the planner in {changed_catalog_rows} row sets"
            ));
        }
    }
    if let Some(path) = &args.catalog_snapshot {
        let snapshot = catalog_snapshot::write_snapshot(current_catalog_rows.clone())?;
        fs::write(path, snapshot).map_err(|error| format!("write {}: {error}", path.display()))?;
        eprintln!("wrote catalog snapshot {}", path.display());
    }
    if let Some(path) = &args.catalog_baseline {
        let baseline = fs::read_to_string(path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        let comparison = catalog_snapshot::compare_snapshots(
            catalog_snapshot::parse_snapshot(&baseline)?,
            current_catalog_rows,
        )?;
        if let Some(report_path) = &args.catalog_report {
            fs::write(report_path, &comparison.report)
                .map_err(|error| format!("write {}: {error}", report_path.display()))?;
            eprintln!(
                "wrote catalog revision comparison {}",
                report_path.display()
            );
        } else {
            eprint!("{}", comparison.report);
        }
        eprintln!(
            "catalog revision comparison: {} added, {} removed, {} changed, {} equal",
            comparison.added_rows,
            comparison.removed_rows,
            comparison.changed_rows,
            comparison.equal_rows,
        );
        if args.require_catalog_baseline_match
            && (comparison.added_rows != 0
                || comparison.removed_rows != 0
                || comparison.changed_rows != 0)
        {
            return Err(format!(
                "generated catalog differs from the required baseline: {} added, {} removed, {} changed",
                comparison.added_rows, comparison.removed_rows, comparison.changed_rows,
            ));
        }
    }
    let publish_started = args.row_progress.then(Instant::now);
    if args.row_progress {
        eprintln!(
            "schedule generation phase: publish {} artifacts",
            outputs.len(),
        );
    }
    let destinations = publish_artifact_outputs(outputs)?;
    if let Some(started) = publish_started {
        eprintln!(
            "schedule generation phase complete: published {} outputs in {:.2?}",
            destinations.len(),
            started.elapsed(),
        );
    }
    for destination in &destinations {
        println!("wrote {}", destination.display());
    }
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
    Ok(())
}

const fn should_emit_catalog_drift_report(
    has_catalog_baseline: bool,
    changed_catalog_rows: usize,
) -> bool {
    !has_catalog_baseline || changed_catalog_rows != 0
}

#[cfg(test)]
#[path = "gen_schedule_artifacts_tests.rs"]
mod tests;
