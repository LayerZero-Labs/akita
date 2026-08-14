//! Generate schedule tables using the offline DP planner.

use akita_planner::emit::{bounded_parallel_filter_map, offline_planning_worker_count};
use akita_planner::generated_families::{
    emit_spec_for_family, wiring_emit_spec, GeneratedFamily, GenerationPreplans,
    ALL_GENERATED_FAMILIES,
};
use akita_planner::{
    publish_generated_outputs, render_generated_outputs_with_validation, EmitSpec,
};
use akita_types::{AkitaScheduleLookupKey, CommittedGroupProfile, PolynomialGroupLayout};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Default)]
struct ExplicitRows {
    final_group: Option<ExplicitGroup>,
    precommitted_groups: Vec<ExplicitGroup>,
}

struct ParsedArgs {
    base_dir: PathBuf,
    wiring_only: bool,
    check_catalog: bool,
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
     [family_module_name ...] \
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
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if raw_args.is_empty() {
        return Err(usage().to_string());
    }
    let base_dir = PathBuf::from(&raw_args[0]);
    let mut wiring_only = false;
    let mut check_catalog = false;
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
    Ok(ParsedArgs {
        base_dir,
        wiring_only,
        check_catalog,
        family_filter,
        explicit_rows,
    })
}

fn validate_materialized_catalog(
    spec: &EmitSpec,
    entries: &[akita_planner::emit::MaterializedEntry],
) -> Result<(), String> {
    let family = family_by_name(spec.module_name)
        .ok_or_else(|| format!("unknown generated family: {}", spec.module_name))?;
    if (family.schedule_catalog)().is_none() {
        return Err(format!(
            "{}: compiled catalog is unavailable; build with all schedule features",
            spec.module_name
        ));
    }
    for (key, expected) in entries {
        let actual = (family.resolve_catalog_row_for_key)(key.clone())
            .map_err(|error| format!("{}: resolve {key:?}: {error}", spec.module_name))?;
        if actual != *expected {
            return Err(format!(
                "{}: compiled catalog row {key:?} disagrees with the planner",
                spec.module_name
            ));
        }
    }
    Ok(())
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
    let args = parse_args()?;
    validate_explicit_output_isolation(&args.base_dir, &args.explicit_rows)?;
    fs::create_dir_all(&args.base_dir)
        .map_err(|e| format!("create {}: {e}", args.base_dir.display()))?;
    let families_to_write = ALL_GENERATED_FAMILIES
        .iter()
        .filter(|family| {
            args.family_filter
                .as_ref()
                .is_none_or(|names| names.iter().any(|name| name == family.module_name))
        })
        .collect::<Vec<_>>();

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
            eprintln!(
                "planning schedule family {}/{}: {}",
                index + 1,
                family_count,
                family.module_name
            );
            emit_spec_with_overrides(
                family,
                &preplans,
                args.base_dir.clone(),
                &args.explicit_rows,
                generator_command,
            )
            .map(Some)
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
    let outputs = render_generated_outputs_with_validation(
        &specs,
        &sorted_unique_specs(&wiring_specs),
        mod_path.as_deref(),
        |spec, entries| {
            if check_catalog {
                validate_materialized_catalog(spec, entries)
            } else {
                Ok(())
            }
        },
    )?;
    for destination in publish_generated_outputs(outputs)? {
        println!("wrote {}", destination.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
