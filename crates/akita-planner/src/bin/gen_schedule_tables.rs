//! Generate schedule tables using the offline DP planner.

use akita_planner::generated_families::{
    emit_spec_for_family, wiring_emit_spec, GeneratedFamily, ALL_GENERATED_FAMILIES,
};
use akita_planner::{
    refresh_generated_wiring, run_regen_fmt, write_family_module,
    write_precommitted_profiles_module, EmitSpec,
};
use akita_types::{AkitaScheduleLookupKey, CommittedGroupProfile, PolynomialGroupLayout};
use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Default)]
struct ExplicitRows {
    final_group: Option<ExplicitGroup>,
    precommitted_groups: Vec<ExplicitGroup>,
}

struct ParsedArgs {
    base_dir: PathBuf,
    wiring_only: bool,
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
     --bin gen_schedule_tables -- <output-dir> [--wiring-only] [family_module_name ...] \
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
    let mut family_args = Vec::new();
    let mut explicit_rows = ExplicitRows::default();
    let mut i = 1;
    while i < raw_args.len() {
        match raw_args[i].as_str() {
            "--wiring-only" => {
                wiring_only = true;
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
    Ok(ParsedArgs {
        base_dir,
        wiring_only,
        family_filter,
        explicit_rows,
    })
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

fn push_unique_profile(profiles: &mut Vec<CommittedGroupProfile>, profile: CommittedGroupProfile) {
    if !profiles.contains(&profile) {
        profiles.push(profile);
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
                    (precommitted_family.explicit_precommitted_group)(layout)
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
    base_dir: PathBuf,
    explicit_rows: &ExplicitRows,
    generator_command: &'static str,
) -> Result<EmitSpec, String> {
    let mut spec = emit_spec_for_family(family, base_dir, generator_command)
        .map_err(|e| format!("{}: emit spec: {e}", family.module_name))?;
    if !explicit_rows.has_family(family) {
        return Ok(spec);
    }

    let final_group = explicit_rows
        .final_group
        .as_ref()
        .ok_or_else(|| format!("{}: missing --final-group", family.module_name))?;
    spec.keys.clear();
    spec.group_batch_keys.clear();
    spec.precommitted_profiles.clear();
    let final_layouts = final_group.layouts();

    if explicit_rows.precommitted_groups.is_empty() {
        for layout in final_layouts {
            push_unique_layout(&mut spec.keys, layout);
        }
        spec.keys
            .sort_by_key(|key| (key.num_vars(), key.num_polynomials()));
        return Ok(spec);
    }

    let precommitted_choices = expand_precommitted_choices(&explicit_rows.precommitted_groups)?;
    let mut precommitted_combinations = Vec::new();
    push_precommitted_combinations(
        &precommitted_choices,
        0,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut precommitted_combinations,
    );

    for (precommitteds, precommitted_honest_fold_policies) in precommitted_combinations {
        for profile in &precommitteds {
            push_unique_profile(&mut spec.precommitted_profiles, *profile);
        }
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
    spec.precommitted_profiles
        .sort_by_key(CommittedGroupProfile::canonical_descriptor_bytes);
    Ok(spec)
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
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

    if !args.wiring_only {
        let generator_command = generator_command();
        let specs = families_to_write
            .iter()
            .map(|family| {
                emit_spec_with_overrides(
                    family,
                    args.base_dir.clone(),
                    &args.explicit_rows,
                    generator_command,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for spec in &specs {
            let dest = write_family_module(spec)
                .map_err(|e| format!("{}: write family module: {e}", spec.module_name))?;
            println!("wrote {}", dest.display());
            let dest = write_precommitted_profiles_module(spec).map_err(|e| {
                format!("{}: write precommit registry module: {e}", spec.module_name)
            })?;
            println!("wrote {}", dest.display());
        }
    }

    let mod_path = args.base_dir.join("mod.rs");
    let wiring_specs = ALL_GENERATED_FAMILIES
        .iter()
        .map(|family| wiring_emit_spec(family, args.base_dir.clone()))
        .collect::<Vec<_>>();
    if mod_path.exists() {
        refresh_generated_wiring(&sorted_unique_specs(&wiring_specs), &mod_path)?;
        println!("updated {}", mod_path.display());
    } else if args.wiring_only {
        return Err(format!("missing {}", mod_path.display()));
    } else {
        println!("skipped missing {}", mod_path.display());
    }
    run_regen_fmt()?;
    Ok(())
}
