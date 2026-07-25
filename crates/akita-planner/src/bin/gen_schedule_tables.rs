//! Generate schedule tables using the offline DP planner.

use std::env;
use std::fs;
use std::path::PathBuf;

use akita_planner::generated_families::{
    emit_spec_for_family, wiring_emit_spec, ALL_GENERATED_FAMILIES,
};
use akita_planner::{refresh_generated_wiring, run_regen_fmt, write_family_module, EmitSpec};

fn generator_command() -> &'static str {
    "cargo run --release -p akita-planner --features catalog-gen --bin gen_schedule_tables -- <output-dir>"
}

fn sorted_unique_specs(specs: &[EmitSpec]) -> Vec<EmitSpec> {
    let mut out: Vec<EmitSpec> = specs.to_vec();
    out.sort_by_key(|spec| spec.module_name);
    out.dedup_by_key(|spec| spec.module_name);
    out
}

fn main() -> Result<(), String> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    if raw_args.is_empty() {
        return Err(
            "usage: cargo run --release -p akita-planner --features catalog-gen \
             --bin gen_schedule_tables -- <output-dir> [--wiring-only] [family_module_name ...]"
                .to_string(),
        );
    }
    let base_dir = PathBuf::from(&raw_args[0]);
    let wiring_only = raw_args.iter().any(|arg| arg == "--wiring-only");
    let family_args: Vec<&str> = raw_args
        .iter()
        .skip(1)
        .map(String::as_str)
        .filter(|arg| *arg != "--wiring-only")
        .collect();
    fs::create_dir_all(&base_dir).map_err(|e| format!("create {}: {e}", base_dir.display()))?;

    let filter: Option<Vec<&str>> = if family_args.is_empty() {
        None
    } else {
        Some(family_args)
    };
    if wiring_only && filter.is_some() {
        return Err("--wiring-only does not accept family filters".to_string());
    }
    if let Some(names) = &filter {
        let unknown = names
            .iter()
            .copied()
            .filter(|name| {
                !ALL_GENERATED_FAMILIES
                    .iter()
                    .any(|family| family.module_name == *name)
            })
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(format!("unknown schedule family: {}", unknown.join(", ")));
        }
    }

    let families_to_write = ALL_GENERATED_FAMILIES
        .iter()
        .filter(|family| {
            filter
                .as_ref()
                .is_none_or(|names| names.contains(&family.module_name))
        })
        .collect::<Vec<_>>();

    if !wiring_only {
        let generator_command = generator_command();
        let specs = families_to_write
            .iter()
            .map(|family| {
                emit_spec_for_family(family, base_dir.clone(), generator_command)
                    .map_err(|e| format!("{}: emit spec: {e}", family.module_name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for spec in &specs {
            let dest = write_family_module(spec)
                .map_err(|e| format!("{}: write family module: {e}", spec.module_name))?;
            println!("wrote {}", dest.display());
        }
    }

    let mod_path = base_dir.join("mod.rs");
    let wiring_specs = ALL_GENERATED_FAMILIES
        .iter()
        .map(|family| wiring_emit_spec(family, base_dir.clone()))
        .collect::<Vec<_>>();
    refresh_generated_wiring(&sorted_unique_specs(&wiring_specs), &mod_path)?;
    println!("updated {}", mod_path.display());
    run_regen_fmt()?;
    Ok(())
}
