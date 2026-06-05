//! Generate schedule tables using the DP planner.
//!
//! Produces one module per family in `GeneratedScheduleTableEntry` format.
//! Each emitted file contains both:
//!
//! - singleton schedules (`num_claims=1`)
//! - batched schedules for 4 polynomials, 1 group, 1 point
//!
//! The family list and per-family `(num_vars, num_claims)` key sequence
//! live in `akita_config::generated_families::ALL_GENERATED_FAMILIES`,
//! which the drift-guard test also consumes — adding a new generated
//! family in one place picks it up in both the emitter and the guard.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use akita_config::generated_families::{family_keys, GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_planner::generated::GeneratedScheduleKey;
use akita_planner::generated_schedule_lookup_key;
use akita_types::sis::min_secure_rank;
use akita_types::{AkitaScheduleLookupKey, DirectStep, FoldStep, LevelParams, Schedule, Step};

/// First-tier `B` rank to store in the compact table.
///
/// For a single-tier level this is just `b_key.row_len()`. For a tiered level
/// the table stores the **un-tiered** rank (the secure rank for the full,
/// pre-split `B` width) so [`GeneratedFoldStep::expand_to_level_params`] can
/// rebuild the un-tiered layout and replay `apply_tiering` to recover the exact
/// `B'`/`F` split. The un-tiered rank equals the pre-tiering rank the DP sized
/// against the full width `b_key.col_len() * tier_split`.
fn untiered_n_b(p: &LevelParams) -> usize {
    if p.f_key.is_none() {
        return p.b_key.row_len();
    }
    let full_width = (p.b_key.col_len() * p.tier_split.max(1)) as u64;
    min_secure_rank(
        p.b_key.sis_family(),
        p.ring_dimension as u32,
        p.b_key.collision_inf(),
        full_width,
    )
    .expect("tiered B' level must have a SIS-secure un-tiered rank for the full width")
}

fn emit_key(key: GeneratedScheduleKey) -> String {
    format!(
        "GeneratedScheduleKey {{ num_vars: {}, num_commitment_groups: {}, num_t_vectors: {}, \
         num_w_vectors: {}, num_z_vectors: {} }}",
        key.num_vars,
        key.num_commitment_groups,
        key.num_t_vectors,
        key.num_w_vectors,
        key.num_z_vectors,
    )
}

fn emit_fold_struct(p: &LevelParams) -> String {
    format!(
        "GeneratedFoldStep {{ \
         ring_d: {}, log_basis: {}, m_vars: {}, r_vars: {}, n_a: {}, n_b: {}, n_d: {} }}",
        p.ring_dimension,
        p.log_basis,
        p.log_block_len(),
        p.log_num_blocks(),
        p.a_key.row_len(),
        untiered_n_b(p),
        p.d_key.row_len(),
    )
}

fn emit_fold(step: &FoldStep) -> String {
    format!(
        "        GeneratedStep::Fold({}),",
        emit_fold_struct(&step.params)
    )
}

fn emit_direct(direct: &DirectStep) -> String {
    // A root-direct schedule carries its brute-forced root commit layout in
    // `DirectStep.params`; a terminal-direct step ships the cleartext
    // witness without committing and carries no commit payload.
    match &direct.params {
        Some(commit) => format!(
            "        GeneratedStep::Direct(GeneratedDirectStep {{ commit: Some({}) }}),",
            emit_fold_struct(commit)
        ),
        None => "        GeneratedStep::Direct(GeneratedDirectStep { commit: None }),".to_string(),
    }
}

fn emit_schedule_entry(out: &mut String, key_str: &str, schedule: &Schedule) -> Result<(), String> {
    writeln!(
        out,
        "    GeneratedScheduleTableEntry {{ key: {key_str}, steps: &[",
    )
    .map_err(|e| e.to_string())?;

    for step in &schedule.steps {
        match step {
            Step::Fold(fold) => {
                writeln!(out, "{}", emit_fold(fold)).map_err(|e| e.to_string())?;
            }
            Step::Direct(direct) => {
                writeln!(out, "{}", emit_direct(direct)).map_err(|e| e.to_string())?;
            }
        }
    }

    writeln!(out, "    ] }},").map_err(|e| e.to_string())
}

fn output_module_name(family: &GeneratedFamily) -> String {
    #[cfg(feature = "zk")]
    {
        format!("{}_zk", family.module_name)
    }
    #[cfg(not(feature = "zk"))]
    {
        family.module_name.to_string()
    }
}

fn output_const_name(family: &GeneratedFamily) -> String {
    #[cfg(feature = "zk")]
    {
        let base = family
            .const_name
            .strip_suffix("_SCHEDULES")
            .expect("generated schedule const name should end in _SCHEDULES");
        format!("{base}_ZK_SCHEDULES")
    }
    #[cfg(not(feature = "zk"))]
    {
        family.const_name.to_string()
    }
}

fn generator_command() -> &'static str {
    #[cfg(feature = "zk")]
    {
        "cargo run --release -p akita-config --features zk --bin gen_schedule_tables -- \
         <output-dir>"
    }
    #[cfg(not(feature = "zk"))]
    {
        "cargo run --release -p akita-config --bin gen_schedule_tables -- <output-dir>"
    }
}

fn emit_module(family: &GeneratedFamily) -> Result<String, String> {
    let mut out = String::new();
    let const_name = output_const_name(family);
    writeln!(out, "// Generated by `{}`", generator_command()).map_err(|e| e.to_string())?;
    // A schedule may contain only `Direct` steps (no `Fold`), e.g. when the
    // SIS-floor tables do not admit a secure commitment for the level's
    // collision bound and the DP falls back to a cleartext schedule. In that
    // case `GeneratedFoldStep` is unused, so allow the dead import rather than
    // making the emitted module shape depend on the schedule contents.
    writeln!(out, "#[allow(unused_imports)]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::{{\n    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleKey, \
         GeneratedScheduleTableEntry,\n    GeneratedStep,\n}};"
    )
    .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;
    writeln!(out, "#[rustfmt::skip]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "pub(crate) static {const_name}: &[GeneratedScheduleTableEntry] = &["
    )
    .map_err(|e| e.to_string())?;

    let keys: Vec<AkitaScheduleLookupKey> =
        family_keys(family).map_err(|e| format!("{}: build keys: {e}", family.module_name))?;
    for key in keys {
        let schedule = (family.regen)(key)
            .map_err(|e| format!("{}: regen {key:?}: {e}", family.module_name))?;
        let key_str = emit_key(generated_schedule_lookup_key(key));
        emit_schedule_entry(&mut out, &key_str, &schedule)?;
    }

    writeln!(out, "];").map_err(|e| e.to_string())?;
    Ok(out)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return Err(
            "usage: cargo run --release -p akita-config --bin gen_schedule_tables -- \
             <output-dir> [family_module_name ...]"
                .to_string(),
        );
    }
    let base_dir = PathBuf::from(&args[0]);
    fs::create_dir_all(&base_dir).map_err(|e| format!("create {}: {e}", base_dir.display()))?;

    let filter: Option<Vec<&str>> = if args.len() > 1 {
        Some(args[1..].iter().map(|s| s.as_str()).collect())
    } else {
        None
    };

    for family in ALL_GENERATED_FAMILIES {
        if let Some(ref names) = filter {
            if !names.contains(&family.module_name) {
                continue;
            }
        }
        let body = emit_module(family)?;
        let dest = base_dir.join(format!("{}.rs", output_module_name(family)));
        fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
        println!("wrote {}", dest.display());
    }

    Ok(())
}
