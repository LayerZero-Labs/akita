//! Generate schedule tables using the refactored DP planner.
//!
//! Produces two sets of table modules in the same `GeneratedScheduleTableEntry`
//! format as the existing generated tables:
//!
//! - **singleton**: one polynomial per entry (nv 1..50)
//! - **batched-4**: 4 polynomials, 1 group, 1 point (nv 1..50)

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use hachi_pcs::planner::proof_size::ring_vec_bytes;
use hachi_pcs::planner::refactored_planner::{
    find_optimal_batched_schedule, BatchConfig, DirectStep, FoldStep, Schedule, Step,
};
use hachi_pcs::protocol::commitment::presets::fp128;
use hachi_pcs::protocol::commitment::{
    CommitmentConfig, CommitmentPreset, GeneratedAdaptivePolicy,
};

type Fp128D128OneHot =
    CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 128, 1>>;

#[derive(Clone, Copy)]
enum FamilyKind {
    Fp128D128Full,
    Fp128D128OneHot,
    Fp128D32Full,
    Fp128D32OneHot,
    Fp128D64Full,
    Fp128D64OneHot,
}

#[derive(Clone, Copy)]
struct FamilySpec {
    module_name: &'static str,
    const_name: &'static str,
    min_num_vars: usize,
    max_num_vars: usize,
    kind: FamilyKind,
}

const ALL_FAMILIES: &[FamilySpec] = &[
    FamilySpec {
        module_name: "fp128_d128_full",
        const_name: "FP128_D128_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D128Full,
    },
    FamilySpec {
        module_name: "fp128_d128_onehot",
        const_name: "FP128_D128_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D128OneHot,
    },
    FamilySpec {
        module_name: "fp128_d32_full",
        const_name: "FP128_D32_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D32Full,
    },
    FamilySpec {
        module_name: "fp128_d32_onehot",
        const_name: "FP128_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D32OneHot,
    },
    FamilySpec {
        module_name: "fp128_d64_full",
        const_name: "FP128_D64_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D64Full,
    },
    FamilySpec {
        module_name: "fp128_d64_onehot",
        const_name: "FP128_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D64OneHot,
    },
];

fn emit_key(
    nv: usize,
    num_claims: usize,
    num_commitment_groups: usize,
    num_points: usize,
) -> String {
    format!(
        "GeneratedScheduleKey {{ max_num_vars: {nv}, num_vars: {nv}, \
         layout_num_claims: {num_claims}, batch_num_claims: {num_claims}, \
         batch_num_commitment_groups: {num_commitment_groups}, \
         batch_num_points: {num_points} }}"
    )
}

fn emit_fold(step: &FoldStep, label: &str) -> String {
    format!(
        "        GeneratedStep::Fold(GeneratedFoldStep {{ \
         current_w_len: {}, d: {}, log_basis: {}, challenge_l1_mass: {}, \
         m_vars: {}, r_vars: {}, n_a: {}, n_b: {}, n_d: {}, \
         delta_open: {}, delta_fold: {}, delta_commit: {}, \
         w_ring: {}, next_w_len: {}, level_bytes: {}, label: {:?} }}),",
        step.current_w_len,
        step.d,
        step.log_basis,
        step.challenge_l1_mass,
        step.m_vars,
        step.r_vars,
        step.n_a,
        step.n_b,
        step.n_d,
        step.delta_open,
        step.delta_fold,
        step.delta_commit,
        step.w_ring,
        step.next_w_len,
        step.level_bytes,
        label,
    )
}

fn emit_direct(direct: &DirectStep, prev_fold: Option<&FoldStep>) -> String {
    let shape = format!(
        "GeneratedDirectWitnessShape::PackedDigits {{ num_elems: {}, bits_per_elem: {} }}",
        direct.current_w_len, direct.bits_per_elem,
    );

    let (entry_d, entry_nb, total_bytes) = if direct.bits_per_elem >= 128 {
        (None, None, direct.direct_bytes)
    } else if let Some(prev) = prev_fold {
        let d = prev.d as usize;
        let nb = prev.n_b;
        let total = direct.direct_bytes + ring_vec_bytes(nb, prev.d);
        (Some(d), Some(nb), total)
    } else {
        (None, None, direct.direct_bytes)
    };

    format!(
        "        GeneratedStep::Direct(GeneratedDirectStep {{ \
         current_w_len: {}, witness_shape: {shape}, \
         entry_d: {entry_d:?}, entry_nb: {entry_nb:?}, \
         direct_bytes: {}, total_bytes: {total_bytes} }}),",
        direct.current_w_len, direct.direct_bytes,
    )
}

fn emit_direct_field_elements(current_w_len: usize, direct_bytes: usize) -> String {
    let shape =
        format!("GeneratedDirectWitnessShape::FieldElements {{ num_elems: {current_w_len} }}");
    format!(
        "        GeneratedStep::Direct(GeneratedDirectStep {{ \
         current_w_len: {current_w_len}, witness_shape: {shape}, \
         entry_d: None, entry_nb: None, \
         direct_bytes: {direct_bytes}, total_bytes: {direct_bytes} }}),",
    )
}

fn emit_schedule_entry(out: &mut String, key_str: &str, schedule: &Schedule) -> Result<(), String> {
    writeln!(
        out,
        "    GeneratedScheduleTableEntry {{ key: {key_str}, total_bytes: {}, steps: &[",
        schedule.total_bytes,
    )
    .map_err(|e| e.to_string())?;

    let mut prev_fold: Option<&FoldStep> = None;
    for step in &schedule.steps {
        match step {
            Step::Fold(fold) => {
                writeln!(out, "{}", emit_fold(fold, "refactored")).map_err(|e| e.to_string())?;
                prev_fold = Some(fold);
            }
            Step::Direct(direct) => {
                if direct.bits_per_elem >= 128 {
                    writeln!(
                        out,
                        "{}",
                        emit_direct_field_elements(direct.current_w_len, direct.direct_bytes)
                    )
                    .map_err(|e| e.to_string())?;
                } else {
                    writeln!(out, "{}", emit_direct(direct, prev_fold))
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }

    writeln!(out, "    ] }},").map_err(|e| e.to_string())
}

fn emit_family_rows<Cfg: CommitmentConfig, const D: usize>(
    spec: FamilySpec,
    batch: BatchConfig,
    out: &mut String,
) -> Result<(), String> {
    let nc = batch.num_claims;
    let ng = batch.num_commitment_groups;
    let np = batch.num_points;

    for nv in spec.min_num_vars..=spec.max_num_vars {
        let schedule = match find_optimal_batched_schedule::<Cfg, D>(nv, batch) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  SKIP {}: nv={nv} claims={nc}: {e}", spec.module_name);
                continue;
            }
        };
        let key_str = emit_key(nv, nc, ng, np);
        emit_schedule_entry(out, &key_str, &schedule)?;
    }
    Ok(())
}

fn emit_module(spec: FamilySpec, batch: BatchConfig) -> Result<String, String> {
    let mut out = String::new();
    writeln!(
        out,
        "// Generated by `cargo run --bin gen_refactored_tables -- <output-dir>`"
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::super::{{\n    GeneratedDirectStep, GeneratedDirectWitnessShape, \
         GeneratedFoldStep, GeneratedScheduleKey,\n    \
         GeneratedScheduleTableEntry, GeneratedStep,\n}};"
    )
    .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;
    writeln!(out, "#[rustfmt::skip]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "pub(crate) static {}: &[GeneratedScheduleTableEntry] = &[",
        spec.const_name
    )
    .map_err(|e| e.to_string())?;

    match spec.kind {
        FamilyKind::Fp128D128Full => {
            emit_family_rows::<fp128::D128Full, 128>(spec, batch, &mut out)?
        }
        FamilyKind::Fp128D128OneHot => {
            emit_family_rows::<Fp128D128OneHot, 128>(spec, batch, &mut out)?
        }
        FamilyKind::Fp128D32Full => emit_family_rows::<fp128::D32Full, 32>(spec, batch, &mut out)?,
        FamilyKind::Fp128D32OneHot => {
            emit_family_rows::<fp128::D32OneHot, 32>(spec, batch, &mut out)?
        }
        FamilyKind::Fp128D64Full => emit_family_rows::<fp128::D64Full, 64>(spec, batch, &mut out)?,
        FamilyKind::Fp128D64OneHot => {
            emit_family_rows::<fp128::D64OneHot, 64>(spec, batch, &mut out)?
        }
    }

    writeln!(out, "];").map_err(|e| e.to_string())?;
    Ok(out)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 1 {
        return Err(
            "usage: cargo run --release --bin gen_refactored_tables -- <output-dir>".to_string(),
        );
    }
    let base_dir = PathBuf::from(&args[0]);

    let singleton_dir = base_dir.join("refactored");
    let batched_dir = base_dir.join("refactored_batched");
    fs::create_dir_all(&singleton_dir)
        .map_err(|e| format!("create {}: {e}", singleton_dir.display()))?;
    fs::create_dir_all(&batched_dir)
        .map_err(|e| format!("create {}: {e}", batched_dir.display()))?;

    let singleton = BatchConfig::singleton();
    let batched_4 = BatchConfig {
        num_claims: 4,
        num_commitment_groups: 1,
        num_points: 1,
    };

    for family in ALL_FAMILIES {
        // Singleton tables
        let body = emit_module(*family, singleton)?;
        let dest = singleton_dir.join(format!("{}.rs", family.module_name));
        fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
        println!("wrote {}", dest.display());

        // Batched-4 tables
        let body = emit_module(*family, batched_4)?;
        let dest = batched_dir.join(format!("{}.rs", family.module_name));
        fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
        println!("wrote {}", dest.display());
    }

    Ok(())
}
