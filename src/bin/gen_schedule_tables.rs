//! Generate schedule tables using the DP planner.
//!
//! Produces one module per family in `GeneratedScheduleTableEntry` format.
//! Each emitted file contains both:
//!
//! - singleton schedules (`num_claims=1`)
//! - batched schedules for 4 polynomials, 1 group, 1 point

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use akita_planner::proof_size::ring_vec_bytes;
use akita_planner::schedule_params::find_optimal_schedule;
use akita_types::{DirectStep, FoldStep, HachiScheduleInputs, Schedule, Step, WitnessShape};
use hachi_pcs::protocol::commitment::current_level_layout_with_log_basis;
use hachi_pcs::protocol::config::proof_optimized::fp128;
use hachi_pcs::protocol::CommitmentConfig;

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
    let p = &step.params;
    format!(
        "        GeneratedStep::Fold(GeneratedFoldStep {{ \
         current_w_len: {}, d: {}, log_basis: {}, challenge_l1_mass: {}, \
         m_vars: {}, r_vars: {}, n_a: {}, n_b: {}, n_d: {}, \
         delta_open: {}, delta_fold: {}, delta_commit: {}, \
         w_ring: {}, next_w_len: {}, level_bytes: {}, label: {:?} }}),",
        step.current_w_len,
        p.ring_dimension,
        p.log_basis,
        p.challenge_l1_mass(),
        p.log_block_len(),
        p.log_num_blocks(),
        p.a_key.row_len(),
        p.b_key.row_len(),
        p.d_key.row_len(),
        p.num_digits_open,
        p.num_digits_fold,
        p.num_digits_commit,
        step.w_ring,
        step.next_w_len,
        step.level_bytes,
        label,
    )
}

fn emit_direct<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    level: usize,
    direct: &DirectStep,
) -> String {
    let shape = format!(
        "GeneratedDirectWitnessShape::PackedDigits {{ num_elems: {}, bits_per_elem: {} }}",
        direct.current_w_len, direct.bits_per_elem,
    );

    let (entry_d, entry_nb, total_bytes) = if direct.bits_per_elem >= 128 {
        (None, None, direct.direct_bytes)
    } else {
        let lp = current_level_layout_with_log_basis::<Cfg>(
            HachiScheduleInputs {
                max_num_vars,
                level,
                current_w_len: direct.current_w_len,
            },
            direct.bits_per_elem,
        )
        .expect("level params for direct step");
        let total =
            direct.direct_bytes + ring_vec_bytes(lp.b_key.row_len(), lp.ring_dimension as u32);
        (Some(lp.ring_dimension), Some(lp.b_key.row_len()), total)
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

fn emit_schedule_entry<Cfg: CommitmentConfig>(
    out: &mut String,
    max_num_vars: usize,
    key_str: &str,
    schedule: &Schedule,
) -> Result<(), String> {
    writeln!(
        out,
        "    GeneratedScheduleTableEntry {{ key: {key_str}, total_bytes: {}, steps: &[",
        schedule.total_bytes,
    )
    .map_err(|e| e.to_string())?;

    let mut level = 0usize;
    for step in &schedule.steps {
        match step {
            Step::Fold(fold) => {
                writeln!(out, "{}", emit_fold(fold, "runtime_exact")).map_err(|e| e.to_string())?;
                level += 1;
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
                    writeln!(out, "{}", emit_direct::<Cfg>(max_num_vars, level, direct))
                        .map_err(|e| e.to_string())?;
                }
            }
        }
    }

    writeln!(out, "    ] }},").map_err(|e| e.to_string())
}

fn emit_family_rows<Cfg: CommitmentConfig>(
    spec: FamilySpec,
    batch: WitnessShape,
    out: &mut String,
) -> Result<(), String> {
    let nc = batch.num_claims;
    let ng = batch.num_commitment_groups;
    let np = batch.num_points;

    for nv in spec.min_num_vars..=spec.max_num_vars {
        let schedule = match find_optimal_schedule::<Cfg>(nv, batch) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  SKIP {}: nv={nv} claims={nc}: {e}", spec.module_name);
                continue;
            }
        };
        let key_str = emit_key(nv, nc, ng, np);
        emit_schedule_entry::<Cfg>(out, nv, &key_str, &schedule)?;
    }
    Ok(())
}

fn emit_module(spec: FamilySpec) -> Result<String, String> {
    let mut out = String::new();
    writeln!(
        out,
        "// Generated by `cargo run -p hachi-pcs --bin gen_schedule_tables -- <output-dir>`"
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::{{\n    GeneratedDirectStep, GeneratedDirectWitnessShape, \
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

    let singleton = WitnessShape::singleton();
    let batched_4 = WitnessShape {
        num_claims: 4,
        num_commitment_groups: 1,
        num_points: 1,
    };

    match spec.kind {
        FamilyKind::Fp128D128Full => {
            emit_family_rows::<fp128::D128Full>(spec, singleton, &mut out)?;
            emit_family_rows::<fp128::D128Full>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D128OneHot => {
            emit_family_rows::<fp128::D128OneHot>(spec, singleton, &mut out)?;
            emit_family_rows::<fp128::D128OneHot>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D32Full => {
            emit_family_rows::<fp128::D32Full>(spec, singleton, &mut out)?;
            emit_family_rows::<fp128::D32Full>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D32OneHot => {
            emit_family_rows::<fp128::D32OneHot>(spec, singleton, &mut out)?;
            emit_family_rows::<fp128::D32OneHot>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D64Full => {
            emit_family_rows::<fp128::D64Full>(spec, singleton, &mut out)?;
            emit_family_rows::<fp128::D64Full>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D64OneHot => {
            emit_family_rows::<fp128::D64OneHot>(spec, singleton, &mut out)?;
            emit_family_rows::<fp128::D64OneHot>(spec, batched_4, &mut out)?;
        }
    }

    writeln!(out, "];").map_err(|e| e.to_string())?;
    Ok(out)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 1 {
        return Err(
            "usage: cargo run --release -p hachi-pcs --bin gen_schedule_tables -- <output-dir>"
                .to_string(),
        );
    }
    let base_dir = PathBuf::from(&args[0]);
    fs::create_dir_all(&base_dir).map_err(|e| format!("create {}: {e}", base_dir.display()))?;

    for family in ALL_FAMILIES {
        let body = emit_module(*family)?;
        let dest = base_dir.join(format!("{}.rs", family.module_name));
        fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
        println!("wrote {}", dest.display());
    }

    Ok(())
}
