//! Generate exact keyed schedule tables and SIS floors.

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use hachi_pcs::planner::proof_size::ring_vec_bytes;
use hachi_pcs::planner::sis_security;
use hachi_pcs::protocol::commitment::presets::fp128;
use hachi_pcs::protocol::commitment::{
    exact_schedule_plan_for_lookup_key, CommitmentConfig, CommitmentPreset,
    GeneratedAdaptivePolicy, HachiPlannedDirectStep, HachiPlannedLevel, HachiPlannedStep,
    HachiRootBatchSummary, HachiScheduleLookupKey, HachiSchedulePlan,
};
use hachi_pcs::protocol::proof::DirectWitnessShape;

type Fp128D128OneHot =
    CommitmentPreset<fp128::Field, GeneratedAdaptivePolicy<fp128::Profile, 128, 1>>;

#[derive(Clone, Copy)]
struct BlessedBatchSpec {
    max_num_vars: usize,
    num_vars: usize,
    layout_num_claims: usize,
    batch_num_claims: usize,
    batch_num_commitment_groups: usize,
    batch_num_points: usize,
}

impl BlessedBatchSpec {
    const fn lookup_key(self) -> HachiScheduleLookupKey {
        HachiScheduleLookupKey::with_batch(
            self.max_num_vars,
            self.num_vars,
            self.layout_num_claims,
            HachiRootBatchSummary {
                num_claims: self.batch_num_claims,
                num_commitment_groups: self.batch_num_commitment_groups,
                num_points: self.batch_num_points,
            },
        )
    }
}

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
    family_name: &'static str,
    module_name: &'static str,
    const_name: &'static str,
    min_num_vars: usize,
    max_num_vars: usize,
    kind: FamilyKind,
    blessed_batches: &'static [BlessedBatchSpec],
}

const D64_ONEHOT_BLESSED_BATCHES: &[BlessedBatchSpec] = &[
    BlessedBatchSpec {
        max_num_vars: 20,
        num_vars: 20,
        layout_num_claims: 6,
        batch_num_claims: 6,
        batch_num_commitment_groups: 3,
        batch_num_points: 1,
    },
    BlessedBatchSpec {
        max_num_vars: 20,
        num_vars: 20,
        layout_num_claims: 6,
        batch_num_claims: 6,
        batch_num_commitment_groups: 3,
        batch_num_points: 2,
    },
];

const ALL_FAMILIES: &[FamilySpec] = &[
    FamilySpec {
        family_name: "fp128_d128_full",
        module_name: "fp128_d128_full",
        const_name: "FP128_D128_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D128Full,
        blessed_batches: &[],
    },
    FamilySpec {
        family_name: "fp128_d128_onehot",
        module_name: "fp128_d128_onehot",
        const_name: "FP128_D128_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D128OneHot,
        blessed_batches: &[],
    },
    FamilySpec {
        family_name: "fp128_d32_full",
        module_name: "fp128_d32_full",
        const_name: "FP128_D32_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D32Full,
        blessed_batches: &[],
    },
    FamilySpec {
        family_name: "fp128_d32_onehot",
        module_name: "fp128_d32_onehot",
        const_name: "FP128_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D32OneHot,
        blessed_batches: &[],
    },
    FamilySpec {
        family_name: "fp128_d64_full",
        module_name: "fp128_d64_full",
        const_name: "FP128_D64_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D64Full,
        blessed_batches: &[],
    },
    FamilySpec {
        family_name: "fp128_d64_onehot",
        module_name: "fp128_d64_onehot",
        const_name: "FP128_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D64OneHot,
        blessed_batches: D64_ONEHOT_BLESSED_BATCHES,
    },
];

fn usage() -> &'static str {
    "usage: cargo run --bin gen_schedule_tables -- <output-dir> [--family <name>]..."
}

fn family_by_name(name: &str) -> Option<FamilySpec> {
    ALL_FAMILIES
        .iter()
        .copied()
        .find(|family| family.family_name == name)
}

fn emit_generated_key(key: HachiScheduleLookupKey) -> String {
    let batch = key.batch;
    format!(
        "GeneratedScheduleKey {{ max_num_vars: {}, num_vars: {}, layout_num_claims: {}, batch_num_claims: {}, batch_num_commitment_groups: {}, batch_num_points: {} }}",
        key.max_num_vars,
        key.num_vars,
        key.layout_num_claims,
        batch.num_claims,
        batch.num_commitment_groups,
        batch.num_points,
    )
}

fn emit_shape(direct: &HachiPlannedDirectStep) -> String {
    match direct.witness_shape {
        DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)) => format!(
            "GeneratedDirectWitnessShape::PackedDigits {{ num_elems: {num_elems}, bits_per_elem: {bits_per_elem} }}"
        ),
        DirectWitnessShape::FieldElements(num_elems) => format!(
            "GeneratedDirectWitnessShape::FieldElements {{ num_elems: {num_elems} }}"
        ),
    }
}

fn emit_direct_entry<Cfg: CommitmentConfig>(
    previous_fold: Option<&HachiPlannedLevel>,
    direct: &HachiPlannedDirectStep,
) -> Result<(Option<usize>, Option<usize>, usize), String> {
    let Some(previous_fold) = previous_fold else {
        return Ok((None, None, direct.direct_bytes));
    };

    let entry_d = Cfg::d_at_level(direct.state.level, direct.state.current_w_len);
    let next_commit_coeffs = previous_fold.next_commit_coeffs;
    if next_commit_coeffs == 0 {
        return Ok((None, None, direct.direct_bytes));
    }
    if next_commit_coeffs % entry_d != 0 {
        return Err(format!(
            "non-integral terminal entry commitment for level {}: coeffs={} d={entry_d}",
            direct.state.level, next_commit_coeffs
        ));
    }

    let entry_nb = next_commit_coeffs / entry_d;
    let total_bytes = direct.direct_bytes + ring_vec_bytes(entry_nb, entry_d as u32);
    Ok((Some(entry_d), Some(entry_nb), total_bytes))
}

fn emit_exact_entry<Cfg: CommitmentConfig>(
    out: &mut String,
    key: HachiScheduleLookupKey,
    plan: &HachiSchedulePlan,
) -> Result<(), String> {
    writeln!(
        out,
        "    GeneratedScheduleTableEntry {{ key: {}, total_bytes: {}, steps: &[",
        emit_generated_key(key),
        plan.exact_proof_bytes
    )
    .map_err(|err| err.to_string())?;

    let mut previous_fold: Option<&HachiPlannedLevel> = None;
    for step in &plan.steps {
        match step {
            HachiPlannedStep::Fold(level) => {
                writeln!(
                    out,
                    "        GeneratedStep::Fold(GeneratedFoldStep {{ current_w_len: {}, d: {}, log_basis: {}, challenge_l1_mass: {}, m_vars: {}, r_vars: {}, n_a: {}, n_b: {}, n_d: {}, delta_open: {}, delta_fold: {}, delta_commit: {}, w_ring: {}, next_w_len: {}, level_bytes: {}, label: {:?} }}),",
                    level.inputs.current_w_len,
                    level.params.d,
                    level.params.log_basis,
                    level.params.challenge_l1_mass,
                    level.layout.m_vars,
                    level.layout.r_vars,
                    level.params.n_a,
                    level.params.n_b,
                    level.params.n_d,
                    level.layout.num_digits_open,
                    level.layout.num_digits_fold,
                    level.layout.num_digits_commit,
                    level.next_inputs.current_w_len / level.params.d,
                    level.next_inputs.current_w_len,
                    level.level_bytes,
                    "runtime_exact",
                )
                .map_err(|err| err.to_string())?;
                previous_fold = Some(level.as_ref());
            }
            HachiPlannedStep::Direct(direct) => {
                let (entry_d, entry_nb, total_bytes) =
                    emit_direct_entry::<Cfg>(previous_fold, direct)?;
                writeln!(
                    out,
                    "        GeneratedStep::Direct(GeneratedDirectStep {{ current_w_len: {}, witness_shape: {}, entry_d: {:?}, entry_nb: {:?}, direct_bytes: {}, total_bytes: {} }}),",
                    direct.state.current_w_len,
                    emit_shape(direct),
                    entry_d,
                    entry_nb,
                    direct.direct_bytes,
                    total_bytes,
                )
                .map_err(|err| err.to_string())?;
            }
        }
    }

    writeln!(out, "    ] }},").map_err(|err| err.to_string())
}

fn emit_family_rows<Cfg: CommitmentConfig, const D: usize>(
    spec: FamilySpec,
    out: &mut String,
) -> Result<(), String> {
    for num_vars in spec.min_num_vars..=spec.max_num_vars {
        let key = HachiScheduleLookupKey::singleton(num_vars, num_vars, 1);
        let plan = exact_schedule_plan_for_lookup_key::<Cfg, D>(key)
            .map_err(|err| format!("failed to plan singleton {key:?}: {err}"))?;
        emit_exact_entry::<Cfg>(out, key, &plan)?;
    }

    for blessed in spec.blessed_batches {
        let key = blessed.lookup_key();
        let plan = exact_schedule_plan_for_lookup_key::<Cfg, D>(key)
            .map_err(|err| format!("failed to plan blessed batch {key:?}: {err}"))?;
        emit_exact_entry::<Cfg>(out, key, &plan)?;
    }

    Ok(())
}

fn emit_family_module(spec: FamilySpec) -> Result<String, String> {
    let mut out = String::new();
    writeln!(
        out,
        "// Generated by `cargo run --bin gen_schedule_tables -- <output-dir>`"
    )
    .map_err(|err| err.to_string())?;
    writeln!(
        out,
        "use super::{{GeneratedDirectStep, GeneratedDirectWitnessShape, GeneratedFoldStep, GeneratedScheduleKey, GeneratedScheduleTableEntry, GeneratedStep}};"
    )
    .map_err(|err| err.to_string())?;
    writeln!(out).map_err(|err| err.to_string())?;
    writeln!(out, "#[rustfmt::skip]").map_err(|err| err.to_string())?;
    writeln!(
        out,
        "pub(crate) static {}: &[GeneratedScheduleTableEntry] = &[",
        spec.const_name
    )
    .map_err(|err| err.to_string())?;

    match spec.kind {
        FamilyKind::Fp128D128Full => emit_family_rows::<fp128::D128Full, 128>(spec, &mut out)?,
        FamilyKind::Fp128D128OneHot => emit_family_rows::<Fp128D128OneHot, 128>(spec, &mut out)?,
        FamilyKind::Fp128D32Full => emit_family_rows::<fp128::D32Full, 32>(spec, &mut out)?,
        FamilyKind::Fp128D32OneHot => emit_family_rows::<fp128::D32OneHot, 32>(spec, &mut out)?,
        FamilyKind::Fp128D64Full => emit_family_rows::<fp128::D64Full, 64>(spec, &mut out)?,
        FamilyKind::Fp128D64OneHot => emit_family_rows::<fp128::D64OneHot, 64>(spec, &mut out)?,
    }

    writeln!(out, "];").map_err(|err| err.to_string())?;
    Ok(out)
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(output_dir) = args.next() else {
        return Err(usage().to_string());
    };
    let mut requested_families = BTreeSet::new();
    while let Some(arg) = args.next() {
        if arg == "--family" {
            let Some(name) = args.next() else {
                return Err(usage().to_string());
            };
            if family_by_name(&name).is_none() {
                return Err(format!(
                    "unknown family {name:?}; expected one of: {}",
                    ALL_FAMILIES
                        .iter()
                        .map(|family| family.family_name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            requested_families.insert(name);
            continue;
        }
        return Err(usage().to_string());
    }

    let output_dir = PathBuf::from(output_dir);
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;

    let families: Vec<FamilySpec> = if requested_families.is_empty() {
        ALL_FAMILIES.to_vec()
    } else {
        requested_families
            .into_iter()
            .map(|name| family_by_name(&name).expect("requested family was prevalidated"))
            .collect()
    };

    for family in families {
        let dest = output_dir.join(format!("{}.rs", family.module_name));
        let body = emit_family_module(family)?;
        fs::write(&dest, body)
            .map_err(|err| format!("failed to write {}: {err}", dest.display()))?;
        println!("wrote {}", dest.display());
    }

    let sis_dest = output_dir.join("sis_floor.rs");
    let sis_body = emit_sis_floor_module();
    fs::write(&sis_dest, &sis_body)
        .map_err(|err| format!("failed to write {}: {err}", sis_dest.display()))?;
    println!("wrote {}", sis_dest.display());

    Ok(())
}

fn emit_sis_floor_module() -> String {
    use std::fmt::Write;

    let mut out = String::new();
    let _ = writeln!(out, "// Generated by `cargo run --bin gen_schedule_tables`");
    let _ = writeln!(out, "//");
    let _ = writeln!(
        out,
        "// SIS width thresholds for 128-bit security (BDGL16 + lgsa, q = 2^128 - 275)."
    );
    let _ = writeln!(
        out,
        "// Source: `src/planner/sis_security.rs`, verified with lattice-estimator."
    );
    let _ = writeln!(out);

    let max_rank = sis_security::MAX_RANK;
    let _ = writeln!(out, "const MAX_RANK: usize = {max_rank};");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "fn sis_max_widths(d: u32, collision_inf: u32) -> Option<[u64; MAX_RANK]> {{"
    );
    let _ = writeln!(out, "    match (d, collision_inf) {{");

    let d_collision_pairs: &[(u32, &[u32])] = &[
        (32, &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]),
        (64, &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047]),
        (128, &[2, 3, 7, 15, 31, 63]),
    ];

    for &(d, collisions) in d_collision_pairs {
        let _ = writeln!(out, "        // D={d}");
        for &c in collisions {
            if let Some(widths) = sis_security::sis_max_widths_public(d, c) {
                let _ = write!(out, "        ({d}, {c}) => Some([");
                for (i, w) in widths.iter().enumerate() {
                    if i > 0 {
                        let _ = write!(out, ", ");
                    }
                    let _ = write!(out, "{w}");
                }
                let _ = writeln!(out, "]),");
            }
        }
    }

    let _ = writeln!(out, "        _ => None,");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "pub(crate) fn min_rank_for_secure_width(d: u32, collision_inf: u32, width: u64) -> Option<usize> {{"
    );
    let _ = writeln!(out, "    let widths = sis_max_widths(d, collision_inf)?;");
    let _ = writeln!(out, "    for (i, &max_w) in widths.iter().enumerate() {{");
    let _ = writeln!(out, "        if width <= max_w {{");
    let _ = writeln!(out, "            return Some(i + 1);");
    let _ = writeln!(out, "        }}");
    let _ = writeln!(out, "    }}");
    let _ = writeln!(out, "    None");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "pub(crate) fn ceil_supported_collision(d: u32, collision_inf: u32) -> Option<u32> {{"
    );
    let _ = writeln!(
        out,
        "    const D32: &[u32] = &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047];"
    );
    let _ = writeln!(
        out,
        "    const D64: &[u32] = &[2, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047];"
    );
    let _ = writeln!(out, "    const D128: &[u32] = &[2, 3, 7, 15, 31, 63];");
    let _ = writeln!(out, "    let buckets = match d {{");
    let _ = writeln!(out, "        32 => D32,");
    let _ = writeln!(out, "        64 => D64,");
    let _ = writeln!(out, "        128 => D128,");
    let _ = writeln!(out, "        _ => return None,");
    let _ = writeln!(out, "    }};");
    let _ = writeln!(out, "    buckets");
    let _ = writeln!(out, "        .iter()");
    let _ = writeln!(out, "        .copied()");
    let _ = writeln!(out, "        .find(|&bucket| collision_inf <= bucket)");
    let _ = writeln!(out, "}}");

    out
}
