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
use std::marker::PhantomData;
use std::path::PathBuf;

use akita_config::proof_optimized::{fp128, fp16, fp32, fp64};
use akita_config::CommitmentConfig;
use akita_planner::schedule_params::find_optimal_schedule_from_scratch;
use akita_types::{
    AjtaiRole, AkitaScheduleLookupKey, ClaimIncidenceSummary, CommitmentEnvelope, DirectStep,
    FoldStep, LevelParams, Schedule, Step,
};

const FRESH_RANK_CONVERGENCE_LIMIT: usize = 1024;

#[derive(Clone)]
struct FreshPlanner<Cfg>(PhantomData<Cfg>);

fn fresh_envelope<Cfg: CommitmentConfig>(num_vars: usize) -> CommitmentEnvelope {
    let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, num_vars);
    let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, num_vars);
    CommitmentEnvelope {
        max_n_a: inner_floor,
        max_n_b: outer_floor,
        max_n_d: outer_floor,
    }
}

fn fresh_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: akita_types::AkitaScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let envelope = fresh_envelope::<Cfg>(inputs.num_vars);
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let fold_shape = Cfg::fold_challenge_shape_at_level(inputs);

    if inputs.level > 0 {
        let mut candidate_n_a = envelope.max_n_a.max(1);
        for _ in 0..FRESH_RANK_CONVERGENCE_LIMIT {
            let tentative = LevelParams::params_only(
                Cfg::sis_modulus_family(),
                d,
                log_basis,
                candidate_n_a,
                envelope.max_n_b.max(1),
                envelope.max_n_d.max(1),
                stage1_config.clone(),
            )
            .with_fold_challenge_shape(fold_shape);
            let Ok(layout) = akita_types::recursive_level_layout_from_params(
                &tentative,
                inputs.current_w_len,
                Cfg::decomposition(),
            ) else {
                break;
            };
            let Some(derived) = akita_types::sis_derived_recursive_params_for_layout(
                Cfg::sis_modulus_family(),
                d,
                log_basis,
                &stage1_config,
                Cfg::ring_subfield_embedding_norm_bound(),
                &envelope,
                &layout,
            ) else {
                break;
            };
            if derived.a_key.row_len() <= candidate_n_a {
                let mut params = derived;
                if let Ok(a_key) = akita_types::AjtaiKeyParams::try_new(
                    params.a_key.sis_family(),
                    candidate_n_a,
                    params.a_key.col_len(),
                    params.a_key.collision_inf(),
                    d,
                ) {
                    params.a_key = a_key;
                    return params.with_layout(&layout);
                }
                break;
            }
            candidate_n_a = derived.a_key.row_len();
        }
    }

    LevelParams::params_only(
        Cfg::sis_modulus_family(),
        d,
        log_basis,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
        stage1_config,
    )
    .with_fold_challenge_shape(fold_shape)
}

fn fresh_root_level_params_for_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: akita_types::AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, akita_field::AkitaError> {
    let params = akita_types::sis_derived_root_params_for_layout(
        Cfg::sis_modulus_family(),
        Cfg::D,
        Cfg::decomposition(),
        Cfg::stage1_challenge_config(Cfg::D),
        Cfg::ring_subfield_embedding_norm_bound(),
        inputs,
        lp,
    )?;
    Ok(params.with_layout(lp))
}

fn fresh_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: akita_types::AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, akita_field::AkitaError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let fold_shape = Cfg::fold_challenge_shape_at_level(inputs);
    let mut candidate_n_a = 1usize;
    for _ in 0..FRESH_RANK_CONVERGENCE_LIMIT {
        let candidate_params = LevelParams::params_only(
            Cfg::sis_modulus_family(),
            Cfg::D,
            log_basis,
            candidate_n_a,
            1,
            1,
            stage1_config.clone(),
        )
        .with_fold_challenge_shape(fold_shape);
        let root_lp = akita_types::derived_root_commitment_layout_from_params(
            inputs,
            Cfg::decomposition(),
            &candidate_params,
            false,
        )?;
        let derived_params =
            fresh_root_level_params_for_layout_with_log_basis::<Cfg>(inputs, &root_lp)?;
        if derived_params.a_key.row_len() <= candidate_n_a {
            let mut result = derived_params;
            result.a_key = akita_types::AjtaiKeyParams::try_new(
                result.a_key.sis_family(),
                candidate_n_a,
                result.a_key.col_len(),
                result.a_key.collision_inf(),
                Cfg::D,
            )?;
            return Ok(result.with_layout(&root_lp));
        }
        candidate_n_a = derived_params.a_key.row_len();
    }
    Err(akita_field::AkitaError::InvalidSetup(format!(
        "failed to converge on self-consistent root A-row rank for D={} lb={log_basis}",
        Cfg::D
    )))
}

impl<Cfg> akita_planner::PlannerConfig for FreshPlanner<Cfg>
where
    Cfg: CommitmentConfig + akita_planner::PlannerConfig,
{
    type PlannerField = Cfg::PlannerField;

    const PLANNER_D: usize = Cfg::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Cfg::planner_field_bits()
    }

    fn planner_challenge_field_bits() -> u32 {
        Cfg::planner_challenge_field_bits()
    }

    fn planner_extension_opening_width() -> usize {
        Cfg::planner_extension_opening_width()
    }

    fn planner_recursive_witness_expansion() -> usize {
        Cfg::planner_recursive_witness_expansion()
    }

    fn planner_recursive_public_rows() -> usize {
        Cfg::planner_recursive_public_rows()
    }

    fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
        Cfg::planner_sis_modulus_family()
    }

    fn planner_stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
        Cfg::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        fresh_root_level_layout_with_log_basis::<Cfg>(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        if inputs.level == 0 {
            fresh_root_level_layout_with_log_basis::<Cfg>(inputs, log_basis)
        } else {
            let params = fresh_level_params_with_log_basis::<Cfg>(inputs, log_basis);
            let layout = akita_types::recursive_level_layout_from_params(
                &params,
                inputs.current_w_len,
                Cfg::decomposition(),
            )?;
            Ok(params.with_layout(&layout))
        }
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        fresh_root_level_params_for_layout_with_log_basis::<Cfg>(inputs, lp)
    }

    fn planner_log_basis_search_range(inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
        Cfg::planner_log_basis_search_range(inputs)
    }

    fn planner_fold_prover_weight() -> usize {
        Cfg::planner_fold_prover_weight()
    }
}

#[derive(Clone, Copy)]
enum FamilyKind {
    Fp128D32Full,
    Fp128D32OneHot,
    Fp128D64Full,
    Fp128D64OneHot,
    Fp128D64OneHotTensor,
    Fp32D32Full,
    Fp32D32OneHot,
    Fp32D64Full,
    Fp32D64OneHot,
    Fp16D32Full,
    Fp16D32OneHot,
    Fp16D64Full,
    Fp16D64OneHot,
    Fp64D32Full,
    Fp64D32OneHot,
    Fp64D64Full,
    Fp64D64OneHot,
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
    FamilySpec {
        module_name: "fp128_d64_onehot_tensor",
        const_name: "FP128_D64_ONEHOT_TENSOR_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 50,
        kind: FamilyKind::Fp128D64OneHotTensor,
    },
    FamilySpec {
        module_name: "fp32_d32",
        const_name: "FP32_D32_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp32D32Full,
    },
    FamilySpec {
        module_name: "fp32_d32_onehot",
        const_name: "FP32_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp32D32OneHot,
    },
    FamilySpec {
        module_name: "fp32_d64",
        const_name: "FP32_D64_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp32D64Full,
    },
    FamilySpec {
        module_name: "fp32_d64_onehot",
        const_name: "FP32_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp32D64OneHot,
    },
    FamilySpec {
        module_name: "fp16_d32_full",
        const_name: "FP16_D32_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp16D32Full,
    },
    FamilySpec {
        module_name: "fp16_d32_onehot",
        const_name: "FP16_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp16D32OneHot,
    },
    FamilySpec {
        module_name: "fp16_d64_full",
        const_name: "FP16_D64_FULL_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp16D64Full,
    },
    FamilySpec {
        module_name: "fp16_d64_onehot",
        const_name: "FP16_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp16D64OneHot,
    },
    FamilySpec {
        module_name: "fp64_d32",
        const_name: "FP64_D32_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp64D32Full,
    },
    FamilySpec {
        module_name: "fp64_d32_onehot",
        const_name: "FP64_D32_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp64D32OneHot,
    },
    FamilySpec {
        module_name: "fp64_d64",
        const_name: "FP64_D64_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp64D64Full,
    },
    FamilySpec {
        module_name: "fp64_d64_onehot",
        const_name: "FP64_D64_ONEHOT_SCHEDULES",
        min_num_vars: 1,
        max_num_vars: 32,
        kind: FamilyKind::Fp64D64OneHot,
    },
];

fn emit_key(key: AkitaScheduleLookupKey) -> String {
    format!(
        "GeneratedScheduleKey {{ num_vars: {}, num_commitment_groups: {}, num_t_vectors: {}, \
         num_w_vectors: {}, num_z_vectors: {} }}",
        key.num_vars, key.num_points, key.num_t_vectors, key.num_w_vectors, key.num_z_vectors,
    )
}

fn emit_fold(step: &FoldStep) -> String {
    let p = &step.params;
    format!(
        "        GeneratedStep::Fold(GeneratedFoldStep {{ \
         ring_d: {}, log_basis: {}, m_vars: {}, r_vars: {}, n_a: {}, n_b: {}, n_d: {} }}),",
        p.ring_dimension,
        p.log_basis,
        p.log_block_len(),
        p.log_num_blocks(),
        p.a_key.row_len(),
        p.b_key.row_len(),
        p.d_key.row_len(),
    )
}

fn emit_direct(_direct: &DirectStep) -> String {
    "        GeneratedStep::Direct(GeneratedDirectStep),".to_string()
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

fn output_module_name(spec: FamilySpec) -> String {
    #[cfg(feature = "zk")]
    {
        format!("{}_zk", spec.module_name)
    }
    #[cfg(not(feature = "zk"))]
    {
        spec.module_name.to_string()
    }
}

fn output_const_name(spec: FamilySpec) -> String {
    #[cfg(feature = "zk")]
    {
        let base = spec
            .const_name
            .strip_suffix("_SCHEDULES")
            .expect("generated schedule const name should end in _SCHEDULES");
        format!("{base}_ZK_SCHEDULES")
    }
    #[cfg(not(feature = "zk"))]
    {
        spec.const_name.to_string()
    }
}

fn generator_command() -> &'static str {
    #[cfg(feature = "zk")]
    {
        "cargo run -p akita-config --features planner,zk --bin gen_schedule_tables -- <output-dir>"
    }
    #[cfg(not(feature = "zk"))]
    {
        "cargo run -p akita-config --features planner --bin gen_schedule_tables -- <output-dir>"
    }
}

fn emit_family_rows<Cfg>(
    spec: FamilySpec,
    incidence_for_nv: impl Fn(usize) -> ClaimIncidenceSummary,
    label_counts: (usize, usize, usize),
    out: &mut String,
) -> Result<(), String>
where
    Cfg: CommitmentConfig + akita_planner::PlannerConfig,
{
    let (num_t_vectors, num_w_vectors, num_z_vectors) = label_counts;

    for nv in spec.min_num_vars..=spec.max_num_vars {
        let incidence = incidence_for_nv(nv);
        let key = AkitaScheduleLookupKey::new_from_incidence(&incidence)
            .map_err(|e| format!("build schedule key: {e}"))?;
        let schedule = match find_optimal_schedule_from_scratch::<FreshPlanner<Cfg>>(key) {
            Ok(s) => s,
            Err(e) => {
                return Err(format!(
                    "{}: nv={nv} t={num_t_vectors} w={num_w_vectors} z={num_z_vectors}: {e}",
                    spec.module_name
                ));
            }
        };
        let key_str = emit_key(key);
        emit_schedule_entry(out, &key_str, &schedule)?;
    }
    Ok(())
}

fn emit_module(spec: FamilySpec) -> Result<String, String> {
    let mut out = String::new();
    let const_name = output_const_name(spec);
    writeln!(out, "// Generated by `{}`", generator_command(),).map_err(|e| e.to_string())?;
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

    let singleton = |nv| ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence");
    let batched_4 = |nv| ClaimIncidenceSummary::same_point(nv, 4).expect("batched incidence");

    match spec.kind {
        FamilyKind::Fp128D32Full => {
            emit_family_rows::<fp128::D32Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp128::D32Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp128D32OneHot => {
            emit_family_rows::<fp128::D32OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp128::D32OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp128D64Full => {
            emit_family_rows::<fp128::D64Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp128::D64Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp128D64OneHot => {
            emit_family_rows::<fp128::D64OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp128::D64OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp128D64OneHotTensor => {
            emit_family_rows::<fp128::D64OneHotTensor>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp128::D64OneHotTensor>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp32D32Full => {
            emit_family_rows::<fp32::D32Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp32::D32Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp32D32OneHot => {
            emit_family_rows::<fp32::D32OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp32::D32OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp32D64Full => {
            emit_family_rows::<fp32::D64Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp32::D64Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp32D64OneHot => {
            emit_family_rows::<fp32::D64OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp32::D64OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp16D32Full => {
            emit_family_rows::<fp16::D32Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp16::D32Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp16D32OneHot => {
            emit_family_rows::<fp16::D32OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp16::D32OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp16D64Full => {
            emit_family_rows::<fp16::D64Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp16::D64Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp16D64OneHot => {
            emit_family_rows::<fp16::D64OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp16::D64OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp64D32Full => {
            emit_family_rows::<fp64::D32Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp64::D32Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp64D32OneHot => {
            emit_family_rows::<fp64::D32OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp64::D32OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp64D64Full => {
            emit_family_rows::<fp64::D64Full>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp64::D64Full>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
        FamilyKind::Fp64D64OneHot => {
            emit_family_rows::<fp64::D64OneHot>(spec, singleton, (1, 1, 1), &mut out)?;
            emit_family_rows::<fp64::D64OneHot>(spec, batched_4, (4, 4, 1), &mut out)?;
        }
    }

    writeln!(out, "];").map_err(|e| e.to_string())?;
    Ok(out)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        return Err(
            "usage: cargo run --release -p akita-config --features planner --bin gen_schedule_tables -- \
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

    for family in ALL_FAMILIES {
        if let Some(ref names) = filter {
            if !names.contains(&family.module_name) {
                continue;
            }
        }
        let body = emit_module(*family)?;
        let dest = base_dir.join(format!("{}.rs", output_module_name(*family)));
        fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
        println!("wrote {}", dest.display());
    }

    Ok(())
}
