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

use akita_challenges::Stage1ChallengeShape;
use akita_config::current_level_layout_with_log_basis;
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_planner::proof_size::ring_vec_bytes;
use akita_planner::schedule_params::find_optimal_schedule;
use akita_types::ScheduleProvider;
use akita_types::{
    AjtaiRole, AkitaScheduleInputs, CommitmentEnvelope, DirectStep, FoldStep, LevelParams,
    Schedule, Step, WitnessShape,
};

#[derive(Clone, Copy)]
struct FreshPlannerCfg<Base>(PhantomData<Base>);

fn fresh_envelope<Cfg: CommitmentConfig>(max_num_vars: usize) -> CommitmentEnvelope {
    let inner_floor = Cfg::audited_root_rank(AjtaiRole::Inner, max_num_vars);
    let outer_floor = Cfg::audited_root_rank(AjtaiRole::Outer, max_num_vars);
    CommitmentEnvelope {
        max_n_a: inner_floor,
        max_n_b: outer_floor,
        max_n_d: outer_floor,
    }
}

fn stage1_challenge_shape_for_config(
    config: &akita_challenges::SparseChallengeConfig,
) -> Stage1ChallengeShape {
    match config {
        akita_challenges::SparseChallengeConfig::BoundedL1Norm => Stage1ChallengeShape::Flat,
        akita_challenges::SparseChallengeConfig::Uniform { .. }
        | akita_challenges::SparseChallengeConfig::ExactShell { .. } => {
            Stage1ChallengeShape::Tensor
        }
    }
}

fn apply_stage1_challenge_shape(mut params: LevelParams) -> LevelParams {
    params.stage1_challenge_shape = stage1_challenge_shape_for_config(&params.stage1_config);
    params
}

fn fresh_level_params_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> LevelParams {
    let envelope = fresh_envelope::<Cfg>(inputs.max_num_vars);
    let d = Cfg::D;
    let stage1_config = Cfg::stage1_challenge_config(d);
    let production_shape = stage1_challenge_shape_for_config(&stage1_config);

    if inputs.level > 0 {
        // Iterated fixed point over (rank -> layout -> rank): each iteration
        // builds the tentative layout under the *production* stage-1 shape so
        // the SIS extraction collision bucket reflects what the runtime will
        // actually face (including the `4ω` tensor degradation when shape =
        // Tensor). The iteration terminates when the rank derived from the
        // SIS table matches the rank used to lay out the level, or after
        // `MAX_RANK + 1` tries.
        let mut candidate_n_a = envelope.max_n_a.max(1);
        for _ in 0..(akita_types::generated::sis_floor::MAX_RANK + 1) {
            let mut tentative = LevelParams::params_only(
                d,
                log_basis,
                candidate_n_a,
                envelope.max_n_b.max(1),
                envelope.max_n_d.max(1),
                stage1_config.clone(),
            );
            tentative.stage1_challenge_shape = production_shape;
            let Ok(layout) = akita_types::recursive_level_layout_from_params(
                &tentative,
                inputs.current_w_len,
                Cfg::decomposition(),
            ) else {
                break;
            };
            let Some(mut derived) = akita_types::sis_derived_recursive_params_for_layout(
                d,
                log_basis,
                &stage1_config,
                &envelope,
                &layout,
            ) else {
                break;
            };
            if derived.a_key.row_len() <= candidate_n_a {
                // Fixed point reached: the candidate's layout is SIS-secure
                // at `derived.a_key.row_len() <= candidate_n_a`, hence also
                // at `candidate_n_a`. Return the candidate's layout with the
                // candidate rank (possibly over-provisioned vs the strict
                // minimum, but always secure).
                derived.stage1_challenge_shape = production_shape;
                derived.a_key = akita_types::AjtaiKeyParams::new_unchecked(
                    candidate_n_a,
                    derived.a_key.col_len(),
                    derived.a_key.collision_inf(),
                    d,
                );
                if let Ok(final_layout) = akita_types::recursive_level_layout_from_params(
                    &derived,
                    inputs.current_w_len,
                    Cfg::decomposition(),
                ) {
                    return derived.with_layout(&final_layout);
                }
                return derived;
            }
            candidate_n_a = derived.a_key.row_len();
        }
        // Iteration did not converge; fall through to the envelope-default
        // params so the planner can reject this configuration explicitly
        // rather than silently shipping an under-secure schedule.
    }

    apply_stage1_challenge_shape(LevelParams::params_only(
        d,
        log_basis,
        envelope.max_n_a,
        envelope.max_n_b,
        envelope.max_n_d,
        stage1_config,
    ))
}

fn fresh_root_level_params_for_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    lp: &LevelParams,
) -> Result<LevelParams, akita_field::AkitaError> {
    let params = akita_types::sis_derived_root_params_for_layout(
        Cfg::D,
        Cfg::decomposition(),
        Cfg::stage1_challenge_config(Cfg::D),
        inputs,
        lp,
    )?;
    Ok(params.with_layout(lp))
}

fn fresh_root_level_layout_with_log_basis<Cfg: CommitmentConfig>(
    inputs: AkitaScheduleInputs,
    log_basis: u32,
) -> Result<LevelParams, akita_field::AkitaError> {
    let stage1_config = Cfg::stage1_challenge_config(Cfg::D);
    let mut candidate_n_a = 1usize;
    for _ in 0..(akita_types::generated::sis_floor::MAX_RANK + 1) {
        let candidate_params = apply_stage1_challenge_shape(LevelParams::params_only(
            Cfg::D,
            log_basis,
            candidate_n_a,
            1,
            1,
            stage1_config.clone(),
        ));
        let root_lp = akita_types::derived_root_commitment_layout_from_params(
            inputs,
            Cfg::decomposition(),
            &candidate_params,
            false,
        )?;
        let derived_params = akita_types::sis_derived_root_params_for_layout(
            Cfg::D,
            Cfg::decomposition(),
            Cfg::stage1_challenge_config(Cfg::D),
            inputs,
            &root_lp,
        )?;
        if derived_params.a_key.row_len() <= candidate_n_a {
            let mut result = derived_params;
            result.a_key = akita_types::AjtaiKeyParams::try_new(
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

impl<Base: CommitmentConfig> ScheduleProvider for FreshPlannerCfg<Base> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn allow_tensor_stage1_schedules() -> bool {
        Base::allow_tensor_stage1_schedules()
    }

    fn schedule_key(key: akita_types::AkitaScheduleLookupKey) -> String {
        format!("fresh/{}", Base::schedule_key(key))
    }

    fn schedule_plan(
        _key: akita_types::AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }
}

impl<Base: CommitmentConfig> CommitmentConfig for FreshPlannerCfg<Base> {
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> akita_types::DecompositionParams {
        Base::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }

    fn use_setup_claim_reduction() -> bool {
        Base::use_setup_claim_reduction()
    }

    fn audited_root_rank(role: akita_types::AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> akita_types::CommitmentEnvelope {
        fresh_envelope::<Self>(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), akita_field::AkitaError> {
        Base::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn level_params_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        log_basis: u32,
    ) -> akita_types::LevelParams {
        fresh_level_params_with_log_basis::<Self>(inputs, log_basis)
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        fresh_root_level_params_for_layout_with_log_basis::<Self>(inputs, lp)
    }

    fn root_level_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        fresh_root_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: akita_types::AkitaScheduleInputs) -> u32 {
        let _ = inputs;
        Base::decomposition().log_basis
    }

    fn log_basis_search_range(inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
        let _ = inputs;
        (2, 6)
    }
}

impl<Base: CommitmentConfig + akita_planner::PlannerConfig> akita_planner::PlannerConfig
    for FreshPlannerCfg<Base>
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> akita_challenges::SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        _key: akita_types::AkitaScheduleLookupKey,
    ) -> Result<Option<akita_types::AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        <Self as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        current_level_layout_with_log_basis::<Self>(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: akita_types::AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        <Self as CommitmentConfig>::root_level_params_for_layout_with_log_basis(inputs, lp)
    }

    fn planner_log_basis_search_range(inputs: akita_types::AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
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
            AkitaScheduleInputs {
                max_num_vars,
                level,
                current_w_len: direct.current_w_len,
            },
            direct.bits_per_elem,
        )
        .expect("level params for direct step");
        let total = direct.direct_bytes
            + ring_vec_bytes(
                lp.b_key.row_len(),
                lp.ring_dimension as u32,
                Cfg::decomposition().field_bits(),
            );
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
        "// Generated by `cargo run -p akita-config --bin gen_schedule_tables -- <output-dir>`"
    )
    .map_err(|e| e.to_string())?;
    writeln!(out, "#![allow(unused_imports)]").map_err(|e| e.to_string())?;
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
            emit_family_rows::<FreshPlannerCfg<fp128::D128Full>>(spec, singleton, &mut out)?;
            emit_family_rows::<FreshPlannerCfg<fp128::D128Full>>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D128OneHot => {
            emit_family_rows::<FreshPlannerCfg<fp128::D128OneHot>>(spec, singleton, &mut out)?;
            emit_family_rows::<FreshPlannerCfg<fp128::D128OneHot>>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D32Full => {
            emit_family_rows::<FreshPlannerCfg<fp128::D32Full>>(spec, singleton, &mut out)?;
            emit_family_rows::<FreshPlannerCfg<fp128::D32Full>>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D32OneHot => {
            emit_family_rows::<FreshPlannerCfg<fp128::D32OneHot>>(spec, singleton, &mut out)?;
            emit_family_rows::<FreshPlannerCfg<fp128::D32OneHot>>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D64Full => {
            emit_family_rows::<FreshPlannerCfg<fp128::D64Full>>(spec, singleton, &mut out)?;
            emit_family_rows::<FreshPlannerCfg<fp128::D64Full>>(spec, batched_4, &mut out)?;
        }
        FamilyKind::Fp128D64OneHot => {
            emit_family_rows::<FreshPlannerCfg<fp128::D64OneHot>>(spec, singleton, &mut out)?;
            emit_family_rows::<FreshPlannerCfg<fp128::D64OneHot>>(spec, batched_4, &mut out)?;
        }
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

    for family in ALL_FAMILIES {
        if let Some(ref names) = filter {
            if !names.contains(&family.module_name) {
                continue;
            }
        }
        let body = emit_module(*family)?;
        let dest = base_dir.join(format!("{}.rs", family.module_name));
        fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
        println!("wrote {}", dest.display());
    }

    Ok(())
}
