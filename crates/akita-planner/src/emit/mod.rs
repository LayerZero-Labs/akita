//! Reusable schedule-table emitter for `akita-schedules` and downstream catalogs.
//!
//! The `akita-config` `gen_schedule_tables` binary adapts preset metadata into
//! [`EmitSpec`] values and calls this module. Jolt can invoke the same API with
//! an explicit [`PlannerPolicy`] and hook function pointers.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, DirectStep, FoldStep, LevelParams,
    PolynomialGroupLayout, PrecommittedGroupParams, Schedule, Step,
};

use crate::catalog_identity::expected_catalog_identity;
use crate::generated::{
    GeneratedDirectStep, GeneratedFoldStep, GeneratedScheduleCatalogIdentity,
    GeneratedScheduleTableEntry, GeneratedStep, SisModulusFamily,
};
use crate::PlannerPolicy;

/// One family the emitter writes to `akita-schedules/src/generated/`.
#[derive(Clone)]
pub struct EmitSpec {
    pub module_name: &'static str,
    pub const_name: &'static str,
    pub family_name: &'static str,
    pub schedule_feature: &'static str,
    pub policy: PlannerPolicy,
    pub keys: Vec<PolynomialGroupLayout>,
    pub group_batch_keys: Vec<AkitaScheduleLookupKey>,
    pub emit_group_batch: bool,
    pub output_dir: PathBuf,
    pub regen: fn(PolynomialGroupLayout) -> Result<Schedule, AkitaError>,
    pub regen_group_batch: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
    pub generator_command: &'static str,
}

const MOD_WIRING_BEGIN: &str = "// @generated schedule module wiring begin";
const MOD_WIRING_END: &str = "// @generated schedule module wiring end";

fn fold_step_from_params(p: &LevelParams) -> GeneratedFoldStep {
    GeneratedFoldStep {
        ring_d: p.ring_dimension as u32,
        log_basis: p.log_basis,
        m_vars: p.log_block_len() as u32,
        r_vars: p.log_num_blocks() as u32,
        n_a: p.a_key.row_len() as u32,
        n_b: p.b_key.row_len() as u32,
        n_d: p.d_key.row_len() as u32,
    }
}

fn schedule_to_generated_steps(schedule: &Schedule) -> Vec<GeneratedStep> {
    schedule
        .steps
        .iter()
        .map(|step| match step {
            Step::Fold(fold) => GeneratedStep::Fold(fold_step_from_params(&fold.params)),
            Step::Direct(direct) => GeneratedStep::Direct(GeneratedDirectStep {
                commit: direct.params.as_ref().map(fold_step_from_params),
            }),
        })
        .collect()
}

fn emit_key(key: PolynomialGroupLayout) -> String {
    format!(
        "PolynomialGroupLayout::new({}, {})",
        key.num_vars(),
        key.num_polynomials(),
    )
}

fn emit_precommitted_group_key(layout: &PrecommittedGroupParams) -> String {
    format!(
        "PrecommittedGroupParams {{ group: {}, m_vars: {}, r_vars: {}, log_basis: {}, n_a: {}, conservative_n_b: {} }}",
        emit_key(layout.group),
        layout.m_vars,
        layout.r_vars,
        layout.log_basis,
        layout.n_a,
        layout.conservative_n_b,
    )
}

fn emit_entry_fields(key: &AkitaScheduleLookupKey) -> String {
    let precommitteds = key
        .precommitteds
        .iter()
        .map(emit_precommitted_group_key)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "final_group: {}, precommitteds: &[{}]",
        emit_key(key.final_group),
        precommitteds,
    )
}

fn emit_fold_struct(p: &LevelParams) -> String {
    let fold = fold_step_from_params(p);
    format!(
        "GeneratedFoldStep {{ \
         ring_d: {}, log_basis: {}, m_vars: {}, r_vars: {}, n_a: {}, n_b: {}, n_d: {} }}",
        fold.ring_d, fold.log_basis, fold.m_vars, fold.r_vars, fold.n_a, fold.n_b, fold.n_d,
    )
}

fn emit_fold(step: &FoldStep) -> String {
    format!(
        "        GeneratedStep::Fold({}),",
        emit_fold_struct(&step.params)
    )
}

fn emit_direct(direct: &DirectStep) -> String {
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
        "    GeneratedScheduleTableEntry {{ {key_str}, steps: &[",
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

fn emit_decomposition(d: akita_types::DecompositionParams) -> String {
    match d.log_open_bound {
        Some(v) => format!(
            "DecompositionParams {{ log_basis: {}, log_commit_bound: {}, log_open_bound: Some({}) }}",
            d.log_basis, d.log_commit_bound, v
        ),
        None => format!(
            "DecompositionParams {{ log_basis: {}, log_commit_bound: {}, log_open_bound: None }}",
            d.log_basis, d.log_commit_bound
        ),
    }
}

fn emit_sis_family(family: SisModulusFamily) -> &'static str {
    match family {
        SisModulusFamily::Q32 => "SisModulusFamily::Q32",
        SisModulusFamily::Q64 => "SisModulusFamily::Q64",
        SisModulusFamily::Q128 => "SisModulusFamily::Q128",
    }
}

fn emit_root_fold_shape(shape: TensorChallengeShape) -> &'static str {
    match shape {
        TensorChallengeShape::Flat => "TensorChallengeShape::Flat",
        TensorChallengeShape::Tensor => "TensorChallengeShape::Tensor",
    }
}

fn emit_witness_chunk(cfg: akita_types::ChunkedWitnessCfg) -> String {
    format!(
        "ChunkedWitnessCfg {{ num_chunks: {}, num_activated_levels: {} }}",
        cfg.num_chunks, cfg.num_activated_levels
    )
}

fn emit_identity_const(identity: &GeneratedScheduleCatalogIdentity) -> String {
    let ring_dims: String = identity
        .ring_dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        concat!(
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_RING_DIMENSIONS: &[usize] = &[{ring_dims}];\n",
            "#[rustfmt::skip]\n",
            "pub(crate) static CATALOG_IDENTITY: GeneratedScheduleCatalogIdentity = ",
            "GeneratedScheduleCatalogIdentity {{\n",
            "    family_name: \"{family_name}\",\n",
            "    sis_family: {sis_family},\n",
            "    min_sis_security_bits: {min_sis_security_bits},\n",
            "    ring_dimension: {ring_dimension},\n",
            "    decomposition: {decomposition},\n",
            "    ring_subfield_norm_bound: {ring_subfield_norm_bound},\n",
            "    claim_ext_degree: {claim_ext_degree},\n",
            "    chal_ext_degree: {chal_ext_degree},\n",
            "    basis_range: ({basis_min}, {basis_max}),\n",
            "    onehot_chunk_size: {onehot_chunk_size},\n",
            "    witness_chunk: {witness_chunk},\n",
            "    root_fold_shape: {root_fold_shape},\n",
            "    ring_dimensions: CATALOG_RING_DIMENSIONS,\n",
            "    ring_challenge_config_digest: {ring_challenge_config_digest},\n",
            "    key_count: {key_count},\n",
            "    key_digest: {key_digest},\n",
            "}};\n",
        ),
        ring_dims = ring_dims,
        family_name = identity.family_name,
        sis_family = emit_sis_family(identity.sis_family),
        min_sis_security_bits = identity.min_sis_security_bits,
        ring_dimension = identity.ring_dimension,
        decomposition = emit_decomposition(identity.decomposition),
        ring_subfield_norm_bound = identity.ring_subfield_norm_bound,
        claim_ext_degree = identity.claim_ext_degree,
        chal_ext_degree = identity.chal_ext_degree,
        basis_min = identity.basis_range.0,
        basis_max = identity.basis_range.1,
        onehot_chunk_size = identity.onehot_chunk_size,
        witness_chunk = emit_witness_chunk(identity.witness_chunk),
        root_fold_shape = emit_root_fold_shape(identity.root_fold_shape),
        ring_challenge_config_digest = identity.ring_challenge_config_digest,
        key_count = identity.key_count,
        key_digest = identity.key_digest,
    )
}

fn output_module_name(spec: &EmitSpec) -> String {
    spec.module_name.to_string()
}

fn output_const_name(spec: &EmitSpec) -> String {
    spec.const_name.to_string()
}

fn materialized_entries(
    spec: &EmitSpec,
) -> Result<Vec<(AkitaScheduleLookupKey, Schedule)>, String> {
    let mut entries = Vec::new();
    for key in &spec.keys {
        let schedule =
            (spec.regen)(*key).map_err(|e| format!("{}: regen {key:?}: {e}", spec.module_name))?;
        entries.push((AkitaScheduleLookupKey::single(*key), schedule));
    }
    for key in &spec.group_batch_keys {
        let schedule = (spec.regen_group_batch)(key.clone())
            .map_err(|e| format!("{}: regen multi-group {key:?}: {e}", spec.module_name))?;
        entries.push((key.clone(), schedule));
    }
    entries
        .sort_by(|(left, _), (right, _)| crate::generated::runtime_schedule_key_cmp(left, right));
    Ok(entries)
}

/// Emit one family module (entries + embedded catalog identity).
pub fn emit_family_module(spec: &EmitSpec) -> Result<String, String> {
    let mut out = String::new();
    let const_name = output_const_name(spec);
    writeln!(out, "// Generated by `{}`", spec.generator_command).map_err(|e| e.to_string())?;
    writeln!(out, "#[allow(unused_imports)]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "use super::{{\n    ChunkedWitnessCfg, GeneratedDirectStep, GeneratedFoldStep, \
         GeneratedScheduleCatalogIdentity, PolynomialGroupLayout, PrecommittedGroupParams, \
         GeneratedScheduleTableEntry, GeneratedStep, DecompositionParams, SisModulusFamily, \
         TensorChallengeShape,\n}};"
    )
    .map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    let mut memory_entries: Vec<GeneratedScheduleTableEntry> = Vec::new();
    let mut leaked_steps: Vec<&'static [GeneratedStep]> = Vec::new();

    writeln!(out, "#[rustfmt::skip]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "pub(crate) static {const_name}: &[GeneratedScheduleTableEntry] = &["
    )
    .map_err(|e| e.to_string())?;

    for (key, schedule) in materialized_entries(spec)? {
        let key_str = emit_entry_fields(&key);
        emit_schedule_entry(&mut out, &key_str, &schedule)?;
        let steps = schedule_to_generated_steps(&schedule);
        let steps_ref = Box::leak(steps.into_boxed_slice());
        leaked_steps.push(steps_ref);
        memory_entries.push(GeneratedScheduleTableEntry {
            final_group: key.final_group,
            precommitteds: Box::leak(key.precommitteds.into_boxed_slice()),
            steps: steps_ref,
        });
    }
    debug_assert!(crate::generated::catalog_entries_sorted_for_lookup(
        &memory_entries
    ));

    writeln!(out, "];").map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;

    let identity = expected_catalog_identity(
        spec.family_name,
        &spec.policy,
        &memory_entries,
        spec.ring_challenge_config,
        spec.fold_challenge_shape_at_level,
    )
    .map_err(|e| format!("{}: catalog identity: {e}", spec.module_name))?;
    out.push_str(&emit_identity_const(&identity));

    Ok(out)
}

fn emit_module_declarations(specs: &[EmitSpec]) -> Result<String, String> {
    let mut out = String::new();
    let mut seen = std::collections::BTreeSet::new();
    for spec in specs {
        if !seen.insert(spec.module_name) {
            continue;
        }
        let module_name = spec.module_name;
        let feat = spec.schedule_feature;
        writeln!(out, "#[cfg(feature = \"{feat}\")]").map_err(|e| e.to_string())?;
        writeln!(out, "pub mod {module_name};").map_err(|e| e.to_string())?;
    }
    writeln!(out).map_err(|e| e.to_string())?;
    Ok(out)
}

fn table_fn_name(module_name: &str) -> String {
    format!("{module_name}_table")
}

fn emit_table_accessor(spec: &EmitSpec) -> Result<String, String> {
    let fn_name = table_fn_name(spec.module_name);
    let feat = spec.schedule_feature;
    let module_name = spec.module_name;
    let const_name = spec.const_name;
    Ok(format!(
        "#[cfg(feature = \"{feat}\")]\n\
         pub fn {fn_name}() -> GeneratedScheduleTable {{\n    GeneratedScheduleTable {{\n        entries: {module_name}::{const_name},\n        identity: {module_name}::CATALOG_IDENTITY,\n    }}\n}}\n"
    ))
}

fn emit_mod_wiring(specs: &[EmitSpec]) -> Result<String, String> {
    let mut out = emit_module_declarations(specs)?;
    let mut seen = std::collections::BTreeSet::new();
    for spec in specs {
        if !seen.insert(spec.module_name) {
            continue;
        }
        out.push_str(&emit_table_accessor(spec)?);
        out.push('\n');
    }
    Ok(out)
}

fn replace_between_markers(
    content: &str,
    begin: &str,
    end: &str,
    replacement: &str,
) -> Result<String, String> {
    let start = content
        .find(begin)
        .ok_or_else(|| format!("missing generated marker `{begin}`"))?
        + begin.len();
    let end_pos = content
        .find(end)
        .ok_or_else(|| format!("missing generated marker `{end}`"))?;
    if end_pos < start {
        return Err(format!(
            "generated markers `{begin}` and `{end}` are out of order"
        ));
    }
    let mut out = String::new();
    out.push_str(&content[..start]);
    out.push('\n');
    out.push_str(replacement.trim_end());
    out.push('\n');
    out.push_str(&content[end_pos..]);
    Ok(out)
}

/// Refresh the `@generated schedule module wiring` block in `mod.rs`.
pub fn refresh_generated_wiring(specs: &[EmitSpec], mod_path: &Path) -> Result<(), String> {
    let mod_src =
        fs::read_to_string(mod_path).map_err(|e| format!("read {}: {e}", mod_path.display()))?;
    let mod_wiring = emit_mod_wiring(specs)?;
    let mod_src = replace_between_markers(&mod_src, MOD_WIRING_BEGIN, MOD_WIRING_END, &mod_wiring)?;
    fs::write(mod_path, mod_src).map_err(|e| format!("write {}: {e}", mod_path.display()))?;
    Ok(())
}

/// Run `cargo fmt` on planner, schedules, and config after regen.
pub fn run_regen_fmt() -> Result<(), String> {
    for package in ["akita-planner", "akita-schedules", "akita-config"] {
        let status = Command::new("cargo")
            .args(["fmt", "-p", package])
            .status()
            .map_err(|e| format!("spawn cargo fmt: {e}"))?;
        if !status.success() {
            return Err(format!("cargo fmt -p {package} failed with {status}"));
        }
    }
    Ok(())
}

/// Write one family module to `spec.output_dir` and return its path.
pub fn write_family_module(spec: &EmitSpec) -> Result<PathBuf, String> {
    let body = emit_family_module(spec)?;
    let dest = spec
        .output_dir
        .join(format!("{}.rs", output_module_name(spec)));
    fs::write(&dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
    Ok(dest)
}
