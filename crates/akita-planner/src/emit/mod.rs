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
    AkitaScheduleInputs, AkitaScheduleLookupKey, LevelParams, PolynomialGroupLayout,
    PrecommittedGroupParams, Schedule, SetupContributionMode,
};

use crate::catalog_identity::expected_catalog_identity;
use crate::generated::{
    GeneratedFold, GeneratedFoldStep, GeneratedFoldStepWithSetupMetadata,
    GeneratedScheduleCatalogIdentity, GeneratedScheduleTableEntry, GeneratedSetupPrefixGroup,
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
        position_index_bits: p.position_index_bits() as u32,
        block_index_bits: p.block_index_bits() as u32,
        num_live_blocks: p.num_live_blocks as u32,
        n_a: p.a_key.row_len() as u32,
        n_b: p.b_key.row_len() as u32,
        n_d: p.d_key.row_len() as u32,
    }
}

fn setup_prefix_group_from_params(
    p: &LevelParams,
    include_setup_prefix_group: bool,
) -> Option<GeneratedSetupPrefixGroup> {
    include_setup_prefix_group
        .then_some(p.setup_prefix.as_ref())
        .flatten()
        .map(|setup_prefix| {
            let group = &setup_prefix.commitment_params;
            GeneratedSetupPrefixGroup {
                natural_len: setup_prefix.natural_len as u32,
                num_live_ring_elements_per_claim: group.layout.num_live_ring_elements_per_claim
                    as u32,
                num_positions_per_block: group.layout.num_positions_per_block as u32,
                num_live_blocks: group.layout.num_live_blocks as u32,
                fold_challenge_shape: group.layout.fold_challenge_shape,
                n_a: group.a_key.row_len() as u32,
                n_b: group.b_key.row_len() as u32,
            }
        })
}

fn schedule_to_generated_folds(
    _key: &AkitaScheduleLookupKey,
    schedule: &Schedule,
) -> Vec<GeneratedFold> {
    schedule
        .folds
        .iter()
        .enumerate()
        .map(|(idx, fold)| {
            let include_setup_prefix_group = idx > 0 && fold.params.setup_prefix.is_some();
            let setup_prefix_group =
                setup_prefix_group_from_params(&fold.params, include_setup_prefix_group);
            let fold_step = fold_step_from_params(&fold.params);
            if setup_prefix_group.is_some()
                || fold.params.setup_contribution_mode != SetupContributionMode::Direct
            {
                GeneratedFold::FoldWithSetupMetadata(GeneratedFoldStepWithSetupMetadata {
                    fold: fold_step,
                    setup_prefix_group,
                    setup_contribution_mode: fold.params.setup_contribution_mode,
                })
            } else {
                GeneratedFold::Fold(fold_step)
            }
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
    let challenge_shape = emit_root_fold_shape(layout.fold_challenge_shape);
    format!(
        "PrecommittedGroupParams {{ group: {}, num_live_ring_elements_per_claim: {}, num_positions_per_block: {}, num_live_blocks: {}, fold_challenge_shape: {}, log_basis: {}, n_a: {}, conservative_n_b: {} }}",
        emit_key(layout.group),
        layout.num_live_ring_elements_per_claim,
        layout.num_positions_per_block,
        layout.num_live_blocks,
        challenge_shape,
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

fn emit_compact_fold_struct(p: &LevelParams) -> String {
    let fold = fold_step_from_params(p);
    format!(
        "GeneratedFoldStep {{ \
         ring_d: {}, log_basis: {}, position_index_bits: {}, block_index_bits: {}, num_live_blocks: {}, n_a: {}, n_b: {}, n_d: {} }}",
        fold.ring_d,
        fold.log_basis,
        fold.position_index_bits,
        fold.block_index_bits,
        fold.num_live_blocks,
        fold.n_a,
        fold.n_b,
        fold.n_d,
    )
}

fn emit_setup_contribution_mode(mode: SetupContributionMode) -> &'static str {
    match mode {
        SetupContributionMode::Direct => "SetupContributionMode::Direct",
        SetupContributionMode::Recursive => "SetupContributionMode::Recursive",
    }
}

fn emit_setup_prefix_group(group: Option<GeneratedSetupPrefixGroup>) -> String {
    match group {
        Some(group) => format!(
            "Some(GeneratedSetupPrefixGroup {{ natural_len: {}, num_live_ring_elements_per_claim: {}, num_positions_per_block: {}, num_live_blocks: {}, fold_challenge_shape: {}, n_a: {}, n_b: {} }})",
            group.natural_len,
            group.num_live_ring_elements_per_claim,
            group.num_positions_per_block,
            group.num_live_blocks,
            emit_root_fold_shape(group.fold_challenge_shape),
            group.n_a,
            group.n_b,
        ),
        None => "None".to_string(),
    }
}

fn emit_fold_step(p: &LevelParams, include_setup_prefix_group: bool) -> String {
    let setup_prefix_group = setup_prefix_group_from_params(p, include_setup_prefix_group);
    if setup_prefix_group.is_none() && p.setup_contribution_mode == SetupContributionMode::Direct {
        return format!("GeneratedFold::Fold({})", emit_compact_fold_struct(p));
    }

    format!(
        "GeneratedFold::FoldWithSetupMetadata(GeneratedFoldStepWithSetupMetadata {{ fold: {}, \
         setup_prefix_group: {}, setup_contribution_mode: {} }})",
        emit_compact_fold_struct(p),
        emit_setup_prefix_group(setup_prefix_group),
        emit_setup_contribution_mode(p.setup_contribution_mode),
    )
}

fn emit_schedule_entry(
    out: &mut String,
    _key: &AkitaScheduleLookupKey,
    key_str: &str,
    schedule: &Schedule,
) -> Result<(), String> {
    writeln!(
        out,
        "    GeneratedScheduleTableEntry {{ {key_str}, folds: &[",
    )
    .map_err(|e| e.to_string())?;

    for (idx, fold) in schedule.folds.iter().enumerate() {
        let include_setup_prefix_group = idx > 0 && fold.params.setup_prefix.is_some();
        writeln!(
            out,
            "        {},",
            emit_fold_step(&fold.params, include_setup_prefix_group)
        )
        .map_err(|e| e.to_string())?;
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

fn emit_sis_modulus_profile(family: akita_types::SisModulusProfileId) -> &'static str {
    match family {
        akita_types::SisModulusProfileId::Q32Offset99 => "SisModulusProfileId::Q32Offset99",
        akita_types::SisModulusProfileId::Q64Offset59 => "SisModulusProfileId::Q64Offset59",
        akita_types::SisModulusProfileId::Q128OffsetA7F7 => "SisModulusProfileId::Q128OffsetA7F7",
    }
}

fn format_bytes(bytes: [u8; 32]) -> String {
    let values = bytes.iter().map(|byte| format!("0x{byte:02x}"));
    format!("[{}]", values.collect::<Vec<_>>().join(", "))
}

fn emit_root_fold_shape(shape: TensorChallengeShape) -> String {
    match shape {
        TensorChallengeShape::Flat => "TensorChallengeShape::Flat".to_string(),
        TensorChallengeShape::Tensor { fold_low_len } => {
            format!("TensorChallengeShape::Tensor {{ fold_low_len: {fold_low_len} }}")
        }
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
            "    sis_modulus_profile: {sis_modulus_profile},\n",
            "    sis_security_policy: SisSecurityPolicyId::{sis_security_policy},\n",
            "    sis_table_digest: SisTableDigest({sis_table_digest}),\n",
            "    ring_dimension: {ring_dimension},\n",
            "    decomposition: {decomposition},\n",
            "    ring_subfield_norm_bound: {ring_subfield_norm_bound},\n",
            "    claim_ext_degree: {claim_ext_degree},\n",
            "    chal_ext_degree: {chal_ext_degree},\n",
            "    basis_range: ({basis_min}, {basis_max}),\n",
            "    onehot_chunk_size: {onehot_chunk_size},\n",
            "    witness_chunk: {witness_chunk},\n",
            "    recursive_setup_planning: {recursive_setup_planning},\n",
            "    root_fold_shape: {root_fold_shape},\n",
            "    ring_dimensions: CATALOG_RING_DIMENSIONS,\n",
            "    ring_challenge_config_digest: {ring_challenge_config_digest},\n",
            "    key_count: {key_count},\n",
            "    key_digest: {key_digest},\n",
            "}};\n",
        ),
        ring_dims = ring_dims,
        family_name = identity.family_name,
        sis_modulus_profile = emit_sis_modulus_profile(identity.sis_modulus_profile),
        sis_security_policy = identity.sis_security_policy.name(),
        sis_table_digest = format_bytes(identity.sis_table_digest.0),
        ring_dimension = identity.ring_dimension,
        decomposition = emit_decomposition(identity.decomposition),
        ring_subfield_norm_bound = identity.ring_subfield_norm_bound,
        claim_ext_degree = identity.claim_ext_degree,
        chal_ext_degree = identity.chal_ext_degree,
        basis_min = identity.basis_range.0,
        basis_max = identity.basis_range.1,
        onehot_chunk_size = identity.onehot_chunk_size,
        witness_chunk = emit_witness_chunk(identity.witness_chunk),
        recursive_setup_planning = identity.recursive_setup_planning,
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
        match (spec.regen)(*key) {
            Ok(schedule) => entries.push((AkitaScheduleLookupKey::single(*key), schedule)),
            Err(akita_field::AkitaError::UnsupportedSchedule(_)) => {}
            Err(error) => {
                return Err(format!("{}: regen {key:?}: {error}", spec.module_name));
            }
        }
    }
    for key in &spec.group_batch_keys {
        match (spec.regen_group_batch)(key.clone()) {
            Ok(schedule) => entries.push((key.clone(), schedule)),
            Err(akita_field::AkitaError::UnsupportedSchedule(_)) => {}
            Err(error) => {
                return Err(format!(
                    "{}: regen multi-group {key:?}: {error}",
                    spec.module_name
                ));
            }
        }
    }
    entries
        .sort_by(|(left, _), (right, _)| crate::generated::runtime_schedule_key_cmp(left, right));
    Ok(entries)
}

fn schedule_uses_fold_with_setup(schedule: &Schedule) -> bool {
    schedule.folds.iter().enumerate().any(|(idx, fold)| {
        let include_setup_prefix_group = idx > 0 && fold.params.setup_prefix.is_some();
        include_setup_prefix_group
            || fold.params.setup_contribution_mode != SetupContributionMode::Direct
    })
}

/// Emit one family module (entries + embedded catalog identity).
pub fn emit_family_module(spec: &EmitSpec) -> Result<String, String> {
    let materialized = materialized_entries(spec)?;
    let uses_fold_with_setup = materialized
        .iter()
        .any(|(_, schedule)| schedule_uses_fold_with_setup(schedule));

    let mut out = String::new();
    let const_name = output_const_name(spec);
    writeln!(out, "// Generated by `{}`", spec.generator_command).map_err(|e| e.to_string())?;
    writeln!(out, "#[allow(unused_imports)]").map_err(|e| e.to_string())?;
    if uses_fold_with_setup {
        writeln!(
            out,
            "use super::{{\n    ChunkedWitnessCfg, DecompositionParams, GeneratedFoldStep, \
             GeneratedFoldStepWithSetupMetadata, GeneratedScheduleCatalogIdentity, \
             GeneratedScheduleTableEntry, GeneratedSetupPrefixGroup, GeneratedFold, \
             PolynomialGroupLayout, PrecommittedGroupParams, SetupContributionMode, \
             SisModulusProfileId, SisSecurityPolicyId, SisTableDigest, TensorChallengeShape,\n}};"
        )
        .map_err(|e| e.to_string())?;
    } else {
        writeln!(
            out,
            "use super::{{\n    ChunkedWitnessCfg, DecompositionParams, GeneratedFoldStep, \
             GeneratedScheduleCatalogIdentity, GeneratedScheduleTableEntry, GeneratedFold, \
             PolynomialGroupLayout, PrecommittedGroupParams, SisModulusProfileId, \
             SisSecurityPolicyId, SisTableDigest, TensorChallengeShape,\n}};"
        )
        .map_err(|e| e.to_string())?;
    }
    writeln!(out).map_err(|e| e.to_string())?;

    let mut memory_entries: Vec<GeneratedScheduleTableEntry> = Vec::new();
    let mut leaked_folds: Vec<&'static [GeneratedFold]> = Vec::new();

    writeln!(out, "#[rustfmt::skip]").map_err(|e| e.to_string())?;
    writeln!(
        out,
        "pub(crate) static {const_name}: &[GeneratedScheduleTableEntry] = &["
    )
    .map_err(|e| e.to_string())?;

    for (key, schedule) in materialized {
        let key_str = emit_entry_fields(&key);
        emit_schedule_entry(&mut out, &key, &key_str, &schedule)?;
        let folds = schedule_to_generated_folds(&key, &schedule);
        let folds_ref = Box::leak(folds.into_boxed_slice());
        leaked_folds.push(folds_ref);
        memory_entries.push(GeneratedScheduleTableEntry {
            final_group: key.final_group,
            precommitteds: Box::leak(key.precommitteds.into_boxed_slice()),
            folds: folds_ref,
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
