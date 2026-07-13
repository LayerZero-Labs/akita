//! Shared metadata describing every `Cfg` family that ships with a
//! generated schedule table in `akita-schedules`.
//!
//! Both the `gen_schedule_tables` binary (the offline table emitter) and
//! the drift-guard test consume [`ALL_GENERATED_FAMILIES`] so the two
//! cannot drift apart: a missing `Cfg` here is missing in both the emitted
//! artifact and the regression guard.
//!
//! This list is the one place a preset `Cfg` type is bound to its regen
//! hook and shipped table, so it lives in `akita-config` (the only crate
//! that can name the presets). The `Cfg`-free DP itself lives in
//! `akita-planner` and is reached through the `regen` glue below, which
//! derives a [`akita_planner::PlannerPolicy`] from each preset via
//! [`crate::policy_of`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_planner::{find_group_batch_schedule, find_schedule, EmitSpec, PlannerPolicy};
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, OpeningClaimsLayout, PolynomialGroupLayout,
    PrecommittedGroupParams, Schedule,
};

use crate::conservative_commitment::conservative_commit_params;
use crate::proof_optimized::{fp128, fp32, fp64};
use crate::{policy_of, tensor_verifier, CommitmentConfig};

/// Default batched opening sizes emitted for every Akita shipped family.
pub const DEFAULT_NUM_POLYS: &[usize] = &[1, 4];

/// Maximum number of precommitted groups emitted for multi-group-root generated tables.
pub const DEFAULT_GROUP_BATCH_MAX_PRECOMMITTED_GROUPS: usize = 2;

/// One generated schedule-table family.
///
/// Function-pointer fields (instead of generic `Fn` closures) keep the
/// list `const`-constructible and `'static`.
#[derive(Clone, Copy)]
pub struct GeneratedFamily {
    /// On-disk module file name (without `.rs`) and the basename used
    /// to derive the static `&[GeneratedScheduleTableEntry]` const name.
    pub module_name: &'static str,
    /// On-disk const name for the table entries array.
    pub const_name: &'static str,
    /// Cargo feature on `akita-schedules` / `akita-config` for this family.
    pub schedule_feature: &'static str,
    /// Inclusive lower bound of the `num_vars` range enumerated for
    /// this family.
    pub min_num_vars: usize,
    /// Inclusive upper bound of the `num_vars` range enumerated for
    /// this family.
    pub max_num_vars: usize,
    /// Opening-batch sizes (`num_polys`) enumerated for this family.
    pub num_polys: &'static [usize],
    /// Pure DP regeneration that ignores any shipped table
    /// (`find_schedule(key, &policy_of::<Cfg>(), …)`).
    pub regen: fn(PolynomialGroupLayout) -> Result<Schedule, AkitaError>,
    /// Pure multi-group DP regeneration that ignores any shipped table.
    pub regen_group_batch: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
    /// Whether this family ships multi-group-root rows in its generated table.
    pub emit_group_batch: bool,
    /// Grouped-root keys enumerated for this generated family.
    pub group_batch_keys: fn(&GeneratedFamily) -> Result<Vec<AkitaScheduleLookupKey>, AkitaError>,
    /// `Cfg::runtime_schedule(key)` — the table fast path when an entry
    /// exists, falling through to the DP otherwise. Used by diagnostic
    /// comparisons against the shipped table.
    pub table_backed: fn(PolynomialGroupLayout) -> Result<Schedule, AkitaError>,
    pub policy: fn() -> PlannerPolicy,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
}

/// Build the ordered key cross-product emitted for `family`.
///
/// Scalar keys emitted for `family`. The emitter combines these with multi-group
/// keys and sorts the unified catalog by the generated schedule lookup order.
///
/// # Errors
///
/// Returns an error if the synthetic opening batch fails to build
/// or the lookup-key derivation fails (both indicate a malformed
/// `(min_num_vars, max_num_vars)` range).
pub fn family_keys(family: &GeneratedFamily) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
    let mut keys = Vec::with_capacity(
        family
            .num_polys
            .len()
            .saturating_mul(family.max_num_vars.saturating_sub(family.min_num_vars) + 1),
    );
    for &num_polys in family.num_polys {
        for nv in family.min_num_vars..=family.max_num_vars {
            let opening_batch = OpeningClaimsLayout::new(nv, num_polys)?;
            keys.push(opening_batch.root_final_group_layout()?);
        }
    }
    Ok(keys)
}

/// Pure DP regeneration for `Cfg` — never consults the shipped table.
fn regen<Cfg: CommitmentConfig>(key: PolynomialGroupLayout) -> Result<Schedule, AkitaError> {
    find_schedule(
        key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
}

/// Pure multi-group DP regeneration for `Cfg` — never consults the shipped table.
fn regen_group_batch<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<Schedule, AkitaError> {
    find_group_batch_schedule(
        &key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
}

/// Table-backed resolution for `Cfg` — table hit when present, otherwise
/// the DP fallback baked into `runtime_schedule`.
fn table_backed<Cfg: CommitmentConfig>(key: PolynomialGroupLayout) -> Result<Schedule, AkitaError> {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(key))
}

fn family_policy<Cfg: CommitmentConfig>() -> PlannerPolicy {
    policy_of::<Cfg>()
}

fn group_batch_keys<Cfg: CommitmentConfig>(
    family: &GeneratedFamily,
) -> Result<Vec<AkitaScheduleLookupKey>, AkitaError> {
    if !family.emit_group_batch {
        return Ok(Vec::new());
    }
    if Cfg::decomposition().log_commit_bound != 1 {
        return Ok(Vec::new());
    }

    let min_precommitted_num_vars = family
        .min_num_vars
        .max(policy_of::<Cfg>().ring_dimension.trailing_zeros() as usize + 1);
    let mut mains = family_keys(family)?;
    // Scalar table emission uses `DEFAULT_NUM_POLYS = [1, 4]`, but the recursive
    // multi-group profile opens a 2-polynomial final group. Enumerate that main
    // shape here so regen keeps the catalog hit (PR #292 hand-inserted one row;
    // a full table regen otherwise drops it and forces DP fallback).
    if !family.num_polys.contains(&2) {
        for nv in family.min_num_vars..=family.max_num_vars {
            mains.push(PolynomialGroupLayout::new(nv, 2));
        }
    }
    let mut keys = Vec::new();
    for main in mains {
        let pre_num_vars = main.num_vars() / 2;
        if pre_num_vars < min_precommitted_num_vars {
            continue;
        }
        let patterns = precommitted_group_patterns(main.num_polynomials());
        for pattern in patterns {
            let mut precommitteds = Vec::with_capacity(pattern.len());
            let mut supported = true;
            for &num_polys in &pattern {
                let pre_key = PolynomialGroupLayout::new(pre_num_vars, num_polys);
                let params = match conservative_commit_params::<Cfg>(&pre_key) {
                    Ok(params) => params,
                    Err(_) => {
                        supported = false;
                        break;
                    }
                };
                precommitteds.push(PrecommittedGroupParams::from_params(pre_key, &params));
            }
            if !supported {
                continue;
            }
            let candidate = AkitaScheduleLookupKey {
                final_group: main,
                precommitteds,
            };
            if regen_group_batch::<Cfg>(candidate.clone()).is_ok() {
                keys.push(candidate);
            }
        }
    }
    keys.sort_by(akita_planner::runtime_schedule_key_cmp);
    Ok(keys)
}

fn precommitted_group_patterns(main_num_polynomials: usize) -> Vec<Vec<usize>> {
    let first_group = 1usize;
    let second_group = (main_num_polynomials / 2).max(1);
    let mut patterns = vec![vec![first_group]];
    if DEFAULT_GROUP_BATCH_MAX_PRECOMMITTED_GROUPS >= 2 {
        patterns.push(vec![first_group, second_group]);
    }
    patterns
}

macro_rules! family_row {
    (group_batch, $module:literal, $const:literal, $feat:literal, $min:expr, $max:expr, $cfg:ty) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            min_num_vars: $min,
            max_num_vars: $max,
            num_polys: DEFAULT_NUM_POLYS,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            emit_group_batch: true,
            group_batch_keys: group_batch_keys::<$cfg>,
            table_backed: table_backed::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            fold_challenge_shape_at_level:
                <$cfg as CommitmentConfig>::fold_challenge_shape_at_level,
        }
    };
    ($module:literal, $const:literal, $feat:literal, $min:expr, $max:expr, $cfg:ty) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            min_num_vars: $min,
            max_num_vars: $max,
            num_polys: DEFAULT_NUM_POLYS,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            emit_group_batch: false,
            group_batch_keys: group_batch_keys::<$cfg>,
            table_backed: table_backed::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            fold_challenge_shape_at_level:
                <$cfg as CommitmentConfig>::fold_challenge_shape_at_level,
        }
    };
}

/// Minimal [`EmitSpec`] for refreshing `generated/mod.rs` wiring only.
pub fn wiring_emit_spec(family: &GeneratedFamily, output_dir: std::path::PathBuf) -> EmitSpec {
    EmitSpec {
        module_name: family.module_name,
        const_name: family.const_name,
        family_name: family.module_name,
        schedule_feature: family.schedule_feature,
        policy: (family.policy)(),
        keys: Vec::new(),
        group_batch_keys: Vec::new(),
        emit_group_batch: family.emit_group_batch,
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
        fold_challenge_shape_at_level: family.fold_challenge_shape_at_level,
        generator_command: "",
    }
}

/// Adapt one [`GeneratedFamily`] into an [`EmitSpec`] for the planner emitter.
pub fn emit_spec_for_family(
    family: &GeneratedFamily,
    output_dir: std::path::PathBuf,
    generator_command: &'static str,
) -> Result<EmitSpec, AkitaError> {
    Ok(EmitSpec {
        module_name: family.module_name,
        const_name: family.const_name,
        family_name: family.module_name,
        schedule_feature: family.schedule_feature,
        policy: (family.policy)(),
        keys: family_keys(family)?,
        group_batch_keys: (family.group_batch_keys)(family)?,
        emit_group_batch: family.emit_group_batch,
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
        fold_challenge_shape_at_level: family.fold_challenge_shape_at_level,
        generator_command,
    })
}

/// Every `Cfg` that ships with a generated schedule table.
///
/// Adding a new preset with a generated table requires adding a row
/// here; both the table emitter and the drift-guard test pick it up
/// automatically.
pub const ALL_GENERATED_FAMILIES: &[GeneratedFamily] = &[
    family_row!(
        "fp128_d128_full",
        "FP128_D128_FULL_SCHEDULES",
        "fp128-d128-full",
        1,
        50,
        fp128::D128Full
    ),
    family_row!(
        group_batch,
        "fp128_d128_onehot",
        "FP128_D128_ONEHOT_SCHEDULES",
        "fp128-d128-onehot",
        1,
        50,
        fp128::D128OneHot
    ),
    family_row!(
        group_batch,
        "fp128_d64_onehot",
        "FP128_D64_ONEHOT_SCHEDULES",
        "fp128-d64-onehot",
        1,
        50,
        fp128::D64OneHot
    ),
    family_row!(
        "fp128_d64_full",
        "FP128_D64_FULL_SCHEDULES",
        "fp128-d64-full",
        1,
        50,
        fp128::D64Full
    ),
    family_row!(
        group_batch,
        "fp128_d64_onehot_tensor",
        "FP128_D64_ONEHOT_TENSOR_SCHEDULES",
        "fp128-d64-onehot-tensor",
        1,
        50,
        tensor_verifier::fp128::D64OneHotTensor
    ),
    // Multi-chunk (distributed-prover) companions of the D64 families. Same
    // `(num_vars, num_polynomials)` keys as their siblings; schedules differ
    // because the policy prices the chunked witness layout.
    family_row!(
        "fp128_d64_onehot_multi_chunk",
        "FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES",
        "fp128-d64-onehot-multi-chunk",
        1,
        50,
        fp128::D64OneHotMultiChunk
    ),
    family_row!(
        "fp128_d64_onehot_multi_chunk_w2r2",
        "FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES",
        "fp128-d64-onehot-multi-chunk-w2r2",
        1,
        50,
        fp128::D64OneHotMultiChunkW2R2
    ),
    family_row!(
        "fp128_d64_onehot_multi_chunk_w4r2",
        "FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES",
        "fp128-d64-onehot-multi-chunk-w4r2",
        1,
        50,
        fp128::D64OneHotMultiChunkW4R2
    ),
    family_row!(
        "fp128_d64_full_multi_chunk",
        "FP128_D64_FULL_MULTI_CHUNK_SCHEDULES",
        "fp128-d64-full-multi-chunk",
        1,
        50,
        fp128::D64FullMultiChunk
    ),
    family_row!(
        "fp64_d128",
        "FP64_D128_SCHEDULES",
        "fp64-d128",
        1,
        32,
        fp64::D128Full
    ),
    family_row!(
        "fp64_d128_onehot",
        "FP64_D128_ONEHOT_SCHEDULES",
        "fp64-d128-onehot",
        1,
        32,
        fp64::D128OneHot
    ),
    family_row!(
        "fp64_d256_onehot",
        "FP64_D256_ONEHOT_SCHEDULES",
        "fp64-d256-onehot",
        1,
        32,
        fp64::D256OneHot
    ),
    family_row!(
        "fp32_d128_onehot",
        "FP32_D128_ONEHOT_SCHEDULES",
        "fp32-d128-onehot",
        1,
        32,
        fp32::D128OneHot
    ),
    family_row!(
        "fp32_d256_onehot",
        "FP32_D256_ONEHOT_SCHEDULES",
        "fp32-d256-onehot",
        1,
        32,
        fp32::D256OneHot
    ),
];
