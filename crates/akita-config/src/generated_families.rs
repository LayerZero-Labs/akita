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
use akita_planner::{find_schedule, EmitSpec, PlannerPolicy};
use akita_types::{AkitaScheduleInputs, AkitaScheduleLookupKey, OpeningBatch, Schedule};

use crate::proof_optimized::{fp128, fp32, fp64};
use crate::{policy_of, tensor_verifier, CommitmentConfig};

/// Default batched opening sizes emitted for every Akita shipped family.
pub const DEFAULT_NUM_POLYS: &[usize] = &[1, 4];

/// One generated schedule-table family.
///
/// Function-pointer fields (instead of generic `Fn` closures) keep the
/// list `const`-constructible and `'static`.
#[derive(Clone, Copy)]
pub struct GeneratedFamily {
    /// On-disk module file name (without `.rs`) and the basename used
    /// to derive the static `&[GeneratedScheduleTableEntry]` const name.
    /// The `_zk` suffix is appended by the binary at emit time.
    pub module_name: &'static str,
    /// On-disk const name for the table entries array. The binary
    /// rewrites `_SCHEDULES` -> `_ZK_SCHEDULES` when the `zk` feature
    /// is enabled.
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
    pub regen: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
    /// `Cfg::runtime_schedule(key)` — the table fast path when an entry
    /// exists, falling through to the DP otherwise. Used by diagnostic
    /// comparisons against the shipped table.
    pub table_backed: fn(AkitaScheduleLookupKey) -> Result<Schedule, AkitaError>,
    pub policy: fn() -> PlannerPolicy,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
}

/// Build the ordered key cross-product emitted for `family`.
///
/// The order matches what `gen_schedule_tables` writes to disk: all
/// singleton keys first, then each batched `num_polys` block, each block
/// ordered by `num_vars` ascending. Drift-guard tests assert positional
/// equality against the shipped table, so this ordering doubles as the
/// canonical row order.
///
/// # Errors
///
/// Returns an error if the synthetic opening batch fails to build
/// or the lookup-key derivation fails (both indicate a malformed
/// `(min_num_vars, max_num_vars)` range).
pub fn family_keys(family: &GeneratedFamily) -> Result<Vec<AkitaScheduleLookupKey>, AkitaError> {
    let mut keys = Vec::with_capacity(
        family
            .num_polys
            .len()
            .saturating_mul(family.max_num_vars.saturating_sub(family.min_num_vars) + 1),
    );
    for &num_polys in family.num_polys {
        for nv in family.min_num_vars..=family.max_num_vars {
            let opening_batch = OpeningBatch::same_point(nv, num_polys)?;
            keys.push(AkitaScheduleLookupKey::new_from_opening_batch(
                &opening_batch,
            )?);
        }
    }
    Ok(keys)
}

/// Pure DP regeneration for `Cfg` — never consults the shipped table.
fn regen<Cfg: CommitmentConfig>(key: AkitaScheduleLookupKey) -> Result<Schedule, AkitaError> {
    find_schedule(
        key,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
}

/// Table-backed resolution for `Cfg` — table hit when present, otherwise
/// the DP fallback baked into `runtime_schedule`.
fn table_backed<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<Schedule, AkitaError> {
    Cfg::runtime_schedule(key)
}

fn family_policy<Cfg: CommitmentConfig>() -> PlannerPolicy {
    policy_of::<Cfg>()
}

macro_rules! family_row {
    ($module:literal, $const:literal, $feat:literal, $min:expr, $max:expr, $cfg:ty) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            min_num_vars: $min,
            max_num_vars: $max,
            num_polys: DEFAULT_NUM_POLYS,
            regen: regen::<$cfg>,
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
        output_dir,
        regen: family.regen,
        ring_challenge_config: family.ring_challenge_config,
        fold_challenge_shape_at_level: family.fold_challenge_shape_at_level,
        zk_enabled: false,
        generator_command: "",
    }
}

/// Adapt one [`GeneratedFamily`] into an [`EmitSpec`] for the planner emitter.
pub fn emit_spec_for_family(
    family: &GeneratedFamily,
    output_dir: std::path::PathBuf,
    zk_enabled: bool,
    generator_command: &'static str,
) -> Result<EmitSpec, AkitaError> {
    Ok(EmitSpec {
        module_name: family.module_name,
        const_name: family.const_name,
        family_name: family.module_name,
        schedule_feature: family.schedule_feature,
        policy: (family.policy)(),
        keys: family_keys(family)?,
        output_dir,
        regen: family.regen,
        ring_challenge_config: family.ring_challenge_config,
        fold_challenge_shape_at_level: family.fold_challenge_shape_at_level,
        zk_enabled,
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
        "fp128_d128_onehot",
        "FP128_D128_ONEHOT_SCHEDULES",
        "fp128-d128-onehot",
        1,
        50,
        fp128::D128OneHot
    ),
    family_row!(
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
        "fp128_d64_onehot_tensor",
        "FP128_D64_ONEHOT_TENSOR_SCHEDULES",
        "fp128-d64-onehot-tensor",
        1,
        50,
        tensor_verifier::fp128::D64OneHotTensor
    ),
    // Tiered companion of `fp128_d64_onehot`
    #[cfg(not(feature = "zk"))]
    family_row!(
        "fp128_d64_onehot_tiered",
        "FP128_D64_ONEHOT_TIERED_SCHEDULES",
        "fp128-d64-onehot-tiered",
        1,
        50,
        fp128::D64OneHotTiered
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
