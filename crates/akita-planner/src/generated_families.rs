//! Shared metadata describing every `Cfg` family that ships with a
//! generated schedule table in `akita-schedules`.
//!
//! Both the `gen_schedule_tables` binary (the offline table emitter) and
//! the drift-guard test consume [`ALL_GENERATED_FAMILIES`] so the two
//! cannot drift apart: a missing `Cfg` here is missing in both the emitted
//! artifact and the regression guard.
//!
//! This list is the one place a preset `Cfg` type is bound to its regen
//! hook and generated table. It is behind the `catalog-gen` feature because
//! that offline path is allowed to name `akita-config` presets. Normal
//! runtime callers consume the generated tables from `akita-schedules`.

use crate::{find_schedule, runtime_schedule_key_cmp, EmitSpec, PlannerPolicy};
use akita_challenges::SparseChallengeConfig;
use akita_field::AkitaError;
use akita_schedules::GeneratedScheduleTable;
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{
    AkitaScheduleLookupKey, CommittedGroupProfile, FoldSchedule, PolynomialGroupLayout,
};

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig, RecursiveCommitmentConfig};

const FP128_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::new(15, 2),
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::new(16, 2),
    PolynomialGroupLayout::new(17, 4),
    PolynomialGroupLayout::singleton(24),
    PolynomialGroupLayout::singleton(26),
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(30),
    PolynomialGroupLayout::singleton(32),
    PolynomialGroupLayout::singleton(44),
    PolynomialGroupLayout::singleton(50),
];

const FP128_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(12),
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::new(14, 2),
    PolynomialGroupLayout::singleton(15),
    PolynomialGroupLayout::new(15, 4),
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::new(16, 2),
    PolynomialGroupLayout::singleton(18),
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::new(20, 2),
    PolynomialGroupLayout::new(20, 4),
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(30),
    PolynomialGroupLayout::new(30, 4),
    PolynomialGroupLayout::singleton(32),
    PolynomialGroupLayout::new(32, 4),
    PolynomialGroupLayout::singleton(36),
    PolynomialGroupLayout::singleton(40),
    PolynomialGroupLayout::singleton(44),
    PolynomialGroupLayout::singleton(50),
];

const FP128_ONEHOT_MULTI_CHUNK_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::singleton(32),
];

const FP128_ONEHOT_RECURSIVE_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(36)];

const FP128_ONEHOT_MULTI_CHUNK_W2R2_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(32),
];

const FP128_ONEHOT_MULTI_CHUNK_W4R2_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(32)];

const FP128_DENSE_MULTI_CHUNK_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(16)];

const FP32_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::singleton(26),
];

const FP32_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::new(16, 2),
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(30),
];

const FP64_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::singleton(26),
];

const FP64_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(30),
];

/// One generated schedule-table family.
///
/// Function-pointer fields (instead of generic `Fn` closures) keep the
/// list `const`-constructible and `'static`.
#[derive(Clone, Copy)]
pub struct GeneratedFamily {
    /// On-disk module file name (without `.rs`) and the basename used
    /// to derive the static `&[GeneratedFoldScheduleEntry]` const name.
    pub module_name: &'static str,
    /// On-disk const name for the table entries array.
    pub const_name: &'static str,
    /// Cargo feature on `akita-schedules` / `akita-config` for this family.
    pub schedule_feature: &'static str,
    /// Scalar opening keys emitted for this family.
    pub scalar_keys: &'static [PolynomialGroupLayout],
    /// Pure DP regeneration that ignores any generated table
    /// (`find_schedule(&single_key, &[], &policy_of::<Cfg>(), …)`).
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    /// Pure multi-group DP regeneration that ignores any generated table.
    pub regen_group_batch:
        fn(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>) -> Result<FoldSchedule, AkitaError>,
    /// Grouped-root keys enumerated for this generated family.
    #[allow(clippy::type_complexity)]
    pub group_batch_keys:
        fn() -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError>,
    /// Strict table-backed runtime resolution. A missing row is unsupported.
    pub select_schedule_for_key: fn(AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError>,
    /// The generated catalog linked for this family, when its feature is active.
    pub schedule_catalog: fn() -> Option<GeneratedScheduleTable>,
    pub policy: fn() -> PlannerPolicy,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    /// Build one caller-requested precommit descriptor and its honest fold policy.
    pub explicit_precommitted_group:
        fn(
            PolynomialGroupLayout,
        ) -> Result<(CommittedGroupProfile, HonestFoldPolicySpec), AkitaError>,
}

/// Build the ordered key cross-product emitted for `family`.
///
/// Scalar keys emitted for `family`. The emitter combines these with multi-group
/// keys and sorts the unified catalog by the generated schedule lookup order.
///
/// # Errors
///
/// Returns an error if key enumeration fails.
pub fn family_keys(family: &GeneratedFamily) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
    let mut keys = family.scalar_keys.to_vec();
    keys.sort_by(|left, right| {
        runtime_schedule_key_cmp(
            &AkitaScheduleLookupKey::single(*left),
            &AkitaScheduleLookupKey::single(*right),
        )
    });
    keys.dedup();
    Ok(keys)
}

/// Scalar keys physically emitted into `family`'s catalog.
///
pub fn emitted_scalar_keys(
    family: &GeneratedFamily,
) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
    family_keys(family)
}

fn plan_regen<Cfg: CommitmentConfig>(
    key: &AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: &[HonestFoldPolicySpec],
) -> Result<FoldSchedule, AkitaError> {
    let planned = find_schedule(
        key,
        honest_fold_policy_of::<Cfg>(),
        precommitted_honest_fold_policies,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}

/// Pure DP regeneration for `Cfg` — never consults the generated table.
fn regen<Cfg: CommitmentConfig>(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    plan_regen::<Cfg>(&AkitaScheduleLookupKey::single(key), &[])
}

fn sole_profile<Cfg: CommitmentConfig>(
    group: PolynomialGroupLayout,
) -> Result<CommittedGroupProfile, AkitaError> {
    let schedule = regen::<Cfg>(group)?;
    Ok(CommittedGroupProfile::from_params(
        group,
        &schedule.root.params.final_group.commitment,
    ))
}

/// Pure multi-group DP regeneration for `Cfg` — never consults the generated table.
fn regen_group_batch<Cfg: CommitmentConfig + 'static>(
    key: AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: Vec<HonestFoldPolicySpec>,
) -> Result<FoldSchedule, AkitaError> {
    plan_regen::<Cfg>(&key, &precommitted_honest_fold_policies)
}

fn select_schedule_for_key<Cfg: CommitmentConfig>(
    key: AkitaScheduleLookupKey,
) -> Result<FoldSchedule, AkitaError> {
    Cfg::select_schedule_for_key(&key).map(akita_schedules::ResolvedScheduleRow::into_schedule)
}

fn schedule_catalog<Cfg: CommitmentConfig>() -> Option<GeneratedScheduleTable> {
    Cfg::schedule_catalog()
}

fn family_policy<Cfg: CommitmentConfig>() -> PlannerPolicy {
    policy_of::<Cfg>()
}

fn sorted_group_batch_keys(
    mut keys: Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>,
) -> Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)> {
    keys.sort_by(|left, right| runtime_schedule_key_cmp(&left.0, &right.0));
    keys
}

fn no_group_batch_keys(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    Ok(Vec::new())
}

fn fp128_onehot_group_batch_keys(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let mut keys = recursive_onehot_profile_keys::<fp128::OneHot>()?;
    keys.push(heterogeneous_onehot_catalog_key()?);
    keys.extend(onehot_group_batch_test_keys::<fp128::OneHot>()?);
    Ok(sorted_group_batch_keys(keys))
}

fn fp128_onehot_multichunk_group_batch_keys(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    Ok(sorted_group_batch_keys(recursive_onehot_profile_keys::<
        fp128::OneHotMultiChunk,
    >()?))
}

fn fp128_onehot_multichunk_w2r2_group_batch_keys(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    type Cfg = fp128::OneHotMultiChunkW2R2;
    let group = PolynomialGroupLayout::new(14, 1);
    let precommitted = sole_profile::<Cfg>(group)?;
    Ok(vec![(
        AkitaScheduleLookupKey {
            final_group: group,
            prior_group_profiles: vec![precommitted],
        },
        vec![honest_fold_policy_of::<Cfg>()],
    )])
}

/// Shipped fp32 precommit-plus-final workload exercised by the extension-field
/// multi-group PCS end-to-end test.
fn fp32_onehot_group_batch_keys(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    type Cfg = fp32::OneHot;
    let group = PolynomialGroupLayout::new(14, 1);
    let precommitted = sole_profile::<Cfg>(group)?;
    Ok(vec![(
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(20, 1),
            prior_group_profiles: vec![precommitted],
        },
        vec![honest_fold_policy_of::<Cfg>()],
    )])
}

fn recursive_onehot_profile_keys<BaseCfg: CommitmentConfig + 'static>(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let precommitted_group = PolynomialGroupLayout::new(16, 1);
    let precommitted = sole_profile::<BaseCfg>(precommitted_group)?;
    Ok(vec![(
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            prior_group_profiles: vec![precommitted, precommitted],
        },
        vec![
            honest_fold_policy_of::<BaseCfg>(),
            honest_fold_policy_of::<BaseCfg>(),
        ],
    )])
}

fn heterogeneous_onehot_catalog_key(
) -> Result<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>), AkitaError> {
    let onehot_group = PolynomialGroupLayout::new(14, 1);
    let dense_group = PolynomialGroupLayout::new(15, 2);
    let onehot_policy = honest_fold_policy_of::<fp128::OneHot>();
    let dense_policy = honest_fold_policy_of::<fp128::Dense>();
    let onehot = sole_profile::<fp128::OneHot>(onehot_group)?;
    let dense = sole_profile::<fp128::Dense>(dense_group)?;
    Ok((
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(16, 1),
            prior_group_profiles: vec![onehot, dense],
        },
        vec![onehot_policy, dense_policy],
    ))
}

fn onehot_group_batch_test_keys<BaseCfg: CommitmentConfig + 'static>(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let singleton_pre = sole_profile::<BaseCfg>(PolynomialGroupLayout::new(14, 1))?;
    let pair_pre = sole_profile::<BaseCfg>(PolynomialGroupLayout::new(14, 2))?;
    let policy = honest_fold_policy_of::<BaseCfg>();
    Ok(vec![
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 2),
                prior_group_profiles: vec![singleton_pre],
            },
            vec![policy],
        ),
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 4),
                prior_group_profiles: vec![singleton_pre, singleton_pre],
            },
            vec![policy, policy],
        ),
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 4),
                prior_group_profiles: vec![singleton_pre, singleton_pre, singleton_pre],
            },
            vec![policy, policy, policy],
        ),
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 1),
                prior_group_profiles: vec![pair_pre],
            },
            vec![policy],
        ),
    ])
}

macro_rules! family_row {
    ($module:literal, $const:literal, $feat:literal, $keys:expr, $cfg:ty, $group_keys:expr) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            scalar_keys: $keys,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            group_batch_keys: $group_keys,
            select_schedule_for_key: select_schedule_for_key::<$cfg>,
            schedule_catalog: schedule_catalog::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            explicit_precommitted_group: explicit_precommitted_group::<$cfg>,
        }
    };
    // Recursion adapter families: like `group_batch`, but grouped keys come from
    // the fixed recursive profiling shape rather than the generic per-`Cfg` grid.
    (recursive, $module:literal, $const:literal, $feat:literal, $keys:expr, $cfg:ty, $base_cfg:ty, $group_keys:expr) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            scalar_keys: $keys,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            group_batch_keys: $group_keys,
            select_schedule_for_key: select_schedule_for_key::<$cfg>,
            schedule_catalog: schedule_catalog::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            explicit_precommitted_group: explicit_precommitted_group::<$base_cfg>,
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
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
        generator_command: "",
    }
}

/// Adapt one [`GeneratedFamily`] into an [`EmitSpec`] for the planner emitter.
pub fn emit_spec_for_family(
    family: &GeneratedFamily,
    output_dir: std::path::PathBuf,
    generator_command: &'static str,
) -> Result<EmitSpec, AkitaError> {
    let policy = (family.policy)();
    let group_batch_keys = (family.group_batch_keys)()?;
    Ok(EmitSpec {
        module_name: family.module_name,
        const_name: family.const_name,
        family_name: family.module_name,
        schedule_feature: family.schedule_feature,
        policy,
        keys: emitted_scalar_keys(family)?,
        group_batch_keys,
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
        generator_command,
    })
}

fn explicit_precommitted_group<Cfg: CommitmentConfig + 'static>(
    group: PolynomialGroupLayout,
) -> Result<(CommittedGroupProfile, HonestFoldPolicySpec), AkitaError> {
    Ok((sole_profile::<Cfg>(group)?, honest_fold_policy_of::<Cfg>()))
}

/// Every `Cfg` that has a generated schedule table.
///
/// Adding a new preset with a generated table requires adding a row
/// here; both the table emitter and the drift-guard test pick it up
/// automatically.
pub const ALL_GENERATED_FAMILIES: &[GeneratedFamily] = &[
    family_row!(
        "fp128_onehot",
        "FP128_ONEHOT_SCHEDULES",
        "fp128-onehot",
        FP128_ONEHOT_KEYS,
        fp128::OneHot,
        fp128_onehot_group_batch_keys
    ),
    family_row!(
        recursive,
        "fp128_onehot_recursive",
        "FP128_ONEHOT_RECURSIVE_SCHEDULES",
        "fp128-onehot-recursive",
        FP128_ONEHOT_RECURSIVE_KEYS,
        RecursiveCommitmentConfig<fp128::OneHot>,
        fp128::OneHot,
        recursive_onehot_profile_keys::<fp128::OneHot>
    ),
    family_row!(
        recursive,
        "fp128_onehot_recursive_multi_chunk_w8r2",
        "FP128_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES",
        "fp128-onehot-recursive-multi-chunk-w8r2",
        &[],
        RecursiveCommitmentConfig<fp128::OneHotMultiChunk>,
        fp128::OneHotMultiChunk,
        recursive_onehot_profile_keys::<fp128::OneHotMultiChunk>
    ),
    family_row!(
        "fp128_dense",
        "FP128_DENSE_SCHEDULES",
        "fp128-dense",
        FP128_DENSE_KEYS,
        fp128::Dense,
        no_group_batch_keys
    ),
    family_row!(
        "fp128_onehot_multi_chunk",
        "FP128_ONEHOT_MULTI_CHUNK_SCHEDULES",
        "fp128-onehot-multi-chunk",
        FP128_ONEHOT_MULTI_CHUNK_KEYS,
        fp128::OneHotMultiChunk,
        fp128_onehot_multichunk_group_batch_keys
    ),
    family_row!(
        "fp128_onehot_multi_chunk_w2r2",
        "FP128_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES",
        "fp128-onehot-multi-chunk-w2r2",
        FP128_ONEHOT_MULTI_CHUNK_W2R2_KEYS,
        fp128::OneHotMultiChunkW2R2,
        fp128_onehot_multichunk_w2r2_group_batch_keys
    ),
    family_row!(
        "fp128_onehot_multi_chunk_w4r2",
        "FP128_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES",
        "fp128-onehot-multi-chunk-w4r2",
        FP128_ONEHOT_MULTI_CHUNK_W4R2_KEYS,
        fp128::OneHotMultiChunkW4R2,
        no_group_batch_keys
    ),
    family_row!(
        "fp128_dense_multi_chunk",
        "FP128_DENSE_MULTI_CHUNK_SCHEDULES",
        "fp128-dense-multi-chunk",
        FP128_DENSE_MULTI_CHUNK_KEYS,
        fp128::DenseMultiChunk,
        no_group_batch_keys
    ),
    family_row!(
        "fp64_dense",
        "FP64_DENSE_SCHEDULES",
        "fp64-dense",
        FP64_DENSE_KEYS,
        fp64::Dense,
        no_group_batch_keys
    ),
    family_row!(
        "fp64_onehot",
        "FP64_ONEHOT_SCHEDULES",
        "fp64-onehot",
        FP64_ONEHOT_KEYS,
        fp64::OneHot,
        no_group_batch_keys
    ),
    family_row!(
        "fp32_dense",
        "FP32_DENSE_SCHEDULES",
        "fp32-dense",
        FP32_DENSE_KEYS,
        fp32::Dense,
        no_group_batch_keys
    ),
    family_row!(
        "fp32_onehot",
        "FP32_ONEHOT_SCHEDULES",
        "fp32-onehot",
        FP32_ONEHOT_KEYS,
        fp32::OneHot,
        fp32_onehot_group_batch_keys
    ),
];
