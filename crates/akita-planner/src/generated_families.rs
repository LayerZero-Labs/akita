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

use std::any::TypeId;

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

struct ScalarPreplan {
    source: TypeId,
    key: PolynomialGroupLayout,
    schedule: FoldSchedule,
}

/// Exact scalar schedules already needed while preparing one generator run.
///
/// Entries are keyed by the producer configuration rather than the consuming
/// family. Recursive families can therefore use a base configuration's frozen
/// profile without aliasing their own scalar schedules. The session is dropped
/// before parallel rendering and is never persisted or iterated for output.
#[derive(Default)]
pub struct GenerationPreplans {
    scalar: Vec<ScalarPreplan>,
}

impl GenerationPreplans {
    fn scalar<Cfg: CommitmentConfig + 'static>(
        &mut self,
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        let source = TypeId::of::<Cfg>();
        if let Some(preplanned) = self
            .scalar
            .iter()
            .find(|preplanned| preplanned.source == source && preplanned.key == key)
        {
            return Ok(preplanned.schedule.clone());
        }
        let schedule = regen::<Cfg>(key)?;
        self.scalar.push(ScalarPreplan {
            source,
            key,
            schedule: schedule.clone(),
        });
        Ok(schedule)
    }

    /// Copy exact producer results into a completed spec before rendering.
    pub fn attach_to_spec(&self, family: &GeneratedFamily, spec: &mut EmitSpec) {
        let source = (family.scalar_plan_source)();
        spec.preplanned_scalar = spec
            .keys
            .iter()
            .filter_map(|key| {
                self.scalar
                    .iter()
                    .find(|preplanned| preplanned.source == source && preplanned.key == *key)
                    .map(|preplanned| (*key, preplanned.schedule.clone()))
            })
            .collect();
    }
}

type GroupBatchKeys = Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>;
type GroupBatchKeyGenerator = fn(&mut GenerationPreplans) -> Result<GroupBatchKeys, AkitaError>;
type ExplicitPrecommittedGroupGenerator =
    fn(
        &mut GenerationPreplans,
        PolynomialGroupLayout,
    ) -> Result<(CommittedGroupProfile, HonestFoldPolicySpec), AkitaError>;

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
    // Produces the frozen profile for the precommit half of
    // `fp64_dense_group_batch_keys`, which needs 16: at 14 or 15 the prover and
    // the planned schedule disagree on the fold-level-1 witness length.
    PolynomialGroupLayout::singleton(16),
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
    /// Exact producer type used to distinguish scalar preplans.
    scalar_plan_source: fn() -> TypeId,
    /// Pure DP regeneration that ignores any generated table
    /// (`find_schedule(&single_key, &[], &policy_of::<Cfg>(), …)`).
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    /// Pure multi-group DP regeneration that ignores any generated table.
    pub regen_group_batch:
        fn(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>) -> Result<FoldSchedule, AkitaError>,
    /// Grouped-root keys enumerated for this generated family.
    pub group_batch_keys: GroupBatchKeyGenerator,
    /// Strict table-backed runtime resolution. A missing row is unsupported.
    pub select_schedule_for_key: fn(AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError>,
    /// The generated catalog linked for this family, when its feature is active.
    pub schedule_catalog: fn() -> Option<GeneratedScheduleTable>,
    pub policy: fn() -> PlannerPolicy,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    /// Build one caller-requested precommit descriptor and its honest fold policy.
    pub explicit_precommitted_group: ExplicitPrecommittedGroupGenerator,
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

/// Frozen profile a group commits with when it has no precommitted groups.
///
/// Generation cannot read the catalog it is producing, so this plans the row
/// instead of selecting it. `CommitmentConfig::profile_without_precommitted_groups`
/// is the runtime counterpart, and
/// `every_grouped_precommitted_descriptor_has_a_generated_producer` asserts the two
/// agree on every shipped descriptor.
fn planned_profile_without_precommitted_groups<Cfg: CommitmentConfig + 'static>(
    preplans: &mut GenerationPreplans,
    group: PolynomialGroupLayout,
) -> Result<CommittedGroupProfile, AkitaError> {
    let schedule = preplans.scalar::<Cfg>(group)?;
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
    _preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    Ok(Vec::new())
}

fn fp128_onehot_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let mut keys = recursive_onehot_profile_keys::<fp128::OneHot>(preplans)?;
    keys.push(heterogeneous_onehot_catalog_key(preplans)?);
    keys.extend(onehot_group_batch_test_keys::<fp128::OneHot>(preplans)?);
    // Single-poly pre + single-poly final: the `fp128 × OneHot × pre` matrix
    // cell. Every other combined OneHot row is heterogeneous or multi-poly.
    keys.extend(single_pre_group_batch_keys::<fp128::OneHot>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(16, 1),
    )?);
    keys.extend(single_pre_group_batch_keys::<fp128::OneHot>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(20, 1),
    )?);
    Ok(sorted_group_batch_keys(keys))
}

fn fp128_onehot_multichunk_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    Ok(sorted_group_batch_keys(recursive_onehot_profile_keys::<
        fp128::OneHotMultiChunk,
    >(preplans)?))
}

fn fp128_onehot_multichunk_w2r2_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    type Cfg = fp128::OneHotMultiChunkW2R2;
    let group = PolynomialGroupLayout::new(14, 1);
    let precommitted = planned_profile_without_precommitted_groups::<Cfg>(preplans, group)?;
    Ok(vec![(
        AkitaScheduleLookupKey {
            final_group: group,
            precommitteds: vec![precommitted],
        },
        vec![honest_fold_policy_of::<Cfg>()],
    )])
}

/// Grouped-root key for one standalone precommit group plus one final group.
///
/// This is the minimal precommit workload: freeze a small group, then commit a
/// final group against it and open both under one root. Families that already
/// ship both a standalone precommit descriptor at the pre size and a scalar row
/// at the final size can resolve each half but not the combination, so this
/// fills that gap. Both sizes are existing production sizes for the family —
/// no key here introduces a new polynomial size or ring dimension.
fn single_pre_group_batch_keys<Cfg: CommitmentConfig + 'static>(
    preplans: &mut GenerationPreplans,
    pre_group: PolynomialGroupLayout,
    final_group: PolynomialGroupLayout,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let precommitted = planned_profile_without_precommitted_groups::<Cfg>(preplans, pre_group)?;
    Ok(vec![(
        AkitaScheduleLookupKey {
            final_group,
            precommitteds: vec![precommitted],
        },
        vec![honest_fold_policy_of::<Cfg>()],
    )])
}

/// Shipped fp32 precommit-plus-final workload exercised by the extension-field
/// multi-group PCS end-to-end test.
fn fp32_onehot_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    single_pre_group_batch_keys::<fp32::OneHot>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(20, 1),
    )
}

/// Precommit-plus-final row backing the `fp32 × Dense × pre` matrix cell.
fn fp32_dense_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    // The precommit half is 20 rather than 14: `fp32::Dense` has no schedule
    // with at least two folds below 20, so 14 cannot produce the row this
    // group's frozen profile is read from.
    single_pre_group_batch_keys::<fp32::Dense>(
        preplans,
        PolynomialGroupLayout::new(20, 1),
        PolynomialGroupLayout::new(20, 1),
    )
}

/// Precommit-plus-final row backing the `fp64 × Dense × pre` matrix cell.
///
/// `pre_nv` is 16 rather than the usual 14: with a 14- or 15-variable
/// pre-group the fp64 dense prover and the planned schedule disagree on the
/// fold-level-1 witness length, so only the 16-variable pre-group yields a
/// schedule the prover can actually execute.
fn fp64_dense_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    single_pre_group_batch_keys::<fp64::Dense>(
        preplans,
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(20, 1),
    )
}

/// Precommit-plus-final row backing the `fp128 × Dense × sc × pre` matrix cell.
fn fp128_dense_group_batch_keys(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    single_pre_group_batch_keys::<fp128::Dense>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(16, 1),
    )
}

fn recursive_onehot_profile_keys<BaseCfg: CommitmentConfig + 'static>(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let precommitted_group = PolynomialGroupLayout::new(16, 1);
    let precommitted =
        planned_profile_without_precommitted_groups::<BaseCfg>(preplans, precommitted_group)?;
    Ok(vec![(
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![precommitted, precommitted],
        },
        vec![
            honest_fold_policy_of::<BaseCfg>(),
            honest_fold_policy_of::<BaseCfg>(),
        ],
    )])
}

fn heterogeneous_onehot_catalog_key(
    preplans: &mut GenerationPreplans,
) -> Result<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>), AkitaError> {
    let onehot_group = PolynomialGroupLayout::new(14, 1);
    let dense_group = PolynomialGroupLayout::new(15, 2);
    let onehot_policy = honest_fold_policy_of::<fp128::OneHot>();
    let dense_policy = honest_fold_policy_of::<fp128::Dense>();
    let onehot =
        planned_profile_without_precommitted_groups::<fp128::OneHot>(preplans, onehot_group)?;
    let dense = planned_profile_without_precommitted_groups::<fp128::Dense>(preplans, dense_group)?;
    Ok((
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(16, 1),
            precommitteds: vec![onehot, dense],
        },
        vec![onehot_policy, dense_policy],
    ))
}

fn onehot_group_batch_test_keys<BaseCfg: CommitmentConfig + 'static>(
    preplans: &mut GenerationPreplans,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let singleton_pre = planned_profile_without_precommitted_groups::<BaseCfg>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
    )?;
    let pair_pre = planned_profile_without_precommitted_groups::<BaseCfg>(
        preplans,
        PolynomialGroupLayout::new(14, 2),
    )?;
    let policy = honest_fold_policy_of::<BaseCfg>();
    Ok(vec![
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 2),
                precommitteds: vec![singleton_pre],
            },
            vec![policy],
        ),
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 4),
                precommitteds: vec![singleton_pre, singleton_pre],
            },
            vec![policy, policy],
        ),
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 4),
                precommitteds: vec![singleton_pre, singleton_pre, singleton_pre],
            },
            vec![policy, policy, policy],
        ),
        (
            AkitaScheduleLookupKey {
                final_group: PolynomialGroupLayout::new(20, 1),
                precommitteds: vec![pair_pre],
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
            scalar_plan_source: TypeId::of::<$cfg>,
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
            scalar_plan_source: TypeId::of::<$cfg>,
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
        preplanned_scalar: Vec::new(),
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
    preplans: &mut GenerationPreplans,
    output_dir: std::path::PathBuf,
    generator_command: &'static str,
) -> Result<EmitSpec, AkitaError> {
    let policy = (family.policy)();
    let group_batch_keys = (family.group_batch_keys)(preplans)?;
    Ok(EmitSpec {
        module_name: family.module_name,
        const_name: family.const_name,
        family_name: family.module_name,
        schedule_feature: family.schedule_feature,
        policy,
        keys: emitted_scalar_keys(family)?,
        group_batch_keys,
        preplanned_scalar: Vec::new(),
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
        generator_command,
    })
}

fn explicit_precommitted_group<Cfg: CommitmentConfig + 'static>(
    preplans: &mut GenerationPreplans,
    group: PolynomialGroupLayout,
) -> Result<(CommittedGroupProfile, HonestFoldPolicySpec), AkitaError> {
    Ok((
        planned_profile_without_precommitted_groups::<Cfg>(preplans, group)?,
        honest_fold_policy_of::<Cfg>(),
    ))
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
        fp128_dense_group_batch_keys
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
        fp64_dense_group_batch_keys
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
        fp32_dense_group_batch_keys
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_preplans_deduplicate_by_exact_producer_and_layout() {
        let key = PolynomialGroupLayout::new(14, 1);
        let mut preplans = GenerationPreplans::default();

        let first = preplans
            .scalar::<fp128::OneHot>(key)
            .expect("first one-hot plan");
        let repeated = preplans
            .scalar::<fp128::OneHot>(key)
            .expect("repeated one-hot plan");
        assert_eq!(first, repeated);
        assert_eq!(preplans.scalar.len(), 1);

        preplans
            .scalar::<fp128::Dense>(key)
            .expect("same layout under a distinct producer");
        assert_eq!(preplans.scalar.len(), 2);

        let family = ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.module_name == "fp128_onehot")
            .expect("known family");
        let mut spec = wiring_emit_spec(family, std::path::PathBuf::new());
        spec.keys = vec![key];
        preplans.attach_to_spec(family, &mut spec);
        assert_eq!(spec.preplanned_scalar, vec![(key, first)]);
    }
}
