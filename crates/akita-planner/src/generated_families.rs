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

use std::{
    any::TypeId,
    collections::HashMap,
    sync::{LazyLock, Mutex, MutexGuard},
};

use crate::{
    derive_standalone_precommit_profile, find_schedule, runtime_schedule_key_cmp, EmitSpec,
    PlannerPolicy,
};
use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_field::AkitaError;
use akita_types::sis::HonestFoldPolicySpec;
use akita_types::{
    AkitaScheduleInputs, AkitaScheduleLookupKey, CommittedGroupProfile, FoldSchedule,
    PolynomialGroupLayout,
};

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{
    honest_fold_policy_of, policy_of, tensor_verifier, CommitmentConfig, RecursiveCommitmentConfig,
};

type RegenScheduleCacheMap =
    HashMap<(TypeId, AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>), FoldSchedule>;
type RegenScheduleCache = LazyLock<Mutex<RegenScheduleCacheMap>>;
type RegenScheduleCacheGuard = MutexGuard<'static, RegenScheduleCacheMap>;

static REGEN_SCHEDULE_CACHE: RegenScheduleCache = LazyLock::new(|| Mutex::new(HashMap::new()));

fn lock_regen_schedule_cache() -> Result<RegenScheduleCacheGuard, AkitaError> {
    REGEN_SCHEDULE_CACHE
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("schedule regen cache poisoned".to_string()))
}

/// Standalone frozen precommit descriptor arities emitted into generated catalogs.
///
/// This list is intentionally finite. Runtime precommit lookup is catalog-backed,
/// but independent precommit commits do not need a multi-group-root schedule row
/// for every descriptor here.
pub const DEFAULT_STANDALONE_PRECOMMIT_NUM_VARS: &[usize] = &[14, 15, 16];

/// Polynomial counts emitted for standalone frozen precommit descriptors.
pub const DEFAULT_STANDALONE_PRECOMMIT_NUM_POLYNOMIALS: &[usize] = &[1, 2];

const FP128_D64_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::new(16, 2),
    PolynomialGroupLayout::new(17, 4),
    PolynomialGroupLayout::singleton(24),
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(30),
    PolynomialGroupLayout::singleton(32),
    PolynomialGroupLayout::singleton(44),
    PolynomialGroupLayout::singleton(50),
];

const FP128_DENSE_KEYS: &[PolynomialGroupLayout] = FP128_D64_DENSE_KEYS;

const FP128_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(10),
    PolynomialGroupLayout::singleton(12),
    PolynomialGroupLayout::singleton(14),
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

const FP128_D64_ONEHOT_TENSOR_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(26)];

const FP128_D64_ONEHOT_MULTI_CHUNK_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(32)];

const FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(32),
];

const FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(32)];

const FP128_D64_DENSE_MULTI_CHUNK_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(16)];

const FP32_D128_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::new(16, 2),
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::singleton(28),
];

const FP32_D256_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[PolynomialGroupLayout::singleton(14)];

const FP64_D128_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(20),
];

const FP64_D128_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[PolynomialGroupLayout::singleton(28)];

const FP64_D256_ONEHOT_KEYS: &[PolynomialGroupLayout] = &[PolynomialGroupLayout::singleton(28)];

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
    /// Whether this family emits multi-group-root rows in its generated table.
    pub emit_group_batch: bool,
    /// Grouped-root keys enumerated for this generated family.
    #[allow(clippy::type_complexity)]
    pub group_batch_keys:
        fn(
            &GeneratedFamily,
        ) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError>,
    /// `Cfg::runtime_schedule(key)` — strict table-backed runtime resolution.
    /// Used by diagnostic comparisons against the generated table.
    pub table_backed: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    pub policy: fn() -> PlannerPolicy,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    pub fold_challenge_shape_at_level: fn(AkitaScheduleInputs) -> TensorChallengeShape,
    /// Standalone precommit profiles emitted for this family.
    pub precommitted_profiles: fn() -> Result<Vec<CommittedGroupProfile>, AkitaError>,
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
/// Recursive companion catalogs contain only genuine multi-group keys. Their
/// adapter delegates scalar resolution to the base config and therefore must
/// not carry a second, unreachable copy of the ordinary scalar table.
pub fn emitted_scalar_keys(
    family: &GeneratedFamily,
) -> Result<Vec<PolynomialGroupLayout>, AkitaError> {
    if (family.policy)().recursive_setup_planning {
        Ok(Vec::new())
    } else {
        family_keys(family)
    }
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
        Cfg::fold_challenge_shape_at_level,
    )?;
    planned.schedule.validate_structure()?;
    Ok(planned.schedule)
}

/// Pure DP regeneration for `Cfg` — never consults the generated table.
fn regen<Cfg: CommitmentConfig>(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    plan_regen::<Cfg>(&AkitaScheduleLookupKey::single(key), &[])
}

/// Offline regeneration for the catalog-backed default fp128 onehot profile.
fn regen_fp128_onehot(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    type Cfg = fp128::OneHot;
    let policy = policy_of::<Cfg>();
    Ok(find_schedule(
        &AkitaScheduleLookupKey::single(key),
        honest_fold_policy_of::<Cfg>(),
        &[],
        &policy,
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?
    .schedule)
}

/// Offline regeneration for the catalog-backed default fp128 dense profile.
fn regen_fp128_dense(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    type Cfg = fp128::Dense;
    Ok(find_schedule(
        &AkitaScheduleLookupKey::single(key),
        honest_fold_policy_of::<Cfg>(),
        &[],
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )?
    .schedule)
}

/// Pure multi-group DP regeneration for `Cfg` — never consults the generated table.
fn regen_group_batch<Cfg: CommitmentConfig + 'static>(
    key: AkitaScheduleLookupKey,
    precommitted_honest_fold_policies: Vec<HonestFoldPolicySpec>,
) -> Result<FoldSchedule, AkitaError> {
    let cache_key = (
        TypeId::of::<Cfg>(),
        key.clone(),
        precommitted_honest_fold_policies.clone(),
    );
    if let Some(schedule) = lock_regen_schedule_cache()?.get(&cache_key).cloned() {
        return Ok(schedule);
    }

    let schedule = plan_regen::<Cfg>(&key, &precommitted_honest_fold_policies)?;
    lock_regen_schedule_cache()?.insert(cache_key, schedule.clone());
    Ok(schedule)
}

/// Table-backed resolution for `Cfg` — table hit when present, otherwise
/// the DP fallback baked into `runtime_schedule`.
fn table_backed<Cfg: CommitmentConfig>(
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, AkitaError> {
    Cfg::runtime_schedule(AkitaScheduleLookupKey::single(key))
}

fn family_policy<Cfg: CommitmentConfig>() -> PlannerPolicy {
    policy_of::<Cfg>()
}

fn supported_group_batch_key<Cfg: CommitmentConfig + 'static>(
    candidate: (AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>),
) -> Option<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)> {
    regen_group_batch::<Cfg>(candidate.0.clone(), candidate.1.clone())
        .is_ok()
        .then_some(candidate)
}

fn supported_group_batch_keys<Cfg: CommitmentConfig + 'static>(
    candidates: Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(candidates.len().max(1));

    if workers <= 1 || candidates.len() < 2 * workers {
        return Ok(candidates
            .into_iter()
            .filter_map(supported_group_batch_key::<Cfg>)
            .collect());
    }

    let chunk_size = candidates.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = candidates
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .cloned()
                        .filter_map(supported_group_batch_key::<Cfg>)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut keys = Vec::new();
        let mut worker_panicked = false;
        for handle in handles {
            match handle.join() {
                Ok(local) => keys.extend(local),
                Err(_) => worker_panicked = true,
            }
        }
        if worker_panicked {
            return Err(AkitaError::InvalidSetup(
                "group-batch key worker panicked".to_string(),
            ));
        }
        Ok(keys)
    })
}

fn standalone_precommit_profile<Cfg: CommitmentConfig>(
    group: PolynomialGroupLayout,
) -> Result<CommittedGroupProfile, AkitaError> {
    derive_standalone_precommit_profile(
        group,
        &policy_of::<Cfg>(),
        honest_fold_policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
        Cfg::fold_challenge_shape_at_level,
    )
}

fn push_unique_profile(profiles: &mut Vec<CommittedGroupProfile>, profile: CommittedGroupProfile) {
    if !profiles.contains(&profile) {
        profiles.push(profile);
    }
}

fn precommitted_profiles<Cfg: CommitmentConfig + 'static>(
) -> Result<Vec<CommittedGroupProfile>, AkitaError> {
    let mut profiles = Vec::new();
    for &num_vars in DEFAULT_STANDALONE_PRECOMMIT_NUM_VARS {
        for &num_polys in DEFAULT_STANDALONE_PRECOMMIT_NUM_POLYNOMIALS {
            let group = PolynomialGroupLayout::new(num_vars, num_polys);
            if let Ok(profile) = standalone_precommit_profile::<Cfg>(group) {
                push_unique_profile(&mut profiles, profile);
            }
        }
    }

    if std::any::TypeId::of::<Cfg>() == std::any::TypeId::of::<fp128::D64OneHot>()
        || std::any::TypeId::of::<Cfg>() == std::any::TypeId::of::<fp128::D64OneHotMultiChunk>()
    {
        push_unique_profile(
            &mut profiles,
            standalone_precommit_profile::<Cfg>(PolynomialGroupLayout::new(16, 1))?,
        );
        push_unique_profile(
            &mut profiles,
            standalone_precommit_profile::<Cfg>(PolynomialGroupLayout::new(15, 2))?,
        );
    }
    if std::any::TypeId::of::<Cfg>() == std::any::TypeId::of::<fp128::D64OneHot>() {
        push_unique_profile(
            &mut profiles,
            standalone_precommit_profile::<Cfg>(PolynomialGroupLayout::new(20, 1))?,
        );
    }
    if std::any::TypeId::of::<Cfg>() == std::any::TypeId::of::<fp128::D64Dense>() {
        push_unique_profile(
            &mut profiles,
            standalone_precommit_profile::<Cfg>(PolynomialGroupLayout::new(15, 2))?,
        );
    }

    profiles.sort_by_key(CommittedGroupProfile::canonical_descriptor_bytes);
    Ok(profiles)
}

fn group_batch_keys<Cfg: CommitmentConfig + 'static>(
    family: &GeneratedFamily,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let mut direct = direct_profile_group_batch_keys_for_cfg::<Cfg>()?;
    if !family.emit_group_batch {
        direct.sort_by(|left, right| runtime_schedule_key_cmp(&left.0, &right.0));
        return Ok(direct);
    }
    if Cfg::decomposition().log_commit_bound != 1 {
        direct.sort_by(|left, right| runtime_schedule_key_cmp(&left.0, &right.0));
        return Ok(direct);
    }

    let mut keys = supported_group_batch_keys::<Cfg>(direct)?;
    keys.sort_by(|left, right| runtime_schedule_key_cmp(&left.0, &right.0));
    Ok(keys)
}

fn direct_profile_group_batch_keys_for_cfg<Cfg: CommitmentConfig + 'static>(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    if std::any::TypeId::of::<Cfg>() == std::any::TypeId::of::<fp128::D64OneHot>() {
        let mut keys = recursive_d64_onehot_profile_keys::<fp128::D64OneHot>()?;
        keys.push(heterogeneous_d64_onehot_catalog_key()?);
        keys.extend(onehot_group_batch_test_keys::<fp128::D64OneHot>()?);
        return Ok(keys);
    }
    if std::any::TypeId::of::<Cfg>() == std::any::TypeId::of::<fp128::D64OneHotMultiChunk>() {
        return recursive_d64_onehot_profile_keys::<fp128::D64OneHotMultiChunk>();
    }
    Ok(Vec::new())
}

fn recursive_profile_group_batch_keys<Cfg: CommitmentConfig + 'static>(
    _family: &GeneratedFamily,
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    recursive_profile_group_batch_keys_for_recursive_cfg::<Cfg>()
}

fn recursive_profile_group_batch_keys_for_recursive_cfg<Cfg: CommitmentConfig + 'static>(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    if std::any::TypeId::of::<Cfg>()
        == std::any::TypeId::of::<RecursiveCommitmentConfig<fp128::D64OneHot>>()
    {
        return recursive_d64_onehot_profile_keys::<fp128::D64OneHot>();
    }
    if std::any::TypeId::of::<Cfg>()
        == std::any::TypeId::of::<RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>>()
    {
        return recursive_d64_onehot_profile_keys::<fp128::D64OneHotMultiChunk>();
    }
    Ok(Vec::new())
}

fn recursive_d64_onehot_profile_keys<BaseCfg: CommitmentConfig + 'static>(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let precommitted_group = PolynomialGroupLayout::new(16, 1);
    let precommitted = standalone_precommit_profile::<BaseCfg>(precommitted_group)?;
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

fn heterogeneous_d64_onehot_catalog_key(
) -> Result<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>), AkitaError> {
    let onehot_group = PolynomialGroupLayout::new(14, 1);
    let dense_group = PolynomialGroupLayout::new(15, 2);
    let onehot_policy = honest_fold_policy_of::<fp128::D64OneHot>();
    let dense_policy = honest_fold_policy_of::<fp128::D64Dense>();
    let onehot = standalone_precommit_profile::<fp128::D64OneHot>(onehot_group)?;
    let dense = standalone_precommit_profile::<fp128::D64Dense>(dense_group)?;
    Ok((
        AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(16, 1),
            precommitteds: vec![onehot, dense],
        },
        vec![onehot_policy, dense_policy],
    ))
}

fn onehot_group_batch_test_keys<BaseCfg: CommitmentConfig + 'static>(
) -> Result<Vec<(AkitaScheduleLookupKey, Vec<HonestFoldPolicySpec>)>, AkitaError> {
    let singleton_pre = standalone_precommit_profile::<BaseCfg>(PolynomialGroupLayout::new(14, 1))?;
    let pair_pre = standalone_precommit_profile::<BaseCfg>(PolynomialGroupLayout::new(14, 2))?;
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

/// Selected multi-group recursive keys for setup-prefix capacity work.
///
/// Returns the bounded supported set: generated-catalog multi-group rows under
/// capacity, plus the explicit recursive profiling key(s). This is intentionally
/// not a dense `1..=max_nv` grid. Setup envelope inflation and exact prefix-slot
/// materialization both walk this set; other recursive shapes remain planner-
/// constructible but are admitted only when their slots already fit the
/// materialized artifact (`ensure_prover_schedule_fits_setup` / missing-slot reject).
///
/// Does not run the planner; callers resolve each selected key.
pub fn recursive_group_batch_candidates_for_capacity<Cfg: CommitmentConfig>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
) -> Result<Vec<AkitaScheduleLookupKey>, AkitaError> {
    if !Cfg::recursive_setup_planning()
        || Cfg::decomposition().log_commit_bound != 1
        || max_num_batched_polys == 0
    {
        return Ok(Vec::new());
    }

    let mut keys = Vec::new();
    if let Some(catalog) = Cfg::schedule_catalog() {
        for entry in catalog.entries {
            if entry.root.precommitted_groups.is_empty() {
                continue;
            }
            let candidate = AkitaScheduleLookupKey {
                final_group: entry.root.final_group.layout,
                precommitteds: entry
                    .root
                    .precommitted_groups
                    .iter()
                    .map(|group| group.descriptor)
                    .collect(),
            };
            if candidate.fits_setup_capacity(max_num_vars, max_num_batched_polys)? {
                push_unique_schedule_key(&mut keys, candidate);
            }
        }
    }

    // Explicit profiling keys stay selected even when the recursive catalog
    // feature is off or the table has not been regenerated yet. The plain
    // recursive adapter and its multi-chunk (distributed-prover) companion share
    // the same profiling key shape; they differ only in the chunked witness
    // layout the policy prices.
    for (candidate, _) in recursive_profile_group_batch_keys_for_recursive_cfg::<Cfg>()? {
        if candidate.fits_setup_capacity(max_num_vars, max_num_batched_polys)? {
            push_unique_schedule_key(&mut keys, candidate);
        }
    }

    keys.sort_by(runtime_schedule_key_cmp);
    Ok(keys)
}

fn push_unique_schedule_key(
    keys: &mut Vec<AkitaScheduleLookupKey>,
    candidate: AkitaScheduleLookupKey,
) {
    // Full-key equality: same group shapes with different frozen precommit
    // metadata (semantic bases / n_a / n_b) stay distinct.
    if !keys.contains(&candidate) {
        keys.push(candidate);
    }
}

macro_rules! family_row {
    (group_batch, $module:literal, $const:literal, $feat:literal, $keys:expr, $cfg:ty) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            scalar_keys: $keys,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            emit_group_batch: true,
            group_batch_keys: group_batch_keys::<$cfg>,
            table_backed: table_backed::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            fold_challenge_shape_at_level:
                <$cfg as CommitmentConfig>::fold_challenge_shape_at_level,
            precommitted_profiles: precommitted_profiles::<$cfg>,
            explicit_precommitted_group: explicit_precommitted_group::<$cfg>,
        }
    };
    // Recursion adapter families: like `group_batch`, but grouped keys come from
    // the fixed recursive profiling shape rather than the generic per-`Cfg` grid.
    (recursive, $module:literal, $const:literal, $feat:literal, $keys:expr, $cfg:ty, $base_cfg:ty) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            scalar_keys: $keys,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            emit_group_batch: true,
            group_batch_keys: recursive_profile_group_batch_keys::<$cfg>,
            table_backed: table_backed::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            fold_challenge_shape_at_level:
                <$cfg as CommitmentConfig>::fold_challenge_shape_at_level,
            precommitted_profiles: precommitted_profiles::<$cfg>,
            explicit_precommitted_group: explicit_precommitted_group::<$base_cfg>,
        }
    };
    ($module:literal, $const:literal, $feat:literal, $keys:expr, $cfg:ty) => {
        GeneratedFamily {
            module_name: $module,
            const_name: $const,
            schedule_feature: $feat,
            scalar_keys: $keys,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            emit_group_batch: false,
            group_batch_keys: group_batch_keys::<$cfg>,
            table_backed: table_backed::<$cfg>,
            policy: family_policy::<$cfg>,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            fold_challenge_shape_at_level:
                <$cfg as CommitmentConfig>::fold_challenge_shape_at_level,
            precommitted_profiles: precommitted_profiles::<$cfg>,
            explicit_precommitted_group: explicit_precommitted_group::<$cfg>,
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
        precommitted_profiles: Vec::new(),
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
    let group_batch_keys = (family.group_batch_keys)(family)?;
    let mut precommitted_profiles = (family.precommitted_profiles)()?;
    precommitted_profiles.sort_by_key(CommittedGroupProfile::canonical_descriptor_bytes);
    Ok(EmitSpec {
        module_name: family.module_name,
        const_name: family.const_name,
        family_name: family.module_name,
        schedule_feature: family.schedule_feature,
        policy,
        keys: emitted_scalar_keys(family)?,
        group_batch_keys,
        emit_group_batch: family.emit_group_batch,
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
        fold_challenge_shape_at_level: family.fold_challenge_shape_at_level,
        precommitted_profiles,
        generator_command,
    })
}

fn explicit_precommitted_group<Cfg: CommitmentConfig + 'static>(
    group: PolynomialGroupLayout,
) -> Result<(CommittedGroupProfile, HonestFoldPolicySpec), AkitaError> {
    Ok((
        standalone_precommit_profile::<Cfg>(group)?,
        honest_fold_policy_of::<Cfg>(),
    ))
}

/// Every `Cfg` that has a generated schedule table.
///
/// Adding a new preset with a generated table requires adding a row
/// here; both the table emitter and the drift-guard test pick it up
/// automatically.
pub const ALL_GENERATED_FAMILIES: &[GeneratedFamily] = &[
    GeneratedFamily {
        module_name: "fp128_onehot",
        const_name: "FP128_ONEHOT_SCHEDULES",
        schedule_feature: "fp128-onehot",
        scalar_keys: FP128_ONEHOT_KEYS,
        regen: regen_fp128_onehot,
        regen_group_batch: regen_group_batch::<fp128::OneHot>,
        emit_group_batch: false,
        group_batch_keys: group_batch_keys::<fp128::OneHot>,
        table_backed: table_backed::<fp128::OneHot>,
        policy: family_policy::<fp128::OneHot>,
        ring_challenge_config: <fp128::OneHot as CommitmentConfig>::ring_challenge_config,
        fold_challenge_shape_at_level:
            <fp128::OneHot as CommitmentConfig>::fold_challenge_shape_at_level,
        precommitted_profiles: precommitted_profiles::<fp128::OneHot>,
        explicit_precommitted_group: explicit_precommitted_group::<fp128::OneHot>,
    },
    family_row!(
        recursive,
        "fp128_d64_onehot_recursive",
        "FP128_D64_ONEHOT_RECURSIVE_SCHEDULES",
        "fp128-d64-onehot-recursive",
        &[],
        RecursiveCommitmentConfig<fp128::D64OneHot>,
        fp128::D64OneHot
    ),
    // Recursive setup offloading combined with the 8-chunk (production
    // distributed-prover) witness layout. `D64OneHotMultiChunk` is the W8R2
    // preset (8 chunks x 2 leading levels); shares the recursive profiling key.
    family_row!(
        recursive,
        "fp128_d64_onehot_recursive_multi_chunk_w8r2",
        "FP128_D64_ONEHOT_RECURSIVE_MULTI_CHUNK_W8R2_SCHEDULES",
        "fp128-d64-onehot-recursive-multi-chunk-w8r2",
        &[],
        RecursiveCommitmentConfig<fp128::D64OneHotMultiChunk>,
        fp128::D64OneHotMultiChunk
    ),
    GeneratedFamily {
        module_name: "fp128_dense",
        const_name: "FP128_DENSE_SCHEDULES",
        schedule_feature: "fp128-dense",
        scalar_keys: FP128_DENSE_KEYS,
        regen: regen_fp128_dense,
        regen_group_batch: regen_group_batch::<fp128::Dense>,
        emit_group_batch: false,
        group_batch_keys: group_batch_keys::<fp128::Dense>,
        table_backed: table_backed::<fp128::Dense>,
        policy: family_policy::<fp128::Dense>,
        ring_challenge_config: <fp128::Dense as CommitmentConfig>::ring_challenge_config,
        fold_challenge_shape_at_level:
            <fp128::Dense as CommitmentConfig>::fold_challenge_shape_at_level,
        precommitted_profiles: precommitted_profiles::<fp128::Dense>,
        explicit_precommitted_group: explicit_precommitted_group::<fp128::Dense>,
    },
    family_row!(
        group_batch,
        "fp128_d64_onehot_tensor",
        "FP128_D64_ONEHOT_TENSOR_SCHEDULES",
        "fp128-d64-onehot-tensor",
        FP128_D64_ONEHOT_TENSOR_KEYS,
        tensor_verifier::fp128::D64OneHotTensor
    ),
    // Multi-chunk (distributed-prover) companions of the D64 families. Same
    // `(num_vars, num_polynomials)` keys as their siblings; schedules differ
    // because the policy prices the chunked witness layout.
    family_row!(
        group_batch,
        "fp128_d64_onehot_multi_chunk",
        "FP128_D64_ONEHOT_MULTI_CHUNK_SCHEDULES",
        "fp128-d64-onehot-multi-chunk",
        FP128_D64_ONEHOT_MULTI_CHUNK_KEYS,
        fp128::D64OneHotMultiChunk
    ),
    family_row!(
        group_batch,
        "fp128_d64_onehot_multi_chunk_w2r2",
        "FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_SCHEDULES",
        "fp128-d64-onehot-multi-chunk-w2r2",
        FP128_D64_ONEHOT_MULTI_CHUNK_W2R2_KEYS,
        fp128::D64OneHotMultiChunkW2R2
    ),
    family_row!(
        group_batch,
        "fp128_d64_onehot_multi_chunk_w4r2",
        "FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_SCHEDULES",
        "fp128-d64-onehot-multi-chunk-w4r2",
        FP128_D64_ONEHOT_MULTI_CHUNK_W4R2_KEYS,
        fp128::D64OneHotMultiChunkW4R2
    ),
    family_row!(
        "fp128_d64_dense_multi_chunk",
        "FP128_D64_DENSE_MULTI_CHUNK_SCHEDULES",
        "fp128-d64-dense-multi-chunk",
        FP128_D64_DENSE_MULTI_CHUNK_KEYS,
        fp128::D64DenseMultiChunk
    ),
    family_row!(
        "fp64_d128_dense",
        "FP64_D128_DENSE_SCHEDULES",
        "fp64-d128-dense",
        FP64_D128_DENSE_KEYS,
        fp64::D128Dense
    ),
    family_row!(
        "fp64_d128_onehot",
        "FP64_D128_ONEHOT_SCHEDULES",
        "fp64-d128-onehot",
        FP64_D128_ONEHOT_KEYS,
        fp64::D128OneHot
    ),
    family_row!(
        "fp64_d256_onehot",
        "FP64_D256_ONEHOT_SCHEDULES",
        "fp64-d256-onehot",
        FP64_D256_ONEHOT_KEYS,
        fp64::D256OneHot
    ),
    family_row!(
        "fp32_d128_onehot",
        "FP32_D128_ONEHOT_SCHEDULES",
        "fp32-d128-onehot",
        FP32_D128_ONEHOT_KEYS,
        fp32::D128OneHot
    ),
    family_row!(
        "fp32_d256_onehot",
        "FP32_D256_ONEHOT_SCHEDULES",
        "fp32-d256-onehot",
        FP32_D256_ONEHOT_KEYS,
        fp32::D256OneHot
    ),
];
