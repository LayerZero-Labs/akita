//! Shared metadata describing every `Cfg` family that ships a schedule artifact.
//!
//! Both the `gen_schedule_artifacts` binary (the offline artifact emitter) and
//! the drift-guard test consume [`ALL_GENERATED_FAMILIES`] so the two
//! cannot drift apart: a missing `Cfg` here is missing in both the emitted
//! artifact and the regression guard.
//!
//! This list is the one place a preset `Cfg` type is bound to its regen
//! hook and artifact policy. It is behind the `catalog-gen` feature because
//! that offline path is allowed to name `akita-config` presets. Normal
//! runtime callers consume validated artifacts through `akita-schedules`.

use std::any::TypeId;
use std::sync::{Arc, Mutex, OnceLock};

pub use crate::emit::{GroupedGenerationRequest, PrecommittedProducer};
use crate::{find_schedule, EmitSpec, PlannerPolicy};
use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::sis::{CommittedSourceContract, HonestFoldPolicySpec};
use akita_types::{
    AkitaScheduleLookupKey, FoldSchedule, GroupCommitPhaseParams, PolynomialGroupLayout,
};

use akita_config::proof_optimized::{fp128, fp32, fp64};
use akita_config::{honest_fold_policy_of, policy_of, CommitmentConfig, RecursiveCommitmentConfig};

struct ScalarPreplan {
    source: TypeId,
    key: PolynomialGroupLayout,
    result: OnceLock<Result<FoldSchedule, AkitaError>>,
}

/// Exact scalar schedules already needed while preparing one generator run.
///
/// Entries are keyed by the producer configuration rather than the consuming
/// family. Recursive families can therefore use a base configuration's frozen
/// profile without aliasing their own scalar schedules. The session is dropped
/// before parallel rendering and is never persisted or iterated for output.
#[derive(Default)]
pub struct GenerationPreplans {
    scalar: Mutex<Vec<Arc<ScalarPreplan>>>,
}

impl GenerationPreplans {
    fn scalar<Cfg: CommitmentConfig + 'static>(
        &self,
        key: PolynomialGroupLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        let source = TypeId::of::<Cfg>();
        let mut scalar = self
            .scalar
            .lock()
            .map_err(|_| AkitaError::InvalidInput("scalar preplan cache poisoned".into()))?;
        let preplanned = scalar
            .iter()
            .find(|preplanned| preplanned.source == source && preplanned.key == key)
            .cloned()
            .unwrap_or_else(|| {
                let preplanned = Arc::new(ScalarPreplan {
                    source,
                    key,
                    result: OnceLock::new(),
                });
                scalar.push(Arc::clone(&preplanned));
                preplanned
            });
        drop(scalar);
        preplanned.result.get_or_init(|| regen::<Cfg>(key)).clone()
    }

    /// Return an exact scalar result previously planned for `family`.
    pub fn scalar_for_family(
        &self,
        family: &GeneratedFamily,
        key: PolynomialGroupLayout,
    ) -> Option<FoldSchedule> {
        let source = (family.scalar_plan_source)();
        self.scalar
            .lock()
            .ok()?
            .iter()
            .find(|preplanned| preplanned.source == source && preplanned.key == key)
            .and_then(|preplanned| preplanned.result.get())
            .and_then(|result| result.as_ref().ok())
            .cloned()
    }

    /// Copy exact producer results into a completed spec before rendering.
    pub fn attach_to_spec(&self, family: &GeneratedFamily, spec: &mut EmitSpec) {
        spec.preplanned_scalar = spec
            .keys
            .iter()
            .filter_map(|key| {
                self.scalar_for_family(family, *key)
                    .map(|schedule| (*key, schedule))
            })
            .collect();
    }
}

type GroupedGenerationRequests = Vec<GroupedGenerationRequest>;
type GroupedRequestGenerator =
    fn(&GenerationPreplans) -> Result<GroupedGenerationRequests, AkitaError>;
type ExplicitPrecommittedGroupGenerator =
    fn(&GenerationPreplans, PolynomialGroupLayout) -> Result<PrecommittedProducer, AkitaError>;

macro_rules! onehot_keys {
    ($(($num_vars:expr, $num_polynomials:expr)),* $(,)?) => {
        &[
            $(
                PolynomialGroupLayout::new($num_vars, $num_polynomials),
            )*
        ]
    };
}

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
];

const FP128_ONEHOT_KEYS: &[PolynomialGroupLayout] = onehot_keys![
    (12, 1),
    (14, 1),
    (14, 2),
    (15, 1),
    (15, 4),
    (16, 1),
    (16, 2),
    (18, 1),
    (20, 1),
    (20, 2),
    (20, 4),
    (28, 1),
    (30, 1),
    (30, 4),
    (32, 1),
    (32, 4),
    (36, 1),
    (40, 1),
    (44, 1),
    (50, 1),
];

const FP128_ONEHOT_MULTI_CHUNK_KEYS: &[PolynomialGroupLayout] = onehot_keys![(16, 1), (32, 1)];

const FP128_ONEHOT_RECURSIVE_KEYS: &[PolynomialGroupLayout] = onehot_keys![(36, 1)];

const FP128_ONEHOT_MULTI_CHUNK_W2R2_KEYS: &[PolynomialGroupLayout] = onehot_keys![(14, 1), (32, 1)];

const FP128_ONEHOT_MULTI_CHUNK_W4R2_KEYS: &[PolynomialGroupLayout] = onehot_keys![(32, 1)];

const FP128_DENSE_MULTI_CHUNK_KEYS: &[PolynomialGroupLayout] =
    &[PolynomialGroupLayout::singleton(16)];

/// Bounded dense keys.
///
/// 14 is the producer for the bounded precommit descriptor stored in the
/// `fp128_onehot` grouped catalog (see [`bounded_dense_onehot_catalog_key`]); 24
/// and 26 are the sizes where the bound's setup and next-witness savings are
/// measured against the matching `fp128_dense` rows.
const FP128_DENSE_BOUNDED_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    PolynomialGroupLayout::singleton(24),
    PolynomialGroupLayout::singleton(26),
];

const FP32_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::singleton(26),
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(30),
];

const FP32_ONEHOT_KEYS: &[PolynomialGroupLayout] = onehot_keys![
    (14, 1),
    (16, 1),
    (16, 2),
    (20, 1),
    (28, 1),
    (30, 1),
    (34, 1)
];

const FP64_DENSE_KEYS: &[PolynomialGroupLayout] = &[
    PolynomialGroupLayout::singleton(14),
    // Produces the frozen profile for the precommit half of
    // `fp64_dense_grouped_requests`, which needs 16: at 14 or 15 the prover and
    // the planned schedule disagree on the fold-level-1 witness length.
    PolynomialGroupLayout::singleton(16),
    PolynomialGroupLayout::singleton(20),
    PolynomialGroupLayout::singleton(26),
    PolynomialGroupLayout::singleton(28),
    PolynomialGroupLayout::singleton(29),
    PolynomialGroupLayout::singleton(30),
];

const FP64_ONEHOT_KEYS: &[PolynomialGroupLayout] = onehot_keys![(28, 1), (30, 1), (34, 1), (35, 1)];

/// One generated schedule-artifact family.
///
/// Function-pointer fields (instead of generic `Fn` closures) keep the
/// list `const`-constructible and `'static`.
#[derive(Clone, Copy)]
pub struct GeneratedFamily {
    /// Artifact family name and on-disk basename (without `.aks`).
    family_name_fn: fn() -> &'static str,
    /// Scalar opening keys emitted for this family.
    pub scalar_keys: &'static [PolynomialGroupLayout],
    /// Exact producer type used to distinguish scalar preplans.
    scalar_plan_source: fn() -> TypeId,
    /// Pure DP regeneration that ignores any checked-in artifact
    /// (`find_schedule(&single_key, &[], &policy_of::<Cfg>(), …)`).
    pub regen: fn(PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError>,
    /// Pure multi-group DP regeneration that ignores any checked-in artifact.
    pub regen_group_batch: fn(GroupedGenerationRequest) -> Result<FoldSchedule, AkitaError>,
    /// Grouped-root keys enumerated for this generated family.
    pub grouped_requests: GroupedRequestGenerator,
    pub policy: fn() -> PlannerPolicy,
    /// The family config's declared producer contract (class plus bound).
    pub source_contract: fn() -> Result<CommittedSourceContract, AkitaError>,
    pub ring_challenge_config: fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    /// Build one caller requested canonical precommit producer record.
    pub explicit_precommitted_group: ExplicitPrecommittedGroupGenerator,
}

impl GeneratedFamily {
    /// Config-owned artifact family name and on-disk basename.
    #[must_use]
    pub fn family_name(self) -> &'static str {
        (self.family_name_fn)()
    }
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
    keys.sort_by_cached_key(|key| AkitaScheduleLookupKey::single(*key).canonical_order_key());
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

/// Pure DP regeneration for `Cfg` — never consults the checked-in artifact.
fn regen<Cfg: CommitmentConfig>(key: PolynomialGroupLayout) -> Result<FoldSchedule, AkitaError> {
    plan_regen::<Cfg>(&AkitaScheduleLookupKey::single(key), &[])
}

/// Frozen profile a group commits with when it has no precommitted groups.
///
/// Generation cannot read the catalog it is producing, so this plans the row
/// instead of selecting it. `TrustedScheduleCatalog::resolve_key` is the runtime counterpart, and
/// `every_grouped_artifact_precommit_has_a_shipped_scalar_producer` asserts the
/// two agree on every shipped descriptor, including recursive adapter/base-family
/// mappings.
fn planned_profile_without_precommitted_groups<Cfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
    group: PolynomialGroupLayout,
) -> Result<GroupCommitPhaseParams, AkitaError> {
    let schedule = preplans.scalar::<Cfg>(group)?;
    GroupCommitPhaseParams::try_from_params(group, &schedule.root.params)
}

/// Pure multi-group DP regeneration for `Cfg` — never consults the checked-in artifact.
fn regen_group_batch<Cfg: CommitmentConfig + 'static>(
    request: GroupedGenerationRequest,
) -> Result<FoldSchedule, AkitaError> {
    // Planning consumes the offline sizing projection; the record owns it beside
    // the descriptor so the two can never drift apart by index.
    let policies = request.fold_policies();
    plan_regen::<Cfg>(&request.key(), &policies)
}

fn family_policy<Cfg: CommitmentConfig>() -> PlannerPolicy {
    policy_of::<Cfg>()
}

fn sorted_grouped_requests(mut requests: GroupedGenerationRequests) -> GroupedGenerationRequests {
    requests.sort_by_cached_key(|request| request.key().canonical_order_key());
    requests
}

fn no_grouped_requests(
    _preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    Ok(Vec::new())
}

fn fp128_onehot_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    let mut keys = recursive_onehot_profile_keys::<fp128::OneHot>(preplans)?;
    keys.push(heterogeneous_onehot_catalog_key(preplans)?);
    keys.push(bounded_dense_onehot_catalog_key(preplans)?);
    keys.extend(onehot_group_batch_test_keys::<fp128::OneHot>(preplans)?);
    // Single-poly pre + single-poly final: the `fp128 × OneHot × pre` matrix
    // cell. Every other combined OneHot row is heterogeneous or multi-poly.
    keys.extend(single_pre_grouped_requests::<fp128::OneHot>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(16, 1),
    )?);
    keys.extend(single_pre_grouped_requests::<fp128::OneHot>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(20, 1),
    )?);
    Ok(sorted_grouped_requests(keys))
}

fn fp128_onehot_multichunk_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    Ok(sorted_grouped_requests(
        recursive_onehot_chunked_profile_keys::<fp128::OneHotMultiChunk>(preplans)?,
    ))
}

fn fp128_onehot_multichunk_w2r2_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    type Cfg = fp128::OneHotMultiChunkW2R2;
    let group = PolynomialGroupLayout::new(14, 1);
    let precommitted = planned_profile_without_precommitted_groups::<Cfg>(preplans, group)?;
    Ok(vec![GroupedGenerationRequest::new(
        group,
        vec![PrecommittedProducer::from_config::<Cfg>(precommitted)?],
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
fn single_pre_grouped_requests<Cfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
    pre_group: PolynomialGroupLayout,
    final_group: PolynomialGroupLayout,
) -> Result<GroupedGenerationRequests, AkitaError> {
    let precommitted = planned_profile_without_precommitted_groups::<Cfg>(preplans, pre_group)?;
    Ok(vec![GroupedGenerationRequest::new(
        final_group,
        vec![PrecommittedProducer::from_config::<Cfg>(precommitted)?],
    )])
}

/// Shipped fp32 precommit-plus-final workload exercised by the extension-field
/// multi-group PCS end-to-end test.
fn fp32_onehot_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    single_pre_grouped_requests::<fp32::OneHot>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(20, 1),
    )
}

/// Precommit-plus-final row backing the `fp32 × Dense × pre` matrix cell.
fn fp32_dense_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    // The precommit half is 20 rather than 14 because the shipped fp32 dense
    // catalog begins at 20. The group's frozen profile must come from a
    // artifact row in that catalog.
    single_pre_grouped_requests::<fp32::Dense>(
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
fn fp64_dense_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    single_pre_grouped_requests::<fp64::Dense>(
        preplans,
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(20, 1),
    )
}

/// Precommit-plus-final row backing the `fp128 × Dense × sc × pre` matrix cell.
fn fp128_dense_grouped_requests(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    single_pre_grouped_requests::<fp128::Dense>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
        PolynomialGroupLayout::new(16, 1),
    )
}

fn recursive_onehot_profile_keys<BaseCfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    // Keep the historical (32, 2) profiling key for e2e catalog tests, and ship
    // (34, 2) as the larger profile-bench fixture.
    recursive_onehot_profile_keys_for_finals::<BaseCfg>(preplans, &[32, 34])
}

fn recursive_onehot_chunked_profile_keys<BaseCfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    recursive_onehot_profile_keys_for_finals::<BaseCfg>(preplans, &[32, 34])
}

fn recursive_onehot_profile_keys_for_finals<BaseCfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
    final_num_vars: &[usize],
) -> Result<GroupedGenerationRequests, AkitaError> {
    let precommitted_group = PolynomialGroupLayout::new(16, 1);
    let precommitted =
        planned_profile_without_precommitted_groups::<BaseCfg>(preplans, precommitted_group)?;
    let producers = vec![
        PrecommittedProducer::from_config::<BaseCfg>(precommitted)?,
        PrecommittedProducer::from_config::<BaseCfg>(precommitted)?,
    ];
    Ok(final_num_vars
        .iter()
        .map(|&num_vars| {
            GroupedGenerationRequest::new(
                PolynomialGroupLayout::new(num_vars, 2),
                producers.clone(),
            )
        })
        .collect())
}

fn heterogeneous_onehot_catalog_key(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequest, AkitaError> {
    let onehot_group = PolynomialGroupLayout::new(14, 1);
    let dense_group = PolynomialGroupLayout::new(15, 2);
    let onehot =
        planned_profile_without_precommitted_groups::<fp128::OneHot>(preplans, onehot_group)?;
    let dense = planned_profile_without_precommitted_groups::<fp128::Dense>(preplans, dense_group)?;
    Ok(GroupedGenerationRequest::new(
        PolynomialGroupLayout::new(16, 1),
        vec![
            PrecommittedProducer::from_config::<fp128::OneHot>(onehot)?,
            PrecommittedProducer::from_config::<fp128::Dense>(dense)?,
        ],
    ))
}

/// Grouped-root key pairing a **bounded** dense precommit with a one-hot final
/// group.
///
/// This is the mixed-bound cell: the precommitted group is frozen by
/// `fp128::DenseBounded` (`log_commit_bound = 65` inside the 128-bit field) while the
/// root is planned under `fp128::OneHot` (`log_commit_bound = 1`). It exercises
/// the fact that a precommitted group carries its own committed-source bound in
/// its frozen `inner_commit_matrix` and does not have to agree with the planning
/// config's bound — only the shared full-width opening geometry has to line up.
fn bounded_dense_onehot_catalog_key(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequest, AkitaError> {
    let bounded_dense_group = PolynomialGroupLayout::new(14, 1);
    let bounded_dense = planned_profile_without_precommitted_groups::<fp128::DenseBounded>(
        preplans,
        bounded_dense_group,
    )?;
    Ok(GroupedGenerationRequest::new(
        PolynomialGroupLayout::new(16, 1),
        vec![PrecommittedProducer::from_config::<fp128::DenseBounded>(
            bounded_dense,
        )?],
    ))
}

fn onehot_group_batch_test_keys<BaseCfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
) -> Result<GroupedGenerationRequests, AkitaError> {
    let singleton_pre = planned_profile_without_precommitted_groups::<BaseCfg>(
        preplans,
        PolynomialGroupLayout::new(14, 1),
    )?;
    let pair_pre = planned_profile_without_precommitted_groups::<BaseCfg>(
        preplans,
        PolynomialGroupLayout::new(14, 2),
    )?;
    let singleton = PrecommittedProducer::from_config::<BaseCfg>(singleton_pre)?;
    let pair = PrecommittedProducer::from_config::<BaseCfg>(pair_pre)?;
    Ok(vec![
        GroupedGenerationRequest::new(PolynomialGroupLayout::new(20, 2), vec![singleton]),
        GroupedGenerationRequest::new(
            PolynomialGroupLayout::new(20, 4),
            vec![singleton, singleton],
        ),
        GroupedGenerationRequest::new(
            PolynomialGroupLayout::new(20, 4),
            vec![singleton, singleton, singleton],
        ),
        GroupedGenerationRequest::new(PolynomialGroupLayout::new(20, 1), vec![pair]),
    ])
}

macro_rules! family_row {
    // Recursion adapter families: like `group_batch`, but grouped keys come from
    // the fixed recursive profiling shape rather than the generic per-`Cfg` grid.
    (recursive, $keys:expr, $cfg:ty, $base_cfg:ty, $group_keys:expr) => {
        GeneratedFamily {
            family_name_fn: <$cfg as CommitmentConfig>::schedule_family_name,
            scalar_keys: $keys,
            scalar_plan_source: TypeId::of::<$cfg>,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            grouped_requests: $group_keys,
            policy: family_policy::<$cfg>,
            source_contract: <$cfg as CommitmentConfig>::committed_source_contract,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            explicit_precommitted_group: explicit_precommitted_group::<$base_cfg>,
        }
    };
    ($keys:expr, $cfg:ty, $group_keys:expr) => {
        GeneratedFamily {
            family_name_fn: <$cfg as CommitmentConfig>::schedule_family_name,
            scalar_keys: $keys,
            scalar_plan_source: TypeId::of::<$cfg>,
            regen: regen::<$cfg>,
            regen_group_batch: regen_group_batch::<$cfg>,
            grouped_requests: $group_keys,
            policy: family_policy::<$cfg>,
            source_contract: <$cfg as CommitmentConfig>::committed_source_contract,
            ring_challenge_config: <$cfg as CommitmentConfig>::ring_challenge_config,
            explicit_precommitted_group: explicit_precommitted_group::<$cfg>,
        }
    };
}

/// Minimal [`EmitSpec`] for a caller-selected artifact sweep.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when the family declares a producer
/// contract it cannot honour.
pub fn empty_emit_spec(
    family: &GeneratedFamily,
    output_dir: std::path::PathBuf,
) -> Result<EmitSpec, AkitaError> {
    Ok(EmitSpec {
        family_name: family.family_name(),
        policy: (family.policy)(),
        source_contract: (family.source_contract)()?,
        keys: Vec::new(),
        grouped_requests: Vec::new(),
        preplanned_scalar: Vec::new(),
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
    })
}

/// Adapt one [`GeneratedFamily`] into an [`EmitSpec`] for the planner emitter.
pub fn emit_spec_for_family(
    family: &GeneratedFamily,
    preplans: &GenerationPreplans,
    output_dir: std::path::PathBuf,
) -> Result<EmitSpec, AkitaError> {
    let policy = (family.policy)();
    let grouped_requests = (family.grouped_requests)(preplans)?;
    Ok(EmitSpec {
        family_name: family.family_name(),
        policy,
        source_contract: (family.source_contract)()?,
        keys: emitted_scalar_keys(family)?,
        grouped_requests,
        preplanned_scalar: Vec::new(),
        output_dir,
        regen: family.regen,
        regen_group_batch: family.regen_group_batch,
        ring_challenge_config: family.ring_challenge_config,
    })
}

fn explicit_precommitted_group<Cfg: CommitmentConfig + 'static>(
    preplans: &GenerationPreplans,
    group: PolynomialGroupLayout,
) -> Result<PrecommittedProducer, AkitaError> {
    PrecommittedProducer::from_config::<Cfg>(planned_profile_without_precommitted_groups::<Cfg>(
        preplans, group,
    )?)
}

/// Every `Cfg` that has a checked-in schedule artifact.
///
/// Adding a new preset with an artifact requires adding a row
/// here; both the artifact emitter and the drift guard pick it up
/// automatically.
pub const ALL_GENERATED_FAMILIES: &[GeneratedFamily] = &[
    family_row!(
        FP128_ONEHOT_KEYS,
        fp128::OneHot,
        fp128_onehot_grouped_requests
    ),
    family_row!(
        recursive,
        FP128_ONEHOT_RECURSIVE_KEYS,
        RecursiveCommitmentConfig<fp128::OneHot>,
        fp128::OneHot,
        recursive_onehot_profile_keys::<fp128::OneHot>
    ),
    family_row!(
        recursive,
        &[],
        RecursiveCommitmentConfig<fp128::OneHotMultiChunk>,
        fp128::OneHotMultiChunk,
        recursive_onehot_chunked_profile_keys::<fp128::OneHotMultiChunk>
    ),
    family_row!(FP128_DENSE_KEYS, fp128::Dense, fp128_dense_grouped_requests),
    family_row!(
        FP128_ONEHOT_MULTI_CHUNK_KEYS,
        fp128::OneHotMultiChunk,
        fp128_onehot_multichunk_grouped_requests
    ),
    family_row!(
        FP128_ONEHOT_MULTI_CHUNK_W2R2_KEYS,
        fp128::OneHotMultiChunkW2R2,
        fp128_onehot_multichunk_w2r2_grouped_requests
    ),
    family_row!(
        FP128_ONEHOT_MULTI_CHUNK_W4R2_KEYS,
        fp128::OneHotMultiChunkW4R2,
        no_grouped_requests
    ),
    family_row!(
        FP128_DENSE_MULTI_CHUNK_KEYS,
        fp128::DenseMultiChunk,
        no_grouped_requests
    ),
    family_row!(
        FP128_DENSE_BOUNDED_KEYS,
        fp128::DenseBounded,
        no_grouped_requests
    ),
    family_row!(FP64_DENSE_KEYS, fp64::Dense, fp64_dense_grouped_requests),
    family_row!(FP64_ONEHOT_KEYS, fp64::OneHot, no_grouped_requests),
    family_row!(FP32_DENSE_KEYS, fp32::Dense, fp32_dense_grouped_requests),
    family_row!(FP32_ONEHOT_KEYS, fp32::OneHot, fp32_onehot_grouped_requests),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_preplans_deduplicate_by_exact_producer_and_layout() {
        let key = PolynomialGroupLayout::new(14, 1);
        let preplans = GenerationPreplans::default();

        let schedules = std::thread::scope(|scope| {
            let handles = (0..3)
                .map(|_| {
                    scope.spawn(|| preplans.scalar::<fp128::OneHot>(key).expect("one-hot plan"))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("preplan worker"))
                .collect::<Vec<_>>()
        });
        let first = schedules[0].clone();
        assert!(schedules.iter().all(|schedule| schedule == &first));
        assert_eq!(preplans.scalar.lock().expect("preplan cache").len(), 1);

        preplans
            .scalar::<fp32::OneHot>(key)
            .expect("same layout under a distinct producer");
        assert_eq!(preplans.scalar.lock().expect("preplan cache").len(), 2);

        let family = ALL_GENERATED_FAMILIES
            .iter()
            .find(|family| family.family_name() == "fp128_onehot")
            .expect("known family");
        let mut spec = empty_emit_spec(family, std::path::PathBuf::new())
            .expect("shipped families declare a valid producer contract");
        spec.keys = vec![key];
        preplans.attach_to_spec(family, &mut spec);
        assert_eq!(spec.preplanned_scalar, vec![(key, first)]);
    }
}
