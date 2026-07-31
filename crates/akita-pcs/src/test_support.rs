//! Test-only mixed ring-dimension schedule builders.
//!
//! Relocated from `akita-config`: after the runtime-resolution move (#327),
//! `akita-planner` depends on `akita-config`, so `akita-config` can no longer
//! depend on the offline planner ([`akita_planner::plan_optimal_suffix`])
//! without a dependency cycle. These builders live here (a crate above both)
//! and are gated behind the `test-support` Cargo feature, which production
//! builds never enable.
//!
//! The non-planner layout helpers (`akita_batched_root_layout`,
//! `ring_plan_test_seed`) remain in [`akita_config::test_support`].

use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
use akita_config::{policy_of, CommitmentConfig, PrecommittedCommitmentConfig};
use akita_field::AkitaError;
use akita_types::sis::{
    compute_num_digits_field_width, decomposed_t_ring_count, decomposed_w_ring_count,
    rounded_up_collision_inf_norm, rounded_up_role_a_inf_norm, InnerCommitMatrixParams,
    OpenCommitMatrixParams, OuterCommitMatrixParams, SisTableKey,
};
use akita_types::{
    active_setup_field_len, padded_setup_prefix_len, setup_prefix_slot_id, AkitaScheduleInputs,
    AkitaScheduleLookupKey, ChunkedWitnessCfg, CommitmentRingDims, CommittedGroupParams,
    DecompositionParams, FoldSchedule, OpeningClaimsLayout, PolynomialGroupLayout,
    RecursiveFoldParams, RecursiveFoldStep, RootFoldStep, RootSource, SetupMatrixEnvelope,
    SisMatrixRole, SisModulusProfileId, SisTableDigest, TerminalFoldParams, TerminalFoldStep,
    WitnessLayout, WitnessPartition,
};
use std::{
    any::TypeId,
    marker::PhantomData,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyntheticScheduleKind {
    MixedD,
    PerMatrixRingDimsRoot,
    RingDimensionTransition,
    RecursiveRingDimensionTransition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SyntheticScheduleCacheKey {
    kind: SyntheticScheduleKind,
    root: TypeId,
    middle: TypeId,
    suffix: TypeId,
    num_vars: usize,
    num_polynomials: usize,
    parameters: [usize; 4],
    lookup_key: Option<AkitaScheduleLookupKey>,
}

/// Cache only the most recently used synthetic schedule.
///
/// These test-only adapters invoke the offline planner, and setup, commit,
/// prove, and verify resolve the same schedule independently. A one-entry
/// cache removes that repeated planning work without allowing verifier-chosen
/// layouts to grow process memory without bound.
fn cached_synthetic_schedule(
    key: SyntheticScheduleCacheKey,
    build: impl FnOnce() -> Result<FoldSchedule, AkitaError>,
) -> Result<FoldSchedule, AkitaError> {
    static CACHE: OnceLock<Mutex<Option<(SyntheticScheduleCacheKey, FoldSchedule)>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(None));
    if let Some(schedule) = cache
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("synthetic schedule cache is poisoned".into()))?
        .as_ref()
        .filter(|(cached_key, _)| cached_key == &key)
        .map(|(_, schedule)| schedule.clone())
    {
        return Ok(schedule);
    }

    let schedule = build()?;
    *cache
        .lock()
        .map_err(|_| AkitaError::InvalidSetup("synthetic schedule cache is poisoned".into()))? =
        Some((key, schedule.clone()));
    Ok(schedule)
}

// -------------------------------------------------------------------------
// Multi-group carrier fixture: precommitted groups use the envelope config,
// while the final group and recursive suffix use a smaller native config.
// -------------------------------------------------------------------------

/// Test config for a multi-group root whose precommitted groups use
/// `Envelope::D` while the final group and recursive suffix use `Final::D`.
///
/// `Self::D` remains the setup-generation envelope. Runtime schedules are
/// selected by `Final`, but [`Self::get_params_for_prove`] freezes preceding
/// groups through `PrecommittedCommitmentConfig<Self>`, so their native
/// dimensions are retained in the grouped schedule.
#[derive(Debug)]
pub struct EnvelopeFinalGroupConfig<Envelope, Final>(PhantomData<fn() -> (Envelope, Final)>);

impl<Envelope, Final> Clone for EnvelopeFinalGroupConfig<Envelope, Final> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Envelope, Final> Copy for EnvelopeFinalGroupConfig<Envelope, Final> {}

impl<Envelope, Final> Default for EnvelopeFinalGroupConfig<Envelope, Final> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Envelope, Final> CommitmentConfig for EnvelopeFinalGroupConfig<Envelope, Final>
where
    Envelope: CommitmentConfig,
    Final: CommitmentConfig<Field = Envelope::Field, ExtField = Envelope::ExtField>,
{
    type Field = Envelope::Field;
    type ExtField = Envelope::ExtField;

    const D: usize = Envelope::D;

    fn decomposition() -> DecompositionParams {
        Envelope::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Envelope::ring_challenge_config(d).or_else(|_| Final::ring_challenge_config(d))
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Envelope::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Envelope::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Envelope::ring_subfield_embedding_norm_bound()
            .max(Final::ring_subfield_embedding_norm_bound())
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        let envelope =
            Envelope::max_setup_matrix_size(max_num_vars, max_num_batched_polys)?.max_setup_len;
        let final_field_elements =
            Final::max_setup_matrix_size(max_num_vars, max_num_batched_polys)?
                .max_setup_len
                .checked_mul(Final::D)
                .ok_or_else(|| AkitaError::InvalidSetup("final setup envelope overflow".into()))?;
        let final_at_envelope = final_field_elements.div_ceil(Envelope::D);
        Ok(SetupMatrixEnvelope {
            max_setup_len: envelope.max(final_at_envelope),
        })
    }

    fn basis_range() -> (u32, u32) {
        Envelope::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Envelope::onehot_chunk_size().min(Final::onehot_chunk_size())
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Envelope::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        let (policy, ring_challenge_config, fold_challenge_shape_at_level) =
            if key.precommitteds.is_empty() {
                (
                    policy_of::<Envelope>(),
                    Envelope::ring_challenge_config
                        as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
                    Envelope::fold_challenge_shape_at_level
                        as fn(AkitaScheduleInputs) -> TensorChallengeShape,
                )
            } else {
                (
                    policy_of::<Final>(),
                    Final::ring_challenge_config
                        as fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
                    Final::fold_challenge_shape_at_level
                        as fn(AkitaScheduleInputs) -> TensorChallengeShape,
                )
            };
        akita_planner::find_group_batch_schedule(
            &key,
            &policy,
            ring_challenge_config,
            fold_challenge_shape_at_level,
        )
        .map(|planned| planned.schedule)
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        opening_batch.check()?;
        let final_group = opening_batch.root_final_group_layout()?;
        let precommitteds = opening_batch
            .root_precommitted_group_layouts()?
            .iter()
            .copied()
            .map(|group| {
                let singleton =
                    OpeningClaimsLayout::new(group.num_vars(), group.num_polynomials())?;
                let params = <PrecommittedCommitmentConfig<Self> as CommitmentConfig>::
                    get_params_for_batched_commitment(&singleton)?;
                Ok(akita_types::PrecommittedGroupDescriptor::from_params(
                    group, &params,
                ))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Self::runtime_schedule(AkitaScheduleLookupKey {
            final_group,
            precommitteds,
        })
    }
}

// -------------------------------------------------------------------------
// Mixed ring-dimension-per-level schedule (runtime ring cutover experiment).
//
// Builds a schedule whose root levels `[0, switch_at_fold)` run at the
// envelope preset's ring dimension and whose recursive/terminal levels run at
// the (smaller) suffix preset's ring dimension. This is the single source of
// truth for the `mixed_d_per_level` acceptance test and the mixed-D profiler
// mode; see `specs/runtime-ring-cutover.md`.
// -------------------------------------------------------------------------

/// Build a mixed ring-dimension-per-level [`FoldSchedule`].
///
/// Fold levels `[0, switch_at_fold)` keep the `EnvelopeCfg` schedule prefix
/// (its root, and any recursive levels before the switch), at the envelope's
/// ring dimension. From `switch_at_fold` onward the schedule is a
/// **proof-size-optimal** continuation planned by [`akita_planner::plan_optimal_suffix`]
/// at `SuffixCfg`'s ring dimension, starting from the envelope prefix's output
/// witness — not the envelope's own (differently sized) suffix repriced in
/// place. This is what lets a large-ring-dimension root fold hand off to a
/// properly planned small-ring-dimension tail instead of terminating early on
/// a huge cleartext witness.
///
/// Fold level 0 is the typed root, levels 1.. are recursive entries, and the
/// last level is the typed direct terminal fold.
///
/// # Errors
///
/// Returns an error when the requested batch is not a singleton, the switch is
/// at the root, the switch skips beyond the envelope suffix, or the planner
/// cannot terminate the suffix.
pub fn mixed_d_per_level_schedule<EnvelopeCfg, SuffixCfg>(
    num_vars: usize,
    num_polynomials: usize,
    switch_at_fold: usize,
) -> Result<FoldSchedule, AkitaError>
where
    EnvelopeCfg: CommitmentConfig,
    SuffixCfg: CommitmentConfig,
{
    cached_synthetic_schedule(
        SyntheticScheduleCacheKey {
            kind: SyntheticScheduleKind::MixedD,
            root: TypeId::of::<EnvelopeCfg>(),
            middle: TypeId::of::<EnvelopeCfg>(),
            suffix: TypeId::of::<SuffixCfg>(),
            num_vars,
            num_polynomials,
            parameters: [switch_at_fold, 0, 0, 0],
            lookup_key: None,
        },
        || {
            if num_polynomials != 1 || switch_at_fold == 0 {
                return Err(AkitaError::InvalidSetup(
                    "mixed-D fixture requires a singleton and a non-root switch".into(),
                ));
            }
            let envelope_policy = policy_of::<EnvelopeCfg>();
            let envelope_domain =
                akita_planner::RingDimensionSearchDomain::uniform(envelope_policy.ring_dimension)?;
            let envelope = akita_planner::find_schedule(
                PolynomialGroupLayout::new(num_vars, num_polynomials),
                &envelope_policy,
                &envelope_domain,
                EnvelopeCfg::ring_challenge_config,
                EnvelopeCfg::fold_challenge_shape_at_level,
            )?
            .schedule;
            let keep_recursive = switch_at_fold - 1;
            if keep_recursive > envelope.recursive_folds.len() {
                return Err(AkitaError::InvalidSetup(
                    "mixed-D switch skips beyond the recursive suffix".into(),
                ));
            }
            // Envelope prefix kept at the envelope ring dimension: root + the recursive
            // folds before the switch point.
            let recursive_prefix = envelope.recursive_folds[..keep_recursive].to_vec();
            let (prefix_output_len, prefix_lb) = match recursive_prefix.last() {
                Some(last) => (last.output_witness_len, last.params.witness.log_basis_open),
                None => (
                    envelope.root.output_witness_len,
                    envelope.root.params.final_group.commitment.log_basis_open,
                ),
            };

            // Optimal small-ring-dimension continuation from the prefix output.
            let suffix = akita_planner::plan_optimal_suffix(
                &policy_of::<SuffixCfg>(),
                SuffixCfg::ring_challenge_config,
                SuffixCfg::fold_challenge_shape_at_level,
                num_vars,
                switch_at_fold,
                prefix_output_len,
                prefix_lb,
            )?;

            let mut recursive_folds = recursive_prefix;
            for fold in &suffix.folds {
                recursive_folds.push(RecursiveFoldStep {
                    params: RecursiveFoldParams {
                        open_commit_matrix: fold.params.open_commit_matrix.clone(),
                        sparse_challenge_config: fold.params.fold_challenge_config,
                        witness: fold.params.clone(),
                        incoming_setup_prefix: None,
                        witness_partition: WitnessPartition::Single,
                    },
                    input_witness_len: fold.input_witness_len,
                    output_witness_len: fold.output_witness_len,
                });
            }

            let schedule = FoldSchedule {
                root: envelope.root,
                recursive_folds,
                terminal: TerminalFoldStep {
                    params: TerminalFoldParams {
                        witness: suffix.terminal.params,
                        sparse_challenge_config: suffix.terminal.sparse_challenge_config,
                        response_shape: suffix.terminal.response_shape,
                    },
                    input_witness_len: suffix.terminal.input_witness_len,
                },
            };
            schedule.validate_structure()?;
            let opening_batch = OpeningClaimsLayout::new(num_vars, 1)?;
            schedule
                .root
                .params
                .final_group
                .commitment
                .validate_opening_batch(&opening_batch)?;
            Ok(schedule)
        },
    )
}

/// Config adapter that opens through a mixed ring-dimension-per-level schedule:
/// levels `[0, SWITCH_AT_FOLD)` at `Env`'s ring dimension, later levels at
/// `Suffix`'s ring dimension. Delegates every policy hook to `Env` (so
/// `Env::D` sets the setup's `gen_ring_dim`) and overrides schedule resolution
/// to build the mixed schedule via [`mixed_d_per_level_schedule`].
#[derive(Debug)]
pub struct MixedDConfig<Env, Suffix, const SWITCH_AT_FOLD: usize>(
    PhantomData<fn() -> (Env, Suffix)>,
);

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> Clone for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> Copy for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD> {}

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> Default
    for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Env, Suffix, const SWITCH_AT_FOLD: usize> CommitmentConfig
    for MixedDConfig<Env, Suffix, SWITCH_AT_FOLD>
where
    Env: CommitmentConfig,
    Suffix: CommitmentConfig<Field = Env::Field, ExtField = Env::ExtField>,
{
    type Field = Env::Field;
    type ExtField = Env::ExtField;

    const D: usize = Env::D;

    fn decomposition() -> DecompositionParams {
        Env::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Env::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Env::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Env::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Env::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        let mut envelope = SetupMatrixEnvelope::minimum();
        for num_polynomials in 1..=max_num_batched_polys.max(1) {
            let schedule = mixed_d_per_level_schedule::<Env, Suffix>(
                max_num_vars,
                num_polynomials,
                SWITCH_AT_FOLD,
            )?;
            let required = akita_types::setup_matrix_envelope_for_schedule(&schedule, Env::D)?;
            envelope.max_setup_len = envelope.max_setup_len.max(required.max_setup_len);
        }
        Ok(envelope)
    }

    fn basis_range() -> (u32, u32) {
        Env::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Env::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Env::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        mixed_d_per_level_schedule::<Env, Suffix>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            SWITCH_AT_FOLD,
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        mixed_d_per_level_schedule::<Env, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            SWITCH_AT_FOLD,
        )
    }
}

// -------------------------------------------------------------------------
// Per-matrix ring dimensions. The A, B, and D labels identify
// the inner/fold-carrier, outer-commitment, and opening-commitment matrices.
// Their protocol jobs never switch; only their ring dimensions vary.
// -------------------------------------------------------------------------

/// Build a root whose A, B, and D commitment matrices use
/// `Env::D`/`b_ring_dim`/`d_ring_dim`, then replan the complete uniform-`Env`
/// suffix from the root's exact outgoing witness.
///
/// # Errors
///
/// Returns an error when the matrix dimensions do not fit the A carrier, an
/// exact matrix width falls outside the audited SIS table, or no terminating
/// suffix can be planned.
pub fn per_matrix_ring_dims_root_schedule<Env: CommitmentConfig>(
    num_vars: usize,
    num_polynomials: usize,
    b_ring_dim: usize,
    d_ring_dim: usize,
) -> Result<FoldSchedule, AkitaError> {
    cached_synthetic_schedule(
        SyntheticScheduleCacheKey {
            kind: SyntheticScheduleKind::PerMatrixRingDimsRoot,
            root: TypeId::of::<Env>(),
            middle: TypeId::of::<Env>(),
            suffix: TypeId::of::<Env>(),
            num_vars,
            num_polynomials,
            parameters: [b_ring_dim, d_ring_dim, 0, 0],
            lookup_key: None,
        },
        || {
            let root_policy = policy_of::<Env>();
            let root_domain =
                akita_planner::RingDimensionSearchDomain::uniform(root_policy.ring_dimension)?;
            let mut root = akita_planner::find_schedule(
                PolynomialGroupLayout::new(num_vars, num_polynomials),
                &root_policy,
                &root_domain,
                Env::ring_challenge_config,
                Env::fold_challenge_shape_at_level,
            )?
            .schedule
            .root;
            retarget_commitment_matrices(
                &mut root.params.final_group.commitment,
                num_polynomials,
                b_ring_dim,
                d_ring_dim,
            )?;
            root.params.open_commit_matrix = root
                .params
                .final_group
                .commitment
                .open_commit_matrix
                .clone();

            let field_bits = Env::decomposition().field_bits();
            let opening_layout = OpeningClaimsLayout::new(num_vars, num_polynomials)?;
            let root_out = outgoing_witness_field_len(
                field_bits,
                &root.params.final_group.commitment,
                &opening_layout,
            )?;
            root.output_witness_len = root_out;
            let suffix = akita_planner::plan_optimal_suffix(
                &policy_of::<Env>(),
                Env::ring_challenge_config,
                Env::fold_challenge_shape_at_level,
                num_vars,
                1,
                root_out,
                root.params.final_group.commitment.log_basis_open,
            )?;
            finish_schedule(root, Vec::new(), suffix, &opening_layout)
        },
    )
}

/// Rebuild the B and D matrices from the final A-carrier geometry.
///
/// The exact widths are the native committed digit counts multiplied by
/// `d_a / d_b` and `d_a / d_d`. Deriving them from the final parameters is
/// essential for promoted carriers such as the temporary D512 experiment:
/// scaling a stale D256 matrix would undercount both widths by two.
fn retarget_commitment_matrices(
    commitment: &mut CommittedGroupParams,
    num_polynomials: usize,
    b_ring_dim: usize,
    d_ring_dim: usize,
) -> Result<(), AkitaError> {
    let dims = CommitmentRingDims {
        inner: commitment.d_a(),
        outer: b_ring_dim,
        opening: d_ring_dim,
    };
    dims.validate_a_carrier()?;
    let projected_width = |label: &str, native_width: usize, target_d: usize| {
        native_width
            .checked_mul(dims.d_a() / target_d)
            .ok_or_else(|| AkitaError::InvalidSetup(format!("{label} matrix width overflow")))
    };

    let native_outer_width = decomposed_t_ring_count(
        commitment.inner_commit_matrix.output_rank(),
        commitment.num_digits_outer,
        commitment.num_live_blocks,
        num_polynomials,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("B matrix width overflow".into()))?;
    let outer_width = projected_width("B", native_outer_width, b_ring_dim)?;
    let outer_key = commitment.outer_commit_matrix.sis_table_key();
    let outer_norm = rounded_up_collision_inf_norm(
        outer_key.policy,
        outer_key.modulus_profile,
        SisMatrixRole::Outer,
        b_ring_dim,
        commitment.log_basis_outer,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup("B matrix norm is outside audited SIS coverage".into())
    })?;
    commitment.outer_commit_matrix = OuterCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            ring_dimension: b_ring_dim as u32,
            coeff_linf_bound: outer_norm,
            ..outer_key
        },
        outer_width,
    )?;

    let native_open_width = decomposed_w_ring_count(
        commitment.num_digits_open,
        commitment.num_live_blocks,
        num_polynomials,
    )
    .ok_or_else(|| AkitaError::InvalidSetup("D matrix width overflow".into()))?;
    let mut open_width = projected_width("D", native_open_width, d_ring_dim)?;
    for group in &commitment.precommitted_groups {
        open_width = open_width
            .checked_add(group.d_segment_width(d_ring_dim)?)
            .ok_or_else(|| AkitaError::InvalidSetup("shared D matrix width overflow".into()))?;
    }
    let open_key = commitment.open_commit_matrix.sis_table_key();
    let open_norm = rounded_up_collision_inf_norm(
        open_key.policy,
        open_key.modulus_profile,
        SisMatrixRole::Open,
        d_ring_dim,
        commitment.log_basis_open,
    )
    .ok_or_else(|| {
        AkitaError::InvalidSetup("D matrix norm is outside audited SIS coverage".into())
    })?;
    commitment.open_commit_matrix = OpenCommitMatrixParams::try_new_with_min_rank(
        SisTableKey {
            ring_dimension: d_ring_dim as u32,
            coeff_linf_bound: open_norm,
            ..open_key
        },
        open_width,
    )?;
    Ok(())
}

/// Field-element length of the outgoing witness produced in the current
/// level's A-carrier ring.
fn outgoing_witness_field_len(
    field_bits: u32,
    commitment: &CommittedGroupParams,
    opening_layout: &OpeningClaimsLayout,
) -> Result<usize, AkitaError> {
    let relation_rows = commitment.relation_matrix_row_count(opening_layout.num_groups())?;
    let layout = WitnessLayout::new(
        commitment,
        opening_layout,
        commitment.witness_chunk.num_chunks,
        relation_rows,
        compute_num_digits_field_width(field_bits, commitment.log_basis_open),
    )?;
    layout
        .total_len()
        .checked_mul(commitment.relation_witness_carrier_ring_dimension())
        .ok_or_else(|| AkitaError::InvalidSetup("outgoing witness length overflow".into()))
}

/// Config adapter for a three-level ring-dimension transition: L0
/// `A/B/D = Env::D/Env::D/ROOT_D_RING_DIM`, L1
/// `A/B/D = Env::D/MID_BD_RING_DIM/MID_BD_RING_DIM`, then uniform `Suffix::D`.
#[derive(Debug)]
pub struct RingDimensionTransitionConfig<
    Env,
    Suffix,
    const MID_BD_RING_DIM: usize,
    const ROOT_D_RING_DIM: usize,
>(PhantomData<fn() -> (Env, Suffix)>);

impl<Env, Suffix, const MID_BD_RING_DIM: usize, const ROOT_D_RING_DIM: usize> Clone
    for RingDimensionTransitionConfig<Env, Suffix, MID_BD_RING_DIM, ROOT_D_RING_DIM>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Env, Suffix, const MID_BD_RING_DIM: usize, const ROOT_D_RING_DIM: usize> Copy
    for RingDimensionTransitionConfig<Env, Suffix, MID_BD_RING_DIM, ROOT_D_RING_DIM>
{
}

impl<Env, Suffix, const MID_BD_RING_DIM: usize, const ROOT_D_RING_DIM: usize> Default
    for RingDimensionTransitionConfig<Env, Suffix, MID_BD_RING_DIM, ROOT_D_RING_DIM>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Env, Suffix, const MID_BD_RING_DIM: usize, const ROOT_D_RING_DIM: usize> CommitmentConfig
    for RingDimensionTransitionConfig<Env, Suffix, MID_BD_RING_DIM, ROOT_D_RING_DIM>
where
    Env: CommitmentConfig,
    Suffix: CommitmentConfig<Field = Env::Field, ExtField = Env::ExtField>,
{
    type Field = Env::Field;
    type ExtField = Env::ExtField;

    const D: usize = Env::D;

    fn decomposition() -> DecompositionParams {
        Env::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Env::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Env::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Env::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Env::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        if max_num_batched_polys > 1 {
            return Err(AkitaError::InvalidSetup(
                "ring-dimension transition requires a singleton batch".into(),
            ));
        }
        let schedule = ring_dimension_transition_schedule::<Env, Env, Suffix>(
            max_num_vars,
            1,
            CommitmentRingDims {
                inner: Env::D,
                outer: Env::D,
                opening: ROOT_D_RING_DIM,
            },
            CommitmentRingDims {
                inner: Env::D,
                outer: MID_BD_RING_DIM,
                opening: MID_BD_RING_DIM,
            },
        )?;
        akita_types::setup_matrix_envelope_for_schedule(&schedule, Env::D)
    }

    fn basis_range() -> (u32, u32) {
        Env::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Env::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Env::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        ring_dimension_transition_schedule::<Env, Env, Suffix>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            CommitmentRingDims {
                inner: Env::D,
                outer: Env::D,
                opening: ROOT_D_RING_DIM,
            },
            CommitmentRingDims {
                inner: Env::D,
                outer: MID_BD_RING_DIM,
                opening: MID_BD_RING_DIM,
            },
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        ring_dimension_transition_schedule::<Env, Env, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            CommitmentRingDims {
                inner: Env::D,
                outer: Env::D,
                opening: ROOT_D_RING_DIM,
            },
            CommitmentRingDims {
                inner: Env::D,
                outer: MID_BD_RING_DIM,
                opening: MID_BD_RING_DIM,
            },
        )
    }
}

/// Convert one planned suffix fold into its wire schedule representation
/// (mirrors the `mixed_d_per_level_schedule` conversion).
fn planned_fold_step(fold: &akita_planner::PlannedSuffixFold) -> RecursiveFoldStep {
    let witness_partition = if fold.params.witness_chunk.num_chunks == 1 {
        WitnessPartition::Single
    } else {
        WitnessPartition::Distributed {
            num_chunks: fold.params.witness_chunk.num_chunks,
        }
    };
    RecursiveFoldStep {
        params: RecursiveFoldParams {
            open_commit_matrix: fold.params.open_commit_matrix.clone(),
            sparse_challenge_config: fold.params.fold_challenge_config,
            witness: fold.params.clone(),
            incoming_setup_prefix: None,
            witness_partition,
        },
        input_witness_len: fold.input_witness_len,
        output_witness_len: fold.output_witness_len,
    }
}

fn finish_schedule(
    root: RootFoldStep,
    mut recursive_folds: Vec<RecursiveFoldStep>,
    suffix: akita_planner::PlannedSuffix,
    opening_layout: &OpeningClaimsLayout,
) -> Result<FoldSchedule, AkitaError> {
    recursive_folds.extend(suffix.folds.iter().map(planned_fold_step));
    let schedule = FoldSchedule {
        root,
        recursive_folds,
        terminal: TerminalFoldStep {
            params: TerminalFoldParams {
                witness: suffix.terminal.params,
                sparse_challenge_config: suffix.terminal.sparse_challenge_config,
                response_shape: suffix.terminal.response_shape,
            },
            input_witness_len: suffix.terminal.input_witness_len,
        },
    };
    schedule.validate_structure()?;
    schedule
        .root
        .params
        .final_group
        .commitment
        .validate_opening_batch(opening_layout)?;
    Ok(schedule)
}

/// Three-band ring-dimension transition:
///
/// - L0 uses `root_dims`, with A fixed to `Root::D`.
/// - L1 uses `middle_dims`, with A fixed to `Mid::D`.
/// - L2+: uniform `Suffix::D`.
///
/// Each continuation is planned only after the preceding level's exact matrix
/// dimensions, SIS ranks, and outgoing witness length are known. The D512 root
/// remains a temporary test-only promotion from D256 planner geometry; all
/// promoted A/B/D matrices are nevertheless rebuilt and priced from their
/// final dimensions. Native per-matrix ring-dimension root planning is
/// required before this D512 experiment can leave `test-support`.
///
/// # Errors
///
/// Returns an error when the batch is not a singleton, either planner pass
/// fails, a matrix width leaves the audited SIS table, or the schedule
/// fails structural validation.
pub fn ring_dimension_transition_schedule<Root, Mid, Suffix>(
    num_vars: usize,
    num_polynomials: usize,
    root_dims: CommitmentRingDims,
    middle_dims: CommitmentRingDims,
) -> Result<FoldSchedule, AkitaError>
where
    Root: CommitmentConfig,
    Mid: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    Suffix: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
{
    cached_synthetic_schedule(
        SyntheticScheduleCacheKey {
            kind: SyntheticScheduleKind::RingDimensionTransition,
            root: TypeId::of::<Root>(),
            middle: TypeId::of::<Mid>(),
            suffix: TypeId::of::<Suffix>(),
            num_vars,
            num_polynomials,
            parameters: [
                root_dims.d_b(),
                root_dims.d_d(),
                middle_dims.d_b(),
                middle_dims.d_d(),
            ],
            lookup_key: None,
        },
        || {
            if num_polynomials != 1 {
                return Err(AkitaError::InvalidSetup(
                    "ring-dimension transition requires a singleton batch".into(),
                ));
            }
            root_dims.validate_a_carrier()?;
            middle_dims.validate_a_carrier()?;
            if root_dims.d_a() != Root::D || middle_dims.d_a() != Mid::D {
                return Err(AkitaError::InvalidSetup(
                    "ring-dimension transition A dimensions must match the Root and Mid policies"
                        .into(),
                ));
            }
            Root::validate_sis_modulus_profile()?;
            Mid::validate_sis_modulus_profile()?;
            Suffix::validate_sis_modulus_profile()?;
            let field_bits = Root::decomposition().field_bits();
            // L0: tableless experiment presets use the offline planner directly.
            // Q128 D512 has audited A-matrix rows but no native per-matrix
            // ring-dimension planner
            // candidate, so temporarily start from D256 root geometry and promote A.
            let mut root_policy = policy_of::<Root>();
            let planned_root_d = if Root::D == 512 { 256 } else { Root::D };
            root_policy.ring_dimension = planned_root_d;
            let root_domain = akita_planner::RingDimensionSearchDomain::uniform(planned_root_d)?;
            let mut root = akita_planner::find_schedule(
                PolynomialGroupLayout::new(num_vars, num_polynomials),
                &root_policy,
                &root_domain,
                Root::ring_challenge_config,
                Root::fold_challenge_shape_at_level,
            )?
            .schedule
            .root;

            if Root::D != planned_root_d {
                if Root::D != 512
                    || Root::sis_modulus_profile() != SisModulusProfileId::Q128OffsetA7F7
                {
                    return Err(AkitaError::InvalidSetup(format!(
                        "ring-dimension transition A promotion from D={planned_root_d} to D={} is unsupported",
                        Root::D
                    )));
                }
                let scale = Root::D / planned_root_d;
                let mut commitment = root.params.final_group.commitment.clone();
                if !commitment
                    .num_live_ring_elements_per_claim
                    .is_multiple_of(scale)
                    || !commitment.num_positions_per_block.is_multiple_of(scale)
                {
                    return Err(AkitaError::InvalidSetup(
                        "D512 transition root cannot preserve the flat source length".into(),
                    ));
                }
                commitment.num_live_ring_elements_per_claim /= scale;
                commitment.num_positions_per_block /= scale;

                let ring_challenge = Root::ring_challenge_config(Root::D)?;
                commitment.fold_challenge_config = ring_challenge;
                let inner_width = commitment
                    .num_positions_per_block
                    .checked_mul(commitment.num_digits_inner)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("D512 A matrix width overflow".into())
                    })?;
                let decomposition = DecompositionParams {
                    log_basis: commitment.log_basis_inner,
                    ..Root::decomposition()
                };
                let norm = rounded_up_role_a_inf_norm(
                    root_policy.sis_security_policy,
                    SisTableDigest::Q128_INNER_D512,
                    Root::sis_modulus_profile(),
                    Root::D,
                    decomposition,
                    commitment.log_basis_open,
                    &ring_challenge,
                    commitment.fold_challenge_shape,
                    true,
                    Root::onehot_chunk_size(),
                    Root::ring_subfield_embedding_norm_bound(),
                    commitment.num_live_blocks,
                    num_polynomials,
                    inner_width as u64,
                )
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "D512 A-matrix norm is outside audited SIS coverage".into(),
                    )
                })?;
                commitment.inner_commit_matrix = InnerCommitMatrixParams::try_new_with_min_rank(
                    SisTableKey {
                        policy: root_policy.sis_security_policy,
                        table_digest: SisTableDigest::Q128_INNER_D512,
                        modulus_profile: Root::sis_modulus_profile(),
                        role: SisMatrixRole::Inner,
                        ring_dimension: Root::D as u32,
                        coeff_linf_bound: norm,
                    },
                    inner_width,
                )?;
                commitment = commitment.with_fold_linf_cap_config(field_bits, num_polynomials)?;
                root.params.final_group.commitment = commitment;
                root.params.sparse_challenge_config = ring_challenge;
            }

            // Rebuild root B/D after the optional A-only promotion. Widths are derived
            // from the final A carrier, not from the stale planned D256 matrices.
            retarget_commitment_matrices(
                &mut root.params.final_group.commitment,
                num_polynomials,
                root_dims.d_b(),
                root_dims.d_d(),
            )?;
            root.params.open_commit_matrix = root
                .params
                .final_group
                .commitment
                .open_commit_matrix
                .clone();
            let root_out = outgoing_witness_field_len(
                field_bits,
                &root.params.final_group.commitment,
                &OpeningClaimsLayout::new(num_vars, num_polynomials)?,
            )?;
            root.output_witness_len = root_out;
            let root_lb = root.params.final_group.commitment.log_basis_open;

            // Band 2 (`Mid`): plan an optimal `Mid`-dim continuation from the root
            // output and keep only its first fold as L1.
            let mid = akita_planner::plan_optimal_suffix(
                &policy_of::<Mid>(),
                Mid::ring_challenge_config,
                Mid::fold_challenge_shape_at_level,
                num_vars,
                1,
                root_out,
                root_lb,
            )?;
            let l1 = mid.folds.first().ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "ring-dimension transition Mid band produced no fold".into(),
                )
            })?;

            let mut l1_step = planned_fold_step(l1);
            // Rebuild L1 B/D from its final A carrier before planning the suffix.
            retarget_commitment_matrices(
                &mut l1_step.params.witness,
                num_polynomials,
                middle_dims.d_b(),
                middle_dims.d_d(),
            )?;
            l1_step.params.open_commit_matrix = l1_step.params.witness.open_commit_matrix.clone();
            let l1_out = outgoing_witness_field_len(
                field_bits,
                &l1_step.params.witness,
                &akita_planner::suffix_opening_layout(l1_step.input_witness_len, None)?,
            )?;
            l1_step.output_witness_len = l1_out;
            let l1_lb = l1_step.params.witness.log_basis_open;

            // Band 3 (`Suffix`): optimal small-ring continuation from L1's output.
            let suffix = akita_planner::plan_optimal_suffix(
                &policy_of::<Suffix>(),
                Suffix::ring_challenge_config,
                Suffix::fold_challenge_shape_at_level,
                num_vars,
                2,
                l1_out,
                l1_lb,
            )?;
            let opening_layout = OpeningClaimsLayout::new(num_vars, num_polynomials)?;
            finish_schedule(root, vec![l1_step], suffix, &opening_layout)
        },
    )
}

mod recursive_transition;
mod setup_prefix_slots;
pub use recursive_transition::{
    recursive_ring_dimension_transition_schedule, RecursiveRingDimensionTransitionConfig,
};
pub use setup_prefix_slots::materialize_schedule_setup_prefix_slots;
/// Config adapter for [`ring_dimension_transition_schedule`].
///
/// `Root`/`Mid`/`Suffix` set the A-matrix dimensions; `ROOT_BD_RING_DIM` and
/// `L1_BD_RING_DIM` set both B and D at L0/L1. `Root::D` sets the setup
/// generation dimension.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct ThreeBandRingDimensionTransitionConfig<
    Root,
    Mid,
    Suffix,
    const ROOT_BD_RING_DIM: usize,
    const L1_BD_RING_DIM: usize,
>(PhantomData<fn() -> (Root, Mid, Suffix)>);

impl<Root, Mid, Suffix, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize> Clone
    for ThreeBandRingDimensionTransitionConfig<Root, Mid, Suffix, ROOT_BD_RING_DIM, L1_BD_RING_DIM>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Root, Mid, Suffix, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize> Copy
    for ThreeBandRingDimensionTransitionConfig<Root, Mid, Suffix, ROOT_BD_RING_DIM, L1_BD_RING_DIM>
{
}

impl<Root, Mid, Suffix, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize> Default
    for ThreeBandRingDimensionTransitionConfig<Root, Mid, Suffix, ROOT_BD_RING_DIM, L1_BD_RING_DIM>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Root, Mid, Suffix, const ROOT_BD_RING_DIM: usize, const L1_BD_RING_DIM: usize> CommitmentConfig
    for ThreeBandRingDimensionTransitionConfig<Root, Mid, Suffix, ROOT_BD_RING_DIM, L1_BD_RING_DIM>
where
    Root: CommitmentConfig,
    Mid: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
    Suffix: CommitmentConfig<Field = Root::Field, ExtField = Root::ExtField>,
{
    type Field = Root::Field;
    type ExtField = Root::ExtField;

    const D: usize = Root::D;

    fn decomposition() -> DecompositionParams {
        Root::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Root::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Root::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Root::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Root::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        if max_num_batched_polys > 1 {
            return Err(AkitaError::InvalidSetup(
                "three-band ring-dimension transition requires a singleton batch".into(),
            ));
        }
        let schedule = ring_dimension_transition_schedule::<Root, Mid, Suffix>(
            max_num_vars,
            1,
            CommitmentRingDims {
                inner: Root::D,
                outer: ROOT_BD_RING_DIM,
                opening: ROOT_BD_RING_DIM,
            },
            CommitmentRingDims {
                inner: Mid::D,
                outer: L1_BD_RING_DIM,
                opening: L1_BD_RING_DIM,
            },
        )?;
        akita_types::setup_matrix_envelope_for_schedule(&schedule, Root::D)
    }

    fn basis_range() -> (u32, u32) {
        Root::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Root::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Root::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        ring_dimension_transition_schedule::<Root, Mid, Suffix>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            CommitmentRingDims {
                inner: Root::D,
                outer: ROOT_BD_RING_DIM,
                opening: ROOT_BD_RING_DIM,
            },
            CommitmentRingDims {
                inner: Mid::D,
                outer: L1_BD_RING_DIM,
                opening: L1_BD_RING_DIM,
            },
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        ring_dimension_transition_schedule::<Root, Mid, Suffix>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            CommitmentRingDims {
                inner: Root::D,
                outer: ROOT_BD_RING_DIM,
                opening: ROOT_BD_RING_DIM,
            },
            CommitmentRingDims {
                inner: Mid::D,
                outer: L1_BD_RING_DIM,
                opening: L1_BD_RING_DIM,
            },
        )
    }
}

/// Config adapter whose root commitment matrices use
/// `A/B/D = Env::D/B_RING_DIM/D_RING_DIM`. The replanned suffix is uniform
/// `Env::D`.
#[derive(Debug)]
pub struct PerMatrixRingDimsRootConfig<Env, const B_RING_DIM: usize, const D_RING_DIM: usize>(
    PhantomData<fn() -> Env>,
);

impl<Env, const B_RING_DIM: usize, const D_RING_DIM: usize> Clone
    for PerMatrixRingDimsRootConfig<Env, B_RING_DIM, D_RING_DIM>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Env, const B_RING_DIM: usize, const D_RING_DIM: usize> Copy
    for PerMatrixRingDimsRootConfig<Env, B_RING_DIM, D_RING_DIM>
{
}

impl<Env, const B_RING_DIM: usize, const D_RING_DIM: usize> Default
    for PerMatrixRingDimsRootConfig<Env, B_RING_DIM, D_RING_DIM>
{
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Env, const B_RING_DIM: usize, const D_RING_DIM: usize> CommitmentConfig
    for PerMatrixRingDimsRootConfig<Env, B_RING_DIM, D_RING_DIM>
where
    Env: CommitmentConfig,
{
    type Field = Env::Field;
    type ExtField = Env::ExtField;

    const D: usize = Env::D;

    fn decomposition() -> DecompositionParams {
        Env::decomposition()
    }

    fn ring_challenge_config(d: usize) -> Result<SparseChallengeConfig, AkitaError> {
        Env::ring_challenge_config(d)
    }

    fn fold_challenge_shape_at_level(inputs: AkitaScheduleInputs) -> TensorChallengeShape {
        Env::fold_challenge_shape_at_level(inputs)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        Env::sis_modulus_profile()
    }

    fn ring_subfield_embedding_norm_bound() -> u32 {
        Env::ring_subfield_embedding_norm_bound()
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
    ) -> Result<SetupMatrixEnvelope, AkitaError> {
        // Smaller B/D rings widen those matrices, so size the setup from the
        // actual per-matrix ring-dimension schedule rather than the uniform
        // envelope.
        let mut max_setup_len = 1usize;
        for num_polys in 1..=max_num_batched_polys.max(1) {
            let schedule = per_matrix_ring_dims_root_schedule::<Env>(
                max_num_vars,
                num_polys,
                B_RING_DIM,
                D_RING_DIM,
            )?;
            let required = akita_types::setup_matrix_envelope_for_schedule(&schedule, Env::D)?;
            max_setup_len = max_setup_len.max(required.max_setup_len);
        }
        Ok(SetupMatrixEnvelope { max_setup_len })
    }

    fn basis_range() -> (u32, u32) {
        Env::basis_range()
    }

    fn onehot_chunk_size() -> usize {
        Env::onehot_chunk_size()
    }

    fn schedule_catalog() -> Option<akita_planner::GeneratedScheduleTable> {
        Env::schedule_catalog()
    }

    fn runtime_schedule(key: AkitaScheduleLookupKey) -> Result<FoldSchedule, AkitaError> {
        per_matrix_ring_dims_root_schedule::<Env>(
            key.final_group.num_vars(),
            key.final_group.num_polynomials(),
            B_RING_DIM,
            D_RING_DIM,
        )
    }

    fn get_params_for_prove(
        opening_batch: &OpeningClaimsLayout,
    ) -> Result<FoldSchedule, AkitaError> {
        per_matrix_ring_dims_root_schedule::<Env>(
            opening_batch.max_num_vars(),
            opening_batch.num_total_polynomials(),
            B_RING_DIM,
            D_RING_DIM,
        )
    }
}
