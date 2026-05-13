//! Phase K.0: hand-built mixed stage-1 shape schedule, end-to-end.
//!
//! Goal: validate that the prover, verifier, and serialization stack support
//! a fold ladder in which *different levels* use *different stage-1 challenge
//! shapes* (some `Flat`, some `Tensor`), without any planner changes.
//!
//! This is the architectural sanity check that gates Phase K.1 (extending
//! the planner DP search to actually pick mixed shapes). If this test
//! passes, the per-level `LevelParams::stage1_challenge_shape` plumbing is
//! end-to-end correct and the planner is the only thing missing.
//!
//! The configuration `HybridFlatRootTensorRecCfg<Base>` exercises the most
//! interesting case for verifier performance per the rev-3 plan: keep the
//! root level flat (smaller L1 mass → lighter digit shape → cheaper
//! verifier work on the largest level) and let recursive levels stay
//! tensor (where the prover-side savings dominate).

#![allow(missing_docs)]

mod common;

use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_planner::PlannerConfig as PlannerConfigTrait;
use akita_prover::CommitmentProver;
use akita_transcript::Blake2bTranscript;
use akita_types::{
    AjtaiRole, AkitaRootBatchSummary, AkitaScheduleInputs, AkitaScheduleLookupKey,
    AkitaSchedulePlan, CommitmentEnvelope, DecompositionParams, Schedule, ScheduleProvider, Step,
    WitnessShape,
};
use akita_verifier::CommitmentVerifier;
use common::*;
use std::marker::PhantomData;
use std::sync::Mutex;

static E2E_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Hybrid stage-1 shape wrapper: root level uses `Flat`, recursive levels
/// inherit the base config's per-level shape (typically `Tensor` for
/// `ExactShell` configs).
#[derive(Clone, Copy, Debug)]
struct HybridFlatRootTensorRecCfg<Base>(PhantomData<Base>);

type HybridOneHotCfg = HybridFlatRootTensorRecCfg<OneHotCfg>;

impl<Base> ScheduleProvider for HybridFlatRootTensorRecCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("hybrid-flat-root-tensor-rec/{}", Base::schedule_key(key))
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }
}

impl<Base> akita_planner::PlannerConfig for HybridFlatRootTensorRecCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        // Root: force Flat shape (overriding any tensor mapping the base would
        // apply via `apply_stage1_challenge_shape`).
        Base::planner_root_level_layout_with_log_basis(inputs, log_basis)
            .map(akita_types::LevelParams::with_flat_stage1_challenges)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        // Recursive: keep base's shape (Tensor for ExactShell configs).
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(
            lp.stage1_challenge_shape,
            akita_challenges::Stage1ChallengeShape::Flat
        ) {
            Ok(params.with_flat_stage1_challenges())
        } else {
            Ok(params.with_tensor_stage1_challenges())
        }
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }
}

fn hybrid_schedule<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<Schedule, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    akita_planner::find_optimal_schedule_with_max::<HybridFlatRootTensorRecCfg<Base>>(
        max_num_vars,
        num_vars,
        WitnessShape::new(
            batch.num_claims,
            batch.num_commitment_groups,
            batch.num_points,
        ),
    )
}

fn hybrid_root_layout<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<akita_types::LevelParams, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    let schedule = hybrid_schedule::<Base>(max_num_vars, num_vars, batch)?;
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
        Some(Step::Direct(_)) | None => Base::commitment_layout(num_vars)
            .map(akita_types::LevelParams::with_flat_stage1_challenges),
    }
}

fn hybrid_schedule_matrix_size<Base>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    let mut max_rows = 1usize;
    let mut max_stride = 1usize;
    let mut visit = |lp: &akita_types::LevelParams| {
        max_rows = max_rows
            .max(lp.a_key.row_len())
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len());
        max_stride = max_stride
            .max(lp.inner_width())
            .max(lp.outer_width())
            .max(lp.d_matrix_width());
    };

    let singleton = hybrid_schedule::<Base>(
        max_num_vars,
        max_num_vars,
        AkitaRootBatchSummary::singleton(),
    )?;
    for step in &singleton.steps {
        if let Step::Fold(fold) = step {
            visit(&fold.params);
        }
    }

    if max_num_batched_polys > 1 {
        let batch = AkitaRootBatchSummary::new(
            max_num_batched_polys,
            max_num_batched_polys,
            max_num_points.min(max_num_batched_polys).max(1),
        )?;
        let batched = hybrid_schedule::<Base>(max_num_vars, max_num_vars, batch)?;
        for step in &batched.steps {
            if let Step::Fold(fold) = step {
                visit(&fold.params);
            }
        }
    }

    Ok((max_rows, max_stride))
}

impl<Base> CommitmentConfig for HybridFlatRootTensorRecCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Base::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), akita_field::AkitaError> {
        let (base_rows, base_stride) =
            Base::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)?;
        let (hybrid_rows, hybrid_stride) = hybrid_schedule_matrix_size::<Base>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )?;
        Ok((base_rows.max(hybrid_rows), base_stride.max(hybrid_stride)))
    }

    fn level_params_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> akita_types::LevelParams {
        Base::level_params_with_log_basis(inputs, log_basis)
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(
            lp.stage1_challenge_shape,
            akita_challenges::Stage1ChallengeShape::Flat
        ) {
            Ok(params.with_flat_stage1_challenges())
        } else {
            Ok(params.with_tensor_stage1_challenges())
        }
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis)
            .map(akita_types::LevelParams::with_flat_stage1_challenges)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Base::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::log_basis_search_range(inputs)
    }

    fn commitment_layout(
        max_num_vars: usize,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        hybrid_root_layout::<Base>(
            max_num_vars,
            max_num_vars,
            AkitaRootBatchSummary::singleton(),
        )
    }

    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_point: usize,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let batch = AkitaRootBatchSummary::new(num_polys_per_point, 1, 1)?;
        hybrid_root_layout::<Base>(num_vars, num_vars, batch)
    }

    fn get_params_for_batched_commitment(
        max_num_vars: usize,
        num_vars: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        hybrid_root_layout::<Base>(max_num_vars, num_vars, batch)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        if layout_num_claims != batch.num_claims {
            return Err(akita_field::AkitaError::InvalidSetup(format!(
                "hybrid test schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        hybrid_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

/// Inspect the schedule the planner produces under the hybrid config and
/// assert that the root is `Flat` while at least one recursive fold is
/// `Tensor`. This is the protocol-level evidence that mixed shapes are
/// actually being exercised by the test.
#[test]
fn hybrid_schedule_has_flat_root_and_tensor_recursive() {
    const NV: usize = 20;
    let schedule = hybrid_schedule::<OneHotCfg>(NV, NV, AkitaRootBatchSummary::singleton())
        .expect("hybrid schedule");
    let mut saw_flat_root = false;
    let mut saw_tensor_recursive = false;
    for (idx, step) in schedule.steps.iter().enumerate() {
        if let Step::Fold(fold) = step {
            let shape = &fold.params.stage1_challenge_shape;
            eprintln!("level {idx} stage1_shape={shape:?}");
            if idx == 0 {
                assert!(
                    matches!(shape, akita_challenges::Stage1ChallengeShape::Flat),
                    "root level must be Flat under HybridFlatRootTensorRecCfg"
                );
                saw_flat_root = true;
            } else if matches!(shape, akita_challenges::Stage1ChallengeShape::Tensor) {
                saw_tensor_recursive = true;
            }
        }
    }
    assert!(saw_flat_root, "expected a flat root fold");
    assert!(
        saw_tensor_recursive,
        "expected at least one tensor recursive fold at NV={NV}; this test \
         needs an NV with recursive folds to actually validate mixing"
    );
}

/// Planner-driven hybrid wrapper: the planner is allowed to pick `Flat`
/// or `Tensor` per fold level (via `planner_stage1_shapes_to_search`).
/// The base config's default shape is reused as the starting point; the
/// SIS-safe `Tensor → Flat` switch is the one the planner can actually
/// take, so this config opts into hybrid only for `ExactShell` /
/// `Uniform` bases whose default shape is `Tensor`.
#[derive(Clone, Copy, Debug)]
struct PlannerHybridCfg<Base>(PhantomData<Base>);

impl<Base> ScheduleProvider for PlannerHybridCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }
    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("planner-hybrid/{}", Base::schedule_key(key))
    }
    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }
}

impl<Base> akita_planner::PlannerConfig for PlannerHybridCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    const PLANNER_D: usize = Base::PLANNER_D;

    fn planner_field_bits() -> u32 {
        Base::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_root_level_layout_with_log_basis_for_shape(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
        shape: Stage1ChallengeShape,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_root_level_layout_with_log_basis_for_shape(inputs, log_basis, shape)
    }

    fn planner_current_level_layout_with_log_basis_for_shape(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
        shape: Stage1ChallengeShape,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_current_level_layout_with_log_basis_for_shape(inputs, log_basis, shape)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(lp.stage1_challenge_shape, Stage1ChallengeShape::Flat) {
            Ok(params.with_flat_stage1_challenges())
        } else {
            Ok(params.with_tensor_stage1_challenges())
        }
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }

    fn planner_stage1_shapes_to_search() -> Vec<Stage1ChallengeShape> {
        vec![Stage1ChallengeShape::Tensor, Stage1ChallengeShape::Flat]
    }
}

fn planner_hybrid_schedule<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<Schedule, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    akita_planner::find_optimal_schedule_with_max::<PlannerHybridCfg<Base>>(
        max_num_vars,
        num_vars,
        WitnessShape::new(
            batch.num_claims,
            batch.num_commitment_groups,
            batch.num_points,
        ),
    )
}

fn planner_hybrid_root_layout<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<akita_types::LevelParams, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    let schedule = planner_hybrid_schedule::<Base>(max_num_vars, num_vars, batch)?;
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
        Some(Step::Direct(_)) | None => Base::commitment_layout(num_vars),
    }
}

fn planner_hybrid_schedule_matrix_size<Base>(
    max_num_vars: usize,
    max_num_batched_polys: usize,
    max_num_points: usize,
) -> Result<(usize, usize), akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    let mut max_rows = 1usize;
    let mut max_stride = 1usize;
    let mut visit = |lp: &akita_types::LevelParams| {
        max_rows = max_rows
            .max(lp.a_key.row_len())
            .max(lp.b_key.row_len())
            .max(lp.d_key.row_len());
        max_stride = max_stride
            .max(lp.inner_width())
            .max(lp.outer_width())
            .max(lp.d_matrix_width());
    };

    let singleton = planner_hybrid_schedule::<Base>(
        max_num_vars,
        max_num_vars,
        AkitaRootBatchSummary::singleton(),
    )?;
    for step in &singleton.steps {
        if let Step::Fold(fold) = step {
            visit(&fold.params);
        }
    }

    if max_num_batched_polys > 1 {
        let batch = AkitaRootBatchSummary::new(
            max_num_batched_polys,
            max_num_batched_polys,
            max_num_points.min(max_num_batched_polys).max(1),
        )?;
        let batched = planner_hybrid_schedule::<Base>(max_num_vars, max_num_vars, batch)?;
        for step in &batched.steps {
            if let Step::Fold(fold) = step {
                visit(&fold.params);
            }
        }
    }

    Ok((max_rows, max_stride))
}

impl<Base> CommitmentConfig for PlannerHybridCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    type Field = Base::Field;
    type ClaimField = Base::ClaimField;
    type ChallengeField = Base::ChallengeField;
    const D: usize = Base::D;

    fn decomposition() -> DecompositionParams {
        Base::decomposition()
    }
    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        Base::stage1_challenge_config(d)
    }
    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        Base::audited_root_rank(role, max_num_vars)
    }
    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        Base::envelope(max_num_vars)
    }
    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), akita_field::AkitaError> {
        let (base_rows, base_stride) =
            Base::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)?;
        let (h_rows, h_stride) = planner_hybrid_schedule_matrix_size::<Base>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )?;
        Ok((base_rows.max(h_rows), base_stride.max(h_stride)))
    }
    fn level_params_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> akita_types::LevelParams {
        Base::level_params_with_log_basis(inputs, log_basis)
    }
    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(lp.stage1_challenge_shape, Stage1ChallengeShape::Flat) {
            Ok(params.with_flat_stage1_challenges())
        } else {
            Ok(params.with_tensor_stage1_challenges())
        }
    }
    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis)
    }
    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        Base::log_basis_at_level(inputs)
    }
    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::log_basis_search_range(inputs)
    }
    fn commitment_layout(
        max_num_vars: usize,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        planner_hybrid_root_layout::<Base>(
            max_num_vars,
            max_num_vars,
            AkitaRootBatchSummary::singleton(),
        )
    }
    fn get_params_for_commitment(
        num_vars: usize,
        num_polys_per_point: usize,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let batch = AkitaRootBatchSummary::new(num_polys_per_point, 1, 1)?;
        planner_hybrid_root_layout::<Base>(num_vars, num_vars, batch)
    }
    fn get_params_for_batched_commitment(
        max_num_vars: usize,
        num_vars: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        planner_hybrid_root_layout::<Base>(max_num_vars, num_vars, batch)
    }
    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        if layout_num_claims != batch.num_claims {
            return Err(akita_field::AkitaError::InvalidSetup(format!(
                "planner-hybrid schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        planner_hybrid_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

#[test]
fn planner_hybrid_schedule_is_self_consistent_nv25() {
    // The bench harness panicked at NV=25 with "scheduled recursive level
    // did not match runtime state". Reproduce minimally and find which
    // level pair has the inconsistency.
    const NV: usize = 25;
    let schedule = planner_hybrid_schedule::<OneHotCfg>(NV, NV, AkitaRootBatchSummary::singleton())
        .expect("hybrid schedule");
    eprintln!("--- hybrid schedule at NV={NV} ---");
    let mut prev_next_w_len: Option<usize> = None;
    for (idx, step) in schedule.steps.iter().enumerate() {
        match step {
            Step::Fold(fold) => {
                let lp = &fold.params;
                eprintln!(
                    "  level {idx} Fold: current_w_len={} log_basis={} shape={:?} \
                     n_a={} n_b={} n_d={} num_digits_fold={} num_blocks={} block_len={}",
                    fold.current_w_len,
                    lp.log_basis,
                    lp.stage1_challenge_shape,
                    lp.a_key.row_len(),
                    lp.b_key.row_len(),
                    lp.d_key.row_len(),
                    lp.num_digits_fold,
                    lp.num_blocks,
                    lp.block_len,
                );
                if let Some(prev) = prev_next_w_len {
                    assert_eq!(
                        prev, fold.current_w_len,
                        "fold step {idx}.current_w_len ({}) must equal prev step's next_w_len ({})",
                        fold.current_w_len, prev
                    );
                }
                prev_next_w_len = Some(akita_types::planned_next_w_len(
                    <OneHotCfg as PlannerConfigTrait>::planner_field_bits(),
                    lp,
                ));
            }
            Step::Direct(direct) => {
                eprintln!(
                    "  level {idx} Direct: current_w_len={} bits_per_elem={}",
                    direct.current_w_len, direct.bits_per_elem,
                );
                if let Some(prev) = prev_next_w_len {
                    assert_eq!(
                        prev, direct.current_w_len,
                        "direct step {idx}.current_w_len ({}) must equal prev step's next_w_len ({})",
                        direct.current_w_len, prev
                    );
                }
            }
        }
    }
}

#[test]
fn planner_hybrid_schedule_is_at_least_as_good_as_tensor_only() {
    const NV: usize = 20;
    let hybrid = planner_hybrid_schedule::<OneHotCfg>(NV, NV, AkitaRootBatchSummary::singleton())
        .expect("hybrid schedule");
    let tensor_only = hybrid_schedule::<OneHotCfg>(NV, NV, AkitaRootBatchSummary::singleton())
        .expect("tensor-only fallback");
    eprintln!(
        "NV={NV}: hybrid total_bytes={} tensor_only total_bytes={}",
        hybrid.total_bytes, tensor_only.total_bytes
    );
    // The hybrid planner has strict-superset of options (tensor included),
    // so it must never be worse than tensor-only.
    assert!(
        hybrid.total_bytes <= tensor_only.total_bytes,
        "hybrid planner found a worse schedule than tensor-only \
         (hybrid={}, tensor-only={}); this means the DP search isn't \
         considering tensor in every state",
        hybrid.total_bytes,
        tensor_only.total_bytes,
    );
    // Log per-level shape choices for postmortem.
    for (idx, step) in hybrid.steps.iter().enumerate() {
        if let Step::Fold(fold) = step {
            eprintln!(
                "  hybrid level {idx} shape={:?} digits_fold={}",
                fold.params.stage1_challenge_shape, fold.params.num_digits_fold,
            );
        }
    }
}

#[test]
fn hybrid_stage1_onehot_prove_verify_singleton() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, HybridOneHotCfg>;

        let layout = HybridOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x715e_d3a0);
        let pt = random_point(NV, 0x715e_d3a1);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"hybrid_stage1_e2e/onehot_singleton");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("hybrid onehot prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"hybrid_stage1_e2e/onehot_singleton");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("hybrid onehot verify");
    });
}

#[test]
fn planner_hybrid_onehot_prove_verify_singleton() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, PlannerHybridCfg<OneHotCfg>>;

        let layout = <PlannerHybridCfg<OneHotCfg> as CommitmentConfig>::commitment_layout(NV)
            .expect("layout");
        let poly = make_onehot_poly(&layout, 0x715e_d3b0);
        let pt = random_point(NV, 0x715e_d3b1);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"hybrid_stage1_e2e/planner_hybrid_onehot");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("planner-hybrid onehot prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"hybrid_stage1_e2e/planner_hybrid_onehot");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("planner-hybrid onehot verify");
    });
}
