#![allow(missing_docs)]

use akita_algebra::poly::multilinear_eval;
use akita_challenges::{SparseChallengeConfig, Stage1ChallengeShape};
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_field::CanonicalField;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, CommittedPolynomials, DensePoly, OneHotPoly};
use akita_transcript::Blake2bTranscript;
use akita_types::{
    AjtaiRole, AkitaBatchedProof, AkitaCommitmentHint, AkitaRootBatchSummary, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaSchedulePlan, AkitaVerifierSetup, BasisMode, CommitmentEnvelope,
    DecompositionParams, RingCommitment, Schedule, ScheduleProvider, Step, WitnessShape,
};
use akita_verifier::{CommitmentVerifier, CommittedOpenings};
use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, BatchSize, BenchmarkGroup, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::marker::PhantomData;
use std::time::Duration;

type F = fp128::Field;

#[derive(Clone, Copy, Debug)]
struct TensorCfg<Base>(PhantomData<Base>);

#[derive(Clone, Copy, Debug)]
struct ClaimReductionCfg<Base>(PhantomData<Base>);

/// Hybrid stage-1 shape wrapper used by the bench harness.
///
/// `planner_stage1_shapes_to_search` exposes both `Tensor` and `Flat`
/// to the DP planner, which then picks the per-level best by
/// objective_cost. For ExactShell-based fp128 configs this is
/// SIS-safe because the only allowed shape switch is Tensor → Flat,
/// which over-secures the SIS rank (smaller mass than was derived).
#[derive(Clone, Copy, Debug)]
struct PlannerHybridCfg<Base>(PhantomData<Base>);

fn apply_claim_reduction(lp: akita_types::LevelParams) -> akita_types::LevelParams {
    lp.with_tensor_stage1_challenges()
        .with_setup_claim_reduction()
}

impl<Base: ScheduleProvider> ScheduleProvider for ClaimReductionCfg<Base> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("claim-reduction/{}", Base::schedule_key(key))
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }
}

impl<Base> akita_planner::PlannerConfig for ClaimReductionCfg<Base>
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
        Base::planner_root_level_layout_with_log_basis(inputs, log_basis).map(apply_claim_reduction)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
            .map(apply_claim_reduction)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        Ok(apply_claim_reduction(params))
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }
}

fn claim_reduction_schedule<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<Schedule, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    akita_planner::find_optimal_schedule_with_max::<ClaimReductionCfg<Base>>(
        max_num_vars,
        num_vars,
        WitnessShape::new(
            batch.num_claims,
            batch.num_commitment_groups,
            batch.num_points,
        ),
    )
}

fn claim_reduction_root_layout<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<akita_types::LevelParams, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    let schedule = claim_reduction_schedule::<Base>(max_num_vars, num_vars, batch)?;
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
        Some(Step::Direct(_)) | None => {
            Base::commitment_layout(num_vars).map(apply_claim_reduction)
        }
    }
}

fn claim_reduction_matrix_size<Base>(
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

    let singleton = claim_reduction_schedule::<Base>(
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
        let batched = claim_reduction_schedule::<Base>(max_num_vars, max_num_vars, batch)?;
        for step in &batched.steps {
            if let Step::Fold(fold) = step {
                visit(&fold.params);
            }
        }
    }

    Ok((max_rows, max_stride))
}

impl<Base> CommitmentConfig for ClaimReductionCfg<Base>
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
        let (cr_rows, cr_stride) = claim_reduction_matrix_size::<Base>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )?;
        Ok((base_rows.max(cr_rows), base_stride.max(cr_stride)))
    }

    fn level_params_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> akita_types::LevelParams {
        apply_claim_reduction(Base::level_params_with_log_basis(inputs, log_basis))
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::root_level_params_for_layout_with_log_basis(inputs, lp)?;
        Ok(apply_claim_reduction(params))
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis).map(apply_claim_reduction)
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
        claim_reduction_root_layout::<Base>(
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
        claim_reduction_root_layout::<Base>(num_vars, num_vars, batch)
    }

    fn get_params_for_batched_commitment(
        max_num_vars: usize,
        num_vars: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        claim_reduction_root_layout::<Base>(max_num_vars, num_vars, batch)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        if layout_num_claims != batch.num_claims {
            return Err(akita_field::AkitaError::InvalidSetup(format!(
                "claim-reduction benchmark schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        claim_reduction_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

impl<Base: ScheduleProvider> ScheduleProvider for TensorCfg<Base> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        None
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        format!("tensor-stage1/{}", Base::schedule_key(key))
    }

    fn schedule_plan(
        _key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Ok(None)
    }
}

impl<Base> akita_planner::PlannerConfig for TensorCfg<Base>
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
            .map(akita_types::LevelParams::with_tensor_stage1_challenges)
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::planner_current_level_layout_with_log_basis(inputs, log_basis)
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &akita_types::LevelParams,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        let params = Base::planner_root_level_params_for_layout_with_log_basis(inputs, lp)?;
        if matches!(
            lp.stage1_challenge_shape,
            akita_challenges::Stage1ChallengeShape::Tensor
        ) {
            Ok(params.with_tensor_stage1_challenges())
        } else {
            Ok(params)
        }
    }

    fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        Base::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        Base::planner_stage1_prover_weight()
    }
}

fn tensor_schedule<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<Schedule, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    akita_planner::find_optimal_schedule_with_max::<TensorCfg<Base>>(
        max_num_vars,
        num_vars,
        WitnessShape::new(
            batch.num_claims,
            batch.num_commitment_groups,
            batch.num_points,
        ),
    )
}

fn tensor_root_layout<Base>(
    max_num_vars: usize,
    num_vars: usize,
    batch: AkitaRootBatchSummary,
) -> Result<akita_types::LevelParams, akita_field::AkitaError>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    let schedule = tensor_schedule::<Base>(max_num_vars, num_vars, batch)?;
    match schedule.steps.first() {
        Some(Step::Fold(root_step)) => Ok(root_step.params.clone()),
        Some(Step::Direct(_)) | None => Base::commitment_layout(num_vars)
            .map(akita_types::LevelParams::with_tensor_stage1_challenges),
    }
}

fn tensor_schedule_matrix_size<Base>(
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

    let singleton = tensor_schedule::<Base>(
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
        let batched = tensor_schedule::<Base>(max_num_vars, max_num_vars, batch)?;
        for step in &batched.steps {
            if let Step::Fold(fold) = step {
                visit(&fold.params);
            }
        }
    }

    Ok((max_rows, max_stride))
}

impl<Base> CommitmentConfig for TensorCfg<Base>
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
        let (tensor_rows, tensor_stride) = tensor_schedule_matrix_size::<Base>(
            max_num_vars,
            max_num_batched_polys,
            max_num_points,
        )?;
        Ok((base_rows.max(tensor_rows), base_stride.max(tensor_stride)))
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
            akita_challenges::Stage1ChallengeShape::Tensor
        ) {
            Ok(params.with_tensor_stage1_challenges())
        } else {
            Ok(params)
        }
    }

    fn root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        Base::root_level_layout_with_log_basis(inputs, log_basis)
            .map(akita_types::LevelParams::with_tensor_stage1_challenges)
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
        tensor_root_layout::<Base>(
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
        tensor_root_layout::<Base>(num_vars, num_vars, batch)
    }

    fn get_params_for_batched_commitment(
        max_num_vars: usize,
        num_vars: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<akita_types::LevelParams, akita_field::AkitaError> {
        tensor_root_layout::<Base>(max_num_vars, num_vars, batch)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        if layout_num_claims != batch.num_claims {
            return Err(akita_field::AkitaError::InvalidSetup(format!(
                "tensor benchmark schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        tensor_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

// ---------- PlannerHybridCfg<Base> ----------

impl<Base: ScheduleProvider> ScheduleProvider for PlannerHybridCfg<Base> {
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
                "planner-hybrid benchmark schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        planner_hybrid_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

fn make_dense_evals<Cfg: CommitmentConfig<Field = F>>(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let len = 1usize << nv;
    let decomp = Cfg::decomposition();
    if decomp.log_commit_bound >= 128 {
        (0..len)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect()
    } else {
        let half_bound = 1i64 << (decomp.log_commit_bound.min(62) - 1);
        (0..len)
            .map(|_| F::from_i64(rng.gen_range(-half_bound..half_bound)))
            .collect()
    }
}

fn random_point(nv: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0xcafe_babe);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn emit_verify_bench_metadata(label: &str, nv: usize, proof: &AkitaBatchedProof<F>) {
    let root_kind = if proof.is_root_direct() {
        "direct"
    } else {
        "fold"
    };
    let final_shape = if proof.is_root_direct() {
        "root-direct".to_string()
    } else {
        match proof.final_witness().shape() {
            akita_types::DirectWitnessShape::PackedDigits((num_elems, bits_per_elem)) => {
                format!("packed:{num_elems}x{bits_per_elem}")
            }
            akita_types::DirectWitnessShape::FieldElements(num_elems) => {
                format!("field:{num_elems}")
            }
        }
    };
    eprintln!(
        "[verify-bench-meta] label={label} nv={nv} proof_bytes={} root={root_kind} recursive_folds={} final_witness={final_shape}",
        proof.size(),
        proof.num_fold_levels(),
    );
}

fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, nv: usize) {
    if nv >= 20 {
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(30));
    }
}

fn bench_dense_phases<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
    >,
{
    let evals = make_dense_evals::<Cfg>(nv);
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point(nv);
    let opening = multilinear_eval(&evals, &pt).unwrap();

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("setup", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(
                    black_box(nv),
                    black_box(1),
                    black_box(1),
                ),
            )
        })
    });

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);

    group.bench_function("commit", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    black_box(std::slice::from_ref(&poly)),
                    black_box(&setup),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .unwrap();

    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    group.bench_function("prove", |b| {
        b.iter_batched(
            || vec![hint.clone()],
            |h| {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench");
                black_box(
                    <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                        &setup,
                        vec![(
                            &pt[..],
                            vec![CommittedPolynomials {
                                polynomials: &poly_refs[..],
                                commitment: &commitments[0],
                                hint: h.into_iter().next().unwrap(),
                            }],
                        )],
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )]),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function("e2e", |b| {
        b.iter(|| {
            let (cm, h) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                std::slice::from_ref(&poly),
                &setup,
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                &setup,
                vec![(
                    &pt[..],
                    vec![CommittedPolynomials {
                        polynomials: &poly_refs[..],
                        commitment: &cms[0],
                        hint: h,
                    }],
                )],
                &mut pt_tr,
                BasisMode::Lagrange,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &cms[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_onehot_phases<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
    >,
{
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let mut rng = StdRng::seed_from_u64(0xbeef_cafe);
    let indices: Vec<Option<usize>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();

    let onehot_poly = OneHotPoly::<F, D>::new(onehot_k, indices.clone()).unwrap();

    let dense_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); total_ring * onehot_k];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let pt = random_point(nv);
    let opening = multilinear_eval(&dense_evals, &pt).unwrap();

    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    configure_group(&mut group, nv);

    group.bench_function("commit_onehot", |b| {
        b.iter(|| {
            black_box(
                <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                    black_box(std::slice::from_ref(&onehot_poly)),
                    black_box(&setup),
                )
                .unwrap(),
            )
        })
    });

    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&onehot_poly),
        &setup,
    )
    .unwrap();

    let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];

    group.bench_function("prove", |b| {
        b.iter_batched(
            || vec![hint.clone()],
            |h| {
                let mut transcript = Blake2bTranscript::<F>::new(b"bench");
                black_box(
                    <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                        &setup,
                        vec![(
                            &pt[..],
                            vec![CommittedPolynomials {
                                polynomials: &poly_refs[..],
                                commitment: &commitments[0],
                                hint: h.into_iter().next().unwrap(),
                            }],
                        )],
                        &mut transcript,
                        BasisMode::Lagrange,
                    )
                    .unwrap(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"bench");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();

    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )]),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });

    group.bench_function("e2e", |b| {
        b.iter(|| {
            let (cm, h) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
                std::slice::from_ref(&onehot_poly),
                &setup,
            )
            .unwrap();
            let cms = [cm];
            let mut pt_tr = Blake2bTranscript::<F>::new(b"bench");
            let pf = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
                &setup,
                vec![(
                    &pt[..],
                    vec![CommittedPolynomials {
                        polynomials: &poly_refs[..],
                        commitment: &cms[0],
                        hint: h,
                    }],
                )],
                &mut pt_tr,
                BasisMode::Lagrange,
            )
            .unwrap();
            let mut vt_tr = Blake2bTranscript::<F>::new(b"bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                &pf,
                &verifier_setup,
                &mut vt_tr,
                vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &cms[0],
                    }],
                )],
                BasisMode::Lagrange,
            )
            .unwrap();
            black_box(())
        })
    });

    group.finish();
}

fn bench_onehot_verify_only<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
    >,
{
    let layout = Cfg::commitment_layout(nv).expect("benchmark layout");
    let total_ring = layout.num_blocks * layout.block_len;
    let onehot_k = D;

    let mut rng = StdRng::seed_from_u64(0x715e_0001);
    let indices: Vec<Option<usize>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..onehot_k)))
        .collect();
    let onehot_poly = OneHotPoly::<F, D>::new(onehot_k, indices.clone()).unwrap();

    let dense_evals: Vec<F> = {
        let mut evals = vec![F::from_u64(0); total_ring * onehot_k];
        for (ci, opt_idx) in indices.iter().enumerate() {
            if let Some(idx) = opt_idx {
                evals[ci * onehot_k + idx] = F::from_u64(1);
            }
        }
        evals
    };
    let pt = random_point(nv);
    let opening = multilinear_eval(&dense_evals, &pt).unwrap();
    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&onehot_poly),
        &setup,
    )
    .unwrap();
    let poly_refs: [&OneHotPoly<F, D>; 1] = [&onehot_poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"tensor-verify-bench");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    emit_verify_bench_metadata(label, nv, &proof);

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"tensor-verify-bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )]),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });
    group.finish();
}

fn bench_dense_verify_only<const D: usize, Cfg: CommitmentConfig<Field = F>>(
    c: &mut Criterion,
    label: &str,
    nv: usize,
) where
    AkitaCommitmentScheme<D, Cfg>: CommitmentProver<
        F,
        D,
        VerifierSetup = AkitaVerifierSetup<F>,
        Commitment = RingCommitment<F, D>,
        CommitHint = AkitaCommitmentHint<F, D>,
        BatchedProof = AkitaBatchedProof<F>,
    >,
{
    let evals = make_dense_evals::<Cfg>(nv);
    let poly = DensePoly::<F, D>::from_field_evals(nv, &evals).unwrap();
    let pt = random_point(nv);
    let opening = multilinear_eval(&evals, &pt).unwrap();
    let setup = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_prover(nv, 1, 1);
    let verifier_setup =
        <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::setup_verifier(&setup);
    let (commitment, hint) = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::commit(
        std::slice::from_ref(&poly),
        &setup,
    )
    .unwrap();
    let poly_refs: [&DensePoly<F, D>; 1] = [&poly];
    let commitments = [commitment];
    let openings = [opening];
    let opening_groups = [&openings[..]];
    let mut prover_transcript = Blake2bTranscript::<F>::new(b"tensor-verify-bench");
    let proof = <AkitaCommitmentScheme<D, Cfg> as CommitmentProver<F, D>>::batched_prove(
        &setup,
        vec![(
            &pt[..],
            vec![CommittedPolynomials {
                polynomials: &poly_refs[..],
                commitment: &commitments[0],
                hint,
            }],
        )],
        &mut prover_transcript,
        BasisMode::Lagrange,
    )
    .unwrap();
    emit_verify_bench_metadata(label, nv, &proof);

    let mut group = c.benchmark_group(format!("akita/{label}/nv{nv}"));
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.bench_function("verify", |b| {
        b.iter(|| {
            let mut transcript = Blake2bTranscript::<F>::new(b"tensor-verify-bench");
            <AkitaCommitmentScheme<D, Cfg> as CommitmentVerifier<F, D>>::batched_verify(
                black_box(&proof),
                black_box(&verifier_setup),
                &mut transcript,
                black_box(vec![(
                    &pt[..],
                    vec![CommittedOpenings {
                        openings: opening_groups[0],
                        commitment: &commitments[0],
                    }],
                )]),
                BasisMode::Lagrange,
            )
            .unwrap();
        })
    });
    group.finish();
}

fn bench_full_nv15(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D128Full::D }, fp128::D128Full>(c, "full-d128", 15);
}
fn bench_full_nv20(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D128Full::D }, fp128::D128Full>(c, "full-d128", 20);
}
fn bench_full_nv25(c: &mut Criterion) {
    bench_dense_phases::<{ fp128::D128Full::D }, fp128::D128Full>(c, "full-d128", 25);
}

fn bench_onehot_nv15(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(c, "onehot-d64", 15);
}
fn bench_onehot_nv20(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(c, "onehot-d64", 20);
}
fn bench_onehot_nv25(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(c, "onehot-d64", 25);
}

// Hybrid (planner-driven Flat/Tensor per level) prove + verify phases.
fn bench_onehot_planner_hybrid_nv15(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, PlannerHybridCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-planner-hybrid",
        15,
    );
}
fn bench_onehot_planner_hybrid_nv20(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, PlannerHybridCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-planner-hybrid",
        20,
    );
}
fn bench_onehot_planner_hybrid_nv25(c: &mut Criterion) {
    bench_onehot_phases::<{ fp128::D64OneHot::D }, PlannerHybridCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-planner-hybrid",
        25,
    );
}

fn bench_d32_full_stage1_verify_flat_nv12(c: &mut Criterion) {
    bench_dense_verify_only::<{ fp128::D32Full::D }, fp128::D32Full>(c, "full-d32-flat-stage1", 12);
}

fn bench_d64_full_stage1_verify_flat_nv12(c: &mut Criterion) {
    bench_dense_verify_only::<{ fp128::D64Full::D }, fp128::D64Full>(c, "full-d64-flat-stage1", 12);
}

fn bench_d64_full_stage1_verify_tensor_nv12(c: &mut Criterion) {
    bench_dense_verify_only::<{ fp128::D64Full::D }, TensorCfg<fp128::D64Full>>(
        c,
        "full-d64-tensor-stage1",
        12,
    );
}

fn bench_d32_onehot_stage1_verify_flat_nv12(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D32OneHot::D }, fp128::D32OneHot>(
        c,
        "onehot-d32-flat-stage1",
        12,
    );
}

fn bench_d64_onehot_stage1_verify_flat_nv12(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(
        c,
        "onehot-d64-flat-stage1",
        12,
    );
}

fn bench_d64_onehot_stage1_verify_tensor_nv12(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, TensorCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-tensor-stage1",
        12,
    );
}

fn bench_d64_onehot_stage1_verify_flat_nv15(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(
        c,
        "onehot-d64-flat-stage1",
        15,
    );
}

fn bench_d64_onehot_stage1_verify_tensor_nv15(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, TensorCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-tensor-stage1",
        15,
    );
}

fn bench_d64_onehot_stage1_verify_flat_nv20(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(
        c,
        "onehot-d64-flat-stage1",
        20,
    );
}

fn bench_d64_onehot_stage1_verify_tensor_nv20(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, TensorCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-tensor-stage1",
        20,
    );
}

fn bench_d64_onehot_stage1_verify_flat_nv25(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, fp128::D64OneHot>(
        c,
        "onehot-d64-flat-stage1",
        25,
    );
}

fn bench_d64_onehot_stage1_verify_tensor_nv25(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, TensorCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-tensor-stage1",
        25,
    );
}

fn bench_d64_onehot_stage1_verify_cr_nv12(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, ClaimReductionCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-claim-reduction",
        12,
    );
}

fn bench_d64_onehot_stage1_verify_cr_nv15(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, ClaimReductionCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-claim-reduction",
        15,
    );
}

fn bench_d64_onehot_stage1_verify_cr_nv20(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, ClaimReductionCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-claim-reduction",
        20,
    );
}

fn bench_d64_onehot_stage1_verify_cr_nv25(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, ClaimReductionCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-claim-reduction",
        25,
    );
}

fn bench_d64_onehot_stage1_verify_planner_hybrid_nv15(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, PlannerHybridCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-planner-hybrid",
        15,
    );
}

fn bench_d64_onehot_stage1_verify_planner_hybrid_nv20(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, PlannerHybridCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-planner-hybrid",
        20,
    );
}

fn bench_d64_onehot_stage1_verify_planner_hybrid_nv25(c: &mut Criterion) {
    bench_onehot_verify_only::<{ fp128::D64OneHot::D }, PlannerHybridCfg<fp128::D64OneHot>>(
        c,
        "onehot-d64-planner-hybrid",
        25,
    );
}

fn bench_d64_full_stage1_verify_cr_nv12(c: &mut Criterion) {
    bench_dense_verify_only::<{ fp128::D64Full::D }, ClaimReductionCfg<fp128::D64Full>>(
        c,
        "full-d64-claim-reduction",
        12,
    );
}

criterion_group!(
    akita_benches,
    bench_full_nv15,
    bench_full_nv20,
    bench_full_nv25,
    bench_onehot_nv15,
    bench_onehot_nv20,
    bench_onehot_nv25,
    bench_onehot_planner_hybrid_nv15,
    bench_onehot_planner_hybrid_nv20,
    bench_onehot_planner_hybrid_nv25,
    bench_d32_full_stage1_verify_flat_nv12,
    bench_d64_full_stage1_verify_flat_nv12,
    bench_d64_full_stage1_verify_tensor_nv12,
    bench_d64_full_stage1_verify_cr_nv12,
    bench_d32_onehot_stage1_verify_flat_nv12,
    bench_d64_onehot_stage1_verify_flat_nv12,
    bench_d64_onehot_stage1_verify_tensor_nv12,
    bench_d64_onehot_stage1_verify_cr_nv12,
    bench_d64_onehot_stage1_verify_flat_nv15,
    bench_d64_onehot_stage1_verify_tensor_nv15,
    bench_d64_onehot_stage1_verify_cr_nv15,
    bench_d64_onehot_stage1_verify_flat_nv20,
    bench_d64_onehot_stage1_verify_tensor_nv20,
    bench_d64_onehot_stage1_verify_cr_nv20,
    bench_d64_onehot_stage1_verify_flat_nv25,
    bench_d64_onehot_stage1_verify_tensor_nv25,
    bench_d64_onehot_stage1_verify_cr_nv25,
    bench_d64_onehot_stage1_verify_planner_hybrid_nv15,
    bench_d64_onehot_stage1_verify_planner_hybrid_nv20,
    bench_d64_onehot_stage1_verify_planner_hybrid_nv25,
);

/// Set `AKITA_PARALLEL=0` to run benchmarks single-threaded.
fn main() {
    #[cfg(feature = "parallel")]
    {
        let num_threads = if std::env::var("AKITA_PARALLEL")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            tracing::info!("AKITA_PARALLEL=0: running single-threaded");
            1
        } else {
            0
        };
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .stack_size(64 * 1024 * 1024)
            .build_global()
            .ok();
    }

    akita_benches();
    criterion::Criterion::default()
        .configure_from_args()
        .final_summary();
}
