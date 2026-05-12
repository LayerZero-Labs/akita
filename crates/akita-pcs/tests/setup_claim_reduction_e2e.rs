//! End-to-end tests for the setup-side claim-reduction sumcheck flow.
//!
//! These tests mirror `tensor_stage1_e2e.rs` but additionally enable
//! `LevelParams::use_setup_claim_reduction` at the root level. The full proof
//! must therefore include a `SetupClaimReductionPayload` that the verifier
//! consumes to close the stage-2 main sumcheck without the materialized
//! M-table contribution from setup-side rows.

#![allow(missing_docs)]

mod common;

use akita_challenges::SparseChallengeConfig;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
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

#[derive(Clone, Copy, Debug)]
struct ClaimReductionCfg<Base>(PhantomData<Base>);

type ClaimReductionOneHotCfg = ClaimReductionCfg<OneHotCfg>;
type ClaimReductionDenseCfg = ClaimReductionCfg<DenseCfg>;

fn apply_claim_reduction(lp: akita_types::LevelParams) -> akita_types::LevelParams {
    lp.with_tensor_stage1_challenges()
        .with_setup_claim_reduction()
}

impl<Base> ScheduleProvider for ClaimReductionCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
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
        Base::level_params_with_log_basis(inputs, log_basis)
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
                "claim-reduction test schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        claim_reduction_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

#[test]
fn setup_claim_reduction_dense_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionDenseCfg>;

        let layout = ClaimReductionDenseCfg::commitment_layout(NV).expect("layout");
        assert!(
            layout.use_setup_claim_reduction,
            "test must exercise setup claim reduction"
        );
        let poly = make_dense_poly(NV, 0x7c1a_1d31);
        let pt = random_point(NV, 0x7c1a_1d32);
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
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/dense_singleton");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("dense claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold()
            .expect("test must exercise tensor root fold");
        assert!(
            fold_root.stage2.setup_claim_reduction.is_some(),
            "fold root stage-2 proof should carry a setup claim-reduction payload"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/dense_singleton");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("dense claim-reduction verify");
    });
}

#[test]
fn setup_claim_reduction_onehot_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionOneHotCfg>;

        let layout = ClaimReductionOneHotCfg::commitment_layout(NV).expect("layout");
        assert!(
            layout.use_setup_claim_reduction,
            "test must exercise setup claim reduction"
        );
        let poly = make_onehot_poly(&layout, 0x7c1a_2222);
        let pt = random_point(NV, 0x7c1a_2223);
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
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_singleton");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold()
            .expect("test must exercise tensor root fold");
        assert!(
            fold_root.stage2.setup_claim_reduction.is_some(),
            "fold root stage-2 proof should carry a setup claim-reduction payload"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_singleton");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction verify");
    });
}

#[test]
fn setup_claim_reduction_rejects_tampered_m_setup_eval() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, ClaimReductionOneHotCfg>;

        let layout = ClaimReductionOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x7c1a_b001);
        let pt = random_point(NV, 0x7c1a_b002);
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
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_tamper");
        let mut proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("onehot claim-reduction prove");

        let fold_root = proof
            .root
            .as_fold_mut()
            .expect("tamper test must exercise tensor root fold");
        let payload = fold_root
            .stage2
            .setup_claim_reduction
            .as_mut()
            .expect("tamper test must have a setup claim-reduction payload");
        payload.m_setup_eval += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"setup_claim_reduction_e2e/onehot_tamper");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered setup claim-reduction m_setup_eval must be rejected"
        );
    });
}
