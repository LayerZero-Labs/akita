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
struct TensorCfg<Base>(PhantomData<Base>);

type TensorOneHotCfg = TensorCfg<OneHotCfg>;
type TensorDenseCfg = TensorCfg<DenseCfg>;

impl<Base> ScheduleProvider for TensorCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
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
                "tensor test schedule requires layout_num_claims ({layout_num_claims}) to match total claims ({})",
                batch.num_claims
            )));
        }
        tensor_schedule::<Base>(max_num_vars, num_vars, batch)
    }
}

#[test]
fn tensor_stage1_dense_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 15;
        const D: usize = DENSE_D;
        type Scheme = AkitaCommitmentScheme<D, TensorDenseCfg>;

        let layout = TensorDenseCfg::commitment_layout(NV).expect("layout");
        let poly = make_dense_poly(NV, 0x715e_d3e5);
        let pt = random_point(NV, 0x715e_d3e6);
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
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/dense_singleton");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("dense tensor prove");
        assert!(
            proof.root.as_fold().is_some(),
            "test must exercise tensor root fold"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/dense_singleton");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("dense tensor verify");
    });
}

#[test]
fn tensor_stage1_rejects_tampered_s_claim() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, TensorOneHotCfg>;

        let layout = TensorOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x715e_5c1a);
        let pt = random_point(NV, 0x715e_5c1b);
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
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/s_claim_tamper");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tensor prove");

        let mut malformed = proof.clone();
        malformed
            .root
            .as_fold_mut()
            .expect("tamper test must exercise tensor root fold")
            .stage1
            .s_claim += F::from_u64(1);

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/s_claim_tamper");
        let result = <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &malformed,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tampered tensor stage-1 s_claim must be rejected"
        );
    });
}

#[test]
fn tensor_stage1_same_point_batched_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, TensorOneHotCfg>;

        let layout = akita_config::akita_batched_root_layout::<TensorOneHotCfg>(NV, 2)
            .expect("batched layout");
        let polys = vec![
            make_onehot_poly(&layout, 0x715e_ba70),
            make_onehot_poly(&layout, 0x715e_ba71),
        ];
        let pt = random_point(NV, 0x715e_ba72);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly::<D, _>(poly, &pt, &layout))
            .collect();
        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, 2, 1);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <Scheme as CommitmentProver<F, D>>::commit(&polys, &setup).expect("commit");
        let poly_refs: Vec<&OneHotPoly<F, D, u8>> = polys.iter().collect();
        let commitments = [commitment];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/d64_same_point");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tensor batched prove");
        assert!(
            proof.root.as_fold().is_some(),
            "test must exercise tensor root fold"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/d64_same_point");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("tensor batched verify");
    });
}

#[test]
fn tensor_stage1_multipoint_prove_verify() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type Scheme = AkitaCommitmentScheme<D, TensorOneHotCfg>;

        let point_group_sizes = [vec![1], vec![1]];
        let total_claims = 2usize;
        let layout = akita_config::akita_batched_root_layout::<TensorOneHotCfg>(NV, total_claims)
            .expect("multipoint layout");
        let point_polys = [
            vec![make_onehot_poly(&layout, 0x715e_0010)],
            vec![make_onehot_poly(&layout, 0x715e_0020)],
        ];
        let opening_points_owned = [random_point(NV, 0x715e_1010), random_point(NV, 0x715e_1020)];
        let openings_by_point: Vec<Vec<F>> = point_polys
            .iter()
            .zip(opening_points_owned.iter())
            .map(|(polys, point)| {
                polys
                    .iter()
                    .map(|poly| opening_from_poly::<D, _>(poly, point, &layout))
                    .collect()
            })
            .collect();
        let polys_by_point: Vec<&[OneHotPoly<F, D, u8>]> =
            point_polys.iter().map(Vec::as_slice).collect();
        let opening_points: Vec<&[F]> = opening_points_owned.iter().map(Vec::as_slice).collect();
        let openings_by_point: Vec<&[F]> = openings_by_point.iter().map(Vec::as_slice).collect();

        let setup = <Scheme as CommitmentProver<F, D>>::setup_prover(NV, total_claims, 2);
        let verifier_setup = <Scheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let point_group_counts: Vec<usize> = point_group_sizes.iter().map(Vec::len).collect();
        let (commitments, hints) = <Scheme as CommitmentProver<F, D>>::batched_commit(
            &polys_by_point,
            &point_group_counts,
            &setup,
        )
        .expect("multipoint commit");

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/d64_multipoint");
        let proof = <Scheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_inputs_from_groups(&opening_points, &polys_by_point, &commitments, hints),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("multipoint tensor prove");
        assert!(
            proof.root.as_fold().is_some(),
            "test must exercise tensor root fold"
        );

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/d64_multipoint");
        <Scheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_inputs_from_groups(&opening_points, &openings_by_point, &commitments),
            BasisMode::Lagrange,
        )
        .expect("multipoint tensor verify");
    });
}

