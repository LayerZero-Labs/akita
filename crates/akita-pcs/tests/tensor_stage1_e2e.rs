#![allow(missing_docs)]

mod common;

use akita_challenges::SparseChallengeConfig;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_transcript::Blake2bTranscript;
use akita_types::{
    direct_witness_bytes, AjtaiRole, AkitaRootBatchSummary, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaSchedulePlan, CommitmentEnvelope, DecompositionParams, DirectStep,
    DirectWitnessShape, Schedule, ScheduleProvider, Step,
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

impl<Base: ScheduleProvider> ScheduleProvider for TensorCfg<Base> {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        Base::schedule_table()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        Base::schedule_key(key)
    }

    fn schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Base::schedule_plan(key)
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
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        Base::planner_schedule_plan(key)
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

impl<Base> CommitmentConfig for TensorCfg<Base>
where
    Base: CommitmentConfig + akita_planner::PlannerConfig,
{
    type Field = Base::Field;
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
        Base::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
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
        Base::commitment_layout(max_num_vars)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        let mut schedule =
            Base::get_params_for_prove(max_num_vars, num_vars, layout_num_claims, batch)?;
        tensorize_root_schedule::<Base>(&mut schedule, batch)?;
        Ok(schedule)
    }
}

fn tensorize_root_schedule<Base: CommitmentConfig>(
    schedule: &mut Schedule,
    batch: AkitaRootBatchSummary,
) -> Result<(), akita_field::AkitaError> {
    let (next_w_len, log_basis) = {
        let Some(Step::Fold(root_step)) = schedule.steps.first_mut() else {
            return Ok(());
        };
        root_step.params = root_step.params.clone().with_tensor_stage1_challenges();
        root_step.delta_fold_per_poly = root_step.params.num_digits_fold;
        root_step.w_ring =
            akita_types::w_ring_element_count_with_batch_summary::<F>(&root_step.params, batch);
        root_step.next_w_len = root_step
            .w_ring
            .checked_mul(root_step.params.ring_dimension)
            .ok_or_else(|| {
                akita_field::AkitaError::InvalidSetup("tensor next-w length overflow".to_string())
            })?;
        (root_step.next_w_len, root_step.params.log_basis)
    };
    if matches!(schedule.steps.get(1), Some(Step::Fold(_))) {
        let direct = DirectStep {
            current_w_len: next_w_len,
            bits_per_elem: log_basis,
            direct_bytes: direct_witness_bytes(
                Base::decomposition().field_bits(),
                &DirectWitnessShape::PackedDigits((next_w_len, log_basis)),
            ),
        };
        schedule.steps.truncate(1);
        schedule.steps.push(Step::Direct(direct));
    } else if let Some(Step::Direct(direct)) = schedule.steps.get_mut(1) {
        direct.current_w_len = next_w_len;
    }
    Ok(())
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

#[test]
fn tensor_proof_rejects_under_flat_schedule() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type TensorScheme = AkitaCommitmentScheme<D, TensorOneHotCfg>;
        type FlatScheme = AkitaCommitmentScheme<D, OneHotCfg>;

        let layout = TensorOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x715e_f1a7);
        let pt = random_point(NV, 0x715e_f1a8);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <TensorScheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <TensorScheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <TensorScheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/tensor_as_flat");
        let proof = <TensorScheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("tensor prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/tensor_as_flat");
        let result = <FlatScheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "tensor proof must reject under flat schedule"
        );
    });
}

#[test]
fn flat_proof_rejects_under_tensor_schedule() {
    init_rayon_pool();
    let _guard = E2E_TEST_LOCK.lock().unwrap();
    run_on_large_stack(|| {
        const NV: usize = 12;
        const D: usize = ONEHOT_D;
        type TensorScheme = AkitaCommitmentScheme<D, TensorOneHotCfg>;
        type FlatScheme = AkitaCommitmentScheme<D, OneHotCfg>;

        let layout = OneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x715e_7e57);
        let pt = random_point(NV, 0x715e_7e58);
        let opening = opening_from_poly::<D, _>(&poly, &pt, &layout);
        let setup = <FlatScheme as CommitmentProver<F, D>>::setup_prover(NV, 1, 1);
        let verifier_setup = <FlatScheme as CommitmentProver<F, D>>::setup_verifier(&setup);
        let (commitment, hint) =
            <FlatScheme as CommitmentProver<F, D>>::commit(std::slice::from_ref(&poly), &setup)
                .expect("commit");
        let poly_refs = [&poly];
        let commitments = [commitment];
        let openings = [opening];
        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/flat_as_tensor");
        let proof = <FlatScheme as CommitmentProver<F, D>>::batched_prove(
            &setup,
            prove_input(&pt, &poly_refs, &commitments[0], hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("flat prove");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"tensor_stage1_e2e/flat_as_tensor");
        let result = <TensorScheme as CommitmentVerifier<F, D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt, &openings, &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_err(),
            "flat proof must reject under tensor schedule"
        );
    });
}
