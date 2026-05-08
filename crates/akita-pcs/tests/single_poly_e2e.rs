//! End-to-end tests for the **single-polynomial** (non-batched) commitment path.
//!
//! Each test commits to one polynomial, produces an opening proof, round-trips
//! the proof through serialization/deserialization, and verifies the result.
//!
//! Two polynomial representations are covered:
//!
//! * **One-hot** — `fp128::D64OneHot` (D = 64, K = D).
//! * **Dense**   — `fp128::D128Full`   (D = 128, full-field coefficients).
//!
//! Variable counts: 10, 15, 20, 25 for each representation (8 tests total).

#![allow(missing_docs)]

mod common;

use akita_challenges::SparseChallengeConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::AkitaBatchedProof;
use akita_types::{
    w_ring_element_count_with_batch_summary, AjtaiRole, AkitaRootBatchSummary, AkitaScheduleInputs,
    AkitaScheduleLookupKey, AkitaSchedulePlan, CommitmentEnvelope, DecompositionParams, Schedule,
    ScheduleProvider, Step,
};
use akita_verifier::CommitmentVerifier;
use common::*;

fn run_single_onehot(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = OneHotCfg::commitment_layout(nv).expect("layout");
        let total_ring = layout.num_blocks * layout.block_len;
        assert_eq!(total_ring * ONEHOT_K, 1usize << nv);

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + nv as u64);
        let indices: Vec<Option<u8>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly = OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices).expect("onehot poly");

        let pt = random_point(nv, 0xcafe_0000 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(nv, 1, 1);
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot");
        let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::batched_prove(
            &setup,
            prove_input(&pt[..], &poly_refs[..], &commitments[0], hints.into_iter().next().unwrap()),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot");
        let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "onehot nv={nv} verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn onehot_tensor_stage1_prove_verify() {
    const NV: usize = 12;
    init_rayon_pool();
    run_on_large_stack(|| {
        let layout = TensorOneHotCfg::commitment_layout(NV).expect("layout");
        let poly = make_onehot_poly(&layout, 0x715e_0000 + NV as u64);
        let pt = random_point(NV, 0x715e_f00d);
        let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

        let setup = <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(NV, 1, 1);
        let verifier_setup =
            <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
        let (commitment, hint) = <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::commit(
            std::slice::from_ref(&poly),
            &setup,
        )
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/tensor_onehot");
        let proof = <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let root = proof.root.as_fold().expect("tensor test should fold root");
        assert!(
            !root.stage1.stages.is_empty(),
            "tensor test should exercise stage-1 folding"
        );

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/tensor_onehot");
        <TensorOneHotScheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
            &proof,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        )
        .expect("verify");
    });
}

// ---------------------------------------------------------------------------
// Dense helpers (D = 128)
// ---------------------------------------------------------------------------

type DenseCfg = fp128::D128Full;
const DENSE_D: usize = DenseCfg::D;

#[derive(Clone, Copy, Debug)]
struct TensorOneHotCfg;

type TensorOneHotScheme = AkitaCommitmentScheme<ONEHOT_D, TensorOneHotCfg>;

impl ScheduleProvider for TensorOneHotCfg {
    fn schedule_table() -> Option<akita_types::generated::GeneratedScheduleTable> {
        OneHotCfg::schedule_table()
    }

    fn schedule_key(key: AkitaScheduleLookupKey) -> String {
        OneHotCfg::schedule_key(key)
    }

    fn schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        OneHotCfg::schedule_plan(key)
    }
}

impl akita_planner::PlannerConfig for TensorOneHotCfg {
    const PLANNER_D: usize = OneHotCfg::D;

    fn planner_field_bits() -> u32 {
        <OneHotCfg as akita_planner::PlannerConfig>::planner_field_bits()
    }

    fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        <OneHotCfg as akita_planner::PlannerConfig>::planner_stage1_challenge_config(d)
    }

    fn planner_schedule_plan(
        key: AkitaScheduleLookupKey,
    ) -> Result<Option<AkitaSchedulePlan>, akita_field::AkitaError> {
        <OneHotCfg as akita_planner::PlannerConfig>::planner_schedule_plan(key)
    }

    fn planner_root_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        <OneHotCfg as akita_planner::PlannerConfig>::planner_root_level_layout_with_log_basis(
            inputs, log_basis,
        )
    }

    fn planner_current_level_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        log_basis: u32,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        <OneHotCfg as akita_planner::PlannerConfig>::planner_current_level_layout_with_log_basis(
            inputs, log_basis,
        )
    }

    fn planner_root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        let params =
            <OneHotCfg as akita_planner::PlannerConfig>::planner_root_level_params_for_layout_with_log_basis(
                inputs, lp,
            )?;
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
        <OneHotCfg as akita_planner::PlannerConfig>::planner_log_basis_search_range(inputs)
    }

    fn planner_stage1_prover_weight() -> usize {
        <OneHotCfg as akita_planner::PlannerConfig>::planner_stage1_prover_weight()
    }
}

impl CommitmentConfig for TensorOneHotCfg {
    type Field = F;
    type ClaimField = <OneHotCfg as CommitmentConfig>::ClaimField;
    type ChallengeField = <OneHotCfg as CommitmentConfig>::ChallengeField;
    const D: usize = ONEHOT_D;

    fn decomposition() -> DecompositionParams {
        OneHotCfg::decomposition()
    }

    fn stage1_challenge_config(d: usize) -> SparseChallengeConfig {
        OneHotCfg::stage1_challenge_config(d)
    }

    fn audited_root_rank(role: AjtaiRole, max_num_vars: usize) -> usize {
        OneHotCfg::audited_root_rank(role, max_num_vars)
    }

    fn envelope(max_num_vars: usize) -> CommitmentEnvelope {
        OneHotCfg::envelope(max_num_vars)
    }

    fn max_setup_matrix_size(
        max_num_vars: usize,
        max_num_batched_polys: usize,
        max_num_points: usize,
    ) -> Result<(usize, usize), akita_field::AkitaError> {
        OneHotCfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
    }

    fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
        OneHotCfg::level_params_with_log_basis(inputs, log_basis)
    }

    fn root_level_params_for_layout_with_log_basis(
        inputs: AkitaScheduleInputs,
        lp: &LevelParams,
    ) -> Result<LevelParams, akita_field::AkitaError> {
        let params = OneHotCfg::root_level_params_for_layout_with_log_basis(inputs, lp)?;
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
    ) -> Result<LevelParams, akita_field::AkitaError> {
        OneHotCfg::root_level_layout_with_log_basis(inputs, log_basis)
    }

    fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
        OneHotCfg::log_basis_at_level(inputs)
    }

    fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
        OneHotCfg::log_basis_search_range(inputs)
    }

    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, akita_field::AkitaError> {
        OneHotCfg::commitment_layout(max_num_vars)
    }

    fn get_params_for_prove(
        max_num_vars: usize,
        num_vars: usize,
        layout_num_claims: usize,
        batch: AkitaRootBatchSummary,
    ) -> Result<Schedule, akita_field::AkitaError> {
        let mut schedule =
            OneHotCfg::get_params_for_prove(max_num_vars, num_vars, layout_num_claims, batch)?;
        tensorize_root_schedule(&mut schedule, batch)?;
        Ok(schedule)
    }
}

fn tensorize_root_schedule(
    schedule: &mut Schedule,
    batch: AkitaRootBatchSummary,
) -> Result<(), akita_field::AkitaError> {
    let next_w_len = {
        let Some(Step::Fold(root_step)) = schedule.steps.first_mut() else {
            return Ok(());
        };
        root_step.params = root_step.params.clone().with_tensor_stage1_challenges();
        root_step.delta_fold_per_poly = root_step.params.num_digits_fold;
        root_step.w_ring = w_ring_element_count_with_batch_summary::<F>(&root_step.params, batch);
        root_step
            .w_ring
            .checked_mul(root_step.params.ring_dimension)
            .ok_or_else(|| {
                akita_field::AkitaError::InvalidSetup("tensor next-w length overflow".to_string())
            })?
    };

    if let Some(Step::Fold(root_step)) = schedule.steps.first_mut() {
        root_step.next_w_len = next_w_len;
    }

    match schedule.steps.get_mut(1) {
        Some(Step::Direct(direct)) => {
            direct.current_w_len = next_w_len;
            Ok(())
        }
        Some(Step::Fold(_)) => Err(akita_field::AkitaError::InvalidSetup(
            "tensor E2E test expects root fold to hand off to direct suffix".to_string(),
        )),
        None => Err(akita_field::AkitaError::InvalidSetup(
            "tensor E2E test schedule missing root successor".to_string(),
        )),
    }
}

fn run_single_dense(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = DenseCfg::commitment_layout(nv).expect("layout");

        let mut rng = StdRng::seed_from_u64(0xface_feed_0000 + nv as u64);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
            .collect();
        let poly = DensePoly::<F, DENSE_D>::from_field_evals(nv, &evals).expect("dense poly");

        let pt = random_point(nv, 0xbabe_0000 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup =
            <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<F, DENSE_D>>::setup_prover(nv, 1, 1);
        let verifier_setup = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let poly_refs: [&DensePoly<F, DENSE_D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/dense");
        let proof = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentProver<
            F,
            DENSE_D,
        >>::batched_prove(
            &setup,
            prove_input(&pt[..], &poly_refs[..], &commitments[0], hints.into_iter().next().unwrap()),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/dense");
        let result = <AkitaCommitmentScheme<DENSE_D, DenseCfg> as CommitmentVerifier<
            F,
            DENSE_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "dense nv={nv} verification failed: {:?}",
            result.err()
        );
    });
}

// ---------------------------------------------------------------------------
// One-hot single-poly tests
// ---------------------------------------------------------------------------

#[test]
fn single_onehot_nv10() {
    run_single_onehot(10);
}

#[test]
fn single_onehot_nv15() {
    run_single_onehot(15);
}

#[test]
fn single_onehot_nv20() {
    run_single_onehot(20);
}

// #[test]
// fn single_onehot_nv25() {
//     run_single_onehot(25);
// }

// ---------------------------------------------------------------------------
// Dense single-poly tests
// ---------------------------------------------------------------------------

#[test]
fn single_dense_nv10() {
    run_single_dense(10);
}

#[test]
fn single_dense_nv15() {
    run_single_dense(15);
}

#[test]
fn single_dense_nv20() {
    run_single_dense(20);
}

// #[test]
// fn single_dense_nv25() {
//     run_single_dense(25);
// }

// ---------------------------------------------------------------------------
// Oversized setup: setup with max_num_vars > actual polynomial num_vars
// ---------------------------------------------------------------------------

#[cfg(feature = "planner")]
fn run_single_onehot_oversized_setup(setup_nv: usize, poly_nv: usize) {
    assert!(setup_nv >= poly_nv);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = OneHotCfg::commitment_layout(poly_nv).expect("layout");
        let total_ring = layout.num_blocks * layout.block_len;
        assert_eq!(total_ring * ONEHOT_K, 1usize << poly_nv);

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + poly_nv as u64);
        let indices: Vec<Option<u8>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly = OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices).expect("onehot poly");

        let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup =
            <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<F, ONEHOT_D>>::setup_prover(setup_nv, 1, 1);
        let verifier_setup = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::commit(commit_input, &setup)
        .expect("commit with oversized setup");

        let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot_oversized");
        let proof = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentProver<
            F,
            ONEHOT_D,
        >>::batched_prove(
            &setup,
            prove_input(&pt[..], &poly_refs[..], &commitments[0], hints.into_iter().next().unwrap()),
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove with oversized setup");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = AkitaBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot_oversized");
        let result = <AkitaCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentVerifier<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "onehot oversized setup (setup_nv={setup_nv}, poly_nv={poly_nv}) verification failed: {:?}",
            result.err()
        );
    });
}

#[cfg(feature = "planner")]
#[test]
fn single_onehot_oversized_setup_15_10() {
    run_single_onehot_oversized_setup(15, 10);
}

#[cfg(feature = "planner")]
#[test]
fn single_onehot_oversized_setup_20_15() {
    run_single_onehot_oversized_setup(20, 15);
}
