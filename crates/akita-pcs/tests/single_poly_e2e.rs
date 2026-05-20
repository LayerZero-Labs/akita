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
#![cfg(not(feature = "zk"))]

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::Blake2bTranscript;
use akita_types::AkitaBatchedProof;
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
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
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

// ---------------------------------------------------------------------------
// Dense helpers (D = 128)
// ---------------------------------------------------------------------------

type DenseCfg = fp128::D128Full;
const DENSE_D: usize = DenseCfg::D;

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
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
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
        let decoded = AkitaBatchedProof::<F, F>::deserialize_compressed(
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

// ---------------------------------------------------------------------------
// Tensor-shaped fold: hand-built activation via a test-only Cfg that mutates
// the production schedule's root step to TensorChallengeShape::Tensor.
// ---------------------------------------------------------------------------

mod tensor_fold {
    use super::*;
    use akita_challenges::{SparseChallengeConfig, TensorChallengeShape};
    use akita_config::CommitmentConfig;
    use akita_field::AkitaError;
    use akita_planner::PlannerConfig;
    use akita_types::generated::GeneratedScheduleTable;
    use akita_types::layout::digit_math::compute_num_digits_fold_with_claims;
    use akita_types::{
        AjtaiRole, AkitaScheduleInputs, AkitaScheduleLookupKey, AkitaSchedulePlan,
        ClaimIncidenceSummary, CommitmentEnvelope, DecompositionParams, Schedule, ScheduleProvider,
        Step,
    };

    /// Test-only config that tensorises the root fold of [`OneHotCfg`]'s
    /// production schedule. Delegates every method except `get_params_for_prove`
    /// to `OneHotCfg`; that method calls into the default impl and then walks
    /// the returned [`Schedule`], replacing the root [`Step::Fold`]'s
    /// `params.fold_challenge_shape` with [`TensorChallengeShape::Tensor`] and
    /// re-deriving the dependent layout (`num_digits_fold`,
    /// `delta_fold_per_poly`, `w_ring`, `next_w_len`, successor's
    /// `current_w_len`).
    #[derive(Clone, Copy, Debug)]
    struct TensorOneHotCfg;

    type TensorOneHotScheme = AkitaCommitmentScheme<ONEHOT_D, TensorOneHotCfg>;

    impl ScheduleProvider for TensorOneHotCfg {
        fn schedule_table() -> Option<GeneratedScheduleTable> {
            OneHotCfg::schedule_table()
        }
        fn schedule_key(key: AkitaScheduleLookupKey) -> String {
            OneHotCfg::schedule_key(key)
        }
        fn schedule_plan(
            key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
            OneHotCfg::schedule_plan(key)
        }
    }

    impl PlannerConfig for TensorOneHotCfg {
        const PLANNER_D: usize = OneHotCfg::D;
        type PlannerField = <OneHotCfg as PlannerConfig>::PlannerField;

        fn planner_field_bits() -> u32 {
            <OneHotCfg as PlannerConfig>::planner_field_bits()
        }
        fn planner_challenge_field_bits() -> u32 {
            <OneHotCfg as PlannerConfig>::planner_challenge_field_bits()
        }
        fn planner_extension_opening_width() -> usize {
            <OneHotCfg as PlannerConfig>::planner_extension_opening_width()
        }
        fn planner_sis_modulus_family() -> akita_types::SisModulusFamily {
            <OneHotCfg as PlannerConfig>::planner_sis_modulus_family()
        }
        fn planner_stage1_challenge_config(d: usize) -> SparseChallengeConfig {
            <OneHotCfg as PlannerConfig>::planner_stage1_challenge_config(d)
        }
        fn planner_schedule_plan(
            key: AkitaScheduleLookupKey,
        ) -> Result<Option<AkitaSchedulePlan>, AkitaError> {
            <OneHotCfg as PlannerConfig>::planner_schedule_plan(key)
        }
        fn planner_root_level_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            <OneHotCfg as PlannerConfig>::planner_root_level_layout_with_log_basis(
                inputs, log_basis,
            )
        }
        fn planner_current_level_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            <OneHotCfg as PlannerConfig>::planner_current_level_layout_with_log_basis(
                inputs, log_basis,
            )
        }
        fn planner_root_level_params_for_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            lp: &LevelParams,
        ) -> Result<LevelParams, AkitaError> {
            <OneHotCfg as PlannerConfig>::planner_root_level_params_for_layout_with_log_basis(
                inputs, lp,
            )
        }
        fn planner_log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
            <OneHotCfg as PlannerConfig>::planner_log_basis_search_range(inputs)
        }
    }

    impl CommitmentConfig for TensorOneHotCfg {
        type Field = F;
        type ClaimField = <OneHotCfg as CommitmentConfig>::ClaimField;
        type ChallengeField = <OneHotCfg as CommitmentConfig>::ChallengeField;
        const D: usize = ONEHOT_D;

        fn sis_modulus_family() -> akita_types::SisModulusFamily {
            <OneHotCfg as CommitmentConfig>::sis_modulus_family()
        }
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
        ) -> Result<(usize, usize), AkitaError> {
            OneHotCfg::max_setup_matrix_size(max_num_vars, max_num_batched_polys, max_num_points)
        }
        fn level_params_with_log_basis(inputs: AkitaScheduleInputs, log_basis: u32) -> LevelParams {
            OneHotCfg::level_params_with_log_basis(inputs, log_basis)
        }
        fn root_level_params_for_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            lp: &LevelParams,
        ) -> Result<LevelParams, AkitaError> {
            OneHotCfg::root_level_params_for_layout_with_log_basis(inputs, lp)
        }
        fn root_level_layout_with_log_basis(
            inputs: AkitaScheduleInputs,
            log_basis: u32,
        ) -> Result<LevelParams, AkitaError> {
            OneHotCfg::root_level_layout_with_log_basis(inputs, log_basis)
        }
        fn log_basis_at_level(inputs: AkitaScheduleInputs) -> u32 {
            OneHotCfg::log_basis_at_level(inputs)
        }
        fn log_basis_search_range(inputs: AkitaScheduleInputs) -> (u32, u32) {
            OneHotCfg::log_basis_search_range(inputs)
        }
        fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, AkitaError> {
            OneHotCfg::commitment_layout(max_num_vars)
        }

        fn get_params_for_prove(incidence: &ClaimIncidenceSummary) -> Result<Schedule, AkitaError> {
            let mut schedule = OneHotCfg::get_params_for_prove(incidence)?;
            tensorise_root_step::<F>(&mut schedule, Self::decomposition().field_bits())?;
            Ok(schedule)
        }
    }

    /// Flip the root fold step's `fold_challenge_shape` to Tensor and
    /// re-derive every layout field that depends on the (now wider) effective
    /// L1 mass: per-poly fold-digit count, ring width, next-witness length,
    /// and the successor step's `current_w_len`. Singleton-incidence
    /// schedules only (the test path), so `num_claims = 1` everywhere.
    fn tensorise_root_step<FF: FieldCore + CanonicalField>(
        schedule: &mut Schedule,
        field_bits: u32,
    ) -> Result<(), AkitaError> {
        let next_w_len = {
            let Some(Step::Fold(root_step)) = schedule.steps.first_mut() else {
                return Ok(());
            };
            if !root_step.params.num_blocks.is_power_of_two() {
                return Err(AkitaError::InvalidSetup(
                    "tensor fold shape requires a power-of-two num_blocks at the root".to_string(),
                ));
            }
            root_step.params = root_step
                .params
                .clone()
                .with_fold_challenge_shape(TensorChallengeShape::Tensor);
            // `with_fold_challenge_shape` only flips the shape; the wider
            // effective L1 mass forces a fresh fold-digit count.
            root_step.params.num_digits_fold = compute_num_digits_fold_with_claims(
                root_step.params.r_vars,
                root_step.params.challenge_l1_mass(),
                root_step.params.log_basis,
                1,
                field_bits,
            );
            root_step.delta_fold_per_poly = root_step.params.num_digits_fold;
            root_step.w_ring =
                akita_types::w_ring_element_count_with_counts::<FF>(&root_step.params, 1, 1, 1, 1)?;
            root_step
                .w_ring
                .checked_mul(root_step.params.ring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("tensor next-w length overflow".to_string())
                })?
        };

        if let Some(Step::Fold(root_step)) = schedule.steps.first_mut() {
            root_step.next_w_len = next_w_len;
        }

        match schedule.steps.get_mut(1) {
            Some(Step::Direct(direct)) => {
                direct.current_w_len = next_w_len;
                // The witness-shape envelope embeds the same length; keep it
                // consistent with `current_w_len` so the prover's logical
                // witness check doesn't trip on a stale shape.
                direct.witness_shape = match direct.witness_shape {
                    akita_types::DirectWitnessShape::FieldElements(_) => {
                        akita_types::DirectWitnessShape::FieldElements(next_w_len)
                    }
                    akita_types::DirectWitnessShape::PackedDigits((_, bits)) => {
                        akita_types::DirectWitnessShape::PackedDigits((next_w_len, bits))
                    }
                };
                Ok(())
            }
            Some(Step::Fold(_)) => Err(AkitaError::InvalidSetup(
                "tensor activation test expects root fold to hand off to a direct step".to_string(),
            )),
            None => Err(AkitaError::InvalidSetup(
                "tensor activation test schedule missing root successor".to_string(),
            )),
        }
    }

    /// Drive the full prove/verify round-trip with the tensor-shaped root
    /// fold and assert acceptance.
    #[test]
    fn onehot_tensor_fold_prove_verify() {
        const NV: usize = 12;
        init_rayon_pool();
        run_on_large_stack(|| {
            let layout = TensorOneHotCfg::commitment_layout(NV).expect("layout");
            let poly = make_onehot_poly(&layout, 0x715e_0000 + NV as u64);
            let pt = random_point(NV, 0x715e_f00d);
            let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

            let setup =
                <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(NV, 1, 1);
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

            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"single_poly_e2e/tensor_onehot");
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

            let root = proof.root.as_fold().expect("tensor test must fold root");
            assert!(
                !root.stage1.stages.is_empty(),
                "tensor test must exercise the stage-1 fold"
            );

            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"single_poly_e2e/tensor_onehot");
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

    /// Negative-path companion: tampering with the prover's stage-1 fold
    /// message after the proof has been built must cause the verifier to
    /// reject. Guards against transcript-binding regressions on the new
    /// tensor sampling labels.
    #[test]
    fn onehot_tensor_fold_rejects_tampered_proof() {
        const NV: usize = 12;
        init_rayon_pool();
        run_on_large_stack(|| {
            let layout = TensorOneHotCfg::commitment_layout(NV).expect("layout");
            let poly = make_onehot_poly(&layout, 0xfa11_0000 + NV as u64);
            let pt = random_point(NV, 0xfa11_f00d);
            let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

            let setup =
                <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(NV, 1, 1);
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

            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"single_poly_e2e/tensor_onehot_tampered");
            let mut proof = <TensorOneHotScheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
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

            // Flip the stage-1 final `s_claim`. The verifier absorbs it
            // before sampling the stage-2 batching coefficient, so any
            // tamper perturbs the reconstructed stage-2 input claim and the
            // sumcheck check must fail.
            let root = proof
                .root
                .as_fold_mut()
                .expect("tensor test must fold root");
            root.stage1.s_claim += F::one();

            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"single_poly_e2e/tensor_onehot_tampered");
            let result = <TensorOneHotScheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            );
            assert!(
                result.is_err(),
                "verifier must reject tampered tensor stage-1 v"
            );
        });
    }
}

// ---------------------------------------------------------------------------
// Planner-driven tensor stage-1 fold preset: D64OneHotTensor.
//
// `D64OneHotTensor` is the public preset whose `fold_challenge_shape_at_level`
// returns `Tensor` at the root level. Unlike `tensor_fold::TensorOneHotCfg`,
// no post-planner schedule mutation is involved: the planner consults the
// selector through `LevelParams::challenge_l1_mass`, sizes `num_digits_fold`
// for the wider tensor envelope, and bakes `TensorChallengeShape::Tensor` into
// the root fold step's `params.fold_challenge_shape`.
// ---------------------------------------------------------------------------

mod planner_tensor_fold {
    use super::*;
    use akita_challenges::TensorChallengeShape;
    use akita_config::proof_optimized::fp128::D64OneHotTensor;
    use akita_config::CommitmentConfig;
    use akita_types::{AkitaScheduleInputs, ClaimIncidenceSummary, Step};

    type TensorPresetScheme = AkitaCommitmentScheme<ONEHOT_D, D64OneHotTensor>;

    fn planned_root_uses_tensor_shape(nv: usize) {
        let incidence = ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence");
        let schedule = D64OneHotTensor::get_params_for_prove(&incidence).expect("prove schedule");
        let Some(Step::Fold(root)) = schedule.steps.first() else {
            panic!("D64OneHotTensor schedule must start with a fold step");
        };
        assert_eq!(
            root.params.fold_challenge_shape,
            TensorChallengeShape::Tensor,
            "planner must bake the tensor shape into the root fold step"
        );
        assert!(
            root.params.num_blocks.is_power_of_two(),
            "tensor sampler requires a power-of-two num_blocks at the root"
        );

        let inputs = AkitaScheduleInputs {
            num_vars: nv,
            level: 0,
            current_w_len: 1usize << nv,
        };
        let log_basis = D64OneHotTensor::log_basis_at_level(inputs);
        let flat_baseline =
            <OneHotCfg as CommitmentConfig>::root_level_layout_with_log_basis(inputs, log_basis)
                .expect("flat baseline layout");
        assert!(
            root.params.num_digits_fold >= flat_baseline.num_digits_fold,
            "tensor envelope must require at least as many fold digits as the flat envelope"
        );
    }

    #[test]
    fn planner_routes_tensor_shape_through_root_fold_step() {
        for nv in [12, 20] {
            planned_root_uses_tensor_shape(nv);
        }
    }

    fn run_prove_verify(nv: usize, label: Vec<u8>) {
        init_rayon_pool();
        run_on_large_stack(move || {
            let layout = D64OneHotTensor::commitment_layout(nv).expect("layout");
            assert_eq!(
                layout.fold_challenge_shape,
                TensorChallengeShape::Tensor,
                "commitment_layout must surface the tensor root shape"
            );
            let poly = make_onehot_poly(&layout, 0x1357_0000 + nv as u64);
            let pt = random_point(nv, 0x1357_f00d + nv as u64);
            let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

            let setup =
                <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(nv, 1, 1);
            let verifier_setup =
                <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
            let (commitment, hint) = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::commit(
                std::slice::from_ref(&poly),
                &setup,
            )
            .expect("commit");

            let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
            let commitments = [commitment];
            let openings = [expected_opening];
            let opening_groups = [&openings[..]];
            let hints = vec![hint];

            let mut prover_transcript = Blake2bTranscript::<F>::new(&label);
            let proof = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
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

            let root = proof.root.as_fold().expect("preset must fold root");
            assert!(
                !root.stage1.stages.is_empty(),
                "preset must exercise the stage-1 fold"
            );

            let mut verifier_transcript = Blake2bTranscript::<F>::new(&label);
            <TensorPresetScheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            )
            .expect("verify");
        });
    }

    #[test]
    fn d64_onehot_tensor_prove_verify_nv20() {
        run_prove_verify(20, b"single_poly_e2e/d64_onehot_tensor_nv20".to_vec());
    }

    #[test]
    fn d64_onehot_tensor_rejects_tampered_proof() {
        const NV: usize = 12;
        init_rayon_pool();
        run_on_large_stack(|| {
            let layout = D64OneHotTensor::commitment_layout(NV).expect("layout");
            let poly = make_onehot_poly(&layout, 0xb1a5_0000 + NV as u64);
            let pt = random_point(NV, 0xb1a5_f00d);
            let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

            let setup =
                <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_prover(NV, 1, 1);
            let verifier_setup =
                <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::setup_verifier(&setup);
            let (commitment, hint) = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::commit(
                std::slice::from_ref(&poly),
                &setup,
            )
            .expect("commit");

            let poly_refs: [&OneHotPoly<F, ONEHOT_D, u8>; 1] = [&poly];
            let commitments = [commitment];
            let openings = [expected_opening];
            let opening_groups = [&openings[..]];
            let hints = vec![hint];

            let mut prover_transcript =
                Blake2bTranscript::<F>::new(b"single_poly_e2e/d64_onehot_tensor_tampered");
            let mut proof = <TensorPresetScheme as CommitmentProver<F, ONEHOT_D>>::batched_prove(
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

            // Tampering with the absorbed stage-1 `s_claim` after the proof
            // is built breaks the verifier's stage-2 reconstruction. The new
            // tensor sampling labels are bound through the same transcript,
            // so the same negative guard exercises label propagation.
            let root = proof.root.as_fold_mut().expect("preset must fold root");
            root.stage1.s_claim += F::one();

            let mut verifier_transcript =
                Blake2bTranscript::<F>::new(b"single_poly_e2e/d64_onehot_tensor_tampered");
            let result = <TensorPresetScheme as CommitmentVerifier<F, ONEHOT_D>>::batched_verify(
                &proof,
                &verifier_setup,
                &mut verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
            );
            assert!(
                result.is_err(),
                "verifier must reject tampered tensor stage-1 v on the production preset"
            );
        });
    }
}
