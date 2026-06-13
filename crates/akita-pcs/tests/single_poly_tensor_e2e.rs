//! End-to-end tests for the tensor-shaped root fold path.

#![allow(missing_docs)]

mod common;

use akita_config::tensor_verifier::fp128::D64OneHotTensor;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{CommitmentProver, ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::AkitaBatchedProof;
use akita_verifier::CommitmentVerifier;
use common::*;

const TENSOR_D: usize = D64OneHotTensor::D;
const TENSOR_K: usize = TENSOR_D;

fn run_single_onehot_tensor(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = D64OneHotTensor::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
        )
        .expect("layout");
        let total_ring = layout.num_blocks * layout.block_len;
        assert_eq!(total_ring * TENSOR_K, 1usize << nv);
        assert_eq!(
            layout.fold_challenge_shape,
            akita_challenges::TensorChallengeShape::Tensor,
            "D64OneHotTensor must emit a tensor-shaped root fold"
        );

        let mut rng = StdRng::seed_from_u64(0xfeed_d00d_0000 + nv as u64);
        let indices: Vec<Option<u8>> = (0..total_ring)
            .map(|_| Some(rng.gen_range(0..TENSOR_K) as u8))
            .collect();
        let poly = OneHotPoly::<F, TENSOR_D, u8>::new(TENSOR_K, indices).expect("onehot poly");

        let pt = random_point(nv, 0xc0ff_ee00 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
            F,
            TENSOR_D,
        >>::setup_prover(nv, 1, 1)
        .expect("setup_prover");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare_setup");
        let verifier_setup =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::commit(
                &setup,
                &CpuBackend,
                &prepared,
                commit_input,
            )
            .expect("commit");

        let poly_refs: [&OneHotPoly<F, TENSOR_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
        let prove_hint = {
            #[cfg(feature = "zk")]
            {
                hint.clone()
            }
            #[cfg(not(feature = "zk"))]
            {
                hint
            }
        };
        let proof = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
            F,
            TENSOR_D,
        >>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&pt[..], &poly_refs[..], &commitments[0], prove_hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");
        assert_zk_tensor_root_proof_shape(&proof);

        #[cfg(feature = "zk")]
        let second_proof = {
            let mut second_prover_transcript =
                AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
                F,
                TENSOR_D,
            >>::batched_prove(
                &setup,
                &CpuBackend,
                &prepared,
                prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
                &mut second_prover_transcript,
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            )
            .expect("second prove")
        };
        #[cfg(feature = "zk")]
        assert_zk_tensor_root_hiding(&proof, &second_proof);

        let decoded = round_trip_proof(&proof);
        #[cfg(feature = "zk")]
        let second_decoded = round_trip_proof(&second_proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
        let result = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentVerifier<
            F,
            TENSOR_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "onehot_tensor nv={nv} verification failed: {:?}",
            result.err()
        );
        #[cfg(feature = "zk")]
        {
            let mut bad_openings = openings;
            bad_openings[0] += F::one();
            let mut bad_verifier_transcript =
                AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
            let bad_result =
                <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentVerifier<
                    F,
                    TENSOR_D,
                >>::batched_verify(
                    &decoded,
                    &verifier_setup,
                    &mut bad_verifier_transcript,
                    verify_input(&pt[..], &bad_openings[..], &commitments[0]),
                    BasisMode::Lagrange,
                    akita_types::SetupContributionMode::Direct,
                );
            assert!(
                bad_result.is_err(),
                "ZK tensor proof should reject a wrong onehot opening"
            );
        }
        #[cfg(feature = "zk")]
        {
            let mut second_verifier_transcript =
                AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentVerifier<
                F,
                TENSOR_D,
            >>::batched_verify(
                &second_decoded,
                &verifier_setup,
                &mut second_verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            )
            .expect("second onehot tensor verify");
        }
    });
}

fn run_single_dense_tensor(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = D64OneHotTensor::get_params_for_batched_commitment(
            &akita_types::ClaimIncidenceSummary::same_point(nv, 1).expect("singleton incidence"),
        )
        .expect("layout");
        let total_ring = layout.num_blocks * layout.block_len;
        assert_eq!(total_ring * TENSOR_D, 1usize << nv);
        assert_eq!(
            layout.fold_challenge_shape,
            akita_challenges::TensorChallengeShape::Tensor,
            "D64OneHotTensor must emit a tensor-shaped root fold"
        );

        let mut rng = StdRng::seed_from_u64(0xd3e5_7000 + nv as u64);
        let evals: Vec<F> = (0..1usize << nv)
            .map(|_| F::from_canonical_u128_reduced(rng.gen_range(0..=1)))
            .collect();
        let poly = DensePoly::<F, TENSOR_D>::from_field_evals(nv, &evals).expect("dense poly");

        let pt = random_point(nv, 0xd3e5_f00d + nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
            F,
            TENSOR_D,
        >>::setup_prover(nv, 1, 1)
        .expect("setup_prover");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare_setup");
        let verifier_setup =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::commit(
                &setup,
                &CpuBackend,
                &prepared,
                commit_input,
            )
            .expect("commit");

        let poly_refs: [&DensePoly<F, TENSOR_D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
        let prove_hint = {
            #[cfg(feature = "zk")]
            {
                hint.clone()
            }
            #[cfg(not(feature = "zk"))]
            {
                hint
            }
        };
        let proof = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
            F,
            TENSOR_D,
        >>::batched_prove(
            &setup,
            &CpuBackend,
            &prepared,
            prove_input(&pt[..], &poly_refs[..], &commitments[0], prove_hint),
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");
        assert_zk_tensor_root_proof_shape(&proof);

        #[cfg(feature = "zk")]
        let second_proof = {
            let mut second_prover_transcript =
                AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
                F,
                TENSOR_D,
            >>::batched_prove(
                &setup,
                &CpuBackend,
                &prepared,
                prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
                &mut second_prover_transcript,
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            )
            .expect("second prove")
        };
        #[cfg(feature = "zk")]
        assert_zk_tensor_root_hiding(&proof, &second_proof);

        let decoded = round_trip_proof(&proof);
        #[cfg(feature = "zk")]
        let second_decoded = round_trip_proof(&second_proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
        let result = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentVerifier<
            F,
            TENSOR_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "dense_tensor nv={nv} verification failed: {:?}",
            result.err()
        );
        #[cfg(feature = "zk")]
        {
            let mut bad_openings = openings;
            bad_openings[0] += F::one();
            let mut bad_verifier_transcript =
                AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
            let bad_result =
                <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentVerifier<
                    F,
                    TENSOR_D,
                >>::batched_verify(
                    &decoded,
                    &verifier_setup,
                    &mut bad_verifier_transcript,
                    verify_input(&pt[..], &bad_openings[..], &commitments[0]),
                    BasisMode::Lagrange,
                    akita_types::SetupContributionMode::Direct,
                );
            assert!(
                bad_result.is_err(),
                "ZK tensor proof should reject a wrong dense opening"
            );
        }
        #[cfg(feature = "zk")]
        {
            let mut second_verifier_transcript =
                AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentVerifier<
                F,
                TENSOR_D,
            >>::batched_verify(
                &second_decoded,
                &verifier_setup,
                &mut second_verifier_transcript,
                verify_input(&pt[..], opening_groups[0], &commitments[0]),
                BasisMode::Lagrange,
                akita_types::SetupContributionMode::Direct,
            )
            .expect("second dense tensor verify");
        }
    });
}

fn round_trip_proof(proof: &AkitaBatchedProof<F, F>) -> AkitaBatchedProof<F, F> {
    let mut serialized = Vec::new();
    let proof_shape = proof.shape();
    proof
        .serialize_compressed(&mut serialized)
        .expect("serialize");
    AkitaBatchedProof::<F, F>::deserialize_compressed(
        &mut std::io::Cursor::new(serialized),
        &proof_shape,
    )
    .expect("deserialize")
}

#[cfg(feature = "zk")]
fn assert_zk_tensor_root_proof_shape(proof: &AkitaBatchedProof<F, F>) {
    let root = proof
        .root
        .as_fold()
        .expect("tensor fixture should use a folded root");
    assert!(
        root.stage1
            .stages
            .iter()
            .all(|stage| !stage.sumcheck_proof_masked.masked_round_polys.is_empty()),
        "ZK tensor root must carry masked stage-1 sumcheck rounds"
    );
    assert!(
        !root.stage2.sumcheck_masked().masked_round_polys.is_empty(),
        "ZK tensor root must carry masked stage-2 sumcheck rounds"
    );
}

#[cfg(feature = "zk")]
fn assert_zk_tensor_root_hiding(
    proof: &AkitaBatchedProof<F, F>,
    second_proof: &AkitaBatchedProof<F, F>,
) {
    let root = proof
        .root
        .as_fold()
        .expect("tensor fixture should use a folded root");
    let second_root = second_proof
        .root
        .as_fold()
        .expect("tensor fixture should use a folded root");
    assert_ne!(
        root.v, second_root.v,
        "ZK tensor root should re-randomize v for the same witness"
    );
    assert_ne!(
        root.stage1.stages[0]
            .sumcheck_proof_masked
            .masked_round_polys,
        second_root.stage1.stages[0]
            .sumcheck_proof_masked
            .masked_round_polys,
        "ZK tensor root should re-randomize masked stage-1 rounds"
    );
    assert_ne!(
        root.stage2
            .as_intermediate()
            .expect("fold root proof must carry intermediate stage-2 proof")
            .sumcheck_proof_masked
            .masked_round_polys,
        second_root
            .stage2
            .as_intermediate()
            .expect("fold root proof must carry intermediate stage-2 proof")
            .sumcheck_proof_masked
            .masked_round_polys,
        "ZK tensor root should re-randomize masked stage-2 rounds"
    );
}

#[cfg(not(feature = "zk"))]
fn assert_zk_tensor_root_proof_shape(_proof: &AkitaBatchedProof<F, F>) {}

// Deferred (D128-tensor follow-up): the tensor fold challenge applies an `ω²`
// factor to the effective challenge L1 mass, and under the safe
// `onehot_chunk_size = 1` default (`nonzeros = D`) the A-role collision pushes
// the D64 tensor family past its secure threshold, so every level degrades to
// cleartext and no tensor-shaped root fold is emitted. Re-enable once the tensor
// family is migrated to D=128.
#[cfg(not(feature = "zk"))]
#[test]
#[ignore = "D64 one-hot tensor degrades to cleartext under onehot_chunk_size=1; pending D128 tensor migration"]
fn single_onehot_tensor_nv15() {
    run_single_onehot_tensor(15);
}

#[cfg(not(feature = "zk"))]
#[test]
#[ignore = "D64 one-hot tensor degrades to cleartext under onehot_chunk_size=1; pending D128 tensor migration"]
fn single_onehot_tensor_nv20() {
    run_single_onehot_tensor(20);
}

#[cfg(not(feature = "zk"))]
#[test]
#[ignore = "D64 one-hot tensor degrades to cleartext under onehot_chunk_size=1; pending D128 tensor migration"]
fn single_onehot_tensor_nv22() {
    run_single_onehot_tensor(22);
}

// Deferred: `D64OneHotTensor` has `log_commit_bound == 1`, so the corrected
// folded-witness bound `β` sizes against one-hot witness sparsity
// (`||s||_inf = 1`). Committing a *dense* poly under this one-hot tensor config
// folds to a larger `||z||_inf` than that `β`, so the prover aborts. Tracked as
// a follow-up to the weak-binding-norm fix (tensor + dense witness interaction).
#[cfg(not(feature = "zk"))]
#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn single_dense_tensor_nv15() {
    run_single_dense_tensor(15);
}

#[cfg(not(feature = "zk"))]
#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn single_dense_tensor_nv20() {
    run_single_dense_tensor(20);
}

#[cfg(not(feature = "zk"))]
#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn single_dense_tensor_nv22() {
    run_single_dense_tensor(22);
}

#[cfg(feature = "zk")]
#[test]
#[ignore = "D64 one-hot tensor degrades to cleartext under onehot_chunk_size=1; pending D128 tensor migration"]
fn zk_single_onehot_tensor_nv20() {
    run_single_onehot_tensor(20);
}

#[cfg(feature = "zk")]
#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn zk_single_dense_tensor_nv20() {
    run_single_dense_tensor(20);
}
