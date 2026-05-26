//! End-to-end tests for the tensor-shaped root fold path.

#![allow(missing_docs)]
#![cfg(not(feature = "zk"))]

mod common;

use akita_config::fast_verifier::fp128::D64OneHotTensor;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::CommitmentProver;
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
        >>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::commit(
                commit_input,
                &setup,
            )
            .expect("commit");

        let poly_refs: [&OneHotPoly<F, TENSOR_D, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
        let proof = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
            F,
            TENSOR_D,
        >>::batched_prove(
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
        );
        assert!(
            result.is_ok(),
            "onehot_tensor nv={nv} verification failed: {:?}",
            result.err()
        );
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
        >>::setup_prover(nv, 1, 1);
        let verifier_setup =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<F, TENSOR_D>>::commit(
                commit_input,
                &setup,
            )
            .expect("commit");

        let poly_refs: [&DensePoly<F, TENSOR_D>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
        let proof = <AkitaCommitmentScheme<TENSOR_D, D64OneHotTensor> as CommitmentProver<
            F,
            TENSOR_D,
        >>::batched_prove(
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
        );
        assert!(
            result.is_ok(),
            "dense_tensor nv={nv} verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn single_onehot_tensor_nv15() {
    run_single_onehot_tensor(15);
}

#[test]
fn single_onehot_tensor_nv20() {
    run_single_onehot_tensor(20);
}

#[test]
fn single_onehot_tensor_nv22() {
    run_single_onehot_tensor(22);
}

#[test]
fn single_dense_tensor_nv15() {
    run_single_dense_tensor(15);
}

#[test]
fn single_dense_tensor_nv20() {
    run_single_dense_tensor(20);
}

#[test]
fn single_dense_tensor_nv22() {
    run_single_dense_tensor(22);
}
