//! End-to-end tests for the tensor-shaped root fold path.

#![cfg(feature = "profile-ci")]
#![allow(missing_docs)]

mod common;

use akita_config::tensor_verifier::fp128::D64OneHotTensor;
use akita_config::CommitmentConfig;
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::{ComputeBackendSetup, CpuBackend};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::AkitaBatchedProof;
use common::*;

const TENSOR_D: usize = D64OneHotTensor::D;

#[cfg(feature = "profile-ci")]
fn run_single_onehot_tensor(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = D64OneHotTensor::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let tensor_k = 256;
        let total_ring = layout.num_live_blocks * layout.num_positions_per_block;
        assert_eq!(total_ring * TENSOR_D, 1usize << nv);
        let num_onehot_chunks = (1usize << nv) / tensor_k;
        assert!(
            matches!(
                layout.fold_challenge_shape,
                akita_challenges::TensorChallengeShape::Tensor { .. }
            ),
            "D64OneHotTensor must emit a tensor-shaped root fold"
        );

        let mut rng = StdRng::seed_from_u64(0xfeed_d00d_0000 + nv as u64);
        let indices: Vec<Option<u8>> = (0..num_onehot_chunks)
            .map(|_| Some(rng.gen_range(0..tensor_k) as u8))
            .collect();
        let poly = OneHotPoly::<F, u8>::new(tensor_k, TENSOR_D, indices).expect("onehot poly");

        let pt = random_point(nv, 0xc0ff_ee00 + nv as u64);
        let expected_opening = opening_from_poly::<TENSOR_D, _>(&poly, &pt, &layout);

        let setup =
            AkitaCommitmentScheme::<D64OneHotTensor>::setup_prover(nv, 1).expect("setup_prover");
        let prepared = CpuBackend.prepare_setup(&setup).expect("prepare_setup");
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = AkitaCommitmentScheme::<D64OneHotTensor>::setup_verifier(&setup)
            .expect("verifier setup");
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            AkitaCommitmentScheme::<D64OneHotTensor>::commit::<_, _>(&setup, commit_input, &stack)
                .expect("commit");

        let poly_refs: [&OneHotPoly<F, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
        let proof = AkitaCommitmentScheme::<D64OneHotTensor>::batched_prove::<_, _, _>(
            &setup,
            prove_input::<D64OneHotTensor, _>(&pt[..], &poly_refs[..], &commitments[0], hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
        )
        .expect("prove");

        let decoded = round_trip_proof(&proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
        let result = AkitaCommitmentScheme::<D64OneHotTensor>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input::<D64OneHotTensor>(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
        );
        assert!(
            result.is_ok(),
            "onehot_tensor nv={nv} verification failed: {:?}",
            result.err()
        );
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

#[test]
#[cfg(feature = "profile-ci")]
fn single_onehot_tensor_nv15() {
    run_single_onehot_tensor(15);
}

#[test]
#[cfg(feature = "profile-ci")]
fn single_onehot_tensor_nv20() {
    run_single_onehot_tensor(20);
}

#[test]
#[cfg(feature = "profile-ci")]
fn single_onehot_tensor_nv22() {
    run_single_onehot_tensor(22);
}
