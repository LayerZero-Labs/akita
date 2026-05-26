//! End-to-end tests for the **tensor-shaped** stage-1 fold path on the
//! `fast_verifier` `D64OneHotTensor` preset.
//!
//! Mirrors `single_poly_e2e.rs::run_single_onehot` but swaps the config from
//! `fp128::D64OneHot` (flat sparse fold) to
//! `fast_verifier::fp128::D64OneHotTensor` (tensor-shaped root fold + flat
//! recursive folds). The test commits one one-hot polynomial, produces a
//! batched proof, round-trips it through serialize/deserialize, and verifies.
//!
//! Coverage for `cargo test`. The runtime profile mode `onehot_d64_tensor`
//! exercises the same code path at NV=32 as a manual benchmark, but only via
//! `cargo run --release --example profile`; this test file is what `cargo
//! test` / `cargo nextest` actually runs.

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
// Same `K = D = 64` one-hot layout as the flat preset.
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
        // The whole point of the test: the root fold's challenge container is
        // the tensor-shaped variant. Recursive levels stay flat by design.
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

#[test]
fn single_onehot_tensor_nv10() {
    run_single_onehot_tensor(15);
}

#[test]
fn single_onehot_tensor_nv15() {
    run_single_onehot_tensor(20);
}

#[test]
fn single_onehot_tensor_nv20() {
    run_single_onehot_tensor(22);
}
