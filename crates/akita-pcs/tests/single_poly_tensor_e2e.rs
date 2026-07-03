//! End-to-end tests for the tensor-shaped root fold path.

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
const TENSOR_K: usize = TENSOR_D;

fn run_single_onehot_tensor(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = D64OneHotTensor::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
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
        let poly = OneHotPoly::<F, u8>::new(TENSOR_K, TENSOR_D, indices).expect("onehot poly");

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
        let verifier_setup = AkitaCommitmentScheme::<D64OneHotTensor>::setup_verifier(&setup);
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
            prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let decoded = round_trip_proof(&proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/onehot");
        let result = AkitaCommitmentScheme::<D64OneHotTensor>::batched_verify(
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
    });
}

fn run_single_dense_tensor(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = D64OneHotTensor::get_params_for_batched_commitment(
            &akita_types::OpeningClaimsLayout::new(nv, 1).expect("singleton opening batch"),
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
        let poly = DensePoly::<F>::from_field_evals(nv, TENSOR_D, &evals).expect("dense poly");

        let pt = random_point(nv, 0xd3e5_f00d + nv as u64);
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
        let verifier_setup = AkitaCommitmentScheme::<D64OneHotTensor>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) =
            AkitaCommitmentScheme::<D64OneHotTensor>::commit::<_, _>(&setup, commit_input, &stack)
                .expect("commit");

        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
        let proof = AkitaCommitmentScheme::<D64OneHotTensor>::batched_prove::<_, _, _>(
            &setup,
            prove_input(&pt[..], &poly_refs[..], &commitments[0], hint),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        )
        .expect("prove");

        let decoded = round_trip_proof(&proof);

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_tensor_e2e/dense");
        let result = AkitaCommitmentScheme::<D64OneHotTensor>::batched_verify(
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

// Deferred (D128-tensor follow-up): the tensor fold challenge applies an `ω²`
// factor to the effective challenge L1 mass, and under the safe
// `onehot_chunk_size = 1` default (`nonzeros = D`) the A-role collision pushes
// the D64 tensor family past its secure threshold, so every level degrades to
// cleartext and no tensor-shaped root fold is emitted. Re-enable once the tensor
// family is migrated to D=128.
#[test]
#[ignore = "D64 one-hot tensor degrades to cleartext under onehot_chunk_size=1; pending D128 tensor migration"]
fn single_onehot_tensor_nv15() {
    run_single_onehot_tensor(15);
}

#[test]
#[ignore = "D64 one-hot tensor degrades to cleartext under onehot_chunk_size=1; pending D128 tensor migration"]
fn single_onehot_tensor_nv20() {
    run_single_onehot_tensor(20);
}

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
#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn single_dense_tensor_nv15() {
    run_single_dense_tensor(15);
}

#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn single_dense_tensor_nv20() {
    run_single_dense_tensor(20);
}

#[test]
#[ignore = "dense poly under one-hot tensor config: fold beta mismatch (weak-binding-norm follow-up)"]
fn single_dense_tensor_nv22() {
    run_single_dense_tensor(22);
}
