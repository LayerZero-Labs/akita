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
//! Variable counts:
//!
//! - one-hot: 10, 15, 20
//! - dense: 10, 15, 18

#![allow(missing_docs)]

use akita_prover::{ComputeBackendSetup, CpuBackend};

mod common;

use akita_pcs::AkitaCommitmentScheme;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::AkitaTranscript;
use akita_types::AkitaBatchedProof;
use common::*;

fn run_single_onehot(nv: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(nv, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let total_field = layout.num_blocks * layout.block_len * ONEHOT_D;
        assert_eq!(total_field, 1usize << nv);
        let total_chunks = total_field / ONEHOT_K;

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + nv as u64);
        let indices: Vec<Option<u8>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly = OneHotPoly::<F, u8>::new(ONEHOT_K, ONEHOT_D, indices).expect("onehot poly");

        let pt = random_point(nv, 0xcafe_0000 + nv as u64);
        let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = AkitaCommitmentScheme::<OneHotCfg>::commit::<_, _, ONEHOT_D>(
            &setup,
            commit_input,
            &stack,
        )
        .expect("commit");

        let poly_refs: [&OneHotPoly<F, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_e2e/onehot");
        let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove::<_, _, _, ONEHOT_D>(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_e2e/onehot");
        let result = AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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
        let layout = DenseCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(nv, 1).expect("singleton opening batch"),
        )
        .expect("layout");

        let evals = dense_field_evals(nv, 0xface_feed_0000 + nv as u64);
        let poly = DensePoly::<F>::from_field_evals(nv, DENSE_D, &evals).expect("dense poly");

        let pt = random_point(nv, 0xbabe_0000 + nv as u64);
        let expected_opening = opening_from_poly::<DENSE_D, _>(&poly, &pt, &layout);

        let setup = AkitaCommitmentScheme::<DenseCfg>::setup_prover(nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = AkitaCommitmentScheme::<DenseCfg>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = AkitaCommitmentScheme::<DenseCfg>::commit::<_, _, DENSE_D>(
            &setup,
            commit_input,
            &stack,
        )
        .expect("commit");

        let poly_refs: [&DensePoly<F>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_e2e/dense");
        let proof = AkitaCommitmentScheme::<DenseCfg>::batched_prove::<_, _, _, DENSE_D>(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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

        let mut verifier_transcript = AkitaTranscript::<F>::new(b"single_poly_e2e/dense");
        let result = AkitaCommitmentScheme::<DenseCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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
fn single_dense_nv18() {
    run_single_dense(18);
}

// #[test]
// fn single_dense_nv25() {
//     run_single_dense(25);
// }

// ---------------------------------------------------------------------------
// Oversized setup: setup with max_num_vars > actual polynomial num_vars
// ---------------------------------------------------------------------------

fn run_single_onehot_oversized_setup(setup_nv: usize, poly_nv: usize) {
    assert!(setup_nv >= poly_nv);
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = OneHotCfg::get_params_for_batched_commitment(
            &akita_types::OpeningBatchShape::new(poly_nv, 1).expect("singleton opening batch"),
        )
        .expect("layout");
        let total_field = layout.num_blocks * layout.block_len * ONEHOT_D;
        assert_eq!(total_field, 1usize << poly_nv);
        let total_chunks = total_field / ONEHOT_K;

        let mut rng = StdRng::seed_from_u64(0xdead_beef_0000 + poly_nv as u64);
        let indices: Vec<Option<u8>> = (0..total_chunks)
            .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
            .collect();
        let poly = OneHotPoly::<F, u8>::new(ONEHOT_K, ONEHOT_D, indices).expect("onehot poly");

        let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
        let expected_opening = opening_from_poly::<ONEHOT_D, _>(&poly, &pt, &layout);

        let setup = AkitaCommitmentScheme::<OneHotCfg>::setup_prover(setup_nv, 1).unwrap();
        let prepared = CpuBackend.prepare_setup(&setup).unwrap();
        let stack = akita_prover::UniformProverStack::uniform(
            &CpuBackend,
            &prepared,
            setup.expanded.as_ref(),
        )
        .expect("stack");
        let verifier_setup = AkitaCommitmentScheme::<OneHotCfg>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = AkitaCommitmentScheme::<OneHotCfg>::commit::<_, _, ONEHOT_D>(
            &setup,
            commit_input,
            &stack,
        )
        .expect("commit with oversized setup");

        let poly_refs: [&OneHotPoly<F, u8>; 1] = [&poly];
        let commitments = [commitment];
        let openings = [expected_opening];
        let opening_groups = [&openings[..]];
        let hints = vec![hint];

        let mut prover_transcript = AkitaTranscript::<F>::new(b"single_poly_e2e/onehot_oversized");
        let proof = AkitaCommitmentScheme::<OneHotCfg>::batched_prove::<_, _, _, ONEHOT_D>(
            &setup,
            prove_input(
                &pt[..],
                &poly_refs[..],
                &commitments[0],
                hints.into_iter().next().unwrap(),
            ),
            &stack,
            &mut prover_transcript,
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
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
            AkitaTranscript::<F>::new(b"single_poly_e2e/onehot_oversized");
        let result = AkitaCommitmentScheme::<OneHotCfg>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            verify_input(&pt[..], opening_groups[0], &commitments[0]),
            BasisMode::Lagrange,
            akita_types::SetupContributionMode::Direct,
        );
        assert!(
            result.is_ok(),
            "onehot oversized setup (setup_nv={setup_nv}, poly_nv={poly_nv}) verification failed: {:?}",
            result.err()
        );
    });
}

#[test]
fn single_onehot_oversized_setup_15_10() {
    run_single_onehot_oversized_setup(15, 10);
}

#[test]
fn single_onehot_oversized_setup_20_15() {
    run_single_onehot_oversized_setup(20, 15);
}
