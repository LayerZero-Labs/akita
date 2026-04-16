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

use common::*;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::proof::HachiProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::{CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript};

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
        let poly =
            OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
                .expect("onehot poly");

        let pt = random_point(nv, 0xcafe_0000 + nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::setup_prover(nv, 1);
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot");
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::prove(
                &setup,
                &poly,
                &pt,
                hint,
                &mut prover_transcript,
                &commitment,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot");
        let result =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                &pt,
                &expected_opening,
                &commitment,
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
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::setup_prover(nv, 1);
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::commit(commit_input, &setup)
        .expect("commit");

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/dense");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::prove(
                &setup,
                &poly,
                &pt,
                hint,
                &mut prover_transcript,
                &commitment,
                BasisMode::Lagrange,
            )
            .expect("prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"single_poly_e2e/dense");
        let result =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                &pt,
                &expected_opening,
                &commitment,
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
        let poly =
            OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
                .expect("onehot poly");

        let pt = random_point(poly_nv, 0xcafe_0000 + poly_nv as u64);
        let expected_opening = opening_from_poly(&poly, &pt, &layout);

        let setup =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::setup_prover(setup_nv, 1);
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);
        let commit_input = std::slice::from_ref(&poly);
        let (commitment, hint) = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::commit(commit_input, &setup)
        .expect("commit with oversized setup");

        let mut prover_transcript =
            Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot_oversized");
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::prove(
                &setup,
                &poly,
                &pt,
                hint,
                &mut prover_transcript,
                &commitment,
                BasisMode::Lagrange,
            )
            .expect("prove with oversized setup");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript =
            Blake2bTranscript::<F>::new(b"single_poly_e2e/onehot_oversized");
        let result =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                &pt,
                &expected_opening,
                &commitment,
                BasisMode::Lagrange,
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
