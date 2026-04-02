//! End-to-end tests for the **single-polynomial** (non-batched) commitment path.
//!
//! Each test commits to one polynomial, produces an opening proof, round-trips
//! the proof through serialization/deserialization, and verifies the result.
//!
//! Two polynomial representations are covered:
//!
//! * **One-hot** — `Fp128OneHotCommitmentConfig` (D = 64, K = D).
//! * **Dense**   — `Fp128FullCommitmentConfig`   (D = 128, full-field coefficients).
//!
//! Variable counts: 10, 15, 20, 25 for each representation (8 tests total).

#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::presets::fp128_5823;
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field, BlockOrder,
};
use hachi_pcs::protocol::proof::HachiProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::{CommitmentConfig, HachiCommitmentLayout};
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Once;

type F = Fp128<0xffffffffffffffffffffffffffffe941>;
const STACK_SIZE: usize = 256 * 1024 * 1024;

static INIT_RAYON: Once = Once::new();

fn init_rayon_pool() {
    INIT_RAYON.call_once(|| {
        #[cfg(feature = "parallel")]
        rayon::ThreadPoolBuilder::new()
            .stack_size(STACK_SIZE)
            .build_global()
            .ok();
    });
}

fn random_point(nv: usize, seed: u64) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect()
}

fn run_on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(f)
        .expect("failed to spawn thread")
        .join()
        .expect("test thread panicked");
}

fn opening_from_poly<const D: usize, P: HachiPolyOps<F, D>>(
    poly: &P,
    point: &[F],
    layout: &HachiCommitmentLayout,
) -> F {
    let alpha_bits = D.trailing_zeros() as usize;
    assert_eq!(point.len(), alpha_bits + layout.m_vars + layout.r_vars);

    let inner_point = &point[..alpha_bits];
    let reduced_point = &point[alpha_bits..];
    let ring_opening_point = ring_opening_point_from_field(
        reduced_point,
        layout.r_vars,
        layout.m_vars,
        BasisMode::Lagrange,
        BlockOrder::RowMajor,
    )
    .expect("opening point shape should match layout");

    let (y_ring, _) = poly.evaluate_and_fold(
        &ring_opening_point.b,
        &ring_opening_point.a,
        layout.block_len,
    );
    let v = reduce_inner_opening_to_ring_element::<F, D>(inner_point, BasisMode::Lagrange)
        .expect("inner opening point should match ring dimension");
    (y_ring * v.sigma_m1()).coefficients()[0]
}

// ---------------------------------------------------------------------------
// One-hot helpers (D = 64)
// ---------------------------------------------------------------------------

type OneHotCfg = fp128_5823::OneHot;
const ONEHOT_D: usize = OneHotCfg::D;
const ONEHOT_K: usize = ONEHOT_D;

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

type DenseCfg = fp128_5823::Full;
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
