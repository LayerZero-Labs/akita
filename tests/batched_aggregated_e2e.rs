//! End-to-end tests for **batched aggregated** commitments.
//!
//! All polynomials in a batch are placed into a single commitment group, so
//! `batched_commit` produces exactly one commitment that aggregates every
//! polynomial.  The test exercises `batched_commit` → `batched_prove` →
//! serialize/deserialize → `batched_verify`.
//!
//! Only the one-hot representation is used (`Fp128OneHotCommitmentConfig`,
//! D = 64, K = D).
//!
//! Variable counts: 10, 15, 20, 25.
//! Batch sizes per variable count: 1, 2, 3, 4, 7, 12, 16 (28 tests total).

#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{hachi_batched_root_layout, Fp128OneHotCommitmentConfig};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{HachiPolyOps, OneHotPoly};
use hachi_pcs::protocol::opening_point::{
    reduce_inner_opening_to_ring_element, ring_opening_point_from_field,
};
use hachi_pcs::protocol::proof::HachiBatchedProof;
use hachi_pcs::protocol::transcript::Blake2bTranscript;
use hachi_pcs::protocol::{CommitmentConfig, HachiCommitmentLayout};
use hachi_pcs::{
    BasisMode, CanonicalField, CommitmentScheme, HachiDeserialize, HachiSerialize, Transcript,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Once;

type F = Fp128<0xffffffffffffffffffffffffffffe941>;
type Cfg = Fp128OneHotCommitmentConfig;
const D: usize = Cfg::D;
const ONEHOT_K: usize = D;
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

fn opening_from_poly<P: HachiPolyOps<F, D>>(
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

fn make_onehot_poly(layout: &HachiCommitmentLayout, seed: u64) -> OneHotPoly<F, D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
        .expect("onehot poly")
}

/// All polynomials are aggregated into a single commitment group.
fn run_aggregated_onehot(nv: usize, batch_size: usize) {
    init_rayon_pool();
    run_on_large_stack(move || {
        let layout = hachi_batched_root_layout::<Cfg, D>(nv, batch_size).expect("layout");

        let polys: Vec<OneHotPoly<F, D, u8>> = (0..batch_size)
            .map(|idx| make_onehot_poly(&layout, 0xa66e_0000 + (nv as u64) * 100 + idx as u64))
            .collect();

        let pt = random_point(nv, 0xf00d_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_prover(nv, batch_size);
        let verifier_setup =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::setup_verifier(&setup);

        let poly_groups: [&[OneHotPoly<F, D, u8>]; 1] = [&polys];
        let (commitments, hints) =
            <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_commit(
                &poly_groups,
                &setup,
                &layout,
            )
            .expect("batched commit");

        assert_eq!(
            commitments.len(),
            1,
            "single group should yield exactly one commitment"
        );

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"batched_aggregated_e2e/onehot");
        let proof = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_prove(
            &setup,
            &poly_groups,
            &pt,
            hints,
            &mut prover_transcript,
            &commitments,
            BasisMode::Lagrange,
            &layout,
        )
        .expect("batched prove");

        let mut serialized = Vec::new();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded =
            HachiBatchedProof::<F>::deserialize_compressed(&mut std::io::Cursor::new(serialized))
                .expect("deserialize");

        let opening_groups: [&[F]; 1] = [&openings];
        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"batched_aggregated_e2e/onehot");
        let result = <HachiCommitmentScheme<D, Cfg> as CommitmentScheme<F, D>>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &pt,
            &opening_groups,
            &commitments,
            BasisMode::Lagrange,
            &layout,
        );
        assert!(
            result.is_ok(),
            "aggregated onehot nv={nv} batch={batch_size} verification failed: {:?}",
            result.err()
        );
    });
}

// ---------------------------------------------------------------------------
// nv = 10
// ---------------------------------------------------------------------------

#[test]
fn aggregated_onehot_nv10_batch1() {
    run_aggregated_onehot(10, 1);
}

#[test]
fn aggregated_onehot_nv10_batch2() {
    run_aggregated_onehot(10, 2);
}

#[test]
fn aggregated_onehot_nv10_batch3() {
    run_aggregated_onehot(10, 3);
}

#[test]
fn aggregated_onehot_nv10_batch4() {
    run_aggregated_onehot(10, 4);
}

#[test]
fn aggregated_onehot_nv10_batch7() {
    run_aggregated_onehot(10, 7);
}

#[test]
fn aggregated_onehot_nv10_batch12() {
    run_aggregated_onehot(10, 12);
}

#[test]
fn aggregated_onehot_nv10_batch16() {
    run_aggregated_onehot(10, 16);
}

// ---------------------------------------------------------------------------
// nv = 15
// ---------------------------------------------------------------------------

#[test]
fn aggregated_onehot_nv15_batch1() {
    run_aggregated_onehot(15, 1);
}

#[test]
fn aggregated_onehot_nv15_batch2() {
    run_aggregated_onehot(15, 2);
}

#[test]
fn aggregated_onehot_nv15_batch3() {
    run_aggregated_onehot(15, 3);
}

#[test]
fn aggregated_onehot_nv15_batch4() {
    run_aggregated_onehot(15, 4);
}

#[test]
fn aggregated_onehot_nv15_batch7() {
    run_aggregated_onehot(15, 7);
}

#[test]
fn aggregated_onehot_nv15_batch12() {
    run_aggregated_onehot(15, 12);
}

#[test]
fn aggregated_onehot_nv15_batch16() {
    run_aggregated_onehot(15, 16);
}

// ---------------------------------------------------------------------------
// nv = 20
// ---------------------------------------------------------------------------

#[test]
fn aggregated_onehot_nv20_batch1() {
    run_aggregated_onehot(20, 1);
}

#[test]
fn aggregated_onehot_nv20_batch2() {
    run_aggregated_onehot(20, 2);
}

#[test]
fn aggregated_onehot_nv20_batch3() {
    run_aggregated_onehot(20, 3);
}

#[test]
fn aggregated_onehot_nv20_batch4() {
    run_aggregated_onehot(20, 4);
}

#[test]
fn aggregated_onehot_nv20_batch7() {
    run_aggregated_onehot(20, 7);
}

#[test]
fn aggregated_onehot_nv20_batch12() {
    run_aggregated_onehot(20, 12);
}

#[test]
fn aggregated_onehot_nv20_batch16() {
    run_aggregated_onehot(20, 16);
}

// ---------------------------------------------------------------------------
// nv = 25
// ---------------------------------------------------------------------------

#[test]
fn aggregated_onehot_nv25_batch1() {
    run_aggregated_onehot(25, 1);
}

#[test]
fn aggregated_onehot_nv25_batch2() {
    run_aggregated_onehot(25, 2);
}

#[test]
fn aggregated_onehot_nv25_batch3() {
    run_aggregated_onehot(25, 3);
}

#[test]
fn aggregated_onehot_nv25_batch4() {
    run_aggregated_onehot(25, 4);
}

#[test]
fn aggregated_onehot_nv25_batch7() {
    run_aggregated_onehot(25, 7);
}

#[test]
fn aggregated_onehot_nv25_batch12() {
    run_aggregated_onehot(25, 12);
}

#[test]
fn aggregated_onehot_nv25_batch16() {
    run_aggregated_onehot(25, 16);
}
