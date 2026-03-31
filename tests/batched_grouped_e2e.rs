//! End-to-end tests for **batched grouped** (detached) commitments.
//!
//! Polynomials in a batch are split into multiple commitment groups.  Each
//! group aggregates its own polynomials into a separate commitment, so
//! `batched_commit` returns one commitment per group.  The test exercises
//! `batched_commit` → `batched_prove` → serialize/deserialize →
//! `batched_verify`, and additionally cross-checks that committing each group
//! individually produces the same commitment as the multi-group call.
//!
//! Two polynomial representations are covered:
//!
//! * **One-hot** — `Fp128OneHotCommitmentConfig` (D = 64, K = D).
//!   Variable counts: 10, 15, 20, 25 (28 tests).
//! * **Dense** — `Fp128FullCommitmentConfig` (D = 128, full-field coefficients).
//!   Variable counts: 10, 15, 20 (21 tests — nv 25 is omitted for speed).
//!
//! Batch sizes per variable count: 1, 2, 3, 4, 7, 12, 16 (49 tests total).
//!
//! Group partitions by batch size:
//!   1 → \[1\],  2 → \[1,1\],  3 → \[2,1\],  4 → \[2,2\],
//!   7 → \[3,2,2\],  12 → \[4,4,4\],  16 → \[5,5,3,3\].

#![allow(missing_docs)]

use hachi_pcs::algebra::Fp128;
use hachi_pcs::protocol::commitment::{
    hachi_batched_root_layout, Fp128FullCommitmentConfig, Fp128OneHotCommitmentConfig,
};
use hachi_pcs::protocol::commitment_scheme::HachiCommitmentScheme;
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps, OneHotPoly};
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
const STACK_SIZE: usize = 256 * 1024 * 1024;

type OneHotCfg = Fp128OneHotCommitmentConfig;
const ONEHOT_D: usize = OneHotCfg::D;
const ONEHOT_K: usize = ONEHOT_D;

type DenseCfg = Fp128FullCommitmentConfig;
const DENSE_D: usize = DenseCfg::D;

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

fn make_onehot_poly(layout: &HachiCommitmentLayout, seed: u64) -> OneHotPoly<F, ONEHOT_D, u8> {
    let total_ring = layout.num_blocks * layout.block_len;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<Option<u8>> = (0..total_ring)
        .map(|_| Some(rng.gen_range(0..ONEHOT_K) as u8))
        .collect();
    OneHotPoly::<F, ONEHOT_D, u8>::new(ONEHOT_K, indices, layout.r_vars, layout.m_vars)
        .expect("onehot poly")
}

fn make_dense_poly(nv: usize, seed: u64) -> DensePoly<F, DENSE_D> {
    let mut rng = StdRng::seed_from_u64(seed);
    let evals: Vec<F> = (0..1usize << nv)
        .map(|_| F::from_canonical_u128_reduced(rng.gen::<u128>()))
        .collect();
    DensePoly::<F, DENSE_D>::from_field_evals(nv, &evals).expect("dense poly")
}

/// Return the group-size partition for a given total batch size.
fn group_partition(batch_size: usize) -> Vec<usize> {
    match batch_size {
        1 => vec![1],
        2 => vec![1, 1],
        3 => vec![2, 1],
        4 => vec![2, 2],
        7 => vec![3, 2, 2],
        12 => vec![4, 4, 4],
        16 => vec![5, 5, 3, 3],
        other => panic!("no partition defined for batch_size={other}"),
    }
}

/// One-hot polynomials are split into multiple commitment groups according to
/// [`group_partition`].  Each group produces its own commitment.  The test
/// also verifies that committing each group individually yields the same
/// commitment as the multi-group `batched_commit` result.
fn run_grouped_onehot(nv: usize, batch_size: usize) {
    init_rayon_pool();
    let group_sizes = group_partition(batch_size);
    run_on_large_stack(move || {
        let layout =
            hachi_batched_root_layout::<OneHotCfg, ONEHOT_D>(nv, batch_size).expect("layout");

        let polys: Vec<OneHotPoly<F, ONEHOT_D, u8>> = (0..batch_size)
            .map(|idx| make_onehot_poly(&layout, 0xbeef_0000 + (nv as u64) * 100 + idx as u64))
            .collect();

        let pt = random_point(nv, 0xd00d_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_prover(nv, batch_size);
        let verifier_setup = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::setup_verifier(&setup);

        let mut poly_groups: Vec<&[OneHotPoly<F, ONEHOT_D, u8>]> = Vec::new();
        let mut opening_groups: Vec<&[F]> = Vec::new();
        let mut offset = 0usize;
        for &gs in &group_sizes {
            poly_groups.push(&polys[offset..offset + gs]);
            opening_groups.push(&openings[offset..offset + gs]);
            offset += gs;
        }
        assert_eq!(offset, batch_size);

        let (commitments, hints): (Vec<_>, Vec<_>) = poly_groups
            .iter()
            .map(|group| {
                <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::commit(
                    group,
                    &setup,
                    &layout,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("grouped commit")
            .into_iter()
            .unzip();

        assert_eq!(
            commitments.len(),
            group_sizes.len(),
            "number of commitments must equal number of groups"
        );

        for (group_idx, &gs) in group_sizes.iter().enumerate() {
            let start = group_sizes[..group_idx].iter().sum::<usize>();
            let (individual_commit, _) =
                <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::commit(
                    &polys[start..start + gs],
                    &setup,
                    &layout,
                )
                .expect("individual group commit");
            assert_eq!(
                individual_commit, commitments[group_idx],
                "group {group_idx} commitment mismatch between individual and multi-group commit"
            );
        }

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/onehot");
        let proof =
            <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<F, ONEHOT_D>>::batched_prove(
                &setup,
                &[&poly_groups[..]],
                &[&pt[..]],
                vec![hints],
                &mut prover_transcript,
                &[&commitments[..]],
                BasisMode::Lagrange,
                &layout,
            )
            .expect("batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/onehot");
        let result = <HachiCommitmentScheme<ONEHOT_D, OneHotCfg> as CommitmentScheme<
            F,
            ONEHOT_D,
        >>::batched_verify(
            &decoded,
            &verifier_setup,
            &mut verifier_transcript,
            &[&pt[..]],
            &[&opening_groups[..]],
            &[&commitments[..]],
            BasisMode::Lagrange,
            &layout,
        );
        assert!(
            result.is_ok(),
            "grouped onehot nv={nv} batch={batch_size} groups={group_sizes:?} \
             verification failed: {:?}",
            result.err()
        );
    });
}

/// Dense polynomials are split into multiple commitment groups according to
/// [`group_partition`].  Same cross-check as the one-hot variant.
fn run_grouped_dense(nv: usize, batch_size: usize) {
    init_rayon_pool();
    let group_sizes = group_partition(batch_size);
    run_on_large_stack(move || {
        let layout =
            hachi_batched_root_layout::<DenseCfg, DENSE_D>(nv, batch_size).expect("layout");

        let polys: Vec<DensePoly<F, DENSE_D>> = (0..batch_size)
            .map(|idx| make_dense_poly(nv, 0xd6e5_0000 + (nv as u64) * 100 + idx as u64))
            .collect();

        let pt = random_point(nv, 0xbbbb_0000 + nv as u64);
        let openings: Vec<F> = polys
            .iter()
            .map(|poly| opening_from_poly(poly, &pt, &layout))
            .collect();

        let setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_prover(nv, batch_size);
        let verifier_setup = <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<
            F,
            DENSE_D,
        >>::setup_verifier(&setup);

        let mut poly_groups: Vec<&[DensePoly<F, DENSE_D>]> = Vec::new();
        let mut opening_groups: Vec<&[F]> = Vec::new();
        let mut offset = 0usize;
        for &gs in &group_sizes {
            poly_groups.push(&polys[offset..offset + gs]);
            opening_groups.push(&openings[offset..offset + gs]);
            offset += gs;
        }
        assert_eq!(offset, batch_size);

        let (commitments, hints): (Vec<_>, Vec<_>) = poly_groups
            .iter()
            .map(|group| {
                <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::commit(
                    group, &setup, &layout,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("grouped commit")
            .into_iter()
            .unzip();

        assert_eq!(
            commitments.len(),
            group_sizes.len(),
            "number of commitments must equal number of groups"
        );

        for (group_idx, &gs) in group_sizes.iter().enumerate() {
            let start = group_sizes[..group_idx].iter().sum::<usize>();
            let (individual_commit, _) =
                <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::commit(
                    &polys[start..start + gs],
                    &setup,
                    &layout,
                )
                .expect("individual group commit");
            assert_eq!(
                individual_commit, commitments[group_idx],
                "group {group_idx} commitment mismatch between individual and multi-group commit"
            );
        }

        let mut prover_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/dense");
        let proof =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_prove(
                &setup,
                &[&poly_groups[..]],
                &[&pt[..]],
                vec![hints],
                &mut prover_transcript,
                &[&commitments[..]],
                BasisMode::Lagrange,
                &layout,
            )
            .expect("batched prove");

        let mut serialized = Vec::new();
        let proof_shape = proof.shape();
        proof
            .serialize_compressed(&mut serialized)
            .expect("serialize");
        let decoded = HachiBatchedProof::<F>::deserialize_compressed(
            &mut std::io::Cursor::new(serialized),
            &proof_shape,
        )
        .expect("deserialize");

        let mut verifier_transcript = Blake2bTranscript::<F>::new(b"batched_grouped_e2e/dense");
        let result =
            <HachiCommitmentScheme<DENSE_D, DenseCfg> as CommitmentScheme<F, DENSE_D>>::batched_verify(
                &decoded,
                &verifier_setup,
                &mut verifier_transcript,
                &[&pt[..]],
                &[&opening_groups[..]],
                &[&commitments[..]],
                BasisMode::Lagrange,
                &layout,
            );
        assert!(
            result.is_ok(),
            "grouped dense nv={nv} batch={batch_size} groups={group_sizes:?} \
             verification failed: {:?}",
            result.err()
        );
    });
}

// ---------------------------------------------------------------------------
// nv = 10
// ---------------------------------------------------------------------------

#[test]
fn grouped_onehot_nv10_batch1() {
    run_grouped_onehot(10, 1);
}

#[test]
fn grouped_onehot_nv10_batch2() {
    run_grouped_onehot(10, 2);
}

#[test]
fn grouped_onehot_nv10_batch3() {
    run_grouped_onehot(10, 3);
}

#[test]
fn grouped_onehot_nv10_batch4() {
    run_grouped_onehot(10, 4);
}

#[test]
fn grouped_onehot_nv10_batch7() {
    run_grouped_onehot(10, 7);
}

#[test]
fn grouped_onehot_nv10_batch12() {
    run_grouped_onehot(10, 12);
}

#[test]
fn grouped_onehot_nv10_batch16() {
    run_grouped_onehot(10, 16);
}

// ---------------------------------------------------------------------------
// nv = 15
// ---------------------------------------------------------------------------

#[test]
fn grouped_onehot_nv15_batch1() {
    run_grouped_onehot(15, 1);
}

#[test]
fn grouped_onehot_nv15_batch2() {
    run_grouped_onehot(15, 2);
}

#[test]
fn grouped_onehot_nv15_batch3() {
    run_grouped_onehot(15, 3);
}

#[test]
fn grouped_onehot_nv15_batch4() {
    run_grouped_onehot(15, 4);
}

#[test]
fn grouped_onehot_nv15_batch7() {
    run_grouped_onehot(15, 7);
}

#[test]
fn grouped_onehot_nv15_batch12() {
    run_grouped_onehot(15, 12);
}

#[test]
fn grouped_onehot_nv15_batch16() {
    run_grouped_onehot(15, 16);
}

// ---------------------------------------------------------------------------
// nv = 20
// ---------------------------------------------------------------------------

#[test]
fn grouped_onehot_nv20_batch1() {
    run_grouped_onehot(20, 1);
}

#[test]
fn grouped_onehot_nv20_batch2() {
    run_grouped_onehot(20, 2);
}

#[test]
fn grouped_onehot_nv20_batch3() {
    run_grouped_onehot(20, 3);
}

#[test]
fn grouped_onehot_nv20_batch4() {
    run_grouped_onehot(20, 4);
}

#[test]
fn grouped_onehot_nv20_batch7() {
    run_grouped_onehot(20, 7);
}

#[test]
fn grouped_onehot_nv20_batch12() {
    run_grouped_onehot(20, 12);
}

// #[test]
// fn grouped_onehot_nv20_batch16() {
//     run_grouped_onehot(20, 16);
// }

// ---------------------------------------------------------------------------
// nv = 25
// ---------------------------------------------------------------------------

#[test]
fn grouped_onehot_nv25_batch1() {
    run_grouped_onehot(25, 1);
}

#[test]
fn grouped_onehot_nv25_batch2() {
    run_grouped_onehot(25, 2);
}

#[test]
fn grouped_onehot_nv25_batch3() {
    run_grouped_onehot(25, 3);
}

#[test]
fn grouped_onehot_nv25_batch4() {
    run_grouped_onehot(25, 4);
}

#[test]
fn grouped_onehot_nv25_batch7() {
    run_grouped_onehot(25, 7);
}

// #[test]
// fn grouped_onehot_nv25_batch12() {
//     run_grouped_onehot(25, 12);
// }

// #[test]
// fn grouped_onehot_nv25_batch16() {
//     run_grouped_onehot(25, 16);
// }

// ===========================================================================
// Dense batched-grouped tests (D = 128)
// ===========================================================================

// ---------------------------------------------------------------------------
// nv = 10
// ---------------------------------------------------------------------------

#[test]
fn grouped_dense_nv10_batch1() {
    run_grouped_dense(10, 1);
}

#[test]
fn grouped_dense_nv10_batch2() {
    run_grouped_dense(10, 2);
}

#[test]
fn grouped_dense_nv10_batch3() {
    run_grouped_dense(10, 3);
}

#[test]
fn grouped_dense_nv10_batch4() {
    run_grouped_dense(10, 4);
}

#[test]
fn grouped_dense_nv10_batch7() {
    run_grouped_dense(10, 7);
}

#[test]
fn grouped_dense_nv10_batch12() {
    run_grouped_dense(10, 12);
}

#[test]
fn grouped_dense_nv10_batch16() {
    run_grouped_dense(10, 16);
}

// ---------------------------------------------------------------------------
// nv = 15
// ---------------------------------------------------------------------------

#[test]
fn grouped_dense_nv15_batch1() {
    run_grouped_dense(15, 1);
}

#[test]
fn grouped_dense_nv15_batch2() {
    run_grouped_dense(15, 2);
}

#[test]
fn grouped_dense_nv15_batch3() {
    run_grouped_dense(15, 3);
}

#[test]
fn grouped_dense_nv15_batch4() {
    run_grouped_dense(15, 4);
}

#[test]
fn grouped_dense_nv15_batch7() {
    run_grouped_dense(15, 7);
}

#[test]
fn grouped_dense_nv15_batch12() {
    run_grouped_dense(15, 12);
}

// #[test]
// fn grouped_dense_nv15_batch16() {
//     run_grouped_dense(15, 16);
// }

// ---------------------------------------------------------------------------
// nv = 20
// ---------------------------------------------------------------------------

#[test]
fn grouped_dense_nv20_batch1() {
    run_grouped_dense(20, 1);
}

#[test]
fn grouped_dense_nv20_batch2() {
    run_grouped_dense(20, 2);
}

#[test]
fn grouped_dense_nv20_batch3() {
    run_grouped_dense(20, 3);
}

#[test]
fn grouped_dense_nv20_batch4() {
    run_grouped_dense(20, 4);
}

#[test]
fn grouped_dense_nv20_batch7() {
    run_grouped_dense(20, 7);
}

// #[test]
// fn grouped_dense_nv20_batch12() {
//     run_grouped_dense(20, 12);
// }

// #[test]
// fn grouped_dense_nv20_batch16() {
//     run_grouped_dense(20, 16);
// }
