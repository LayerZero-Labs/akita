#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::protocol::commitment::{
    utils::linear::decompose_block, CommitmentConfig, DecompositionParams, HachiCommitmentCore,
    HachiCommitmentLayout, RingCommitmentScheme, SmallTestCommitmentConfig,
};
use hachi_pcs::test_utils::*;
use hachi_pcs::{FromSmallInt, HachiError};
use std::array::from_fn;

#[derive(Clone)]
struct BadDegreeConfig;

impl CommitmentConfig for BadDegreeConfig {
    const D: usize = 32;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const CHALLENGE_WEIGHT: usize = 3;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(4, 2, &Self::decomposition())
    }
}

#[test]
fn setup_shape_is_consistent() {
    let (p1, v1) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let (p2, v2) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();

    assert_eq!(p1.expanded.seed.max_num_vars, 16);
    assert_eq!(v1.expanded.seed.max_num_vars, 16);
    assert_eq!(p2.expanded.seed.max_num_vars, 16);
    assert_eq!(v2.expanded.seed.max_num_vars, 16);
    assert_eq!(p1.expanded.A.num_rows(), TinyConfig::N_A);
    assert!(p1.expanded.A.num_cols_at::<D>() >= BLOCK_LEN * num_digits_commit());
    assert_eq!(p1.expanded.B.num_rows(), TinyConfig::N_B);
    assert!(p1.expanded.B.num_cols_at::<D>() >= TinyConfig::N_A * num_digits_open() * NUM_BLOCKS);
}

#[test]
fn commit_is_deterministic_and_shape_consistent() {
    let (psetup, _) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let blocks = sample_blocks();

    let w1 = <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
        &blocks, &psetup,
    )
    .unwrap();
    let w2 = <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
        &blocks, &psetup,
    )
    .unwrap();

    assert_eq!(w1.commitment, w2.commitment);
    assert_eq!(w1.t_hat, w2.t_hat);

    let num_blocks = NUM_BLOCKS;
    assert_eq!(w1.commitment.u.len(), TinyConfig::N_B);
    assert_eq!(w1.t_hat.len(), num_blocks);
    let depth = num_digits_commit();
    assert!(w1.t_hat.iter().all(|t| t.len() == TinyConfig::N_A * depth));
}

#[test]
fn commit_ring_coeffs_matches_block_commitment() {
    let (psetup, _) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let blocks = sample_blocks();

    let wb = <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
        &blocks, &psetup,
    )
    .unwrap();

    // Sequential layout: block 0 elements, then block 1 elements, etc.
    let f_coeffs: Vec<_> = blocks
        .iter()
        .flat_map(|block| block.iter().copied())
        .collect();

    let wc = <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_coeffs(
        &f_coeffs, &psetup,
    )
    .unwrap();

    assert_eq!(wb.commitment, wc.commitment);
    assert_eq!(wb.t_hat, wc.t_hat);
}

#[test]
fn opening_satisfies_inner_and_outer_equations() {
    let (psetup, _) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let blocks = sample_blocks();
    let w = <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
        &blocks, &psetup,
    )
    .unwrap();

    let depth = num_digits_commit();
    let log_basis = log_basis();
    for (i, block) in blocks.iter().enumerate() {
        let s_i = decompose_block(block, depth, log_basis);
        let lhs = mat_vec_mul(&psetup.expanded.A, &s_i);
        let rhs: Vec<CyclotomicRing<F, D>> = (0..TinyConfig::N_A)
            .map(|j| {
                let start = j * depth;
                let end = start + depth;
                CyclotomicRing::gadget_recompose_pow2_i8(&w.t_hat[i][start..end], log_basis)
            })
            .collect();
        assert_eq!(lhs, rhs);
    }

    let t_hat_flat_ring: Vec<CyclotomicRing<F, D>> = w
        .t_hat
        .iter()
        .flat_map(|x| x.iter())
        .map(|plane| {
            let coeffs: [F; D] = from_fn(|k| F::from_i64(plane[k] as i64));
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let outer = mat_vec_mul(&psetup.expanded.B, &t_hat_flat_ring);
    assert_eq!(outer, w.commitment.u);
}

#[test]
fn small_test_config_has_expected_shape() {
    assert_eq!(SmallTestCommitmentConfig::D, 16);
    let layout = SmallTestCommitmentConfig::commitment_layout(8).unwrap();
    assert_eq!(layout.block_len, 16);
    assert_eq!(layout.num_blocks, 4);
    let depth = layout.num_digits_commit;
    assert!(depth > 0);
}

#[test]
fn setup_rejects_mismatched_degree() {
    let err = <HachiCommitmentCore as RingCommitmentScheme<F, D, BadDegreeConfig>>::setup(16)
        .unwrap_err();
    match err {
        HachiError::InvalidSetup(msg) => assert!(msg.contains("mismatches")),
        other => panic!("unexpected error: {other:?}"),
    }
}
