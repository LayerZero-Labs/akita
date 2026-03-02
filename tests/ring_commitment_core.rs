#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::error::HachiError;
use hachi_pcs::protocol::commitment::{
    CommitmentConfig, HachiCommitmentCore, HachiCommitmentLayout, RingCommitmentScheme,
    SmallTestCommitmentConfig,
};
use hachi_pcs::test_utils::*;

#[derive(Clone)]
struct BadDegreeConfig;

impl CommitmentConfig for BadDegreeConfig {
    const D: usize = 32;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 8;
    const TAU: usize = 4;
    const CHALLENGE_WEIGHT: usize = 3;

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(4, 2)
    }
}

#[derive(Clone)]
struct BadDigitBudgetConfig;

impl CommitmentConfig for BadDigitBudgetConfig {
    const D: usize = 64;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const N_D: usize = 4;
    const LOG_BASIS: u32 = 32;
    const DELTA: usize = 5; // 160 > 128
    const TAU: usize = 4;
    const CHALLENGE_WEIGHT: usize = 3;

    fn commitment_layout(_max_num_vars: usize) -> Result<HachiCommitmentLayout, HachiError> {
        HachiCommitmentLayout::new::<Self>(4, 2)
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
    assert_eq!(p1.expanded.A.len(), TinyConfig::N_A);
    assert_eq!(
        p1.expanded.A[0].len(),
        hachi_pcs::test_utils::BLOCK_LEN * TinyConfig::DELTA
    );
    assert_eq!(p1.expanded.B.len(), TinyConfig::N_B);
    assert_eq!(
        p1.expanded.B[0].len(),
        TinyConfig::N_A * TinyConfig::DELTA * hachi_pcs::test_utils::NUM_BLOCKS
    );
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

    let num_blocks = hachi_pcs::test_utils::NUM_BLOCKS;
    assert_eq!(w1.commitment.u.len(), TinyConfig::N_B);
    assert_eq!(w1.t_hat.len(), num_blocks);
    assert!(w1
        .t_hat
        .iter()
        .all(|t| t.len() == TinyConfig::N_A * TinyConfig::DELTA));
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

    for (i, block) in blocks.iter().enumerate() {
        let s_i = hachi_pcs::protocol::commitment::utils::linear::decompose_block(
            block,
            TinyConfig::DELTA,
            TinyConfig::LOG_BASIS,
        );
        let lhs = mat_vec_mul(&psetup.expanded.A, &s_i);
        let rhs: Vec<CyclotomicRing<F, D>> = (0..TinyConfig::N_A)
            .map(|j| {
                let start = j * TinyConfig::DELTA;
                let end = start + TinyConfig::DELTA;
                CyclotomicRing::gadget_recompose_pow2(
                    &w.t_hat[i][start..end],
                    TinyConfig::LOG_BASIS,
                )
            })
            .collect();
        assert_eq!(lhs, rhs);
    }

    let t_hat_flat: Vec<CyclotomicRing<F, D>> =
        w.t_hat.iter().flat_map(|x| x.iter().copied()).collect();
    let outer = mat_vec_mul(&psetup.expanded.B, &t_hat_flat);
    assert_eq!(outer, w.commitment.u);
}

#[test]
fn small_test_config_has_expected_shape() {
    assert_eq!(SmallTestCommitmentConfig::D, 16);
    let layout = SmallTestCommitmentConfig::commitment_layout(8).unwrap();
    assert_eq!(layout.block_len, 16);
    assert_eq!(layout.num_blocks, 4);
    let delta = SmallTestCommitmentConfig::DELTA;
    assert!(delta > 0);
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

#[test]
fn setup_rejects_invalid_digit_budget() {
    let err = <HachiCommitmentCore as RingCommitmentScheme<F, D, BadDigitBudgetConfig>>::setup(16)
        .unwrap_err();
    match err {
        HachiError::InvalidSetup(msg) => assert!(msg.contains("DELTA * LOG_BASIS")),
        other => panic!("unexpected error: {other:?}"),
    }
}
