#![allow(missing_docs)]

use hachi_pcs::algebra::{CyclotomicRing, Fp64};
use hachi_pcs::error::HachiError;
use hachi_pcs::protocol::commitment::{
    CommitmentConfig, DefaultCommitmentConfig, HachiCommitmentCore, RingCommitmentScheme,
};
use hachi_pcs::CanonicalField;

type F = Fp64<4294967197>;
const D: usize = 64;

#[derive(Clone)]
struct TinyConfig;

impl CommitmentConfig for TinyConfig {
    const D: usize = 64;
    const M: usize = 1;
    const R: usize = 1;
    const N_A: usize = 2;
    const N_B: usize = 2;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 8;
}

#[derive(Clone)]
struct BadDegreeConfig;

impl CommitmentConfig for BadDegreeConfig {
    const D: usize = 32;
    const M: usize = 4;
    const R: usize = 2;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const LOG_BASIS: u32 = 4;
    const DELTA: usize = 8;
}

#[derive(Clone)]
struct BadDigitBudgetConfig;

impl CommitmentConfig for BadDigitBudgetConfig {
    const D: usize = 64;
    const M: usize = 4;
    const R: usize = 2;
    const N_A: usize = 8;
    const N_B: usize = 4;
    const LOG_BASIS: u32 = 32;
    const DELTA: usize = 5; // 160 > 128
}

fn sample_blocks() -> Vec<Vec<CyclotomicRing<F, D>>> {
    let num_blocks = 1usize << TinyConfig::R;
    let block_len = 1usize << TinyConfig::M;
    (0..num_blocks)
        .map(|bi| {
            (0..block_len)
                .map(|bj| {
                    let coeffs = std::array::from_fn(|k| {
                        let v = (bi * 1_000 + bj * 100 + k) as u64;
                        F::from_u64(v)
                    });
                    CyclotomicRing::from_coefficients(coeffs)
                })
                .collect()
        })
        .collect()
}

fn mat_vec_mul(
    mat: &[Vec<CyclotomicRing<F, D>>],
    vec: &[CyclotomicRing<F, D>],
) -> Vec<CyclotomicRing<F, D>> {
    mat.iter()
        .map(|row| {
            assert_eq!(row.len(), vec.len());
            row.iter()
                .zip(vec.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (a, x)| {
                    acc + (*a * *x)
                })
        })
        .collect()
}

#[test]
fn setup_shape_is_consistent() {
    let (p1, v1) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let (p2, v2) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();

    assert_eq!(p1.max_num_vars, 16);
    assert_eq!(v1.max_num_vars, 16);
    assert_eq!(p2.max_num_vars, 16);
    assert_eq!(v2.max_num_vars, 16);
    assert_eq!(p1.A.len(), TinyConfig::N_A);
    assert_eq!(p1.A[0].len(), (1usize << TinyConfig::M) * TinyConfig::DELTA);
    assert_eq!(p1.B.len(), TinyConfig::N_B);
    assert_eq!(
        p1.B[0].len(),
        TinyConfig::N_A * TinyConfig::DELTA * (1usize << TinyConfig::R)
    );
}

#[test]
fn commit_is_deterministic_and_shape_consistent() {
    let (psetup, _) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let blocks = sample_blocks();

    let (c1, o1) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
            &blocks, &psetup,
        )
        .unwrap();
    let (c2, o2) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
            &blocks, &psetup,
        )
        .unwrap();

    assert_eq!(c1, c2);
    assert_eq!(o1, o2);

    let num_blocks = 1usize << TinyConfig::R;
    let block_len = 1usize << TinyConfig::M;
    assert_eq!(c1.u.len(), TinyConfig::N_B);
    assert_eq!(o1.s.len(), num_blocks);
    assert_eq!(o1.t_hat.len(), num_blocks);
    assert!(o1
        .s
        .iter()
        .all(|s| s.len() == block_len * TinyConfig::DELTA));
    assert!(o1
        .t_hat
        .iter()
        .all(|t| t.len() == TinyConfig::N_A * TinyConfig::DELTA));
}

#[test]
fn opening_satisfies_inner_and_outer_equations() {
    let (psetup, _) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::setup(16).unwrap();
    let blocks = sample_blocks();
    let (commitment, opening) =
        <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
            &blocks, &psetup,
        )
        .unwrap();

    for i in 0..opening.s.len() {
        let lhs = mat_vec_mul(&psetup.A, &opening.s[i]);
        let rhs: Vec<CyclotomicRing<F, D>> = (0..TinyConfig::N_A)
            .map(|j| {
                let start = j * TinyConfig::DELTA;
                let end = start + TinyConfig::DELTA;
                CyclotomicRing::gadget_recompose_pow2(
                    &opening.t_hat[i][start..end],
                    TinyConfig::LOG_BASIS,
                )
            })
            .collect();
        assert_eq!(lhs, rhs);
    }

    let t_hat_flat: Vec<CyclotomicRing<F, D>> = opening
        .t_hat
        .iter()
        .flat_map(|x| x.iter().copied())
        .collect();
    let outer = mat_vec_mul(&psetup.B, &t_hat_flat);
    assert_eq!(outer, commitment.u);
}

#[test]
fn default_config_has_expected_shape() {
    assert_eq!(DefaultCommitmentConfig::D, 64);
    assert_eq!(1usize << DefaultCommitmentConfig::M, 16);
    assert_eq!(1usize << DefaultCommitmentConfig::R, 4);
    let delta = DefaultCommitmentConfig::DELTA;
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
