#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::protocol::commitment::{
    utils::linear::decompose_block, CommitmentConfig, CommitmentEnvelope, DecompositionParams,
    HachiCommitmentCore, RingCommitmentScheme, SmallTestCommitmentConfig,
};
use hachi_pcs::protocol::params::LevelParams;
use hachi_pcs::protocol::preprocessing::HachiProverSetup;
use hachi_pcs::test_utils::*;
use hachi_pcs::{FromSmallInt, HachiError};
use std::array::from_fn;

#[derive(Clone)]
struct BadDegreeConfig;

impl CommitmentConfig for BadDegreeConfig {
    type Field = F;
    const D: usize = 32;

    fn decomposition() -> DecompositionParams {
        DecompositionParams {
            log_basis: 3,
            log_commit_bound: 32,
            log_open_bound: None,
        }
    }

    fn envelope(_max_num_vars: usize) -> CommitmentEnvelope {
        CommitmentEnvelope {
            max_n_a: 4,
            max_n_b: 4,
            max_n_d: 4,
        }
    }

    fn stage1_challenge_config(d: usize) -> hachi_pcs::algebra::SparseChallengeConfig {
        assert_eq!(d, Self::D, "unsupported ring dim {d}");
        hachi_pcs::algebra::SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        }
    }

    fn commitment_layout(max_num_vars: usize) -> Result<LevelParams, HachiError> {
        Self::root_level_layout_with_log_basis(
            hachi_pcs::protocol::commitment::HachiScheduleInputs {
                max_num_vars,
                level: 0,
                current_w_len: 1usize.checked_shl(max_num_vars as u32).unwrap_or(0),
            },
            Self::decomposition().log_basis,
        )
    }
}

#[test]
fn setup_shape_is_consistent() {
    let envelope = TinyConfig::envelope(16);
    let p1 = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1).unwrap();
    let v1 = p1.verifier_setup();
    let p2 = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1).unwrap();
    let v2 = p2.verifier_setup();

    assert_eq!(p1.expanded.seed.max_num_vars, 16);
    assert_eq!(v1.expanded.seed.max_num_vars, 16);
    assert_eq!(p2.expanded.seed.max_num_vars, 16);
    assert_eq!(v2.expanded.seed.max_num_vars, 16);
    let total = p1.expanded.shared_matrix.total_ring_elements_at::<D>();
    let inner_width = BLOCK_LEN * num_digits_commit();
    let outer_width = envelope.max_n_a * num_digits_open() * NUM_BLOCKS;
    assert!(total >= envelope.max_n_a * inner_width);
    assert!(total >= envelope.max_n_b * outer_width);
}

#[test]
fn commit_is_deterministic_and_shape_consistent() {
    let psetup = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1).unwrap();
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
    assert_eq!(w1.commitment.u.len(), TinyConfig::envelope(16).max_n_b);
    assert_eq!(w1.t_hat.len(), num_blocks);
    let depth = num_digits_commit();
    assert!(w1
        .t_hat
        .iter()
        .all(|t| t.len() == TinyConfig::envelope(16).max_n_a * depth));
}

#[test]
fn commit_ring_coeffs_matches_block_commitment() {
    let psetup = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1).unwrap();
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
fn commit_ring_coeffs_rejects_short_input() {
    let psetup = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1).unwrap();
    let blocks = sample_blocks();

    let mut f_coeffs: Vec<_> = blocks
        .iter()
        .flat_map(|block| block.iter().copied())
        .collect();
    let _ = f_coeffs.pop();

    match <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_coeffs(
        &f_coeffs, &psetup,
    ) {
        Err(HachiError::InvalidSize {
            expected: _,
            actual,
        }) => assert_eq!(actual, f_coeffs.len()),
        Err(other) => panic!("unexpected error: {other:?}"),
        Ok(_) => panic!("expected short coefficient table to be rejected"),
    }
}

#[test]
fn opening_satisfies_inner_and_outer_equations() {
    let psetup = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1).unwrap();
    let blocks = sample_blocks();
    let w = <HachiCommitmentCore as RingCommitmentScheme<F, D, TinyConfig>>::commit_ring_blocks(
        &blocks, &psetup,
    )
    .unwrap();

    let depth = num_digits_commit();
    let log_basis = log_basis();
    for (i, block) in blocks.iter().enumerate() {
        let s_i = decompose_block(block, depth, log_basis);
        let lhs = mat_vec_mul(
            &psetup.expanded.shared_matrix,
            TinyConfig::envelope(16).max_n_a,
            psetup.expanded.seed.max_stride,
            &s_i,
        );
        let t_hat_block = w
            .t_hat
            .iter()
            .nth(i)
            .expect("commit witness should retain every block");
        let rhs: Vec<CyclotomicRing<F, D>> = (0..TinyConfig::envelope(16).max_n_a)
            .map(|j| {
                let start = j * depth;
                let end = start + depth;
                CyclotomicRing::gadget_recompose_pow2_i8(&t_hat_block[start..end], log_basis)
            })
            .collect();
        assert_eq!(lhs, rhs);
    }

    let t_hat_flat_ring: Vec<CyclotomicRing<F, D>> = w
        .t_hat
        .flat_digits()
        .iter()
        .map(|plane| {
            let coeffs: [F; D] = from_fn(|k| F::from_i64(plane[k] as i64));
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let outer = mat_vec_mul(
        &psetup.expanded.shared_matrix,
        TinyConfig::envelope(16).max_n_b,
        psetup.expanded.seed.max_stride,
        &t_hat_flat_ring,
    );
    assert_eq!(outer, w.commitment.u);
}

#[test]
fn small_test_config_has_expected_shape() {
    assert_eq!(SmallTestCommitmentConfig::D, 32);
    let lp = SmallTestCommitmentConfig::commitment_layout(8).unwrap();
    assert_eq!(lp.block_len, 16);
    assert_eq!(lp.num_blocks, 4);
    let depth = lp.num_digits_commit;
    assert!(depth > 0);
}

#[test]
fn setup_rejects_mismatched_degree() {
    let err = HachiProverSetup::<F, D>::new::<BadDegreeConfig>(16, 1).unwrap_err();
    match err {
        HachiError::InvalidSetup(msg) => assert!(msg.contains("mismatches")),
        other => panic!("unexpected error: {other:?}"),
    }
}
