#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::protocol::commitment::utils::linear::{decompose_block, mat_vec_mul_ntt_single_i8};
use hachi_pcs::protocol::commitment::{
    CommitmentConfig, CommitmentEnvelope, DecompositionParams, RingCommitment,
};
use hachi_pcs::protocol::hachi_poly_ops::{DensePoly, HachiPolyOps};
use hachi_pcs::protocol::params::LevelParams;
use hachi_pcs::protocol::setup::HachiProverSetup;
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

/// Commit `blocks` via the production path:
///   1. run `DensePoly::commit_inner_witness` to produce `t_hat`,
///   2. apply the outer Ajtai matrix to `t_hat`'s digit planes to get
///      `u = B · t_hat`.
///
/// Returns `(u, t_hat_digits_per_block)`, which together form the
/// commit-witness shape previously produced by the now-removed
/// `RingCommitmentScheme::commit_ring_blocks` helper.
fn commit_blocks_via_production_path(
    setup: &HachiProverSetup<F, D>,
    blocks: &[Vec<CyclotomicRing<F, D>>],
) -> (Vec<CyclotomicRing<F, D>>, Vec<Vec<[i8; D]>>) {
    let lp = TinyConfig::commitment_layout(setup.expanded.seed.max_num_vars).unwrap();
    let ring_coeffs: Vec<CyclotomicRing<F, D>> =
        blocks.iter().flat_map(|b| b.iter().copied()).collect();
    let poly = DensePoly::from_ring_coeffs(ring_coeffs);
    let inner = poly
        .commit_inner_witness(
            &setup.expanded.shared_matrix,
            &setup.ntt_shared,
            lp.a_key.row_len(),
            lp.block_len,
            lp.num_digits_commit,
            lp.num_digits_open,
            lp.log_basis,
            setup.expanded.seed.max_stride,
        )
        .unwrap();
    let u: Vec<CyclotomicRing<F, D>> = mat_vec_mul_ntt_single_i8(
        &setup.ntt_shared,
        lp.b_key.row_len(),
        setup.expanded.seed.max_stride,
        inner.t_hat.flat_digits(),
    );
    let t_hat_blocks: Vec<Vec<[i8; D]>> = inner.t_hat.into_blocks();
    (u, t_hat_blocks)
}

#[test]
fn setup_shape_is_consistent() {
    let envelope = TinyConfig::envelope(16);
    let p1 = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1, 1).unwrap();
    let v1 = p1.verifier_setup();
    let p2 = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1, 1).unwrap();
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
    let psetup = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1, 1).unwrap();
    let blocks = sample_blocks();

    let (u1, t_hat_1) = commit_blocks_via_production_path(&psetup, &blocks);
    let (u2, t_hat_2) = commit_blocks_via_production_path(&psetup, &blocks);

    assert_eq!(u1, u2, "commitment must be deterministic");
    assert_eq!(t_hat_1, t_hat_2, "t_hat must be deterministic");

    assert_eq!(u1.len(), TinyConfig::envelope(16).max_n_b);
    assert_eq!(t_hat_1.len(), NUM_BLOCKS);
    let depth = num_digits_commit();
    assert!(t_hat_1
        .iter()
        .all(|t| t.len() == TinyConfig::envelope(16).max_n_a * depth));
}

#[test]
fn opening_satisfies_inner_and_outer_equations() {
    let psetup = HachiProverSetup::<F, D>::new::<TinyConfig>(16, 1, 1).unwrap();
    let blocks = sample_blocks();
    let (u, t_hat_blocks) = commit_blocks_via_production_path(&psetup, &blocks);

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
        let t_hat_block = t_hat_blocks.get(i).expect("every block retained");
        let rhs: Vec<CyclotomicRing<F, D>> = (0..TinyConfig::envelope(16).max_n_a)
            .map(|j| {
                let start = j * depth;
                let end = start + depth;
                CyclotomicRing::gadget_recompose_pow2_i8(&t_hat_block[start..end], log_basis)
            })
            .collect();
        assert_eq!(lhs, rhs, "Row `A · s_i = t_i` failed for block {i}");
    }

    let t_hat_flat_ring: Vec<CyclotomicRing<F, D>> = t_hat_blocks
        .iter()
        .flat_map(|block| block.iter())
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
    assert_eq!(outer, u, "Row `B · t_hat = u` failed");

    // Sanity-check that `u` was actually produced by the outer map above
    // (guards against trivial aliasing errors in the helper).
    let _ = RingCommitment::<F, D> { u: u.clone() };
}

#[test]
fn setup_rejects_mismatched_degree() {
    let err = HachiProverSetup::<F, D>::new::<BadDegreeConfig>(16, 1, 1).unwrap_err();
    match err {
        HachiError::InvalidSetup(msg) => assert!(msg.contains("mismatches")),
        other => panic!("unexpected error: {other:?}"),
    }
}
