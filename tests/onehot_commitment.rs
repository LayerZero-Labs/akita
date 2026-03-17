#![allow(missing_docs)]

use hachi_pcs::algebra::CyclotomicRing;
use hachi_pcs::protocol::commitment::{HachiCommitmentCore, RingCommitmentScheme};
use hachi_pcs::{FieldCore, FromSmallInt};
use hachi_test_support::*;

type Core = HachiCommitmentCore;

fn psetup() -> <Core as RingCommitmentScheme<F, D, TinyConfig>>::ProverSetup {
    <Core as RingCommitmentScheme<F, D, TinyConfig>>::setup(16)
        .unwrap()
        .0
}

/// Compare the optimized one-hot path against the default dense path.
///
/// The default implementation materializes the full vector and calls
/// `commit_coeffs`. The optimized impl uses sparse inner Ajtai.
/// Both must produce identical (commitment, s_all, t_hat_all).
fn assert_onehot_matches_dense(onehot_k: usize, indices: &[usize]) {
    let opt_indices: Vec<Option<usize>> = indices.iter().map(|&i| Some(i)).collect();
    let setup = psetup();

    // Optimized sparse path.
    let w_sparse = <Core as RingCommitmentScheme<F, D, TinyConfig>>::commit_onehot(
        onehot_k,
        &opt_indices,
        &setup,
    )
    .unwrap();

    // Reference: materialize the full one-hot vector, pack into ring elements,
    // and commit via the dense path.
    let total_field = indices.len() * onehot_k;
    let total_ring = total_field / D;
    let mut field_elems = vec![F::zero(); total_field];
    for (c, &idx) in indices.iter().enumerate() {
        field_elems[c * onehot_k + idx] = F::from_u64(1);
    }
    let ring_coeffs: Vec<CyclotomicRing<F, D>> = (0..total_ring)
        .map(|r| {
            let coeffs: [F; D] = std::array::from_fn(|i| field_elems[r * D + i]);
            CyclotomicRing::from_coefficients(coeffs)
        })
        .collect();
    let w_dense =
        <Core as RingCommitmentScheme<F, D, TinyConfig>>::commit_coeffs(&ring_coeffs, &setup)
            .unwrap();

    assert_eq!(
        w_sparse.commitment, w_dense.commitment,
        "commitments must match"
    );
    assert_eq!(
        w_sparse.t_hat, w_dense.t_hat,
        "t_hat_all (decomposed inner output) must match"
    );
}

#[test]
fn onehot_k_gt_d_basic() {
    // K=128, D=64 => K/D=2, T=2 => T*K=256 => 4 ring elements
    assert_onehot_matches_dense(128, &[0, 64]);
}

#[test]
fn onehot_k_gt_d_various_positions() {
    assert_onehot_matches_dense(128, &[127, 0]);
    assert_onehot_matches_dense(128, &[63, 65]);
    assert_onehot_matches_dense(128, &[32, 96]);
}

#[test]
fn onehot_k_much_gt_d() {
    // K=256, D=64 => K/D=4, T=1 => T*K=256 => 4 ring elements
    assert_onehot_matches_dense(256, &[0]);
    assert_onehot_matches_dense(256, &[63]);
    assert_onehot_matches_dense(256, &[64]);
    assert_onehot_matches_dense(256, &[255]);
    assert_onehot_matches_dense(256, &[100]);
}

#[test]
fn onehot_k_eq_d_basic() {
    // K=64=D, T=4 => 4 ring elements, each is a monomial X^{idx}.
    assert_onehot_matches_dense(64, &[0, 0, 0, 0]);
}

#[test]
fn onehot_k_eq_d_varied() {
    assert_onehot_matches_dense(64, &[0, 31, 32, 63]);
    assert_onehot_matches_dense(64, &[1, 2, 3, 4]);
    assert_onehot_matches_dense(64, &[63, 63, 63, 63]);
}

#[test]
fn onehot_k_lt_d_basic() {
    // K=16, D=64 => D/K=4, T=16 => T*K=256 => 4 ring elements.
    // Each ring element spans 4 chunks, so has 4 nonzero coefficients.
    let indices: Vec<usize> = (0..16).map(|i| i % 16).collect();
    assert_onehot_matches_dense(16, &indices);
}

#[test]
fn onehot_k_lt_d_all_zeros() {
    let indices = vec![0; 16];
    assert_onehot_matches_dense(16, &indices);
}

#[test]
fn onehot_k_lt_d_all_max() {
    let indices = vec![15; 16];
    assert_onehot_matches_dense(16, &indices);
}

#[test]
fn onehot_k_lt_d_mixed() {
    let indices = vec![0, 15, 7, 3, 12, 1, 8, 14, 5, 10, 2, 9, 6, 11, 4, 13];
    assert_onehot_matches_dense(16, &indices);
}

#[test]
fn onehot_k_lt_d_ratio_2() {
    // K=32, D=64 => D/K=2, T=8 => T*K=256 => 4 ring elements.
    let indices = vec![0, 31, 16, 8, 24, 4, 12, 20];
    assert_onehot_matches_dense(32, &indices);
}

#[test]
fn onehot_rejects_non_divisible_k_and_d() {
    let setup = psetup();
    let result = <Core as RingCommitmentScheme<F, D, TinyConfig>>::commit_onehot(
        17,
        &[Some(0usize); 4],
        &setup,
    );
    assert!(result.is_err());
}

#[test]
fn onehot_rejects_out_of_range_index() {
    let setup = psetup();
    let result = <Core as RingCommitmentScheme<F, D, TinyConfig>>::commit_onehot(
        64,
        &[Some(0usize), Some(64), Some(0), Some(0)],
        &setup,
    );
    assert!(result.is_err());
}

#[test]
fn onehot_rejects_wrong_total_size() {
    let setup = psetup();
    let result = <Core as RingCommitmentScheme<F, D, TinyConfig>>::commit_onehot(
        64,
        &[Some(0usize), Some(0), Some(0)],
        &setup,
    );
    assert!(result.is_err());
}
