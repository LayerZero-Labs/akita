#![allow(missing_docs)]

use hachi_pcs::algebra::{CyclotomicRing, Fp32, Fp64, Prime128M8M4M1M0};
use hachi_pcs::protocol::ajtai::ajtai_commit::AjtaiCommitmentScheme;
use hachi_pcs::protocol::ajtai::coeff::{CoeffAjtai, CoeffAjtaiConfig};
use hachi_pcs::protocol::ajtai::ntt_backend::NttAjtaiBackend;
use hachi_pcs::protocol::commitment::utils::crt_ntt::{build_ntt_slot, NttSlotCache};
use hachi_pcs::protocol::commitment::utils::flat_matrix::FlatMatrix;
use hachi_pcs::{CanonicalField, FieldCore, FromSmallInt};

const WITNESS_LEN: usize = 40;
const NUM_WITNESS_ROWS: usize = 12;
const TEST_SALT_BASE: usize = 11;

fn test_config() -> CoeffAjtaiConfig {
    CoeffAjtaiConfig {
        inner_rows: 16,
        outer_rows: 10,
        num_digits: 4,
        decompose_modulus: 8,
    }
}

fn sample_small_instances<F: FieldCore + FromSmallInt, const D: usize>(
    rows: usize,
    cols: usize,
    salt: usize,
) -> Vec<Vec<CyclotomicRing<F, D>>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                        let v = ((r * 17 + c * 13 + k * 7 + salt) % 7) as i64 - 3;
                        F::from_i64(v)
                    }))
                })
                .collect()
        })
        .collect()
}

fn build_slot<F: FieldCore + CanonicalField, const D: usize>(
    mat: &[Vec<CyclotomicRing<F, D>>],
) -> NttSlotCache<D> {
    let flat = FlatMatrix::from_ring_matrix(mat);
    build_ntt_slot(flat.view::<D>()).unwrap()
}

fn assert_ntt_matches_coeff<F: FieldCore + CanonicalField + FromSmallInt, const D: usize>() {
    let cfg = test_config();
    let witness = sample_small_instances::<F, D>(NUM_WITNESS_ROWS, WITNESS_LEN, TEST_SALT_BASE);
    let matrix_a = sample_small_instances::<F, D>(cfg.inner_rows, WITNESS_LEN, TEST_SALT_BASE + 1);

    let decomp_len = NUM_WITNESS_ROWS * cfg.inner_rows * cfg.num_digits;
    let matrix_b = sample_small_instances::<F, D>(cfg.outer_rows, decomp_len, TEST_SALT_BASE + 2);

    let a_ntt = build_slot::<F, D>(&matrix_a);
    let b_ntt = build_slot::<F, D>(&matrix_b);

    let (coeff_t_hat, coeff_u) =
        CoeffAjtai::two_tier_commit(&matrix_a, &matrix_b, &witness, &cfg).unwrap();
    let (ntt_t_hat, ntt_u) = <NttAjtaiBackend as AjtaiCommitmentScheme<F, D>>::two_tier_commit(
        &a_ntt, &b_ntt, &witness, &cfg,
    )
    .unwrap();

    assert!(coeff_t_hat == ntt_t_hat);
    assert!(coeff_u == ntt_u);
}

#[test]
fn ntt_matches_coeff_fp32() {
    type F = Fp32<4294967197u32>;
    const D: usize = 64;
    assert_ntt_matches_coeff::<F, D>();
}

#[test]
fn ntt_matches_coeff_fp64() {
    type F = Fp64<4294967197u64>;
    const D: usize = 64;
    assert_ntt_matches_coeff::<F, D>();
}

#[test]
fn ntt_matches_coeff_fp128() {
    type F = Prime128M8M4M1M0;
    const D: usize = 64;
    assert_ntt_matches_coeff::<F, D>();
}
