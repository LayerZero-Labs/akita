use super::coeffs::balanced_decompose_centered_i32_i8_into;
use super::coeffs::build_w_coeffs;
use crate::protocol::ring_relation::RelationQuotientOutput;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallengeConfig;
use akita_field::Prime128OffsetA7F7;
use akita_types::{
    r_decomp_levels, CommitmentRingDims, DigitBlocks, LevelParams, SisModulusFamily,
};
use std::array::from_fn;

#[test]
fn centered_i32_decompose_matches_ring_decompose() {
    type F = Prime128OffsetA7F7;
    const D: usize = 128;

    let centered = from_fn(|i| ((37 * i as i32 + 11) % 95) - 47);
    let ring =
        CyclotomicRing::<F, D>::from_coefficients(from_fn(|i| F::from_i64(centered[i] as i64)));

    for (num_digits, log_basis) in [
        (7usize, 3u32),
        (10usize, 2u32),
        (5usize, 5u32),
        (4usize, 6u32),
    ] {
        let mut got = vec![[0i8; D]; num_digits];
        balanced_decompose_centered_i32_i8_into(&centered, &mut got, log_basis);

        let mut expected = vec![[0i8; D]; num_digits];
        ring.balanced_decompose_pow2_i8_into(&mut expected, log_basis);
        assert_eq!(
            got, expected,
            "centered i32 decomposition mismatch for num_digits={num_digits} log_basis={log_basis}"
        );
    }
}

#[test]
fn build_w_coeffs_accepts_mixed_opening_stride_and_mixed_quotient_rows() {
    type F = Prime128OffsetA7F7;
    const D_A: usize = 128;
    const D_D: usize = 32;

    let lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        D_A,
        4,
        1,
        1,
        1,
        SparseChallengeConfig::Uniform {
            weight: 3,
            nonzero_coeffs: vec![-1, 1],
        },
    )
    .with_decomp(0, 0, 1, 2, 0)
    .expect("valid level params")
    .with_role_dims(CommitmentRingDims {
        inner: D_A,
        outer: 64,
        opening: D_D,
    })
    .expect("nested role dims");

    let num_claims = 1;
    let e_hat = DigitBlocks::zeroed(vec![lp.num_digits_open], D_D).expect("opening digits");
    let t_hat = DigitBlocks::zeroed(vec![lp.a_key.row_len() * lp.num_digits_open], D_A)
        .expect("inner digits");
    let z_folded_centered_per_chunk = vec![vec![[0i32; D_A]; lp.inner_width()]];

    let mut r = RelationQuotientOutput::new();
    r.push_ring(CyclotomicRing::<F, D_A>::zero());
    r.push_ring(CyclotomicRing::<F, D_D>::zero());

    let witness = build_w_coeffs::<F, D_A>(
        &e_hat,
        &t_hat,
        &z_folded_centered_per_chunk,
        &r,
        &lp,
        num_claims,
    )
    .expect("mixed-stride witness assembly");

    let z_digits = lp.inner_width() * lp.num_digits_fold(num_claims, 128).unwrap() * D_A;
    let r_digits = (D_A + D_D) * r_decomp_levels::<F>(lp.log_basis);
    assert_eq!(
        witness.len(),
        z_digits + e_hat.digits().len() + t_hat.digits().len() + r_digits
    );
}
