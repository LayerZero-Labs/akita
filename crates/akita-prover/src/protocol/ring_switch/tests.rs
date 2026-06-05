use super::coeffs::balanced_decompose_centered_i32_i8_into;
#[cfg(not(feature = "zk"))]
use super::coeffs::build_terminal_direct_w_coeffs;
use akita_algebra::CyclotomicRing;
use akita_field::Prime128OffsetA7F7;
#[cfg(not(feature = "zk"))]
use akita_types::{FlatDigitBlocks, LevelParams, SisModulusFamily};
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

#[cfg(not(feature = "zk"))]
#[test]
fn terminal_direct_witness_coeffs_emit_no_r_hat_suffix() {
    type F = Prime128OffsetA7F7;
    const D: usize = 2;

    let lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        D,
        3,
        1,
        1,
        0,
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        },
    )
    .with_decomp(1, 1, 1, 1, 1)
    .expect("level params");
    let w_hat = FlatDigitBlocks::from_blocks(vec![vec![[1, 2]], vec![[-1, -2]]]);
    let t_hat = FlatDigitBlocks::from_blocks(vec![vec![[3, -3]], vec![[0, 1]]]);
    let z_pre_centered = [[2, -2], [1, 0]];

    let got = build_terminal_direct_w_coeffs::<F, D>(&w_hat, &t_hat, &z_pre_centered, &lp, 1);

    assert_eq!(
        got.as_i8_digits(),
        &[2, -2, 1, 0, 0, 0, 0, 0, 1, 2, -1, -2, 3, -3, 0, 1],
        "terminal direct witness must contain z_pre, w_hat, and t_hat only"
    );
}

#[cfg(not(feature = "zk"))]
#[test]
fn terminal_direct_witness_coeffs_preserve_w_first_order() {
    type F = Prime128OffsetA7F7;
    const D: usize = 2;

    let lp = LevelParams::params_only(
        SisModulusFamily::Q128,
        D,
        3,
        1,
        1,
        0,
        akita_challenges::SparseChallengeConfig::Uniform {
            weight: 1,
            nonzero_coeffs: vec![-1, 1],
        },
    )
    .with_decomp(0, 1, 1, 1, 1)
    .expect("level params");
    let w_hat = FlatDigitBlocks::from_blocks(vec![vec![[1, 2]], vec![[-1, -2]]]);
    let t_hat = FlatDigitBlocks::from_blocks(vec![vec![[3, -3]], vec![[0, 1]]]);
    let z_pre_centered = [[2, -2]];

    let got = build_terminal_direct_w_coeffs::<F, D>(&w_hat, &t_hat, &z_pre_centered, &lp, 1);

    assert_eq!(
        got.as_i8_digits(),
        &[1, 2, -1, -2, 3, -3, 0, 1, 2, -2, 0, 0],
        "w-first terminal direct witness must omit only the r_hat suffix"
    );
}
