//! Equivalence and boundary tests for the secret-input algebra entry points.

use crate::ring::cyclotomic::BalancedDecomposePow2Params;
use crate::*;
use jolt_field::{Prime128Offset6A23801F, Prime32Offset99, Prime64Offset59};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn check_digits<F: Field + CanonicalEncoding>() {
    let q = (-F::one()).to_u128_checked().unwrap() + 1;
    let bits = 128 - (q - 1).leading_zeros();
    let mut rng = StdRng::seed_from_u64(0x000a_1c7d_1617);
    for log_basis in 1..=8 {
        for levels in [
            0,
            1,
            (bits / log_basis) as usize,
            bits.div_ceil(log_basis) as usize,
        ] {
            let params = BalancedDecomposePow2Params::<F>::new(levels, log_basis);
            let threshold =
                crate::ring::cyclotomic::decompose_centering_threshold(levels, log_basis, q);
            let half = 1u128 << (log_basis - 1);
            let mut coefficients = [
                0,
                1,
                half - 1,
                half,
                half + 1,
                q - half,
                q - half - 1,
                q / 2,
                q / 2 + 1,
                q - 1,
                threshold,
                threshold.saturating_add(1).min(q - 1),
            ]
            .map(F::from_u128_reduced)
            .to_vec();
            coefficients.extend((0..193).map(|_| F::from_u128_reduced(rng.gen())));
            // Empty, scalar, SIMD boundaries, and a partial chunk.
            for len in [0, 1, 4, 8, 63, 64, 65, coefficients.len()] {
                let mut expected = vec![0; len * levels];
                balanced_decompose_coefficients_pow2_i8_into(
                    &coefficients[..len],
                    &mut expected,
                    &params,
                );
                let mut actual = vec![42; len * levels];
                balanced_decompose_coefficients_pow2_i8_into_ct(
                    &coefficients[..len],
                    &mut actual,
                    &params,
                )
                .unwrap();
                assert_eq!(
                    actual, expected,
                    "basis={log_basis}, levels={levels}, len={len}"
                );
            }
        }
    }
}

#[test]
fn ct_digits_match_variable_time_random_and_boundaries() {
    check_digits::<Prime32Offset99>();
    check_digits::<Prime64Offset59>();
    check_digits::<Prime128Offset6A23801F>();
}

#[test]
fn ct_digits_reject_public_shape_and_basis_without_writing() {
    type F = Prime128Offset6A23801F;
    let mut output = [42];
    let params = BalancedDecomposePow2Params::<F>::new(2, 8);
    assert!(
        balanced_decompose_coefficients_pow2_i8_into_ct(&[F::zero()], &mut output, &params)
            .is_err()
    );
    assert_eq!(output, [42]);
    let params = BalancedDecomposePow2Params::<F>::new(1, 9);
    assert!(
        balanced_decompose_coefficients_pow2_i8_into_ct(&[F::zero()], &mut output, &params)
            .is_err()
    );
    assert_eq!(output, [42]);
}

fn check_lifts<F: Field + CanonicalEncoding>() {
    let mut rng = StdRng::seed_from_u64(0x00a1_11f7);
    for x in [
        i64::MIN,
        i64::MIN + 1,
        -129,
        -128,
        -1,
        0,
        1,
        127,
        128,
        i64::MAX,
    ]
    .into_iter()
    .chain((0..1024).map(|_| rng.gen()))
    {
        assert_eq!(lift_ct::<F>(x), F::from_i64(x));
        assert_eq!(lift_biased::<F>(x) - F::from_u64(LIFT_BIAS), F::from_i64(x));
    }
}

#[test]
fn signed_lifts_match_variable_time_at_both_i64_extremes() {
    check_lifts::<Prime32Offset99>();
    check_lifts::<Prime64Offset59>();
    check_lifts::<Prime128Offset6A23801F>();
}

fn check_crt<F: CrtNttConvertibleField, W: PrimeWidth, const K: usize, const D: usize>(
    params: &CrtNttParamSet<W, K, D>,
) {
    let mut rng = StdRng::seed_from_u64(0x000a_1c47);
    for round in 0..24 {
        let input = CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|index| {
            match (round + index) % 7 {
                0 => F::zero(),
                1 => -F::one(),
                2 => F::one(),
                _ => F::from_u128_reduced(rng.gen()),
            }
        }));
        let ntt = CyclotomicCrtNtt::from_ring(&input, params);
        // Small CRT profiles need not hold a full field coefficient, but the
        // two reconstruction paths must agree even when the integer wraps.
        assert_eq!(ntt.to_ring_ct::<F>(params), ntt.to_ring::<F>(params));
    }
}

#[test]
fn ct_reconstruction_matches_all_residue_widths_and_fields() {
    check_crt::<Prime128Offset6A23801F, _, 6, 64>(&CrtNttParamSet::new(tables::q128_primes()));
    check_crt::<Prime64Offset59, _, 3, 64>(&CrtNttParamSet::new(tables::Q64_PRIMES));
    check_crt::<Prime32Offset99, _, 2, 64>(&CrtNttParamSet::new(tables::Q32_PRIMES));
    check_crt::<Prime128Offset6A23801F, _, 1, 64>(&CrtNttParamSet::new([tables::I16_TAIL_PRIME]));
}

#[test]
fn ct_i32_lift_matches_full_range_and_digit_boundaries() {
    type F = Prime128Offset6A23801F;
    const D: usize = 64;
    let params = CrtNttParamSet::new(tables::q128_primes());
    let mut rng = StdRng::seed_from_u64(0x000a_1132);
    for _ in 0..32 {
        let mut coeffs = std::array::from_fn::<_, D, _>(|_| rng.gen::<i32>());
        coeffs[..8].copy_from_slice(&[i32::MIN, i32::MAX, -129, -128, -1, 0, 1, 127]);
        let mut ct = CyclotomicCrtNtt::zero();
        ct.assign_centered_i32_ct(&coeffs, &params);
        let expected =
            CyclotomicRing::<F, D>::from_coefficients(coeffs.map(|x| F::from_i64(x.into())));
        assert_eq!(ct.to_ring_ct::<F>(&params), expected);
        assert_eq!(ct.to_ring::<F>(&params), expected);
        let vt = CyclotomicCrtNtt::from_centered_coefficients(&coeffs.map(i128::from), &params);
        assert_eq!(ct.to_ring_ct::<F>(&params), vt.to_ring::<F>(&params));
    }
}
