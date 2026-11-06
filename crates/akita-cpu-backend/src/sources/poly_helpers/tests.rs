use super::rotated_accum::accumulate_rotated_digit_plane;
use super::{
    fill_rotated_challenge, sparse_mul_acc, sparse_mul_acc_i16, sparse_mul_acc_i16_pm1,
    sparse_mul_acc_i16_scalar, sparse_mul_acc_pm1, sparse_mul_acc_scalar, SignedDigitBasis,
    ValidatedSparseChallenge, ValidatedSparseChallenges,
};
use akita_algebra::ring::cyclotomic::BalancedDecomposePow2Params;
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use jolt_algebra::{Fp64, Ring};

/// SIMD-vs-scalar parity for the sparse-multiply-accumulate decompose-fold
/// kernel, exercising whichever SIMD backend is active (NEON / AVX2 /
/// AVX-512). Restricted to `|coeff| <= 2` so the SIMD fast path fires.
/// `D = 128` matches typical small-field schedules and gives both kernels
/// multiple full-width iterations to chew through.
#[test]
fn sparse_mul_acc_simd_matches_scalar_small_coeffs() {
    const D: usize = 128;

    // Construct a small-coefficient challenge that hits both positive and
    // negative paths for both magnitudes 1 and 2. Positions cover both the
    // pure-prefix (split == D, no wrap) and the wrap-around case.
    let positions: Vec<u32> = (0..32u32).map(|k| k * 4).collect();
    let coeffs: Vec<i8> = (0..32)
        .map(|k| match k % 4 {
            0 => 1,
            1 => -1,
            2 => 2,
            _ => -2,
        })
        .collect();
    let challenge = SparseChallenge {
        positions: positions.into(),
        coeffs: coeffs.into(),
    };

    let digit_plane: [i8; D] = std::array::from_fn(|k| (((7 * k as i64) % 13) - 6) as i8);

    let mut simd_acc = [0i32; D];
    let mut scalar_acc = [0i32; D];

    sparse_mul_acc::<D>(
        &digit_plane,
        ValidatedSparseChallenge::new(&challenge).unwrap(),
        &mut simd_acc,
    );
    sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

    assert_eq!(
        simd_acc, scalar_acc,
        "SIMD sparse_mul_acc disagreed with scalar reference"
    );
}

#[test]
fn sparse_mul_acc_i16_simd_matches_scalar() {
    const D: usize = 128;
    let challenge = SparseChallenge {
        positions: (0..32u32).map(|k| k * 4).collect(),
        coeffs: (0..32)
            .map(|k| match k % 4 {
                0 => 1,
                1 => -1,
                2 => 2,
                _ => -2,
            })
            .collect(),
    };
    let digit_plane: [i16; D] = std::array::from_fn(|k| (((811 * k as i64) % 1024) - 512) as i16);
    let mut simd_acc = [0i32; D];
    let mut scalar_acc = [0i32; D];
    sparse_mul_acc_i16::<D>(
        &digit_plane,
        ValidatedSparseChallenge::new(&challenge).unwrap(),
        &mut simd_acc,
    );
    sparse_mul_acc_i16_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);
    assert_eq!(simd_acc, scalar_acc);
}

#[test]
fn prepared_pm1_kernels_match_generic_sparse_accumulation() {
    const D: usize = 256;
    let positive = vec![0, 17, 61, 128, 251];
    let negative = vec![3, 29, 97, 191, 255];
    let challenge = SparseChallenge {
        positions: positive.iter().chain(&negative).copied().collect(),
        coeffs: std::iter::repeat_n(1, positive.len())
            .chain(std::iter::repeat_n(-1, negative.len()))
            .collect(),
    };

    let i8_plane = std::array::from_fn(|index| ((index * 17) % 127) as i8 - 63);
    let mut expected_i8 = [0i32; D];
    sparse_mul_acc(
        &i8_plane,
        ValidatedSparseChallenge::new(&challenge).unwrap(),
        &mut expected_i8,
    );
    let mut actual_i8 = [0i32; D];
    sparse_mul_acc_pm1(&i8_plane, &positive, &negative, &mut actual_i8);
    assert_eq!(actual_i8, expected_i8);

    let i16_plane = std::array::from_fn(|index| ((index * 509) % 1024) as i16 - 512);
    let mut expected_i16 = [0i32; D];
    sparse_mul_acc_i16(
        &i16_plane,
        ValidatedSparseChallenge::new(&challenge).unwrap(),
        &mut expected_i16,
    );
    let mut actual_i16 = [0i32; D];
    sparse_mul_acc_i16_pm1(&i16_plane, &positive, &negative, &mut actual_i16);
    assert_eq!(actual_i16, expected_i16);
}

#[test]
fn sparse_mul_acc_rejects_out_of_range_challenge_before_dispatch() {
    const D: usize = 64;
    let challenge = SparseChallenge {
        positions: vec![D as u32].into(),
        coeffs: vec![1].into(),
    };
    assert_eq!(
        ValidatedSparseChallenge::<D>::new(&challenge).unwrap_err(),
        akita_error::AkitaError::InvalidInput(
            "sparse fold challenge position exceeds the ring dimension".into()
        ),
    );
}

#[test]
fn sparse_fold_batch_rejects_mismatched_support_lengths() {
    let challenges = [SparseChallenge {
        positions: vec![0, 1].into(),
        coeffs: vec![1].into(),
    }];
    assert!(matches!(
        ValidatedSparseChallenges::<64>::new(&challenges, 1, 1),
        Err(akita_error::AkitaError::InvalidSize {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn signed_digit_basis_rejects_unsupported_exponents() {
    for log_basis in [0, 17, u32::MAX] {
        assert_eq!(
            SignedDigitBasis::new(log_basis).unwrap_err(),
            akita_error::AkitaError::InvalidInput(
                "signed-digit log basis is outside the supported range".into()
            ),
        );
    }
}

/// Edge case: challenge with `pos == 0` so `split == D` and the second
/// (wrap) segment is empty.
#[test]
fn sparse_mul_acc_simd_zero_position() {
    const D: usize = 64;
    let challenge = SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    };
    let digit_plane: [i8; D] = std::array::from_fn(|k| (k as i8) - 32);

    let mut simd_acc = [0i32; D];
    let mut scalar_acc = [0i32; D];
    sparse_mul_acc::<D>(
        &digit_plane,
        ValidatedSparseChallenge::new(&challenge).unwrap(),
        &mut simd_acc,
    );
    sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

    assert_eq!(simd_acc, scalar_acc);
}

/// Edge case: challenge with `pos == D - 1` so `split == 1` and the
/// post-split (wrap) segment is the bulk of the work.
#[test]
fn sparse_mul_acc_simd_max_position() {
    const D: usize = 64;
    let challenge = SparseChallenge {
        positions: vec![(D - 1) as u32].into(),
        coeffs: vec![-2].into(),
    };
    let digit_plane: [i8; D] = std::array::from_fn(|k| ((k as i8) - 32).wrapping_mul(3));

    let mut simd_acc = [0i32; D];
    let mut scalar_acc = [0i32; D];
    sparse_mul_acc::<D>(
        &digit_plane,
        ValidatedSparseChallenge::new(&challenge).unwrap(),
        &mut simd_acc,
    );
    sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

    assert_eq!(simd_acc, scalar_acc);
}

#[test]
fn rotated_digit_plane_accumulate_matches_generic_sparse_path() {
    type F = Fp64<4294967197>;
    const D: usize = 32;
    let num_digits = 4;
    let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
        let v = ((7 * k as i64) % 17) - 8;
        F::from_i64(v)
    }));
    let challenge = SparseChallenge {
        positions: (0..D as u32).collect(),
        coeffs: (0..D)
            .map(|k| match k % 5 {
                0 => -3,
                1 => -1,
                2 => 1,
                3 => 2,
                _ => 4,
            })
            .collect(),
    };
    let log_basis = 3u32;
    let params = BalancedDecomposePow2Params::new(num_digits, log_basis);

    let mut digits = vec![[0i8; D]; num_digits];
    ring.balanced_decompose_pow2_i8_into_with_params(&mut digits, &params);
    let mut generic_acc = vec![[0i32; D]; num_digits];
    for digit in 0..num_digits {
        sparse_mul_acc::<D>(
            &digits[digit],
            ValidatedSparseChallenge::new(&challenge).unwrap(),
            &mut generic_acc[digit],
        );
    }

    let mut rotated = vec![[0i16; D]; D];
    fill_rotated_challenge::<D>(&mut rotated, &challenge);
    let mut rotated_i8_acc = vec![[0i32; D]; num_digits];
    let mut rotated_i16_acc = vec![[0i32; D]; num_digits];
    for digit in 0..num_digits {
        accumulate_rotated_digit_plane(&digits[digit], &rotated, &mut rotated_i8_acc[digit]);
        accumulate_rotated_digit_plane(
            &digits[digit].map(i16::from),
            &rotated,
            &mut rotated_i16_acc[digit],
        );
    }

    assert_eq!(rotated_i8_acc, generic_acc);
    assert_eq!(rotated_i16_acc, generic_acc);
}
