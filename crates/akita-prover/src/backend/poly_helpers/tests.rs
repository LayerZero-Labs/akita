use super::rotated_accum::{
    decompose_ring_full_challenge_accumulate, should_use_rotated_challenge,
};
use super::{
    balanced_ring_decompose_fold_partitioned, decompose_ring_interleaved, fill_rotated_challenge,
    sparse_mul_acc, sparse_mul_acc_i16, sparse_mul_acc_i16_pm1, sparse_mul_acc_i16_scalar,
    sparse_mul_acc_pm1, sparse_mul_acc_scalar, DecomposeParams,
};
use akita_algebra::CyclotomicRing;
use akita_challenges::SparseChallenge;
use akita_field::CanonicalField;
use akita_field::{Fp64, Prime128Offset275};
use akita_types::sis::compute_num_digits_field_width;

#[test]
fn compact_subfield_fold_matches_materialized_ring_oracle_for_all_sources() {
    use crate::backend::{DensePoly, OneHotPoly, RecursiveWitnessFlat, SparseRingPoly};
    use akita_field::{ExtField, FpExt4, Prime32Offset99};
    use akita_types::{prepare_opening_point, BasisMode};

    type F = Prime32Offset99;
    type E = FpExt4<F>;
    const D: usize = 32;
    const POSITIONS: usize = 4;

    let mut point = vec![E::zero(); 8];
    for (index, coordinate) in point[5..].iter_mut().enumerate() {
        *coordinate = E::from_base_slice(&[
            F::from_u64(index as u64 + 2),
            F::from_u64(3 * index as u64 + 5),
            F::from_u64(5 * index as u64 + 7),
            F::from_u64(7 * index as u64 + 11),
        ]);
    }
    let prepared = prepare_opening_point::<F, E, D>(
        &point,
        BasisMode::Lagrange,
        POSITIONS,
        2,
        D.trailing_zeros() as usize,
    )
    .expect("valid compact opening point");
    let multipliers = &prepared.ring_multiplier_point;
    let subfield_multipliers = multipliers
        .as_subfield()
        .expect("proper extension multipliers");
    let position_rings = multipliers
        .materialize_position_rings::<D>()
        .expect("valid ring dimension")
        .expect("proper extension multipliers");
    let fold_rings = multipliers
        .materialize_fold_rings::<D>()
        .expect("valid ring dimension")
        .expect("proper extension multipliers");

    let assert_output = |actual: (CyclotomicRing<F, D>, Vec<CyclotomicRing<F, D>>),
                         expected_folded: Vec<CyclotomicRing<F, D>>| {
        let expected_eval = expected_folded
            .iter()
            .zip(&fold_rings)
            .fold(CyclotomicRing::zero(), |acc, (folded, weight)| {
                acc + *folded * *weight
            });
        assert_eq!(actual.1, expected_folded);
        assert_eq!(actual.0, expected_eval);
    };

    let dense = DensePoly::from_ring_coeffs::<D>(
        (0..8)
            .map(|ring| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_u64((ring * D + coefficient + 1) as u64)
                }))
            })
            .collect(),
    );
    assert_output(
        dense
            .evaluate_and_fold_subfield(subfield_multipliers, POSITIONS)
            .expect("dense compact fold"),
        dense.fold_blocks_ring(&position_rings, POSITIONS),
    );

    let onehot = OneHotPoly::<F>::new(
        D,
        D,
        vec![
            Some(0),
            Some(31),
            None,
            Some(5),
            Some(17),
            None,
            Some(9),
            Some(23),
        ],
    )
    .expect("one-hot source");
    assert_output(
        onehot
            .evaluate_and_fold_subfield(subfield_multipliers, POSITIONS)
            .expect("one-hot compact fold"),
        onehot.fold_blocks_ring(&position_rings, POSITIONS),
    );

    let sparse = SparseRingPoly::<F>::from_signed_coeffs(
        8,
        D,
        8,
        vec![(0, 1, 1), (2, 30, -1), (4, 17, 1), (7, 31, -1)],
    )
    .expect("sparse-ring source");
    assert_output(
        sparse
            .evaluate_and_fold_subfield(subfield_multipliers, POSITIONS)
            .expect("sparse-ring compact fold"),
        sparse.fold_blocks_ring(&position_rings, POSITIONS),
    );

    let digits = (0..8 * D).map(|index| (index % 7) as i8 - 3).collect();
    let witness = RecursiveWitnessFlat::from_i8_digits(digits);
    let suffix = witness.view::<F, D>().expect("suffix source");
    assert_output(
        suffix
            .evaluate_and_fold_subfield(subfield_multipliers, POSITIONS)
            .expect("suffix compact fold"),
        suffix.fold_blocks_ring(&position_rings, POSITIONS),
    );
}

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

    sparse_mul_acc::<D>(&digit_plane, &challenge, &mut simd_acc);
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
    sparse_mul_acc_i16::<D>(&digit_plane, &challenge, &mut simd_acc);
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
    sparse_mul_acc(&i8_plane, &challenge, &mut expected_i8);
    let mut actual_i8 = [0i32; D];
    sparse_mul_acc_pm1(&i8_plane, &positive, &negative, &mut actual_i8);
    assert_eq!(actual_i8, expected_i8);

    let i16_plane = std::array::from_fn(|index| ((index * 509) % 1024) as i16 - 512);
    let mut expected_i16 = [0i32; D];
    sparse_mul_acc_i16(&i16_plane, &challenge, &mut expected_i16);
    let mut actual_i16 = [0i32; D];
    sparse_mul_acc_i16_pm1(&i16_plane, &positive, &negative, &mut actual_i16);
    assert_eq!(actual_i16, expected_i16);
}

#[test]
#[should_panic]
fn sparse_mul_acc_rejects_out_of_range_challenge_before_dispatch() {
    const D: usize = 64;
    let challenge = SparseChallenge {
        positions: vec![D as u32].into(),
        coeffs: vec![1].into(),
    };
    sparse_mul_acc_i16(&[0; D], &challenge, &mut [0; D]);
}

#[test]
fn large_basis_partitioned_fold_preserves_i16_digits() {
    type F = Prime128Offset275;
    const D: usize = 128;
    const POSITIONS: usize = 2;
    let log_basis = 10;
    let num_digits = compute_num_digits_field_width(128, log_basis);
    let rings = (0..4)
        .map(|ring| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                F::from_canonical_u128_reduced(((ring * D + coefficient) as u128 + 1) * 509)
            }))
        })
        .collect::<Vec<_>>();
    let challenges = vec![
        SparseChallenge {
            positions: vec![0, 7, 63].into(),
            coeffs: vec![1, -2, 2].into(),
        },
        SparseChallenge {
            positions: vec![3, 19, 91].into(),
            coeffs: vec![-1, 2, 1].into(),
        },
    ];
    let q = (-F::one()).to_canonical_u128() + 1;
    let threshold =
        akita_algebra::ring::cyclotomic::decompose_centering_threshold(num_digits, log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    let actual = balanced_ring_decompose_fold_partitioned(
        &rings,
        &challenges,
        POSITIONS,
        num_digits,
        &params,
    );
    let mut expected = vec![[0i32; D]; POSITIONS * num_digits];
    for (block, challenge) in challenges.iter().enumerate() {
        for position in 0..POSITIONS {
            let digits = rings[block * POSITIONS + position]
                .balanced_decompose_pow2_i16(num_digits, log_basis);
            for digit in 0..num_digits {
                sparse_mul_acc_i16_scalar(
                    &digits[digit],
                    challenge,
                    &mut expected[position * num_digits + digit],
                );
            }
        }
    }
    assert_eq!(actual, expected);
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
    sparse_mul_acc::<D>(&digit_plane, &challenge, &mut simd_acc);
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
    sparse_mul_acc::<D>(&digit_plane, &challenge, &mut simd_acc);
    sparse_mul_acc_scalar::<D>(&digit_plane, &challenge, &mut scalar_acc);

    assert_eq!(simd_acc, scalar_acc);
}

#[test]
fn fused_full_challenge_accumulate_matches_generic_sparse_path() {
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
    let q = (-F::one()).to_canonical_u128() + 1;
    let log_basis = 3u32;
    let threshold =
        akita_algebra::ring::cyclotomic::decompose_centering_threshold(num_digits, log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    let mut generic_digits = vec![[0i8; D]; num_digits];
    decompose_ring_interleaved::<F, D>(&ring, &mut generic_digits, num_digits, &params);
    let mut generic_acc = vec![[0i32; D]; num_digits];
    for digit in 0..num_digits {
        sparse_mul_acc::<D>(&generic_digits[digit], &challenge, &mut generic_acc[digit]);
    }

    let mut rotated = vec![[0i16; D]; D];
    fill_rotated_challenge::<D>(&mut rotated, &challenge);
    let mut fused_acc = vec![[0i32; D]; num_digits];
    decompose_ring_full_challenge_accumulate::<F, D>(&ring, &rotated, &mut fused_acc, &params);

    assert_eq!(fused_acc, generic_acc);
}

#[test]
fn partitioned_full_challenge_accumulate_matches_generic_sparse_path() {
    type F = Fp64<4294967197>;
    const D: usize = 32;
    let num_positions_per_block = 3;
    let num_digits = 4;
    let coeffs: Vec<_> = (0..6)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                let v = (((idx * 11 + k * 7) as i64) % 19) - 9;
                F::from_i64(v)
            }))
        })
        .collect();
    let challenges = vec![
        SparseChallenge {
            positions: (0..D as u32).collect(),
            coeffs: (0..D)
                .map(|k| match k % 4 {
                    0 => -2,
                    1 => -1,
                    2 => 1,
                    _ => 3,
                })
                .collect(),
        },
        SparseChallenge {
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
        },
    ];
    let q = (-F::one()).to_canonical_u128() + 1;
    let log_basis = 3u32;
    let threshold =
        akita_algebra::ring::cyclotomic::decompose_centering_threshold(num_digits, log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    let fused = balanced_ring_decompose_fold_partitioned::<F, D>(
        &coeffs,
        &challenges,
        num_positions_per_block,
        num_digits,
        &params,
    );

    let mut generic = vec![[0i32; D]; num_positions_per_block * num_digits];
    let mut digit_buf = vec![[0i8; D]; num_digits];
    for (block_idx, challenge) in challenges.iter().enumerate() {
        let block_start = block_idx * num_positions_per_block;
        for local_idx in 0..num_positions_per_block {
            let ring = &coeffs[block_start + local_idx];
            decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, &params);
            let base = local_idx * num_digits;
            for digit in 0..num_digits {
                sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut generic[base + digit]);
            }
        }
    }

    assert_eq!(fused, generic);
}

#[test]
fn partitioned_high_density_d32_challenge_uses_rotated_path() {
    type F = Fp64<4294967197>;
    const D: usize = 32;
    let num_positions_per_block = 3;
    let num_digits = 4;
    let coeffs: Vec<_> = (0..6)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                let v = (((idx * 13 + k * 5) as i64) % 23) - 11;
                F::from_i64(v)
            }))
        })
        .collect();
    let high_density = SparseChallenge {
        positions: vec![
            0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31,
        ]
        .into(),
        coeffs: vec![
            2, 2, -1, 4, 1, -1, 5, 4, -3, -4, -3, -6, 2, -8, -4, -3, -7, -3, 4, -1, 4, -4, 5, -2,
            -4, 6, 6, -3, 4, 4,
        ]
        .into(),
    };
    let sparse = SparseChallenge {
        positions: vec![1, 7, 19].into(),
        coeffs: vec![2, -1, 3].into(),
    };
    assert!(should_use_rotated_challenge::<D>(&high_density));
    assert!(!should_use_rotated_challenge::<D>(&sparse));
    let challenges = vec![high_density, sparse];
    let q = (-F::one()).to_canonical_u128() + 1;
    let log_basis = 3u32;
    let threshold =
        akita_algebra::ring::cyclotomic::decompose_centering_threshold(num_digits, log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    let mixed = balanced_ring_decompose_fold_partitioned::<F, D>(
        &coeffs,
        &challenges,
        num_positions_per_block,
        num_digits,
        &params,
    );

    let mut generic = vec![[0i32; D]; num_positions_per_block * num_digits];
    let mut digit_buf = vec![[0i8; D]; num_digits];
    for (block_idx, challenge) in challenges.iter().enumerate() {
        let block_start = block_idx * num_positions_per_block;
        for local_idx in 0..num_positions_per_block {
            let ring = &coeffs[block_start + local_idx];
            decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, &params);
            let base = local_idx * num_digits;
            for digit in 0..num_digits {
                sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut generic[base + digit]);
            }
        }
    }

    assert_eq!(mixed, generic);
}

#[test]
fn partitioned_high_density_d64_challenge_uses_rotated_path() {
    type F = Fp64<4294967197>;
    const D: usize = 64;
    let num_positions_per_block = 2;
    let num_digits = 3;
    let coeffs: Vec<_> = (0..4)
        .map(|idx| {
            CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
                let v = (((idx * 17 + k * 7) as i64) % 31) - 15;
                F::from_i64(v)
            }))
        })
        .collect();
    let high_density = SparseChallenge {
        positions: (0..42u32).collect(),
        coeffs: (0..42)
            .map(|k| match k % 4 {
                0 => -2,
                1 => -1,
                2 => 1,
                _ => 2,
            })
            .collect(),
    };
    let sparse = SparseChallenge {
        positions: vec![1, 17, 33, 49].into(),
        coeffs: vec![2, -1, 1, -2].into(),
    };
    assert!(should_use_rotated_challenge::<D>(&high_density));
    assert!(!should_use_rotated_challenge::<D>(&sparse));
    let challenges = vec![high_density, sparse];
    let q = (-F::one()).to_canonical_u128() + 1;
    let log_basis = 4u32;
    let threshold =
        akita_algebra::ring::cyclotomic::decompose_centering_threshold(num_digits, log_basis, q);
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    let mixed = balanced_ring_decompose_fold_partitioned::<F, D>(
        &coeffs,
        &challenges,
        num_positions_per_block,
        num_digits,
        &params,
    );

    let mut generic = vec![[0i32; D]; num_positions_per_block * num_digits];
    let mut digit_buf = vec![[0i8; D]; num_digits];
    for (block_idx, challenge) in challenges.iter().enumerate() {
        let block_start = block_idx * num_positions_per_block;
        for local_idx in 0..num_positions_per_block {
            let ring = &coeffs[block_start + local_idx];
            decompose_ring_interleaved::<F, D>(ring, &mut digit_buf, num_digits, &params);
            let base = local_idx * num_digits;
            for digit in 0..num_digits {
                sparse_mul_acc::<D>(&digit_buf[digit], challenge, &mut generic[base + digit]);
            }
        }
    }

    assert_eq!(mixed, generic);
}

#[test]
fn fp128_overflow_paths_match_direct_and_fused_sparse_path() {
    type F = Prime128Offset275;
    const D: usize = 32;

    let log_basis = 4u32;
    let num_digits = compute_num_digits_field_width(128, log_basis);
    let q = (-F::one()).to_canonical_u128() + 1;
    let threshold =
        akita_algebra::ring::cyclotomic::decompose_centering_threshold(num_digits, log_basis, q);
    let i128_max = i128::MAX as u128;
    let boundary_values = [
        0,
        threshold,
        threshold + 1,
        q - i128_max - 1,
        q - i128_max,
        q - 1,
    ];
    let ring = CyclotomicRing::from_coefficients(std::array::from_fn(|k| {
        F::from_canonical_u128_reduced(boundary_values[k % boundary_values.len()])
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
    let params = DecomposeParams {
        threshold,
        q,
        mask: (1i128 << log_basis) - 1,
        half_b: 1i128 << (log_basis - 1),
        b_val: 1i128 << log_basis,
        log_basis,
        overflow_possible: q.saturating_sub(threshold) > i128::MAX as u128,
    };

    assert!(
        params.overflow_possible,
        "test must exercise the overflow path"
    );

    let mut actual_digits = vec![[0i8; D]; num_digits];
    decompose_ring_interleaved::<F, D>(&ring, &mut actual_digits, num_digits, &params);
    let mut expected_digits = vec![[0i8; D]; num_digits];
    ring.balanced_decompose_pow2_i8_into(&mut expected_digits, log_basis);
    assert_eq!(actual_digits, expected_digits);

    let mut generic_acc = vec![[0i32; D]; num_digits];
    for digit in 0..num_digits {
        sparse_mul_acc::<D>(&actual_digits[digit], &challenge, &mut generic_acc[digit]);
    }

    let mut rotated = vec![[0i16; D]; D];
    fill_rotated_challenge::<D>(&mut rotated, &challenge);
    let mut fused_acc = vec![[0i32; D]; num_digits];
    decompose_ring_full_challenge_accumulate::<F, D>(&ring, &rotated, &mut fused_acc, &params);
    assert_eq!(fused_acc, generic_acc);
}
