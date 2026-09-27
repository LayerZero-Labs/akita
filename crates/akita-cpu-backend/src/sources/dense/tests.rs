use super::poly::DensePoly;
use crate::commitment::CommitmentSource;
use crate::opaque::RootPolyMeta;
use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use jolt_field::Prime128OffsetA7F7 as F;
use jolt_field::{CanonicalEncoding, Ring, Zero};

#[test]
fn chunked_fold_matches_windowed_reference_and_global() {
    use akita_challenges::SparseChallenge;

    const D: usize = 64;
    const POSITIONS: usize = 2;
    let poly =
        DensePoly::<F>::from_ring_coeffs((0..8).map(|index| ring::<D>(index * 10)).collect())
            .unwrap();
    let challenges = (0..4)
        .map(|block| SparseChallenge {
            positions: vec![(block * 2) as u32, (block * 2 + 1) as u32].into(),
            coeffs: vec![1, -1].into(),
        })
        .collect::<Vec<_>>();
    let global = poly.decompose_fold::<D>(&challenges, POSITIONS, 2, 4);

    for chunk_count in [2, 4, 8] {
        let ranges = akita_types::dyadic_block_ranges(challenges.len(), chunk_count).unwrap();
        let chunks = poly.decompose_fold_chunked::<D>(&challenges, &ranges, POSITIONS, 2, 4);
        assert_eq!(chunks.len(), chunk_count);
        for (range, chunk) in ranges.iter().zip(&chunks) {
            let window = challenges
                .iter()
                .enumerate()
                .map(|(block, challenge)| {
                    if range.contains(&block) {
                        challenge.clone()
                    } else {
                        SparseChallenge {
                            positions: Vec::new().into(),
                            coeffs: Vec::new().into(),
                        }
                    }
                })
                .collect::<Vec<_>>();
            let expected = poly.decompose_fold::<D>(&window, POSITIONS, 2, 4);
            assert_eq!(
                chunk.centered_coeffs_flat(),
                expected.centered_coeffs_flat()
            );
        }
        let combined = crate::opaque::aggregate_decompose_fold_witnesses::<D>(
            chunks
                .iter()
                .map(|chunk| Ok::<_, AkitaError>(chunk.clone())),
        )
        .unwrap();
        assert_eq!(
            combined.centered_coeffs_flat(),
            global.centered_coeffs_flat()
        );
    }
}

fn ring<const D: usize>(offset: u64) -> CyclotomicRing<F, D> {
    CyclotomicRing::from_coefficients(std::array::from_fn(|idx| {
        F::from_u64(offset + idx as u64 + 1)
    }))
}

#[test]
fn ring_fold_matches_dense_multiplication_reference() {
    const D: usize = 8;
    let coeffs = (0..2).map(|idx| ring::<D>(10 * idx)).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_ring_coeffs(coeffs.clone()).unwrap();
    let scalars = vec![
        ring::<D>(100),
        ring::<D>(200),
        ring::<D>(300),
        ring::<D>(400),
    ];
    let got = poly.fold_blocks_ring(&scalars, 4);
    let expected = coeffs
        .chunks(4)
        .map(|block| {
            block
                .iter()
                .zip(scalars.iter())
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (coeff, scalar)| {
                    acc + (*coeff * *scalar)
                })
        })
        .collect::<Vec<_>>();

    assert_eq!(got, expected);
}

#[test]
fn dense_constructor_reuses_owned_evaluation_buffer() {
    let evals = (0..2048).map(F::from_u64).collect::<Vec<_>>();
    let allocation = evals.as_ptr();
    let poly = DensePoly::<F>::from_field_evals(11, evals).unwrap();
    assert_eq!(poly.field_coeffs().as_ptr(), allocation);
}

#[test]
fn signed_constructor_preserves_every_byte_and_padding() {
    fn check<Fld: jolt_field::Field + CanonicalEncoding>() {
        let bytes = (u8::MIN..=u8::MAX).map(|v| v as i8).collect::<Vec<_>>();
        let poly = DensePoly::<Fld>::from_i8_evals(8, bytes.clone()).unwrap();
        for (&actual, &value) in poly.field_coeffs().iter().zip(&bytes) {
            assert_eq!(actual, Fld::from_i64(i64::from(value)));
        }
        assert!(poly.field_coeffs()[256..].iter().all(|v| v.is_zero()));
        for value in [i8::MIN, -1, 0, 1, i8::MAX] {
            let single = DensePoly::<Fld>::from_i8_evals(0, vec![value]).unwrap();
            assert_eq!(single.field_coeffs()[0], Fld::from_i64(i64::from(value)));
            assert!(single.field_coeffs()[1..].iter().all(|v| v.is_zero()));
        }
    }
    check::<F>();
    check::<jolt_field::Prime64Offset59>();
    check::<jolt_field::Prime32Offset99>();
    check::<jolt_field::Fp64<251>>();
    for num_vars in [usize::BITS as usize, usize::MAX] {
        assert!(DensePoly::<F>::from_i8_evals(num_vars, vec![0]).is_err());
    }
    for len in [0, 1, 7, 9] {
        assert!(matches!(
            DensePoly::<F>::from_i8_evals(3, vec![0; len]),
            Err(AkitaError::InvalidSize { expected: 8, actual }) if actual == len
        ));
    }
}

#[test]
fn signed_digit_borrow_obeys_each_balanced_interval_and_ring_view() {
    fn check<const D: usize>() {
        for log_basis in 1..=8 {
            let half = 1i16 << (log_basis - 1);
            for value in [-half - 1, -half, -1, 0, half - 1, half] {
                let Ok(value) = i8::try_from(value) else {
                    continue;
                };
                let poly = DensePoly::<F>::from_i8_evals(3, vec![value; 8]).unwrap();
                let admitted = i16::from(value) >= -half && i16::from(value) < half;
                assert_eq!(poly.cached_digit_parts(D, 1, log_basis).is_some(), admitted);
                assert!(poly.cached_digit_parts(D, 2, log_basis).is_none());
                let planes = poly.digit_planes_for::<D>(1, log_basis).unwrap();
                let expected = (i16::from(value) + half).rem_euclid(2 * half) - half;
                for (index, &digit) in planes.as_flattened().iter().enumerate() {
                    assert_eq!(i16::from(digit), if index < 8 { expected } else { 0 });
                }
                if admitted {
                    assert_eq!(
                        planes.as_ptr().cast::<i8>(),
                        poly.small_i8_coeffs.as_ref().unwrap().as_ptr()
                    );
                    let cloned = poly.clone();
                    assert_eq!(cloned.small_i8_bounds(), poly.small_i8_bounds());
                    assert!(cloned.cached_digit_parts(D, 1, log_basis).is_some());
                }
                // A different ring view must remain exact after the first request.
                let other = poly.digit_planes_for::<64>(1, log_basis);
                if admitted || D == 64 {
                    assert_eq!(i16::from(other.unwrap()[0][0]), expected);
                }
            }
        }
    }
    check::<64>();
    check::<128>();
    check::<1024>();
    let small = DensePoly::<jolt_field::Fp64<251>>::from_i8_evals(0, vec![-128]).unwrap();
    assert!(small.small_i8_bounds().is_none());
    assert!(small.cached_digit_parts(64, 1, 8).is_none());
}

#[test]
fn cached_reach_preserves_arbitrary_centering_thresholds() {
    let poly = DensePoly::<F>::from_i8_evals(3, vec![-128, -7, -1, 0, 1, 7, 126, 127]).unwrap();
    let q = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    for modulus in [q, q + 1] {
        for threshold in [0, 126, 127, q / 2, q - 129, q - 128, q - 1] {
            let mut expected = (0, 0);
            for value in poly.field_coeffs() {
                let value = value.to_u128_checked().unwrap();
                if value <= threshold {
                    expected.1 = expected.1.max(value);
                } else {
                    expected.0 = expected.0.max(modulus - value);
                }
            }
            assert_eq!(
                poly.committed_centered_reach(modulus, threshold).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn incompatible_digit_cache_falls_back_to_balanced_sparse_convolution() {
    use akita_challenges::SparseChallenge;
    const D: usize = 64;
    let bytes = (0..512).map(|i| (i * 137 + 41) as i8).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_i8_evals(9, bytes.clone()).unwrap();
    // Occupy the cache at a different dimension and digit count.
    poly.digit_planes_for::<128>(2, 4).unwrap();
    for positions in [vec![], vec![0], vec![1, 17, 63]] {
        let challenges = (0..4)
            .map(|block| SparseChallenge {
                positions: positions.clone().into(),
                coeffs: positions
                    .iter()
                    .enumerate()
                    .map(|(i, _)| if (block + i) % 2 == 0 { 1 } else { -1 })
                    .collect::<Vec<_>>()
                    .into(),
            })
            .collect::<Vec<_>>();
        let got = poly.decompose_fold::<D>(&challenges, 2, 1, 3);
        let mut expected = vec![0i32; 2 * D];
        for (block, challenge) in challenges.iter().enumerate() {
            for position in 0..2 {
                for coefficient in 0..D {
                    let value = i32::from(bytes[(block * 2 + position) * D + coefficient]);
                    let digit = (value + 4).rem_euclid(8) - 4;
                    for (&shift, &sign) in challenge.positions.iter().zip(challenge.coeffs.iter()) {
                        let shifted = coefficient + shift as usize;
                        let sign = i32::from(sign) * if shifted < D { 1 } else { -1 };
                        expected[position * D + shifted % D] += digit * sign;
                    }
                }
            }
        }
        assert_eq!(got.centered_coeffs_flat(), expected);
    }
}

#[cfg(feature = "parallel")]
#[test]
fn signed_bounds_are_consistent_across_concurrent_queries() {
    use rayon::prelude::*;
    let poly = DensePoly::<F>::from_i8_evals(17, vec![-7; 1 << 17]).unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            (0..16).into_par_iter().for_each(|_| {
                assert_eq!(poly.small_i8_bounds(), Some((-7, 0)));
            });
        });
}

#[test]
fn dense_source_has_exact_views_across_supported_ring_dimensions() {
    let evals = (1..=32).map(F::from_u64).collect::<Vec<_>>();
    let poly = DensePoly::<F>::from_field_evals(5, evals.clone()).unwrap();

    fn assert_view<const D: usize>(poly: &DensePoly<F>, evals: &[F]) {
        let rings = poly.ring_coeffs::<D>().expect("supported dense view");
        let flat = rings
            .iter()
            .flat_map(|ring| ring.coefficients().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(&flat[..evals.len()], evals);
        assert!(flat[evals.len()..].iter().all(|value| *value == F::zero()));
    }

    assert_view::<64>(&poly, &evals);
    assert_view::<128>(&poly, &evals);
    assert_view::<256>(&poly, &evals);
    assert_view::<512>(&poly, &evals);
    assert_view::<1024>(&poly, &evals);
}

#[test]
fn dense_ring_constructor_rejects_empty_and_irregular_sources() {
    const D: usize = 512;
    // 96 rings previously exposed only the first 32 of the supplied rings.
    for num_rings in [0, 3, 96] {
        let result =
            DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, D>::zero(); num_rings]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
    let zero_degree = DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, 0>::zero()]);
    assert!(matches!(zero_degree, Err(AkitaError::InvalidInput(_))));
    let odd_degree = DensePoly::<F>::from_ring_coeffs(vec![CyclotomicRing::<F, 3>::zero(); 2]);
    assert!(matches!(odd_degree, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn dense_ring_constructor_preserves_the_entire_commitment_source_and_bound_scan() {
    const D: usize = 512;
    let modulus = (-F::from_u64(1)).to_u128_checked().unwrap() + 1;
    // Exercise both small-i8 and full-field storage, including physical padding.
    for num_rings in [1, 32, 64] {
        for reach in [127u64, 128] {
            let mut evals = (0..num_rings * D)
                .map(|idx| F::from_u64((idx % 32) as u64))
                .collect::<Vec<_>>();
            evals[num_rings * D - 2] = -F::from_u64(reach);
            evals[num_rings * D - 1] = F::from_u64(reach);
            let rings = evals
                .chunks_exact(D)
                .map(|chunk| CyclotomicRing::<F, D>::from_coefficients(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let poly = DensePoly::<F>::from_ring_coeffs(rings.clone()).unwrap();

            assert_eq!(1usize << RootPolyMeta::num_vars(&poly), evals.len());
            assert_eq!(&poly.field_coeffs()[..evals.len()], evals);
            assert_eq!(poly.ring_coeffs::<D>().unwrap(), rings);
            let descriptor = <DensePoly<F> as CommitmentSource<F>>::descriptor(&poly).unwrap();
            assert_eq!(descriptor.num_vars(), RootPolyMeta::num_vars(&poly));
            assert_eq!(descriptor.live_coefficient_len(), evals.len());
            assert_eq!(
                <DensePoly<F> as CommitmentSource<F>>::committed_centered_reach(
                    &poly,
                    modulus,
                    modulus / 2,
                )
                .unwrap(),
                (u128::from(reach), u128::from(reach)),
            );
            assert_eq!(
                poly,
                DensePoly::from_field_evals(RootPolyMeta::num_vars(&poly), evals).unwrap(),
            );
        }
    }
}

#[test]
fn dense_field_constructor_rejects_unrepresentable_arities() {
    for num_vars in [usize::BITS as usize, usize::MAX] {
        let result = DensePoly::<F>::from_field_evals(num_vars, vec![F::zero()]);
        assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn dense_field_constructor_rejects_arity_that_truncates_to_a_valid_shift() {
    let result = DensePoly::<F>::from_field_evals((1usize << 32) + 14, vec![F::zero(); 1 << 14]);
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}

#[test]
fn batch_fold_rejects_mixed_extents_and_count_mismatch() {
    use crate::opaque::{
        CpuBackend, DecomposeFoldBatchPlan, OpeningBatchKernel, RootOpeningSource,
    };
    use akita_challenges::SparseChallenge;

    const D: usize = 64;
    let polys = [
        DensePoly::from_field_evals(6, vec![F::from_u64(1); 1 << 6]).unwrap(),
        DensePoly::from_field_evals(7, vec![F::from_u64(1); 1 << 7]).unwrap(),
    ];
    let challenges = vec![
        SparseChallenge {
            positions: vec![0].into(),
            coeffs: vec![1].into(),
        };
        2
    ];
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let run = |refs: &[&DensePoly<F>]| {
        OpeningBatchKernel::decompose_fold_batch(
            &backend,
            None,
            <DensePoly<F> as RootOpeningSource<F, D>>::opening_batch(refs).unwrap(),
            DecomposeFoldBatchPlan::Sparse {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 1,
            },
        )
    };

    assert!(matches!(
        run(&[&polys[0], &polys[1]]),
        Err(AkitaError::InvalidInput(_))
    ));
    assert!(matches!(
        run(&[&polys[0], &polys[0], &polys[0]]),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn scalar_fold_rejects_short_and_excess_challenges() {
    use crate::opaque::{CpuBackend, DecomposeFoldPlan, OpeningFoldKernel, RootOpeningSource};
    use akita_challenges::SparseChallenge;

    const D: usize = 8;
    // Two rings at one position per block give two live blocks.
    let poly = DensePoly::from_ring_coeffs(vec![ring::<D>(0), ring::<D>(10)]).unwrap();
    let challenge = SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    };
    let backend = CpuBackend::<F, F>::for_arithmetic_tests();
    let run = |count: usize| {
        let challenges = vec![challenge.clone(); count];
        OpeningFoldKernel::decompose_fold(
            &backend,
            None,
            <DensePoly<F> as RootOpeningSource<F, D>>::opening_view(&poly).unwrap(),
            DecomposeFoldPlan {
                challenges: &challenges,
                num_positions_per_block: 1,
                num_digits: 1,
                log_basis: 6,
            },
        )
    };

    assert!(run(2).is_ok());
    for count in [0, 1, 3] {
        assert!(
            matches!(
                run(count),
                Err(AkitaError::InvalidSize { expected: 2, actual }) if actual == count
            ),
            "{count} challenges for two live blocks must be rejected"
        );
    }
}

// A +1 monomial challenge must leave every single balanced digit unchanged.
// The expectation is the input integer itself, independent of decomposition.
#[test]
fn single_digit_fold_preserves_signed_i8_i16_boundaries() {
    use akita_challenges::SparseChallenge;
    use jolt_field::Prime64Offset59;
    const D: usize = 128;
    let challenge = SparseChallenge {
        positions: vec![0].into(),
        coeffs: vec![1].into(),
    };
    let mut mismatches = Vec::new();
    for log_basis in [8_u32, 9] {
        let half = 1_i64 << (log_basis - 1);
        for value in [127_i64, 128, 255, -128, -129, -256] {
            if !(-half..half).contains(&value) {
                continue; // Only admissible balanced digits belong in this oracle.
            }
            let mut coefficients = vec![Prime64Offset59::zero(); D];
            coefficients[0] = Prime64Offset59::from_i64(value);
            let poly = DensePoly::from_field_evals(7, coefficients).unwrap();
            let actual =
                poly.decompose_fold::<D>(std::slice::from_ref(&challenge), 1, 1, log_basis);
            let mut expected = vec![0_i32; D];
            expected[0] = value as i32;
            if actual.centered_coeffs_flat() != expected {
                mismatches.push((log_basis, value, actual.centered_coeffs_flat()[0]));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "(basis, expected, actual): {mismatches:?}"
    );
}
