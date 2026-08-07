use super::*;
use akita_algebra::CyclotomicRing;
use akita_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59};
use core::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn flat_zeros<F: FieldCore, const D: usize>(len: usize) -> crate::FlatMatrix<F> {
    crate::FlatMatrix::from_ring_slice(&vec![CyclotomicRing::<F, D>::zero(); len])
}

#[test]
fn prefix_requirements_join_by_maximum_in_one_dimension() {
    let short = NttPrefixRequirement::from_matrix_shape(64, 2, 3).expect("short prefix");
    let long = NttPrefixRequirement::from_matrix_shape(64, 4, 5).expect("long prefix");
    assert_eq!(short.join(long).expect("join"), long);
    assert_eq!(long.num_field_elements().expect("field count"), 20 * 64);

    let other_dimension =
        NttPrefixRequirement::from_matrix_shape(128, 1, 1).expect("other dimension");
    assert!(short.join(other_dimension).is_err());
}

#[test]
fn prepare_materializes_exactly_the_requested_layout() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime32Offset99, D>(10);
    let view = flat.ring_view::<D>(1, 10).expect("matrix view");
    let both = prepare_ntt_cache(view, NttCacheMode::BothTransforms).expect("both transforms");
    assert!(both.has_negacyclic());
    assert!(both.has_cyclic());
    assert!(!both.has_i16_tail());

    let view = flat.ring_view::<D>(1, 7).expect("matrix view");
    let cyclic = prepare_ntt_cache(view, NttCacheMode::Cyclic).expect("cyclic transform");
    assert!(!cyclic.has_negacyclic());
    assert!(cyclic.has_cyclic());
    assert_eq!(
        cyclic.cache_bytes(),
        7 * D * Q32_NUM_PRIMES * size_of::<i32>()
    );

    let view = flat.ring_view::<D>(1, 10).expect("matrix view");
    let exact = prepare_ntt_cache(
        view,
        NttCacheMode::ExactNegacyclic {
            width: 5,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("base negacyclic");
    assert!(exact.has_negacyclic());
    assert!(!exact.has_cyclic());
    assert_eq!(exact.has_i16_tail(), ifma52_cache_enabled::<D>());

    let flat = flat_zeros::<Prime128Offset275, D>(10);
    let view = flat.ring_view::<D>(1, 10).expect("matrix view");
    let tail = prepare_ntt_cache(
        view,
        NttCacheMode::ExactNegacyclic {
            width: 5,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("tail negacyclic");
    assert!(!tail.has_cyclic());
    assert!(tail.has_i16_tail());
    let bytes_per_ring = if ifma52_cache_enabled::<D>() {
        3 * size_of::<u64>() + size_of::<i16>()
    } else {
        Q128_NUM_PRIMES * size_of::<i32>() + size_of::<i16>()
    };
    assert_eq!(tail.cache_bytes(), 10 * D * bytes_per_ring);
}

#[test]
fn exact_mode_rejects_invalid_bounds() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime64Offset59, D>(1);
    for mode in [
        NttCacheMode::ExactNegacyclic {
            width: 0,
            rhs_abs_bound: 1,
        },
        NttCacheMode::ExactNegacyclic {
            width: 1,
            rhs_abs_bound: 0,
        },
    ] {
        let view = flat.ring_view::<D>(1, 1).expect("matrix view");
        assert!(matches!(
            prepare_ntt_cache(view, mode),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}

#[test]
fn exact_selector_changes_layout_at_the_strict_capacity_boundary() {
    const D: usize = 64;
    let ProtocolCrtNttParams::Q128(params) =
        select_crt_ntt_params::<Prime128Offset275, D>().expect("Q128 params")
    else {
        panic!("Q128 field must select Q128 params");
    };
    let safe = params
        .crt_capacity()
        .max_safe_width::<Prime128Offset275, D>(1 << 15)
        .expect("one term fits");
    assert!(!ntt_cache_requires_i16_tail::<Prime128Offset275, D>(safe, 1 << 15).unwrap());
    assert!(ntt_cache_requires_i16_tail::<Prime128Offset275, D>(safe + 1, 1 << 15).unwrap());
}

#[test]
fn q128_a7f7_selector_accepts_d512() {
    assert!(matches!(
        select_crt_ntt_params::<Prime128OffsetA7F7, 512>(),
        Ok(ProtocolCrtNttParams::Q128(_))
    ));
}

#[test]
fn q64_exact_cache_uses_ifma52_when_enabled() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime64Offset59, D>(2);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(1, 2).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: 2,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("exact cache");
    if ifma52_cache_enabled::<D>() {
        assert!(ntt_cache_requires_i16_tail::<Prime32Offset99, D>(2, 1 << 15).unwrap());
        assert!(cache.uses_ifma52());
        assert_eq!(cache.cache_bytes(), 2 * 2 * D * size_of::<u64>());
    } else {
        assert!(!cache.uses_ifma52());
    }
    assert_eq!(
        cache
            .mat_vec_i16::<Prime64Offset59>(16, 1, &[[i16::MAX; D], [i16::MIN; D]])
            .expect("IFMA52 exact matvec"),
        vec![CyclotomicRing::zero()]
    );
}

#[test]
fn q32_exact_cache_uses_mixed_ifma52_when_enabled() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime32Offset99, D>(2);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(1, 2).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: 2,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("exact cache");
    if ifma52_cache_enabled::<D>() {
        assert!(cache.uses_ifma52());
        assert!(cache.has_i16_tail());
        assert_eq!(
            cache.cache_bytes(),
            2 * D * (size_of::<u64>() + size_of::<i16>())
        );
    } else {
        assert!(!cache.uses_ifma52());
    }
    assert_eq!(
        cache
            .mat_vec_i16::<Prime32Offset99>(16, 1, &[[i16::MAX; D], [i16::MIN; D]])
            .expect("mixed IFMA52 exact matvec"),
        vec![CyclotomicRing::zero()]
    );
}

#[test]
fn q32_i16_vnni_exact_cache_matches_ring_arithmetic() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let available = std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512dq")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vnni")
        && std::env::var("AKITA_IFMA52").ok().as_deref() == Some("0");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let available = false;
    if !available {
        assert_ne!(
            std::env::var("AKITA_REQUIRE_AVX512VNNI").ok().as_deref(),
            Some("1"),
            "required Q32 i16 VNNI cache backend is unavailable"
        );
        return;
    }

    const D: usize = 64;
    const ROWS: usize = 2;
    const COLS: usize = 6;
    type F = Prime32Offset99;
    let matrix = (0..ROWS * COLS)
        .map(|entry| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                let magnitude = (Q32_MODULUS / 2) as i64 - (entry * 257 + coefficient * 17) as i64;
                F::from_i64(if (entry + coefficient) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                })
            }))
        })
        .collect::<Vec<_>>();
    let rhs = (0..COLS)
        .map(|column| {
            std::array::from_fn(|coefficient| {
                if (column + coefficient) % 2 == 0 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            })
        })
        .collect::<Vec<_>>();
    let flat = crate::FlatMatrix::from_ring_slice(&matrix);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(ROWS, COLS).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: COLS,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("Q32 i16 cache");
    assert!(cache.q32_i16_base().is_some());
    assert_eq!(
        cache.cache_bytes(),
        ROWS * COLS * D * Q32_I16_NUM_PRIMES * size_of::<i16>()
    );

    let actual = cache
        .mat_vec_i16::<F>(16, ROWS, &rhs)
        .expect("Q32 i16 matvec");
    let expected = matrix
        .chunks_exact(COLS)
        .map(|row| {
            row.iter()
                .zip(&rhs)
                .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                    sum + *lhs
                        * CyclotomicRing::from_coefficients(
                            rhs.map(|value| F::from_i64(value.into())),
                        )
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

fn assert_q32_exact_cache_matches_ring_arithmetic<const D: usize>() {
    const ROWS: usize = 2;
    const COLS: usize = 3;
    type F = Prime32Offset99;
    let matrix = (0..ROWS * COLS)
        .map(|entry| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                let magnitude = (Q32_MODULUS / 2) as i64 - (entry * 257 + coefficient * 17) as i64;
                F::from_i64(if (entry + coefficient) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                })
            }))
        })
        .collect::<Vec<_>>();
    let flat = crate::FlatMatrix::from_ring_slice(&matrix);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(ROWS, COLS).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: COLS,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("exact cache");
    let rhs = (0..COLS)
        .map(|column| {
            std::array::from_fn(|coefficient| {
                if (column + coefficient) % 2 == 0 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            })
        })
        .collect::<Vec<_>>();
    let actual = cache
        .mat_vec_i16::<F>(16, ROWS, &rhs)
        .expect("exact matvec");
    let expected = matrix
        .chunks_exact(COLS)
        .map(|row| {
            row.iter()
                .zip(&rhs)
                .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                    sum + *lhs
                        * CyclotomicRing::from_coefficients(
                            rhs.map(|value| F::from_i64(value.into())),
                        )
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn q32_exact_cache_matches_ring_arithmetic_at_all_ifma_dimensions() {
    assert_q32_exact_cache_matches_ring_arithmetic::<64>();
    assert_q32_exact_cache_matches_ring_arithmetic::<128>();
    assert_q32_exact_cache_matches_ring_arithmetic::<256>();
    assert_q32_exact_cache_matches_ring_arithmetic::<512>();
}

fn assert_q128_exact_cache_matches_ring_arithmetic<const D: usize>() {
    const ROWS: usize = 2;
    const COLS: usize = 3;
    type F = Prime128OffsetA7F7;
    let modulus = u128::MAX - (<F as PseudoMersenneField>::MODULUS_OFFSET - 1);
    let matrix = (0..ROWS * COLS)
        .map(|entry| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                let magnitude = modulus / 2 - (entry * 257 + coefficient * 17) as u128;
                let value = F::from_canonical_u128_reduced(magnitude);
                if (entry + coefficient) % 2 == 0 {
                    value
                } else {
                    -value
                }
            }))
        })
        .collect::<Vec<_>>();
    let flat = crate::FlatMatrix::from_ring_slice(&matrix);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(ROWS, COLS).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: COLS,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("exact cache");
    if ifma52_cache_enabled::<D>() {
        assert!(cache.uses_ifma52());
        assert!(cache.has_i16_tail());
        assert_eq!(
            cache.cache_bytes(),
            ROWS * COLS * D * (3 * size_of::<u64>() + size_of::<i16>())
        );
    }
    let rhs = (0..COLS)
        .map(|column| {
            std::array::from_fn(|coefficient| {
                if (column + coefficient) % 2 == 0 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            })
        })
        .collect::<Vec<_>>();
    let actual = cache
        .mat_vec_i16::<F>(16, ROWS, &rhs)
        .expect("exact matvec");
    let expected = matrix
        .chunks_exact(COLS)
        .map(|row| {
            row.iter()
                .zip(&rhs)
                .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                    sum + *lhs
                        * CyclotomicRing::from_coefficients(
                            rhs.map(|value| F::from_i64(value.into())),
                        )
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn q128_exact_cache_matches_ring_arithmetic_at_all_ifma_dimensions() {
    assert_q128_exact_cache_matches_ring_arithmetic::<64>();
    assert_q128_exact_cache_matches_ring_arithmetic::<128>();
    assert_q128_exact_cache_matches_ring_arithmetic::<256>();
    assert_q128_exact_cache_matches_ring_arithmetic::<512>();
}

#[test]
fn protocol_selector_rejects_compression_only_q128_d8_while_compression_prep_succeeds() {
    assert!(matches!(
        select_crt_ntt_params::<Prime128OffsetA7F7, 8>(),
        Err(AkitaError::InvalidSetup(_))
    ));
    assert!(matches!(
        select_compression_crt_ntt_params::<Prime128OffsetA7F7, 8>(),
        Ok(ProtocolCrtNttParams::Q128(_))
    ));
    let flat = flat_zeros::<Prime128OffsetA7F7, 8>(1);
    let cache = prepare_compression_ntt_cache(flat.ring_view::<8>(1, 1).expect("matrix view"))
        .expect("compression-only D8 cache");
    assert!(cache.has_cyclic());
    assert!(matches!(
        prepare_ntt_cache(
            flat.ring_view::<8>(1, 1).expect("matrix view"),
            NttCacheMode::BothTransforms,
        ),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn signed_i16_cache_checks_shape_and_digit_class() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime32Offset99, D>(2);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(1, 2).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: 2,
            rhs_abs_bound: 1 << 9,
        },
    )
    .expect("cache");
    assert!(cache
        .mat_vec_i16::<Prime32Offset99>(10, 1, &[[511; D], [-512; D]])
        .is_ok());
    assert!(matches!(
        cache.mat_vec_i16::<Prime32Offset99>(10, 1, &[[512; D], [0; D]]),
        Err(AkitaError::InvalidProof)
    ));
    assert!(cache
        .mat_vec_i16::<Prime32Offset99>(10, 1, &[[0; D]])
        .is_ok());

    let short = prepare_ntt_cache(
        flat.ring_view::<D>(1, 1).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: 1,
            rhs_abs_bound: 1 << 9,
        },
    )
    .expect("short cache");
    assert!(matches!(
        short.mat_vec_i16::<Prime32Offset99>(10, 1, &[[0; D], [0; D]]),
        Err(AkitaError::InvalidSetup(_))
    ));
}

#[test]
fn erased_cache_mismatches_return_errors_without_panicking() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime32Offset99, D>(1);
    let cache = Arc::new(
        prepare_ntt_cache(
            flat.ring_view::<D>(1, 1).expect("matrix view"),
            NttCacheMode::ExactNegacyclic {
                width: 1,
                rhs_abs_bound: 1 << 7,
            },
        )
        .expect("cache"),
    );
    let bytes = cache.cache_bytes();
    let wrong_degree = Arc::new(ErasedVerifierNttCache {
        ring_d: D,
        base_prefix_len: 1,
        tail_prefix_len: 0,
        cache_bytes: bytes,
        cache: Arc::clone(&cache) as Arc<dyn Any + Send + Sync>,
    });
    let result = catch_unwind(AssertUnwindSafe(|| {
        downcast_verifier_cache::<32>(wrong_degree)
    }));
    assert!(matches!(result, Ok(Err(AkitaError::InvalidSetup(_)))));

    let wrong_type = Arc::new(ErasedVerifierNttCache {
        ring_d: D,
        base_prefix_len: 1,
        tail_prefix_len: 0,
        cache_bytes: 0,
        cache: Arc::new(17usize),
    });
    let result = catch_unwind(AssertUnwindSafe(|| {
        downcast_verifier_cache::<D>(wrong_type)
    }));
    assert!(matches!(result, Ok(Err(AkitaError::InvalidSetup(_)))));
}
