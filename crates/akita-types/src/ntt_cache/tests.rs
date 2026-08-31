use super::*;
use akita_algebra::CyclotomicRing;
use core::mem::size_of;
use jolt_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59, Ring};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn flat_zeros<F: Field, const D: usize>(len: usize) -> crate::FlatMatrix<F> {
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
fn verifier_cache_key_binds_exact_plan_requirement() {
    let strong = VerifierNttCacheKey {
        ring_d: 64,
        width: 8,
        rhs_abs_bound: 1 << 15,
    };
    let weak = VerifierNttCacheKey {
        ring_d: 64,
        width: 16,
        rhs_abs_bound: 3,
    };
    assert_ne!(strong, weak);
    assert_eq!(strong, strong);
}

#[test]
fn verifier_cache_preserves_strong_entry_across_weaker_growth() {
    const D: usize = 64;
    let ring_count = 32;
    let matrix = (0..ring_count)
        .map(|ring| {
            CyclotomicRing::<Prime128Offset275, D>::from_coefficients(std::array::from_fn(
                |coefficient| Prime128Offset275::from_u64((ring * D + coefficient + 1) as u64),
            ))
        })
        .collect::<Vec<_>>();
    let expanded = crate::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
        crate::AkitaSetupDescriptor {
            max_num_vars: 0,
            max_num_batched_polys: 0,
            num_field_elements: ring_count * D,
            setup_seed: [0u8; 32].into(),
        },
        crate::FlatMatrix::from_ring_slice(&matrix),
    );
    let cache = VerifierNttCache::default();
    let strong_mode = NttCacheMode::ExactNegacyclic {
        width: 8,
        rhs_abs_bound: 1 << 15,
    };
    let strong_tail = usize::from(
        ntt_cache_requires_exactness_tail::<Prime128Offset275, D>(8, 1 << 15)
            .expect("strong exactness requirement"),
    ) * 8;
    let strong = cache
        .prepare::<Prime128Offset275, D>(
            &expanded,
            NttCacheKey {
                ring_d: D,
                num_ring_elements: 8,
                domain: NttTransformDomain::Negacyclic,
            },
            strong_tail,
            strong_mode,
        )
        .expect("strong small prefix");

    let weak_tail = usize::from(
        ntt_cache_requires_exactness_tail::<Prime128Offset275, D>(16, 1)
            .expect("weak exactness requirement"),
    ) * 16;
    cache
        .prepare::<Prime128Offset275, D>(
            &expanded,
            NttCacheKey {
                ring_d: D,
                num_ring_elements: ring_count,
                domain: NttTransformDomain::Negacyclic,
            },
            weak_tail,
            NttCacheMode::ExactNegacyclic {
                width: 16,
                rhs_abs_bound: 1,
            },
        )
        .expect("weak large prefix");

    let strong_again = cache
        .prepare::<Prime128Offset275, D>(
            &expanded,
            NttCacheKey {
                ring_d: D,
                num_ring_elements: 8,
                domain: NttTransformDomain::Negacyclic,
            },
            strong_tail,
            strong_mode,
        )
        .expect("strong entry remains available");
    assert!(Arc::ptr_eq(&strong, &strong_again));
    assert_eq!(cache.slots.lock().expect("cache slots").len(), 2);

    let rhs = vec![[1i16; D]; 8];
    let cached_output = strong_again
        .mat_vec_i16::<Prime128Offset275>(3, 1, &rhs)
        .expect("cached strong plan computes");
    let uncached = prepare_ntt_cache(
        expanded
            .shared_matrix()
            .ring_view::<D>(1, 8)
            .expect("uncached matrix view"),
        strong_mode,
    )
    .expect("uncached strong plan");
    let uncached_output = uncached
        .mat_vec_i16::<Prime128Offset275>(3, 1, &rhs)
        .expect("uncached strong plan computes");
    assert_eq!(cached_output, uncached_output);
}

#[test]
fn prepare_materializes_exactly_the_requested_layout() {
    const D: usize = 64;
    let flat = flat_zeros::<Prime32Offset99, D>(10);
    let view = flat.ring_view::<D>(1, 10).expect("matrix view");
    let both = prepare_ntt_cache(view, NttCacheMode::BothTransforms).expect("both transforms");
    assert!(both.has_negacyclic());
    assert!(both.has_cyclic());
    assert!(!both.has_exactness_tail());

    let view = flat.ring_view::<D>(1, 10).expect("matrix view");
    let tail_pair =
        prepare_ntt_cache(view, NttCacheMode::I16TailBothTransforms).expect("tail transform pair");
    assert!(tail_pair.i16_tail_pair().is_some());
    assert_eq!(tail_pair.cache_bytes(), 10 * D * 2 * size_of::<i16>());

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
    assert_eq!(exact.has_exactness_tail(), ifma52_cache_enabled::<D>());

    let flat = flat_zeros::<Prime128Offset275, D>(10);
    let view = flat.ring_view::<D>(1, 10).expect("matrix view");
    let q128_exact = prepare_ntt_cache(
        view,
        NttCacheMode::ExactNegacyclic {
            width: 5,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("tail negacyclic");
    assert!(!q128_exact.has_cyclic());
    assert_eq!(q128_exact.has_exactness_tail(), ifma52_cache_enabled::<D>());
    let bytes_per_ring = if ifma52_cache_enabled::<D>() {
        IFMA52_PRIMES.len() * size_of::<u64>()
            + usize::from(q128_exact.has_exactness_tail()) * size_of::<i32>()
    } else {
        Q128_NUM_PRIMES * size_of::<i32>()
    };
    assert_eq!(q128_exact.cache_bytes(), 10 * D * bytes_per_ring);
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
    let capacity = if ifma52_cache_enabled::<D>() {
        CrtCapacity::from_prime_moduli(IFMA52_PRIMES.map(u128::from))
    } else {
        params.crt_capacity()
    };
    let safe = capacity
        .max_safe_width::<Prime128Offset275, D>(1 << 15)
        .expect("one term fits");
    assert!(!ntt_cache_requires_exactness_tail::<Prime128Offset275, D>(safe, 1 << 15).unwrap());
    assert!(ntt_cache_requires_exactness_tail::<Prime128Offset275, D>(safe + 1, 1 << 15).unwrap());
}

fn assert_quotient_tail_selectors_agree<F: Field + CanonicalEncoding, const D: usize>(
    profile: SisModulusProfileId,
) {
    for rhs_abs_bound in [1, 1 << 15, 1_000_000, u64::from(u32::MAX)] {
        let profile_result = centered_quotient_requires_i16_tail(profile, D, rhs_abs_bound);
        let field_result = centered_quotient_requires_i16_tail_for_field::<F, D>(rhs_abs_bound);
        match (profile_result, field_result) {
            (Ok(profile_tail), Ok(field_tail)) => assert_eq!(profile_tail, field_tail),
            (Err(_), Err(_)) => {}
            (profile_result, field_result) => panic!(
                "profile and field quotient-tail selectors disagree for D={D}, bound={rhs_abs_bound}: profile={profile_result:?}, field={field_result:?}"
            ),
        }
    }
}

#[test]
fn quotient_tail_planning_and_runtime_selectors_agree_for_all_fields() {
    assert_quotient_tail_selectors_agree::<Prime32Offset99, 128>(SisModulusProfileId::Q32Offset99);
    assert_quotient_tail_selectors_agree::<Prime32Offset99, 256>(SisModulusProfileId::Q32Offset99);
    assert_quotient_tail_selectors_agree::<Prime64Offset59, 64>(SisModulusProfileId::Q64Offset59);
    assert_quotient_tail_selectors_agree::<Prime64Offset59, 128>(SisModulusProfileId::Q64Offset59);
    assert_quotient_tail_selectors_agree::<Prime64Offset59, 256>(SisModulusProfileId::Q64Offset59);
    assert_quotient_tail_selectors_agree::<Prime128OffsetA7F7, 64>(
        SisModulusProfileId::Q128OffsetA7F7,
    );
    assert_quotient_tail_selectors_agree::<Prime128OffsetA7F7, 128>(
        SisModulusProfileId::Q128OffsetA7F7,
    );
}

#[test]
fn dense_i8_exact_ifma52_requires_q128_tail_capacity() {
    let q128 = SisModulusProfileId::Q128OffsetA7F7.modulus();
    let base_max = CrtCapacity::from_prime_moduli(IFMA52_PRIMES.map(u128::from))
        .max_safe_width_for_modulus(512, q128, 64)
        .expect("three-prime IFMA base capacity");
    let hybrid_max = CrtCapacity::from_prime_moduli(IFMA52_PRIMES.map(u128::from))
        .with_prime_modulus(q128_primes()[0].p as u128)
        .max_safe_width_for_modulus(512, q128, 64)
        .expect("hybrid IFMA capacity");
    assert!(dense_i8_exact_ifma52_is_profitable(
        q128,
        512,
        base_max + 1,
        64,
        true,
    ));
    assert!(!dense_i8_exact_ifma52_is_profitable(
        q128,
        512,
        hybrid_max + 1,
        64,
        true,
    ));
    assert!(!dense_i8_exact_ifma52_is_profitable(
        q128,
        512,
        base_max + 1,
        64,
        false,
    ));
    assert!(!dense_i8_exact_ifma52_is_profitable(
        SisModulusProfileId::Q64Offset59.modulus(),
        512,
        19_456,
        64,
        true,
    ));
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
    assert_eq!(
        planned_exact_ntt_cache_bytes::<Prime64Offset59, D>(2, 2, 1 << 15)
            .expect("planned exact cache bytes"),
        cache.cache_bytes()
    );
    if ifma52_cache_enabled::<D>() {
        assert!(ntt_cache_requires_exactness_tail::<Prime32Offset99, D>(2, 1 << 15).unwrap());
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
#[ignore = "requires AVX-512F/DQ/IFMA hardware or emulation"]
fn exact_cache_selects_ifma52_on_supported_hardware() {
    const D: usize = 64;
    assert!(
        ifma52_cache_enabled::<D>(),
        "IFMA52 dispatch is unavailable"
    );
    let flat = flat_zeros::<Prime64Offset59, D>(2);
    let cache = prepare_ntt_cache(
        flat.ring_view::<D>(1, 2).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: 2,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("exact cache");
    assert!(cache.uses_ifma52());
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
    assert_eq!(
        planned_exact_ntt_cache_bytes::<Prime32Offset99, D>(2, 2, 1 << 15)
            .expect("planned exact cache bytes"),
        cache.cache_bytes()
    );
    if ifma52_cache_enabled::<D>() {
        assert!(cache.uses_ifma52());
        assert!(cache.has_exactness_tail());
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

fn run_with_large_test_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("exact-ntt-cache-test".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(f)
        .expect("spawn exact NTT cache test")
        .join()
        .expect("exact NTT cache test thread");
}

#[test]
fn q32_exact_cache_matches_ring_arithmetic_at_all_ifma_dimensions() {
    run_with_large_test_stack(|| {
        assert_q32_exact_cache_matches_ring_arithmetic::<64>();
        assert_q32_exact_cache_matches_ring_arithmetic::<128>();
        assert_q32_exact_cache_matches_ring_arithmetic::<256>();
        assert_q32_exact_cache_matches_ring_arithmetic::<512>();
        assert_q32_exact_cache_matches_ring_arithmetic::<1024>();
        assert_q32_exact_cache_matches_ring_arithmetic::<2048>();
    });
}

fn assert_q128_exact_cache_matches_ring_arithmetic<const D: usize>() {
    const ROWS: usize = 2;
    const COLS: usize = 3;
    type F = Prime128OffsetA7F7;
    let modulus = u128::MAX - (<F as PseudoMersenne>::OFFSET - 1);
    let matrix = (0..ROWS * COLS)
        .map(|entry| {
            CyclotomicRing::<F, D>::from_coefficients(std::array::from_fn(|coefficient| {
                let magnitude = modulus / 2 - (entry * 257 + coefficient * 17) as u128;
                let value = F::from_u128_reduced(magnitude);
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
    assert_eq!(
        planned_exact_ntt_cache_bytes::<F, D>(ROWS * COLS, COLS, 1 << 15)
            .expect("planned exact cache bytes"),
        cache.cache_bytes()
    );
    if ifma52_cache_enabled::<D>() {
        assert!(cache.uses_ifma52());
        assert!(cache.has_exactness_tail());
        assert_eq!(
            cache.cache_bytes(),
            ROWS * COLS * D * (IFMA52_PRIMES.len() * size_of::<u64>() + size_of::<i32>())
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
    run_with_large_test_stack(|| {
        assert_q128_exact_cache_matches_ring_arithmetic::<64>();
        assert_q128_exact_cache_matches_ring_arithmetic::<128>();
        assert_q128_exact_cache_matches_ring_arithmetic::<256>();
        assert_q128_exact_cache_matches_ring_arithmetic::<512>();
        assert_q128_exact_cache_matches_ring_arithmetic::<1024>();
    });
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
    let reduced =
        prepare_reduced_compression_ntt_cache(flat.ring_view::<8>(1, 1).expect("matrix view"))
            .expect("reduced compression-only D8 cache");
    assert!(reduced.has_negacyclic());
    assert!(!reduced.has_cyclic());
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

    let full_i16 = prepare_ntt_cache(
        flat.ring_view::<D>(1, 2).expect("matrix view"),
        NttCacheMode::ExactNegacyclic {
            width: 2,
            rhs_abs_bound: 1 << 15,
        },
    )
    .expect("full-i16 cache");
    assert!(full_i16
        .mat_vec_i16::<Prime32Offset99>(16, 1, &[[i16::MIN; D], [i16::MAX; D]])
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
