//! Exact negacyclic NTT kernels for terminal verifier matrix relations.

use akita_algebra::CyclotomicRing;
use akita_error::AkitaError;
use akita_types::{
    dispatch_for_field, ntt_cache_requires_exactness_tail, AkitaVerifierSetup, FoldSchedule,
};
use jolt_field::{CanonicalEncoding, Field};

use crate::prepared_cache::{
    terminal_ntt_cache_requirement, TERMINAL_I16_ABS_BOUND, TERMINAL_I16_LOG_BASIS,
};

/// Warm every exact terminal i16 representation selected by a validated schedule.
pub(super) fn warm_for_schedule<
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
>(
    setup: &AkitaVerifierSetup<F>,
    schedule: &FoldSchedule,
) -> Result<(), AkitaError> {
    let requirement = terminal_ntt_cache_requirement(schedule)?;
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        requirement.ring_dimension,
        |D| {
            let tail_prefix_len = if ntt_cache_requires_exactness_tail::<F, D>(
                requirement.width,
                TERMINAL_I16_ABS_BOUND,
            )? {
                requirement.prefix_len
            } else {
                0
            };
            setup.prepared_verifier_ntt_prefix::<D>(
                requirement.prefix_len,
                tail_prefix_len,
                requirement.width,
                TERMINAL_I16_ABS_BOUND,
            )?;
            Ok::<(), AkitaError>(())
        }
    )
}

/// Compute the terminal prepared negacyclic matrix product for signed-i16 rings.
pub(super) fn centered_rows<F, const D: usize>(
    setup: &AkitaVerifierSetup<F>,
    num_rows: usize,
    rhs: &[[i16; D]],
    prepared_prefix_len: usize,
) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
{
    let _span = tracing::info_span!(
        "terminal_ntt_a_product",
        ring_d = D,
        num_rows,
        num_cols = rhs.len(),
        prepared_prefix_len
    )
    .entered();
    let required = num_rows
        .checked_mul(rhs.len())
        .ok_or(AkitaError::InvalidProof)?;
    if prepared_prefix_len < required {
        return Err(AkitaError::InvalidSetup(
            "verifier A cache prefix is undersized".into(),
        ));
    }
    if num_rows == 0 || rhs.is_empty() {
        return Ok(vec![CyclotomicRing::zero(); num_rows]);
    }

    let slot = {
        let _span = tracing::info_span!("terminal_ntt_a_i16_cache_lookup").entered();
        let tail_prefix_len =
            if ntt_cache_requires_exactness_tail::<F, D>(rhs.len(), TERMINAL_I16_ABS_BOUND)? {
                prepared_prefix_len
            } else {
                0
            };
        setup.prepared_verifier_ntt_prefix::<D>(
            prepared_prefix_len,
            tail_prefix_len,
            rhs.len(),
            TERMINAL_I16_ABS_BOUND,
        )?
    };
    let _span = tracing::info_span!("terminal_ntt_a_i16_accumulate").entered();
    slot.mat_vec_i16(TERMINAL_I16_LOG_BASIS, num_rows, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ntt::ifma52::ifma52_enabled;
    use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES};
    use akita_config::proof_optimized::fp128::OneHot;
    use akita_types::{
        prepare_ntt_cache, AkitaExpandedSetup, AkitaScheduleLookupKey, AkitaSetupDescriptor,
        FlatMatrix, NttCacheMode, PolynomialGroupLayout, SetupPrefixVerifierRegistry,
    };
    use jolt_field::Ring;
    use jolt_field::{Prime128Offset275 as F, Prime32Offset99 as F32, Prime64Offset59 as F64};
    use std::sync::Arc;

    const D: usize = 64;

    fn centered_inner_product<F, const D: usize>(
        lhs: &[CyclotomicRing<F, D>],
        rhs: &[[i16; D]],
    ) -> Result<CyclotomicRing<F, D>, AkitaError>
    where
        F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize,
    {
        if lhs.len() != rhs.len() {
            return Err(AkitaError::InvalidProof);
        }
        if lhs.is_empty() {
            return Ok(CyclotomicRing::zero());
        }
        let rhs_abs_bound = 1u64
            .checked_shl(TERMINAL_I16_LOG_BASIS - 1)
            .ok_or(AkitaError::InvalidProof)?;
        let flat = FlatMatrix::from_ring_slice(lhs);
        let matrix = flat.ring_view::<D>(1, lhs.len())?;
        let prepared = prepare_ntt_cache(
            matrix,
            NttCacheMode::ExactNegacyclic {
                width: lhs.len(),
                rhs_abs_bound,
            },
        )?;
        prepared
            .mat_vec_i16::<F>(TERMINAL_I16_LOG_BASIS, 1, rhs)?
            .pop()
            .ok_or(AkitaError::InvalidProof)
    }

    fn q128_base_cache_bytes(entries: usize) -> usize {
        let bytes_per_coefficient = if ifma52_enabled() {
            3 * core::mem::size_of::<u64>()
        } else {
            Q128_NUM_PRIMES * core::mem::size_of::<i32>()
        };
        entries * D * bytes_per_coefficient
    }

    fn matrix() -> Vec<CyclotomicRing<F, D>> {
        (0..10)
            .map(|entry| {
                CyclotomicRing::from_coefficients(std::array::from_fn(|coefficient| {
                    F::from_i64(((entry * 17 + coefficient * 5) % 31) as i64 - 15)
                }))
            })
            .collect()
    }

    fn expected(
        matrix: &[CyclotomicRing<F, D>],
        rhs: &[CyclotomicRing<F, D>],
    ) -> Vec<CyclotomicRing<F, D>> {
        matrix
            .chunks_exact(rhs.len())
            .map(|row| {
                row.iter()
                    .zip(rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + (*lhs * *rhs)
                    })
            })
            .collect()
    }

    fn verifier_setup(matrix: &[CyclotomicRing<F, D>]) -> AkitaVerifierSetup<F> {
        AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 1,
                        max_num_batched_polys: 1,
                        num_field_elements: matrix.len(),
                        setup_seed: [9; 32].into(),
                    },
                    FlatMatrix::from_ring_slice(matrix),
                ),
            ),
            SetupPrefixVerifierRegistry::new([9; 32].into()),
        )
        .expect("matching public-matrix identity")
    }

    fn centered_rings(values: &[[i16; D]]) -> Vec<CyclotomicRing<F, D>> {
        values
            .iter()
            .map(|ring| {
                CyclotomicRing::from_coefficients(ring.map(|value| F::from_i64(i64::from(value))))
            })
            .collect()
    }

    #[test]
    fn terminal_i16_path_materializes_the_selected_exact_profile() {
        let matrix = matrix();
        let setup = verifier_setup(&matrix);
        let centered = (0..5)
            .map(|column| {
                std::array::from_fn(|coefficient| match (column + coefficient) % 5 {
                    0 => i16::MIN,
                    1 => -1024,
                    2 => -1,
                    3 => 1023,
                    _ => i16::MAX,
                })
            })
            .collect::<Vec<_>>();
        let needs_tail =
            ntt_cache_requires_exactness_tail::<F, D>(centered.len(), TERMINAL_I16_ABS_BOUND)
                .expect("tail capability");
        assert_eq!(
            centered_rows(&setup, 2, &centered, 10).expect("mixed i16 terminal matvec"),
            expected(&matrix, &centered_rings(&centered))
        );
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("cache bytes"),
            q128_base_cache_bytes(10)
                + usize::from(needs_tail) * 10 * D * core::mem::size_of::<i32>()
        );
    }

    #[test]
    fn terminal_cache_rejects_an_undersized_setup_without_panicking() {
        let matrix = matrix();
        let setup = verifier_setup(&matrix[..1]);
        let centered = vec![[1i16; D]; 5];
        assert!(matches!(
            centered_rows(&setup, 2, &centered, 10),
            Err(AkitaError::InvalidSetup(_))
        ));
    }

    #[test]
    fn q32_terminal_i16_width_uses_the_selected_exact_layout() {
        let matrix = (0..10)
            .map(|entry| {
                CyclotomicRing::<F32, D>::from_coefficients(std::array::from_fn(|coefficient| {
                    F32::from_i64(((entry * 17 + coefficient * 5) % 31) as i64 - 15)
                }))
            })
            .collect::<Vec<_>>();
        let setup = AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 1,
                        max_num_batched_polys: 1,
                        num_field_elements: matrix.len(),
                        setup_seed: [8; 32].into(),
                    },
                    FlatMatrix::from_ring_slice(&matrix),
                ),
            ),
            SetupPrefixVerifierRegistry::new([8; 32].into()),
        )
        .expect("matching public-matrix identity");
        let rhs = vec![[i16::MAX; D]; 5];
        let needs_tail =
            ntt_cache_requires_exactness_tail::<F32, D>(rhs.len(), TERMINAL_I16_ABS_BOUND)
                .expect("q32 terminal capability");
        let actual = centered_rows(&setup, 2, &rhs, 10).expect("q32 i16 terminal matvec");
        let centered_rhs = rhs
            .iter()
            .map(|ring| {
                CyclotomicRing::from_coefficients(ring.map(|value| F32::from_i64(i64::from(value))))
            })
            .collect::<Vec<_>>();
        let expected = matrix
            .chunks_exact(rhs.len())
            .map(|row| {
                row.iter()
                    .zip(&centered_rhs)
                    .fold(CyclotomicRing::zero(), |sum, (lhs, rhs)| {
                        sum + (*lhs * *rhs)
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        let base_bytes_per_coefficient = if ifma52_enabled() {
            core::mem::size_of::<u64>()
        } else {
            Q32_NUM_PRIMES * core::mem::size_of::<i32>()
        };
        let tail_bytes_per_coefficient = usize::from(needs_tail) * core::mem::size_of::<i16>();
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("cache bytes"),
            10 * D * (base_bytes_per_coefficient + tail_bytes_per_coefficient)
        );
    }

    #[test]
    fn dynamic_d128_inner_product_matches_schoolbook() {
        const D128: usize = 128;
        const WIDTH: usize = 17;
        let lhs = (0..WIDTH)
            .map(|column| {
                CyclotomicRing::<F64, D128>::from_coefficients(std::array::from_fn(|index| {
                    F64::from_i64(((column * 11 + index * 7) % 29) as i64 - 14)
                }))
            })
            .collect::<Vec<_>>();
        let rhs = (0..WIDTH)
            .map(|column| std::array::from_fn(|index| ((column * 5 + index * 3) % 31) as i16 - 15))
            .collect::<Vec<_>>();
        let expected =
            lhs.iter()
                .zip(&rhs)
                .fold(CyclotomicRing::<F64, D128>::zero(), |sum, (lhs, rhs)| {
                    let rhs = CyclotomicRing::from_coefficients(
                        rhs.map(|coefficient| F64::from_i64(i64::from(coefficient))),
                    );
                    sum + *lhs * rhs
                });
        assert_eq!(
            centered_inner_product(&lhs, &rhs).expect("exact dynamic terminal inner product"),
            expected
        );
    }

    #[test]
    fn schedule_warm_builds_terminal_cache_once_before_arithmetic() {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
            .expect("workspace schedule catalog");
        let group = PolynomialGroupLayout::new(15, 1);
        let schedule = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(group))
            .expect("adaptive schedule")
            .schedule()
            .clone();
        let params = &schedule.terminal;
        let prefix_len = params
            .inner
            .matrix
            .output_rank()
            .checked_mul(params.inner_width())
            .expect("terminal prefix");
        let field_len = prefix_len
            .checked_mul(params.d_a())
            .expect("terminal setup field length");
        let setup = AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 1,
                        max_num_batched_polys: 1,
                        num_field_elements: field_len,
                        setup_seed: [9; 32].into(),
                    },
                    FlatMatrix::from_flat_data(vec![F::from_i64(0); field_len]),
                ),
            ),
            SetupPrefixVerifierRegistry::new([9; 32].into()),
        )
        .expect("matching public-matrix identity");

        assert_eq!(setup.verifier_ntt_cache_bytes().expect("empty cache"), 0);
        warm_for_schedule(&setup, &schedule).expect("warm cache");
        let warmed_bytes = setup.verifier_ntt_cache_bytes().expect("warmed cache");
        assert!(warmed_bytes > 0);
        warm_for_schedule(&setup, &schedule).expect("reuse warm cache");
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("reused cache"),
            warmed_bytes
        );
    }

    #[test]
    fn distinct_exact_plans_preserve_smaller_flat_matrix_geometry() {
        let matrix = matrix();
        let setup = verifier_setup(&matrix);
        let wide_rhs = (0..5)
            .map(|column| {
                std::array::from_fn(|coefficient| ((column * 7 + coefficient) % 17) as i16 - 8)
            })
            .collect::<Vec<_>>();
        centered_rows(&setup, 2, &wide_rhs, 10).expect("wide cached product");
        let wide_cache_bytes = setup.verifier_ntt_cache_bytes().expect("wide cache bytes");

        let narrow_rhs = &wide_rhs[..3];
        assert_eq!(
            centered_rows(&setup, 2, narrow_rhs, 6).expect("narrow cached product"),
            expected(&matrix[..6], &centered_rings(narrow_rhs)),
        );
        let distinct_plan_bytes = setup
            .verifier_ntt_cache_bytes()
            .expect("distinct plan cache bytes");
        assert!(distinct_plan_bytes > wide_cache_bytes);
        centered_rows(&setup, 2, narrow_rhs, 6).expect("reuse narrow exact plan");
        assert_eq!(
            setup
                .verifier_ntt_cache_bytes()
                .expect("reused exact plans"),
            distinct_plan_bytes,
        );
    }

    #[test]
    fn exact_capabilities_do_not_alias_cache_entries() {
        let setup = verifier_setup(&matrix());
        let initial_needs_tail =
            ntt_cache_requires_exactness_tail::<F, D>(4, TERMINAL_I16_ABS_BOUND)
                .expect("initial exactness requirement");
        let initial_tail_len = usize::from(initial_needs_tail) * 4;

        let initial_tail = setup
            .prepared_verifier_ntt_prefix::<D>(4, initial_tail_len, 4, TERMINAL_I16_ABS_BOUND)
            .expect("initial exact prefix");
        assert_eq!(initial_tail.has_exactness_tail(), initial_needs_tail);

        let combined = setup
            .prepared_verifier_ntt_prefix::<D>(10, 0, 1, TERMINAL_I16_ABS_BOUND)
            .expect("larger base-only prefix");
        assert!(!combined.has_exactness_tail());
        assert!(!Arc::ptr_eq(&initial_tail, &combined));
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("separate bytes"),
            q128_base_cache_bytes(4)
                + initial_tail_len * D * core::mem::size_of::<i32>()
                + q128_base_cache_bytes(10)
        );
        let other_basis = setup
            .prepared_verifier_ntt_prefix::<D>(10, 0, 1, 1 << 14)
            .expect("distinct exact bound");
        assert!(!Arc::ptr_eq(&combined, &other_basis));

        let reused_tail = setup
            .prepared_verifier_ntt_prefix::<D>(4, initial_tail_len, 4, TERMINAL_I16_ABS_BOUND)
            .expect("reused exact prefix");
        assert!(Arc::ptr_eq(&initial_tail, &reused_tail));
    }
}
