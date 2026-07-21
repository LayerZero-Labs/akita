//! Exact negacyclic NTT kernels for terminal verifier matrix relations.

use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{
    dispatch_for_field, ntt_cache_requires_i16_tail, AkitaVerifierSetup, OpeningClaimsLayout,
    PolynomialGroupLayout, Schedule,
};

pub(super) const TERMINAL_I16_LOG_BASIS: u32 = 16;

/// Warm every exact terminal i16 representation selected by a validated schedule.
pub(super) fn warm_for_schedule<F: FieldCore + CanonicalField>(
    setup: &AkitaVerifierSetup<F>,
    schedule: &Schedule,
) -> Result<(), AkitaError> {
    let terminal_params = &schedule
        .folds
        .last()
        .ok_or_else(|| AkitaError::InvalidSetup("schedule has no terminal fold".into()))?
        .params;
    let precommitteds = terminal_params
        .precommitted_group_iter()
        .map(|params| params.layout.group)
        .collect::<Vec<_>>();
    let opening_batch = OpeningClaimsLayout::from_root_groups(
        &precommitteds,
        PolynomialGroupLayout::new(terminal_params.recursive_opening_num_vars()?, 1),
    )?;
    let ring_d = terminal_params.role_dims().d_a();
    dispatch_for_field!(
        akita_types::ProtocolDispatchSlot::Role(akita_types::RingRole::Inner),
        F,
        ring_d,
        |D| {
            let mut base_prefix_len = 0usize;
            let mut tail_prefix_len = 0usize;
            let mut max_width = 0usize;
            for group_index in 0..opening_batch.num_groups() {
                let params = terminal_params.group_params(&opening_batch, group_index)?;
                let width = params.a_col_len();
                let prefix_len =
                    params
                        .a_rows_len()
                        .checked_mul(width)
                        .ok_or(AkitaError::InvalidSetup(
                            "terminal A cache prefix length overflow".into(),
                        ))?;
                base_prefix_len = base_prefix_len.max(prefix_len);
                max_width = max_width.max(width);
                if ntt_cache_requires_i16_tail::<F, D>(width, TERMINAL_I16_LOG_BASIS)? {
                    tail_prefix_len = tail_prefix_len.max(prefix_len);
                }
            }
            if base_prefix_len > 0 {
                setup.prepared_verifier_ntt_prefix::<D>(
                    base_prefix_len,
                    tail_prefix_len,
                    max_width,
                    TERMINAL_I16_LOG_BASIS,
                )?;
            }
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
    F: FieldCore + CanonicalField,
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
            if ntt_cache_requires_i16_tail::<F, D>(rhs.len(), TERMINAL_I16_LOG_BASIS)? {
                prepared_prefix_len
            } else {
                0
            };
        setup.prepared_verifier_ntt_prefix::<D>(
            prepared_prefix_len,
            tail_prefix_len,
            rhs.len(),
            TERMINAL_I16_LOG_BASIS,
        )?
    };
    let _span = tracing::info_span!("terminal_ntt_a_i16_accumulate").entered();
    slot.mat_vec_i16(TERMINAL_I16_LOG_BASIS, num_rows, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES};
    use akita_config::{proof_optimized::fp128::D64OneHot, CommitmentConfig};
    use akita_field::{Prime128Offset275 as F, Prime32Offset99 as F32};
    use akita_types::{
        max_safe_crt_accumulation_width, select_crt_ntt_params, AkitaExpandedSetup,
        AkitaScheduleLookupKey, AkitaSetupSeed, FlatMatrix, LevelParamsLike, PolynomialGroupLayout,
        ProtocolCrtNttParams, SetupPrefixVerifierRegistry,
    };
    use std::sync::Arc;

    const D: usize = 64;

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
                    AkitaSetupSeed {
                        max_num_vars: 1,
                        max_num_batched_polys: 1,
                        gen_ring_dim: D,
                        max_setup_len: matrix.len(),
                        public_matrix_seed: [9; 32],
                    },
                    FlatMatrix::from_ring_slice(matrix),
                ),
            ),
            SetupPrefixVerifierRegistry::new(),
        )
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
    fn terminal_i16_tail_path_materializes_exactly_one_small_prime() {
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
        assert!(
            ntt_cache_requires_i16_tail::<F, D>(5, TERMINAL_I16_LOG_BASIS)
                .expect("tail capability")
        );
        assert_eq!(
            centered_rows(&setup, 2, &centered, 10).expect("mixed i16 terminal matvec"),
            expected(&matrix, &centered_rings(&centered))
        );
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("cache bytes"),
            10 * D * (Q128_NUM_PRIMES * core::mem::size_of::<i32>() + core::mem::size_of::<i16>())
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
    fn q32_terminal_i16_width_uses_only_the_base_cache() {
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
                    AkitaSetupSeed {
                        max_num_vars: 1,
                        max_num_batched_polys: 1,
                        gen_ring_dim: D,
                        max_setup_len: matrix.len(),
                        public_matrix_seed: [8; 32],
                    },
                    FlatMatrix::from_ring_slice(&matrix),
                ),
            ),
            SetupPrefixVerifierRegistry::new(),
        );
        let rhs = vec![[i16::MAX; D]; 5];
        assert!(
            !ntt_cache_requires_i16_tail::<F32, D>(rhs.len(), TERMINAL_I16_LOG_BASIS)
                .expect("q32 terminal capability")
        );
        centered_rows(&setup, 2, &rhs, 10).expect("q32 i16 terminal matvec");
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("cache bytes"),
            10 * D * Q32_NUM_PRIMES * core::mem::size_of::<i32>()
        );
    }

    #[test]
    fn schedule_warm_builds_terminal_cache_once_before_arithmetic() {
        let group = PolynomialGroupLayout::new(15, 1);
        let schedule = D64OneHot::runtime_schedule(AkitaScheduleLookupKey::single(group))
            .expect("D64 schedule");
        let params = &schedule.folds.last().expect("terminal fold").params;
        let prefix_len = params
            .a_rows_len()
            .checked_mul(params.a_col_len())
            .expect("terminal prefix");
        let matrix = vec![CyclotomicRing::<F, D>::zero(); prefix_len];
        let setup = verifier_setup(&matrix);

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
    fn base_and_tail_requests_share_one_strongest_prefix() {
        let setup = verifier_setup(&matrix());
        let ProtocolCrtNttParams::Q128(params) =
            select_crt_ntt_params::<F, D>().expect("Q128 params")
        else {
            panic!("Q128 field must select Q128 params");
        };
        let safe_width = max_safe_crt_accumulation_width::<F, _, Q128_NUM_PRIMES, D>(
            &params,
            1 << (TERMINAL_I16_LOG_BASIS - 1),
        )
        .expect("base profile supports a terminal width");
        assert!(safe_width < 4);

        let initial_tail = setup
            .prepared_verifier_ntt_prefix::<D>(4, 4, safe_width + 1, TERMINAL_I16_LOG_BASIS)
            .expect("tail prefix");
        assert!(initial_tail.has_i16_tail());

        let combined = setup
            .prepared_verifier_ntt_prefix::<D>(10, 0, safe_width, TERMINAL_I16_LOG_BASIS)
            .expect("larger base-only prefix");
        assert!(combined.has_i16_tail());
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("combined bytes"),
            10 * D * Q128_NUM_PRIMES * core::mem::size_of::<i32>()
                + 4 * D * core::mem::size_of::<i16>()
        );
        let reused_other_basis = setup
            .prepared_verifier_ntt_prefix::<D>(10, 0, 1, 15)
            .expect("same physical cache for another exact bound");
        assert!(Arc::ptr_eq(&combined, &reused_other_basis));

        let reused_tail = setup
            .prepared_verifier_ntt_prefix::<D>(4, 4, safe_width + 1, TERMINAL_I16_LOG_BASIS)
            .expect("reused tail prefix");
        assert!(Arc::ptr_eq(&combined, &reused_tail));
    }
}
