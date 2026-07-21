//! Exact negacyclic NTT kernels for terminal verifier matrix relations.

use akita_algebra::{CyclotomicCrtNtt, CyclotomicRing, MixedCrtNtt};
use akita_field::{AkitaError, CanonicalField, FieldCore};
use akita_types::{
    dispatch_for_field, select_crt_ntt_capability, AkitaVerifierSetup, CrtAccumulationProfile,
    OpeningClaimsLayout, PolynomialGroupLayout, PreparedNttCapabilitySlot, PreparedNttSlot,
    PreparedVerifierNttSlotAny, Schedule,
};

pub(super) const TERMINAL_I16_ABS_BOUND: u64 = 1 << 15;

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
            let mut base_requirement = None;
            let mut tail_requirement = None;
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
                let update = |requirement: &mut Option<(usize, usize)>| {
                    if requirement.is_none_or(|(_, current_prefix)| prefix_len > current_prefix) {
                        *requirement = Some((width, prefix_len));
                    }
                };
                match select_crt_ntt_capability::<F, D>(width, TERMINAL_I16_ABS_BOUND)?.profile() {
                    CrtAccumulationProfile::Base => update(&mut base_requirement),
                    CrtAccumulationProfile::I16Tail => update(&mut tail_requirement),
                }
            }
            for requirement in [base_requirement, tail_requirement] {
                if let Some((width, prefix_len)) = requirement {
                    setup.prepared_verifier_ntt_prefix::<D>(
                        prefix_len,
                        width,
                        TERMINAL_I16_ABS_BOUND,
                    )?;
                }
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

    let profile = select_crt_ntt_capability::<F, D>(rhs.len(), TERMINAL_I16_ABS_BOUND)?.profile();
    let slot = {
        let _span = tracing::info_span!("terminal_ntt_a_i16_cache_lookup", ?profile).entered();
        setup.prepared_verifier_ntt_prefix::<D>(
            prepared_prefix_len,
            rhs.len(),
            TERMINAL_I16_ABS_BOUND,
        )?
    };
    let _span = tracing::info_span!("terminal_ntt_a_i16_accumulate", ?profile).entered();
    match (profile, slot.as_ref()) {
        (CrtAccumulationProfile::Base, PreparedVerifierNttSlotAny::Base(slot)) => {
            match slot.as_d::<D>()? {
                PreparedNttSlot::Q32 { neg, params, .. } => {
                    CyclotomicCrtNtt::mat_vec_i16(neg, num_rows, rhs.len(), rhs, params)
                }
                PreparedNttSlot::Q64 { neg, params, .. } => {
                    CyclotomicCrtNtt::mat_vec_i16(neg, num_rows, rhs.len(), rhs, params)
                }
                PreparedNttSlot::Q128 { neg, params, .. } => {
                    CyclotomicCrtNtt::mat_vec_i16(neg, num_rows, rhs.len(), rhs, params)
                }
            }
        }
        (CrtAccumulationProfile::I16Tail, PreparedVerifierNttSlotAny::I16Tail(slot)) => {
            match slot.as_d::<D>()? {
                PreparedNttCapabilitySlot::Q32I16Tail { neg, params } => {
                    MixedCrtNtt::mat_vec_i16(neg, num_rows, rhs.len(), rhs, params)
                }
                PreparedNttCapabilitySlot::Q64I16Tail { neg, params } => {
                    MixedCrtNtt::mat_vec_i16(neg, num_rows, rhs.len(), rhs, params)
                }
                PreparedNttCapabilitySlot::Q128I16Tail { neg, params } => {
                    MixedCrtNtt::mat_vec_i16(neg, num_rows, rhs.len(), rhs, params)
                }
                PreparedNttCapabilitySlot::Q32Base { .. }
                | PreparedNttCapabilitySlot::Q64Base { .. }
                | PreparedNttCapabilitySlot::Q128Base { .. } => Err(AkitaError::InvalidSetup(
                    "verifier i16-tail cache contains a base-only slot".into(),
                )),
            }
        }
        _ => Err(AkitaError::InvalidSetup(
            "verifier NTT cache profile does not match the requested capability".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::ntt::tables::{Q128_NUM_PRIMES, Q32_NUM_PRIMES};
    use akita_config::{proof_optimized::fp128::D64OneHot, CommitmentConfig};
    use akita_field::{Prime128Offset275 as F, Prime32Offset99 as F32};
    use akita_types::{
        AkitaExpandedSetup, AkitaScheduleLookupKey, AkitaSetupSeed, FlatMatrix, LevelParamsLike,
        PolynomialGroupLayout, SetupPrefixVerifierRegistry,
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
        assert_eq!(
            select_crt_ntt_capability::<F, D>(5, TERMINAL_I16_ABS_BOUND)
                .expect("tail capability")
                .profile(),
            CrtAccumulationProfile::I16Tail
        );
        assert_eq!(
            centered_rows(&setup, 2, &centered, 10).expect("mixed i16 terminal matvec"),
            expected(&matrix, &centered_rings(&centered))
        );
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("cache bytes"),
            10 * core::mem::size_of::<MixedCrtNtt<Q128_NUM_PRIMES, D>>()
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
        assert_eq!(
            select_crt_ntt_capability::<F32, D>(rhs.len(), TERMINAL_I16_ABS_BOUND)
                .expect("q32 terminal capability")
                .profile(),
            CrtAccumulationProfile::Base
        );
        centered_rows(&setup, 2, &rhs, 10).expect("q32 i16 terminal matvec");
        assert_eq!(
            setup.verifier_ntt_cache_bytes().expect("cache bytes"),
            10 * core::mem::size_of::<CyclotomicCrtNtt<i32, Q32_NUM_PRIMES, D>>()
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
}
