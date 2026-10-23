//! Experimental source-aware NTT commitment variants.
use super::{CpuBackend, CpuPreparedSetup};
use crate::opaque::{CommitInnerPlan, OneHotSource};
use crate::OneHotIndex;
use akita_error::{checked, AkitaError};
use akita_types::{NttCacheKey, NttTransformDomain, RingVec};
use jolt_field::{CanonicalEncoding, Field};
#[cfg(test)]
mod benchmark;
mod decode;
mod legacy;
mod panel;

impl<F: Field + CanonicalEncoding, E> CpuBackend<F, E> {
    pub(crate) fn onehot_ntt_commit_rows<const D: usize, I: OneHotIndex>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        sources: &[OneHotSource<'_, I>],
        plan: CommitInnerPlan,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        let mode = std::env::var("AKITA_ONEHOT_COMMIT").unwrap_or_else(|_| "ntt".into());
        let group = std::env::var("AKITA_ONEHOT_NTT_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        self.onehot_ntt_variant::<D, I>(prepared, sources, plan, &mode, group)
    }

    fn onehot_ntt_variant<const D: usize, I: OneHotIndex>(
        &self,
        prepared: &CpuPreparedSetup<F>,
        sources: &[OneHotSource<'_, I>],
        plan: CommitInnerPlan,
        mode: &str,
        group: usize,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        if mode == "ntt-legacy" {
            return self.onehot_ntt_legacy_rows::<D, I>(prepared, sources, plan);
        }
        if group == 0 || group > 32 {
            return Err(AkitaError::InvalidInput(
                "NTT block panel must be in 1..=32".into(),
            ));
        }
        let width = checked::product([plan.num_positions_per_block, plan.num_digits_inner])
            .ok_or_else(|| AkitaError::InvalidSetup("one-hot NTT width overflow".into()))?;
        if mode == "ntt-exact" {
            // Direct one-hot decode emits only 0/1, which the basis-2 matvec
            // validates independently of the protocol's inner gadget basis.
            let bound = 2;
            let key = NttCacheKey::from_matrix_shape(
                D,
                plan.n_a,
                width,
                NttTransformDomain::ExactNegacyclicI16 {
                    width,
                    rhs_abs_bound: bound,
                },
            )?;
            return prepared.with_shared_ntt::<D, _>(key, |ntt| {
                use jolt_field::solinas::parallel::*;
                eprintln!(
                    "[onehot_ntt] variant=exact uses_ifma52={} cache_bytes={}",
                    ntt.uses_ifma52(),
                    ntt.cache_bytes()
                );
                sources
                    .iter()
                    .map(|source| {
                        decode::validate::<D, I>(*source, plan)?;
                        let rows = cfg_into_iter!(0..plan.num_live_blocks)
                            .map(|block| {
                                let mut rhs = vec![[0i16; D]; width];
                                decode::fill(*source, plan, block, 0, &mut rhs)?;
                                ntt.mat_vec_i16::<F>(2, plan.n_a, &rhs)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        crate::typed_inner_rows(rows)
                    })
                    .collect()
            });
        }
        let key =
            NttCacheKey::from_matrix_shape(D, plan.n_a, width, NttTransformDomain::Negacyclic)?;
        prepared.with_shared_ntt::<D, _>(key, |ntt| {
            let base = ntt.q128_base().ok_or_else(|| {
                AkitaError::InvalidSetup("one-hot panel experiment requires fp128".into())
            })?;
            let matrix = base
                .negacyclic()
                .ok_or_else(|| AkitaError::InvalidSetup("missing negacyclic cache".into()))?;
            sources
                .iter()
                .map(|source| {
                    decode::validate::<D, I>(*source, plan)?;
                    let rows = match mode {
                        "ntt" | "ntt-panel5" | "ntt-binary5" => panel::commit::<F, I, 5, D>(
                            matrix,
                            *source,
                            plan,
                            group,
                            mode == "ntt-binary5",
                            usize::MAX,
                        ),
                        "ntt-panel6" => panel::commit::<F, I, 6, D>(
                            matrix,
                            *source,
                            plan,
                            group,
                            false,
                            usize::MAX,
                        ),
                        _ => Err(AkitaError::InvalidInput(format!(
                            "unknown one-hot NTT variant {mode}"
                        ))),
                    }?;
                    crate::typed_inner_rows(rows)
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opaque::ComputeBackendSetup;
    use crate::sources::onehot::column_sweep_ajtai_onehot_multi;
    use crate::AkitaProverSetup;
    use akita_params::SetupMatrixCapacity;
    use jolt_field::Prime128OffsetA7F7 as F;

    fn compare<const D: usize>() {
        let backend = CpuBackend::<F, F>::for_arithmetic_tests();
        for digits in [1, 2] {
            let plan = CommitInnerPlan {
                ring_dimension: D,
                num_live_blocks: 2,
                n_a: 2,
                num_positions_per_block: 8,
                num_digits_inner: digits,
                log_basis_inner: 3,
            };
            let setup = AkitaProverSetup::<F>::generate_with_capacity(
                D.trailing_zeros() as usize + 4,
                1,
                SetupMatrixCapacity {
                    num_field_elements: 2 * 8 * digits * D,
                },
            )
            .unwrap();
            let prepared = backend.prepare_setup(&setup).unwrap();
            for chunk_size in [16, D, D * 2] {
                // Sixteen rings over two blocks, including missing chunks.
                let indices = (0..D * 16 / chunk_size)
                    .map(|i| (!i.is_multiple_of(7)).then_some((i * 11) % chunk_size))
                    .collect::<Vec<_>>();
                let source = OneHotSource {
                    indices: &indices,
                    chunk_size,
                    num_vars: D.trailing_zeros() as usize + 4,
                };

                let view = setup
                    .expanded
                    .shared_matrix
                    .ring_view::<D>(2, 8 * digits)
                    .unwrap();
                let expected = column_sweep_ajtai_onehot_multi::<F, D, _>(
                    &view,
                    &[source],
                    2,
                    8 * digits,
                    digits,
                )
                .unwrap();
                let expected =
                    crate::typed_inner_rows(expected.into_iter().next().unwrap()).unwrap();
                let width = plan.num_positions_per_block * plan.num_digits_inner;
                let key = NttCacheKey::from_matrix_shape(
                    D,
                    plan.n_a,
                    width,
                    NttTransformDomain::Negacyclic,
                )
                .unwrap();
                let chunked = prepared
                    .with_shared_ntt::<D, _>(key, |ntt| {
                        let base = ntt.q128_base().unwrap();
                        panel::commit::<F, _, 5, D>(
                            base.negacyclic().unwrap(),
                            source,
                            plan,
                            2,
                            false,
                            3,
                        )
                    })
                    .unwrap();
                assert_eq!(
                    crate::typed_inner_rows(chunked).unwrap().coeffs(),
                    expected.coeffs()
                );
                for mode in [
                    "ntt-legacy",
                    "ntt-panel6",
                    "ntt-panel5",
                    "ntt-binary5",
                    "ntt-exact",
                ] {
                    for group in [1, 4] {
                        let actual = backend
                            .onehot_ntt_variant::<D, _>(&prepared, &[source], plan, mode, group)
                            .unwrap();
                        assert_eq!(
                            actual[0].coeffs(),
                            expected.coeffs(),
                            "{mode} group={group} D={D} digits={digits} chunk={chunk_size}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn onehot_ntt_matches_accumulation_d256() {
        compare::<256>();
    }
    #[test]
    fn onehot_ntt_matches_accumulation_d512() {
        compare::<512>();
    }
}
