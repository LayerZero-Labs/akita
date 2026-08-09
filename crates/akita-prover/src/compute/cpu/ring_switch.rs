use super::*;
use crate::backend::{RingSwitchQuotientView, RingSwitchRelationView};

impl<F, const D: usize> RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: RingSwitchRelationView<'_, D>,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        // The root-level relation spans nearly the whole matrix but reads
        // each element exactly once per prove — stream its transforms from
        // the field form instead of materializing a matrix-scale NTT cache
        // for one pass. Small (deeper-level) extents keep the cached path,
        // which is shared with the per-level digit-row products.
        let stream_extent = fused_quotient_matrix_extent(
            plan.n_d,
            source.e_hat.len(),
            plan.n_b,
            source.t_hat.len(),
            plan.n_a,
            source.z_segment.len(),
        )?;
        if !self.ntt_operation_uses_cache(NttOperationCluster::RingSwitch, stream_extent) {
            let view = prepared
                .expanded
                .shared_matrix()
                .ring_view::<D>(1, stream_extent)?;
            let streamed = fused_split_eq_quotients_streamed_prover_bounds(
                view.as_slice(),
                plan.n_d,
                plan.n_b,
                plan.n_a,
                source.e_hat,
                source.t_hat,
                source.z_segment,
                source.z_folded_centered_inf_norm,
                plan.log_basis_open,
                plan.log_basis_outer,
            )?;
            if let Some((d_cyclic, b_cyclic, a_quotients)) = streamed {
                return Ok(RingSwitchRelationRows {
                    d_cyclic,
                    b_cyclic,
                    a_quotients,
                });
            }
        }
        let mut cyclic_requirement: Option<NttCacheKey> = None;
        for (rows, width) in [
            (plan.n_d, source.e_hat.len()),
            (plan.n_b, source.t_hat.len()),
            (plan.n_a, source.z_segment.len()),
        ] {
            if rows == 0 && width == 0 {
                continue;
            }
            let role_requirement =
                NttCacheKey::from_matrix_shape(D, rows, width, NttTransformDomain::Cyclic)?;
            cyclic_requirement = Some(match cyclic_requirement {
                Some(current) => current.join(role_requirement)?,
                None => role_requirement,
            });
        }
        let cyclic_requirement = cyclic_requirement.ok_or_else(|| {
            AkitaError::InvalidSetup("ring-switch relation has no active rows".into())
        })?;
        prepared.with_shared_ntt::<D, _>(cyclic_requirement, |cyclic_ntt| {
            if plan.n_a == 0 {
                let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                    cyclic_ntt,
                    cyclic_ntt,
                    plan.n_d,
                    plan.n_b,
                    0,
                    source.e_hat,
                    source.t_hat,
                    &[],
                    0,
                    plan.log_basis_open,
                    plan.log_basis_outer,
                )?;
                return Ok(RingSwitchRelationRows {
                    d_cyclic,
                    b_cyclic,
                    a_quotients,
                });
            }
            let negacyclic_requirement = NttCacheKey::from_matrix_shape(
                D,
                plan.n_a,
                source.z_segment.len(),
                NttTransformDomain::Negacyclic,
            )?;
            prepared.with_shared_ntt::<D, _>(negacyclic_requirement, |negacyclic_ntt| {
                let (d_cyclic, b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    plan.n_d,
                    plan.n_b,
                    plan.n_a,
                    source.e_hat,
                    source.t_hat,
                    source.z_segment,
                    source.z_folded_centered_inf_norm,
                    plan.log_basis_open,
                    plan.log_basis_outer,
                )?;
                Ok(RingSwitchRelationRows {
                    d_cyclic,
                    b_cyclic,
                    a_quotients,
                })
            })
        })
    }
}

impl<F, const D: usize> RingSwitchQuotientKernel<RingSwitchQuotientView<'_, D>, F, D> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn quotient_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: RingSwitchQuotientView<'_, D>,
        plan: RingSwitchQuotientPlan,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        let stream_extent =
            fused_quotient_matrix_extent(0, 0, 0, 0, plan.n_a, source.z_segment.len())?;
        if !self.ntt_operation_uses_cache(NttOperationCluster::RingSwitch, stream_extent) {
            let view = prepared
                .expanded
                .shared_matrix()
                .ring_view::<D>(1, stream_extent)?;
            let streamed = fused_split_eq_quotients_streamed_prover_bounds(
                view.as_slice(),
                0,
                0,
                plan.n_a,
                &[][..],
                &[][..],
                source.z_segment,
                source.z_folded_centered_inf_norm,
                1,
                1,
            )?;
            if let Some((_d_cyclic, _b_cyclic, a_quotients)) = streamed {
                return Ok(a_quotients);
            }
        }
        let cyclic = NttCacheKey::from_matrix_shape(
            D,
            plan.n_a,
            source.z_segment.len(),
            NttTransformDomain::Cyclic,
        )?;
        let negacyclic = NttCacheKey::from_matrix_shape(
            D,
            plan.n_a,
            source.z_segment.len(),
            NttTransformDomain::Negacyclic,
        )?;
        prepared.with_shared_ntt::<D, _>(cyclic, |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic, |negacyclic_ntt| {
                let (_d_cyclic, _b_cyclic, a_quotients) = fused_split_eq_quotients_prover_bounds(
                    negacyclic_ntt,
                    cyclic_ntt,
                    0,
                    0,
                    plan.n_a,
                    &[][..],
                    &[][..],
                    source.z_segment,
                    source.z_folded_centered_inf_norm,
                    1,
                    1,
                )?;
                Ok(a_quotients)
            })
        })
    }
}
