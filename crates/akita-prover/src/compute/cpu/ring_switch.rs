use super::CpuBackend;
use crate::backend::{RingSwitchQuotientView, RingSwitchRelationView};
use crate::compute::kernels::{RingSwitchQuotientKernel, RingSwitchRelationKernel};
use crate::compute::operation_plans::{RingSwitchQuotientPlan, RingSwitchRelationPlan};
use crate::compute::plans::RingSwitchRelationRows;
use crate::compute::requirements::NttOperationCluster;
use crate::kernels::linear::{
    centered_quotient_rows_with_i16_tail, fused_quotient_matrix_extent,
    fused_split_eq_quotients_prover_bounds, fused_split_eq_quotients_streamed_prover_bounds,
};
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_types::{centered_quotient_requires_i16_tail_for_field, NttCacheKey, NttTransformDomain};

fn centered_rhs_abs_bound<const D: usize>(rows: &[[i32; D]], claimed: u32) -> u64 {
    rows.iter()
        .flat_map(|row| row.iter())
        .map(|value| u64::from(value.unsigned_abs()))
        .max()
        .unwrap_or(0)
        .max(u64::from(claimed))
}

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
            if let Some((d_cyclic, b_cyclic, a_quotients)) =
                fused_split_eq_quotients_streamed_prover_bounds(
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
                )?
            {
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
                if centered_quotient_requires_i16_tail_for_field::<F, D>(centered_rhs_abs_bound(
                    source.z_segment,
                    source.z_folded_centered_inf_norm,
                ))? {
                    let tail_requirement = NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_a,
                        source.z_segment.len(),
                        NttTransformDomain::I16TailBothTransforms,
                    )?;
                    return prepared.with_shared_ntt::<D, _>(tail_requirement, |tail_ntt| {
                        let (d_cyclic, b_cyclic, _) = fused_split_eq_quotients_prover_bounds(
                            negacyclic_ntt,
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
                        let a_quotients = centered_quotient_rows_with_i16_tail(
                            negacyclic_ntt,
                            cyclic_ntt,
                            tail_ntt,
                            plan.n_a,
                            source.z_segment,
                            source.z_folded_centered_inf_norm,
                        )?;
                        Ok(RingSwitchRelationRows {
                            d_cyclic,
                            b_cyclic,
                            a_quotients,
                        })
                    });
                }
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
            if let Some((_d_cyclic, _b_cyclic, a_quotients)) =
                fused_split_eq_quotients_streamed_prover_bounds(
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
                )?
            {
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
                if centered_quotient_requires_i16_tail_for_field::<F, D>(centered_rhs_abs_bound(
                    source.z_segment,
                    source.z_folded_centered_inf_norm,
                ))? {
                    let tail = NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_a,
                        source.z_segment.len(),
                        NttTransformDomain::I16TailBothTransforms,
                    )?;
                    return prepared.with_shared_ntt::<D, _>(tail, |tail_ntt| {
                        centered_quotient_rows_with_i16_tail(
                            negacyclic_ntt,
                            cyclic_ntt,
                            tail_ntt,
                            plan.n_a,
                            source.z_segment,
                            source.z_folded_centered_inf_norm,
                        )
                    });
                }
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
