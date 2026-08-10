use super::CpuBackend;
use crate::compute::backend::RingSwitchComputeBackend;
use crate::compute::plans::{
    RingSwitchQuotientRowsPlan, RingSwitchRelationRows, RingSwitchRelationRowsPlan,
};
use crate::kernels::linear::{
    centered_quotient_rows_with_i16_tail, fused_split_eq_quotients_prover_bounds,
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

impl<F> RingSwitchComputeBackend<F> for CpuBackend
where
    F: FieldCore + CanonicalField,
{
    fn ring_switch_relation_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchRelationRowsPlan<'_, D>,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: HalvingField,
    {
        let mut cyclic_requirement: Option<NttCacheKey> = None;
        for (rows, width) in [
            (plan.n_d, plan.e_hat.len()),
            (plan.n_b, plan.t_hat.len()),
            (plan.n_a, plan.z_segment.len()),
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
                    plan.e_hat,
                    plan.t_hat,
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
                plan.z_segment.len(),
                NttTransformDomain::Negacyclic,
            )?;
            prepared.with_shared_ntt::<D, _>(negacyclic_requirement, |negacyclic_ntt| {
                if centered_quotient_requires_i16_tail_for_field::<F, D>(centered_rhs_abs_bound(
                    plan.z_segment,
                    plan.z_folded_centered_inf_norm,
                ))? {
                    let tail_requirement = NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_a,
                        plan.z_segment.len(),
                        NttTransformDomain::I16TailBothTransforms,
                    )?;
                    return prepared.with_shared_ntt::<D, _>(tail_requirement, |tail_ntt| {
                        let (d_cyclic, b_cyclic, _) = fused_split_eq_quotients_prover_bounds(
                            negacyclic_ntt,
                            cyclic_ntt,
                            plan.n_d,
                            plan.n_b,
                            0,
                            plan.e_hat,
                            plan.t_hat,
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
                            plan.z_segment,
                            plan.z_folded_centered_inf_norm,
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
                    plan.e_hat,
                    plan.t_hat,
                    plan.z_segment,
                    plan.z_folded_centered_inf_norm,
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

    fn ring_switch_quotient_rows<const D: usize>(
        &self,
        prepared: &Self::PreparedSetup,
        plan: RingSwitchQuotientRowsPlan<'_, D>,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError>
    where
        F: HalvingField,
    {
        let cyclic = NttCacheKey::from_matrix_shape(
            D,
            plan.n_a,
            plan.z_segment.len(),
            NttTransformDomain::Cyclic,
        )?;
        let negacyclic = NttCacheKey::from_matrix_shape(
            D,
            plan.n_a,
            plan.z_segment.len(),
            NttTransformDomain::Negacyclic,
        )?;
        prepared.with_shared_ntt::<D, _>(cyclic, |cyclic_ntt| {
            prepared.with_shared_ntt::<D, _>(negacyclic, |negacyclic_ntt| {
                if centered_quotient_requires_i16_tail_for_field::<F, D>(centered_rhs_abs_bound(
                    plan.z_segment,
                    plan.z_folded_centered_inf_norm,
                ))? {
                    let tail = NttCacheKey::from_matrix_shape(
                        D,
                        plan.n_a,
                        plan.z_segment.len(),
                        NttTransformDomain::I16TailBothTransforms,
                    )?;
                    return prepared.with_shared_ntt::<D, _>(tail, |tail_ntt| {
                        centered_quotient_rows_with_i16_tail(
                            negacyclic_ntt,
                            cyclic_ntt,
                            tail_ntt,
                            plan.n_a,
                            plan.z_segment,
                            plan.z_folded_centered_inf_norm,
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
                    plan.z_segment,
                    plan.z_folded_centered_inf_norm,
                    1,
                    1,
                )?;
                Ok(a_quotients)
            })
        })
    }
}
