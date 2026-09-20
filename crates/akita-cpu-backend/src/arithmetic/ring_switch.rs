use super::{CpuBackend, CpuPreparedSetup};
use crate::arithmetic::requirements::NttOperationCluster;
use crate::kernels::linear::{
    centered_quotient_rows_with_i16_tail, digit_relation_matrix_extent,
    digit_relation_rows_cached_prover_bounds, digit_relation_rows_streamed_prover_bounds,
    fused_quotient_matrix_extent, fused_split_eq_quotients_prover_bounds,
    fused_split_eq_quotients_streamed_prover_bounds, DigitRelationRows, FusedQuotientRows,
};
use crate::opaque::plans::RingSwitchRelationRows;
use crate::opaque::RingSwitchRelationKernel;
use crate::opaque::RingSwitchRelationPlan;
use crate::opaque::RingSwitchRelationView;
use akita_error::AkitaError;
use akita_types::{centered_quotient_requires_i16_tail_for_field, NttCacheKey, NttTransformDomain};
use jolt_field::{CanonicalEncoding, Field};

fn centered_rhs_abs_bound<const D: usize>(rows: &[[i32; D]], claimed: u32) -> u64 {
    rows.iter()
        .flat_map(|row| row.iter())
        .map(|value| u64::from(value.unsigned_abs()))
        .max()
        .unwrap_or(0)
        .max(u64::from(claimed))
}

fn validate_role_shape(role: &str, rows: usize, width: usize) -> Result<(), AkitaError> {
    if rows != 0 && width == 0 {
        return Err(AkitaError::InvalidInput(format!(
            "active ring-switch {role} role must have a nonzero source width"
        )));
    }
    Ok(())
}

fn cached_b_a_rows<F: Field + CanonicalEncoding, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    t_hat: &[[i8; D]],
    z: &[[i32; D]],
    z_inf_norm: u32,
    plan: RingSwitchRelationPlan,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
    let mut cyclic_requirement: Option<NttCacheKey> = None;
    for (rows, width) in [(plan.n_b, t_hat.len()), (plan.n_a, z.len())] {
        if rows == 0 {
            continue;
        }
        let role_requirement =
            NttCacheKey::from_matrix_shape(D, rows, width, NttTransformDomain::Cyclic)?;
        cyclic_requirement = Some(match cyclic_requirement {
            Some(current) => current.join(role_requirement)?,
            None => role_requirement,
        });
    }
    let cyclic_requirement = cyclic_requirement
        .ok_or_else(|| AkitaError::InvalidSetup("ring-switch relation has no B/A rows".into()))?;
    prepared.with_shared_ntt::<D, _>(cyclic_requirement, |cyclic_ntt| {
        if plan.n_a == 0 {
            let rows = fused_split_eq_quotients_prover_bounds(
                cyclic_ntt,
                cyclic_ntt,
                plan.n_b,
                0,
                t_hat,
                &[],
                0,
                plan.log_basis_outer,
            )?;
            return Ok(rows);
        }
        let negacyclic_requirement =
            NttCacheKey::from_matrix_shape(D, plan.n_a, z.len(), NttTransformDomain::Negacyclic)?;
        prepared.with_shared_ntt::<D, _>(negacyclic_requirement, |negacyclic_ntt| {
            if centered_quotient_requires_i16_tail_for_field::<F, D>(centered_rhs_abs_bound(
                z, z_inf_norm,
            ))? {
                let tail_requirement = NttCacheKey::from_matrix_shape(
                    D,
                    plan.n_a,
                    z.len(),
                    NttTransformDomain::I16TailBothTransforms,
                )?;
                return prepared.with_shared_ntt::<D, _>(tail_requirement, |tail_ntt| {
                    let b_rows = fused_split_eq_quotients_prover_bounds(
                        negacyclic_ntt,
                        cyclic_ntt,
                        plan.n_b,
                        0,
                        t_hat,
                        &[],
                        0,
                        plan.log_basis_outer,
                    )?;
                    let a_quotients = centered_quotient_rows_with_i16_tail(
                        negacyclic_ntt,
                        cyclic_ntt,
                        tail_ntt,
                        plan.n_a,
                        z,
                        z_inf_norm,
                    )?;
                    Ok(FusedQuotientRows {
                        b_cyclic: b_rows.b_cyclic,
                        a_quotients,
                    })
                });
            }
            let rows = fused_split_eq_quotients_prover_bounds(
                negacyclic_ntt,
                cyclic_ntt,
                plan.n_b,
                plan.n_a,
                t_hat,
                z,
                z_inf_norm,
                plan.log_basis_outer,
            )?;
            Ok(rows)
        })
    })
}

pub(crate) fn relation_b_a_rows<F: Field + CanonicalEncoding, const D: usize>(
    backend: &CpuBackend,
    prepared: &CpuPreparedSetup<F>,
    t_hat: &[[i8; D]],
    z: &[[i32; D]],
    z_inf_norm: u32,
    plan: RingSwitchRelationPlan,
) -> Result<FusedQuotientRows<F, D>, AkitaError> {
    let extent = fused_quotient_matrix_extent(plan.n_b, t_hat.len(), plan.n_a, z.len())?;
    if extent == 0 {
        return Ok(FusedQuotientRows {
            b_cyclic: Vec::new(),
            a_quotients: Vec::new(),
        });
    }
    if backend.ntt_operation_uses_cache(NttOperationCluster::RingSwitch, extent) {
        return cached_b_a_rows(prepared, t_hat, z, z_inf_norm, plan);
    }
    let view = prepared
        .expanded
        .shared_matrix()
        .ring_view::<D>(1, extent)?;
    fused_split_eq_quotients_streamed_prover_bounds(
        view.as_slice(),
        plan.n_b,
        plan.n_a,
        t_hat,
        z,
        z_inf_norm,
        plan.log_basis_outer,
    )
}

impl<F, const D: usize> RingSwitchRelationKernel<RingSwitchRelationView<'_, D>, F, D> for CpuBackend
where
    F: Field + CanonicalEncoding,
{
    fn relation_rows(
        &self,
        prepared: &Self::PreparedSetup,
        source: RingSwitchRelationView<'_, D>,
        plan: RingSwitchRelationPlan,
    ) -> Result<RingSwitchRelationRows<F, D>, AkitaError>
    where
        F: Field,
    {
        validate_role_shape("D", plan.n_d, source.e_hat.len())?;
        validate_role_shape("B", plan.n_b, source.t_hat.len())?;
        if plan.n_a != 0 {
            return Err(AkitaError::InvalidInput(
                "A-side rows require an accepted fold handle".into(),
            ));
        }
        let d_extent = digit_relation_matrix_extent(plan.n_d, source.e_hat.len())?;
        let b_a_extent = fused_quotient_matrix_extent(plan.n_b, source.t_hat.len(), 0, 0)?;
        let stream_extent = d_extent.max(b_a_extent);
        if !self.ntt_operation_uses_cache(NttOperationCluster::RingSwitch, stream_extent) {
            let view = prepared
                .expanded
                .shared_matrix()
                .ring_view::<D>(1, stream_extent)?;
            let d_rows = if plan.n_d == 0 {
                DigitRelationRows {
                    negacyclic: Vec::new(),
                    cyclic: Vec::new(),
                }
            } else {
                digit_relation_rows_streamed_prover_bounds(
                    view.as_slice(),
                    plan.n_d,
                    source.e_hat,
                    plan.log_basis_open,
                )?
            };
            let rows = relation_b_a_rows(self, prepared, source.t_hat, &[], 0, plan)?;
            return Ok(RingSwitchRelationRows {
                d_negacyclic: d_rows.negacyclic,
                d_cyclic: d_rows.cyclic,
                b_cyclic: rows.b_cyclic,
                a_quotients: rows.a_quotients,
            });
        }

        let d_rows = if plan.n_d == 0 {
            DigitRelationRows {
                negacyclic: Vec::new(),
                cyclic: Vec::new(),
            }
        } else {
            let negacyclic_requirement = NttCacheKey::from_matrix_shape(
                D,
                plan.n_d,
                source.e_hat.len(),
                NttTransformDomain::Negacyclic,
            )?;
            let cyclic_requirement = NttCacheKey::from_matrix_shape(
                D,
                plan.n_d,
                source.e_hat.len(),
                NttTransformDomain::Cyclic,
            )?;
            prepared.with_shared_ntt::<D, _>(negacyclic_requirement, |negacyclic_ntt| {
                prepared.with_shared_ntt::<D, _>(cyclic_requirement, |cyclic_ntt| {
                    digit_relation_rows_cached_prover_bounds(
                        negacyclic_ntt,
                        cyclic_ntt,
                        plan.n_d,
                        source.e_hat,
                        plan.log_basis_open,
                    )
                })
            })?
        };

        let (b_cyclic, a_quotients) = if b_a_extent == 0 {
            (Vec::new(), Vec::new())
        } else {
            let rows = relation_b_a_rows(self, prepared, source.t_hat, &[], 0, plan)?;
            (rows.b_cyclic, rows.a_quotients)
        };
        Ok(RingSwitchRelationRows {
            d_negacyclic: d_rows.negacyclic,
            d_cyclic: d_rows.cyclic,
            b_cyclic,
            a_quotients,
        })
    }
}
