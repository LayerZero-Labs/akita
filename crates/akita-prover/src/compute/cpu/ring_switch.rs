use super::{CpuBackend, CpuPreparedSetup};
use crate::backend::RingSwitchRelationView;
use crate::compute::kernels::RingSwitchRelationKernel;
use crate::compute::operation_plans::RingSwitchRelationPlan;
use crate::compute::plans::RingSwitchRelationRows;
use crate::compute::requirements::NttOperationCluster;
use crate::kernels::linear::{
    centered_quotient_rows_with_i16_tail, digit_relation_matrix_extent,
    digit_relation_rows_cached_prover_bounds, digit_relation_rows_streamed_prover_bounds,
    fused_quotient_matrix_extent, fused_split_eq_quotients_prover_bounds,
    fused_split_eq_quotients_streamed_prover_bounds, DigitRelationRows, FusedBARows,
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

fn validate_role_shape(role: &str, rows: usize, width: usize) -> Result<(), AkitaError> {
    if rows != 0 && width == 0 {
        return Err(AkitaError::InvalidInput(format!(
            "active ring-switch {role} role must have a nonzero source width"
        )));
    }
    Ok(())
}

struct BARelationRows<F: FieldCore, const D: usize> {
    cyclic: Vec<CyclotomicRing<F, D>>,
    quotients: Vec<CyclotomicRing<F, D>>,
}

fn cached_b_a_rows<F: FieldCore + CanonicalField + HalvingField, const D: usize>(
    prepared: &CpuPreparedSetup<F>,
    source: RingSwitchRelationView<'_, D>,
    plan: RingSwitchRelationPlan,
) -> Result<BARelationRows<F, D>, AkitaError> {
    let mut cyclic_requirement: Option<NttCacheKey> = None;
    for (rows, width) in [
        (plan.n_b, source.t_hat.len()),
        (plan.n_a, source.z_segment.len()),
    ] {
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
                source.t_hat,
                &[],
                0,
                plan.log_basis_outer,
            )?;
            return Ok(BARelationRows {
                cyclic: rows.b_cyclic,
                quotients: rows.a_quotients,
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
                    let b_rows = fused_split_eq_quotients_prover_bounds(
                        negacyclic_ntt,
                        cyclic_ntt,
                        plan.n_b,
                        0,
                        source.t_hat,
                        &[],
                        0,
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
                    Ok(BARelationRows {
                        cyclic: b_rows.b_cyclic,
                        quotients: a_quotients,
                    })
                });
            }
            let rows = fused_split_eq_quotients_prover_bounds(
                negacyclic_ntt,
                cyclic_ntt,
                plan.n_b,
                plan.n_a,
                source.t_hat,
                source.z_segment,
                source.z_folded_centered_inf_norm,
                plan.log_basis_outer,
            )?;
            Ok(BARelationRows {
                cyclic: rows.b_cyclic,
                quotients: rows.a_quotients,
            })
        })
    })
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
        validate_role_shape("D", plan.n_d, source.e_hat.len())?;
        validate_role_shape("B", plan.n_b, source.t_hat.len())?;
        validate_role_shape("A", plan.n_a, source.z_segment.len())?;
        let d_extent = digit_relation_matrix_extent(plan.n_d, source.e_hat.len())?;
        let b_a_extent = fused_quotient_matrix_extent(
            plan.n_b,
            source.t_hat.len(),
            plan.n_a,
            source.z_segment.len(),
        )?;
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
            let rows = if b_a_extent == 0 {
                FusedBARows {
                    b_cyclic: Vec::new(),
                    a_quotients: Vec::new(),
                }
            } else {
                fused_split_eq_quotients_streamed_prover_bounds(
                    view.as_slice(),
                    plan.n_b,
                    plan.n_a,
                    source.t_hat,
                    source.z_segment,
                    source.z_folded_centered_inf_norm,
                    plan.log_basis_outer,
                )?
            };
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
            let rows = cached_b_a_rows(prepared, source, plan)?;
            (rows.cyclic, rows.quotients)
        };
        Ok(RingSwitchRelationRows {
            d_negacyclic: d_rows.negacyclic,
            d_cyclic: d_rows.cyclic,
            b_cyclic,
            a_quotients,
        })
    }
}
