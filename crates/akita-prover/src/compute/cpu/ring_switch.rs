use super::*;

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
        // The root-level relation spans nearly the whole matrix but reads
        // each element exactly once per prove — stream its transforms from
        // the field form instead of materializing a matrix-scale NTT cache
        // for one pass. Small (deeper-level) extents keep the cached path,
        // which is shared with the per-level digit-row products.
        let stream_extent = plan
            .n_d
            .saturating_mul(plan.e_hat.len())
            .max(plan.n_b.saturating_mul(plan.t_hat.len()))
            .max(plan.n_a.saturating_mul(plan.z_segment.len()));
        if stream_extent > NTT_STREAM_THRESHOLD_RING_ELEMENTS {
            let view = prepared
                .expanded
                .shared_matrix()
                .ring_view::<D>(1, stream_extent)?;
            let source = StreamedASource::new(view.as_slice());
            let streamed = fused_split_eq_quotients_streamed_prover_bounds(
                &source,
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
        let stream_extent = plan.n_a.saturating_mul(plan.z_segment.len());
        if stream_extent > NTT_STREAM_THRESHOLD_RING_ELEMENTS {
            let view = prepared
                .expanded
                .shared_matrix()
                .ring_view::<D>(1, stream_extent)?;
            let source = StreamedASource::new(view.as_slice());
            let streamed = fused_split_eq_quotients_streamed_prover_bounds(
                &source,
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
            if let Some((_d_cyclic, _b_cyclic, a_quotients)) = streamed {
                return Ok(a_quotients);
            }
        }
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
