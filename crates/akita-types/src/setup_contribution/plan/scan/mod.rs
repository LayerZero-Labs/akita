mod group;

use super::*;
use akita_algebra::ring::eval_ring_at_pows_fast;

impl<E: FieldCore> SetupContributionPlan<E> {
    pub fn evaluate_direct<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
    ) -> Result<E, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let geometry = self.projection_geometry;
        geometry.validate_alpha_power_lengths(
            alpha_pows_a.len(),
            alpha_pows_b.len(),
            alpha_pows_d.len(),
        )?;
        let base_d = geometry.base_ring_dim();
        let base_pows = alpha_pows_d.get(..base_d).ok_or(AkitaError::InvalidProof)?;
        let a_projection = role_projection(alpha_pows_a, base_pows, geometry.a_ratio())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "A alpha powers do not decompose over base dimension".into(),
                )
            })?;
        let b_projection = role_projection(alpha_pows_b, base_pows, geometry.b_ratio())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "B alpha powers do not decompose over base dimension".into(),
                )
            })?;
        let d_projection = role_projection(alpha_pows_d, base_pows, geometry.d_ratio())
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "D alpha powers do not decompose over base dimension".into(),
                )
            })?;

        dispatch_for_field!(
            ProtocolDispatchSlot::Role(RingRole::Opening),
            F,
            base_d,
            |BASE_D| {
                self.evaluate_role_dims_direct_typed::<F, BASE_D>(
                    setup,
                    base_pows,
                    &a_projection,
                    &b_projection,
                    &d_projection,
                )
            }
        )
    }

    fn evaluate_role_dims_direct_typed<F, const BASE_D: usize>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        base_pows: &[E],
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let required = self.projection_geometry.required();
        let fused_groups = self.groups.len() > 1;
        let logical_group_rings = self
            .groups
            .iter()
            .fold(0usize, |sum, group| sum.saturating_add(group.required));
        let physical_ring_evaluations = if fused_groups {
            self.projection_geometry.required()
        } else {
            logical_group_rings
        };
        let jobs = if fused_groups {
            self.projection_geometry
                .required()
                .div_ceil(super::segments::SETUP_SCAN_JOB_RINGS)
        } else {
            self.groups
                .iter()
                .map(|group| group.segments.len())
                .sum::<usize>()
        };
        let _span = tracing::info_span!(
            "setup_contribution_scan",
            required,
            groups = self.groups.len(),
            logical_group_rings,
            physical_ring_evaluations,
            jobs,
            fused_groups,
            base_d = BASE_D,
            a_ratio = self.projection_geometry.a_ratio(),
            b_ratio = self.projection_geometry.b_ratio(),
            d_ratio = self.projection_geometry.d_ratio()
        )
        .entered();
        if base_pows.len() != BASE_D {
            return Err(AkitaError::InvalidSize {
                expected: BASE_D,
                actual: base_pows.len(),
            });
        }
        let setup_len = setup.shared_matrix().total_ring_elements_at::<BASE_D>()?;
        if required > setup_len {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        let setup_view = setup.shared_matrix().ring_view::<BASE_D>(1, setup_len)?;
        if fused_groups {
            return self.evaluate_groups_fused::<F, BASE_D>(
                &setup_view,
                base_pows,
                a_projection,
                b_projection,
                d_projection,
            );
        }
        let mut acc = E::zero();
        for group in &self.groups {
            acc += group.evaluate_base_ring_direct::<F, BASE_D>(
                &setup_view,
                base_pows,
                &self.d_weights,
                a_projection,
                b_projection,
                d_projection,
                self.d_rows,
                self.d_physical_cols,
            )?;
        }
        Ok(acc)
    }

    fn evaluate_groups_fused<F, const BASE_D: usize>(
        &self,
        setup_view: &RingMatrixView<'_, F, BASE_D>,
        base_pows: &[E],
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let setup_flat = setup_view.as_slice();
        let required = self.projection_geometry.required();
        if self.d_weights.len() != self.d_rows {
            return Err(AkitaError::InvalidSetup(
                "cached setup scan geometry is malformed".into(),
            ));
        }
        let job_rings = super::segments::SETUP_SCAN_JOB_RINGS;
        let num_jobs = required.div_ceil(job_rings);
        cfg_try_fold_reduce!(
            0..num_jobs,
            E::zero,
            |acc, job| {
                let lo = job.checked_mul(job_rings).ok_or(AkitaError::InvalidProof)?;
                let hi = lo.saturating_add(job_rings).min(required);
                let setup = setup_flat.get(lo..hi).ok_or(AkitaError::InvalidProof)?;
                let mut weights = vec![E::zero(); setup.len()];
                for group in &self.groups {
                    let first = group.segments.partition_point(|segment| segment.hi <= lo);
                    for segment in group.segments.iter().skip(first) {
                        if segment.lo >= hi {
                            break;
                        }
                        let overlap = segment.lo.max(lo)..segment.hi.min(hi);
                        let weight_start = overlap
                            .start
                            .checked_sub(lo)
                            .ok_or(AkitaError::InvalidProof)?;
                        dispatch_segment_roles!(segment, Ok(()), |HAS_D, HAS_B, HAS_A| {
                            for_each_base_ring_segment_weight_typed::<E, HAS_D, HAS_B, HAS_A>(
                                overlap,
                                segment,
                                &group.e_eq_slice,
                                &group.t_eq_slice,
                                &group.z_eq_slice,
                                d_projection,
                                b_projection,
                                a_projection,
                                |offset, weight| {
                                    let slot = weight_start
                                        .checked_add(offset)
                                        .and_then(|index| weights.get_mut(index))
                                        .ok_or(AkitaError::InvalidProof)?;
                                    *slot += weight;
                                    Ok(())
                                },
                            )
                        })?;
                    }
                }
                let mut term = E::zero();
                for (ring, weight) in setup.iter().zip(weights) {
                    if !weight.is_zero() {
                        term += eval_ring_at_pows_fast(ring, base_pows) * weight;
                    }
                }
                Ok(acc + term)
            },
            |lhs, rhs| Ok(lhs + rhs)
        )
    }
}
