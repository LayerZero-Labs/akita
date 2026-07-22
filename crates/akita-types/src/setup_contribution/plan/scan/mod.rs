use super::*;
use akita_algebra::ring::eval_ring_at_pows_fast;

/// Target scan-job size. At fp128/D64 this is 2 MiB of contiguous setup data,
/// large enough to amortize scheduling while exposing hundreds of root jobs.
const SETUP_SCAN_JOB_RINGS: usize = 2048;

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
        let job_rings = SETUP_SCAN_JOB_RINGS;
        let required = self.projection_geometry.required();
        let jobs = required.div_ceil(job_rings);
        let _span = tracing::info_span!(
            "setup_contribution_scan",
            required = self.projection_geometry.required(),
            groups = self.groups.len(),
            jobs,
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
        let setup_flat = setup_view.as_slice();
        cfg_try_fold_reduce!(
            0..jobs,
            E::zero,
            |acc, job| {
                let lo = job.checked_mul(job_rings).ok_or(AkitaError::InvalidProof)?;
                let hi = lo.saturating_add(job_rings).min(required);
                let mut term = E::zero();
                for setup_idx in lo..hi {
                    let ring = setup_flat.get(setup_idx).ok_or(AkitaError::InvalidProof)?;
                    let weight = self.base_setup_weight_at(
                        setup_idx,
                        a_projection,
                        b_projection,
                        d_projection,
                    )?;
                    if !weight.is_zero() {
                        term += eval_ring_at_pows_fast(ring, base_pows) * weight;
                    }
                }
                Ok(acc + term)
            },
            |lhs, rhs| Ok(lhs + rhs)
        )
    }

    fn base_setup_weight_at(
        &self,
        setup_idx: usize,
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
    ) -> Result<E, AkitaError> {
        let mut weight = E::zero();
        if self.d_physical_cols != 0 {
            let (d_idx, scale) = projected_logical_index(setup_idx, d_projection)?;
            let d_footprint = self
                .d_rows
                .checked_mul(self.d_physical_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
            if d_idx < d_footprint {
                let d_col = d_idx % self.d_physical_cols;
                let d_row = d_idx / self.d_physical_cols;
                for group in &self.groups {
                    if group.d_col_range.contains(&d_col) {
                        let local_col = d_col
                            .checked_sub(group.d_col_range.start)
                            .ok_or(AkitaError::InvalidProof)?;
                        let row_weight = *self
                            .eq_tau1
                            .get(self.d_row_start + d_row)
                            .ok_or(AkitaError::InvalidProof)?;
                        weight += scale * row_weight * group.d_eq_at(local_col, &self.eq_window)?;
                    }
                }
            }
        }
        let (b_idx, b_scale) = projected_logical_index(setup_idx, b_projection)?;
        for group in &self.groups {
            let b_footprint = group
                .n_b
                .checked_mul(group.t_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
            if b_idx < b_footprint {
                let b_col = b_idx % group.t_cols;
                let b_row = b_idx / group.t_cols;
                let row_weight = *self
                    .eq_tau1
                    .get(group.b_row_start + b_row)
                    .ok_or(AkitaError::InvalidProof)?;
                weight += b_scale * row_weight * group.b_eq_at(b_col, &self.eq_window)?;
            }

            let (a_idx, a_scale) = projected_logical_index(setup_idx, a_projection)?;
            let a_footprint = group
                .n_a
                .checked_mul(group.z_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
            if a_idx < a_footprint {
                let a_col = a_idx % group.z_cols;
                let a_row = a_idx / group.z_cols;
                let row_weight = *self
                    .eq_tau1
                    .get(group.a_row_start + a_row)
                    .ok_or(AkitaError::InvalidProof)?;
                weight += a_scale
                    * row_weight
                    * group.a_eq_at(a_col, &self.eq_window, &self.fold_gadget)?;
            }
        }
        Ok(weight)
    }
}

fn projected_logical_index<E: FieldCore>(
    setup_idx: usize,
    projection: &RoleProjection<E>,
) -> Result<(usize, E), AkitaError> {
    if projection.is_identity() {
        return Ok((setup_idx, E::one()));
    }
    let scale = *projection
        .scales
        .get(setup_idx & projection.mask)
        .ok_or(AkitaError::InvalidProof)?;
    Ok((setup_idx >> projection.shift, scale))
}
