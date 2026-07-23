mod group;

use super::*;

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
        let required = self
            .groups
            .iter()
            .map(|group| group.required)
            .max()
            .unwrap_or(0);
        let jobs = self
            .groups
            .iter()
            .map(|group| group.segments.len())
            .sum::<usize>();
        let _span = tracing::info_span!(
            "setup_contribution_scan",
            required,
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
        let mut weight = E::zero();
        for group in &self.groups {
            weight += group.evaluate_base_ring_direct::<F, BASE_D>(
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
        Ok(weight)
    }
}
