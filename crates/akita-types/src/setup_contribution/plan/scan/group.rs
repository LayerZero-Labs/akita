use super::super::*;

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_base_ring_direct<F, const BASE_D: usize>(
        &self,
        setup_view: &RingMatrixView<'_, F, BASE_D>,
        base_pows: &[E],
        d_weights: &[E],
        a_projection: &RoleProjection<E>,
        b_projection: &RoleProjection<E>,
        d_projection: &RoleProjection<E>,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let setup_flat = setup_view.as_slice();
        if self.required > setup_flat.len() {
            return Err(AkitaError::InvalidSetup(
                "shared matrix is too small for selected verifier layout".into(),
            ));
        }
        if d_weights.len() != d_rows
            || self.d_col_range.end > d_physical_cols
            || self.d_col_range.len() != self.e_eq_slice.len()
        {
            return Err(AkitaError::InvalidSetup(
                "cached setup scan geometry is malformed".into(),
            ));
        }

        cfg_try_fold_reduce!(
            self.segments.as_ref(),
            E::zero,
            |acc, segment| {
                dispatch_segment_roles!(segment, Ok(acc), |HAS_D, HAS_B, HAS_A| {
                    base_ring_segment_inner_sum_typed::<F, E, BASE_D, HAS_D, HAS_B, HAS_A>(
                        segment.lo..segment.hi,
                        setup_flat,
                        base_pows,
                        segment,
                        &self.e_eq_slice,
                        &self.t_eq_slice,
                        &self.z_eq_slice,
                        d_projection,
                        b_projection,
                        a_projection,
                    )
                    .map(|term| acc + term)
                })
            },
            |lhs, rhs| Ok(lhs + rhs)
        )
    }
}
