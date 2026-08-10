use super::*;

#[cfg(test)]
impl<E: FieldCore> SetupContributionPlan<E> {
    pub(crate) fn evaluate_direct_by_rows<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_a: usize,
    ) -> Result<E, AkitaError>
    where
        F: FieldCore,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let d_d = alpha_pows_d.len();
        let d_b = alpha_pows_b.len();
        let mut acc = E::zero();
        if self.d_rows != 0 {
            let d_view =
                setup
                    .shared_matrix
                    .ring_view_dyn(self.d_rows, self.d_physical_cols, d_d)?;
            for group in &self.groups {
                let (e_eq_slice, _, _) = group.require_column_eq_slices()?;
                for (row_idx, &row_weight) in self.d_weights.iter().enumerate() {
                    if row_weight.is_zero() {
                        continue;
                    }
                    let row = d_view.row_flat(row_idx)?;
                    acc += evaluate_weighted_setup_row::<F, E>(
                        row,
                        group.d_col_range.start,
                        e_eq_slice,
                        row_weight,
                        alpha_pows_d,
                    )?;
                }
            }
        }

        for group in &self.groups {
            let (_, _t_eq_slice, z_eq_slice) = group.require_column_eq_slices()?;
            let a_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_a, group.z_cols, d_a)?;
            for (row_idx, &row_weight) in group.a_row_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = a_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    z_eq_slice,
                    row_weight,
                    alpha_pows_a,
                )?;
            }

            let b_view = setup.shared_matrix.ring_view_dyn(
                group.physical_n_b,
                group.physical_t_cols,
                d_b,
            )?;
            let b_setup_weights = &group
                .direct_scan_weights
                .as_ref()
                .ok_or(AkitaError::InvalidProof)?
                .b_setup;
            for row_idx in 0..group.physical_n_b {
                let row = b_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    checked_slice(
                        b_setup_weights,
                        row_idx * group.physical_t_cols,
                        group.physical_t_cols,
                        "physical B setup weights",
                    )?,
                    E::one(),
                    alpha_pows_b,
                )?;
            }
        }

        Ok(acc)
    }
}
