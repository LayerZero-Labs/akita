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
                for (row_idx, &row_weight) in self.d_weights.iter().enumerate() {
                    if row_weight.is_zero() {
                        continue;
                    }
                    let row = d_view.row_flat(row_idx)?;
                    acc += evaluate_weighted_setup_row::<F, E>(
                        row,
                        group.d_col_range.start,
                        &group.e_eq_slice,
                        row_weight,
                        alpha_pows_d,
                    )?;
                }
            }
        }

        for group in &self.groups {
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
                    &group.z_eq_slice,
                    row_weight,
                    alpha_pows_a,
                )?;
            }

            let b_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_b, group.t_cols, d_b)?;
            for (row_idx, &row_weight) in group.b_weights.iter().enumerate() {
                if row_weight.is_zero() {
                    continue;
                }
                let row = b_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    &group.t_eq_slice,
                    row_weight,
                    alpha_pows_b,
                )?;
            }
        }

        Ok(acc)
    }
}
