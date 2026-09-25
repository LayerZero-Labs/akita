use super::*;

impl<E: Field> DirectScan<E> {
    /// Row-by-row reference evaluation of this scan, for tests.
    #[cfg(test)]
    pub(crate) fn evaluate_direct_by_rows<F>(
        &self,
        setup: &AkitaExpandedSetup<F>,
        alpha_pows_a: &[E],
        alpha_pows_b: &[E],
        alpha_pows_d: &[E],
        d_a: usize,
    ) -> Result<E, AkitaError>
    where
        F: Field,
        E: ExtField<F> + MulBaseUnreduced<F>,
    {
        let plan = &self.plan;
        let d_d = alpha_pows_d.len();
        let d_b = alpha_pows_b.len();
        let mut acc = E::zero();
        if plan.d_rows() != 0 {
            let d_view =
                setup
                    .shared_matrix
                    .ring_view_dyn(plan.d_rows(), plan.d_physical_cols(), d_d)?;
            for (group_index, group) in plan.groups().iter().enumerate() {
                let (e_eq_slice, _, _) = self
                    .mode
                    .weights(group_index)
                    .ok_or(AkitaError::InvalidProof)?
                    .slices();
                for (row_idx, &row_weight) in plan.d_weights().iter().enumerate() {
                    if row_weight.is_zero() {
                        continue;
                    }
                    let row = d_view.row_flat(row_idx)?;
                    acc += evaluate_weighted_setup_row::<F, E>(
                        row,
                        group.d_col_range().start,
                        e_eq_slice,
                        row_weight,
                        alpha_pows_d,
                    )?;
                }
            }
        }

        for (group_index, group) in plan.groups().iter().enumerate() {
            let direct = self
                .mode
                .weights(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            let (_, _t_eq_slice, z_eq_slice) = direct.slices();
            let a_view = setup
                .shared_matrix
                .ring_view_dyn(group.n_a(), group.z_cols(), d_a)?;
            for (row_idx, &row_weight) in group.a_row_weights().iter().enumerate() {
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
                group.physical_b().physical_rows(),
                group.physical_b().physical_input_width(),
                d_b,
            )?;
            let b_setup_weights = group
                .physical_b()
                .contract_logical_column_weights(&direct.t)?;
            for row_idx in 0..group.physical_b().physical_rows() {
                let row = b_view.row_flat(row_idx)?;
                acc += evaluate_weighted_setup_row::<F, E>(
                    row,
                    0,
                    checked_slice(
                        &b_setup_weights,
                        row_idx * group.physical_b().physical_input_width(),
                        group.physical_b().physical_input_width(),
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
