use super::*;
use akita_algebra::offset_eq::eval_offset_eq_interval;

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Dense setup weight vector `setup_index_weight[i]`.
    ///
    /// For one commitment group, a packed setup position `i` receives
    ///
    /// ```text
    /// 1_D(i) * tau_D[row_D(i)] * E_col[col_D(i)]
    /// + 1_B(i) * tau_B[row_B(i)] * T_col[col_B(i)]
    /// + 1_A(i) * tau_A[row_A(i)] * Z_col[col_A(i)].
    /// ```
    ///
    /// Here `Z_col[blk, dc]` is the column weight for
    /// `A * G_fold * z_hat`. If several groups write to the same packed setup
    /// position, their scalar weights are added.
    pub fn materialize_setup_index_weights(&self) -> Result<Vec<E>, AkitaError> {
        let required = self.required()?;
        let mut setup_index_weight = vec![E::zero(); required];
        for group in &self.groups {
            let (_, segments) = group.packed_segments(self.d_rows, self.d_physical_cols)?;
            let segment_values = cfg_into_iter!(0..segments.len())
                .map(|idx| {
                    let segment = &segments[idx];
                    let values = (segment.lo..segment.hi)
                        .map(|setup_idx| {
                            segment.weight_at(
                                setup_idx,
                                &group.e_eq_slice,
                                &group.t_eq_slice,
                                &group.z_eq_slice,
                            )
                        })
                        .collect::<Vec<_>>();
                    (segment.lo, values)
                })
                .collect::<Vec<_>>();
            for (lo, values) in segment_values {
                for (offset, value) in values.into_iter().enumerate() {
                    setup_index_weight[lo + offset] += value;
                }
            }
        }
        Ok(setup_index_weight)
    }

    /// Evaluate the multilinear extension of `setup_index_weight` at
    /// `rho_setup_idx`.
    ///
    /// This computes
    ///
    /// ```text
    /// sum_i eq(rho_setup_idx, i) * setup_index_weight[i]
    /// ```
    ///
    /// without building the full setup-index equality table and without
    /// materializing `setup_index_weight`.
    pub fn evaluate_setup_index_weight_mle(&self, rho_setup_idx: &[E]) -> Result<E, AkitaError> {
        let required = self.required()?;
        let setup_idx_len = required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup weight index length overflow".into()))?;
        let setup_idx_bits = setup_idx_len.trailing_zeros() as usize;
        if rho_setup_idx.len() != setup_idx_bits {
            return Err(AkitaError::InvalidSize {
                expected: setup_idx_bits,
                actual: rho_setup_idx.len(),
            });
        }
        let mut acc = E::zero();
        for group in &self.groups {
            if let Some(value) = group.evaluate_setup_index_weight_mle_factored(
                rho_setup_idx,
                self.d_rows,
                self.d_physical_cols,
            )? {
                acc += value;
            } else {
                let (_, segments) = group.packed_segments(self.d_rows, self.d_physical_cols)?;
                let group_sum = cfg_fold_reduce!(
                    0..segments.len(),
                    || Ok::<E, AkitaError>(E::zero()),
                    |acc, idx| {
                        let acc = acc?;
                        let segment = &segments[idx];
                        let value = setup_index_weight_segment_mle(
                            rho_setup_idx,
                            segment,
                            &group.e_eq_slice,
                            &group.t_eq_slice,
                            &group.z_eq_slice,
                        )?;
                        Ok::<E, AkitaError>(acc + value)
                    },
                    |lhs, rhs| match (lhs, rhs) {
                        (Ok(lhs), Ok(rhs)) => Ok(lhs + rhs),
                        (Err(err), _) | (_, Err(err)) => Err(err),
                    }
                )?;
                acc += group_sum;
            }
        }
        Ok(acc)
    }
}

impl<E: FieldCore> SetupContributionGroupPlan<E> {
    fn evaluate_setup_index_weight_mle_factored(
        &self,
        rho_setup_idx: &[E],
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<Option<E>, AkitaError> {
        if !role_width_is_factored(d_rows, d_physical_cols)
            || !role_width_is_factored(self.n_b, self.t_cols)
            || !role_width_is_factored(self.n_a, self.z_cols)
        {
            return Ok(None);
        }

        let mut acc = E::zero();
        acc += role_setup_index_weight_mle(
            rho_setup_idx,
            &self.d_weights,
            d_rows,
            d_physical_cols,
            &self.e_eq_slice,
            self.e_col_offset,
        )?;
        acc += role_setup_index_weight_mle(
            rho_setup_idx,
            &self.b_weights,
            self.n_b,
            self.t_cols,
            &self.t_eq_slice,
            0,
        )?;
        acc += role_setup_index_weight_mle(
            rho_setup_idx,
            &self.a_row_weights,
            self.n_a,
            self.z_cols,
            &self.z_eq_slice,
            0,
        )?;
        Ok(Some(acc))
    }
}

fn role_width_is_factored(rows: usize, width: usize) -> bool {
    rows == 0 || width.is_power_of_two()
}

fn role_setup_index_weight_mle<E: FieldCore>(
    rho_setup_idx: &[E],
    row_weights: &[E],
    rows: usize,
    width: usize,
    col_weights: &[E],
    col_offset: usize,
) -> Result<E, AkitaError> {
    if rows == 0 || width == 0 || col_weights.is_empty() {
        return Ok(E::zero());
    }
    if !width.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(
            "factored setup-index weight role has non-power-of-two width".into(),
        ));
    }
    if row_weights.len() != rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: row_weights.len(),
        });
    }
    let width_bits = width.trailing_zeros() as usize;
    if width_bits > rho_setup_idx.len() {
        return Err(AkitaError::InvalidProof);
    }
    let col_point = &rho_setup_idx[..width_bits];
    let row_point = &rho_setup_idx[width_bits..];
    let row_eval = eval_offset_eq_interval(row_point, 0, E::one(), row_weights)?;
    let col_eval = eval_offset_eq_interval(col_point, col_offset, E::one(), col_weights)?;
    Ok(row_eval * col_eval)
}

fn setup_index_weight_segment_mle<E: FieldCore>(
    rho_setup_idx: &[E],
    segment: &GroupSetupSegment<E>,
    e_eq: &[E],
    t_eq: &[E],
    z_eq: &[E],
) -> Result<E, AkitaError> {
    let mut acc = E::zero();
    if segment.has_d {
        acc += eval_offset_eq_interval(
            rho_setup_idx,
            segment.lo,
            segment.d_weight,
            segment_slice(e_eq, segment.lo, segment.hi, segment.d_start_abs, "D")?,
        )?;
    }
    if segment.has_b {
        acc += eval_offset_eq_interval(
            rho_setup_idx,
            segment.lo,
            segment.b_weight,
            segment_slice(t_eq, segment.lo, segment.hi, segment.b_start_abs, "B")?,
        )?;
    }
    if segment.has_a {
        acc += eval_offset_eq_interval(
            rho_setup_idx,
            segment.lo,
            segment.a_row_weight,
            segment_slice(z_eq, segment.lo, segment.hi, segment.a_start_abs, "A")?,
        )?;
    }
    Ok(acc)
}

fn segment_slice<'a, E>(
    values: &'a [E],
    lo: usize,
    hi: usize,
    row_start_abs: usize,
    role: &str,
) -> Result<&'a [E], AkitaError> {
    let start = lo.checked_sub(row_start_abs).ok_or_else(|| {
        AkitaError::InvalidSetup(format!("setup {role} segment starts before row"))
    })?;
    let end = hi
        .checked_sub(row_start_abs)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("setup {role} segment ends before row")))?;
    values
        .get(start..end)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("setup {role} segment exceeds row width")))
}
