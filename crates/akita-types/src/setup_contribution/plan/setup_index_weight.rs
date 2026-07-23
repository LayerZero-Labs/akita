use super::*;
use akita_algebra::offset_eq::eval_compact_pair_eq;
use akita_algebra::ring::scalar_powers;

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
    pub fn materialize_setup_index_weights(&self, alpha: E) -> Result<Vec<E>, AkitaError> {
        let scales = self.projection_scales(alpha);
        (0..self.required())
            .map(|setup_idx| self.setup_index_weight_at(setup_idx, &scales))
            .collect()
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
    pub fn evaluate_setup_index_weight_mle(
        &self,
        rho_setup_idx: &[E],
        alpha: E,
    ) -> Result<E, AkitaError> {
        let geometry = self.projection_geometry;
        let setup_idx_bits = geometry.setup_index_len().trailing_zeros() as usize;
        if rho_setup_idx.len() != setup_idx_bits {
            return Err(AkitaError::InvalidSize {
                expected: setup_idx_bits,
                actual: rho_setup_idx.len(),
            });
        }
        let _span = tracing::info_span!("stage3_setup_index_weight_mle").entered();
        let scales = self.projection_scales(alpha);
        let mut acc = E::zero();
        for group in &self.groups {
            acc += self.evaluate_d_spans_at_point(group, rho_setup_idx, &scales[2])?;
            acc += self.evaluate_b_spans_at_point(group, rho_setup_idx, &scales[1])?;
            acc += self.evaluate_a_spans_at_point(group, rho_setup_idx, &scales[0])?;
        }
        Ok(acc)
    }

    fn projection_scales(&self, alpha: E) -> [Vec<E>; 3] {
        let geometry = self.projection_geometry;
        let role_scales = |role_dim: usize| {
            scalar_powers(alpha, role_dim)
                .chunks(geometry.base_ring_dim())
                .map(|chunk| chunk[0])
                .collect()
        };
        [
            role_scales(geometry.role_dims().d_a()),
            role_scales(geometry.role_dims().d_b()),
            role_scales(geometry.role_dims().d_d()),
        ]
    }

    fn setup_index_weight_at(
        &self,
        setup_idx: usize,
        scales: &[Vec<E>; 3],
    ) -> Result<E, AkitaError> {
        let geometry = self.projection_geometry;
        if setup_idx >= geometry.required() {
            return Err(AkitaError::InvalidSize {
                expected: geometry.required(),
                actual: setup_idx,
            });
        }
        let mut weight = E::zero();
        for group in &self.groups {
            let d_idx = setup_idx / geometry.d_ratio();
            let d_footprint = self
                .d_rows
                .checked_mul(self.d_physical_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
            if d_idx < d_footprint {
                let d_col = d_idx % self.d_physical_cols;
                let d_row = d_idx / self.d_physical_cols;
                if group.d_col_range.contains(&d_col) {
                    let local_col = d_col
                        .checked_sub(group.d_col_range.start)
                        .ok_or(AkitaError::InvalidProof)?;
                    weight += scales[2][setup_idx % geometry.d_ratio()]
                        * self.eq_tau1[self.d_row_start + d_row]
                        * group.d_eq_at(local_col, &self.eq_window)?;
                }
            }

            let b_idx = setup_idx / geometry.b_ratio();
            let b_footprint = group
                .n_b
                .checked_mul(group.t_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
            if b_idx < b_footprint {
                let b_col = b_idx % group.t_cols;
                let b_row = b_idx / group.t_cols;
                weight += scales[1][setup_idx % geometry.b_ratio()]
                    * self.eq_tau1[group.b_row_start + b_row]
                    * group.b_eq_at(b_col, &self.eq_window)?;
            }

            let a_idx = setup_idx / geometry.a_ratio();
            let a_footprint = group
                .n_a
                .checked_mul(group.z_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
            if a_idx < a_footprint {
                let a_col = a_idx % group.z_cols;
                let a_row = a_idx / group.z_cols;
                weight += scales[0][setup_idx % geometry.a_ratio()]
                    * self.eq_tau1[group.a_row_start + a_row]
                    * group.a_eq_at(a_col, &self.eq_window, &self.fold_gadget)?;
            }
        }
        Ok(weight)
    }

    fn evaluate_d_spans_at_point(
        &self,
        group: &SetupContributionGroupPlan<E>,
        rho_setup_idx: &[E],
        scales: &[E],
    ) -> Result<E, AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(E::zero());
        }
        let mut acc = E::zero();
        for &(setup_start, witness_start, len) in &group.d_spans {
            let setup_col = group
                .d_col_range
                .start
                .checked_add(setup_start)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D address overflow".into()))?;
            for row in 0..self.d_rows {
                let row_weight = *self
                    .eq_tau1
                    .get(self.d_row_start + row)
                    .ok_or(AkitaError::InvalidProof)?;
                for (lane, &scale) in scales.iter().enumerate() {
                    let setup_index = projected_setup_offset(
                        self.projection_geometry.d_ratio(),
                        self.d_physical_cols,
                        row,
                        setup_col,
                        lane,
                    )?;
                    let pair = eval_compact_pair_eq(
                        rho_setup_idx,
                        setup_index,
                        self.projection_geometry.d_ratio(),
                        &self.x_challenges,
                        witness_start,
                        1,
                        len,
                    )?;
                    acc += row_weight * scale * pair;
                }
            }
        }
        Ok(acc)
    }

    fn evaluate_b_spans_at_point(
        &self,
        group: &SetupContributionGroupPlan<E>,
        rho_setup_idx: &[E],
        scales: &[E],
    ) -> Result<E, AkitaError> {
        if group.n_b == 0 {
            return Ok(E::zero());
        }
        let mut acc = E::zero();
        for &(setup_start, witness_start, len) in &group.b_spans {
            for row in 0..group.n_b {
                let row_weight = *self
                    .eq_tau1
                    .get(group.b_row_start + row)
                    .ok_or(AkitaError::InvalidProof)?;
                for (lane, &scale) in scales.iter().enumerate() {
                    let setup_index = projected_setup_offset(
                        self.projection_geometry.b_ratio(),
                        group.t_cols,
                        row,
                        setup_start,
                        lane,
                    )?;
                    let pair = eval_compact_pair_eq(
                        rho_setup_idx,
                        setup_index,
                        self.projection_geometry.b_ratio(),
                        &self.x_challenges,
                        witness_start,
                        1,
                        len,
                    )?;
                    acc += row_weight * scale * pair;
                }
            }
        }
        Ok(acc)
    }

    fn evaluate_a_spans_at_point(
        &self,
        group: &SetupContributionGroupPlan<E>,
        rho_setup_idx: &[E],
        scales: &[E],
    ) -> Result<E, AkitaError> {
        if group.n_a == 0 {
            return Ok(E::zero());
        }
        let mut acc = E::zero();
        for &(setup_start, witness_start, len, fold_digit) in &group.a_spans {
            let fold = *self
                .fold_gadget
                .get(fold_digit)
                .ok_or(AkitaError::InvalidProof)?;
            for row in 0..group.n_a {
                let row_weight = *self
                    .eq_tau1
                    .get(group.a_row_start + row)
                    .ok_or(AkitaError::InvalidProof)?;
                for (lane, &scale) in scales.iter().enumerate() {
                    let setup_index = projected_setup_offset(
                        self.projection_geometry.a_ratio(),
                        group.z_cols,
                        row,
                        setup_start,
                        lane,
                    )?;
                    let pair = eval_compact_pair_eq(
                        rho_setup_idx,
                        setup_index,
                        self.projection_geometry.a_ratio(),
                        &self.x_challenges,
                        witness_start,
                        group.depth_fold,
                        len,
                    )?;
                    acc -= row_weight * scale * fold * pair;
                }
            }
        }
        Ok(acc)
    }
}

fn projected_setup_offset(
    ratio: usize,
    width: usize,
    row: usize,
    column: usize,
    lane: usize,
) -> Result<usize, AkitaError> {
    if column >= width || lane >= ratio {
        return Err(AkitaError::InvalidSetup(
            "setup projected address out of range".into(),
        ));
    }
    let logical = width
        .checked_mul(row)
        .and_then(|base| base.checked_add(column))
        .ok_or_else(|| AkitaError::InvalidSetup("setup role index overflow".into()))?;
    ratio
        .checked_mul(logical)
        .and_then(|base| base.checked_add(lane))
        .ok_or_else(|| AkitaError::InvalidSetup("setup base index overflow".into()))
}
