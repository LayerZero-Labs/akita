use super::*;
use akita_algebra::offset_eq::OffsetEqWindow;
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
        let scales = self.projection_scales(alpha);
        // Share one bounded equality window across every packed setup position
        // instead of recomputing a full-width equality product per index, and
        // evaluate the (independent) per-position terms in parallel. This is the
        // fallback used at the root level when groups do not share a fold gadget
        // (e.g. multi-group with precommitted singletons), and it was the
        // dominant recursive-mode verifier cost there.
        let _span = tracing::info_span!("stage3_setup_index_weight_mle").entered();
        let eq_window = OffsetEqWindow::new(rho_setup_idx)?;
        let terms = cfg_into_iter!(0..geometry.required())
            .map(|setup_idx| -> Result<E, AkitaError> {
                Ok(eq_window.eval(setup_idx) * self.setup_index_weight_at(setup_idx, &scales)?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(terms.into_iter().fold(E::zero(), |acc, value| acc + value))
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
                    weight += scales[2][setup_idx % geometry.d_ratio()]
                        * self.d_weights[d_row]
                        * group.e_eq_slice[d_col - group.d_col_range.start];
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
                    * group.b_weights[b_row]
                    * group.t_eq_slice[b_col];
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
                    * group.a_row_weights[a_row]
                    * group.z_eq_slice[a_col];
            }
        }
        Ok(weight)
    }
}
