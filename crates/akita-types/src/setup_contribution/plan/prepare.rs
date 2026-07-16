use super::*;

impl<E: FieldCore> SetupContributionPlan<E> {
    pub fn prepare_static(
        inputs: &SetupContributionPlanInputs<E>,
        layout: &SetupContributionLayout,
    ) -> Result<SetupContributionStatic<E>, AkitaError> {
        let groups = layout.groups();
        let d_rows = match inputs.relation_matrix_row_layout {
            crate::RelationMatrixRowLayout::WithDBlock => inputs.n_d,
            crate::RelationMatrixRowLayout::WithoutDBlock => 0,
        };
        let d_row_start = inputs
            .rows
            .checked_sub(d_rows)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D rows exceed relation rows".into()))?;
        let d_physical_cols = layout.d_physical_cols();
        let d_weights: std::sync::Arc<[E]> = if d_rows == 0 {
            Vec::new().into()
        } else {
            checked_slice(&inputs.eq_tau1, d_row_start, d_rows, "setup D rows")?
                .to_vec()
                .into()
        };
        let static_groups = groups
            .iter()
            .map(|group| {
                let d_col_range = layout.d_col_range(group.group_id)?;
                let t_cols = group
                    .num_claims
                    .checked_mul(group.t_cols_per_vector)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
                let z_cols = group
                    .positions_per_block
                    .checked_mul(group.depth_commit)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                let a_row_weights: std::sync::Arc<[E]> = checked_slice(
                    &inputs.eq_tau1,
                    group.a_row_start,
                    group.n_a,
                    "setup A rows",
                )?
                .to_vec()
                .into();
                let b_weights: std::sync::Arc<[E]> = checked_slice(
                    &inputs.eq_tau1,
                    group.b_row_start,
                    group.n_b,
                    "setup B rows",
                )?
                .to_vec()
                .into();
                let e_cols = group
                    .num_claims
                    .checked_mul(group.live_block_count)
                    .and_then(|cols| cols.checked_mul(group.depth_open))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup E active width overflow".into())
                    })?;
                let (required, segments) = build_packed_segments(
                    d_col_range.start,
                    e_cols,
                    t_cols,
                    z_cols,
                    group.n_a,
                    group.n_b,
                    &a_row_weights,
                    &b_weights,
                    &d_weights,
                    d_rows,
                    d_physical_cols,
                )?;
                Ok(SetupContributionGroupStatic {
                    d_col_range,
                    t_cols,
                    z_cols,
                    n_a: group.n_a,
                    n_b: group.n_b,
                    required,
                    segments: segments.into(),
                    a_row_weights,
                    b_weights,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        Ok(SetupContributionStatic {
            groups: static_groups,
            d_rows,
            d_physical_cols,
            d_weights,
        })
    }

    pub fn finish_plan<F>(
        static_plan: &SetupContributionStatic<E>,
        full_vec_randomness: &[E],
        _eq_low: Option<&[E]>,
        _z_block_low_eq: Option<&[E]>,
        fold_gadget: Option<&[F]>,
        layout: &SetupContributionLayout,
        role_dims: CommitmentRingDims,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
        let groups = layout.groups();
        if static_plan.groups.len() != groups.len() {
            return Err(AkitaError::InvalidSize {
                expected: groups.len(),
                actual: static_plan.groups.len(),
            });
        }
        // Build the bounded equality window once and share it across every E/T/Z
        // column weight. Each canonical column address then costs one bounded
        // low-table lookup plus a short high evaluation instead of a full
        // `O(col_bits+ring_bits)` equality product recomputed per column, which
        // was the dominant verifier setup-plan cost after the digit-innermost
        // cutover (root cause 4). The `_eq_low`/`_z_block_low_eq` parameters are
        // subsumed by this window.
        let eq_window = akita_algebra::offset_eq::OffsetEqWindow::new(full_vec_randomness)?;
        let dynamic_groups = static_plan
            .groups
            .iter()
            .zip(groups)
            .map(|(static_group, group)| {
                let e_eq_slice = {
                    let _span = tracing::info_span!("setup_prepare_e_weights").entered();
                    setup_e_col_weights::<E>(
                        layout.witness_layout(),
                        layout.opening_source_len(),
                        group.group_id,
                        group.live_block_count,
                        group.num_claims,
                        group.depth_open,
                        &eq_window,
                    )?
                };
                let t_eq_slice = {
                    let _span = tracing::info_span!("setup_prepare_t_weights").entered();
                    setup_t_col_weights::<E>(
                        layout.witness_layout(),
                        layout.opening_source_len(),
                        group.group_id,
                        group.live_block_count,
                        group.depth_open,
                        group.n_a,
                        group.t_cols_per_vector,
                        group.num_claims,
                        group.num_claims,
                        0,
                        &eq_window,
                    )?
                };
                let fold_gadget_storage;
                let fold_gadget = if let Some(fold_gadget) = fold_gadget {
                    if fold_gadget.len() < group.depth_fold {
                        return Err(AkitaError::InvalidSize {
                            expected: group.depth_fold,
                            actual: fold_gadget.len(),
                        });
                    }
                    fold_gadget
                } else {
                    fold_gadget_storage =
                        crate::gadget_row_scalars::<F>(group.depth_fold, group.log_basis);
                    &fold_gadget_storage
                };
                let z_range = group
                    .positions_per_block
                    .checked_mul(group.depth_commit)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                let mut z_eq_slice = vec![E::zero(); z_range];
                {
                    let _span = tracing::info_span!("setup_prepare_z_weights").entered();
                    setup_z_col_weights::<F, E>(
                        layout.witness_layout(),
                        layout.opening_source_len(),
                        group.group_id,
                        group.positions_per_block,
                        group.depth_commit,
                        group.depth_fold,
                        &eq_window,
                        fold_gadget,
                        &mut z_eq_slice,
                    )?;
                }
                let a_row_weights = static_group.a_row_weights.clone();
                let b_weights = static_group.b_weights.clone();
                let d_weights = static_plan.d_weights.clone();
                Ok(SetupContributionGroupPlan {
                    d_col_range: static_group.d_col_range.clone(),
                    t_cols: static_group.t_cols,
                    z_cols: static_group.z_cols,
                    n_a: static_group.n_a,
                    n_b: static_group.n_b,
                    required: static_group.required,
                    segments: static_group.segments.clone(),
                    e_eq_slice,
                    t_eq_slice,
                    z_eq_slice,
                    a_row_weights,
                    b_weights,
                    d_weights,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_groups = dynamic_groups
            .iter()
            .zip(groups)
            .map(|(planned, group)| {
                let d_active_cols = group
                    .num_claims
                    .checked_mul(group.live_block_count)
                    .and_then(|cols| cols.checked_mul(group.depth_open))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup D active width overflow".into())
                    })?;
                Ok(SetupProjectionGroupGeometry {
                    a_rows: planned.n_a,
                    a_cols: planned.z_cols,
                    b_rows: planned.n_b,
                    b_cols: planned.t_cols,
                    d_active_cols,
                    ownership_units: layout
                        .witness_layout()
                        .units_for_group(group.group_id)?
                        .len(),
                    depth_fold: group.depth_fold,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_geometry = crate::SetupProjectionGeometry::from_groups(
            role_dims,
            static_plan.d_rows,
            static_plan.d_physical_cols,
            &projection_groups,
        )?;
        Ok(SetupContributionPlan {
            groups: dynamic_groups,
            d_rows: static_plan.d_rows,
            d_physical_cols: static_plan.d_physical_cols,
            projection_geometry,
        })
    }

    /// Common-base packed-scan footprint.
    #[must_use]
    pub const fn required(&self) -> usize {
        self.projection_geometry.required()
    }

    /// Canonical common-base Stage 3 projection geometry.
    #[must_use]
    pub const fn projection_geometry(&self) -> SetupProjectionGeometry {
        self.projection_geometry
    }
}
