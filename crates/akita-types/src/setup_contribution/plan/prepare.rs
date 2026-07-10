use super::*;

impl<E: FieldCore> SetupContributionPlan<E> {
    pub fn prepare_static(
        inputs: &SetupContributionPlanInputs<E>,
        groups: &[SetupContributionGroupInputs],
        d_row_start: usize,
        d_rows: usize,
        d_physical_cols: usize,
    ) -> Result<SetupContributionStatic<E>, AkitaError> {
        let d_weights: std::sync::Arc<[E]> = if d_rows == 0 {
            Vec::new().into()
        } else {
            checked_slice(&inputs.eq_tau1, d_row_start, d_rows, "setup D rows")?
                .to_vec()
                .into()
        };
        let num_groups = groups.len();
        let static_groups = groups
            .iter()
            .map(|group| {
                validate_group_chunk_layout(group, num_groups)?;
                let t_cols = group
                    .num_claims
                    .checked_mul(group.t_cols_per_vector)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
                let z_cols = group
                    .block_len
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
                    .checked_mul(group.num_blocks)
                    .and_then(|cols| cols.checked_mul(group.depth_open))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup E active width overflow".into())
                    })?;
                let (required, segments) = build_packed_segments(
                    group.e_col_offset,
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
                    e_col_offset: group.e_col_offset,
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
        groups: &[SetupContributionGroupInputs],
        role_dims: CommitmentRingDims,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
        if static_plan.groups.len() != groups.len() {
            return Err(AkitaError::InvalidSize {
                expected: groups.len(),
                actual: static_plan.groups.len(),
            });
        }
        let dynamic_groups = static_plan
            .groups
            .iter()
            .zip(groups)
            .map(|(static_group, group)| {
                let e_eq_slice = {
                    let _span = tracing::info_span!("setup_prepare_e_weights").entered();
                    setup_e_col_weights::<E>(
                        &group.layout,
                        group.opening_layout,
                        group.group_id,
                        group.num_blocks,
                        group.num_claims,
                        group.depth_open,
                        full_vec_randomness,
                    )?
                };
                let t_eq_slice = {
                    let _span = tracing::info_span!("setup_prepare_t_weights").entered();
                    setup_t_col_weights::<E>(
                        &group.layout,
                        group.opening_layout,
                        group.group_id,
                        group.num_blocks,
                        group.depth_open,
                        group.n_a,
                        group.t_cols_per_vector,
                        group.num_claims,
                        group.num_claims,
                        0,
                        full_vec_randomness,
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
                    .block_len
                    .checked_mul(group.depth_commit)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                let mut z_eq_slice = vec![E::zero(); z_range];
                {
                    let _span = tracing::info_span!("setup_prepare_z_weights").entered();
                    setup_z_col_weights::<F, E>(
                        &group.layout,
                        group.opening_layout,
                        group.group_id,
                        group.block_len,
                        group.depth_commit,
                        group.depth_fold,
                        full_vec_randomness,
                        fold_gadget,
                        &mut z_eq_slice,
                    )?;
                }
                let a_row_weights = static_group.a_row_weights.clone();
                let b_weights = static_group.b_weights.clone();
                let d_weights = static_plan.d_weights.clone();
                Ok(SetupContributionGroupPlan {
                    e_col_offset: static_group.e_col_offset,
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
                    .checked_mul(group.num_blocks)
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
                    ownership_units: group.layout.units_for_group(group.group_id)?.len(),
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
