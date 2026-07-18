use super::*;

impl<E: FieldCore> SetupContributionPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        level_params: &LevelParams,
        opening_batch: &OpeningClaimsLayout,
        relation_matrix_row_layout: RelationMatrixRowLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        opening_source_len: usize,
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        role_dims: CommitmentRingDims,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
        let rows = {
            let _span = tracing::info_span!("setup_prepare_validate").entered();
            validate_setup_inputs(
                level_params,
                opening_batch,
                relation_matrix_row_layout,
                witness_layout,
                groups,
            )?;
            validate_static_inputs(
                level_params,
                opening_batch,
                relation_matrix_row_layout,
                &eq_tau1,
            )?
        };
        let (d_rows, d_physical_cols, d_weights) = {
            let _span = tracing::info_span!("setup_prepare_global_geometry").entered();
            let d_rows = match relation_matrix_row_layout {
                crate::RelationMatrixRowLayout::WithDBlock => level_params.d_key.row_len(),
                crate::RelationMatrixRowLayout::WithoutCommitmentBlocks => 0,
            };
            let d_row_start = rows.checked_sub(d_rows).ok_or_else(|| {
                AkitaError::InvalidSetup("setup D rows exceed relation rows".into())
            })?;
            let d_physical_cols = get_total_d(level_params, opening_batch, groups)?;
            let d_weights: std::sync::Arc<[E]> = if d_rows == 0 {
                Vec::new().into()
            } else {
                checked_slice(&eq_tau1, d_row_start, d_rows, "setup D rows")?
                    .to_vec()
                    .into()
            };
            (d_rows, d_physical_cols, d_weights)
        };
        // Build the bounded equality window once and share it across every E/T/Z
        // column weight. Each canonical column address then costs one bounded
        // low-table lookup plus a short high evaluation instead of a full
        // `O(col_bits+ring_bits)` equality product recomputed per column, which
        // was the dominant verifier setup-plan cost after the digit-innermost
        // cutover (root cause 4).
        let eq_window = {
            let _span = tracing::info_span!("setup_prepare_eq_window").entered();
            akita_algebra::offset_eq::OffsetEqWindow::new(full_vec_randomness)?
        };
        let mut dynamic_groups = groups
            .iter()
            .map(|group| {
                let geometry_span =
                    tracing::info_span!("setup_prepare_group_geometry", group_id = group.group_id)
                        .entered();
                let num_live_blocks = group.num_live_blocks(level_params, opening_batch)?;
                let num_positions_per_block =
                    group.num_positions_per_block(level_params, opening_batch)?;
                let depth_open = group.depth_open(level_params, opening_batch)?;
                let depth_commit = group.depth_commit(level_params, opening_batch)?;
                let log_basis = group.log_basis(level_params, opening_batch)?;
                let n_a = group.n_a(level_params, opening_batch)?;
                let n_b = group.n_b(level_params, opening_batch, relation_matrix_row_layout)?;
                let t_vector_width = group.t_vector_width(level_params, opening_batch)?;
                let d_col_range =
                    get_d_col_range(level_params, opening_batch, groups, group.group_id)?;
                let t_cols = group
                    .num_claims
                    .checked_mul(t_vector_width)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
                let z_cols = num_positions_per_block
                    .checked_mul(depth_commit)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                let a_row_weights: std::sync::Arc<[E]> =
                    checked_slice(&eq_tau1, group.a_row_start, n_a, "setup A rows")?
                        .to_vec()
                        .into();
                let b_weights: std::sync::Arc<[E]> =
                    checked_slice(&eq_tau1, group.b_row_start, n_b, "setup B rows")?
                        .to_vec()
                        .into();
                drop(geometry_span);
                let e_eq_slice = {
                    let _span = tracing::info_span!("setup_prepare_e_weights").entered();
                    setup_e_col_weights::<E>(
                        witness_layout,
                        opening_source_len,
                        group.group_id,
                        num_live_blocks,
                        group.num_claims,
                        depth_open,
                        &eq_window,
                    )?
                };
                let t_eq_slice = {
                    let _span = tracing::info_span!("setup_prepare_t_weights").entered();
                    setup_t_col_weights::<E>(
                        witness_layout,
                        opening_source_len,
                        group.group_id,
                        num_live_blocks,
                        depth_open,
                        n_a,
                        group.num_claims,
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
                        crate::gadget_row_scalars::<F>(group.depth_fold, log_basis);
                    &fold_gadget_storage
                };
                let z_range = num_positions_per_block
                    .checked_mul(depth_commit)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                let mut z_eq_slice = vec![E::zero(); z_range];
                {
                    let _span = tracing::info_span!("setup_prepare_z_weights").entered();
                    setup_z_col_weights::<F, E>(
                        witness_layout,
                        opening_source_len,
                        group.group_id,
                        num_positions_per_block,
                        depth_commit,
                        group.depth_fold,
                        &eq_window,
                        fold_gadget,
                        &mut z_eq_slice,
                    )?;
                }

                Ok(SetupContributionGroupPlan {
                    d_col_range,
                    t_cols,
                    z_cols,
                    n_a,
                    n_b,
                    required: 0,
                    segments: Vec::new().into(),
                    a_row_weights,
                    b_weights,
                    e_eq_slice,
                    t_eq_slice,
                    z_eq_slice,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_groups = dynamic_groups
            .iter()
            .zip(groups)
            .map(|(planned, group)| {
                let d_active_cols = group.d_active_cols(level_params, opening_batch)?;
                Ok(SetupProjectionGroupGeometry {
                    a_rows: planned.n_a,
                    a_cols: planned.z_cols,
                    b_rows: planned.n_b,
                    b_cols: planned.t_cols,
                    d_active_cols,
                    ownership_units: witness_layout.units_for_group(group.group_id)?.len(),
                    depth_fold: group.depth_fold,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_geometry = crate::SetupProjectionGeometry::from_groups(
            role_dims,
            d_rows,
            d_physical_cols,
            &projection_groups,
        )?;
        for group in &mut dynamic_groups {
            group.refresh_segments(
                &d_weights,
                d_rows,
                d_physical_cols,
                projection_geometry.a_ratio(),
                projection_geometry.b_ratio(),
                projection_geometry.d_ratio(),
            )?;
        }
        Ok(SetupContributionPlan {
            groups: dynamic_groups,
            d_rows,
            d_physical_cols,
            d_weights,
            projection_geometry,
            eq_window,
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

fn validate_static_inputs<E: FieldCore>(
    level_params: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    relation_matrix_row_layout: RelationMatrixRowLayout,
    eq_tau1: &[E],
) -> Result<usize, AkitaError> {
    opening_batch.check()?;
    let num_groups = opening_batch.num_groups();
    let num_polynomials = opening_batch.num_total_polynomials();
    let depth_fold =
        level_params.num_digits_fold(num_polynomials, level_params.field_bits_for_cache())?;
    if level_params.num_live_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "num_live_blocks must be positive".into(),
        ));
    }
    if depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "setup evaluator layout has zero width".into(),
        ));
    }
    for group_index in 0..num_groups {
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_params = level_params.group_params(opening_batch, group_index)?;
        let depth_commit = group_params.num_digits_commit();
        let depth_open = group_params.num_digits_open();
        let num_positions_per_block = group_params.num_positions_per_block();
        let num_live_blocks = group_params.num_live_blocks();
        if num_positions_per_block == 0 || depth_commit == 0 || depth_open == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        let inner_width = num_positions_per_block
            .checked_mul(depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".into()))?;
        if group_params.a_col_len() < inner_width {
            return Err(AkitaError::InvalidSetup(
                "A-key column width is too small for setup contribution layout".into(),
            ));
        }
        let expected_b_width = group_layout
            .num_polynomials()
            .checked_mul(group_params.a_rows_len())
            .and_then(|width| width.checked_mul(depth_open))
            .and_then(|width| width.checked_mul(num_live_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".into()))?;
        if group_params.b_col_len() < expected_b_width {
            return Err(AkitaError::InvalidSetup(
                "B-key column width is too small for setup contribution layout".into(),
            ));
        }
    }
    let rows =
        level_params.relation_matrix_row_count_for(num_groups, relation_matrix_row_layout)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    Ok(rows)
}
