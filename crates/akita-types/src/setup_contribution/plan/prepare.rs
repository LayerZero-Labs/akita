use super::*;

impl<E: FieldCore> SetupContributionPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        groups: &[SetupContributionGroupInputs],
        relation_address: PreparedRelationAddress<E>,
        fold_gadget: Option<&[F]>,
        relation_address_geometry: RelationAddressGeometry,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
        let full_vec_randomness = relation_address.point();
        let expected_address_variables = relation_address_geometry.relation_lane_variable_count();
        if full_vec_randomness.len() != expected_address_variables {
            return Err(AkitaError::InvalidSize {
                expected: expected_address_variables,
                actual: full_vec_randomness.len(),
            });
        }
        let rows = {
            let _span = tracing::info_span!("setup_prepare_validate").entered();
            validate_setup_inputs(level_params, opening_batch, witness_layout, groups)?;
            validate_static_inputs(level_params, opening_batch, &eq_tau1)?
        };
        let group_geometry = groups
            .iter()
            .map(|group| {
                let role_dims = level_params.group_role_dims(opening_batch, group.group_id)?;
                let (b_subcolumns, d_subcolumns) =
                    SetupProjectionGeometry::native_role_subcolumn_counts(role_dims)?;
                let raw_d_cols = group.d_active_cols(level_params, opening_batch)?;
                Ok((role_dims, b_subcolumns, d_subcolumns, raw_d_cols))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut d_cursor = 0usize;
        let d_col_ranges = group_geometry
            .iter()
            .map(|(_, _, d_subcolumns, raw_d_cols)| {
                let width = raw_d_cols.checked_mul(*d_subcolumns).ok_or_else(|| {
                    AkitaError::InvalidSetup("setup D subcolumn width overflow".into())
                })?;
                let end = d_cursor
                    .checked_add(width)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
                let range = d_cursor..end;
                d_cursor = end;
                Ok(range)
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let (d_rows, d_physical_cols, d_weights) = {
            let _span = tracing::info_span!("setup_prepare_global_geometry").entered();
            let d_rows = level_params.open_commit_matrix.output_rank();
            let relation_geometry = crate::RelationWitnessGeometry::for_evaluation_trace_execution(
                level_params,
                opening_batch,
            )?;
            let row_families = relation_geometry.rhs_layout().row_families()?;
            let d_row_start = row_families
                .iter()
                .position(|family| matches!(family, crate::RelationRowFamily::Opening { .. }))
                .ok_or_else(|| AkitaError::InvalidSetup("setup D rows are missing".into()))?;
            let d_row_end = d_row_start
                .checked_add(d_rows)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D row range overflow".into()))?;
            if d_row_end > rows
                || row_families[d_row_start..d_row_end]
                    .iter()
                    .any(|family| !matches!(family, crate::RelationRowFamily::Opening { .. }))
            {
                return Err(AkitaError::InvalidSetup(
                    "setup D rows disagree with the relation layout".into(),
                ));
            }
            let d_physical_cols = d_cursor;
            let d_weights: std::sync::Arc<[E]> = if d_rows == 0 {
                Vec::new().into()
            } else {
                checked_slice(&eq_tau1, d_row_start, d_rows, "setup D rows")?
                    .to_vec()
                    .into()
            };
            (d_rows, d_physical_cols, d_weights)
        };
        let mut dynamic_groups = groups
            .iter()
            .zip(&group_geometry)
            .zip(&d_col_ranges)
            .map(|((group, (role_dims, b_subcolumns, _, _)), d_col_range)| {
                let geometry_span =
                    tracing::info_span!("setup_prepare_group_geometry", group_id = group.group_id)
                        .entered();
                let num_live_blocks = group.num_live_blocks(level_params, opening_batch)?;
                let num_positions_per_block =
                    group.num_positions_per_block(level_params, opening_batch)?;
                let depth_witness = group.depth_witness(level_params, opening_batch)?;
                let depth_commit = group.depth_commit(level_params, opening_batch)?;
                let depth_open = group.depth_open(level_params, opening_batch)?;
                let log_basis_open = group.log_basis_open(level_params, opening_batch)?;
                let group_params = level_params.group_params(opening_batch, group.group_id)?;
                let log_basis_inner = group_params.log_basis_inner();
                let log_basis_outer = group_params.log_basis_outer();
                let n_a = group.n_a(level_params, opening_batch)?;
                let physical_n_b = group_params.b_rows_len();
                let t_vector_width = group.t_vector_width(level_params, opening_batch)?;
                let d_col_range = d_col_range.clone();
                let expected_logical_t_cols = group
                    .num_claims
                    .checked_mul(t_vector_width)
                    .and_then(|cols| cols.checked_mul(*b_subcolumns))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
                let slice_geometry = crate::CommitmentSliceGeometry::try_new(
                    group_params.outer_slice_count(),
                    num_live_blocks,
                    group.num_claims,
                    n_a,
                    depth_commit,
                    role_dims.d_a(),
                    role_dims.d_b(),
                )?;
                if slice_geometry.logical_input_width() != expected_logical_t_cols {
                    return Err(AkitaError::InvalidSetup(
                        "logical B columns disagree with slice geometry".into(),
                    ));
                }
                let z_cols = num_positions_per_block
                    .checked_mul(depth_witness)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                let a_row_weights: std::sync::Arc<[E]> =
                    checked_slice(&eq_tau1, group.a_row_start, n_a, "setup A rows")?
                        .to_vec()
                        .into();
                let logical_n_b = slice_geometry.logical_output_rows(physical_n_b)?;
                let b_weights: std::sync::Arc<[E]> =
                    checked_slice(&eq_tau1, group.b_row_start, logical_n_b, "setup B rows")?
                        .to_vec()
                        .into();
                let physical_b = PhysicalBSetupPlan::new(slice_geometry, physical_n_b, b_weights)?;
                let consistency_weight = *eq_tau1
                    .get(level_params.consistency_row_index(opening_batch, group.group_id)?)
                    .ok_or(AkitaError::InvalidProof)?;
                let num_physical_units = witness_layout.units_for_group(group.group_id)?.count();
                let active_unit_ranges = witness_layout
                    .units_for_group(group.group_id)?
                    .filter(|unit| unit.num_live_blocks() != 0)
                    .map(|unit| SetupUnitRange {
                        global_block_start: unit.global_block_start(),
                        num_live_blocks: unit.num_live_blocks(),
                    })
                    .collect::<Vec<_>>()
                    .into();
                drop(geometry_span);
                let fold_gadget_storage;
                let group_fold_gadget = if let Some(fold_gadget) = fold_gadget {
                    if fold_gadget.len() < group.depth_fold {
                        return Err(AkitaError::InvalidSize {
                            expected: group.depth_fold,
                            actual: fold_gadget.len(),
                        });
                    }
                    fold_gadget
                        .get(..group.depth_fold)
                        .ok_or(AkitaError::InvalidProof)?
                } else {
                    fold_gadget_storage =
                        crate::gadget_row_scalars::<F>(group.depth_fold, log_basis_open);
                    &fold_gadget_storage
                };
                let fold_gadget: std::sync::Arc<[E]> = group_fold_gadget
                    .iter()
                    .take(group.depth_fold)
                    .copied()
                    .map(|fold| E::one().mul_base(fold))
                    .collect::<Vec<_>>()
                    .into();
                Ok(SetupContributionGroupPlan {
                    group_id: group.group_id,
                    role_dims: *role_dims,
                    a_ratio: 0,
                    b_ratio: 0,
                    d_ratio: 0,
                    consistency_weight,
                    num_claims: group.num_claims,
                    num_live_blocks,
                    num_positions_per_block,
                    depth_witness,
                    depth_commit,
                    depth_open,
                    log_basis_inner,
                    log_basis_outer,
                    log_basis_open,
                    d_col_range,
                    z_cols,
                    n_a,
                    physical_b,
                    required: 0,
                    segments: Vec::new().into(),
                    a_row_weights,
                    fold_gadget,
                    direct_scan_weights: None,
                    active_unit_ranges,
                    num_physical_units,
                    d_tensors: Vec::new(),
                    a_tensors: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_groups = dynamic_groups
            .iter()
            .map(|planned| {
                Ok(SetupProjectionGroupGeometry {
                    role_dims: planned.role_dims,
                    a_rows: planned.n_a,
                    a_cols: planned.z_cols,
                    b_rows: planned.physical_b.physical_rows(),
                    b_cols: planned.physical_b.physical_input_width(),
                    d_active_cols: planned.d_col_range.len(),
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_geometry = crate::SetupProjectionGeometry::from_groups(
            relation_address_geometry.role_dims(),
            d_rows,
            d_physical_cols,
            &projection_groups,
        )?;
        if projection_geometry.base_ring_dim()
            != relation_address_geometry.relation_coefficient_block_len()
        {
            return Err(AkitaError::InvalidSetup(
                "Stage 3 setup projection and relation point use different current-role bases"
                    .into(),
            ));
        }
        let base = projection_geometry.base_ring_dim();
        for group in &mut dynamic_groups {
            group.set_projection_ratios(base)?;
        }
        let mut plan = SetupContributionPlan {
            groups: dynamic_groups,
            d_rows,
            d_physical_cols,
            d_weights,
            setup_index_tensors: Vec::new(),
            relation_address,
            relation_address_geometry,
            projection_geometry,
            direct_scan_alpha: None,
        };
        plan.setup_index_tensors = plan.prepare_setup_index_tensors(witness_layout)?;
        Ok(plan)
    }

    /// Materialize the derived column-weight and scan caches used only by the
    /// direct setup scan.
    pub fn materialize_direct_scan(&mut self, alpha: E) -> Result<(), AkitaError> {
        if self
            .direct_scan_alpha
            .is_some_and(|prepared| prepared != alpha)
        {
            return Err(AkitaError::InvalidInput(
                "direct setup weights were prepared for a different alpha".into(),
            ));
        }
        self.direct_scan_alpha = Some(alpha);
        for group_index in 0..self.groups.len() {
            if self
                .groups
                .get(group_index)
                .is_some_and(|group| group.direct_scan_weights.is_some())
            {
                continue;
            }
            let (e, t, z) = {
                let group = self
                    .groups
                    .get(group_index)
                    .ok_or(AkitaError::InvalidProof)?;
                // `materialize_role_tensor_weights` already gates its own
                // parallelism on this threshold per output length, so forking
                // on the largest of the three keeps the outer decision on the
                // same policy: below it every job stays sequential and
                // `rayon::join` would only add scheduling cost.
                const PARALLEL_THRESHOLD: usize = 1 << 14;
                let largest_output = group
                    .d_col_range
                    .len()
                    .max(group.physical_b.logical_input_width())
                    .max(group.z_cols);
                if largest_output >= PARALLEL_THRESHOLD {
                    // Shared reborrow alongside group's sub-borrow; both are &T so
                    // NLL allows them to coexist for parallel closure capture.
                    let plan: &Self = self;
                    let (e_res, (t_res, z_res)) = cfg_join!(
                        || {
                            let _span =
                                tracing::info_span!("setup_materialize_e_weights").entered();
                            plan.materialize_role_tensor_weights(
                                group.d_ratio,
                                &group.d_tensors,
                                group.d_col_range.len(),
                                alpha,
                            )
                        },
                        || cfg_join!(
                            || {
                                let _span =
                                    tracing::info_span!("setup_materialize_t_weights").entered();
                                plan.materialize_role_tensor_weights(
                                    group.b_ratio,
                                    &group.physical_b.relation_tensors,
                                    group.physical_b.logical_input_width(),
                                    alpha,
                                )
                            },
                            || {
                                let _span =
                                    tracing::info_span!("setup_materialize_z_weights").entered();
                                plan.materialize_role_tensor_weights(
                                    group.a_ratio,
                                    &group.a_tensors,
                                    group.z_cols,
                                    alpha,
                                )
                            }
                        )
                    );
                    (e_res?, t_res?, z_res?)
                } else {
                    let e = {
                        let _span = tracing::info_span!("setup_materialize_e_weights").entered();
                        self.materialize_role_tensor_weights(
                            group.d_ratio,
                            &group.d_tensors,
                            group.d_col_range.len(),
                            alpha,
                        )?
                    };
                    let t = {
                        let _span = tracing::info_span!("setup_materialize_t_weights").entered();
                        self.materialize_role_tensor_weights(
                            group.b_ratio,
                            &group.physical_b.relation_tensors,
                            group.physical_b.logical_input_width(),
                            alpha,
                        )?
                    };
                    let z = {
                        let _span = tracing::info_span!("setup_materialize_z_weights").entered();
                        self.materialize_role_tensor_weights(
                            group.a_ratio,
                            &group.a_tensors,
                            group.z_cols,
                            alpha,
                        )?
                    };
                    (e, t, z)
                }
            };
            let group = self
                .groups
                .get_mut(group_index)
                .ok_or(AkitaError::InvalidProof)?;
            group.direct_scan_weights = Some(DirectScanWeights { e, t, z });
            {
                let _span = tracing::info_span!("setup_materialize_scan_segments").entered();
                group.refresh_segments(
                    &self.d_weights,
                    self.d_rows,
                    self.d_physical_cols,
                    group.a_ratio,
                    group.b_ratio,
                    group.d_ratio,
                )?;
            }
        }
        Ok(())
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

    /// Canonical relation-address geometry used by every setup contribution
    /// tensor.
    #[must_use]
    pub const fn relation_address_geometry(&self) -> RelationAddressGeometry {
        self.relation_address_geometry
    }
}

fn validate_static_inputs<E: FieldCore>(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    eq_tau1: &[E],
) -> Result<usize, AkitaError> {
    opening_batch.check()?;
    let num_groups = opening_batch.num_groups();
    let depth_fold = level_params.num_digits_fold();
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
        let depth_witness = group_params.num_digits_inner();
        let depth_commit = group_params.num_digits_outer();
        let depth_open = group_params.num_digits_open();
        let num_positions_per_block = group_params.num_positions_per_block();
        let num_live_blocks = group_params.num_live_blocks();
        if num_positions_per_block == 0
            || depth_witness == 0
            || depth_commit == 0
            || depth_open == 0
        {
            return Err(AkitaError::InvalidSetup(
                "setup evaluator layout has zero width".into(),
            ));
        }
        let inner_width = num_positions_per_block
            .checked_mul(depth_witness)
            .ok_or_else(|| AkitaError::InvalidSetup("inner width overflow".into()))?;
        if group_params.a_col_len() < inner_width {
            return Err(AkitaError::InvalidSetup(
                "A-key column width is too small for setup contribution layout".into(),
            ));
        }
        let role_dims = level_params.group_role_dims(opening_batch, group_index)?;
        let expected_b_width = crate::CommitmentSliceGeometry::try_new(
            group_params.outer_slice_count(),
            num_live_blocks,
            group_layout.num_polynomials(),
            group_params.a_rows_len(),
            depth_commit,
            role_dims.d_a(),
            role_dims.d_b(),
        )?
        .physical_input_width();
        if group_params.b_col_len() < expected_b_width {
            return Err(AkitaError::InvalidSetup(
                "B-key column width is too small for setup contribution layout".into(),
            ));
        }
    }
    let rows = level_params.relation_matrix_row_count(num_groups)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    Ok(rows)
}
