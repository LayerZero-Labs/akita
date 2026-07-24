use super::*;

impl<E: FieldCore> SetupContributionPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        opening_source_len: usize,
        full_vec_randomness: &[E],
        role_dims: CommitmentRingDims,
        outgoing_ring_dim: usize,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
        let groups = validate_setup_inputs(level_params, opening_batch, witness_layout)?;
        // The relation address domain is sized from the padded opening-source
        // length that the ring-switch evaluation point is built from — not from
        // `witness_layout.total_len()`, which can round to a smaller power of two
        // in mixed-D folds and would desync this plan from the actual point.
        if witness_layout.total_len() > opening_source_len {
            return Err(AkitaError::InvalidSetup(
                "witness layout exceeds opening source length".into(),
            ));
        }
        crate::validate_role_dims(role_dims)?;
        if outgoing_ring_dim == 0 || !outgoing_ring_dim.is_power_of_two() {
            return Err(AkitaError::InvalidSetup(
                "outgoing witness ring dimension must be a non-zero power of two".into(),
            ));
        }
        let common_coeff_count = role_dims.common_relation_witness_coeff_count(outgoing_ring_dim);
        if common_coeff_count == 0
            || !common_coeff_count.is_power_of_two()
            || !role_dims.d_a().is_multiple_of(common_coeff_count)
            || !role_dims.d_b().is_multiple_of(common_coeff_count)
            || !role_dims.d_d().is_multiple_of(common_coeff_count)
            || !outgoing_ring_dim.is_multiple_of(common_coeff_count)
        {
            return Err(AkitaError::InvalidSetup(
                "relation and outgoing witness do not admit a common coefficient block".into(),
            ));
        }
        let relation_field_len = crate::opening_domain_len(opening_source_len)?
            .checked_mul(outgoing_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation point domain overflow".into()))?;
        let relation_address_len = relation_field_len
            .checked_div(common_coeff_count)
            .filter(|len| len.is_power_of_two())
            .ok_or_else(|| {
                AkitaError::InvalidSetup("relation address domain must be a power of two".into())
            })?;
        let expected_address_bits = relation_address_len.trailing_zeros() as usize;
        if full_vec_randomness.len() != expected_address_bits {
            return Err(AkitaError::InvalidSize {
                expected: expected_address_bits,
                actual: full_vec_randomness.len(),
            });
        }
        let inner_lane_count = role_dims.d_a() / common_coeff_count;
        let outer_lane_count = role_dims.d_b() / common_coeff_count;
        let opening_lane_count = role_dims.d_d() / common_coeff_count;
        let (outer_subcolumns, opening_subcolumns) =
            SetupProjectionGeometry::witness_subcolumn_ratios(role_dims)?;
        let rows = {
            let _span = tracing::info_span!("setup_prepare_validate").entered();
            validate_static_inputs(level_params, opening_batch, &eq_tau1)?
        };
        let (d_row_start, d_rows, d_physical_cols) = {
            let _span = tracing::info_span!("setup_prepare_global_geometry").entered();
            let d_rows = level_params.open_commit_matrix.output_rank();
            let d_row_start = rows.checked_sub(d_rows).ok_or_else(|| {
                AkitaError::InvalidSetup("setup D rows exceed relation rows".into())
            })?;
            let d_physical_cols = get_total_d(level_params, opening_batch, &groups)?
                .checked_mul(opening_subcolumns)
                .ok_or_else(|| AkitaError::InvalidSetup("native setup D width overflow".into()))?;
            (d_row_start, d_rows, d_physical_cols)
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
        let max_depth_fold = groups
            .iter()
            .map(|&group_id| {
                let group_params = level_params.group_params(opening_batch, group_id)?;
                let num_claims = opening_batch.group_layout(group_id)?.num_polynomials();
                level_params.num_digits_fold_for_params(
                    group_params,
                    num_claims,
                    level_params.field_bits_for_cache(),
                )
            })
            .collect::<Result<Vec<_>, AkitaError>>()?
            .into_iter()
            .max()
            .ok_or_else(|| AkitaError::InvalidSetup("setup groups are empty".into()))?;
        let fold_gadget_base =
            crate::gadget_row_scalars::<F>(max_depth_fold, level_params.log_basis_open);
        let fold_gadget: std::sync::Arc<[E]> = fold_gadget_base
            .iter()
            .copied()
            .map(|fold| E::one().mul_base(fold))
            .collect::<Vec<_>>()
            .into();

        let dynamic_groups = groups
            .iter()
            .map(|&group_id| {
                let geometry_span =
                    tracing::info_span!("setup_prepare_group_geometry", group_id).entered();
                let group_params = level_params.group_params(opening_batch, group_id)?;
                let group_layout = opening_batch.group_layout(group_id)?;
                let num_claims = group_layout.num_polynomials();
                let num_live_blocks = group_params.num_live_blocks();
                let num_positions_per_block = group_params.num_positions_per_block();
                let depth_witness = group_params.num_digits_inner();
                let depth_commit = group_params.num_digits_outer();
                let depth_open = group_params.num_digits_open();
                let depth_fold = level_params.num_digits_fold_for_params(
                    group_params,
                    num_claims,
                    level_params.field_bits_for_cache(),
                )?;
                let n_a = group_params.a_rows_len();
                let n_b = group_params.b_rows_len();
                let t_vector_width = n_a
                    .checked_mul(depth_commit)
                    .and_then(|width| width.checked_mul(num_live_blocks))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup B vector width overflow".into())
                    })?;
                let a_row_start = level_params.a_row_range(opening_batch, group_id)?.start;
                let b_row_start = level_params
                    .commitment_row_range(opening_batch, group_id)?
                    .start;
                let semantic_d_col_range =
                    get_d_col_range(level_params, opening_batch, &groups, group_id)?;
                let d_native_start = groups
                    .iter()
                    .take_while(|&&candidate| candidate != group_id)
                    .try_fold(0usize, |start, &candidate| {
                        d_active_cols(level_params, opening_batch, candidate)?
                            .checked_mul(opening_subcolumns)
                            .and_then(|width| start.checked_add(width))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("native setup D width overflow".into())
                            })
                    })?;
                let d_native_len = semantic_d_col_range
                    .len()
                    .checked_mul(opening_subcolumns)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("native setup D width overflow".into())
                    })?;
                let d_col_range = d_native_start
                    ..d_native_start.checked_add(d_native_len).ok_or_else(|| {
                        AkitaError::InvalidSetup("native setup D width overflow".into())
                    })?;
                let t_cols = group_layout
                    .num_polynomials()
                    .checked_mul(t_vector_width)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
                let b_native_cols = t_cols.checked_mul(outer_subcolumns).ok_or_else(|| {
                    AkitaError::InvalidSetup("native setup B width overflow".into())
                })?;
                let z_cols = num_positions_per_block
                    .checked_mul(depth_witness)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                if fold_gadget_base.len() < depth_fold {
                    return Err(AkitaError::InvalidSize {
                        expected: depth_fold,
                        actual: fold_gadget_base.len(),
                    });
                }
                drop(geometry_span);
                let mut d_spans = Vec::new();
                let mut b_spans = Vec::new();
                let mut a_spans = Vec::new();
                for unit in witness_layout.units_for_group(group_id)? {
                    for claim in 0..num_claims {
                        for subcolumn in 0..opening_subcolumns {
                            for digit in 0..depth_open {
                                let d_setup_start = claim
                                    .checked_mul(num_live_blocks)
                                    .and_then(|base| base.checked_add(unit.global_block_start()))
                                    .and_then(|base| base.checked_mul(opening_subcolumns))
                                    .and_then(|base| base.checked_add(subcolumn))
                                    .and_then(|base| base.checked_mul(depth_open))
                                    .and_then(|base| base.checked_add(digit))
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup("setup D address overflow".into())
                                    })?;
                                let d_witness_column = unit.e_index(
                                    num_claims,
                                    depth_open,
                                    claim,
                                    unit.global_block_start(),
                                    digit,
                                )?;
                                let d_witness_start = d_witness_column
                                    .checked_mul(inner_lane_count)
                                    .and_then(|base| {
                                        subcolumn
                                            .checked_mul(opening_lane_count)
                                            .and_then(|offset| base.checked_add(offset))
                                    })
                                    .ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "setup D relation address overflow".into(),
                                        )
                                    })?;
                                d_spans.push(SetupContributionSpan::new(
                                    d_setup_start,
                                    opening_subcolumns.checked_mul(depth_open).ok_or_else(
                                        || {
                                            AkitaError::InvalidSetup(
                                                "setup D stride overflow".into(),
                                            )
                                        },
                                    )?,
                                    d_witness_start,
                                    depth_open.checked_mul(inner_lane_count).ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "setup D relation stride overflow".into(),
                                        )
                                    })?,
                                    unit.num_live_blocks(),
                                    None,
                                    d_native_len,
                                    relation_address_len,
                                    opening_lane_count,
                                )?);
                            }
                        }

                        for a_row in 0..n_a {
                            for digit in 0..depth_commit {
                                for subcolumn in 0..outer_subcolumns {
                                    let b_setup_start = claim
                                        .checked_mul(num_live_blocks)
                                        .and_then(|base| {
                                            base.checked_add(unit.global_block_start())
                                        })
                                        .and_then(|base| base.checked_mul(n_a))
                                        .and_then(|base| base.checked_add(a_row))
                                        .and_then(|base| base.checked_mul(depth_commit))
                                        .and_then(|base| base.checked_add(digit))
                                        .and_then(|base| base.checked_mul(outer_subcolumns))
                                        .and_then(|base| base.checked_add(subcolumn))
                                        .ok_or_else(|| {
                                            AkitaError::InvalidSetup(
                                                "setup B address overflow".into(),
                                            )
                                        })?;
                                    let b_witness_column = unit.t_index(
                                        num_claims,
                                        n_a,
                                        depth_commit,
                                        claim,
                                        unit.global_block_start(),
                                        a_row,
                                        digit,
                                    )?;
                                    let b_witness_start = b_witness_column
                                        .checked_mul(inner_lane_count)
                                        .and_then(|base| {
                                            subcolumn
                                                .checked_mul(outer_lane_count)
                                                .and_then(|offset| base.checked_add(offset))
                                        })
                                        .ok_or_else(|| {
                                            AkitaError::InvalidSetup(
                                                "setup B relation address overflow".into(),
                                            )
                                        })?;
                                    b_spans.push(SetupContributionSpan::new(
                                        b_setup_start,
                                        n_a.checked_mul(depth_commit)
                                            .and_then(|stride| stride.checked_mul(outer_subcolumns))
                                            .ok_or_else(|| {
                                                AkitaError::InvalidSetup(
                                                    "setup B stride overflow".into(),
                                                )
                                            })?,
                                        b_witness_start,
                                        n_a.checked_mul(depth_commit)
                                            .and_then(|stride| stride.checked_mul(inner_lane_count))
                                            .ok_or_else(|| {
                                                AkitaError::InvalidSetup(
                                                    "setup B relation stride overflow".into(),
                                                )
                                            })?,
                                        unit.num_live_blocks(),
                                        None,
                                        b_native_cols,
                                        relation_address_len,
                                        outer_lane_count,
                                    )?);
                                }
                            }
                        }
                    }
                    for fold_digit in 0..depth_fold {
                        let a_witness_start = unit.z_index(
                            num_positions_per_block,
                            depth_witness,
                            depth_fold,
                            0,
                            0,
                            fold_digit,
                        )?;
                        a_spans.push(SetupContributionSpan::new(
                            0,
                            1,
                            a_witness_start
                                .checked_mul(inner_lane_count)
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup(
                                        "setup A relation address overflow".into(),
                                    )
                                })?,
                            depth_fold.checked_mul(inner_lane_count).ok_or_else(|| {
                                AkitaError::InvalidSetup("setup A relation stride overflow".into())
                            })?,
                            z_cols,
                            Some(fold_digit),
                            z_cols,
                            relation_address_len,
                            inner_lane_count,
                        )?);
                    }
                }
                Ok(SetupContributionGroupPlan {
                    group_id,
                    num_claims,
                    num_live_blocks,
                    num_positions_per_block,
                    depth_witness,
                    depth_commit,
                    depth_open,
                    depth_fold,
                    log_basis_inner: group_params.log_basis_inner(),
                    log_basis_outer: group_params.log_basis_outer(),
                    log_basis_open: group_params.log_basis_open(),
                    a_row_start,
                    b_row_start,
                    d_col_range,
                    t_cols: b_native_cols,
                    z_cols,
                    n_a,
                    n_b,
                    d_spans,
                    b_spans,
                    a_spans,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_groups = dynamic_groups
            .iter()
            .map(|planned| {
                Ok(SetupProjectionGroupGeometry {
                    a_rows: planned.n_a,
                    a_cols: planned.z_cols,
                    b_rows: planned.n_b,
                    b_cols: planned.t_cols,
                    d_active_cols: planned.d_col_range.len(),
                    ownership_units: witness_layout.units_for_group(planned.group_id)?.len(),
                    depth_fold: planned.depth_fold,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_geometry = crate::SetupProjectionGeometry::from_groups(
            role_dims,
            d_rows,
            d_physical_cols,
            &projection_groups,
        )?;
        projection_geometry.ensure_evaluation_budget()?;
        Ok(SetupContributionPlan {
            groups: dynamic_groups,
            eq_tau1,
            x_challenges: full_vec_randomness.to_vec().into(),
            fold_gadget,
            common_coeff_count,
            inner_lane_count,
            outer_lane_count,
            opening_lane_count,
            d_row_start,
            d_rows,
            d_physical_cols,
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
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
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
        let expected_b_width = group_layout
            .num_polynomials()
            .checked_mul(group_params.a_rows_len())
            .and_then(|width| width.checked_mul(depth_commit))
            .and_then(|width| width.checked_mul(num_live_blocks))
            .ok_or_else(|| AkitaError::InvalidSetup("B-matrix width overflow".into()))?;
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
