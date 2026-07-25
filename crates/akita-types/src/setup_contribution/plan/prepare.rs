use super::*;
use akita_algebra::ring::scalar_powers;

/// Per-role relation lane geometry derived from `role_dims` and the folding
/// challenge `alpha`. For uniform roles every ratio is 1 and every spec reduces
/// to the pre-existing single-`eq` fast path.
struct RelationLaneGeometry<E> {
    /// `d_a / base_ring_dim` (total relation lanes per inner ring element).
    a_ratio: usize,
    /// Distinct physical subcolumns per role: `a_ratio / (d_role / base)`.
    d_subcolumns: usize,
    b_subcolumns: usize,
    /// `α^{base_ring_dim · l}` tables, one per role (length `d_role / base`).
    a_lane_alpha: Vec<E>,
    b_lane_alpha: Vec<E>,
    d_lane_alpha: Vec<E>,
}

impl<E: FieldCore> RelationLaneGeometry<E> {
    fn new(role_dims: CommitmentRingDims, alpha: E) -> Result<Self, AkitaError> {
        let base = role_dims.d_a().min(role_dims.d_b()).min(role_dims.d_d());
        if base == 0 {
            return Err(AkitaError::InvalidSetup("zero base ring dimension".into()));
        }
        let ratio = |d: usize| -> Result<usize, AkitaError> {
            if !d.is_multiple_of(base) {
                return Err(AkitaError::InvalidSetup(
                    "role dimension does not decompose over the Stage 3 base".into(),
                ));
            }
            Ok(d / base)
        };
        let a_ratio = ratio(role_dims.d_a())?;
        let b_lanes = ratio(role_dims.d_b())?;
        let d_lanes = ratio(role_dims.d_d())?;
        let subcolumns = |lanes: usize| -> Result<usize, AkitaError> {
            a_ratio
                .checked_div(lanes)
                .filter(|s| *s != 0 && a_ratio == s * lanes)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("role lanes do not divide the inner witness".into())
                })
        };
        // `α^base`, then per-role lane tables `[1, α^base, α^{2·base}, …]`.
        let alpha_base = *scalar_powers(alpha, base + 1)
            .get(base)
            .ok_or(AkitaError::InvalidProof)?;
        let lane_alpha = |lanes: usize| scalar_powers(alpha_base, lanes);
        Ok(Self {
            a_ratio,
            d_subcolumns: subcolumns(d_lanes)?,
            b_subcolumns: subcolumns(b_lanes)?,
            a_lane_alpha: lane_alpha(a_ratio),
            b_lane_alpha: lane_alpha(b_lanes),
            d_lane_alpha: lane_alpha(d_lanes),
        })
    }

    fn a_spec(&self) -> RoleLaneSpec<'_, E> {
        RoleLaneSpec {
            a_ratio: self.a_ratio,
            role_subcolumns: 1,
            role_lanes: self.a_lane_alpha.len(),
            role_lane_alpha: &self.a_lane_alpha,
        }
    }

    fn b_spec(&self) -> RoleLaneSpec<'_, E> {
        RoleLaneSpec {
            a_ratio: self.a_ratio,
            role_subcolumns: self.b_subcolumns,
            role_lanes: self.b_lane_alpha.len(),
            role_lane_alpha: &self.b_lane_alpha,
        }
    }

    fn d_spec(&self) -> RoleLaneSpec<'_, E> {
        RoleLaneSpec {
            a_ratio: self.a_ratio,
            role_subcolumns: self.d_subcolumns,
            role_lanes: self.d_lane_alpha.len(),
            role_lane_alpha: &self.d_lane_alpha,
        }
    }
}

impl<E: FieldCore> SetupContributionPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        opening_source_len: usize,
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        role_dims: CommitmentRingDims,
        alpha: E,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        Self::prepare_with_mode::<F>(
            level_params,
            opening_batch,
            eq_tau1,
            witness_layout,
            opening_source_len,
            groups,
            full_vec_randomness,
            fold_gadget,
            role_dims,
            alpha,
            SetupPlanMaterialization::Full,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_deferred<F>(
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        opening_source_len: usize,
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        role_dims: CommitmentRingDims,
        alpha: E,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        Self::prepare_with_mode::<F>(
            level_params,
            opening_batch,
            eq_tau1,
            witness_layout,
            opening_source_len,
            groups,
            full_vec_randomness,
            fold_gadget,
            role_dims,
            alpha,
            SetupPlanMaterialization::Deferred,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_with_mode<F>(
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        opening_source_len: usize,
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        role_dims: CommitmentRingDims,
        alpha: E,
        materialization: SetupPlanMaterialization,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
        let lanes = RelationLaneGeometry::new(role_dims, alpha)?;
        let rows = {
            let _span = tracing::info_span!("setup_prepare_validate").entered();
            validate_setup_inputs(level_params, opening_batch, witness_layout, groups)?;
            validate_static_inputs(level_params, opening_batch, &eq_tau1)?
        };
        let (d_rows, d_physical_cols, d_weights) = {
            let _span = tracing::info_span!("setup_prepare_global_geometry").entered();
            let d_rows = level_params.open_commit_matrix.output_rank();
            let d_row_start = rows.checked_sub(d_rows).ok_or_else(|| {
                AkitaError::InvalidSetup("setup D rows exceed relation rows".into())
            })?;
            let d_physical_cols = get_total_d(level_params, opening_batch, groups)?
                .checked_mul(lanes.d_subcolumns)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup D subcolumn width overflow".into())
                })?;
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
                let depth_witness = group.depth_witness(level_params, opening_batch)?;
                let depth_commit = group.depth_commit(level_params, opening_batch)?;
                let depth_open = group.depth_open(level_params, opening_batch)?;
                let log_basis_open = group.log_basis_open(level_params, opening_batch)?;
                let n_a = group.n_a(level_params, opening_batch)?;
                let n_b = group.n_b(level_params, opening_batch)?;
                let t_vector_width = group.t_vector_width(level_params, opening_batch)?;
                let d_col_range = {
                    let range =
                        get_d_col_range(level_params, opening_batch, groups, group.group_id)?;
                    // Expand the physical D range by the per-role subcolumn count
                    // (1 for uniform roles), keeping groups contiguous.
                    let start = range.start.checked_mul(lanes.d_subcolumns).ok_or_else(|| {
                        AkitaError::InvalidSetup("setup D subcolumn range overflow".into())
                    })?;
                    let end = range.end.checked_mul(lanes.d_subcolumns).ok_or_else(|| {
                        AkitaError::InvalidSetup("setup D subcolumn range overflow".into())
                    })?;
                    start..end
                };
                let t_cols = group
                    .num_claims
                    .checked_mul(t_vector_width)
                    .and_then(|cols| cols.checked_mul(lanes.b_subcolumns))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
                let z_cols = num_positions_per_block
                    .checked_mul(depth_witness)
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
                let (e_eq_slice, t_eq_slice, z_eq_slice) = if materialization
                    .materializes_column_slices()
                {
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
                            &lanes.d_spec(),
                        )?
                    };
                    let t_eq_slice = {
                        let _span = tracing::info_span!("setup_prepare_t_weights").entered();
                        setup_t_col_weights::<E>(
                            witness_layout,
                            opening_source_len,
                            group.group_id,
                            num_live_blocks,
                            depth_commit,
                            n_a,
                            group.num_claims,
                            &eq_window,
                            &lanes.b_spec(),
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
                            crate::gadget_row_scalars::<F>(group.depth_fold, log_basis_open);
                        &fold_gadget_storage
                    };
                    let z_range = num_positions_per_block
                        .checked_mul(depth_witness)
                        .ok_or_else(|| AkitaError::InvalidSetup("setup Z range overflow".into()))?;
                    let mut z_eq_slice = vec![E::zero(); z_range];
                    {
                        let _span = tracing::info_span!("setup_prepare_z_weights").entered();
                        setup_z_col_weights::<F, E>(
                            witness_layout,
                            opening_source_len,
                            group.group_id,
                            num_positions_per_block,
                            depth_witness,
                            group.depth_fold,
                            &eq_window,
                            fold_gadget,
                            &lanes.a_spec(),
                            &mut z_eq_slice,
                        )?;
                    }
                    (e_eq_slice, t_eq_slice, z_eq_slice)
                } else {
                    if let Some(fold_gadget) = fold_gadget {
                        if fold_gadget.len() < group.depth_fold {
                            return Err(AkitaError::InvalidSize {
                                expected: group.depth_fold,
                                actual: fold_gadget.len(),
                            });
                        }
                    }
                    (Vec::new(), Vec::new(), Vec::new())
                };

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
                let d_active_cols = group
                    .d_active_cols(level_params, opening_batch)?
                    .checked_mul(lanes.d_subcolumns)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup D active subcolumn overflow".into())
                    })?;
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
        if materialization.builds_scan_segments() {
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

#[derive(Clone, Copy)]
enum SetupPlanMaterialization {
    Full,
    Deferred,
}

impl SetupPlanMaterialization {
    const fn materializes_column_slices(self) -> bool {
        matches!(self, Self::Full)
    }

    const fn builds_scan_segments(self) -> bool {
        matches!(self, Self::Full)
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
