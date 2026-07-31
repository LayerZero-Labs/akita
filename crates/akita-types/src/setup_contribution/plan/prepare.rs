use super::*;
use crate::CommitmentRingDims;
use akita_algebra::ring::scalar_powers;

/// Challenge-dependent lane weights over checked relation-address geometry.
///
/// Address counts come from [`RelationAddressGeometry`]; this type owns only
/// the `alpha` powers needed to contract consecutive lanes into one physical
/// setup column.
struct PreparedRelationLanes<E> {
    carrier_lane_count: usize,
    inner_lane_count: usize,
    outer_lane_count: usize,
    opening_lane_count: usize,
    d_subcolumns: usize,
    b_subcolumns: usize,
    inner_alpha: Vec<E>,
    outer_alpha: Vec<E>,
    opening_alpha: Vec<E>,
}

impl<E: FieldCore> PreparedRelationLanes<E> {
    fn new(
        role_dims: CommitmentRingDims,
        common_coeff_count: usize,
        carrier_ring_dimension: usize,
        alpha: E,
    ) -> Result<Self, AkitaError> {
        role_dims.validate_a_carrier()?;
        if common_coeff_count == 0 {
            return Err(AkitaError::InvalidSetup(
                "setup relation lane base must be nonzero".into(),
            ));
        }
        if role_dims == CommitmentRingDims::uniform(common_coeff_count)
            && carrier_ring_dimension == common_coeff_count
        {
            return Ok(Self {
                carrier_lane_count: 1,
                inner_lane_count: 1,
                outer_lane_count: 1,
                opening_lane_count: 1,
                d_subcolumns: 1,
                b_subcolumns: 1,
                inner_alpha: vec![E::one()],
                outer_alpha: vec![E::one()],
                opening_alpha: vec![E::one()],
            });
        }
        let lane_count = |role: RingRole| {
            role_dims
                .dim_for(role)
                .checked_div(common_coeff_count)
                .filter(|count| *count != 0)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(
                        "group role dimension does not decompose over relation lane base".into(),
                    )
                })
        };
        let inner_lane_count = lane_count(RingRole::Inner)?;
        let outer_lane_count = lane_count(RingRole::Outer)?;
        let opening_lane_count = lane_count(RingRole::Opening)?;
        let carrier_lane_count = carrier_ring_dimension
            .checked_div(common_coeff_count)
            .filter(|count| {
                carrier_ring_dimension.is_multiple_of(common_coeff_count) && *count != 0
            })
            .ok_or_else(|| {
                AkitaError::InvalidSetup(
                    "setup witness carrier does not decompose over relation lane base".into(),
                )
            })?;
        let (b_subcolumns, d_subcolumns) =
            SetupProjectionGeometry::a_carrier_subcolumn_counts(role_dims)?;
        let alpha_base = *scalar_powers(alpha, common_coeff_count + 1)
            .get(common_coeff_count)
            .ok_or(AkitaError::InvalidProof)?;
        let lane_alpha = |lanes: usize| scalar_powers(alpha_base, lanes);
        Ok(Self {
            carrier_lane_count,
            inner_lane_count,
            outer_lane_count,
            opening_lane_count,
            d_subcolumns,
            b_subcolumns,
            inner_alpha: lane_alpha(inner_lane_count),
            outer_alpha: lane_alpha(outer_lane_count),
            opening_alpha: lane_alpha(opening_lane_count),
        })
    }
}

impl<E: FieldCore> SetupContributionPlan<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prepare<F>(
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        eq_tau1: std::sync::Arc<[E]>,
        witness_layout: &WitnessLayout,
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        relation_address_geometry: RelationAddressGeometry,
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
            groups,
            full_vec_randomness,
            fold_gadget,
            relation_address_geometry,
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
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        relation_address_geometry: RelationAddressGeometry,
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
            groups,
            full_vec_randomness,
            fold_gadget,
            relation_address_geometry,
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
        groups: &[SetupContributionGroupInputs],
        full_vec_randomness: &[E],
        fold_gadget: Option<&[F]>,
        relation_address_geometry: RelationAddressGeometry,
        alpha: E,
        materialization: SetupPlanMaterialization,
    ) -> Result<SetupContributionPlan<E>, AkitaError>
    where
        F: FieldCore + CanonicalField,
        E: MulBase<F>,
    {
        let _span = tracing::info_span!("setup_prepare_plan").entered();
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
        let common_coeff_count = relation_address_geometry.common_relation_witness_coeff_count();
        let group_geometry = groups
            .iter()
            .map(|group| {
                let role_dims = level_params.group_role_dims(opening_batch, group.group_id)?;
                let lanes = PreparedRelationLanes::new(
                    role_dims,
                    common_coeff_count,
                    relation_address_geometry.carrier_ring_dimension(),
                    alpha,
                )?;
                let raw_d_cols = group.d_active_cols(level_params, opening_batch)?;
                Ok((role_dims, lanes, raw_d_cols))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let mut d_cursor = 0usize;
        let d_col_ranges = group_geometry
            .iter()
            .map(|(_, lanes, raw_d_cols)| {
                let width = raw_d_cols.checked_mul(lanes.d_subcolumns).ok_or_else(|| {
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
            let d_row_start = rows.checked_sub(d_rows).ok_or_else(|| {
                AkitaError::InvalidSetup("setup D rows exceed relation rows".into())
            })?;
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
            .zip(&group_geometry)
            .zip(&d_col_ranges)
            .map(|((group, (role_dims, lanes, _)), d_col_range)| {
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
                let n_b = group.n_b(level_params, opening_batch)?;
                let t_vector_width = group.t_vector_width(level_params, opening_batch)?;
                let d_col_range = d_col_range.clone();
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
                let consistency_weight = *eq_tau1
                    .get(level_params.consistency_row_index(opening_batch, group.group_id)?)
                    .ok_or(AkitaError::InvalidProof)?;
                drop(geometry_span);
                let (d_spans, b_spans, a_spans) = build_setup_contribution_spans(
                    witness_layout,
                    group,
                    num_live_blocks,
                    num_positions_per_block,
                    depth_witness,
                    depth_commit,
                    depth_open,
                    n_a,
                    d_col_range.len(),
                    t_cols,
                    z_cols,
                    relation_address_geometry.relation_lane_capacity(),
                    lanes,
                )?;
                let fold_gadget_storage;
                let group_fold_gadget = if let Some(fold_gadget) = fold_gadget {
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
                let fold_gadget: std::sync::Arc<[E]> = group_fold_gadget
                    .iter()
                    .take(group.depth_fold)
                    .copied()
                    .map(|fold| E::one().mul_base(fold))
                    .collect::<Vec<_>>()
                    .into();
                let (e_eq_slice, t_eq_slice, z_eq_slice) =
                    if materialization.materializes_column_slices() {
                        let e_eq_slice = {
                            let _span = tracing::info_span!("setup_prepare_e_weights").entered();
                            if lanes.carrier_lane_count == 1
                                && lanes.opening_lane_count == 1
                                && lanes.d_subcolumns == 1
                            {
                                materialize_uniform_role_weights(
                                    witness_layout,
                                    group.group_id,
                                    num_live_blocks,
                                    group.num_claims,
                                    depth_open,
                                    d_col_range.len(),
                                    relation_address_geometry.relation_lane_capacity(),
                                    &eq_window,
                                    RingRole::Opening,
                                )?
                            } else {
                                materialize_span_weights::<F, E>(
                                    d_col_range.len(),
                                    &d_spans,
                                    &eq_window,
                                    &lanes.opening_alpha,
                                    None,
                                )?
                            }
                        };
                        let t_eq_slice = {
                            let _span = tracing::info_span!("setup_prepare_t_weights").entered();
                            if lanes.carrier_lane_count == 1
                                && lanes.outer_lane_count == 1
                                && lanes.b_subcolumns == 1
                            {
                                let columns_per_block =
                                    n_a.checked_mul(depth_commit).ok_or_else(|| {
                                        AkitaError::InvalidSetup(
                                            "setup B columns per block overflow".into(),
                                        )
                                    })?;
                                materialize_uniform_role_weights(
                                    witness_layout,
                                    group.group_id,
                                    num_live_blocks,
                                    group.num_claims,
                                    columns_per_block,
                                    t_cols,
                                    relation_address_geometry.relation_lane_capacity(),
                                    &eq_window,
                                    RingRole::Outer,
                                )?
                            } else {
                                materialize_span_weights::<F, E>(
                                    t_cols,
                                    &b_spans,
                                    &eq_window,
                                    &lanes.outer_alpha,
                                    None,
                                )?
                            }
                        };
                        let z_eq_slice = {
                            let _span = tracing::info_span!("setup_prepare_z_weights").entered();
                            if lanes.carrier_lane_count == 1
                                && lanes.inner_lane_count == 1
                                && lanes.d_subcolumns == 1
                            {
                                materialize_uniform_inner_weights(
                                    witness_layout,
                                    group.group_id,
                                    num_positions_per_block,
                                    depth_witness,
                                    group.depth_fold,
                                    relation_address_geometry.relation_lane_capacity(),
                                    &eq_window,
                                    group_fold_gadget,
                                )?
                            } else {
                                materialize_span_weights::<F, E>(
                                    z_cols,
                                    &a_spans,
                                    &eq_window,
                                    &lanes.inner_alpha,
                                    Some(group_fold_gadget),
                                )?
                            }
                        };
                        (e_eq_slice, t_eq_slice, z_eq_slice)
                    } else {
                        (Vec::new(), Vec::new(), Vec::new())
                    };

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
                    t_cols,
                    z_cols,
                    n_a,
                    n_b,
                    required: 0,
                    segments: Vec::new().into(),
                    a_row_weights,
                    b_weights,
                    fold_gadget,
                    e_eq_slice,
                    t_eq_slice,
                    z_eq_slice,
                    d_spans,
                    b_spans,
                    a_spans,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_groups = dynamic_groups
            .iter()
            .zip(groups)
            .map(|(planned, group)| {
                Ok(SetupProjectionGroupGeometry {
                    role_dims: planned.role_dims,
                    a_rows: planned.n_a,
                    a_cols: planned.z_cols,
                    b_rows: planned.n_b,
                    b_cols: planned.t_cols,
                    d_active_cols: planned.d_col_range.len(),
                    ownership_units: witness_layout.units_for_group(group.group_id)?.len(),
                    depth_fold: group.depth_fold,
                })
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let projection_geometry = crate::SetupProjectionGeometry::from_groups(
            relation_address_geometry.role_dims(),
            d_rows,
            d_physical_cols,
            &projection_groups,
        )?;
        if materialization.builds_scan_segments() {
            for group in &mut dynamic_groups {
                let base = projection_geometry.base_ring_dim();
                group.set_projection_ratios(base)?;
                group.refresh_segments(
                    &d_weights,
                    d_rows,
                    d_physical_cols,
                    group.a_ratio,
                    group.b_ratio,
                    group.d_ratio,
                )?;
            }
        } else {
            let base = projection_geometry.base_ring_dim();
            for group in &mut dynamic_groups {
                group.set_projection_ratios(base)?;
            }
        }
        Ok(SetupContributionPlan {
            groups: dynamic_groups,
            d_rows,
            d_physical_cols,
            d_weights,
            address_point: full_vec_randomness.to_vec().into(),
            relation_address_geometry,
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

    /// Canonical relation-address geometry used by every setup contribution
    /// span.
    #[must_use]
    pub const fn relation_address_geometry(&self) -> RelationAddressGeometry {
        self.relation_address_geometry
    }
}

type GroupContributionSpans = (
    Vec<SetupContributionSpan>,
    Vec<SetupContributionSpan>,
    Vec<SetupContributionSpan>,
);

#[allow(clippy::too_many_arguments)]
fn build_setup_contribution_spans<E: FieldCore>(
    witness_layout: &WitnessLayout,
    group: &SetupContributionGroupInputs,
    num_live_blocks: usize,
    num_positions_per_block: usize,
    depth_witness: usize,
    depth_commit: usize,
    depth_open: usize,
    n_a: usize,
    d_column_count: usize,
    b_column_count: usize,
    a_column_count: usize,
    relation_lane_capacity: usize,
    lanes: &PreparedRelationLanes<E>,
) -> Result<GroupContributionSpans, AkitaError> {
    let mut d_spans = Vec::new();
    let mut b_spans = Vec::new();
    let mut a_spans = Vec::new();
    let d_setup_stride = lanes
        .d_subcolumns
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D column stride overflow".into()))?;
    let d_relation_stride = depth_open
        .checked_mul(lanes.carrier_lane_count)
        .ok_or_else(|| AkitaError::InvalidSetup("setup D relation stride overflow".into()))?;
    let b_setup_stride = n_a
        .checked_mul(depth_commit)
        .and_then(|stride| stride.checked_mul(lanes.b_subcolumns))
        .ok_or_else(|| AkitaError::InvalidSetup("setup B column stride overflow".into()))?;
    let b_relation_stride = n_a
        .checked_mul(depth_commit)
        .and_then(|stride| stride.checked_mul(lanes.carrier_lane_count))
        .ok_or_else(|| AkitaError::InvalidSetup("setup B relation stride overflow".into()))?;
    let a_relation_stride = group
        .depth_fold
        .checked_mul(lanes.carrier_lane_count)
        .ok_or_else(|| AkitaError::InvalidSetup("setup A relation stride overflow".into()))?;

    for unit in witness_layout.units_for_group(group.group_id)? {
        for claim in 0..group.num_claims {
            for subcolumn in 0..lanes.d_subcolumns {
                for digit in 0..depth_open {
                    let setup_column_start = claim
                        .checked_mul(num_live_blocks)
                        .and_then(|base| base.checked_add(unit.global_block_start()))
                        .and_then(|base| base.checked_mul(lanes.d_subcolumns))
                        .and_then(|base| base.checked_add(subcolumn))
                        .and_then(|base| base.checked_mul(depth_open))
                        .and_then(|base| base.checked_add(digit))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("setup D column address overflow".into())
                        })?;
                    let witness_column = unit.e_index(
                        group.num_claims,
                        depth_open,
                        claim,
                        unit.global_block_start(),
                        digit,
                    )?;
                    let relation_lane_start = witness_column
                        .checked_mul(lanes.carrier_lane_count)
                        .and_then(|base| {
                            subcolumn
                                .checked_mul(lanes.opening_lane_count)
                                .and_then(|offset| base.checked_add(offset))
                        })
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("setup D relation address overflow".into())
                        })?;
                    d_spans.push(SetupContributionSpan::new(
                        setup_column_start,
                        d_setup_stride,
                        relation_lane_start,
                        d_relation_stride,
                        unit.num_live_blocks(),
                        lanes.opening_lane_count,
                        None,
                        d_column_count,
                        relation_lane_capacity,
                    )?);
                }
            }

            for a_row in 0..n_a {
                for digit in 0..depth_commit {
                    for subcolumn in 0..lanes.b_subcolumns {
                        let setup_column_start = claim
                            .checked_mul(num_live_blocks)
                            .and_then(|base| base.checked_add(unit.global_block_start()))
                            .and_then(|base| base.checked_mul(n_a))
                            .and_then(|base| base.checked_add(a_row))
                            .and_then(|base| base.checked_mul(depth_commit))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|base| base.checked_mul(lanes.b_subcolumns))
                            .and_then(|base| base.checked_add(subcolumn))
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("setup B column address overflow".into())
                            })?;
                        let witness_column = unit.t_index(
                            group.num_claims,
                            n_a,
                            depth_commit,
                            claim,
                            unit.global_block_start(),
                            a_row,
                            digit,
                        )?;
                        let relation_lane_start = witness_column
                            .checked_mul(lanes.carrier_lane_count)
                            .and_then(|base| {
                                subcolumn
                                    .checked_mul(lanes.outer_lane_count)
                                    .and_then(|offset| base.checked_add(offset))
                            })
                            .ok_or_else(|| {
                                AkitaError::InvalidSetup("setup B relation address overflow".into())
                            })?;
                        b_spans.push(SetupContributionSpan::new(
                            setup_column_start,
                            b_setup_stride,
                            relation_lane_start,
                            b_relation_stride,
                            unit.num_live_blocks(),
                            lanes.outer_lane_count,
                            None,
                            b_column_count,
                            relation_lane_capacity,
                        )?);
                    }
                }
            }
        }

        for fold_digit in 0..group.depth_fold {
            let witness_column = unit.z_index(
                num_positions_per_block,
                depth_witness,
                group.depth_fold,
                0,
                0,
                fold_digit,
            )?;
            let relation_lane_start = witness_column
                .checked_mul(lanes.carrier_lane_count)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup A relation address overflow".into())
                })?;
            a_spans.push(SetupContributionSpan::new(
                0,
                1,
                relation_lane_start,
                a_relation_stride,
                a_column_count,
                lanes.inner_lane_count,
                Some(fold_digit),
                a_column_count,
                relation_lane_capacity,
            )?);
        }
    }

    Ok((d_spans, b_spans, a_spans))
}

/// Materialize the original contiguous-column path when the relation carrier
/// and the selected setup role are the same ring. In that geometry each
/// physical setup column maps to exactly one witness column, so filling whole
/// unit intervals avoids compiling one span job per column.
#[allow(clippy::too_many_arguments)]
#[inline]
fn materialize_uniform_role_weights<E: FieldCore>(
    witness_layout: &WitnessLayout,
    group_id: usize,
    num_live_blocks: usize,
    num_claims: usize,
    columns_per_block: usize,
    column_count: usize,
    relation_lane_capacity: usize,
    eq_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    role: RingRole,
) -> Result<Vec<E>, AkitaError> {
    if !matches!(role, RingRole::Outer | RingRole::Opening) {
        return Err(AkitaError::InvalidSetup(
            "uniform setup interval requires the B or D role".into(),
        ));
    }
    let mut weights = vec![E::zero(); column_count];
    for claim in 0..num_claims {
        for unit in witness_layout.units_for_group(group_id)? {
            let unit_width = unit
                .num_live_blocks()
                .checked_mul(columns_per_block)
                .ok_or_else(|| AkitaError::InvalidSetup("setup unit width overflow".into()))?;
            let expected_source_len = num_claims
                .checked_mul(unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup unit shape overflow".into()))?;
            let source_range = match role {
                RingRole::Outer => unit.t_range(),
                RingRole::Opening => unit.e_range(),
                RingRole::Inner => {
                    return Err(AkitaError::InvalidSetup(
                        "uniform setup interval does not support the A role".into(),
                    ));
                }
            };
            if source_range.len() != expected_source_len {
                return Err(AkitaError::InvalidSetup(
                    "setup unit shape disagrees with resolved witness range".into(),
                ));
            }
            let claim_offset = claim
                .checked_mul(unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup source offset overflow".into()))?;
            let source_start = source_range
                .start
                .checked_add(claim_offset)
                .ok_or_else(|| AkitaError::InvalidSetup("setup source interval overflow".into()))?;
            let source_end = source_start
                .checked_add(unit_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup source interval overflow".into()))?;
            if source_end > source_range.end || source_end > relation_lane_capacity {
                return Err(AkitaError::InvalidInput(
                    "setup source interval is out of range".into(),
                ));
            }

            let destination_start = claim
                .checked_mul(num_live_blocks)
                .and_then(|base| base.checked_add(unit.global_block_start()))
                .and_then(|block| block.checked_mul(columns_per_block))
                .ok_or_else(|| {
                    AkitaError::InvalidSetup("setup destination interval overflow".into())
                })?;
            let destination_end = destination_start.checked_add(unit_width).ok_or_else(|| {
                AkitaError::InvalidSetup("setup destination interval overflow".into())
            })?;
            let destination = weights
                .get_mut(destination_start..destination_end)
                .ok_or(AkitaError::InvalidProof)?;
            eq_window.fill_interval(source_start, destination)?;
        }
    }
    Ok(weights)
}

/// Materialize A-role weights directly when one physical setup column maps to
/// one relation lane. This keeps the established uniform recursive kernel
/// while mixed dimensions continue through the span evaluator below.
#[allow(clippy::too_many_arguments)]
fn materialize_uniform_inner_weights<F, E>(
    witness_layout: &WitnessLayout,
    group_id: usize,
    num_positions_per_block: usize,
    depth_witness: usize,
    depth_fold: usize,
    relation_lane_capacity: usize,
    eq_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    fold_gadget: &[F],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    if fold_gadget.len() < depth_fold {
        return Err(AkitaError::InvalidSetup(
            "setup A weights have malformed fold geometry".into(),
        ));
    }
    let z_cols = num_positions_per_block
        .checked_mul(depth_witness)
        .ok_or_else(|| AkitaError::InvalidSetup("setup A width overflow".into()))?;
    let units = witness_layout.units_for_group(group_id)?;
    let mut weights = vec![E::zero(); z_cols];
    cfg_iter_mut!(weights)
        .enumerate()
        .try_for_each(|(column, destination)| {
            let position = column / depth_witness;
            let witness_digit = column % depth_witness;
            let mut weight = E::zero();
            for unit in &units {
                for (fold_digit, &fold) in fold_gadget.iter().enumerate().take(depth_fold) {
                    let witness_index = unit.z_index(
                        num_positions_per_block,
                        depth_witness,
                        depth_fold,
                        position,
                        witness_digit,
                        fold_digit,
                    )?;
                    if witness_index >= relation_lane_capacity {
                        return Err(AkitaError::InvalidInput(
                            "setup A relation address is out of range".into(),
                        ));
                    }
                    weight -= eq_window.eval(witness_index).mul_base(fold);
                }
            }
            *destination = weight;
            Ok(())
        })?;
    Ok(weights)
}

fn materialize_span_weights<F, E>(
    column_count: usize,
    spans: &[SetupContributionSpan],
    eq_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    lane_alpha: &[E],
    fold_gadget: Option<&[F]>,
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    let mut weights = vec![E::zero(); column_count];

    // Dense overlapping families (notably one span per fold digit/unit) all
    // cover the same destination domain. Assign one destination per worker so
    // every overlap is accumulated locally without synchronization.
    let dense_overlaps = spans.iter().all(|span| {
        span.setup_column_start == 0
            && span.setup_column_stride == 1
            && span.occurrence_count == column_count
    });
    if dense_overlaps {
        cfg_iter_mut!(weights)
            .enumerate()
            .try_for_each(|(setup_column, destination)| {
                for span in spans {
                    if span.relation_lane_count != lane_alpha.len() {
                        return Err(AkitaError::InvalidSetup(
                            "setup contribution span disagrees with role lane geometry".into(),
                        ));
                    }
                    let relation_lane_start = span
                        .relation_lane_stride
                        .checked_mul(setup_column)
                        .and_then(|offset| span.relation_lane_start.checked_add(offset))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "setup contribution relation span overflow".into(),
                            )
                        })?;
                    *destination += materialized_span_contribution(
                        span,
                        relation_lane_start,
                        eq_window,
                        lane_alpha,
                        fold_gadget,
                    )?;
                }
                Ok::<_, AkitaError>(())
            })?;
        return Ok(weights);
    }

    // For a disjoint family, first compile the affine spans into one checked
    // job per destination and then evaluate those jobs in parallel. The test is
    // purely geometric, so mixed physical subcolumns use the same path.
    let mut jobs = vec![None; column_count];
    let mut disjoint = true;
    'spans: for (span_index, span) in spans.iter().enumerate() {
        if span.relation_lane_count != lane_alpha.len() {
            return Err(AkitaError::InvalidSetup(
                "setup contribution span disagrees with role lane geometry".into(),
            ));
        }
        for addresses in span.occurrences() {
            let (setup_column, relation_lane_start) = addresses?;
            let slot = jobs.get_mut(setup_column).ok_or(AkitaError::InvalidProof)?;
            if slot.is_some() {
                disjoint = false;
                break 'spans;
            }
            *slot = Some((span_index, relation_lane_start));
        }
    }
    if disjoint {
        cfg_iter_mut!(weights)
            .enumerate()
            .try_for_each(|(setup_column, destination)| {
                let Some(&(span_index, relation_lane_start)) =
                    jobs.get(setup_column).and_then(Option::as_ref)
                else {
                    return Ok(());
                };
                let span = spans.get(span_index).ok_or(AkitaError::InvalidProof)?;
                *destination = materialized_span_contribution(
                    span,
                    relation_lane_start,
                    eq_window,
                    lane_alpha,
                    fold_gadget,
                )?;
                Ok::<_, AkitaError>(())
            })?;
        return Ok(weights);
    }

    for span in spans {
        if span.relation_lane_count != lane_alpha.len() {
            return Err(AkitaError::InvalidSetup(
                "setup contribution span disagrees with role lane geometry".into(),
            ));
        }
        for occurrence in span.occurrences() {
            let (setup_column, relation_lane_start) = occurrence?;
            let contribution = materialized_span_contribution(
                span,
                relation_lane_start,
                eq_window,
                lane_alpha,
                fold_gadget,
            )?;
            let destination = weights
                .get_mut(setup_column)
                .ok_or(AkitaError::InvalidProof)?;
            *destination += contribution;
        }
    }
    Ok(weights)
}

fn materialized_span_contribution<F, E>(
    span: &SetupContributionSpan,
    relation_lane_start: usize,
    eq_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    lane_alpha: &[E],
    fold_gadget: Option<&[F]>,
) -> Result<E, AkitaError>
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    let weight = lane_alpha
        .iter()
        .enumerate()
        .try_fold(E::zero(), |weight, (lane, &alpha)| {
            let relation_lane = relation_lane_start.checked_add(lane).ok_or_else(|| {
                AkitaError::InvalidSetup("setup contribution relation lane overflow".into())
            })?;
            Ok::<_, AkitaError>(weight + eq_window.eval(relation_lane) * alpha)
        })?;
    if let Some(fold_digit) = span.fold_digit {
        let fold = fold_gadget
            .and_then(|gadget| gadget.get(fold_digit))
            .copied()
            .ok_or(AkitaError::InvalidProof)?;
        Ok(-weight.mul_base(fold))
    } else {
        Ok(weight)
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
