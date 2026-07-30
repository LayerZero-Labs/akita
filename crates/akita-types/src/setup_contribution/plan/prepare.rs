use super::*;
use crate::CommitmentRingDims;
use akita_algebra::ring::scalar_powers_with_stride;

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
        let lane_alpha = |lanes: usize| scalar_powers_with_stride(alpha, common_coeff_count, lanes);
        Ok(Self {
            carrier_lane_count,
            inner_lane_count,
            outer_lane_count,
            opening_lane_count,
            d_subcolumns,
            b_subcolumns,
            inner_alpha: lane_alpha(inner_lane_count)?,
            outer_alpha: lane_alpha(outer_lane_count)?,
            opening_alpha: lane_alpha(opening_lane_count)?,
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
        relation_address: PreparedRelationAddress<E>,
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
            relation_address,
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
        relation_address: PreparedRelationAddress<E>,
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
            relation_address,
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
        relation_address: PreparedRelationAddress<E>,
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
        let common_coeff_count = relation_address_geometry.relation_coefficient_block_len();
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
        // The caller prepares this checked point/window pair once. Stage 2
        // shares it with quotient evaluation and the cached Stage-3 plan.
        let eq_window = relation_address.equality_window();
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
                let (d_spans, b_spans, a_families) = build_setup_contribution_spans(
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
                let column_weights = if materialization.materializes_column_slices() {
                    let e = {
                        let _span = tracing::info_span!("setup_prepare_e_weights").entered();
                        materialize_span_weights::<F, E>(
                            d_col_range.len(),
                            &d_spans,
                            eq_window,
                            &lanes.opening_alpha,
                            None,
                        )?
                    };
                    let t = {
                        let _span = tracing::info_span!("setup_prepare_t_weights").entered();
                        materialize_span_weights::<F, E>(
                            t_cols,
                            &b_spans,
                            eq_window,
                            &lanes.outer_alpha,
                            None,
                        )?
                    };
                    let z = {
                        let _span = tracing::info_span!("setup_prepare_z_weights").entered();
                        materialize_span_weights::<F, E>(
                            z_cols,
                            &a_families,
                            eq_window,
                            &lanes.inner_alpha,
                            Some(group_fold_gadget),
                        )?
                    };
                    SetupContributionColumnWeights::Prepared { e, t, z }
                } else {
                    SetupContributionColumnWeights::Deferred
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
                    column_weights,
                    d_spans,
                    b_spans,
                    a_families,
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
                    b_rows: planned.n_b,
                    b_cols: planned.t_cols,
                    d_active_cols: planned.d_col_range.len(),
                    d_span_count: planned.d_spans.len(),
                    b_span_count: planned.b_spans.len(),
                    // Stage 3 expands each coarse physical A family into one
                    // recurrence term per fold digit. Budget the emitted
                    // stream, not the compact plan metadata.
                    a_span_count: planned
                        .a_families
                        .len()
                        .checked_mul(planned.fold_gadget.len())
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("setup A expanded span count overflow".into())
                        })?,
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
        projection_geometry.ensure_setup_index_evaluation_budget()?;
        let mut plan = SetupContributionPlan {
            groups: dynamic_groups,
            d_rows,
            d_physical_cols,
            d_weights,
            setup_index_terms: Default::default(),
            relation_address,
            relation_address_geometry,
            projection_geometry,
        };
        plan.setup_index_terms = plan.prepare_setup_index_terms()?;
        Ok(plan)
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
    let mut a_families = Vec::new();
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
                            b_column_count,
                            relation_lane_capacity,
                        )?);
                    }
                }
            }
        }

        let witness_column = unit.z_index(
            num_positions_per_block,
            depth_witness,
            group.depth_fold,
            0,
            0,
            0,
        )?;
        let relation_lane_start = witness_column
            .checked_mul(lanes.carrier_lane_count)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A relation address overflow".into()))?;
        a_families.push(SetupContributionSpan::new_fold_family(
            0,
            1,
            relation_lane_start,
            a_relation_stride,
            a_column_count,
            lanes.inner_lane_count,
            group.depth_fold,
            lanes.carrier_lane_count,
            a_column_count,
            relation_lane_capacity,
        )?);
    }

    Ok((d_spans, b_spans, a_families))
}

/// Materialize any identity-weighted span partition whose interleaved streams
/// form contiguous source and destination rectangles.
///
/// This is an address-geometry optimization: it is available to every role
/// shape that degenerates to one lane with unit weight. No caller selects it
/// from a projection ratio.
fn materialize_contiguous_span_partition<E: FieldCore>(
    column_count: usize,
    spans: &[SetupContributionSpan],
    eq_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
) -> Result<Option<Vec<E>>, AkitaError> {
    let mut intervals = Vec::<(usize, usize, usize)>::new();
    intervals.try_reserve(spans.len()).map_err(|_| {
        AkitaError::InvalidSetup("setup contiguous interval allocation failed".into())
    })?;
    let mut index = 0usize;
    while index < spans.len() {
        let base = spans.get(index).ok_or(AkitaError::InvalidProof)?;
        let width = base.setup_column_stride;
        if width == 0
            || base.relation_lane_stride != width
            || base.relation_lane_count != 1
            || base.fold_count != 1
        {
            return Ok(None);
        }
        let family_end = index.checked_add(width).ok_or_else(|| {
            AkitaError::InvalidSetup("setup contiguous family width overflow".into())
        })?;
        let Some(family) = spans.get(index..family_end) else {
            return Ok(None);
        };
        let complete = family.iter().enumerate().all(|(lane, span)| {
            span.setup_column_stride == width
                && span.relation_lane_stride == width
                && span.occurrence_count == base.occurrence_count
                && span.relation_lane_count == 1
                && span.fold_count == 1
                && base.setup_column_start.checked_add(lane) == Some(span.setup_column_start)
                && base.relation_lane_start.checked_add(lane) == Some(span.relation_lane_start)
        });
        if !complete {
            return Ok(None);
        }
        let len = base
            .occurrence_count
            .checked_mul(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup contiguous interval overflow".into()))?;
        intervals.push((base.setup_column_start, base.relation_lane_start, len));
        index = family_end;
    }

    intervals.sort_unstable_by_key(|&(destination, _, _)| destination);
    let mut previous_end = 0usize;
    for &(destination, _, len) in &intervals {
        let end = destination.checked_add(len).ok_or_else(|| {
            AkitaError::InvalidSetup("setup contiguous destination overflow".into())
        })?;
        if destination < previous_end || end > column_count {
            return Ok(None);
        }
        previous_end = end;
    }

    let mut weights = vec![E::zero(); column_count];
    for (destination, source, len) in intervals {
        let end = destination.checked_add(len).ok_or_else(|| {
            AkitaError::InvalidSetup("setup contiguous destination overflow".into())
        })?;
        let output = weights
            .get_mut(destination..end)
            .ok_or(AkitaError::InvalidProof)?;
        eq_window.fill_interval(source, output)?;
    }
    Ok(Some(weights))
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
    if fold_gadget.is_none() && lane_alpha == [E::one()] {
        if let Some(weights) =
            materialize_contiguous_span_partition(column_count, spans, eq_window)?
        {
            return Ok(weights);
        }
    }

    let prepared_spans = spans
        .iter()
        .map(|span| prepare_span_contribution(span, lane_alpha, fold_gadget))
        .collect::<Result<Vec<_>, _>>()?;
    let mut weights = vec![E::zero(); column_count];

    // Dense overlapping families cover the same destination domain. Assign one
    // destination per worker so every overlap is accumulated locally without
    // synchronization, using fold/lane weights prepared once above.
    let dense_overlaps = spans.iter().all(|span| {
        span.setup_column_start == 0
            && span.setup_column_stride == 1
            && span.occurrence_count == column_count
    });
    if dense_overlaps {
        cfg_iter_mut!(weights)
            .enumerate()
            .try_for_each(|(setup_column, destination)| {
                for (span, prepared) in spans.iter().zip(&prepared_spans) {
                    let relation_lane_start = span
                        .relation_lane_stride
                        .checked_mul(setup_column)
                        .and_then(|offset| span.relation_lane_start.checked_add(offset))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup(
                                "setup contribution relation span overflow".into(),
                            )
                        })?;
                    *destination +=
                        prepared_span_contribution(prepared, relation_lane_start, eq_window)?;
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
                let prepared = prepared_spans
                    .get(span_index)
                    .ok_or(AkitaError::InvalidProof)?;
                *destination =
                    prepared_span_contribution(prepared, relation_lane_start, eq_window)?;
                Ok::<_, AkitaError>(())
            })?;
        return Ok(weights);
    }

    for (span, prepared) in spans.iter().zip(&prepared_spans) {
        for occurrence in span.occurrences() {
            let (setup_column, relation_lane_start) = occurrence?;
            let contribution =
                prepared_span_contribution(prepared, relation_lane_start, eq_window)?;
            let destination = weights
                .get_mut(setup_column)
                .ok_or(AkitaError::InvalidProof)?;
            *destination += contribution;
        }
    }
    Ok(weights)
}

fn prepare_span_contribution<F, E>(
    span: &SetupContributionSpan,
    lane_alpha: &[E],
    fold_gadget: Option<&[F]>,
) -> Result<Vec<(usize, E)>, AkitaError>
where
    F: FieldCore,
    E: FieldCore + MulBase<F>,
{
    if span.relation_lane_count != lane_alpha.len()
        || span.fold_count != fold_gadget.map_or(1, <[F]>::len)
    {
        return Err(AkitaError::InvalidSetup(
            "setup contribution span disagrees with role lane geometry".into(),
        ));
    }
    let digit_count = span
        .fold_count
        .checked_mul(span.relation_lane_count)
        .ok_or_else(|| {
            AkitaError::InvalidSetup("setup contribution digit count overflow".into())
        })?;
    let mut digits = Vec::new();
    digits.try_reserve_exact(digit_count).map_err(|_| {
        AkitaError::InvalidSetup("setup contribution digit allocation failed".into())
    })?;
    if let Some(fold_gadget) = fold_gadget {
        for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
            for (lane, &alpha) in lane_alpha.iter().enumerate() {
                let offset = fold_digit
                    .checked_mul(span.fold_lane_stride)
                    .and_then(|offset| offset.checked_add(lane))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup contribution relation lane overflow".into())
                    })?;
                digits.push((offset, -alpha.mul_base(fold)));
            }
        }
    } else {
        digits.extend(lane_alpha.iter().copied().enumerate());
    }
    Ok(digits)
}

fn prepared_span_contribution<E: FieldCore>(
    digits: &[(usize, E)],
    relation_lane_start: usize,
    eq_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
) -> Result<E, AkitaError> {
    digits
        .iter()
        .try_fold(E::zero(), |weight, &(offset, digit_weight)| {
            let relation_lane = relation_lane_start.checked_add(offset).ok_or_else(|| {
                AkitaError::InvalidSetup("setup contribution relation lane overflow".into())
            })?;
            Ok(weight + eq_window.eval(relation_lane) * digit_weight)
        })
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
