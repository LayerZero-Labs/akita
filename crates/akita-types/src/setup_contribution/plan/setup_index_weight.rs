use super::*;
use akita_algebra::{
    offset_eq::{eval_weighted_compact_pair_eq, WeightedCompactPairTerm, MAX_COMPACT_STRIDE_TERMS},
    ring::scalar_powers,
};

struct GroupSetupIndexWeights<E> {
    projection_scales: [Vec<E>; 3],
    relation_lane_powers: [Vec<E>; 3],
}

impl<E: FieldCore> SetupContributionPlan<E> {
    /// Materialize the dense packed setup-position weight vector.
    pub fn materialize_setup_index_weights(&self, alpha: E) -> Result<Vec<E>, AkitaError> {
        // Both power families depend only on the group. Hoist their allocation
        // and exponentiation out of the potentially million-element setup loop.
        let group_weights = self
            .groups
            .iter()
            .map(|group| GroupSetupIndexWeights {
                projection_scales: self.group_projection_scales(group, alpha),
                relation_lane_powers: self.group_relation_lane_powers(group, alpha),
            })
            .collect::<Vec<_>>();
        (0..self.required())
            .map(|setup_idx| self.setup_index_weight_at(setup_idx, &group_weights))
            .collect()
    }

    /// Evaluate the packed setup-position weight polynomial at
    /// `rho_setup_idx` directly from its canonical contribution spans.
    pub fn evaluate_setup_index_weight_mle(
        &self,
        rho_setup_idx: &[E],
        alpha: E,
    ) -> Result<E, AkitaError> {
        let expected = self.projection_geometry.setup_index_len().trailing_zeros() as usize;
        if rho_setup_idx.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: rho_setup_idx.len(),
            });
        }
        self.projection_geometry.ensure_evaluation_budget()?;
        let _span = tracing::info_span!("stage3_setup_index_weight_mle").entered();
        let mut terms = Vec::new();
        for group in &self.groups {
            let scales = self.group_projection_scales(group, alpha);
            let lane_powers = self.group_relation_lane_powers(group, alpha);
            self.append_d_span_terms(group, &scales[2], &lane_powers[2], &mut terms)?;
            self.append_b_span_terms(group, &scales[1], &lane_powers[1], &mut terms)?;
            self.append_a_span_terms(group, &scales[0], &lane_powers[0], &mut terms)?;
        }
        eval_weighted_compact_pair_eq(rho_setup_idx, &self.address_point, &terms)
    }

    fn group_projection_scales(
        &self,
        group: &SetupContributionGroupPlan<E>,
        alpha: E,
    ) -> [Vec<E>; 3] {
        let role_scales = |role_dim: usize| {
            scalar_powers(alpha, role_dim)
                .chunks(self.projection_geometry.base_ring_dim())
                .map(|chunk| chunk[0])
                .collect()
        };
        [
            role_scales(group.role_dims.d_a()),
            role_scales(group.role_dims.d_b()),
            role_scales(group.role_dims.d_d()),
        ]
    }

    fn setup_index_weight_at(
        &self,
        setup_idx: usize,
        group_weights: &[GroupSetupIndexWeights<E>],
    ) -> Result<E, AkitaError> {
        let geometry = self.projection_geometry;
        if setup_idx >= geometry.required() {
            return Err(AkitaError::InvalidSize {
                expected: geometry.required(),
                actual: setup_idx,
            });
        }
        let mut weight = E::zero();
        for (group, weights) in self.groups.iter().zip(group_weights) {
            let scales = &weights.projection_scales;
            let lane_powers = &weights.relation_lane_powers;
            let d_idx = setup_idx / group.d_ratio;
            let d_footprint = self
                .d_rows
                .checked_mul(self.d_physical_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D footprint overflow".into()))?;
            if d_idx < d_footprint {
                let d_col = d_idx % self.d_physical_cols;
                let d_row = d_idx / self.d_physical_cols;
                if group.d_col_range.contains(&d_col) {
                    weight += scales[2][setup_idx % group.d_ratio]
                        * self.d_weights[d_row]
                        * role_column_weight_or_materialized(
                            &group.d_spans,
                            &group.e_eq_slice,
                            d_col - group.d_col_range.start,
                            &self.eq_window,
                            &lane_powers[2],
                            None,
                        )?;
                }
            }

            let b_idx = setup_idx / group.b_ratio;
            let b_footprint = group
                .n_b
                .checked_mul(group.t_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B footprint overflow".into()))?;
            if b_idx < b_footprint {
                let b_col = b_idx % group.t_cols;
                let b_row = b_idx / group.t_cols;
                weight += scales[1][setup_idx % group.b_ratio]
                    * group.b_weights[b_row]
                    * role_column_weight_or_materialized(
                        &group.b_spans,
                        &group.t_eq_slice,
                        b_col,
                        &self.eq_window,
                        &lane_powers[1],
                        None,
                    )?;
            }

            let a_idx = setup_idx / group.a_ratio;
            let a_footprint = group
                .n_a
                .checked_mul(group.z_cols)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A footprint overflow".into()))?;
            if a_idx < a_footprint {
                let a_col = a_idx % group.z_cols;
                let a_row = a_idx / group.z_cols;
                weight += scales[0][setup_idx % group.a_ratio]
                    * group.a_row_weights[a_row]
                    * role_column_weight_or_materialized(
                        &group.a_spans,
                        &group.z_eq_slice,
                        a_col,
                        &self.eq_window,
                        &lane_powers[0],
                        Some(&group.fold_gadget),
                    )?;
            }
        }
        Ok(weight)
    }

    fn append_d_span_terms(
        &self,
        group: &SetupContributionGroupPlan<E>,
        scales: &[E],
        lane_powers: &[E],
        terms: &mut Vec<WeightedCompactPairTerm<E>>,
    ) -> Result<(), AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(());
        }
        for span in &group.d_spans {
            super::structured::ensure_lane_count(span, lane_powers)?;
            let setup_col = group
                .d_col_range
                .start
                .checked_add(span.setup_column_start)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D address overflow".into()))?;
            let setup_stride = group
                .d_ratio
                .checked_mul(span.setup_column_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D stride overflow".into()))?;
            for (row, &row_weight) in self.d_weights.iter().enumerate() {
                for (lane, &scale) in scales.iter().enumerate() {
                    let setup_index = projected_setup_offset(
                        group.d_ratio,
                        self.d_physical_cols,
                        row,
                        setup_col,
                        lane,
                    )?;
                    for (relation_lane, &lane_power) in lane_powers.iter().enumerate() {
                        push_weighted_term(
                            terms,
                            WeightedCompactPairTerm {
                                left_offset: setup_index,
                                left_stride: setup_stride,
                                right_offset: relation_lane_start(span, relation_lane)?,
                                right_stride: span.relation_lane_stride,
                                len: span.occurrence_count,
                                weight: row_weight * scale * lane_power,
                            },
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn append_b_span_terms(
        &self,
        group: &SetupContributionGroupPlan<E>,
        scales: &[E],
        lane_powers: &[E],
        terms: &mut Vec<WeightedCompactPairTerm<E>>,
    ) -> Result<(), AkitaError> {
        for span in &group.b_spans {
            super::structured::ensure_lane_count(span, lane_powers)?;
            let setup_stride = group
                .b_ratio
                .checked_mul(span.setup_column_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B stride overflow".into()))?;
            for (row, &row_weight) in group.b_weights.iter().enumerate() {
                for (lane, &scale) in scales.iter().enumerate() {
                    let setup_index = projected_setup_offset(
                        group.b_ratio,
                        group.t_cols,
                        row,
                        span.setup_column_start,
                        lane,
                    )?;
                    for (relation_lane, &lane_power) in lane_powers.iter().enumerate() {
                        push_weighted_term(
                            terms,
                            WeightedCompactPairTerm {
                                left_offset: setup_index,
                                left_stride: setup_stride,
                                right_offset: relation_lane_start(span, relation_lane)?,
                                right_stride: span.relation_lane_stride,
                                len: span.occurrence_count,
                                weight: row_weight * scale * lane_power,
                            },
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn append_a_span_terms(
        &self,
        group: &SetupContributionGroupPlan<E>,
        scales: &[E],
        lane_powers: &[E],
        terms: &mut Vec<WeightedCompactPairTerm<E>>,
    ) -> Result<(), AkitaError> {
        for span in &group.a_spans {
            super::structured::ensure_lane_count(span, lane_powers)?;
            let fold = *group
                .fold_gadget
                .get(span.fold_digit.ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?;
            let setup_stride = group
                .a_ratio
                .checked_mul(span.setup_column_stride)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A stride overflow".into()))?;
            for (row, &row_weight) in group.a_row_weights.iter().enumerate() {
                for (lane, &scale) in scales.iter().enumerate() {
                    let setup_index = projected_setup_offset(
                        group.a_ratio,
                        group.z_cols,
                        row,
                        span.setup_column_start,
                        lane,
                    )?;
                    for (relation_lane, &lane_power) in lane_powers.iter().enumerate() {
                        push_weighted_term(
                            terms,
                            WeightedCompactPairTerm {
                                left_offset: setup_index,
                                left_stride: setup_stride,
                                right_offset: relation_lane_start(span, relation_lane)?,
                                right_stride: span.relation_lane_stride,
                                len: span.occurrence_count,
                                weight: -(row_weight * scale * fold * lane_power),
                            },
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn push_weighted_term<E: FieldCore>(
    terms: &mut Vec<WeightedCompactPairTerm<E>>,
    term: WeightedCompactPairTerm<E>,
) -> Result<(), AkitaError> {
    if terms.len() >= MAX_COMPACT_STRIDE_TERMS {
        return Err(AkitaError::InvalidSize {
            expected: MAX_COMPACT_STRIDE_TERMS,
            actual: terms.len().saturating_add(1),
        });
    }
    if terms.len() == terms.capacity() {
        terms.try_reserve(1).map_err(|_| {
            AkitaError::InvalidSetup("setup evaluation term allocation failed".into())
        })?;
    }
    terms.push(term);
    Ok(())
}

fn role_column_weight_or_materialized<E: FieldCore>(
    spans: &[SetupContributionSpan],
    materialized: &[E],
    column: usize,
    equality_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    lane_powers: &[E],
    fold_gadget: Option<&[E]>,
) -> Result<E, AkitaError> {
    if !materialized.is_empty() {
        return materialized
            .get(column)
            .copied()
            .ok_or(AkitaError::InvalidProof);
    }
    if spans.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "setup contribution plan is missing canonical spans".into(),
        ));
    }
    role_column_weight(spans, column, equality_window, lane_powers, fold_gadget)
}

fn role_column_weight<E: FieldCore>(
    spans: &[SetupContributionSpan],
    column: usize,
    equality_window: &akita_algebra::offset_eq::OffsetEqWindow<E>,
    lane_powers: &[E],
    fold_gadget: Option<&[E]>,
) -> Result<E, AkitaError> {
    let mut weight = E::zero();
    for span in spans {
        super::structured::ensure_lane_count(span, lane_powers)?;
        let Some(relation_lane_start) = span.relation_lane_start_for_setup_column(column)? else {
            continue;
        };
        let lane_equality =
            lane_powers
                .iter()
                .copied()
                .enumerate()
                .try_fold(E::zero(), |sum, (lane, power)| {
                    Ok::<_, AkitaError>(
                        sum + equality_window
                            .eval(checked_add_relation_lane(relation_lane_start, lane)?)
                            * power,
                    )
                })?;
        if let Some(fold_digit) = span.fold_digit {
            weight -= lane_equality
                * *fold_gadget
                    .and_then(|gadget| gadget.get(fold_digit))
                    .ok_or(AkitaError::InvalidProof)?;
        } else {
            weight += lane_equality;
        }
    }
    Ok(weight)
}

fn relation_lane_start(span: &SetupContributionSpan, lane: usize) -> Result<usize, AkitaError> {
    checked_add_relation_lane(span.relation_lane_start, lane)
}

fn checked_add_relation_lane(start: usize, lane: usize) -> Result<usize, AkitaError> {
    start
        .checked_add(lane)
        .ok_or_else(|| AkitaError::InvalidSetup("setup contribution relation lane overflow".into()))
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
    width
        .checked_mul(row)
        .and_then(|base| base.checked_add(column))
        .and_then(|logical| ratio.checked_mul(logical))
        .and_then(|base| base.checked_add(lane))
        .ok_or_else(|| AkitaError::InvalidSetup("setup projected address overflow".into()))
}
