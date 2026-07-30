use super::*;
use akita_algebra::{
    offset_eq::{eval_weighted_compact_pair_eq, WeightedCompactPairTerm, MAX_COMPACT_STRIDE_TERMS},
    ring::{evaluate_power_sequence_mle, scalar_powers},
};
use akita_field::fft::field_pow;
use std::collections::BTreeMap;

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

    /// Evaluate the packed setup-position weight polynomial from its canonical
    /// contribution spans.
    ///
    /// For a role dimension `d_R`, let `q = d_R / base_ring_dim` and
    /// `beta = alpha^base_ring_dim`. The `q` setup subrings and `q` relation
    /// lanes carry the separable weight `beta^u * beta^v`. Their low Boolean
    /// coordinates are evaluated once with [`evaluate_power_sequence_mle`],
    /// leaving one compact term per semantic span and matrix row.
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
        self.projection_geometry
            .ensure_setup_index_evaluation_budget()?;
        let _span = tracing::info_span!("stage3_setup_index_weight_mle").entered();
        let mut terms_by_projection_ratio =
            BTreeMap::<usize, Vec<WeightedCompactPairTerm<E>>>::new();
        for group in &self.groups {
            self.append_d_span_terms(
                group,
                terms_by_projection_ratio.entry(group.d_ratio).or_default(),
            )?;
            self.append_b_span_terms(
                group,
                terms_by_projection_ratio.entry(group.b_ratio).or_default(),
            )?;
            self.append_a_span_terms(
                group,
                terms_by_projection_ratio.entry(group.a_ratio).or_default(),
            )?;
        }
        let base_ring_dim = self.projection_geometry.base_ring_dim();
        let base_ring_dim_u64 = u64::try_from(base_ring_dim).map_err(|_| {
            AkitaError::InvalidSetup("setup base ring dimension does not fit u64".into())
        })?;
        let alpha_per_base_ring = field_pow(alpha, base_ring_dim_u64);
        terms_by_projection_ratio
            .into_iter()
            .try_fold(E::zero(), |evaluation, (ratio, terms)| {
                if ratio == 0 || !ratio.is_power_of_two() {
                    return Err(AkitaError::InvalidSetup(
                        "setup projection ratio must be a non-zero power of two".into(),
                    ));
                }
                let low_variable_count = ratio.trailing_zeros() as usize;
                let setup_low_point =
                    rho_setup_idx
                        .get(..low_variable_count)
                        .ok_or(AkitaError::InvalidSize {
                            expected: low_variable_count,
                            actual: rho_setup_idx.len(),
                        })?;
                let setup_high_point =
                    rho_setup_idx
                        .get(low_variable_count..)
                        .ok_or(AkitaError::InvalidSize {
                            expected: low_variable_count,
                            actual: rho_setup_idx.len(),
                        })?;
                let relation_low_point = self.address_point.get(..low_variable_count).ok_or(
                    AkitaError::InvalidSize {
                        expected: low_variable_count,
                        actual: self.address_point.len(),
                    },
                )?;
                let relation_high_point = self.address_point.get(low_variable_count..).ok_or(
                    AkitaError::InvalidSize {
                        expected: low_variable_count,
                        actual: self.address_point.len(),
                    },
                )?;
                let power_factor =
                    evaluate_power_sequence_mle(alpha_per_base_ring, setup_low_point)
                        * evaluate_power_sequence_mle(alpha_per_base_ring, relation_low_point);
                Ok(evaluation
                    + power_factor
                        * eval_weighted_compact_pair_eq(
                            setup_high_point,
                            relation_high_point,
                            &terms,
                        )?)
            })
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
                        &group.a_families,
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
        terms: &mut Vec<WeightedCompactPairTerm<E>>,
    ) -> Result<(), AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(());
        }
        for span in &group.d_spans {
            ensure_span_lane_count(span, group.d_ratio)?;
            let setup_col = group
                .d_col_range
                .start
                .checked_add(span.setup_column_start)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D address overflow".into()))?;
            let relation_start = divide_aligned(
                span.relation_lane_start,
                group.d_ratio,
                "setup D relation address is not aligned to its projection ratio",
            )?;
            let relation_stride = divide_aligned(
                span.relation_lane_stride,
                group.d_ratio,
                "setup D relation stride is not aligned to its projection ratio",
            )?;
            for (row, &row_weight) in self.d_weights.iter().enumerate() {
                push_weighted_term(
                    terms,
                    WeightedCompactPairTerm {
                        left_offset: row_major_index(self.d_physical_cols, row, setup_col)?,
                        left_stride: span.setup_column_stride,
                        right_offset: relation_start,
                        right_stride: relation_stride,
                        len: span.occurrence_count,
                        weight: row_weight,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn append_b_span_terms(
        &self,
        group: &SetupContributionGroupPlan<E>,
        terms: &mut Vec<WeightedCompactPairTerm<E>>,
    ) -> Result<(), AkitaError> {
        for span in &group.b_spans {
            ensure_span_lane_count(span, group.b_ratio)?;
            let relation_start = divide_aligned(
                span.relation_lane_start,
                group.b_ratio,
                "setup B relation address is not aligned to its projection ratio",
            )?;
            let relation_stride = divide_aligned(
                span.relation_lane_stride,
                group.b_ratio,
                "setup B relation stride is not aligned to its projection ratio",
            )?;
            for (row, &row_weight) in group.b_weights.iter().enumerate() {
                push_weighted_term(
                    terms,
                    WeightedCompactPairTerm {
                        left_offset: row_major_index(group.t_cols, row, span.setup_column_start)?,
                        left_stride: span.setup_column_stride,
                        right_offset: relation_start,
                        right_stride: relation_stride,
                        len: span.occurrence_count,
                        weight: row_weight,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn append_a_span_terms(
        &self,
        group: &SetupContributionGroupPlan<E>,
        terms: &mut Vec<WeightedCompactPairTerm<E>>,
    ) -> Result<(), AkitaError> {
        for span in &group.a_families {
            if span.relation_lane_count != group.a_ratio
                || span.fold_count != group.fold_gadget.len()
            {
                return Err(AkitaError::InvalidSetup(
                    "setup A span is not one coarse fold family".into(),
                ));
            }
            let relation_stride = divide_aligned(
                span.relation_lane_stride,
                group.a_ratio,
                "setup A relation stride is not aligned to its projection ratio",
            )?;
            for (fold_digit, &fold) in group.fold_gadget.iter().enumerate() {
                let fold_lane_offset = fold_digit
                    .checked_mul(span.fold_lane_stride)
                    .and_then(|offset| span.relation_lane_start.checked_add(offset))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("setup A relation address overflow".into())
                    })?;
                let relation_start = divide_aligned(
                    fold_lane_offset,
                    group.a_ratio,
                    "setup A relation address is not aligned to its projection ratio",
                )?;
                for (row, &row_weight) in group.a_row_weights.iter().enumerate() {
                    push_weighted_term(
                        terms,
                        WeightedCompactPairTerm {
                            left_offset: row_major_index(
                                group.z_cols,
                                row,
                                span.setup_column_start,
                            )?,
                            left_stride: span.setup_column_stride,
                            right_offset: relation_start,
                            right_stride: relation_stride,
                            len: span.occurrence_count,
                            weight: -(row_weight * fold),
                        },
                    )?;
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
        let Some(relation_lane_start) = span.relation_lane_start_for_setup_column(column)? else {
            continue;
        };
        if let Some(fold_gadget) = fold_gadget {
            if span.relation_lane_count != lane_powers.len() || span.fold_count != fold_gadget.len()
            {
                return Err(AkitaError::InvalidSetup(
                    "setup A span is not one coarse fold family".into(),
                ));
            }
            for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                for (lane, &power) in lane_powers.iter().enumerate() {
                    let offset = fold_digit
                        .checked_mul(span.fold_lane_stride)
                        .and_then(|offset| offset.checked_add(lane))
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("setup A fold lane overflow".into())
                        })?;
                    weight -= equality_window
                        .eval(checked_add_relation_lane(relation_lane_start, offset)?)
                        * power
                        * fold;
                }
            }
        } else {
            super::structured::ensure_lane_count(span, lane_powers)?;
            for (lane, &power) in lane_powers.iter().enumerate() {
                weight += equality_window
                    .eval(checked_add_relation_lane(relation_lane_start, lane)?)
                    * power;
            }
        }
    }
    Ok(weight)
}

fn ensure_span_lane_count(span: &SetupContributionSpan, ratio: usize) -> Result<(), AkitaError> {
    if ratio == 0 || !ratio.is_power_of_two() || span.relation_lane_count != ratio {
        return Err(AkitaError::InvalidSetup(
            "setup span lane count disagrees with its projection ratio".into(),
        ));
    }
    Ok(())
}

fn checked_add_relation_lane(start: usize, lane: usize) -> Result<usize, AkitaError> {
    start
        .checked_add(lane)
        .ok_or_else(|| AkitaError::InvalidSetup("setup contribution relation lane overflow".into()))
}

fn divide_aligned(
    value: usize,
    divisor: usize,
    context: &'static str,
) -> Result<usize, AkitaError> {
    value
        .checked_div(divisor)
        .filter(|_| divisor != 0 && value.is_multiple_of(divisor))
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}

fn row_major_index(width: usize, row: usize, column: usize) -> Result<usize, AkitaError> {
    if column >= width {
        return Err(AkitaError::InvalidSetup(
            "setup row-major column out of range".into(),
        ));
    }
    width
        .checked_mul(row)
        .and_then(|base| base.checked_add(column))
        .ok_or_else(|| AkitaError::InvalidSetup("setup row-major index overflow".into()))
}
