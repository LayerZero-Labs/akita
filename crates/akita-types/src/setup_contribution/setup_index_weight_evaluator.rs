use akita_algebra::offset_eq::eval_compact_pair_eq;
use akita_field::{AkitaError, FieldCore};

use crate::{CommitmentRingDims, SetupContributionPlan, SetupProjectionGeometry};

use super::plan::SetupContributionGroupPlan;

/// Shape-selecting evaluator for the setup-index weight MLE.
///
/// The evaluator owns the canonical contribution plan. Exact uniform geometry
/// evaluates the plan's D/B/A intervals directly; other geometries remain on
/// the plan's general span evaluator until they gain an equally compact
/// shape-specific kernel.
pub struct SetupIndexWeightEvaluator<E: FieldCore> {
    plan: SetupContributionPlan<E>,
    alpha: E,
    uses_uniform_intervals: bool,
}

impl<E: FieldCore> SetupIndexWeightEvaluator<E> {
    /// Attach Stage 3's shape-specific evaluator to its canonical setup plan.
    pub fn new(plan: SetupContributionPlan<E>, alpha: E) -> Result<Self, AkitaError> {
        let geometry = plan.projection_geometry();
        geometry.ensure_evaluation_budget()?;
        let base_ring_dim = geometry.base_ring_dim();
        let relation_geometry = plan.relation_address_geometry;
        let uses_uniform_intervals = relation_geometry.carrier_ring_dimension() == base_ring_dim
            && plan.groups.iter().all(|group| {
                group.role_dims == CommitmentRingDims::uniform(base_ring_dim)
                    && group.a_ratio == 1
                    && group.b_ratio == 1
                    && group.d_ratio == 1
            });
        Ok(Self {
            plan,
            alpha,
            uses_uniform_intervals,
        })
    }

    /// Canonical common-base Stage 3 projection geometry.
    #[must_use]
    pub const fn projection_geometry(&self) -> SetupProjectionGeometry {
        self.plan.projection_geometry()
    }

    /// Number of base setup positions covered by this evaluator.
    #[must_use]
    pub fn required(&self) -> usize {
        self.plan.required()
    }

    /// Evaluate `setup_index_weight~(rho_setup_idx)` exactly.
    #[inline]
    pub fn evaluate(&self, rho_setup_idx: &[E]) -> Result<E, AkitaError> {
        if !self.uses_uniform_intervals {
            return self
                .plan
                .evaluate_setup_index_weight_mle(rho_setup_idx, self.alpha);
        }
        let _span = tracing::info_span!("stage3_setup_index_weight_uniform_intervals").entered();
        let expected = self
            .plan
            .projection_geometry()
            .setup_index_len()
            .trailing_zeros() as usize;
        if rho_setup_idx.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: rho_setup_idx.len(),
            });
        }

        let mut acc = E::zero();
        for group in &self.plan.groups {
            acc += self.evaluate_d_intervals(group, rho_setup_idx)?;
            acc += self.evaluate_b_intervals(group, rho_setup_idx)?;
            acc += self.evaluate_a_intervals(group, rho_setup_idx)?;
        }
        Ok(acc)
    }

    fn evaluate_d_intervals(
        &self,
        group: &SetupContributionGroupPlan<E>,
        rho_setup_idx: &[E],
    ) -> Result<E, AkitaError> {
        if self.plan.d_rows == 0 || self.plan.d_physical_cols == 0 {
            return Ok(E::zero());
        }
        if group.d_col_range.end > self.plan.d_physical_cols {
            return Err(AkitaError::InvalidSetup(
                "setup D active range exceeds physical width".into(),
            ));
        }

        let mut acc = E::zero();
        for span in &group.d_spans {
            if span.relation_lane_count != 1 {
                return Err(AkitaError::InvalidSetup(
                    "contiguous setup D span must address one relation lane".into(),
                ));
            }
            let setup_col = group
                .d_col_range
                .start
                .checked_add(span.setup_column_start)
                .ok_or_else(|| AkitaError::InvalidSetup("setup D address overflow".into()))?;
            for (row, &row_weight) in self.plan.d_weights.iter().enumerate() {
                let setup_index = row_major_index(self.plan.d_physical_cols, row, setup_col)?;
                acc += row_weight
                    * eval_compact_pair_eq(
                        rho_setup_idx,
                        setup_index,
                        span.setup_column_stride,
                        &self.plan.address_point,
                        span.relation_lane_start,
                        span.relation_lane_stride,
                        span.occurrence_count,
                    )?;
            }
        }
        Ok(acc)
    }

    fn evaluate_b_intervals(
        &self,
        group: &SetupContributionGroupPlan<E>,
        rho_setup_idx: &[E],
    ) -> Result<E, AkitaError> {
        if group.n_b == 0 {
            return Ok(E::zero());
        }
        let mut acc = E::zero();
        for span in &group.b_spans {
            if span.relation_lane_count != 1 {
                return Err(AkitaError::InvalidSetup(
                    "contiguous setup B span must address one relation lane".into(),
                ));
            }
            for (row, &row_weight) in group.b_weights.iter().enumerate() {
                let setup_index = row_major_index(group.t_cols, row, span.setup_column_start)?;
                acc += row_weight
                    * eval_compact_pair_eq(
                        rho_setup_idx,
                        setup_index,
                        span.setup_column_stride,
                        &self.plan.address_point,
                        span.relation_lane_start,
                        span.relation_lane_stride,
                        span.occurrence_count,
                    )?;
            }
        }
        Ok(acc)
    }

    fn evaluate_a_intervals(
        &self,
        group: &SetupContributionGroupPlan<E>,
        rho_setup_idx: &[E],
    ) -> Result<E, AkitaError> {
        if group.n_a == 0 {
            return Ok(E::zero());
        }
        let mut acc = E::zero();
        for span in &group.a_spans {
            if span.relation_lane_count != 1 {
                return Err(AkitaError::InvalidSetup(
                    "contiguous setup A span must address one relation lane".into(),
                ));
            }
            let fold = *group
                .fold_gadget
                .get(span.fold_digit.ok_or(AkitaError::InvalidProof)?)
                .ok_or(AkitaError::InvalidProof)?;
            for (row, &row_weight) in group.a_row_weights.iter().enumerate() {
                let setup_index = row_major_index(group.z_cols, row, span.setup_column_start)?;
                acc -= row_weight
                    * fold
                    * eval_compact_pair_eq(
                        rho_setup_idx,
                        setup_index,
                        span.setup_column_stride,
                        &self.plan.address_point,
                        span.relation_lane_start,
                        span.relation_lane_stride,
                        span.occurrence_count,
                    )?;
            }
        }
        Ok(acc)
    }
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
