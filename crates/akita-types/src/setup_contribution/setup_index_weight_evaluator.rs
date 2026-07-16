use akita_algebra::offset_eq::{eq_eval_at_index, OffsetEqWindow};
use akita_algebra::ring::scalar_powers;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::{
    SetupContributionGroupInputs, SetupContributionLayout, SetupContributionPlan,
    SetupContributionPlanInputs, SetupProjectionGeometry,
};

/// Succinct evaluator for the setup-index weight multilinear extension.
///
/// The evaluator contracts live D, B, and A setup spans with an exact sparse
/// pair-carry recurrence. It does not materialize the packed setup-weight
/// vector or a Cartesian equality domain. Witness addresses and setup columns
/// are resolved by one canonical shared [`SetupContributionLayout`].
#[derive(Clone)]
pub struct SetupIndexWeightEvaluator<E> {
    tau1: Vec<E>,
    x_challenges: Vec<E>,
    layout: SetupContributionLayout,
    d_row_start: usize,
    d_rows: usize,
    d_physical_cols: usize,
    a_projection: SetupRoleProjection<E>,
    b_projection: SetupRoleProjection<E>,
    d_projection: SetupRoleProjection<E>,
    fold_gadget: Vec<E>,
    required: usize,
}

#[derive(Clone)]
struct SetupRoleProjection<E> {
    ratio: usize,
    scales: Vec<E>,
}

impl<E: FieldCore> SetupIndexWeightEvaluator<E> {
    /// Build a succinct evaluator for `setup_index_weight~`.
    ///
    /// `setup_ring_dim` is the base ring dimension used by the setup prefix.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F>(
        inputs: &SetupContributionPlanInputs<E>,
        plan: &SetupContributionPlan<E>,
        layout: &SetupContributionLayout,
        tau1: &[E],
        x_challenges: &[E],
        fold_gadget: &[F],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        if layout.groups().is_empty() {
            return Err(AkitaError::InvalidSetup(
                "setup-index weight evaluator requires at least one group".into(),
            ));
        }
        let geometry = plan.projection_geometry();
        geometry.ensure_evaluation_budget()?;
        validate_tau_domain(tau1, inputs.rows)?;

        let d_rows = plan.d_rows;
        let d_physical_cols = plan.d_physical_cols;
        let d_row_start = inputs.rows.checked_sub(d_rows).ok_or_else(|| {
            AkitaError::InvalidSetup("setup D rows exceed relation row count".into())
        })?;
        let a_projection = setup_role_projection(alpha, geometry, geometry.a_ratio(), "A")?;
        let b_projection = setup_role_projection(alpha, geometry, geometry.b_ratio(), "B")?;
        let d_projection = setup_role_projection(alpha, geometry, geometry.d_ratio(), "D")?;

        for group in layout.groups() {
            if fold_gadget.len() < group.depth_fold {
                return Err(AkitaError::InvalidSize {
                    expected: group.depth_fold,
                    actual: fold_gadget.len(),
                });
            }
        }
        let fold_gadget = fold_gadget
            .iter()
            .copied()
            .map(|fold| E::one().mul_base(fold))
            .collect::<Vec<_>>();
        Ok(Self {
            tau1: tau1.to_vec(),
            x_challenges: x_challenges.to_vec(),
            layout: layout.clone(),
            d_row_start,
            d_rows,
            d_physical_cols,
            a_projection,
            b_projection,
            d_projection,
            fold_gadget,
            required: geometry.required(),
        })
    }

    /// Number of base setup positions covered by this evaluator.
    #[must_use]
    pub fn required(&self) -> usize {
        self.required
    }

    /// Evaluate `setup_index_weight~(rho_setup_idx)` exactly.
    #[tracing::instrument(skip_all, name = "stage3_setup_index_weight")]
    pub fn evaluate(&self, rho_setup_idx: &[E]) -> Result<E, AkitaError> {
        let setup_idx_bits = self.setup_idx_bits()?;
        if rho_setup_idx.len() != setup_idx_bits {
            return Err(AkitaError::InvalidSize {
                expected: setup_idx_bits,
                actual: rho_setup_idx.len(),
            });
        }

        // Build bounded equality windows once and share them across every role
        // and every strided address in the inner sums. The setup-index and
        // opening addresses are each visited O(setup columns) times per level,
        // so recomputing a full O(bits) equality product per address dominated
        // the recursive-mode verifier (setup-product stage 3).
        let setup_window = OffsetEqWindow::new(rho_setup_idx)?;
        let x_window = OffsetEqWindow::new(&self.x_challenges)?;

        let mut acc = E::zero();
        for group in self.layout.groups() {
            acc += self.evaluate_d_role(group, &setup_window, &x_window)?;
            acc += self.evaluate_b_role(group, &setup_window, &x_window)?;
            acc += self.evaluate_a_role(group, &setup_window, &x_window)?;
        }
        Ok(acc)
    }

    fn setup_idx_bits(&self) -> Result<usize, AkitaError> {
        let setup_idx_len = self
            .required
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("setup-index weight length overflow".into()))?;
        Ok(setup_idx_len.trailing_zeros() as usize)
    }

    fn evaluate_d_role(
        &self,
        group: &SetupContributionGroupInputs,
        setup_window: &OffsetEqWindow<E>,
        x_window: &OffsetEqWindow<E>,
    ) -> Result<E, AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(E::zero());
        }
        let active_cols = checked_mul3(
            group.num_claims,
            group.live_fold_count,
            group.depth_open,
            "setup D active width overflow",
        )?;
        let d_col_range = self.layout.d_col_range(group.group_id)?;
        if d_col_range.len() != active_cols || d_col_range.end > self.d_physical_cols {
            return Err(AkitaError::InvalidSetup(
                "setup D active range exceeds physical width".into(),
            ));
        }

        let mut acc = E::zero();
        let units = self
            .layout
            .witness_layout()
            .units_for_group(group.group_id)?;
        for claim in 0..group.num_claims {
            for unit in &units {
                let setup_col = group
                    .live_fold_count
                    .checked_mul(claim)
                    .and_then(|base| base.checked_add(unit.global_fold_start()))
                    .and_then(|base| base.checked_mul(group.depth_open))
                    .and_then(|local| d_col_range.start.checked_add(local))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup D address overflow".into()))?;
                let witness_index = self.layout.witness_layout().e_index(
                    unit,
                    group.num_claims,
                    group.depth_open,
                    claim,
                    unit.global_fold_start(),
                    0,
                )?;
                let len = unit
                    .live_fold_count()
                    .checked_mul(group.depth_open)
                    .ok_or_else(|| AkitaError::InvalidSetup("setup D span overflow".into()))?;
                for row in 0..self.d_rows {
                    let row_weight = eq_eval_at_index(&self.tau1, self.d_row_start + row);
                    for (lane, &scale) in self.d_projection.scales.iter().enumerate() {
                        let setup_index = projected_setup_offset(
                            &self.d_projection,
                            self.d_physical_cols,
                            row,
                            setup_col,
                            lane,
                        )?;
                        let mut pair = E::zero();
                        for index in 0..len {
                            let setup_delta =
                                index.checked_mul(self.d_projection.ratio).ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup D address overflow".into())
                                })?;
                            let setup_address =
                                setup_index.checked_add(setup_delta).ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup D address overflow".into())
                                })?;
                            let physical_address =
                                witness_index.checked_add(index).ok_or_else(|| {
                                    AkitaError::InvalidSetup("witness D address overflow".into())
                                })?;
                            let opening_address = crate::checked_opening_source_index(
                                self.layout.opening_source_len(),
                                physical_address,
                            )?;
                            pair +=
                                setup_window.eval(setup_address) * x_window.eval(opening_address);
                        }
                        acc += row_weight * scale * pair;
                    }
                }
            }
        }
        Ok(acc)
    }

    fn evaluate_b_role(
        &self,
        group: &SetupContributionGroupInputs,
        setup_window: &OffsetEqWindow<E>,
        x_window: &OffsetEqWindow<E>,
    ) -> Result<E, AkitaError> {
        if group.n_b == 0 {
            return Ok(E::zero());
        }
        let t_cols = group
            .num_claims
            .checked_mul(group.t_cols_per_vector)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
        let mut acc = E::zero();
        let units = self
            .layout
            .witness_layout()
            .units_for_group(group.group_id)?;
        for claim in 0..group.num_claims {
            for unit in &units {
                let setup_col = group
                    .live_fold_count
                    .checked_mul(claim)
                    .and_then(|base| base.checked_add(unit.global_fold_start()))
                    .and_then(|base| base.checked_mul(group.n_a))
                    .and_then(|base| base.checked_mul(group.depth_open))
                    .ok_or_else(|| AkitaError::InvalidSetup("setup B address overflow".into()))?;
                let witness_index = self.layout.witness_layout().t_index(
                    unit,
                    group.num_claims,
                    group.n_a,
                    group.depth_open,
                    claim,
                    unit.global_fold_start(),
                    0,
                    0,
                )?;
                let len = checked_mul3(
                    unit.live_fold_count(),
                    group.n_a,
                    group.depth_open,
                    "setup B span overflow",
                )?;
                for row in 0..group.n_b {
                    let row_weight = eq_eval_at_index(&self.tau1, group.b_row_start + row);
                    for (lane, &scale) in self.b_projection.scales.iter().enumerate() {
                        let setup_index = projected_setup_offset(
                            &self.b_projection,
                            t_cols,
                            row,
                            setup_col,
                            lane,
                        )?;
                        let mut pair = E::zero();
                        for index in 0..len {
                            let setup_delta =
                                index.checked_mul(self.b_projection.ratio).ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup B address overflow".into())
                                })?;
                            let setup_address =
                                setup_index.checked_add(setup_delta).ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup B address overflow".into())
                                })?;
                            let physical_address =
                                witness_index.checked_add(index).ok_or_else(|| {
                                    AkitaError::InvalidSetup("witness B address overflow".into())
                                })?;
                            let opening_address = crate::checked_opening_source_index(
                                self.layout.opening_source_len(),
                                physical_address,
                            )?;
                            pair +=
                                setup_window.eval(setup_address) * x_window.eval(opening_address);
                        }
                        acc += row_weight * scale * pair;
                    }
                }
            }
        }
        Ok(acc)
    }

    fn evaluate_a_role(
        &self,
        group: &SetupContributionGroupInputs,
        setup_window: &OffsetEqWindow<E>,
        x_window: &OffsetEqWindow<E>,
    ) -> Result<E, AkitaError> {
        if group.n_a == 0 {
            return Ok(E::zero());
        }
        let z_cols = group
            .fold_position_count
            .checked_mul(group.depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A width overflow".into()))?;
        let units = self
            .layout
            .witness_layout()
            .units_for_group(group.group_id)?;
        let setup_col = 0;
        let mut acc = E::zero();
        for unit in &units {
            for (fold_digit, &fold) in self.fold_gadget.iter().enumerate().take(group.depth_fold) {
                let witness_index = self.layout.witness_layout().z_index(
                    unit,
                    group.fold_position_count,
                    group.depth_commit,
                    group.depth_fold,
                    0,
                    0,
                    fold_digit,
                )?;
                for row in 0..group.n_a {
                    let row_weight = eq_eval_at_index(&self.tau1, group.a_row_start + row);
                    for (lane, &scale) in self.a_projection.scales.iter().enumerate() {
                        let setup_index = projected_setup_offset(
                            &self.a_projection,
                            z_cols,
                            row,
                            setup_col,
                            lane,
                        )?;
                        let mut pair = E::zero();
                        for index in 0..z_cols {
                            let setup_delta =
                                index.checked_mul(self.a_projection.ratio).ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup A address overflow".into())
                                })?;
                            let setup_address =
                                setup_index.checked_add(setup_delta).ok_or_else(|| {
                                    AkitaError::InvalidSetup("setup A address overflow".into())
                                })?;
                            let witness_delta =
                                index.checked_mul(group.depth_fold).ok_or_else(|| {
                                    AkitaError::InvalidSetup("witness A address overflow".into())
                                })?;
                            let physical_address =
                                witness_index.checked_add(witness_delta).ok_or_else(|| {
                                    AkitaError::InvalidSetup("witness A address overflow".into())
                                })?;
                            let opening_address = crate::checked_opening_source_index(
                                self.layout.opening_source_len(),
                                physical_address,
                            )?;
                            pair +=
                                setup_window.eval(setup_address) * x_window.eval(opening_address);
                        }
                        acc -= row_weight * scale * fold * pair;
                    }
                }
            }
        }
        Ok(acc)
    }
}

fn projected_setup_offset<E: FieldCore>(
    projection: &SetupRoleProjection<E>,
    width: usize,
    row: usize,
    column: usize,
    lane: usize,
) -> Result<usize, AkitaError> {
    if column >= width {
        return Err(AkitaError::InvalidSetup(
            "setup column exceeds role width".into(),
        ));
    }
    if lane >= projection.ratio {
        return Err(AkitaError::InvalidSetup(
            "setup projection lane out of range".into(),
        ));
    }
    let logical = width
        .checked_mul(row)
        .and_then(|base| base.checked_add(column))
        .ok_or_else(|| AkitaError::InvalidSetup("setup role index overflow".into()))?;
    let base = projection
        .ratio
        .checked_mul(logical)
        .ok_or_else(|| AkitaError::InvalidSetup("setup base index overflow".into()))?;
    base.checked_add(lane)
        .ok_or_else(|| AkitaError::InvalidSetup("setup lane index overflow".into()))
}

fn setup_role_projection<E: FieldCore>(
    alpha: E,
    geometry: SetupProjectionGeometry,
    ratio: usize,
    role: &'static str,
) -> Result<SetupRoleProjection<E>, AkitaError> {
    let role_dim = geometry
        .base_ring_dim()
        .checked_mul(ratio)
        .ok_or_else(|| AkitaError::InvalidSetup(format!("{role} ring dimension overflow")))?;
    if ratio == 1 {
        return Ok(SetupRoleProjection {
            ratio,
            scales: vec![E::one()],
        });
    }

    let role_pows = scalar_powers(alpha, role_dim);
    let base_pows = &role_pows[..geometry.base_ring_dim()];
    let mut scales = Vec::with_capacity(ratio);
    for lane in 0..ratio {
        let offset = lane * geometry.base_ring_dim();
        let scale = role_pows[offset];
        for idx in 0..geometry.base_ring_dim() {
            if role_pows[offset + idx] != scale * base_pows[idx] {
                return Err(AkitaError::InvalidSetup(format!(
                    "{role} setup-index weight alpha powers do not decompose over base setup ring"
                )));
            }
        }
        scales.push(scale);
    }
    Ok(SetupRoleProjection { ratio, scales })
}

fn validate_tau_domain<E: FieldCore>(tau1: &[E], rows: usize) -> Result<(), AkitaError> {
    if tau1.len() < usize::BITS as usize && rows > (1usize << tau1.len()) {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: 1usize << tau1.len(),
        });
    }
    Ok(())
}

fn checked_mul3(a: usize, b: usize, c: usize, context: &'static str) -> Result<usize, AkitaError> {
    a.checked_mul(b)
        .and_then(|ab| ab.checked_mul(c))
        .ok_or_else(|| AkitaError::InvalidSetup(context.into()))
}
