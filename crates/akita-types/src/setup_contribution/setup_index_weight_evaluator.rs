use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::ring::scalar_powers;
use akita_field::{AkitaError, FieldCore, MulBase};

use crate::{
    SetupContributionGroupInputs, SetupContributionPlan, SetupContributionPlanInputs,
    SetupProjectionGeometry,
};

/// Succinct evaluator for the setup-index weight multilinear extension.
///
/// The evaluator contracts live D, B, and A setup spans with an exact sparse
/// pair-carry recurrence. It does not materialize the packed setup-weight
/// vector or a Cartesian equality domain. Witness addresses and setup columns
/// are resolved by the canonical [`crate::OpeningBatchWitnessLayout`] carried
/// by each group.
#[derive(Clone)]
pub struct SetupIndexWeightEvaluator<E> {
    tau1: Vec<E>,
    x_challenges: Vec<E>,
    groups: Vec<SetupContributionGroupInputs>,
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
        groups: &[SetupContributionGroupInputs],
        tau1: &[E],
        x_challenges: &[E],
        fold_gadget: &[F],
        alpha: E,
    ) -> Result<Self, AkitaError>
    where
        F: FieldCore,
        E: MulBase<F>,
    {
        if groups.is_empty() {
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

        for group in groups {
            if fold_gadget.len() < group.depth_fold {
                return Err(AkitaError::InvalidSize {
                    expected: group.depth_fold,
                    actual: fold_gadget.len(),
                });
            }
            validate_group_layout(group)?;
        }
        let fold_gadget = fold_gadget
            .iter()
            .copied()
            .map(|fold| E::one().mul_base(fold))
            .collect::<Vec<_>>();
        Ok(Self {
            tau1: tau1.to_vec(),
            x_challenges: x_challenges.to_vec(),
            groups: groups.to_vec(),
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
    pub fn evaluate(&self, rho_setup_idx: &[E]) -> Result<E, AkitaError> {
        let setup_idx_bits = self.setup_idx_bits()?;
        if rho_setup_idx.len() != setup_idx_bits {
            return Err(AkitaError::InvalidSize {
                expected: setup_idx_bits,
                actual: rho_setup_idx.len(),
            });
        }

        let mut acc = E::zero();
        for group in &self.groups {
            acc += self.evaluate_d_role(group, rho_setup_idx)?;
            acc += self.evaluate_b_role(group, rho_setup_idx)?;
            acc += self.evaluate_a_role(group, rho_setup_idx)?;
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
        rho_setup_idx: &[E],
    ) -> Result<E, AkitaError> {
        if self.d_rows == 0 || self.d_physical_cols == 0 {
            return Ok(E::zero());
        }
        let active_cols = checked_mul3(
            group.num_claims,
            group.num_blocks,
            group.depth_open,
            "setup D active width overflow",
        )?;
        let active_end = group
            .e_col_offset
            .checked_add(active_cols)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D active range overflow".into()))?;
        if active_end > self.d_physical_cols {
            return Err(AkitaError::InvalidSetup(
                "setup D active range exceeds physical width".into(),
            ));
        }

        let mut acc = E::zero();
        let units = group.layout.units_for_group(group.group_id)?;
        for claim in 0..group.num_claims {
            for unit in &units {
                let setup_col = group.layout.e_setup_col_index(
                    group.group_id,
                    claim,
                    unit.global_block_base,
                    0,
                )?;
                let witness_index = group
                    .layout
                    .e_index(unit, claim, unit.global_block_base, 0)?;
                let len = unit
                    .blocks
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
                            let opening_address = group
                                .opening_layout
                                .opening_index_for_physical(physical_address)?;
                            pair += eq_eval_at_index(rho_setup_idx, setup_address)
                                * eq_eval_at_index(&self.x_challenges, opening_address);
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
        rho_setup_idx: &[E],
    ) -> Result<E, AkitaError> {
        if group.n_b == 0 {
            return Ok(E::zero());
        }
        let t_cols = group
            .num_claims
            .checked_mul(group.t_cols_per_vector)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".into()))?;
        let mut acc = E::zero();
        let units = group.layout.units_for_group(group.group_id)?;
        for claim in 0..group.num_claims {
            for unit in &units {
                let setup_col = group.layout.t_setup_col_index(
                    group.group_id,
                    claim,
                    unit.global_block_base,
                    0,
                    0,
                )?;
                let witness_index =
                    group
                        .layout
                        .t_index(unit, claim, unit.global_block_base, 0, 0)?;
                let len = checked_mul3(
                    unit.blocks,
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
                            let opening_address = group
                                .opening_layout
                                .opening_index_for_physical(physical_address)?;
                            pair += eq_eval_at_index(rho_setup_idx, setup_address)
                                * eq_eval_at_index(&self.x_challenges, opening_address);
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
        rho_setup_idx: &[E],
    ) -> Result<E, AkitaError> {
        if group.n_a == 0 {
            return Ok(E::zero());
        }
        let z_cols = group
            .block_len
            .checked_mul(group.depth_commit)
            .ok_or_else(|| AkitaError::InvalidSetup("setup A width overflow".into()))?;
        let units = group.layout.units_for_group(group.group_id)?;
        let setup_col = group.layout.z_setup_col_index(group.group_id, 0, 0)?;
        let mut acc = E::zero();
        for unit in &units {
            for (fold_digit, &fold) in self.fold_gadget.iter().enumerate().take(group.depth_fold) {
                let witness_index = group.layout.z_index(unit, 0, 0, fold_digit)?;
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
                            let opening_address = group
                                .opening_layout
                                .opening_index_for_physical(physical_address)?;
                            pair += eq_eval_at_index(rho_setup_idx, setup_address)
                                * eq_eval_at_index(&self.x_challenges, opening_address);
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

fn validate_group_layout(group: &SetupContributionGroupInputs) -> Result<(), AkitaError> {
    let descriptor = group.layout.group(group.group_id)?;
    if descriptor.num_claims != group.num_claims
        || descriptor.num_blocks != group.num_blocks
        || descriptor.block_len != group.block_len
        || descriptor.depth_open != group.depth_open
        || descriptor.depth_commit != group.depth_commit
        || descriptor.depth_fold != group.depth_fold
        || descriptor.n_a != group.n_a
        || descriptor.e_setup_col_offset != group.e_col_offset
    {
        return Err(AkitaError::InvalidSetup(
            "setup group dimensions disagree with witness layout".into(),
        ));
    }
    Ok(())
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
