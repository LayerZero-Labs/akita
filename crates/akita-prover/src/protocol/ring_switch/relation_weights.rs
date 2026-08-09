//! Semantic relation-weight events and their canonical consumers.

use std::ops::Range;

use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::offset_eq::{eq_eval_at_index, OffsetEqWindow};
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
};
use akita_types::{
    gadget_row_scalars, r_decomp_levels, relation_rhs_layout_for, AkitaExpandedSetup,
    CommittedGroupParams, FpExtEncoding, OpeningClaimsLayout, RelationAddressGeometry,
    RelationRowFamily, RingRelationInstance, SetupProjectionGeometry,
};

/// Whether one relation event belongs to the protocol constraint or setup matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationWeightContribution {
    /// Consistency, A-row, opening, and quotient-denominator arithmetic.
    Constraint,
    /// D/B/A setup-matrix arithmetic replaceable by one offloaded setup claim.
    SetupMatrix,
}

/// One aligned consecutive-alpha contribution to the flat relation weight table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightEvent<E: FieldCore> {
    physical_coefficients: Range<usize>,
    alpha_exponent_start: usize,
    scalar: E,
    contribution: RelationWeightContribution,
}

impl<E: FieldCore> RelationWeightEvent<E> {
    /// Flat physical coefficient interval receiving this contribution.
    #[must_use]
    pub fn physical_coefficients(&self) -> Range<usize> {
        self.physical_coefficients.clone()
    }

    /// Alpha exponent attached to the first coefficient in the interval.
    #[must_use]
    pub fn alpha_exponent_start(&self) -> usize {
        self.alpha_exponent_start
    }

    /// Scalar multiplying the consecutive alpha powers.
    #[must_use]
    pub fn scalar(&self) -> E {
        self.scalar
    }

    /// Whether this is constraint or setup-matrix arithmetic.
    #[must_use]
    pub fn contribution(&self) -> RelationWeightContribution {
        self.contribution
    }
}

/// Source of setup-matrix relation weights for this evaluation.
#[derive(Clone, Copy)]
pub enum RelationSetupSource<'a, F: FieldCore> {
    /// Emit setup events directly from the expanded setup matrix.
    Matrix(&'a AkitaExpandedSetup<F>),
    /// Omit setup events because their complete evaluation is supplied separately.
    DeferredClaim,
}

/// Inputs to the one semantic relation-event builder.
pub struct RelationWeightEventInputs<'a, F: FieldCore, E: FieldCore> {
    pub setup: RelationSetupSource<'a, F>,
    pub instance: &'a RingRelationInstance<F>,
    pub alpha: E,
    pub level_params: &'a CommittedGroupParams,
    pub relation_row_point: &'a [E],
    pub claim_coefficients: &'a [E],
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
}

/// Checked relation events plus the domain data needed by every consumer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightEvents<E: FieldCore> {
    events: Vec<RelationWeightEvent<E>>,
    alpha_powers: Vec<E>,
    relation_coefficient_block_len: usize,
    physical_field_len: usize,
    setup_is_deferred: bool,
}

/// Exact common-alpha factorization of the padded relation-weight table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightFactorization<E: FieldCore> {
    common_alpha_factor: Vec<E>,
    relation_lane_weights: Vec<E>,
}

impl<E: FieldCore> RelationWeightFactorization<E> {
    /// Alpha powers on the low coefficient block shared by every role.
    #[must_use]
    pub fn common_alpha_factor(&self) -> &[E] {
        &self.common_alpha_factor
    }

    /// Relation weights after removing the shared low alpha factor.
    #[must_use]
    pub fn relation_lane_weights(&self) -> &[E] {
        &self.relation_lane_weights
    }

    /// Consume the factorization without recomputing either component.
    #[must_use]
    pub fn into_common_alpha_factor_and_relation_lane_weights(self) -> (Vec<E>, Vec<E>) {
        (self.common_alpha_factor, self.relation_lane_weights)
    }

    /// Expand this factorization over its complete padded flat domain.
    pub fn materialize_dense(&self) -> Result<Vec<E>, AkitaError> {
        let length = self
            .common_alpha_factor
            .len()
            .checked_mul(self.relation_lane_weights.len())
            .ok_or_else(|| AkitaError::InvalidSetup("relation weight length overflow".into()))?;
        let mut weights = Vec::with_capacity(length);
        for &lane in &self.relation_lane_weights {
            weights.extend(
                self.common_alpha_factor
                    .iter()
                    .map(|&coefficient| lane * coefficient),
            );
        }
        Ok(weights)
    }
}

impl<E: FieldCore> RelationWeightEvents<E> {
    fn push(
        &mut self,
        physical_start: usize,
        coefficient_count: usize,
        alpha_exponent_start: usize,
        scalar: E,
        contribution: RelationWeightContribution,
    ) -> Result<(), AkitaError> {
        if scalar.is_zero() {
            return Ok(());
        }
        let physical_end = physical_start
            .checked_add(coefficient_count)
            .ok_or_else(|| AkitaError::InvalidSetup("relation event address overflow".into()))?;
        let alpha_exponent_end = alpha_exponent_start
            .checked_add(coefficient_count)
            .ok_or_else(|| AkitaError::InvalidSetup("relation alpha range overflow".into()))?;
        if coefficient_count == 0
            || !coefficient_count.is_power_of_two()
            || !physical_start.is_multiple_of(self.relation_coefficient_block_len)
            || !coefficient_count.is_multiple_of(self.relation_coefficient_block_len)
            || !alpha_exponent_start.is_multiple_of(self.relation_coefficient_block_len)
            || physical_end > self.physical_field_len
            || alpha_exponent_end > self.alpha_powers.len()
            || (self.setup_is_deferred && contribution == RelationWeightContribution::SetupMatrix)
        {
            return Err(AkitaError::InvalidSetup(
                "relation event is unaligned or outside its checked domain".into(),
            ));
        }
        self.events.push(RelationWeightEvent {
            physical_coefficients: physical_start..physical_end,
            alpha_exponent_start,
            scalar,
            contribution,
        });
        Ok(())
    }

    fn push_native_ring(
        &mut self,
        physical_start: usize,
        role_ring_dimension: usize,
        scalar: E,
        contribution: RelationWeightContribution,
    ) -> Result<(), AkitaError> {
        if role_ring_dimension == 0 {
            return Err(AkitaError::InvalidProof);
        }
        self.push(physical_start, role_ring_dimension, 0, scalar, contribution)
    }

    /// Semantic events in emission order. Overlaps are intentionally additive.
    #[must_use]
    pub fn events(&self) -> &[RelationWeightEvent<E>] {
        &self.events
    }

    /// Materialize the complete padded flat coefficient table.
    pub fn materialize_dense(&self) -> Result<Vec<E>, AkitaError> {
        if self.setup_is_deferred {
            return Err(AkitaError::InvalidInput(
                "cannot materialize relation weights with a deferred setup claim".into(),
            ));
        }
        let mut weights = vec![E::zero(); self.physical_field_len];
        for event in &self.events {
            for (offset, alpha_power) in self.alpha_powers[event.alpha_exponent_start
                ..event.alpha_exponent_start + event.physical_coefficients.len()]
                .iter()
                .copied()
                .enumerate()
            {
                let physical = event.physical_coefficients.start + offset;
                *weights.get_mut(physical).ok_or(AkitaError::InvalidProof)? +=
                    event.scalar * alpha_power;
            }
        }
        Ok(weights)
    }

    /// Compile the exact common-alpha factorization shared by all role dimensions.
    pub fn factor_common_alpha(&self) -> Result<RelationWeightFactorization<E>, AkitaError> {
        if self.setup_is_deferred {
            return Err(AkitaError::InvalidSetup(
                "relation factorization requires direct setup contributions".into(),
            ));
        }
        let coeff_count = self.relation_coefficient_block_len;
        let lane_capacity = self
            .physical_field_len
            .checked_div(coeff_count)
            .filter(|capacity| capacity.is_power_of_two())
            .ok_or_else(|| AkitaError::InvalidSetup("relation lane capacity is invalid".into()))?;
        let mut relation_lane_weights = vec![E::zero(); lane_capacity];
        for event in &self.events {
            if !event
                .physical_coefficients
                .start
                .is_multiple_of(coeff_count)
                || !event
                    .physical_coefficients
                    .len()
                    .is_multiple_of(coeff_count)
                || !event.alpha_exponent_start.is_multiple_of(coeff_count)
            {
                return Err(AkitaError::InvalidSetup(
                    "relation event does not preserve the common alpha factor".into(),
                ));
            }
            for coefficient_offset in (0..event.physical_coefficients.len()).step_by(coeff_count) {
                let physical = event.physical_coefficients.start + coefficient_offset;
                if !physical.is_multiple_of(coeff_count) {
                    return Err(AkitaError::InvalidSetup(
                        "flat relation layout breaks relation lane alignment".into(),
                    ));
                }
                let lane = physical / coeff_count;
                let alpha_exponent = event.alpha_exponent_start + coefficient_offset;
                let alpha_power = *self
                    .alpha_powers
                    .get(alpha_exponent)
                    .ok_or(AkitaError::InvalidProof)?;
                *relation_lane_weights
                    .get_mut(lane)
                    .ok_or(AkitaError::InvalidProof)? += event.scalar * alpha_power;
            }
        }
        let common_alpha_factor = self
            .alpha_powers
            .get(..coeff_count)
            .ok_or(AkitaError::InvalidProof)?
            .to_vec();
        Ok(RelationWeightFactorization {
            common_alpha_factor,
            relation_lane_weights,
        })
    }

    /// Evaluate the relation-weight MLE directly at one flat coefficient point.
    pub fn evaluate_at_point(
        &self,
        point: &[E],
        deferred_setup_claim: Option<E>,
    ) -> Result<E, AkitaError> {
        match (self.setup_is_deferred, deferred_setup_claim) {
            (true, None) | (false, Some(_)) => return Err(AkitaError::InvalidProof),
            _ => {}
        }
        if self.physical_field_len != 1usize.checked_shl(point.len() as u32).unwrap_or(0) {
            return Err(AkitaError::InvalidSize {
                expected: self.physical_field_len.trailing_zeros() as usize,
                actual: point.len(),
            });
        }

        let equality = OffsetEqWindow::new(point)?;
        let mut low_factor_cache = Vec::new();
        let mut evaluation = deferred_setup_claim.unwrap_or_else(E::zero);
        for event in &self.events {
            let coefficient_count = event.physical_coefficients.len();
            if !event
                .physical_coefficients
                .start
                .is_multiple_of(coefficient_count)
            {
                let alpha_powers = &self.alpha_powers
                    [event.alpha_exponent_start..event.alpha_exponent_start + coefficient_count];
                let interval = alpha_powers.iter().copied().enumerate().fold(
                    E::zero(),
                    |sum, (offset, alpha_power)| {
                        sum + alpha_power
                            * equality.eval(event.physical_coefficients.start + offset)
                    },
                );
                evaluation += event.scalar * interval;
                continue;
            }
            let low_variable_count = coefficient_count.trailing_zeros() as usize;
            let cache_key = (event.alpha_exponent_start, coefficient_count);
            let low_factor = if let Some((_, cached)) = low_factor_cache
                .iter()
                .find(|(cached_key, _)| *cached_key == cache_key)
            {
                *cached
            } else {
                let alpha_powers = &self.alpha_powers
                    [event.alpha_exponent_start..event.alpha_exponent_start + coefficient_count];
                let factor = multilinear_eval(alpha_powers, &point[..low_variable_count])?;
                low_factor_cache.push((cache_key, factor));
                factor
            };
            let high_index = event.physical_coefficients.start >> low_variable_count;
            let high_factor = eq_eval_at_index(&point[low_variable_count..], high_index);
            evaluation += event.scalar * low_factor * high_factor;
        }
        Ok(evaluation)
    }
}

fn relation_d_group_width(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params(opening_batch, group_index)?;
    let group_dims = lp.group_role_dims(opening_batch, group_index)?;
    let (_, d_subcolumns) = SetupProjectionGeometry::native_role_subcolumn_counts(group_dims)?;
    let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
    num_claims
        .checked_mul(group_lp.num_live_blocks())
        .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
        .and_then(|n| n.checked_mul(d_subcolumns))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
}

fn relation_d_column_ranges(
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
) -> Result<Vec<Range<usize>>, AkitaError> {
    let mut cursor = 0usize;
    let mut seen = vec![false; opening_batch.num_groups()];
    let mut ranges = vec![0..0; opening_batch.num_groups()];
    for group_id in opening_batch.root_group_order()? {
        let slot = seen
            .get_mut(group_id)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D group id out of range".into()))?;
        if std::mem::replace(slot, true) {
            return Err(AkitaError::InvalidSetup(
                "setup D group id appears more than once".into(),
            ));
        }
        let width = relation_d_group_width(lp, opening_batch, group_id)?;
        let end = cursor
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".into()))?;
        ranges[group_id] = cursor..end;
        cursor = end;
    }
    if seen.iter().any(|present| !present) {
        return Err(AkitaError::InvalidSetup(
            "setup D group ids are not contiguous".into(),
        ));
    }
    Ok(ranges)
}

/// Emit the complete checked relation semantics for one fold.
#[tracing::instrument(skip_all, name = "build_relation_weight_events")]
pub fn build_relation_weight_events<F, E>(
    inputs: RelationWeightEventInputs<'_, F, E>,
) -> Result<RelationWeightEvents<E>, AkitaError>
where
    F: FieldCore + CanonicalField,
    E: FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F> + MulBaseUnreduced<F>,
{
    let RelationWeightEventInputs {
        setup,
        instance,
        alpha,
        level_params: lp,
        relation_row_point: tau1,
        claim_coefficients: gamma,
        opening_source_len,
        opening_ring_dim,
    } = inputs;
    let opening_batch = instance.opening_batch();
    lp.witness_chunk.validate()?;
    lp.validate_opening_batch(opening_batch)?;
    if gamma.len() != opening_batch.num_total_polynomials() {
        return Err(AkitaError::InvalidProof);
    }
    let role_dims = instance.role_dims();
    if role_dims != lp.role_dims() {
        return Err(AkitaError::InvalidSetup(
            "relation instance and level role dimensions disagree".into(),
        ));
    }
    let d_a = role_dims.d_a();
    let d_b = role_dims.d_b();
    let d_d = role_dims.d_d();
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let relation_rhs_layout = relation_rhs_layout_for(lp, opening_batch)?;
    let row_families = relation_rhs_layout.row_families()?;
    let quotient_row_dims = row_families
        .iter()
        .map(|row| row.ring_dim())
        .collect::<Vec<_>>();
    let rows = quotient_row_dims.len();
    if rows != lp.relation_matrix_row_count(opening_batch.num_groups())? {
        return Err(AkitaError::InvalidSetup(
            "relation quotient row dimensions disagree with the matrix layout".into(),
        ));
    }
    let mut additional_quotient_alpha_powers = Vec::new();
    for &row_dim in &quotient_row_dims {
        if row_dim != d_a
            && row_dim != d_b
            && row_dim != d_d
            && additional_quotient_alpha_powers
                .iter()
                .all(|(dimension, _): &(usize, Vec<E>)| *dimension != row_dim)
        {
            additional_quotient_alpha_powers.push((row_dim, scalar_powers(alpha, row_dim)));
        }
    }
    let eq_tau1 = SplitEqEvals::new(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.open_commit_matrix.output_rank();
    let levels = r_decomp_levels::<F>(lp.log_basis_open);
    let witness_layout = instance.segment_layout(lp, None)?;
    if witness_layout.r_rows().len() != rows || witness_layout.quotient_depth() != levels {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    for (row, &row_dim) in witness_layout.r_rows().iter().zip(&quotient_row_dims) {
        if row.ring_dim() != row_dim {
            return Err(AkitaError::InvalidSetup(
                "relation quotient dimensions disagree with witness layout".into(),
            ));
        }
    }
    let live_witness_coeff_len = witness_layout.live_coeff_len();
    let physical_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
    if live_witness_coeff_len > physical_field_len {
        return Err(AkitaError::InvalidSize {
            expected: physical_field_len,
            actual: live_witness_coeff_len,
        });
    }
    let setup_matrix = match setup {
        RelationSetupSource::Matrix(setup) => Some(setup),
        RelationSetupSource::DeferredClaim => None,
    };
    let setup_is_deferred = setup_matrix.is_none();
    let d_column_ranges = if setup_matrix.is_some() {
        relation_d_column_ranges(lp, opening_batch)?
    } else {
        Vec::new()
    };
    let group_role_dims = (0..opening_batch.num_groups())
        .map(|group_index| lp.group_role_dims(opening_batch, group_index))
        .collect::<Result<Vec<_>, _>>()?;
    let relation_coefficient_block_len = RelationAddressGeometry::new_for_groups(
        role_dims,
        &group_role_dims,
        opening_ring_dim,
        live_witness_coeff_len,
    )?
    .relation_coefficient_block_len();
    let mut relation_events = RelationWeightEvents {
        events: Vec::new(),
        alpha_powers: scalar_powers(
            alpha,
            quotient_row_dims
                .iter()
                .copied()
                .max()
                .ok_or(AkitaError::InvalidProof)?,
        ),
        relation_coefficient_block_len,
        physical_field_len,
        setup_is_deferred,
    };
    let d_view = if let Some(setup) = setup_matrix {
        let d_physical_columns = d_column_ranges
            .iter()
            .map(|range| range.end)
            .max()
            .unwrap_or(0);
        Some(setup.shared_matrix.ring_view_dyn(
            lp.open_commit_matrix.output_rank(),
            d_physical_columns,
            d_d,
        )?)
    } else {
        None
    };
    let d_rows = if let Some(d_view) = &d_view {
        (0..lp.open_commit_matrix.output_rank())
            .map(|row| d_view.row_flat(row))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let d_start = row_families
        .iter()
        .position(|row| matches!(row, akita_types::RelationRowFamily::Opening { .. }))
        .ok_or(AkitaError::InvalidProof)?;
    for group_index in 0..opening_batch.num_groups() {
        let e_setup_offset = if setup_matrix.is_some() {
            d_column_ranges
                .get(group_index)
                .ok_or(AkitaError::InvalidProof)?
                .start
        } else {
            0
        };
        let group_lp = lp.group_params(opening_batch, group_index)?;
        let group_dims = lp.group_role_dims(opening_batch, group_index)?;
        let group_d_a = group_dims.d_a();
        let group_d_b = group_dims.d_b();
        let group_d_d = group_dims.d_d();
        let (b_ratio, d_ratio) = SetupProjectionGeometry::native_role_subcolumn_counts(group_dims)?;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_id = group_index;
        let units = witness_layout.units_for_group(group_id)?;
        let k_g = group_layout.num_polynomials();
        let ring_multiplier_point = instance.group_ring_multiplier_point(group_index)?;
        let challenges = &instance.group_challenges()[group_index];
        if ring_multiplier_point.position_len() != group_lp.num_positions_per_block()
            || ring_multiplier_point.fold_len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval multiplier layout mismatch".to_string(),
            ));
        }
        let total_blocks = k_g
            .checked_mul(group_lp.num_live_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if challenges.len() != total_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = group_lp.num_digits_fold();
        let log_basis_inner = group_lp.log_basis_inner();
        let log_basis_outer = group_lp.log_basis_outer();
        let log_basis_open = group_lp.log_basis_open();
        let n_a = group_lp.a_rows_len();
        let n_b = group_lp.b_rows_len();
        let inner_width = group_lp.a_col_len();
        // Hoist per-group geometry into `Copy` locals so the parallel closures
        // below capture scalars instead of the `!Sync` `&dyn LevelParamsLike`.
        let num_live_blocks_g = group_lp.num_live_blocks();
        let num_positions_per_block_g = group_lp.num_positions_per_block();
        let semantic_t_vector_width = n_a
            .checked_mul(depth_commit)
            .and_then(|len| len.checked_mul(num_live_blocks_g))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let t_vector_width = semantic_t_vector_width
            .checked_mul(b_ratio)
            .ok_or_else(|| {
                AkitaError::InvalidSetup("multi-group B vector width overflow".to_string())
            })?;
        let b_width = k_g
            .checked_mul(t_vector_width)
            .ok_or_else(|| AkitaError::InvalidSetup("setup B width overflow".to_string()))?;
        let setup_views = if let Some(setup) = setup_matrix {
            Some((
                setup
                    .shared_matrix
                    .ring_view_dyn(n_a, inner_width, group_d_a)?,
                setup.shared_matrix.ring_view_dyn(n_b, b_width, group_d_b)?,
            ))
        } else {
            None
        };
        let (setup_a_rows, b_rows) = if let Some((setup_a_view, b_view)) = &setup_views {
            let setup_a_rows = (0..n_a)
                .map(|row| setup_a_view.row_flat(row))
                .collect::<Result<Vec<_>, _>>()?;
            let b_rows = (0..n_b)
                .map(|row| b_view.row_flat(row))
                .collect::<Result<Vec<_>, _>>()?;
            (setup_a_rows, b_rows)
        } else {
            (Vec::new(), Vec::new())
        };
        let a_range = lp.a_row_range(opening_batch, group_index)?;
        let b_range = lp.commitment_row_range(opening_batch, group_index)?;
        let consistency_weight =
            eq_tau1.eval_at(lp.consistency_row_index(opening_batch, group_index)?)?;
        if a_range.end > eq_tau1.len() || b_range.end > eq_tau1.len() {
            return Err(AkitaError::InvalidProof);
        }
        let g_open: Vec<E> = gadget_row_scalars::<F>(depth_open, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let t_commit_gadget: Vec<E> = gadget_row_scalars::<F>(depth_commit, log_basis_outer)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let witness_gadget: Vec<E> = gadget_row_scalars::<F>(depth_witness, log_basis_inner)
            .into_iter()
            .map(E::lift_base)
            .collect();
        let fold_gadget: Vec<E> = gadget_row_scalars::<F>(depth_fold, log_basis_open)
            .into_iter()
            .map(E::lift_base)
            .collect();

        for claim in 0..k_g {
            for global_block in 0..num_live_blocks_g {
                let unit = witness_layout.unit_for_block(group_id, global_block)?;
                let challenge_index = claim
                    .checked_mul(num_live_blocks_g)
                    .and_then(|base| base.checked_add(global_block))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation challenge index overflow".into())
                    })?;
                let challenge_alpha =
                    challenges.eval_at_pows::<F, E>(challenge_index, &group_alpha_pows_a)?;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    for role_subcol in 0..d_ratio {
                        let physical_start = unit.e_coefficient_index(
                            group_d_a,
                            group_d_d,
                            k_g,
                            depth_open,
                            claim,
                            global_block,
                            role_subcol,
                            digit,
                            0,
                        )?;
                        let logical_block = claim * num_live_blocks_g + global_block;
                        let d_phys_col = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(role_subcol))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|local| e_setup_offset.checked_add(local))
                            .ok_or(AkitaError::InvalidProof)?;
                        let consistency_acc = consistency_weight * challenge_alpha * opening_gadget;
                        let mut setup_acc = E::zero();
                        for (di, d_row) in d_rows.iter().take(n_d_active).enumerate() {
                            let eq_i = eq_tau1.eval_at(d_start + di)?;
                            if !eq_i.is_zero() {
                                setup_acc += eq_i
                                    * eval_flat_ring_at_pows_fast(
                                        &d_row
                                            [d_phys_col * group_d_d..(d_phys_col + 1) * group_d_d],
                                        &group_alpha_pows_d,
                                    );
                            }
                        }
                        relation_events.push(
                            physical_start,
                            group_d_d,
                            role_subcol * group_d_d,
                            consistency_acc,
                            RelationWeightContribution::Constraint,
                        )?;
                        if setup_matrix.is_some() {
                            relation_events.push(
                                physical_start,
                                group_d_d,
                                0,
                                setup_acc,
                                RelationWeightContribution::SetupMatrix,
                            )?;
                        }
                    }
                }
                for a_idx in 0..n_a {
                    let a_row_weight = eq_tau1.eval_at(a_range.start + a_idx)?;
                    for (digit, &opening_gadget) in t_commit_gadget.iter().enumerate() {
                        let block_claim = num_live_blocks_g
                            .checked_mul(claim)
                            .and_then(|base| base.checked_add(global_block))
                            .ok_or(AkitaError::InvalidProof)?;
                        let row_block_claim = n_a
                            .checked_mul(block_claim)
                            .and_then(|base| base.checked_add(a_idx))
                            .ok_or(AkitaError::InvalidProof)?;
                        for role_subcol in 0..b_ratio {
                            let local_col = row_block_claim
                                .checked_mul(b_ratio)
                                .and_then(|base| base.checked_add(role_subcol))
                                .and_then(|base| base.checked_mul(depth_commit))
                                .and_then(|base| base.checked_add(digit))
                                .ok_or(AkitaError::InvalidProof)?;
                            let physical_start = unit.t_coefficient_index(
                                group_d_a,
                                group_d_b,
                                k_g,
                                n_a,
                                depth_commit,
                                claim,
                                global_block,
                                a_idx,
                                role_subcol,
                                digit,
                                0,
                            )?;
                            let a_acc = a_row_weight * challenge_alpha * opening_gadget;
                            let mut b_acc = E::zero();
                            for (row_idx, b_row) in b_rows.iter().take(n_b).enumerate() {
                                let eq_i = eq_tau1.eval_at(b_range.start + row_idx)?;
                                if !eq_i.is_zero() {
                                    b_acc += eq_i
                                        * eval_flat_ring_at_pows_fast(
                                            &b_row[local_col * group_d_b
                                                ..(local_col + 1) * group_d_b],
                                            &group_alpha_pows_b,
                                        );
                                }
                            }
                            relation_events.push(
                                physical_start,
                                group_d_b,
                                role_subcol * group_d_b,
                                a_acc,
                                RelationWeightContribution::Constraint,
                            )?;
                            if setup_matrix.is_some() {
                                relation_events.push(
                                    physical_start,
                                    group_d_b,
                                    0,
                                    b_acc,
                                    RelationWeightContribution::SetupMatrix,
                                )?;
                            }
                        }
                    }
                }
            }
        }

        // For z_hat[blk, dc, df], the column value is:
        //
        // -G_fold[df] * (
        //     tau_consistency * a_alpha[blk] * G_commit[dc]
        //     + sum_r tau_A[r] * A_alpha[r, blk, dc]
        //   ).
        //
        // The first term is the opening row. The second term is the A-row setup
        // contribution. A is already digit-domain, so the A-row setup term does
        // not multiply by G_commit.
        let z_bases = cfg_into_iter!(0..inner_width)
            .map(|k| {
                let block_idx = k / depth_witness;
                let digit_idx = k % depth_witness;
                let opening_a_eval =
                    ring_multiplier_point.eval_position_at::<E>(block_idx, &group_alpha_pows_a)?;
                let constraint = consistency_weight * opening_a_eval * witness_gadget[digit_idx];
                let mut setup = E::zero();
                for (a_idx, a_row) in setup_a_rows.iter().take(n_a).enumerate() {
                    let eq_i = eq_tau1.eval_at(a_range.start + a_idx)?;
                    if !eq_i.is_zero() {
                        setup += eq_i
                            * eval_flat_ring_at_pows_fast(
                                &a_row[k * group_d_a..(k + 1) * group_d_a],
                                &group_alpha_pows_a,
                            );
                    }
                }
                Ok((constraint, setup))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        for unit in units {
            for position in 0..num_positions_per_block_g {
                for commit_digit in 0..depth_witness {
                    for (fold_digit, &fold) in fold_gadget.iter().enumerate() {
                        let phys_k = position * depth_witness + commit_digit;
                        let physical_start = unit.z_coefficient_index(
                            group_d_a,
                            num_positions_per_block_g,
                            depth_witness,
                            depth_fold,
                            position,
                            commit_digit,
                            fold_digit,
                            0,
                        )?;
                        relation_events.push_native_ring(
                            physical_start,
                            group_d_a,
                            -(z_bases[phys_k].0 * fold),
                            RelationWeightContribution::Constraint,
                        )?;
                        if setup_matrix.is_some() {
                            relation_events.push_native_ring(
                                physical_start,
                                group_d_a,
                                -(z_bases[phys_k].1 * fold),
                                RelationWeightContribution::SetupMatrix,
                            )?;
                        }
                    }
                }
            }
        }
    }
    let r_gadget: Vec<E> = gadget_row_scalars::<F>(levels, lp.log_basis_open)
        .into_iter()
        .map(E::lift_base)
        .collect();
    for (row, &row_dim) in quotient_row_dims.iter().enumerate() {
        if matches!(
            row_families[row],
            RelationRowFamily::CompressionF { .. } | RelationRowFamily::CompressionH { .. }
        ) {
            continue;
        }
        let eq_weight = eq_tau1.eval_at(row)?;
        let row_alpha_pows = if row_dim == d_a {
            relation_events.alpha_powers.as_slice()
        } else if row_dim == d_b {
            alpha_pows_b.as_slice()
        } else if row_dim == d_d {
            alpha_pows_d.as_slice()
        } else {
            additional_quotient_alpha_powers
                .iter()
                .find_map(|(dimension, powers)| {
                    (*dimension == row_dim).then_some(powers.as_slice())
                })
                .ok_or(AkitaError::InvalidProof)?
        };
        let row_denom = row_alpha_pows[row_dim - 1] * alpha + E::one();
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let physical_start = witness_layout.r_coefficient_index(row, digit, 0)?;
            relation_events.push_native_ring(
                physical_start,
                row_dim,
                -(eq_weight * row_denom * *gadget),
                RelationWeightContribution::Constraint,
            )?;
        }
    }
    Ok(relation_events)
}

#[cfg(test)]
#[path = "relation_weights_tests.rs"]
mod tests;
