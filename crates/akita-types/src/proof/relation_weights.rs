//! Semantic relation-weight events and their canonical consumers.

use std::ops::Range;

use crate::layout::CommitmentRingDims;
use crate::proof::ring_relation::RingRelationInstance;
use crate::{
    gadget_row_scalars, r_decomp_levels, AkitaExpandedSetup, FpExtEncoding, LevelParams,
    OpeningClaimsLayout, RelationMatrixRowLayout, SetupProjectionGeometry,
};
use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
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
    pub level_params: &'a LevelParams,
    pub relation_row_point: &'a [E],
    pub claim_coefficients: &'a [E],
    pub relation_matrix_row_layout: RelationMatrixRowLayout,
    pub opening_source_len: usize,
    pub opening_ring_dim: usize,
}

/// Checked relation events plus the domain data needed by every consumer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightEvents<E: FieldCore> {
    events: Vec<RelationWeightEvent<E>>,
    inner_alpha_powers: Vec<E>,
    role_dims: CommitmentRingDims,
    opening_source_len: usize,
    opening_ring_dim: usize,
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
            || !physical_start.is_multiple_of(coefficient_count)
            || physical_end > self.physical_field_len
            || alpha_exponent_end > self.inner_alpha_powers.len()
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

    fn push_role(
        &mut self,
        witness_column: usize,
        role_subcolumn: usize,
        role_ring_dimension: usize,
        alpha_exponent_start: usize,
        scalar: E,
        contribution: RelationWeightContribution,
    ) -> Result<(), AkitaError> {
        let inner_ring_dimension = self.role_dims.d_a();
        if role_ring_dimension == 0
            || !inner_ring_dimension.is_multiple_of(role_ring_dimension)
            || role_subcolumn >= inner_ring_dimension / role_ring_dimension
        {
            return Err(AkitaError::InvalidProof);
        }
        let physical_start = witness_column
            .checked_mul(inner_ring_dimension)
            .and_then(|base| base.checked_add(role_subcolumn * role_ring_dimension))
            .ok_or_else(|| AkitaError::InvalidSetup("relation event address overflow".into()))?;
        self.push(
            physical_start,
            role_ring_dimension,
            alpha_exponent_start,
            scalar,
            contribution,
        )
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
        let opening_field_len = crate::opening_domain_len(self.opening_source_len)?
            .checked_mul(self.opening_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation weight length overflow".into()))?;
        let mut weights = vec![E::zero(); opening_field_len];
        for event in &self.events {
            for (offset, alpha_power) in self.inner_alpha_powers[event.alpha_exponent_start
                ..event.alpha_exponent_start + event.physical_coefficients.len()]
                .iter()
                .copied()
                .enumerate()
            {
                let physical = event.physical_coefficients.start + offset;
                let opening_column = crate::checked_opening_source_index(
                    self.opening_source_len,
                    physical / self.opening_ring_dim,
                )?;
                let opening_index = opening_column
                    .checked_mul(self.opening_ring_dim)
                    .and_then(|base| base.checked_add(physical % self.opening_ring_dim))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation weight address overflow".into())
                    })?;
                *weights
                    .get_mut(opening_index)
                    .ok_or(AkitaError::InvalidProof)? += event.scalar * alpha_power;
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
        let common_coefficient_count = self
            .role_dims
            .common_stage2_coefficient_count(self.opening_ring_dim);
        if common_coefficient_count == 0
            || !common_coefficient_count.is_power_of_two()
            || !self
                .role_dims
                .d_a()
                .is_multiple_of(common_coefficient_count)
            || !self
                .role_dims
                .d_b()
                .is_multiple_of(common_coefficient_count)
            || !self
                .role_dims
                .d_d()
                .is_multiple_of(common_coefficient_count)
            || !self
                .opening_ring_dim
                .is_multiple_of(common_coefficient_count)
        {
            return Err(AkitaError::InvalidSetup(
                "relation roles do not admit a common alpha factor".into(),
            ));
        }
        let opening_field_len = crate::opening_domain_len(self.opening_source_len)?
            .checked_mul(self.opening_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation lane length overflow".into()))?;
        let lane_count = opening_field_len / common_coefficient_count;
        let mut relation_lane_weights = vec![E::zero(); lane_count];
        for event in &self.events {
            if !event
                .physical_coefficients
                .start
                .is_multiple_of(common_coefficient_count)
                || !event
                    .physical_coefficients
                    .len()
                    .is_multiple_of(common_coefficient_count)
                || !event
                    .alpha_exponent_start
                    .is_multiple_of(common_coefficient_count)
            {
                return Err(AkitaError::InvalidSetup(
                    "relation event does not preserve the common alpha factor".into(),
                ));
            }
            for coefficient_offset in
                (0..event.physical_coefficients.len()).step_by(common_coefficient_count)
            {
                let physical = event.physical_coefficients.start + coefficient_offset;
                let opening_column = crate::checked_opening_source_index(
                    self.opening_source_len,
                    physical / self.opening_ring_dim,
                )?;
                let opening_coefficient = opening_column
                    .checked_mul(self.opening_ring_dim)
                    .and_then(|base| base.checked_add(physical % self.opening_ring_dim))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation lane address overflow".into())
                    })?;
                if !opening_coefficient.is_multiple_of(common_coefficient_count) {
                    return Err(AkitaError::InvalidSetup(
                        "opening layout breaks relation lane alignment".into(),
                    ));
                }
                let lane = opening_coefficient / common_coefficient_count;
                let alpha_exponent = event.alpha_exponent_start + coefficient_offset;
                let alpha_power = *self
                    .inner_alpha_powers
                    .get(alpha_exponent)
                    .ok_or(AkitaError::InvalidProof)?;
                *relation_lane_weights
                    .get_mut(lane)
                    .ok_or(AkitaError::InvalidProof)? += event.scalar * alpha_power;
            }
        }
        let common_alpha_factor = self
            .inner_alpha_powers
            .get(..common_coefficient_count)
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
        let opening_field_len = crate::opening_domain_len(self.opening_source_len)?
            .checked_mul(self.opening_ring_dim)
            .ok_or_else(|| AkitaError::InvalidSetup("relation weight length overflow".into()))?;
        let expected_point_len = opening_field_len.trailing_zeros() as usize;
        if !opening_field_len.is_power_of_two() || point.len() != expected_point_len {
            return Err(AkitaError::InvalidSize {
                expected: expected_point_len,
                actual: point.len(),
            });
        }

        let mut low_factor_cache = Vec::new();
        let mut evaluation = deferred_setup_claim.unwrap_or_else(E::zero);
        for event in &self.events {
            let coefficient_count = event.physical_coefficients.len();
            let low_variable_count = coefficient_count.trailing_zeros() as usize;
            let cache_key = (event.alpha_exponent_start, coefficient_count);
            let low_factor = if let Some((_, cached)) = low_factor_cache
                .iter()
                .find(|(cached_key, _)| *cached_key == cache_key)
            {
                *cached
            } else {
                let alpha_powers = &self.inner_alpha_powers
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
    lp: &LevelParams,
    opening_batch: &OpeningClaimsLayout,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params(opening_batch, group_index)?;
    let num_claims = opening_batch.group_layout(group_index)?.num_polynomials();
    num_claims
        .checked_mul(group_lp.num_live_blocks())
        .and_then(|n| n.checked_mul(group_lp.num_digits_open()))
        .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))
}

fn relation_d_column_ranges(
    lp: &LevelParams,
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
        relation_matrix_row_layout,
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
    let alpha_pows_a = scalar_powers(alpha, d_a);
    let alpha_pows_b = scalar_powers(alpha, d_b);
    let alpha_pows_d = scalar_powers(alpha, d_d);
    let rows =
        lp.relation_matrix_row_count_for(opening_batch.num_groups(), relation_matrix_row_layout)?;
    let eq_tau1 = SplitEqEvals::new(tau1)?;
    if eq_tau1.len() < rows {
        return Err(AkitaError::InvalidSize {
            expected: rows,
            actual: eq_tau1.len(),
        });
    }
    let n_d_active = lp.n_d_active_for(relation_matrix_row_layout);
    let levels = r_decomp_levels::<F>(lp.log_basis_open);
    let witness_layout = instance.segment_layout(lp, None)?;
    let expected_r_len = rows.checked_mul(levels).ok_or_else(|| {
        AkitaError::InvalidSetup("relation quotient witness width overflow".to_string())
    })?;
    if witness_layout.r_range().len() != expected_r_len {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    let (b_ratio, d_ratio) = SetupProjectionGeometry::witness_subcolumn_ratios(role_dims)?;
    let physical_field_len = witness_layout
        .total_len()
        .checked_mul(d_a)
        .ok_or_else(|| AkitaError::InvalidSetup("relation weight length overflow".into()))?;
    let expected_field_len = opening_source_len
        .checked_mul(opening_ring_dim)
        .ok_or_else(|| AkitaError::InvalidSetup("opening field length overflow".into()))?;
    if physical_field_len > expected_field_len {
        return Err(AkitaError::InvalidSize {
            expected: expected_field_len,
            actual: physical_field_len,
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
    let mut relation_events = RelationWeightEvents {
        events: Vec::new(),
        inner_alpha_powers: alpha_pows_a,
        role_dims,
        opening_source_len,
        opening_ring_dim,
        physical_field_len,
        setup_is_deferred,
    };
    let d_view = if let Some(setup) = setup_matrix {
        let d_physical_columns = d_column_ranges
            .iter()
            .map(|range| range.end)
            .max()
            .unwrap_or(0);
        let e_total = d_physical_columns
            .checked_mul(d_ratio)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))?;
        Some(
            setup
                .shared_matrix
                .ring_view_dyn(lp.d_key.row_len(), e_total, d_d)?,
        )
    } else {
        None
    };
    let d_rows = if let Some(d_view) = &d_view {
        (0..lp.d_key.row_len())
            .map(|row| d_view.row_flat(row))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let d_start = rows
        .checked_sub(n_d_active)
        .ok_or(AkitaError::InvalidProof)?;
    let consistency_weight = eq_tau1.eval_at(0)?;

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
        let group_layout = opening_batch.group_layout(group_index)?;
        let group_id = group_index;
        let units = witness_layout.units_for_group(group_id)?;
        let k_g = group_layout.num_polynomials();
        let opening_point = instance.group_opening_point(group_index)?;
        let ring_multiplier_point = instance.group_ring_multiplier_point(group_index)?;
        let challenges = &instance.group_challenges()[group_index];
        if opening_point.position_weights.len() != group_lp.num_positions_per_block()
            || opening_point.live_block_weights.len() != group_lp.num_live_blocks()
        {
            return Err(AkitaError::InvalidInput(
                "relation matrix col eval opening-point layout mismatch".to_string(),
            ));
        }
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
        if challenges.logical_len() != total_blocks {
            return Err(AkitaError::InvalidProof);
        }
        let depth_witness = group_lp.num_digits_inner();
        let depth_commit = group_lp.num_digits_outer();
        let depth_open = group_lp.num_digits_open();
        let depth_fold = lp.num_digits_fold_for_params(group_lp, k_g, lp.field_bits_for_cache())?;
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
                setup.shared_matrix.ring_view_dyn(n_a, inner_width, d_a)?,
                setup.shared_matrix.ring_view_dyn(n_b, b_width, d_b)?,
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
        let a_range = lp.a_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
        let b_range =
            lp.commitment_row_range(opening_batch, group_index, relation_matrix_row_layout)?;
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
                let challenge_alpha = challenges.eval_logical_at_pows::<F, E>(
                    challenge_index,
                    &relation_events.inner_alpha_powers,
                )?;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    let witness_col = unit.e_index(k_g, depth_open, claim, global_block, digit)?;
                    for role_subcol in 0..d_ratio {
                        let logical_block = claim * num_live_blocks_g + global_block;
                        let d_phys_col = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(role_subcol))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|local| {
                                e_setup_offset
                                    .checked_mul(d_ratio)
                                    .and_then(|offset| offset.checked_add(local))
                            })
                            .ok_or(AkitaError::InvalidProof)?;
                        let consistency_acc = consistency_weight * challenge_alpha * opening_gadget;
                        let mut setup_acc = E::zero();
                        for (di, d_row) in d_rows.iter().take(n_d_active).enumerate() {
                            let eq_i = eq_tau1.eval_at(d_start + di)?;
                            if !eq_i.is_zero() {
                                setup_acc += eq_i
                                    * eval_flat_ring_at_pows_fast(
                                        &d_row[d_phys_col * d_d..(d_phys_col + 1) * d_d],
                                        &alpha_pows_d,
                                    );
                            }
                        }
                        relation_events.push_role(
                            witness_col,
                            role_subcol,
                            d_d,
                            role_subcol * d_d,
                            consistency_acc,
                            RelationWeightContribution::Constraint,
                        )?;
                        if setup_matrix.is_some() {
                            relation_events.push_role(
                                witness_col,
                                role_subcol,
                                d_d,
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
                        let semantic_col = depth_commit
                            .checked_mul(row_block_claim)
                            .and_then(|base| base.checked_add(digit))
                            .ok_or(AkitaError::InvalidProof)?;
                        let witness_col = unit.t_index(
                            k_g,
                            n_a,
                            depth_commit,
                            claim,
                            global_block,
                            a_idx,
                            digit,
                        )?;
                        for role_subcol in 0..b_ratio {
                            let local_col = semantic_col
                                .checked_mul(b_ratio)
                                .and_then(|base| base.checked_add(role_subcol))
                                .ok_or(AkitaError::InvalidProof)?;
                            let a_acc = a_row_weight * challenge_alpha * opening_gadget;
                            let mut b_acc = E::zero();
                            for (row_idx, b_row) in b_rows.iter().take(n_b).enumerate() {
                                let eq_i = eq_tau1.eval_at(b_range.start + row_idx)?;
                                if !eq_i.is_zero() {
                                    b_acc += eq_i
                                        * eval_flat_ring_at_pows_fast(
                                            &b_row[local_col * d_b..(local_col + 1) * d_b],
                                            &alpha_pows_b,
                                        );
                                }
                            }
                            relation_events.push_role(
                                witness_col,
                                role_subcol,
                                d_b,
                                role_subcol * d_b,
                                a_acc,
                                RelationWeightContribution::Constraint,
                            )?;
                            if setup_matrix.is_some() {
                                relation_events.push_role(
                                    witness_col,
                                    role_subcol,
                                    d_b,
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
                let opening_a_eval = ring_multiplier_point
                    .eval_position_at_dyn::<E>(block_idx, &relation_events.inner_alpha_powers)?;
                let constraint = consistency_weight * opening_a_eval * witness_gadget[digit_idx];
                let mut setup = E::zero();
                for (a_idx, a_row) in setup_a_rows.iter().take(n_a).enumerate() {
                    let eq_i = eq_tau1.eval_at(a_range.start + a_idx)?;
                    if !eq_i.is_zero() {
                        setup += eq_i
                            * eval_flat_ring_at_pows_fast(
                                &a_row[k * d_a..(k + 1) * d_a],
                                &relation_events.inner_alpha_powers,
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
                        let witness_col = unit.z_index(
                            num_positions_per_block_g,
                            depth_witness,
                            depth_fold,
                            position,
                            commit_digit,
                            fold_digit,
                        )?;
                        relation_events.push_role(
                            witness_col,
                            0,
                            d_a,
                            0,
                            -(z_bases[phys_k].0 * fold),
                            RelationWeightContribution::Constraint,
                        )?;
                        if setup_matrix.is_some() {
                            relation_events.push_role(
                                witness_col,
                                0,
                                d_a,
                                0,
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
    for row in 0..rows {
        let eq_weight = eq_tau1.eval_at(row)?;
        let is_b_row = (0..opening_batch.num_groups()).try_fold(false, |found, group| {
            Ok::<_, AkitaError>(
                found
                    || lp
                        .commitment_row_range(opening_batch, group, relation_matrix_row_layout)?
                        .contains(&row),
            )
        })?;
        let (row_dim, row_alpha_pows): (usize, &[E]) = if row >= d_start {
            (d_d, alpha_pows_d.as_slice())
        } else if is_b_row {
            (d_b, alpha_pows_b.as_slice())
        } else {
            (d_a, relation_events.inner_alpha_powers.as_slice())
        };
        let row_denom = row_alpha_pows[row_dim - 1] * alpha + E::one();
        for (digit, gadget) in r_gadget.iter().enumerate() {
            let witness_col = witness_layout.r_index(levels, row, digit)?;
            relation_events.push_role(
                witness_col,
                0,
                row_dim,
                0,
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
