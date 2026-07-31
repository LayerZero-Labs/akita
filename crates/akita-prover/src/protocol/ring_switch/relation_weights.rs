//! Semantic relation-weight events and their canonical consumers.

use std::ops::Range;

/// One family of setup-matrix rows read as per-column ring slices.
///
/// `Flat` borrows the materialized store; `Seed` derives each touched slice
/// on demand — the weight events read sparse column slices strided across
/// the FULL matrix width, so no retained prefix can serve them once the
/// store is released, and materializing the whole extent for a handful of
/// slices is exactly the residency this path is trying to avoid.
#[allow(clippy::large_enum_variant)]
enum SetupRowFamily<'a, F: akita_field::FieldCore> {
    Flat {
        rows: Vec<&'a [F]>,
        ring_d: usize,
    },
    Seed {
        deriver: akita_types::MatrixElementDeriver<F>,
        row_width_rings: usize,
        ring_d: usize,
    },
}

impl<F: akita_field::FieldCore> SetupRowFamily<'_, F> {
    fn ring_slice(&self, row: usize, col: usize) -> Result<std::borrow::Cow<'_, [F]>, AkitaError> {
        match self {
            Self::Flat { rows, ring_d } => rows
                .get(row)
                .and_then(|row| row.get(col * ring_d..(col + 1) * ring_d))
                .map(std::borrow::Cow::Borrowed)
                .ok_or(AkitaError::InvalidProof),
            Self::Seed {
                deriver,
                row_width_rings,
                ring_d,
            } => {
                let flat_ring = row
                    .checked_mul(*row_width_rings)
                    .and_then(|base| base.checked_add(col))
                    .ok_or(AkitaError::InvalidProof)?;
                Ok(std::borrow::Cow::Owned(
                    deriver.coeff_range(flat_ring * ring_d, *ring_d),
                ))
            }
        }
    }
}

use akita_algebra::eq_poly::SplitEqEvals;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::{eval_flat_ring_at_pows_fast, scalar_powers};
use akita_field::parallel::*;
use akita_field::{
    AkitaError, CanonicalField, FieldCore, FromPrimitiveInt, LiftBase, MulBase, MulBaseUnreduced,
};
use akita_types::{
    checked_opening_source_index, gadget_row_scalars, opening_domain_len, r_decomp_levels,
    relation_rhs_layout_for, AkitaExpandedSetup, CommitmentRingDims, CommittedGroupParams,
    FpExtEncoding, OpeningClaimsLayout, RelationAddressGeometry, RingRelationInstance,
    SetupProjectionGeometry,
};

fn evaluate_setup_columns<F, E>(
    family: &SetupRowFamily<'_, F>,
    columns: Range<usize>,
    row_weights: &[(usize, E)],
    alpha_powers: &[E],
) -> Result<Vec<E>, AkitaError>
where
    F: FieldCore,
    E: FieldCore + MulBaseUnreduced<F>,
{
    cfg_into_iter!(columns)
        .map(|col| {
            row_weights
                .iter()
                .try_fold(E::zero(), |acc, &(row, weight)| {
                    Ok(acc
                        + weight
                            * eval_flat_ring_at_pows_fast(
                                family.ring_slice(row, col)?.as_ref(),
                                alpha_powers,
                            ))
                })
        })
        .collect()
}

/// Whether one relation event belongs to the protocol constraint or setup matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationWeightContribution {
    /// Consistency, A-row, opening, and quotient-denominator arithmetic.
    Constraint,
    /// D/B/A setup-matrix arithmetic replaceable by one offloaded setup claim.
    SetupMatrix,
    /// Constraint and setup-matrix terms sharing one physical alpha interval.
    ConstraintAndSetupMatrix,
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
    inner_alpha_powers: Vec<E>,
    role_dims: CommitmentRingDims,
    group_role_dims: Vec<CommitmentRingDims>,
    carrier_ring_dimension: usize,
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
            || (self.setup_is_deferred && contribution != RelationWeightContribution::Constraint)
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
        let inner_ring_dimension = self.carrier_ring_dimension;
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

    #[allow(clippy::too_many_arguments)]
    fn push_role_with_setup(
        &mut self,
        witness_column: usize,
        role_subcolumn: usize,
        role_ring_dimension: usize,
        constraint_alpha_exponent_start: usize,
        constraint_scalar: E,
        setup_scalar: Option<E>,
    ) -> Result<(), AkitaError> {
        match setup_scalar {
            Some(setup_scalar) if constraint_alpha_exponent_start == 0 => self.push_role(
                witness_column,
                role_subcolumn,
                role_ring_dimension,
                0,
                constraint_scalar + setup_scalar,
                RelationWeightContribution::ConstraintAndSetupMatrix,
            ),
            Some(setup_scalar) => {
                self.push_role(
                    witness_column,
                    role_subcolumn,
                    role_ring_dimension,
                    constraint_alpha_exponent_start,
                    constraint_scalar,
                    RelationWeightContribution::Constraint,
                )?;
                self.push_role(
                    witness_column,
                    role_subcolumn,
                    role_ring_dimension,
                    0,
                    setup_scalar,
                    RelationWeightContribution::SetupMatrix,
                )
            }
            None => self.push_role(
                witness_column,
                role_subcolumn,
                role_ring_dimension,
                constraint_alpha_exponent_start,
                constraint_scalar,
                RelationWeightContribution::Constraint,
            ),
        }
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
        let opening_field_len = opening_domain_len(self.opening_source_len)?
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
                let opening_column = checked_opening_source_index(
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
        let geometry = RelationAddressGeometry::new_for_groups(
            self.role_dims,
            &self.group_role_dims,
            self.opening_ring_dim,
            self.opening_source_len,
        )?;
        let coeff_count = geometry.common_relation_witness_coeff_count();
        let mut relation_lane_weights = vec![E::zero(); geometry.relation_lane_capacity()];
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
                let opening_column = checked_opening_source_index(
                    self.opening_source_len,
                    physical / self.opening_ring_dim,
                )?;
                let opening_coefficient = opening_column
                    .checked_mul(self.opening_ring_dim)
                    .and_then(|base| base.checked_add(physical % self.opening_ring_dim))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation lane address overflow".into())
                    })?;
                if !opening_coefficient.is_multiple_of(coeff_count) {
                    return Err(AkitaError::InvalidSetup(
                        "opening layout breaks relation lane alignment".into(),
                    ));
                }
                let lane = opening_coefficient / coeff_count;
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
        RelationAddressGeometry::new_for_groups(
            self.role_dims,
            &self.group_role_dims,
            self.opening_ring_dim,
            self.opening_source_len,
        )?
        .validate_relation_point_len(point.len())?;

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
    lp: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    group_index: usize,
) -> Result<usize, AkitaError> {
    let group_lp = lp.group_params(opening_batch, group_index)?;
    let group_dims = lp.group_role_dims(opening_batch, group_index)?;
    let (_, d_subcolumns) = SetupProjectionGeometry::a_carrier_subcolumn_counts(group_dims)?;
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
    let quotient_row_dims = relation_rhs_layout.row_ring_dims()?;
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
    let carrier_ring_dimension = lp.relation_witness_carrier_ring_dimension();
    let expected_r_len = rows.checked_mul(levels).ok_or_else(|| {
        AkitaError::InvalidSetup("relation quotient witness width overflow".to_string())
    })?;
    if witness_layout.r_range().len() != expected_r_len {
        return Err(AkitaError::InvalidSetup(
            "relation matrix dimensions disagree with witness layout".to_string(),
        ));
    }
    let physical_field_len = witness_layout
        .total_len()
        .checked_mul(carrier_ring_dimension)
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
        inner_alpha_powers: scalar_powers(alpha, carrier_ring_dimension),
        role_dims,
        group_role_dims: (0..opening_batch.num_groups())
            .map(|group_index| lp.group_role_dims(opening_batch, group_index))
            .collect::<Result<Vec<_>, _>>()?,
        carrier_ring_dimension,
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
        let rank = lp.open_commit_matrix.output_rank();
        let extent = rank
            .checked_mul(d_physical_columns)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D extent overflow".to_string()))?;
        Some((
            setup
                .shared_matrix
                .materialized_covering_at_dyn(extent, d_d),
            rank,
            d_physical_columns,
        ))
    } else {
        None
    };
    let d_family = match &d_view {
        Some((Some(matrix), rows, cols)) => {
            let view = matrix.ring_view_dyn(*rows, *cols, d_d)?;
            Some(SetupRowFamily::Flat {
                rows: (0..*rows)
                    .map(|row| view.row_flat(row))
                    .collect::<Result<Vec<_>, _>>()?,
                ring_d: d_d,
            })
        }
        Some((None, _, cols)) => Some(SetupRowFamily::Seed {
            deriver: setup_matrix
                .ok_or(AkitaError::InvalidProof)?
                .shared_matrix
                .element_deriver(),
            row_width_rings: *cols,
            ring_d: d_d,
        }),
        None => None,
    };
    let d_start = rows
        .checked_sub(n_d_active)
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
        let (b_ratio, d_ratio) = SetupProjectionGeometry::a_carrier_subcolumn_counts(group_dims)?;
        let group_alpha_pows_a = scalar_powers(alpha, group_d_a);
        let group_alpha_pows_b = scalar_powers(alpha, group_d_b);
        let group_alpha_pows_d = scalar_powers(alpha, group_d_d);
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
        let setup_arcs = if let Some(setup) = setup_matrix {
            let a_extent = n_a
                .checked_mul(inner_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup A extent overflow".to_string()))?;
            let b_extent = n_b
                .checked_mul(b_width)
                .ok_or_else(|| AkitaError::InvalidSetup("setup B extent overflow".to_string()))?;
            Some((
                setup
                    .shared_matrix
                    .materialized_covering_at_dyn(a_extent, group_d_a),
                setup
                    .shared_matrix
                    .materialized_covering_at_dyn(b_extent, group_d_b),
            ))
        } else {
            None
        };
        let (setup_a_family, b_family) = if let Some((a_matrix, b_matrix)) = &setup_arcs {
            let setup_shared = setup_matrix
                .map(|setup| &setup.shared_matrix)
                .ok_or(AkitaError::InvalidProof)?;
            let a_family = match a_matrix {
                Some(matrix) => {
                    let view = matrix.ring_view_dyn(n_a, inner_width, group_d_a)?;
                    SetupRowFamily::Flat {
                        rows: (0..n_a)
                            .map(|row| view.row_flat(row))
                            .collect::<Result<Vec<_>, _>>()?,
                        ring_d: group_d_a,
                    }
                }
                None => SetupRowFamily::Seed {
                    deriver: setup_shared.element_deriver(),
                    row_width_rings: inner_width,
                    ring_d: group_d_a,
                },
            };
            let b_family = match b_matrix {
                Some(matrix) => {
                    let view = matrix.ring_view_dyn(n_b, b_width, group_d_b)?;
                    SetupRowFamily::Flat {
                        rows: (0..n_b)
                            .map(|row| view.row_flat(row))
                            .collect::<Result<Vec<_>, _>>()?,
                        ring_d: group_d_b,
                    }
                }
                None => SetupRowFamily::Seed {
                    deriver: setup_shared.element_deriver(),
                    row_width_rings: b_width,
                    ring_d: group_d_b,
                },
            };
            (Some(a_family), Some(b_family))
        } else {
            (None, None)
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
        let d_setup_start = e_setup_offset;
        let d_setup_len = total_blocks
            .checked_mul(d_ratio)
            .and_then(|len| len.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("setup D width overflow".to_string()))?;
        let d_setup_end = d_setup_start
            .checked_add(d_setup_len)
            .ok_or_else(|| AkitaError::InvalidSetup("setup D extent overflow".to_string()))?;
        let d_setup_accs = if let Some(d_family) = &d_family {
            let _span = tracing::info_span!("relation_weight_d_setup_columns").entered();
            let row_weights = (0..n_d_active)
                .map(|row| Ok((row, eq_tau1.eval_at(d_start + row)?)))
                .filter_map(|result| match result {
                    Ok((_, weight)) if weight.is_zero() => None,
                    other => Some(other),
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Some(evaluate_setup_columns(
                d_family,
                d_setup_start..d_setup_end,
                &row_weights,
                &group_alpha_pows_d,
            )?)
        } else {
            None
        };
        let b_setup_accs = if let Some(b_family) = &b_family {
            let _span = tracing::info_span!("relation_weight_b_setup_columns").entered();
            let row_weights = (0..n_b)
                .map(|row| Ok((row, eq_tau1.eval_at(b_range.start + row)?)))
                .filter_map(|result| match result {
                    Ok((_, weight)) if weight.is_zero() => None,
                    other => Some(other),
                })
                .collect::<Result<Vec<_>, AkitaError>>()?;
            Some(evaluate_setup_columns(
                b_family,
                0..b_width,
                &row_weights,
                &group_alpha_pows_b,
            )?)
        } else {
            None
        };

        for claim in 0..k_g {
            for global_block in 0..num_live_blocks_g {
                let unit = witness_layout.unit_for_block(group_id, global_block)?;
                let challenge_index = claim
                    .checked_mul(num_live_blocks_g)
                    .and_then(|base| base.checked_add(global_block))
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("relation challenge index overflow".into())
                    })?;
                let challenge_alpha = challenges
                    .eval_logical_at_pows::<F, E>(challenge_index, &group_alpha_pows_a)?;
                for (digit, &opening_gadget) in g_open.iter().enumerate() {
                    let witness_col = unit.e_index(k_g, depth_open, claim, global_block, digit)?;
                    for role_subcol in 0..d_ratio {
                        let logical_block = claim * num_live_blocks_g + global_block;
                        let d_phys_col = logical_block
                            .checked_mul(d_ratio)
                            .and_then(|base| base.checked_add(role_subcol))
                            .and_then(|base| base.checked_mul(depth_open))
                            .and_then(|base| base.checked_add(digit))
                            .and_then(|local| e_setup_offset.checked_add(local))
                            .ok_or(AkitaError::InvalidProof)?;
                        let consistency_acc = consistency_weight * challenge_alpha * opening_gadget;
                        let setup_acc = d_setup_accs
                            .as_ref()
                            .map_or(E::zero(), |weights| weights[d_phys_col - d_setup_start]);
                        relation_events.push_role_with_setup(
                            witness_col,
                            role_subcol,
                            group_d_d,
                            role_subcol * group_d_d,
                            consistency_acc,
                            d_setup_accs.as_ref().map(|_| setup_acc),
                        )?;
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
                            let b_acc = b_setup_accs
                                .as_ref()
                                .map_or(E::zero(), |weights| weights[local_col]);
                            relation_events.push_role_with_setup(
                                witness_col,
                                role_subcol,
                                group_d_b,
                                role_subcol * group_d_b,
                                a_acc,
                                b_setup_accs.as_ref().map(|_| b_acc),
                            )?;
                        }
                    }
                }
            }
        }
        drop(d_setup_accs);
        drop(b_setup_accs);

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
                    .eval_position_at_dyn::<E>(block_idx, &group_alpha_pows_a)?;
                let constraint = consistency_weight * opening_a_eval * witness_gadget[digit_idx];
                let mut setup = E::zero();
                if let Some(setup_a_family) = &setup_a_family {
                    for a_idx in 0..n_a {
                        let eq_i = eq_tau1.eval_at(a_range.start + a_idx)?;
                        if !eq_i.is_zero() {
                            setup += eq_i
                                * eval_flat_ring_at_pows_fast(
                                    setup_a_family.ring_slice(a_idx, k)?.as_ref(),
                                    &group_alpha_pows_a,
                                );
                        }
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
                        relation_events.push_role_with_setup(
                            witness_col,
                            0,
                            group_d_a,
                            0,
                            -(z_bases[phys_k].0 * fold),
                            setup_matrix
                                .is_some()
                                .then_some(-(z_bases[phys_k].1 * fold)),
                        )?;
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
        let eq_weight = eq_tau1.eval_at(row)?;
        let row_alpha_pows = if row_dim == d_a {
            relation_events.inner_alpha_powers.as_slice()
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
