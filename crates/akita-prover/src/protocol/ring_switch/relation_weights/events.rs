use akita_algebra::offset_eq::{eq_eval_at_index, OffsetEqWindow};
use akita_algebra::poly::multilinear_eval;
use akita_error::AkitaError;
use akita_types::{RelationWeightContribution, RelationWeightEvent};
use jolt_field::Field;

/// Checked relation events plus the domain data needed by every consumer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightEvents<E: Field> {
    pub(super) events: Vec<RelationWeightEvent<E>>,
    pub(super) alpha_powers: Vec<E>,
    pub(super) relation_coefficient_block_len: usize,
    pub(super) physical_field_len: usize,
    pub(super) setup_is_deferred: bool,
}

/// Exact common-alpha factorization of the padded relation-weight table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelationWeightFactorization<E: Field> {
    common_alpha_factor: Vec<E>,
    relation_lane_weights: Vec<E>,
}

impl<E: Field> RelationWeightFactorization<E> {
    pub(crate) fn new(
        common_alpha_factor: Vec<E>,
        relation_lane_weights: Vec<E>,
    ) -> Result<Self, AkitaError> {
        if common_alpha_factor.is_empty()
            || !common_alpha_factor.len().is_power_of_two()
            || relation_lane_weights.is_empty()
            || !relation_lane_weights.len().is_power_of_two()
        {
            return Err(AkitaError::InvalidSetup(
                "relation factorization dimensions must be nonzero powers of two".into(),
            ));
        }
        Ok(Self {
            common_alpha_factor,
            relation_lane_weights,
        })
    }

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

    pub(crate) fn components_mut(&mut self) -> (&mut Vec<E>, &mut Vec<E>) {
        (
            &mut self.common_alpha_factor,
            &mut self.relation_lane_weights,
        )
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

impl<E: Field> RelationWeightEvents<E> {
    pub(super) fn push(
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
        self.events.push(RelationWeightEvent::new(
            physical_start..physical_end,
            alpha_exponent_start,
            scalar,
            contribution,
        )?);
        Ok(())
    }

    pub(super) fn extend_events(
        &mut self,
        events: impl IntoIterator<Item = RelationWeightEvent<E>>,
    ) -> Result<(), AkitaError> {
        for event in events {
            self.push(
                event.physical_coefficients().start,
                event.physical_coefficients().len(),
                event.alpha_exponent_start(),
                event.scalar(),
                event.contribution(),
            )?;
        }
        Ok(())
    }

    pub(super) fn push_native_ring(
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
            let coefficients = event.physical_coefficients();
            for (offset, alpha_power) in self.alpha_powers
                [event.alpha_exponent_start()..event.alpha_exponent_start() + coefficients.len()]
                .iter()
                .copied()
                .enumerate()
            {
                let physical = coefficients.start + offset;
                *weights.get_mut(physical).ok_or(AkitaError::InvalidProof)? +=
                    event.scalar() * alpha_power;
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
            let coefficients = event.physical_coefficients();
            if !coefficients.start.is_multiple_of(coeff_count)
                || !coefficients.len().is_multiple_of(coeff_count)
                || !event.alpha_exponent_start().is_multiple_of(coeff_count)
            {
                return Err(AkitaError::InvalidSetup(
                    "relation event does not preserve the common alpha factor".into(),
                ));
            }
            for coefficient_offset in (0..coefficients.len()).step_by(coeff_count) {
                let physical = coefficients.start + coefficient_offset;
                if !physical.is_multiple_of(coeff_count) {
                    return Err(AkitaError::InvalidSetup(
                        "flat relation layout breaks relation lane alignment".into(),
                    ));
                }
                let lane = physical / coeff_count;
                let alpha_exponent = event.alpha_exponent_start() + coefficient_offset;
                let alpha_power = *self
                    .alpha_powers
                    .get(alpha_exponent)
                    .ok_or(AkitaError::InvalidProof)?;
                *relation_lane_weights
                    .get_mut(lane)
                    .ok_or(AkitaError::InvalidProof)? += event.scalar() * alpha_power;
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
            let coefficients = event.physical_coefficients();
            let coefficient_count = coefficients.len();
            if !coefficients.start.is_multiple_of(coefficient_count) {
                let alpha_powers = &self.alpha_powers[event.alpha_exponent_start()
                    ..event.alpha_exponent_start() + coefficient_count];
                let interval = alpha_powers.iter().copied().enumerate().fold(
                    E::zero(),
                    |sum, (offset, alpha_power)| {
                        sum + alpha_power * equality.eval(coefficients.start + offset)
                    },
                );
                evaluation += event.scalar() * interval;
                continue;
            }
            let low_variable_count = coefficient_count.trailing_zeros() as usize;
            let cache_key = (event.alpha_exponent_start(), coefficient_count);
            let low_factor = if let Some((_, cached)) = low_factor_cache
                .iter()
                .find(|(cached_key, _)| *cached_key == cache_key)
            {
                *cached
            } else {
                let alpha_powers = &self.alpha_powers[event.alpha_exponent_start()
                    ..event.alpha_exponent_start() + coefficient_count];
                let factor = multilinear_eval(alpha_powers, &point[..low_variable_count])?;
                low_factor_cache.push((cache_key, factor));
                factor
            };
            let high_index = coefficients.start >> low_variable_count;
            let high_factor = eq_eval_at_index(&point[low_variable_count..], high_index);
            evaluation += event.scalar() * low_factor * high_factor;
        }
        Ok(evaluation)
    }
}
