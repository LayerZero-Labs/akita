use std::ops::Range;

use akita_error::AkitaError;
use akita_types::{RelationWeightContribution, RelationWeightEvent};
use jolt_field::Field;

use super::split_disjoint_mut;

/// Relation weights accumulated per relation coefficient lane.
///
/// A contribution adds `scalar * alpha^(e + c)` to physical coefficient
/// `p + c` for every `c` below its width, where the first coefficient `p`, the
/// first alpha exponent `e`, and the width are multiples of the relation
/// coefficient block `L`. Coefficient `lane * L + c` with `c < L` then carries
/// `lanes[lane] * alpha^c`, so the contribution adds `scalar * alpha^(e + j L)`
/// to lane `p / L + j`. Overlapping contributions are intentionally additive.
pub(crate) struct RelationLaneWeights<E: Field> {
    lanes: Vec<E>,
    alpha_powers: Vec<E>,
    relation_coefficient_block_len: usize,
    setup_is_deferred: bool,
}

/// Exact common-alpha factorization of the padded relation-weight table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RelationWeightFactorization<E: Field> {
    common_alpha_factor: Vec<E>,
    relation_lane_weights: Vec<E>,
}

impl<E: Field> RelationWeightFactorization<E> {
    #[cfg(test)]
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
    pub(crate) fn common_alpha_factor(&self) -> &[E] {
        &self.common_alpha_factor
    }

    /// Relation weights after removing the shared low alpha factor.
    #[must_use]
    pub(crate) fn relation_lane_weights(&self) -> &[E] {
        &self.relation_lane_weights
    }

    pub(crate) fn common_alpha_factor_mut(&mut self) -> &mut Vec<E> {
        &mut self.common_alpha_factor
    }

    pub(crate) fn take_lane_weights(&mut self) -> Vec<E> {
        std::mem::take(&mut self.relation_lane_weights)
    }

    /// Expand this factorization over its complete padded flat domain.
    #[cfg(test)]
    pub(crate) fn materialize_dense(&self) -> Result<Vec<E>, AkitaError> {
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

impl<E: Field> RelationLaneWeights<E> {
    /// Empty weights over `physical_field_len` coefficients.
    ///
    /// `alpha_powers` must cover every contribution's alpha exponents.
    pub(super) fn new(
        alpha_powers: Vec<E>,
        relation_coefficient_block_len: usize,
        physical_field_len: usize,
        setup_is_deferred: bool,
    ) -> Result<Self, AkitaError> {
        let lane_capacity = physical_field_len
            .checked_div(relation_coefficient_block_len)
            .filter(|capacity| {
                capacity.is_power_of_two()
                    && physical_field_len.is_multiple_of(relation_coefficient_block_len)
            })
            .ok_or_else(|| AkitaError::InvalidSetup("relation lane capacity is invalid".into()))?;
        if alpha_powers.len() < relation_coefficient_block_len {
            return Err(AkitaError::InvalidSetup(
                "relation alpha powers do not cover one coefficient block".into(),
            ));
        }
        Ok(Self {
            lanes: vec![E::zero(); lane_capacity],
            alpha_powers,
            relation_coefficient_block_len,
            setup_is_deferred,
        })
    }

    #[must_use]
    pub(super) fn alpha_powers(&self) -> &[E] {
        &self.alpha_powers
    }

    /// `alpha^(j L)` for every lane-aligned exponent the alpha powers cover.
    #[must_use]
    pub(super) fn lane_alpha_powers(&self) -> Vec<E> {
        self.alpha_powers
            .iter()
            .step_by(self.relation_coefficient_block_len)
            .copied()
            .collect()
    }

    /// Accumulated lanes; a deferred setup claim leaves out setup terms.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn lanes(&self) -> &[E] {
        &self.lanes
    }

    fn lane_range(
        &self,
        physical_start: usize,
        coefficient_count: usize,
    ) -> Result<Range<usize>, AkitaError> {
        aligned_lane_range(
            physical_start,
            coefficient_count,
            self.relation_coefficient_block_len,
        )
        .filter(|lanes| lanes.end <= self.lanes.len())
        .ok_or_else(unaligned_event)
    }

    /// Accumulated lanes, for scatter windows.
    pub(super) fn lanes_mut(&mut self) -> &mut [E] {
        &mut self.lanes
    }

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
        let alpha_exponent_end = alpha_exponent_start
            .checked_add(coefficient_count)
            .ok_or_else(|| AkitaError::InvalidSetup("relation alpha range overflow".into()))?;
        if !coefficient_count.is_power_of_two()
            || !alpha_exponent_start.is_multiple_of(self.relation_coefficient_block_len)
            || alpha_exponent_end > self.alpha_powers.len()
            || (self.setup_is_deferred && contribution == RelationWeightContribution::SetupMatrix)
        {
            return Err(unaligned_event());
        }
        let range = self.lane_range(physical_start, coefficient_count)?;
        let alpha_powers = self.alpha_powers[alpha_exponent_start..alpha_exponent_end]
            .iter()
            .step_by(self.relation_coefficient_block_len);
        for (lane, &alpha_power) in self.lanes[range].iter_mut().zip(alpha_powers) {
            *lane += scalar * alpha_power;
        }
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

    /// Split the weights into the shared low alpha factor and the lanes.
    pub(crate) fn into_factorization(self) -> Result<RelationWeightFactorization<E>, AkitaError> {
        if self.setup_is_deferred {
            return Err(AkitaError::InvalidSetup(
                "relation factorization requires direct setup contributions".into(),
            ));
        }
        let mut common_alpha_factor = self.alpha_powers;
        common_alpha_factor.truncate(self.relation_coefficient_block_len);
        Ok(RelationWeightFactorization {
            common_alpha_factor,
            relation_lane_weights: self.lanes,
        })
    }
}

/// Disjoint windows of `lanes`, a table of `lane_len`-coefficient lanes, over
/// the aligned physical coefficient `extents`, in the order of `extents`. An
/// empty extent has an empty window.
pub(super) fn lane_windows_mut<'a, E>(
    lanes: &'a mut [E],
    extents: &[Range<usize>],
    lane_len: usize,
) -> Result<Vec<LaneWindow<'a, E>>, AkitaError> {
    let lane_ranges = extents
        .iter()
        .map(|extent| {
            if extent.is_empty() {
                Ok(0..0)
            } else {
                aligned_lane_range(extent.start, extent.len(), lane_len).ok_or_else(unaligned_event)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(split_disjoint_mut(lanes, &lane_ranges)?
        .into_iter()
        .zip(lane_ranges)
        .map(|(lanes, range)| LaneWindow {
            lanes,
            first_lane: range.start,
            relation_coefficient_block_len: lane_len,
        })
        .collect())
}

/// Mutable lanes of one aligned physical coefficient extent. A dense
/// coefficient table is the one-coefficient-lane case.
pub(super) struct LaneWindow<'a, E> {
    lanes: &'a mut [E],
    first_lane: usize,
    relation_coefficient_block_len: usize,
}

impl<E> LaneWindow<'_, E> {
    /// Lanes of the `coefficient_count` coefficients from `physical_start`,
    /// which must lie inside this window.
    pub(super) fn lanes_mut(
        &mut self,
        physical_start: usize,
        coefficient_count: usize,
    ) -> Result<&mut [E], AkitaError> {
        aligned_lane_range(
            physical_start,
            coefficient_count,
            self.relation_coefficient_block_len,
        )
        .and_then(|lanes| {
            let start = lanes.start.checked_sub(self.first_lane)?;
            self.lanes.get_mut(start..start + lanes.len())
        })
        .ok_or_else(|| AkitaError::InvalidSetup("relation address lies outside its window".into()))
    }
}

/// Lanes of `coefficient_count > 0` coefficients from `physical_start`, both
/// multiples of the relation coefficient block.
fn aligned_lane_range(
    physical_start: usize,
    coefficient_count: usize,
    relation_coefficient_block_len: usize,
) -> Option<Range<usize>> {
    let first = physical_start / relation_coefficient_block_len;
    (coefficient_count != 0
        && physical_start.is_multiple_of(relation_coefficient_block_len)
        && coefficient_count.is_multiple_of(relation_coefficient_block_len))
    .then(|| first.checked_add(coefficient_count / relation_coefficient_block_len))
    .flatten()
    .map(|end| first..end)
}

fn unaligned_event() -> AkitaError {
    AkitaError::InvalidSetup("relation event is unaligned or outside its checked domain".into())
}
