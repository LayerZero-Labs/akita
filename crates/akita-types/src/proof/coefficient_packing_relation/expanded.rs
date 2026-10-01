use super::*;
#[cfg(test)]
use std::sync::Arc;

/// Checked inputs for one coefficient-packing group's shared relation semantics.
pub(super) struct CoefficientPackingGroupSemanticInputs<'authority, 'output, F: Field, E: Field> {
    pub level_params: &'authority CommittedGroupParams,
    pub opening_batch: &'authority OpeningClaimsLayout,
    pub relation_plan: &'authority RelationRangeImagePlan,
    pub relation: &'authority RingRelationInstance<F>,
    pub group_index: usize,
    pub prepared_point: &'output PreparedSubringCoefficientPackingPoint<E>,
    pub alpha: E,
    pub tau1: &'authority [E],
    /// Global claim coefficients in authenticated opening-batch order.
    pub claim_coefficients: &'output [E],
}

/// Packing-specific E and quotient events over the checked flat witness domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CoefficientPackingRelationEvents<E: Field> {
    pub(super) events: Vec<RelationWeightEvent<E>>,
    #[cfg(test)]
    pub(super) alpha_powers: Arc<[E]>,
    #[cfg(test)]
    pub(super) relation_coefficient_block_len: usize,
    #[cfg(test)]
    pub(super) physical_field_len: usize,
}

#[cfg(test)]
impl<E: Field> CoefficientPackingRelationEvents<E> {
    #[must_use]
    pub(super) fn events(&self) -> &[RelationWeightEvent<E>] {
        &self.events
    }

    /// Canonical powers of the alpha used to prepare every event scalar.
    #[must_use]
    pub(super) fn alpha_powers(&self) -> &[E] {
        &self.alpha_powers
    }

    #[must_use]
    pub(super) const fn relation_coefficient_block_len(&self) -> usize {
        self.relation_coefficient_block_len
    }

    #[must_use]
    pub(super) const fn physical_field_len(&self) -> usize {
        self.physical_field_len
    }

    /// Evaluate the sparse packing E and quotient events at one flat point.
    ///
    /// The returned value already includes every event's alpha powers. A
    /// caller that separately contracts the native common-alpha factor must
    /// add this value afterwards, without multiplying by that factor again.
    pub(super) fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
        let point_variables = u32::try_from(point.len())
            .map_err(|_| AkitaError::InvalidSetup("packing point domain overflow".into()))?;
        let expected = 1usize
            .checked_shl(point_variables)
            .ok_or_else(|| AkitaError::InvalidSetup("packing point domain overflow".into()))?;
        let padded_field_len = self
            .physical_field_len
            .checked_next_power_of_two()
            .ok_or_else(|| AkitaError::InvalidSetup("packing field domain overflow".into()))?;
        if expected != padded_field_len {
            return Err(AkitaError::InvalidSize {
                expected: padded_field_len.trailing_zeros() as usize,
                actual: point.len(),
            });
        }
        let block = self.relation_coefficient_block_len;
        let low_variables = block.trailing_zeros() as usize;
        let equality = OffsetEqWindow::new(point.get(low_variables..).ok_or_else(|| {
            AkitaError::Internal("packing high point exceeds the checked point dimension".into())
        })?)?;
        let mut work = 0usize;
        for event in &self.events {
            work = work
                .checked_add(event.physical_coefficients().len() / block)
                .ok_or_else(|| AkitaError::InvalidSetup("packing event work overflow".into()))?;
        }
        if work > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: work,
            });
        }
        let alpha_block_count = self.alpha_powers.len() / block;
        let mut alpha_cache = Vec::new();
        alpha_cache
            .try_reserve_exact(alpha_block_count)
            .map_err(|_| {
                AkitaError::InvalidInput("packing alpha cache allocation failed".into())
            })?;
        for alpha_block in 0..alpha_block_count {
            let alpha_start = alpha_block
                .checked_mul(block)
                .ok_or_else(|| AkitaError::InvalidSetup("packing alpha range overflow".into()))?;
            let alpha_end = alpha_start
                .checked_add(block)
                .ok_or_else(|| AkitaError::InvalidSetup("packing alpha range overflow".into()))?;
            alpha_cache.push(multilinear_eval(
                self.alpha_powers
                    .get(alpha_start..alpha_end)
                    .ok_or_else(|| {
                        AkitaError::Internal(
                            "packing alpha block exceeds the generated power table".into(),
                        )
                    })?,
                &point[..low_variables],
            )?);
        }
        let evaluate_event = |sum: Result<E, AkitaError>, event_index: usize| {
            let sum = sum?;
            let event = self.events.get(event_index).ok_or_else(|| {
                AkitaError::Internal("packing event index exceeds the generated event table".into())
            })?;
            let coefficients = event.physical_coefficients();
            if !coefficients.start.is_multiple_of(block)
                || !coefficients.len().is_multiple_of(block)
                || !event.alpha_exponent_start().is_multiple_of(block)
            {
                return Err(AkitaError::Internal(
                    "packing event is not aligned to its coefficient block".into(),
                ));
            }
            (0..coefficients.len())
                .step_by(block)
                .try_fold(sum, |acc, coefficient_offset| {
                    let alpha_start = event
                        .alpha_exponent_start()
                        .checked_add(coefficient_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("packing alpha range overflow".into())
                        })?;
                    let alpha_eval = *alpha_cache.get(alpha_start / block).ok_or_else(|| {
                        AkitaError::Internal(
                            "packing event exceeds the generated alpha evaluation cache".into(),
                        )
                    })?;
                    let physical = coefficients
                        .start
                        .checked_add(coefficient_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("packing event address overflow".into())
                        })?;
                    Ok(acc + event.scalar() * alpha_eval * equality.eval(physical / block))
                })
        };
        (0..self.events.len()).fold(Ok(E::zero()), evaluate_event)
    }
}

/// One group's checked coefficient-packing relation semantics and canonical
/// factors from which a compute backend can prepare its own execution plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingGroupSemantics<'a, E: Field> {
    pub(super) group_index: usize,
    pub(super) geometry: SubringCoefficientPackingGeometry,
    pub(super) relation_events: CoefficientPackingRelationEvents<E>,
    pub(super) witness_units: Vec<WitnessUnitLayout>,
    pub(super) prepared_point: &'a PreparedSubringCoefficientPackingPoint<E>,
    pub(super) group_claim_range: Range<usize>,
    pub(super) group_claim_coefficients: &'a [E],
    pub(super) scalar_claim_weight: E,
    pub(super) consistency_weight: E,
    pub(super) d_d: usize,
    pub(super) num_digits_open: usize,
    pub(super) num_digits_inner: usize,
    pub(super) num_digits_fold: usize,
    pub(super) num_positions_per_block: usize,
    pub(super) physical_field_len: usize,
    pub(super) relation_coefficient_block_len: usize,
    pub(super) alpha_powers: Vec<E>,
    pub(super) basis: Vec<E>,
    pub(super) opening_gadget: Vec<E>,
    pub(super) witness_gadget: Vec<E>,
    pub(super) fold_gadget: Vec<E>,
}

/// Exact authority used to prepare every packing group in one fold.
pub struct CoefficientPackingBatchSemanticInputs<'authority, 'output, 'points, F: Field, E: Field> {
    pub level_params: &'authority CommittedGroupParams,
    pub opening_batch: &'authority OpeningClaimsLayout,
    pub relation_plan: &'authority RelationRangeImagePlan,
    pub relation: &'authority RingRelationInstance<F>,
    /// Prepared public points keyed by authenticated group index.
    pub prepared_points: &'points [(usize, &'output PreparedSubringCoefficientPackingPoint<E>)],
    pub alpha: E,
    pub tau1: &'authority [E],
    pub claim_coefficients: &'output [E],
}

/// Checked packing semantics for every packing group in one exact relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingBatchSemantics<'a, E: Field> {
    pub(super) groups: Vec<CoefficientPackingGroupSemantics<'a, E>>,
}

impl<'a, E: Field> CoefficientPackingBatchSemantics<'a, E> {
    #[must_use]
    pub fn groups(&self) -> &[CoefficientPackingGroupSemantics<'a, E>] {
        &self.groups
    }

    #[must_use]
    pub fn into_groups(self) -> Vec<CoefficientPackingGroupSemantics<'a, E>> {
        self.groups
    }
}

impl<'a, E: Field> CoefficientPackingGroupSemantics<'a, E> {
    #[must_use]
    pub const fn group_index(&self) -> usize {
        self.group_index
    }

    #[must_use]
    pub const fn geometry(&self) -> SubringCoefficientPackingGeometry {
        self.geometry
    }

    #[cfg(test)]
    #[must_use]
    pub(super) const fn relation_events(&self) -> &CoefficientPackingRelationEvents<E> {
        &self.relation_events
    }

    /// Relation events prepared from this group's validated packing authority.
    #[must_use]
    pub fn relation_weight_events(&self) -> &[RelationWeightEvent<E>] {
        &self.relation_events.events
    }

    #[must_use]
    pub fn witness_units(&self) -> &[WitnessUnitLayout] {
        &self.witness_units
    }

    #[must_use]
    pub const fn prepared_point(&self) -> &'a PreparedSubringCoefficientPackingPoint<E> {
        self.prepared_point
    }

    #[must_use]
    pub fn group_claim_range(&self) -> Range<usize> {
        self.group_claim_range.clone()
    }

    #[must_use]
    pub fn group_claim_coefficients(&self) -> &'a [E] {
        self.group_claim_coefficients
    }

    #[must_use]
    pub const fn scalar_claim_weight(&self) -> E {
        self.scalar_claim_weight
    }

    #[must_use]
    pub const fn consistency_weight(&self) -> E {
        self.consistency_weight
    }

    #[must_use]
    pub const fn d_d(&self) -> usize {
        self.d_d
    }

    #[must_use]
    pub const fn num_digits_open(&self) -> usize {
        self.num_digits_open
    }

    #[must_use]
    pub const fn num_digits_inner(&self) -> usize {
        self.num_digits_inner
    }

    #[must_use]
    pub const fn num_digits_fold(&self) -> usize {
        self.num_digits_fold
    }

    #[must_use]
    pub const fn num_positions_per_block(&self) -> usize {
        self.num_positions_per_block
    }

    #[must_use]
    pub const fn physical_field_len(&self) -> usize {
        self.physical_field_len
    }

    #[must_use]
    pub const fn relation_coefficient_block_len(&self) -> usize {
        self.relation_coefficient_block_len
    }

    #[must_use]
    pub fn alpha_powers(&self) -> &[E] {
        &self.alpha_powers
    }

    #[must_use]
    pub fn basis(&self) -> &[E] {
        &self.basis
    }

    #[must_use]
    pub fn opening_gadget(&self) -> &[E] {
        &self.opening_gadget
    }

    #[must_use]
    pub fn witness_gadget(&self) -> &[E] {
        &self.witness_gadget
    }

    #[must_use]
    pub fn fold_gadget(&self) -> &[E] {
        &self.fold_gadget
    }
}
