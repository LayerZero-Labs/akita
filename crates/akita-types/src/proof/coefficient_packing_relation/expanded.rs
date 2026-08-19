use super::*;
#[cfg(test)]
use std::sync::Arc;

/// Checked inputs for one coefficient-packing group's shared relation semantics.
pub(super) struct CoefficientPackingGroupSemanticInputs<'a, F: FieldCore, E: FieldCore> {
    pub level_params: &'a CommittedGroupParams,
    pub opening_batch: &'a OpeningClaimsLayout,
    pub relation_plan: &'a RelationRangeImagePlan,
    pub relation: &'a RingRelationInstance<F>,
    pub group_index: usize,
    pub prepared_point: &'a PreparedSubringCoefficientPackingPoint<E>,
    pub alpha: E,
    pub tau1: &'a [E],
    /// Global claim coefficients in authenticated opening-batch order.
    pub claim_coefficients: &'a [E],
}

/// Packing-specific E and quotient events over the checked flat witness domain.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CoefficientPackingRelationEvents<E: FieldCore> {
    pub(super) events: Vec<RelationWeightEvent<E>>,
    pub(super) alpha_powers: Arc<[E]>,
    pub(super) relation_coefficient_block_len: usize,
    pub(super) physical_field_len: usize,
}

#[cfg(test)]
impl<E: FieldCore> CoefficientPackingRelationEvents<E> {
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
        let equality =
            OffsetEqWindow::new(point.get(low_variables..).ok_or(AkitaError::InvalidProof)?)?;
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
                    .ok_or(AkitaError::InvalidProof)?,
                &point[..low_variables],
            )?);
        }
        let evaluate_event = |sum: Result<E, AkitaError>, event_index: usize| {
            let sum = sum?;
            let event = self
                .events
                .get(event_index)
                .ok_or(AkitaError::InvalidProof)?;
            let coefficients = event.physical_coefficients();
            if !coefficients.start.is_multiple_of(block)
                || !coefficients.len().is_multiple_of(block)
                || !event.alpha_exponent_start().is_multiple_of(block)
            {
                return Err(AkitaError::InvalidProof);
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
                    let alpha_eval = *alpha_cache
                        .get(alpha_start / block)
                        .ok_or(AkitaError::InvalidProof)?;
                    let physical = coefficients
                        .start
                        .checked_add(coefficient_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("packing event address overflow".into())
                        })?;
                    Ok(acc + event.scalar() * alpha_eval * equality.eval(physical / block))
                })
        };
        if work < 1024 {
            (0..self.events.len()).fold(Ok(E::zero()), evaluate_event)
        } else {
            cfg_fold_reduce!(
                0..self.events.len(),
                || Ok(E::zero()),
                evaluate_event,
                |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
            )
        }
    }
}

/// Shared source selected by one structured Stage 2 term.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoefficientPackingStage2Source {
    DirectOpening,
    PackingZ,
}

/// One checked source-to-witness segment for a structured Stage 2 term.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingStage2Segment {
    pub(super) physical_coefficients: Range<usize>,
    pub(super) source_coefficients: Range<usize>,
}

impl CoefficientPackingStage2Segment {
    #[must_use]
    pub fn physical_coefficients(&self) -> Range<usize> {
        self.physical_coefficients.clone()
    }

    #[must_use]
    pub fn source_coefficients(&self) -> Range<usize> {
        self.source_coefficients.clone()
    }
}

/// One scalar-times-source structured Stage 2 contribution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingStage2Term<E: FieldCore> {
    pub(super) source: CoefficientPackingStage2Source,
    pub(super) factor: E,
    pub(super) segments: Range<usize>,
}

impl<E: FieldCore> CoefficientPackingStage2Term<E> {
    #[must_use]
    pub const fn source(&self) -> CoefficientPackingStage2Source {
        self.source
    }

    #[must_use]
    pub const fn factor(&self) -> E {
        self.factor
    }

    #[must_use]
    pub fn segments(&self) -> Range<usize> {
        self.segments.clone()
    }
}

/// Direct-opening and packing-Z structured terms for one group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingStage2Terms<E: FieldCore> {
    pub(super) direct_opening_source: Vec<E>,
    pub(super) packing_z_source: Vec<E>,
    pub(super) segments: Vec<CoefficientPackingStage2Segment>,
    pub(super) terms: Vec<CoefficientPackingStage2Term<E>>,
    pub(super) physical_field_len: usize,
    pub(super) relation_coefficient_block_len: usize,
    pub(super) group_claim_range: Range<usize>,
    pub(super) scalar_claim_weight: E,
}

impl<E: FieldCore> CoefficientPackingStage2Terms<E> {
    #[must_use]
    pub fn direct_opening_source(&self) -> &[E] {
        &self.direct_opening_source
    }

    #[must_use]
    pub fn packing_z_source(&self) -> &[E] {
        &self.packing_z_source
    }

    #[must_use]
    pub fn segments(&self) -> &[CoefficientPackingStage2Segment] {
        &self.segments
    }

    #[must_use]
    pub fn terms(&self) -> &[CoefficientPackingStage2Term<E>] {
        &self.terms
    }

    #[must_use]
    pub const fn physical_field_len(&self) -> usize {
        self.physical_field_len
    }

    #[must_use]
    pub fn group_claim_range(&self) -> Range<usize> {
        self.group_claim_range.clone()
    }

    #[must_use]
    pub const fn scalar_claim_weight(&self) -> E {
        self.scalar_claim_weight
    }

    #[must_use]
    pub const fn relation_coefficient_block_len(&self) -> usize {
        self.relation_coefficient_block_len
    }

    #[must_use]
    pub fn into_linear_parts(
        self,
    ) -> (
        [Vec<E>; 2],
        Vec<CoefficientPackingStage2Segment>,
        Vec<CoefficientPackingStage2Term<E>>,
    ) {
        (
            [self.direct_opening_source, self.packing_z_source],
            self.segments,
            self.terms,
        )
    }

    /// Evaluate the structured direct-opening and packing-Z terms at one flat
    /// witness point without materializing a witness-sized weight table.
    pub fn evaluate_at_point(&self, point: &[E]) -> Result<E, AkitaError> {
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
        let equality =
            OffsetEqWindow::new(point.get(low_variables..).ok_or(AkitaError::InvalidProof)?)?;
        let mut work = 0usize;
        for term in &self.terms {
            for segment in self
                .segments
                .get(term.segments())
                .ok_or(AkitaError::InvalidProof)?
            {
                work = work
                    .checked_add(segment.physical_coefficients().len() / block)
                    .ok_or_else(|| {
                        AkitaError::InvalidSetup("packing Stage 2 work overflow".into())
                    })?;
            }
        }
        if work > MAX_COMPACT_STRIDE_TERMS {
            return Err(AkitaError::InvalidSize {
                expected: MAX_COMPACT_STRIDE_TERMS,
                actual: work,
            });
        }
        let evaluate_source = |source: &[E]| -> Result<Vec<E>, AkitaError> {
            if !source.len().is_multiple_of(block) {
                return Err(AkitaError::InvalidProof);
            }
            let block_count = source.len() / block;
            let mut cache = Vec::new();
            cache.try_reserve_exact(block_count).map_err(|_| {
                AkitaError::InvalidInput("packing source cache allocation failed".into())
            })?;
            for source_block in 0..block_count {
                let source_start = source_block.checked_mul(block).ok_or_else(|| {
                    AkitaError::InvalidSetup("packing Stage 2 source overflow".into())
                })?;
                let source_end = source_start.checked_add(block).ok_or_else(|| {
                    AkitaError::InvalidSetup("packing Stage 2 source overflow".into())
                })?;
                cache.push(multilinear_eval(
                    source
                        .get(source_start..source_end)
                        .ok_or(AkitaError::InvalidProof)?,
                    &point[..low_variables],
                )?);
            }
            Ok(cache)
        };
        let direct_opening_cache = evaluate_source(self.direct_opening_source())?;
        let packing_z_cache = evaluate_source(self.packing_z_source())?;
        let evaluate_term = |sum: Result<E, AkitaError>, term_index: usize| {
            let sum = sum?;
            let term = self.terms.get(term_index).ok_or(AkitaError::InvalidProof)?;
            let source_cache = match term.source() {
                CoefficientPackingStage2Source::DirectOpening => &direct_opening_cache,
                CoefficientPackingStage2Source::PackingZ => &packing_z_cache,
            };
            let segments = self
                .segments
                .get(term.segments())
                .ok_or(AkitaError::InvalidProof)?;
            segments.iter().try_fold(sum, |term_sum, segment| {
                let physical = segment.physical_coefficients();
                let source_range = segment.source_coefficients();
                if physical.len() != source_range.len()
                    || !physical.start.is_multiple_of(block)
                    || !physical.len().is_multiple_of(block)
                    || !source_range.start.is_multiple_of(block)
                {
                    return Err(AkitaError::InvalidProof);
                }
                (0..physical.len())
                    .step_by(block)
                    .try_fold(term_sum, |acc, offset| {
                        let physical_index =
                            physical.start.checked_add(offset).ok_or_else(|| {
                                AkitaError::InvalidSetup("packing Stage 2 address overflow".into())
                            })?;
                        let source_index =
                            source_range.start.checked_add(offset).ok_or_else(|| {
                                AkitaError::InvalidSetup("packing Stage 2 source overflow".into())
                            })?;
                        let source_value = *source_cache
                            .get(source_index / block)
                            .ok_or(AkitaError::InvalidProof)?;
                        Ok(acc
                            + term.factor() * source_value * equality.eval(physical_index / block))
                    })
            })
        };
        if work < 1024 {
            (0..self.terms.len()).fold(Ok(E::zero()), evaluate_term)
        } else {
            cfg_fold_reduce!(
                0..self.terms.len(),
                || Ok(E::zero()),
                evaluate_term,
                |lhs: Result<E, AkitaError>, rhs: Result<E, AkitaError>| Ok(lhs? + rhs?)
            )
        }
    }
}

/// One group's joined coefficient-packing relation semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingGroupSemantics<E: FieldCore> {
    pub(super) group_index: usize,
    pub(super) geometry: SubringCoefficientPackingGeometry,
    #[cfg(test)]
    pub(super) relation_events: CoefficientPackingRelationEvents<E>,
    pub(super) stage2_terms: CoefficientPackingStage2Terms<E>,
}

/// Exact authority used to prepare every packing group in one fold.
pub struct CoefficientPackingBatchSemanticInputs<'a, F: FieldCore, E: FieldCore> {
    pub level_params: &'a CommittedGroupParams,
    pub opening_batch: &'a OpeningClaimsLayout,
    pub relation_plan: &'a RelationRangeImagePlan,
    pub relation: &'a RingRelationInstance<F>,
    /// Prepared public points keyed by authenticated group index.
    pub prepared_points: &'a [(usize, &'a PreparedSubringCoefficientPackingPoint<E>)],
    pub alpha: E,
    pub tau1: &'a [E],
    pub claim_coefficients: &'a [E],
}

/// Checked packing semantics for every packing group in one exact relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingBatchSemantics<E: FieldCore> {
    pub(super) groups: Vec<CoefficientPackingGroupSemantics<E>>,
}

impl<E: FieldCore> CoefficientPackingBatchSemantics<E> {
    #[must_use]
    pub fn groups(&self) -> &[CoefficientPackingGroupSemantics<E>] {
        &self.groups
    }

    #[must_use]
    pub fn into_groups(self) -> Vec<CoefficientPackingGroupSemantics<E>> {
        self.groups
    }
}

impl<E: FieldCore> CoefficientPackingGroupSemantics<E> {
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

    #[must_use]
    pub const fn stage2_terms(&self) -> &CoefficientPackingStage2Terms<E> {
        &self.stage2_terms
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        usize,
        SubringCoefficientPackingGeometry,
        CoefficientPackingStage2Terms<E>,
    ) {
        (self.group_index, self.geometry, self.stage2_terms)
    }
}
