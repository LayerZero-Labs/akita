//! Shared coefficient-packing relation semantics for prover and verifier.

use std::ops::Range;
use std::sync::Arc;

use akita_algebra::offset_eq::{OffsetEqWindow, MAX_COMPACT_STRIDE_TERMS};
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::scalar_powers;
use akita_field::{
    canonical_extension_basis, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    LiftBase, MulBase,
};

use super::{relation_row_weight, RingRelationGroupOpeningView, RingRelationInstance};
use crate::{
    gadget_row_scalars, r_decomp_levels, validate_role_dims_for_field, CommittedGroupParams,
    FpExtEncoding, OpeningClaimsLayout, OpeningMethod, PreparedSubringCoefficientPackingPoint,
    RelationRangeImagePlan, RelationRowFamily, RelationWitnessGeometry, SignedDigitKernel,
    SubringCoefficientPackingGeometry,
};

fn checked_product(label: &str, factors: &[usize]) -> Result<usize, AkitaError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            AkitaError::InvalidSetup(format!("coefficient-packing {label} count overflow"))
        })
    })
}

#[derive(Clone, Copy)]
struct RelationEventDomain {
    alpha_power_count: usize,
    coefficient_block: usize,
    physical_field_len: usize,
}

/// Checked inputs for one coefficient-packing group's shared relation semantics.
pub struct CoefficientPackingGroupSemanticInputs<'a, F: FieldCore, E: FieldCore> {
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

/// One consecutive-alpha contribution to the packing-specific relation weights.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingRelationEvent<E: FieldCore> {
    physical_coefficients: Range<usize>,
    alpha_exponent_start: usize,
    scalar: E,
}

impl<E: FieldCore> CoefficientPackingRelationEvent<E> {
    #[must_use]
    pub fn physical_coefficients(&self) -> Range<usize> {
        self.physical_coefficients.clone()
    }

    #[must_use]
    pub const fn alpha_exponent_start(&self) -> usize {
        self.alpha_exponent_start
    }

    #[must_use]
    pub const fn scalar(&self) -> E {
        self.scalar
    }
}

/// Packing-specific E and quotient events over the checked flat witness domain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingRelationEvents<E: FieldCore> {
    events: Vec<CoefficientPackingRelationEvent<E>>,
    alpha_powers: Arc<[E]>,
    relation_coefficient_block_len: usize,
    physical_field_len: usize,
}

impl<E: FieldCore> CoefficientPackingRelationEvents<E> {
    #[must_use]
    pub fn events(&self) -> &[CoefficientPackingRelationEvent<E>] {
        &self.events
    }

    /// Canonical powers of the alpha used to prepare every event scalar.
    #[must_use]
    pub fn alpha_powers(&self) -> &[E] {
        &self.alpha_powers
    }

    #[must_use]
    pub const fn relation_coefficient_block_len(&self) -> usize {
        self.relation_coefficient_block_len
    }

    #[must_use]
    pub const fn physical_field_len(&self) -> usize {
        self.physical_field_len
    }

    /// Evaluate the sparse packing E and quotient events at one flat point.
    ///
    /// The returned value already includes every event's alpha powers. A
    /// caller that separately contracts the native common-alpha factor must
    /// add this value afterwards, without multiplying by that factor again.
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
        let mut alpha_cache = Vec::new();
        self.events.iter().try_fold(E::zero(), |sum, event| {
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
                    let alpha_eval = if let Some((_, value)) = alpha_cache
                        .iter()
                        .find(|(cached_start, _)| *cached_start == alpha_start)
                    {
                        *value
                    } else {
                        let alpha_end = alpha_start.checked_add(block).ok_or_else(|| {
                            AkitaError::InvalidSetup("packing alpha range overflow".into())
                        })?;
                        let value = multilinear_eval(
                            self.alpha_powers
                                .get(alpha_start..alpha_end)
                                .ok_or(AkitaError::InvalidProof)?,
                            &point[..low_variables],
                        )?;
                        alpha_cache.push((alpha_start, value));
                        value
                    };
                    let physical = coefficients
                        .start
                        .checked_add(coefficient_offset)
                        .ok_or_else(|| {
                            AkitaError::InvalidSetup("packing event address overflow".into())
                        })?;
                    Ok(acc + event.scalar() * alpha_eval * equality.eval(physical / block))
                })
        })
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
    physical_coefficients: Range<usize>,
    source_coefficients: Range<usize>,
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
    source: CoefficientPackingStage2Source,
    factor: E,
    segments: Range<usize>,
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
    direct_opening_source: Arc<[E]>,
    packing_z_source: Arc<[E]>,
    segments: Vec<CoefficientPackingStage2Segment>,
    terms: Vec<CoefficientPackingStage2Term<E>>,
    physical_field_len: usize,
    relation_coefficient_block_len: usize,
    group_claim_range: Range<usize>,
    scalar_claim_weight: E,
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
        let mut source_cache = Vec::new();
        self.terms.iter().try_fold(E::zero(), |sum, term| {
            let source = match term.source() {
                CoefficientPackingStage2Source::DirectOpening => self.direct_opening_source(),
                CoefficientPackingStage2Source::PackingZ => self.packing_z_source(),
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
                        let source_value = if let Some((_, _, value)) =
                            source_cache
                                .iter()
                                .find(|(cached_source, cached_index, _)| {
                                    *cached_source == term.source() && *cached_index == source_index
                                }) {
                            *value
                        } else {
                            let source_end = source_index.checked_add(block).ok_or_else(|| {
                                AkitaError::InvalidSetup("packing Stage 2 source overflow".into())
                            })?;
                            let value = multilinear_eval(
                                source
                                    .get(source_index..source_end)
                                    .ok_or(AkitaError::InvalidProof)?,
                                &point[..low_variables],
                            )?;
                            source_cache.push((term.source(), source_index, value));
                            value
                        };
                        Ok(acc
                            + term.factor() * source_value * equality.eval(physical_index / block))
                    })
            })
        })
    }
}

/// One group's joined coefficient-packing relation semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoefficientPackingGroupSemantics<E: FieldCore> {
    group_index: usize,
    geometry: SubringCoefficientPackingGeometry,
    relation_events: CoefficientPackingRelationEvents<E>,
    stage2_terms: CoefficientPackingStage2Terms<E>,
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
pub struct CoefficientPackingBatchSemantics<F: FieldCore, E: FieldCore> {
    level_params: CommittedGroupParams,
    opening_batch: OpeningClaimsLayout,
    relation_plan: RelationRangeImagePlan,
    relation: RingRelationInstance<F>,
    alpha: E,
    tau1: Arc<[E]>,
    claim_coefficients: Arc<[E]>,
    groups: Vec<CoefficientPackingGroupSemantics<E>>,
}

impl<F: FieldCore, E: FieldCore> CoefficientPackingBatchSemantics<F, E> {
    #[must_use]
    pub fn groups(&self) -> &[CoefficientPackingGroupSemantics<E>] {
        &self.groups
    }

    #[must_use]
    pub const fn relation_plan(&self) -> &RelationRangeImagePlan {
        &self.relation_plan
    }

    /// Rejoin this prepared batch to the exact authority that created it.
    pub fn validate_context(
        &self,
        level_params: &CommittedGroupParams,
        opening_batch: &OpeningClaimsLayout,
        relation: &RingRelationInstance<F>,
        alpha: E,
        tau1: &[E],
        claim_coefficients: &[E],
    ) -> Result<(), AkitaError> {
        if &self.level_params != level_params
            || &self.opening_batch != opening_batch
            || &self.relation != relation
            || self.alpha != alpha
            || self.tau1.as_ref() != tau1
            || self.claim_coefficients.as_ref() != claim_coefficients
        {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing batch authority disagrees with its consumer".into(),
            ));
        }
        Ok(())
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

    #[must_use]
    pub const fn relation_events(&self) -> &CoefficientPackingRelationEvents<E> {
        &self.relation_events
    }

    #[must_use]
    pub const fn stage2_terms(&self) -> &CoefficientPackingStage2Terms<E> {
        &self.stage2_terms
    }
}

fn push_event<E: FieldCore>(
    events: &mut Vec<CoefficientPackingRelationEvent<E>>,
    physical_start: usize,
    coefficient_count: usize,
    alpha_exponent_start: usize,
    scalar: E,
    domain: RelationEventDomain,
) -> Result<(), AkitaError> {
    let physical_end = physical_start
        .checked_add(coefficient_count)
        .ok_or_else(|| AkitaError::InvalidSetup("packing relation event overflow".into()))?;
    let alpha_exponent_end = alpha_exponent_start
        .checked_add(coefficient_count)
        .ok_or_else(|| AkitaError::InvalidSetup("packing alpha range overflow".into()))?;
    if coefficient_count == 0
        || !physical_start.is_multiple_of(domain.coefficient_block)
        || !coefficient_count.is_multiple_of(domain.coefficient_block)
        || !alpha_exponent_start.is_multiple_of(domain.coefficient_block)
        || alpha_exponent_end > domain.alpha_power_count
        || physical_end > domain.physical_field_len
    {
        return Err(AkitaError::InvalidSetup(
            "packing relation event is not aligned to its checked domain".into(),
        ));
    }
    if !scalar.is_zero() {
        events.push(CoefficientPackingRelationEvent {
            physical_coefficients: physical_start..physical_end,
            alpha_exponent_start,
            scalar,
        });
    }
    Ok(())
}

/// Build one group's shared coefficient-packing relation semantics.
pub fn prepare_coefficient_packing_group_semantics<F, E>(
    inputs: CoefficientPackingGroupSemanticInputs<'_, F, E>,
) -> Result<CoefficientPackingGroupSemantics<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    if SignedDigitKernel::for_log_basis(inputs.level_params.log_basis_open)
        != Some(SignedDigitKernel::I8)
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing level opening basis requires the i8 digit kernel".into(),
        ));
    }
    for group_index in 0..inputs.opening_batch.num_groups() {
        let group_params = inputs
            .level_params
            .group_params_geometry(inputs.opening_batch, group_index)?;
        if SignedDigitKernel::for_log_basis(group_params.log_basis_open())
            != Some(SignedDigitKernel::I8)
            || SignedDigitKernel::for_log_basis(group_params.log_basis_inner()).is_none()
        {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing group opening bases require i8 digits and inner bases must be supported"
                    .into(),
            ));
        }
    }
    let level_role_dims = inputs.level_params.role_dims();
    validate_role_dims_for_field::<F>(level_role_dims)?;
    if inputs.relation.role_dims() != level_role_dims {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing relation role dimensions disagree with the level".into(),
        ));
    }
    for group_index in 0..inputs.opening_batch.num_groups() {
        validate_role_dims_for_field::<F>(
            inputs
                .level_params
                .group_role_dims_geometry(inputs.opening_batch, group_index)?,
        )?;
    }
    let expected_geometry = RelationWitnessGeometry::for_level(
        inputs.level_params,
        inputs.opening_batch,
        E::EXT_DEGREE,
    )?;
    let expected_witness_layout = inputs.relation.segment_layout(inputs.level_params, None)?;
    if inputs.relation.opening_batch() != inputs.opening_batch
        || inputs.relation.extension_degree() != E::EXT_DEGREE
        || inputs.relation_plan.relation_witness_geometry() != &expected_geometry
        || inputs.relation_plan.witness_layout() != &expected_witness_layout
        || inputs.claim_coefficients.len() != inputs.opening_batch.num_total_polynomials()
        || inputs.tau1.len() != inputs.relation_plan.relation_row_index_num_vars()?
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing relation authorities disagree".into(),
        ));
    }
    let group_plan = inputs
        .relation_plan
        .groups()
        .iter()
        .find(|group| group.group_index() == inputs.group_index)
        .ok_or(AkitaError::InvalidProof)?;
    let group_claim_range = group_plan.claim_range();
    let group_claim_coefficients = inputs
        .claim_coefficients
        .get(group_claim_range.clone())
        .ok_or(AkitaError::InvalidProof)?;
    let group_layout = inputs.opening_batch.group_layout(inputs.group_index)?;
    let group_params = inputs
        .level_params
        .group_params_geometry(inputs.opening_batch, inputs.group_index)?;
    let challenge_subring_dimension = match group_params.opening_method() {
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => challenge_subring_dimension,
        OpeningMethod::EvaluationTrace => {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing semantics require the packing method".into(),
            ));
        }
    };
    let geometry = SubringCoefficientPackingGeometry::try_new(
        E::EXT_DEGREE,
        group_params.inner_commit_matrix_params().ring_dimension(),
        challenge_subring_dimension,
    )?;
    let canonical_challenges = match inputs.relation.group_opening_view(inputs.group_index)? {
        RingRelationGroupOpeningView::SubringCoefficientPacking {
            geometry: actual,
            canonical_subring_challenges,
            ..
        } if actual == geometry => canonical_subring_challenges,
        _ => {
            return Err(AkitaError::InvalidSetup(
                "relation opening does not carry the scheduled packing geometry".into(),
            ));
        }
    };
    let opening_geometry = expected_geometry.group_opening_geometry(inputs.group_index)?;
    if inputs.prepared_point.geometry() != geometry
        || opening_geometry.polynomial_modulus_dimension() != geometry.challenge_subring_dimension()
        || opening_geometry.coordinate_plane_count() != geometry.extension_degree()
        || inputs.prepared_point.source_num_vars() != group_layout.num_vars()
        || inputs.prepared_point.num_live_positions()
            != group_params.num_live_ring_elements_per_claim()
        || inputs.prepared_point.num_positions_per_block() != group_params.num_positions_per_block()
        || inputs.prepared_point.num_live_blocks() != group_params.num_live_blocks()
        || canonical_challenges.num_claims() != group_layout.num_polynomials()
        || canonical_challenges.num_live_blocks_per_claim() != group_params.num_live_blocks()
        || group_claim_coefficients.len() != group_layout.num_polynomials()
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing group geometry, point, or claims disagree".into(),
        ));
    }

    let row_families = expected_geometry.rhs_layout().row_families()?;
    let consistency_row = inputs
        .relation_plan
        .consistency_row_index(inputs.group_index)?;
    if !matches!(
        row_families.get(consistency_row),
        Some(RelationRowFamily::Consistency {
            group_index,
            opening_method: OpeningMethod::SubringCoefficientPacking { .. },
            ..
        }) if *group_index == inputs.group_index
    ) {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing consistency row identity disagrees".into(),
        ));
    }
    let mut rhs_offset = 0usize;
    for (row, family) in row_families.iter().enumerate() {
        let width = family.geometry().physical_coefficient_width();
        let end = rhs_offset
            .checked_add(width)
            .ok_or_else(|| AkitaError::InvalidSetup("relation RHS offset overflow".into()))?;
        if row == consistency_row
            && inputs
                .relation
                .rhs()
                .coeffs()
                .get(rhs_offset..end)
                .ok_or(AkitaError::InvalidProof)?
                .iter()
                .any(|coefficient| !coefficient.is_zero())
        {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing consistency RHS must be zero".into(),
            ));
        }
        rhs_offset = end;
    }
    if rhs_offset != inputs.relation.rhs().coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: rhs_offset,
            actual: inputs.relation.rhs().coeff_len(),
        });
    }

    let consistency_weight = relation_row_weight(consistency_row, inputs.tau1)?;
    let scalar_claim_weight = relation_row_weight(
        inputs.relation_plan.scalar_opening_row_index()?,
        inputs.tau1,
    )?;
    let s = geometry.challenge_subring_dimension();
    let d_a = geometry.a_ring_dimension();
    let d_d = inputs.level_params.role_dims().d_d();
    let coefficient_block = inputs
        .relation_plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let physical_field_len = inputs.relation_plan.digit_witness_domain().live_len();
    let alpha_powers = scalar_powers(inputs.alpha, s);
    let event_domain = RelationEventDomain {
        alpha_power_count: alpha_powers.len(),
        coefficient_block,
        physical_field_len,
    };
    let basis = canonical_extension_basis::<F, E>(geometry.extension_degree())?;
    let opening_gadget = gadget_row_scalars::<F>(
        group_params.num_digits_open(),
        group_params.log_basis_open(),
    )
    .into_iter()
    .map(E::lift_base)
    .collect::<Vec<_>>();

    let e_event_capacity = checked_product(
        "E event",
        &[
            group_layout.num_polynomials(),
            group_params.num_live_blocks(),
            opening_gadget.len(),
            geometry.extension_degree(),
            s.div_ceil(d_d),
        ],
    )?;
    let q_event_capacity = checked_product(
        "quotient event",
        &[
            inputs.relation_plan.witness_layout().quotient_depth(),
            geometry.extension_degree(),
        ],
    )?;
    let event_capacity = e_event_capacity
        .checked_add(q_event_capacity)
        .ok_or_else(|| AkitaError::InvalidSetup("packing event count overflow".into()))?;
    let mut events = Vec::new();
    events
        .try_reserve_exact(event_capacity)
        .map_err(|_| AkitaError::InvalidInput("packing event allocation failed".into()))?;
    for claim in 0..group_layout.num_polynomials() {
        for unit in inputs
            .relation_plan
            .witness_layout()
            .units_for_group(inputs.group_index)?
        {
            for global_block in unit.global_block_range() {
                let challenge_index = claim
                    .checked_mul(group_params.num_live_blocks())
                    .and_then(|base| base.checked_add(global_block))
                    .ok_or_else(|| AkitaError::InvalidSetup("challenge index overflow".into()))?;
                let challenge_alpha =
                    canonical_challenges.eval_at_pows::<F, E>(challenge_index, &alpha_powers)?;
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    for (plane, &basis_element) in basis.iter().enumerate() {
                        let mut plane_offset = 0usize;
                        while plane_offset < s {
                            let flat = plane
                                .checked_mul(s)
                                .and_then(|base| base.checked_add(plane_offset))
                                .ok_or_else(|| {
                                    AkitaError::InvalidSetup("packing E plane overflow".into())
                                })?;
                            let role_subcolumn = flat / d_d;
                            let role_coefficient = flat % d_d;
                            let count = (d_d - role_coefficient).min(s - plane_offset);
                            let physical_start = unit.e_coefficient_index(
                                d_d,
                                group_layout.num_polynomials(),
                                group_params.num_digits_open(),
                                claim,
                                global_block,
                                role_subcolumn,
                                digit,
                                role_coefficient,
                            )?;
                            push_event(
                                &mut events,
                                physical_start,
                                count,
                                plane_offset,
                                consistency_weight * challenge_alpha * gadget * basis_element,
                                event_domain,
                            )?;
                            plane_offset += count;
                        }
                    }
                }
            }
        }
    }

    let quotient_gadget = gadget_row_scalars::<F>(
        r_decomp_levels::<F>(inputs.level_params.log_basis_open),
        inputs.level_params.log_basis_open,
    )
    .into_iter()
    .map(E::lift_base)
    .collect::<Vec<_>>();
    if quotient_gadget.len() != inputs.relation_plan.witness_layout().quotient_depth() {
        return Err(AkitaError::InvalidSetup(
            "packing quotient depth disagrees with witness layout".into(),
        ));
    }
    let denominator = alpha_powers
        .last()
        .copied()
        .ok_or(AkitaError::InvalidProof)?
        * inputs.alpha
        + E::one();
    for (digit, &gadget) in quotient_gadget.iter().enumerate() {
        for (plane, &basis_element) in basis.iter().enumerate() {
            let physical_start = inputs.relation_plan.witness_layout().r_coefficient_index(
                consistency_row,
                digit,
                plane,
                0,
            )?;
            push_event(
                &mut events,
                physical_start,
                s,
                0,
                -(consistency_weight * gadget * basis_element * denominator),
                event_domain,
            )?;
        }
    }

    let mut direct_opening_source = Vec::new();
    direct_opening_source
        .try_reserve_exact(geometry.partial_base_field_width())
        .map_err(|_| AkitaError::InvalidInput("direct-opening source allocation failed".into()))?;
    for &basis_element in &basis {
        direct_opening_source.extend(
            inputs
                .prepared_point
                .tail_weights()
                .iter()
                .map(|&tail_weight| basis_element * tail_weight),
        );
    }
    let mut packing_z_source = vec![E::zero(); d_a];
    for (low_index, &packing_weight) in inputs.prepared_point.packing_weights().iter().enumerate() {
        for (subring_index, &alpha_power) in alpha_powers.iter().enumerate() {
            let physical = geometry.a_ring_coefficient_index(low_index, subring_index)?;
            *packing_z_source
                .get_mut(physical)
                .ok_or(AkitaError::InvalidProof)? = packing_weight * alpha_power;
        }
    }
    let direct_term_capacity = checked_product(
        "direct-opening term",
        &[
            group_layout.num_polynomials(),
            group_params.num_live_blocks(),
            opening_gadget.len(),
        ],
    )?;
    let direct_segment_capacity = direct_term_capacity
        .checked_mul(geometry.partial_base_field_width() / d_d)
        .ok_or_else(|| AkitaError::InvalidSetup("direct-opening segment count overflow".into()))?;
    let z_term_capacity = checked_product(
        "packing-Z term",
        &[
            inputs
                .relation_plan
                .witness_layout()
                .units_for_group(inputs.group_index)?
                .count(),
            group_params.num_positions_per_block(),
            group_params.num_digits_inner(),
            group_params.num_digits_fold(),
        ],
    )?;
    let segment_capacity = direct_segment_capacity
        .checked_add(z_term_capacity)
        .ok_or_else(|| AkitaError::InvalidSetup("packing segment count overflow".into()))?;
    let term_capacity = direct_term_capacity
        .checked_add(z_term_capacity)
        .ok_or_else(|| AkitaError::InvalidSetup("packing term count overflow".into()))?;
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(segment_capacity)
        .map_err(|_| AkitaError::InvalidInput("packing segment allocation failed".into()))?;
    let mut terms = Vec::new();
    terms
        .try_reserve_exact(term_capacity)
        .map_err(|_| AkitaError::InvalidInput("packing term allocation failed".into()))?;
    for (claim, &claim_coefficient) in group_claim_coefficients.iter().enumerate() {
        for unit in inputs
            .relation_plan
            .witness_layout()
            .units_for_group(inputs.group_index)?
        {
            for global_block in unit.global_block_range() {
                let block_weight = *inputs
                    .prepared_point
                    .live_block_weights()
                    .get(global_block)
                    .ok_or(AkitaError::InvalidProof)?;
                for (digit, &gadget) in opening_gadget.iter().enumerate() {
                    let segment_start = segments.len();
                    for role_subcolumn in 0..geometry.partial_base_field_width() / d_d {
                        let physical_start = unit.e_coefficient_index(
                            d_d,
                            group_layout.num_polynomials(),
                            group_params.num_digits_open(),
                            claim,
                            global_block,
                            role_subcolumn,
                            digit,
                            0,
                        )?;
                        let source_start = role_subcolumn * d_d;
                        let physical_end = physical_start.checked_add(d_d).ok_or_else(|| {
                            AkitaError::InvalidSetup("direct-opening segment overflow".into())
                        })?;
                        let source_end = source_start.checked_add(d_d).ok_or_else(|| {
                            AkitaError::InvalidSetup("direct-opening source overflow".into())
                        })?;
                        segments.push(CoefficientPackingStage2Segment {
                            physical_coefficients: physical_start..physical_end,
                            source_coefficients: source_start..source_end,
                        });
                    }
                    terms.push(CoefficientPackingStage2Term {
                        source: CoefficientPackingStage2Source::DirectOpening,
                        factor: scalar_claim_weight * claim_coefficient * block_weight * gadget,
                        segments: segment_start..segments.len(),
                    });
                }
            }
        }
    }
    let witness_gadget = gadget_row_scalars::<F>(
        group_params.num_digits_inner(),
        group_params.log_basis_inner(),
    )
    .into_iter()
    .map(E::lift_base)
    .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(
        group_params.num_digits_fold(),
        group_params.log_basis_open(),
    )
    .into_iter()
    .map(E::lift_base)
    .collect::<Vec<_>>();
    for unit in inputs
        .relation_plan
        .witness_layout()
        .units_for_group(inputs.group_index)?
    {
        for (position, &position_weight) in
            inputs.prepared_point.position_weights().iter().enumerate()
        {
            for (witness_digit, &witness_weight) in witness_gadget.iter().enumerate() {
                for (fold_digit, &fold_weight) in fold_gadget.iter().enumerate() {
                    let physical_start = unit.z_coefficient_index(
                        d_a,
                        group_params.num_positions_per_block(),
                        group_params.num_digits_inner(),
                        group_params.num_digits_fold(),
                        position,
                        witness_digit,
                        fold_digit,
                        0,
                    )?;
                    let segment_start = segments.len();
                    let physical_end = physical_start.checked_add(d_a).ok_or_else(|| {
                        AkitaError::InvalidSetup("packing-Z segment overflow".into())
                    })?;
                    segments.push(CoefficientPackingStage2Segment {
                        physical_coefficients: physical_start..physical_end,
                        source_coefficients: 0..d_a,
                    });
                    terms.push(CoefficientPackingStage2Term {
                        source: CoefficientPackingStage2Source::PackingZ,
                        factor: -(consistency_weight
                            * position_weight
                            * witness_weight
                            * fold_weight),
                        segments: segment_start..segments.len(),
                    });
                }
            }
        }
    }

    Ok(CoefficientPackingGroupSemantics {
        group_index: inputs.group_index,
        geometry,
        relation_events: CoefficientPackingRelationEvents {
            events,
            alpha_powers: alpha_powers.into(),
            relation_coefficient_block_len: coefficient_block,
            physical_field_len,
        },
        stage2_terms: CoefficientPackingStage2Terms {
            direct_opening_source: direct_opening_source.into(),
            packing_z_source: packing_z_source.into(),
            segments,
            terms,
            physical_field_len,
            relation_coefficient_block_len: coefficient_block,
            group_claim_range,
            scalar_claim_weight,
        },
    })
}

/// Prepare all packing groups for one exact fold authority.
pub fn prepare_coefficient_packing_batch_semantics<F, E>(
    inputs: CoefficientPackingBatchSemanticInputs<'_, F, E>,
) -> Result<CoefficientPackingBatchSemantics<F, E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FpExtEncoding<F> + FromPrimitiveInt + LiftBase<F> + MulBase<F>,
{
    let expected_geometry = RelationWitnessGeometry::for_level(
        inputs.level_params,
        inputs.opening_batch,
        E::EXT_DEGREE,
    )?;
    if inputs.relation_plan.relation_witness_geometry() != &expected_geometry
        || inputs.relation.opening_batch() != inputs.opening_batch
        || inputs.prepared_points.len() > inputs.opening_batch.num_groups()
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing batch authorities disagree".into(),
        ));
    }
    let mut points = vec![None; inputs.opening_batch.num_groups()];
    for &(group_index, point) in inputs.prepared_points {
        let slot = points
            .get_mut(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if slot.replace(point).is_some() {
            return Err(AkitaError::InvalidInput(
                "coefficient-packing prepared point appears more than once".into(),
            ));
        }
    }
    let mut groups = Vec::new();
    for group_plan in inputs.relation_plan.groups() {
        let group_index = group_plan.group_index();
        match expected_geometry.group_opening_method(group_index)? {
            OpeningMethod::EvaluationTrace => {
                if points[group_index].is_some() {
                    return Err(AkitaError::InvalidInput(
                        "EvaluationTrace group supplied a packing point".into(),
                    ));
                }
            }
            OpeningMethod::SubringCoefficientPacking { .. } => {
                let prepared_point = points[group_index].ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "coefficient-packing group is missing its prepared point".into(),
                    )
                })?;
                groups.push(prepare_coefficient_packing_group_semantics(
                    CoefficientPackingGroupSemanticInputs {
                        level_params: inputs.level_params,
                        opening_batch: inputs.opening_batch,
                        relation_plan: inputs.relation_plan,
                        relation: inputs.relation,
                        group_index,
                        prepared_point,
                        alpha: inputs.alpha,
                        tau1: inputs.tau1,
                        claim_coefficients: inputs.claim_coefficients,
                    },
                )?);
            }
        }
    }
    if points.iter().enumerate().any(|(group, point)| {
        point.is_some() && groups.iter().all(|plan| plan.group_index() != group)
    }) {
        return Err(AkitaError::InvalidInput(
            "coefficient-packing prepared point is outside the relation group order".into(),
        ));
    }
    Ok(CoefficientPackingBatchSemantics {
        level_params: inputs.level_params.clone(),
        opening_batch: inputs.opening_batch.clone(),
        relation_plan: inputs.relation_plan.clone(),
        relation: inputs.relation.clone(),
        alpha: inputs.alpha,
        tau1: inputs.tau1.into(),
        claim_coefficients: inputs.claim_coefficients.into(),
        groups,
    })
}

#[cfg(test)]
#[path = "coefficient_packing_relation_tests.rs"]
mod tests;
