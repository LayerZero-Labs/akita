//! Shared coefficient-packing relation semantics for prover and verifier.

use std::ops::Range;

#[cfg(test)]
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_algebra::offset_eq::{OffsetEqWindow, MAX_COMPACT_STRIDE_TERMS};
use akita_algebra::poly::multilinear_eval;
use akita_algebra::ring::scalar_powers;
use akita_error::{checked, AkitaError};
use jolt_field::solinas::parallel::*;
use jolt_field::{canonical_extension_basis, CanonicalEncoding, ExtField, Field};

use super::{
    relation_row_weight, RelationWeightContribution, RelationWeightEvent,
    RingRelationGroupOpeningView, RingRelationInstance,
};
use crate::{
    gadget_row_scalars, r_decomp_levels, validate_role_dims_for_field, CommittedGroupParams,
    FpExtEncoding, OpeningClaimsLayout, OpeningMethod, PreparedSubringCoefficientPackingPoint,
    RelationRangeImagePlan, RelationRowFamily, RelationWitnessGeometry, SignedDigitKernel,
    SubringCoefficientPackingGeometry,
};

mod compact;
mod expanded;

#[derive(Clone, Copy)]
struct RelationEventDomain {
    alpha_power_count: usize,
    coefficient_block: usize,
    physical_field_len: usize,
}

#[cfg(test)]
use compact::CoefficientPackingAffineRelationFamily;
pub use compact::{
    CoefficientPackingCompactFactors, CoefficientPackingVerifierBatchSemantics,
    CoefficientPackingVerifierGroupSemantics,
};
use expanded::CoefficientPackingGroupSemanticInputs;
#[cfg(test)]
use expanded::CoefficientPackingRelationEvents;
pub use expanded::{
    CoefficientPackingBatchSemanticInputs, CoefficientPackingBatchSemantics,
    CoefficientPackingGroupSemantics, CoefficientPackingStage2Segment,
    CoefficientPackingStage2Source, CoefficientPackingStage2Term, CoefficientPackingStage2Terms,
};

struct ValidatedCoefficientPackingGroup<'a, F: Field, E: Field> {
    inputs: CoefficientPackingGroupSemanticInputs<'a, F, E>,
    geometry: SubringCoefficientPackingGeometry,
    group_claim_range: Range<usize>,
    consistency_row: usize,
    consistency_weight: E,
    scalar_claim_weight: E,
    d_d: usize,
    coefficient_block: usize,
    physical_field_len: usize,
    alpha_powers: Vec<E>,
    basis: Vec<E>,
    opening_gadget: Vec<E>,
    challenge_alpha_values: Vec<E>,
    quotient_gadget: Vec<E>,
    denominator: E,
    witness_gadget: Vec<E>,
    fold_gadget: Vec<E>,
}

struct CoefficientPackingBatchAuthority {
    relation_geometry: RelationWitnessGeometry,
    row_families: Vec<RelationRowFamily>,
}

fn validate_coefficient_packing_batch_authority<F, E>(
    level_params: &CommittedGroupParams,
    opening_batch: &OpeningClaimsLayout,
    relation_plan: &RelationRangeImagePlan,
    relation: &RingRelationInstance<F>,
    tau1: &[E],
    claim_coefficients: &[E],
) -> Result<CoefficientPackingBatchAuthority, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    if SignedDigitKernel::for_log_basis(level_params.open().digits.log_basis)
        != Some(SignedDigitKernel::I8)
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing level opening basis requires the i8 digit kernel".into(),
        ));
    }
    for group_index in 0..opening_batch.num_groups() {
        let group_params = level_params.group_params_geometry(opening_batch, group_index)?;
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
    let level_role_dims = level_params.role_dims();
    validate_role_dims_for_field::<F>(level_role_dims)?;
    if relation.role_dims() != level_role_dims {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing relation role dimensions disagree with the level".into(),
        ));
    }
    for group_index in 0..opening_batch.num_groups() {
        validate_role_dims_for_field::<F>(
            level_params.group_role_dims_geometry(opening_batch, group_index)?,
        )?;
    }
    let relation_geometry =
        RelationWitnessGeometry::for_level(level_params, opening_batch, E::DEGREE)?;
    let expected_witness_layout = relation.segment_layout(level_params, None)?;
    if relation.opening_batch() != opening_batch
        || relation.extension_degree() != E::DEGREE
        || relation_plan.relation_witness_geometry() != &relation_geometry
        || relation_plan.witness_layout() != &expected_witness_layout
        || claim_coefficients.len() != opening_batch.num_total_polynomials()
        || tau1.len() != relation_plan.relation_row_index_num_vars()?
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing relation authorities disagree".into(),
        ));
    }
    let row_families = relation_geometry.rhs_layout().row_families()?;
    let expected_rhs_len = row_families.iter().try_fold(0usize, |sum, family| {
        sum.checked_add(family.geometry().physical_coefficient_width())
            .ok_or_else(|| AkitaError::InvalidSetup("relation RHS offset overflow".into()))
    })?;
    if expected_rhs_len != relation.rhs().coeff_len() {
        return Err(AkitaError::InvalidSize {
            expected: expected_rhs_len,
            actual: relation.rhs().coeff_len(),
        });
    }
    Ok(CoefficientPackingBatchAuthority {
        relation_geometry,
        row_families,
    })
}

fn push_event<E: Field>(
    events: &mut Vec<RelationWeightEvent<E>>,
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
        events.push(RelationWeightEvent::new(
            physical_start..physical_end,
            alpha_exponent_start,
            scalar,
            RelationWeightContribution::Constraint,
        )?);
    }
    Ok(())
}

/// Build one group's shared coefficient-packing relation semantics.
#[cfg(test)]
fn prepare_coefficient_packing_group_semantics<F, E>(
    inputs: CoefficientPackingGroupSemanticInputs<'_, F, E>,
) -> Result<CoefficientPackingGroupSemantics<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let authority = validate_coefficient_packing_batch_authority::<F, E>(
        inputs.level_params,
        inputs.opening_batch,
        inputs.relation_plan,
        inputs.relation,
        inputs.tau1,
        inputs.claim_coefficients,
    )?;
    prepare_coefficient_packing_prover_group(validate_coefficient_packing_group(
        inputs, &authority,
    )?)
    .map(|(_, semantics)| semantics)
}

fn validate_coefficient_packing_group<'a, F, E>(
    inputs: CoefficientPackingGroupSemanticInputs<'a, F, E>,
    authority: &CoefficientPackingBatchAuthority,
) -> Result<ValidatedCoefficientPackingGroup<'a, F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
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
        E::DEGREE,
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
    let opening_geometry = authority
        .relation_geometry
        .group_opening_geometry(inputs.group_index)?;
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

    let row_families = &authority.row_families;
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
    let consistency_weight = relation_row_weight(consistency_row, inputs.tau1)?;
    let scalar_claim_weight = relation_row_weight(
        inputs.relation_plan.scalar_opening_row_index()?,
        inputs.tau1,
    )?;
    let s = geometry.challenge_subring_dimension();
    let d_d = inputs.level_params.role_dims().d_d();
    let coefficient_block = inputs
        .relation_plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    let physical_field_len = inputs.relation_plan.digit_witness_domain().live_len();
    let alpha_powers = scalar_powers(inputs.alpha, s);
    let basis = canonical_extension_basis::<F, E>(geometry.extension_degree())
        .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
    let opening_gadget = gadget_row_scalars::<F>(
        group_params.num_digits_open(),
        group_params.log_basis_open(),
    )
    .into_iter()
    .map(E::lift_base)
    .collect::<Vec<_>>();
    let challenge_count = group_layout
        .num_polynomials()
        .checked_mul(group_params.num_live_blocks())
        .ok_or_else(|| AkitaError::InvalidSetup("challenge count overflow".into()))?;
    let mut challenge_alpha_values = Vec::new();
    challenge_alpha_values
        .try_reserve_exact(challenge_count)
        .map_err(|_| AkitaError::InvalidInput("challenge evaluation allocation failed".into()))?;
    for challenge_index in 0..challenge_count {
        challenge_alpha_values
            .push(canonical_challenges.eval_at_pows::<F, E>(challenge_index, &alpha_powers)?);
    }

    let quotient_gadget = gadget_row_scalars::<F>(
        r_decomp_levels::<F>(inputs.level_params.open().digits.log_basis),
        inputs.level_params.open().digits.log_basis,
    )
    .into_iter()
    .map(E::lift_base)
    .collect::<Vec<_>>();
    let quotient_depth = inputs
        .relation_plan
        .witness_layout()
        .quotient_depth()
        .ok_or_else(|| {
            AkitaError::InvalidSetup(
                "coefficient packing requires a quotient-lift witness layout".into(),
            )
        })?;
    if quotient_gadget.len() != quotient_depth {
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
    Ok(ValidatedCoefficientPackingGroup {
        inputs,
        geometry,
        group_claim_range,
        consistency_row,
        consistency_weight,
        scalar_claim_weight,
        d_d,
        coefficient_block,
        physical_field_len,
        alpha_powers,
        basis,
        opening_gadget,
        challenge_alpha_values,
        quotient_gadget,
        denominator,
        witness_gadget,
        fold_gadget,
    })
}

fn prepare_coefficient_packing_prover_group<F, E>(
    validated: ValidatedCoefficientPackingGroup<'_, F, E>,
) -> Result<
    (
        Vec<RelationWeightEvent<E>>,
        CoefficientPackingGroupSemantics<E>,
    ),
    AkitaError,
>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let ValidatedCoefficientPackingGroup {
        inputs,
        geometry,
        group_claim_range,
        consistency_row,
        consistency_weight,
        scalar_claim_weight,
        d_d,
        coefficient_block,
        physical_field_len,
        alpha_powers,
        basis,
        opening_gadget,
        challenge_alpha_values,
        quotient_gadget,
        denominator,
        witness_gadget,
        fold_gadget,
    } = validated;
    let group_layout = inputs.opening_batch.group_layout(inputs.group_index)?;
    let group_params = inputs
        .level_params
        .group_params_geometry(inputs.opening_batch, inputs.group_index)?;
    let group_claim_coefficients = inputs
        .claim_coefficients
        .get(group_claim_range.clone())
        .ok_or(AkitaError::InvalidProof)?;
    let s = geometry.challenge_subring_dimension();
    let d_a = geometry.a_ring_dimension();
    let event_domain = RelationEventDomain {
        alpha_power_count: alpha_powers.len(),
        coefficient_block,
        physical_field_len,
    };

    let e_event_capacity = checked::product([
        group_layout.num_polynomials(),
        group_params.num_live_blocks(),
        opening_gadget.len(),
        geometry.extension_degree(),
        s.div_ceil(d_d),
    ])
    .ok_or_else(|| AkitaError::InvalidSetup("coefficient-packing E event count overflow".into()))?;
    let q_event_capacity = checked::product([quotient_gadget.len(), geometry.extension_degree()])
        .ok_or_else(|| {
        AkitaError::InvalidSetup("coefficient-packing quotient event count overflow".into())
    })?;
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
                let challenge_alpha = *challenge_alpha_values
                    .get(challenge_index)
                    .ok_or(AkitaError::InvalidProof)?;
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
    let direct_term_capacity = checked::product([
        group_layout.num_polynomials(),
        group_params.num_live_blocks(),
        opening_gadget.len(),
    ])
    .ok_or_else(|| {
        AkitaError::InvalidSetup("coefficient-packing direct-opening term count overflow".into())
    })?;
    let direct_segment_capacity = direct_term_capacity
        .checked_mul(geometry.partial_base_field_width() / d_d)
        .ok_or_else(|| AkitaError::InvalidSetup("direct-opening segment count overflow".into()))?;
    let z_term_capacity = checked::product([
        inputs
            .relation_plan
            .witness_layout()
            .units_for_group(inputs.group_index)?
            .count(),
        group_params.num_positions_per_block(),
        group_params.num_digits_inner(),
        group_params.num_digits_fold(),
    ])
    .ok_or_else(|| {
        AkitaError::InvalidSetup("coefficient-packing packing-Z term count overflow".into())
    })?;
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

    #[cfg(test)]
    let relation_events = CoefficientPackingRelationEvents {
        events: events.clone(),
        alpha_powers: alpha_powers.clone().into(),
        relation_coefficient_block_len: coefficient_block,
        physical_field_len,
    };
    Ok((
        events,
        CoefficientPackingGroupSemantics {
            group_index: inputs.group_index,
            geometry,
            #[cfg(test)]
            relation_events,
            stage2_terms: CoefficientPackingStage2Terms {
                direct_opening_source,
                packing_z_source,
                segments,
                terms,
                physical_field_len,
                relation_coefficient_block_len: coefficient_block,
                group_claim_range,
                scalar_claim_weight,
            },
        },
    ))
}

fn prepare_coefficient_packing_verifier_group<F, E>(
    validated: ValidatedCoefficientPackingGroup<'_, F, E>,
) -> Result<CoefficientPackingVerifierGroupSemantics<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let inputs = &validated.inputs;
    let group_layout = inputs.opening_batch.group_layout(inputs.group_index)?;
    let group_params = inputs
        .level_params
        .group_params_geometry(inputs.opening_batch, inputs.group_index)?;
    let group_claim_coefficients = inputs
        .claim_coefficients
        .get(validated.group_claim_range.clone())
        .ok_or(AkitaError::InvalidProof)?;
    let compact_factors = compact::prepare_compact_factors(compact::CompactFactorInputs {
        geometry: validated.geometry,
        prepared_point: inputs.prepared_point,
        witness_layout: inputs.relation_plan.witness_layout(),
        group_index: inputs.group_index,
        num_claims: group_layout.num_polynomials(),
        num_live_blocks: group_params.num_live_blocks(),
        d_d: validated.d_d,
        consistency_row: validated.consistency_row,
        physical_field_len: validated.physical_field_len,
        consistency_weight: validated.consistency_weight,
        scalar_claim_weight: validated.scalar_claim_weight,
        denominator: validated.denominator,
        claim_coefficients: group_claim_coefficients,
        challenge_alpha: &validated.challenge_alpha_values,
        alpha_powers: &validated.alpha_powers,
        basis_elements: &validated.basis,
        opening_gadget: &validated.opening_gadget,
        quotient_gadget: &validated.quotient_gadget,
        witness_gadget: &validated.witness_gadget,
        fold_gadget: &validated.fold_gadget,
    })?;
    Ok(CoefficientPackingVerifierGroupSemantics {
        group_index: inputs.group_index,
        geometry: validated.geometry,
        group_claim_range: validated.group_claim_range,
        scalar_claim_weight: validated.scalar_claim_weight,
        compact_factors,
    })
}

fn prepare_coefficient_packing_batch_groups<'a, F, E, T>(
    inputs: &CoefficientPackingBatchSemanticInputs<'a, F, E>,
    mut project: impl FnMut(ValidatedCoefficientPackingGroup<'a, F, E>) -> Result<T, AkitaError>,
) -> Result<Vec<T>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let authority = validate_coefficient_packing_batch_authority::<F, E>(
        inputs.level_params,
        inputs.opening_batch,
        inputs.relation_plan,
        inputs.relation,
        inputs.tau1,
        inputs.claim_coefficients,
    )?;
    if inputs.prepared_points.len() > inputs.opening_batch.num_groups() {
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
        let point = points.get(group_index).ok_or(AkitaError::InvalidProof)?;
        match authority
            .relation_geometry
            .group_opening_method(group_index)?
        {
            OpeningMethod::EvaluationTrace => {
                if point.is_some() {
                    return Err(AkitaError::InvalidInput(
                        "EvaluationTrace group supplied a packing point".into(),
                    ));
                }
            }
            OpeningMethod::SubringCoefficientPacking { .. } => {
                let prepared_point = point.ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "coefficient-packing group is missing its prepared point".into(),
                    )
                })?;
                groups.push(project(validate_coefficient_packing_group(
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
                    &authority,
                )?)?);
            }
        }
    }
    if points.iter().enumerate().any(|(group, point)| {
        point.is_some()
            && inputs
                .relation_plan
                .groups()
                .iter()
                .all(|plan| plan.group_index() != group)
    }) {
        return Err(AkitaError::InvalidInput(
            "coefficient-packing prepared point is outside the relation group order".into(),
        ));
    }
    Ok(groups)
}

/// Prepare all packing groups for one exact fold authority.
pub fn prepare_coefficient_packing_batch_semantics<F, E>(
    inputs: CoefficientPackingBatchSemanticInputs<'_, F, E>,
) -> Result<
    (
        Vec<RelationWeightEvent<E>>,
        CoefficientPackingBatchSemantics<E>,
    ),
    AkitaError,
>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let prepared = prepare_coefficient_packing_batch_groups(
        &inputs,
        prepare_coefficient_packing_prover_group,
    )?;
    let mut events = Vec::new();
    let mut groups = Vec::with_capacity(prepared.len());
    for (group_events, group) in prepared {
        events.extend(group_events);
        groups.push(group);
    }
    Ok((events, CoefficientPackingBatchSemantics { groups }))
}

/// Prepare the compact packing factors used by the Stage 2 verifier without
/// constructing the prover's expanded event or segment tables.
pub fn prepare_coefficient_packing_verifier_batch_semantics<F, E>(
    inputs: CoefficientPackingBatchSemanticInputs<'_, F, E>,
) -> Result<CoefficientPackingVerifierBatchSemantics<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let groups = prepare_coefficient_packing_batch_groups(
        &inputs,
        prepare_coefficient_packing_verifier_group,
    )?;
    Ok(CoefficientPackingVerifierBatchSemantics { groups })
}

#[cfg(test)]
#[path = "coefficient_packing_relation_tests.rs"]
mod tests;
