//! Structured Stage 2 terms for subring coefficient packing.

use std::sync::Arc;

use super::{
    PreparedProverLinearTerms, StructuredLinearSegment, StructuredLinearTerm,
    StructuredLinearWeights,
};
use crate::compute::SubringCoefficientPackingPartials;
use akita_field::{
    canonical_extension_basis, AkitaError, CanonicalField, ExtField, FieldCore, FromPrimitiveInt,
    LiftBase,
};
use akita_types::proof::relation::relation_row_weight;
use akita_types::{
    coefficient_packing_scalar_opening, gadget_row_scalars, OpeningClaimsLayout, OpeningMethod,
    PreparedSubringCoefficientPackingPoint, RelationRangeImagePlan, RelationWitnessGeometry,
    SubringCoefficientPackingGeometry,
};

/// Checked inputs for one group's two structured Stage 2 contributions.
pub(crate) struct CoefficientPackingLinearTermInputs<'a, F: FieldCore, E: FieldCore> {
    pub(crate) level_params: &'a akita_types::CommittedGroupParams,
    pub(crate) opening_batch: &'a OpeningClaimsLayout,
    pub(crate) relation_plan: &'a RelationRangeImagePlan,
    pub(crate) group_index: usize,
    pub(crate) prepared_point: &'a PreparedSubringCoefficientPackingPoint<E>,
    pub(crate) partials_by_claim: &'a [SubringCoefficientPackingPartials<F>],
    /// Global claim coefficients in authenticated opening-batch order.
    pub(crate) claim_coefficients: &'a [E],
    pub(crate) claimed_scalar_opening: E,
    pub(crate) alpha: E,
    pub(crate) tau1: &'a [E],
}

/// Prepared method-specific weights and their independently computed input claim.
pub(crate) struct PreparedCoefficientPackingLinearTerms<E: FieldCore> {
    pub(crate) group_index: usize,
    pub(crate) geometry: SubringCoefficientPackingGeometry,
    pub(crate) linear_terms: PreparedProverLinearTerms<E>,
    pub(crate) scalar_opening: E,
    pub(crate) weighted_scalar_opening_claim: E,
}

fn packing_geometry<F: FieldCore, E: ExtField<F>>(
    inputs: &CoefficientPackingLinearTermInputs<'_, F, E>,
) -> Result<SubringCoefficientPackingGeometry, AkitaError> {
    let expected_relation_geometry = RelationWitnessGeometry::for_level(
        inputs.level_params,
        inputs.opening_batch,
        E::EXT_DEGREE,
    )?;
    if inputs.relation_plan.relation_witness_geometry() != &expected_relation_geometry {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing Stage 2 plan belongs to different level parameters".into(),
        ));
    }
    let group_params = inputs
        .level_params
        .group_params_geometry(inputs.opening_batch, inputs.group_index)?;
    let challenge_subring_dimension = match group_params.opening_method() {
        OpeningMethod::SubringCoefficientPacking {
            challenge_subring_dimension,
        } => challenge_subring_dimension,
        OpeningMethod::EvaluationTrace => {
            return Err(AkitaError::InvalidSetup(
                "coefficient-packing Stage 2 terms require the packing method".into(),
            ));
        }
    };
    let geometry = SubringCoefficientPackingGeometry::try_new(
        E::EXT_DEGREE,
        group_params.inner_commit_matrix_params().ring_dimension(),
        challenge_subring_dimension,
    )?;
    let opening_geometry = inputs
        .relation_plan
        .relation_witness_geometry()
        .group_opening_geometry(inputs.group_index)?;
    if inputs
        .relation_plan
        .relation_witness_geometry()
        .extension_degree()
        != E::EXT_DEGREE
        || inputs
            .relation_plan
            .relation_witness_geometry()
            .group_opening_method(inputs.group_index)?
            != group_params.opening_method()
        || opening_geometry.polynomial_modulus_dimension() != geometry.challenge_subring_dimension()
        || opening_geometry.coordinate_plane_count() != geometry.extension_degree()
        || opening_geometry.physical_coefficient_width() != geometry.partial_base_field_width()
        || inputs.prepared_point.geometry() != geometry
        || inputs.prepared_point.source_num_vars()
            != inputs
                .opening_batch
                .group_layout(inputs.group_index)?
                .num_vars()
        || inputs.prepared_point.num_live_positions()
            != group_params.num_live_ring_elements_per_claim()
        || inputs.prepared_point.num_positions_per_block() != group_params.num_positions_per_block()
        || inputs.prepared_point.num_live_blocks() != group_params.num_live_blocks()
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing Stage 2 authorities disagree".into(),
        ));
    }
    Ok(geometry)
}

/// Prepare the direct scalar-opening and packing-Z consistency terms.
///
/// Each contribution stays factored. Their sum is the method-specific linear
/// weight function added to the ordinary relation factorization in Stage 2.
pub(crate) fn prepare_coefficient_packing_linear_terms<F, E>(
    inputs: CoefficientPackingLinearTermInputs<'_, F, E>,
) -> Result<PreparedCoefficientPackingLinearTerms<E>, AkitaError>
where
    F: FieldCore + CanonicalField + FromPrimitiveInt,
    E: ExtField<F> + FromPrimitiveInt + LiftBase<F>,
{
    let geometry = packing_geometry(&inputs)?;
    let group_params = inputs
        .level_params
        .group_params_geometry(inputs.opening_batch, inputs.group_index)?;
    let group_layout = inputs.opening_batch.group_layout(inputs.group_index)?;
    let num_claims = group_layout.num_polynomials();
    let num_blocks = group_params.num_live_blocks();
    if inputs.claim_coefficients.len() != inputs.opening_batch.num_total_polynomials()
        || inputs.tau1.len() != inputs.relation_plan.relation_row_index_num_vars()?
    {
        return Err(AkitaError::InvalidInput(
            "coefficient-packing Stage 2 global claim or row point disagrees".into(),
        ));
    }
    let group_plan = inputs
        .relation_plan
        .groups()
        .iter()
        .find(|group| group.group_index() == inputs.group_index)
        .ok_or(AkitaError::InvalidProof)?;
    let group_claim_coefficients = inputs
        .claim_coefficients
        .get(group_plan.claim_range())
        .ok_or(AkitaError::InvalidProof)?;
    if num_claims == 0
        || inputs.partials_by_claim.len() != num_claims
        || group_claim_coefficients.len() != num_claims
        || inputs.partials_by_claim.iter().any(|partials| {
            partials.geometry() != geometry || partials.num_live_blocks() != num_blocks
        })
    {
        return Err(AkitaError::InvalidInput(
            "coefficient-packing Stage 2 claims or blocks disagree".into(),
        ));
    }
    let scalar_opening = coefficient_packing_scalar_opening::<F, E>(
        geometry,
        num_blocks,
        inputs.partials_by_claim,
        group_claim_coefficients,
        inputs.prepared_point.live_block_weights(),
        inputs.prepared_point.tail_weights(),
    )?;
    if scalar_opening != inputs.claimed_scalar_opening {
        return Err(AkitaError::InvalidProof);
    }
    let consistency_row_weight = relation_row_weight(
        inputs
            .relation_plan
            .consistency_row_index(inputs.group_index)?,
        inputs.tau1,
    )?;
    let scalar_opening_row_weight = relation_row_weight(
        inputs.relation_plan.scalar_opening_row_index()?,
        inputs.tau1,
    )?;
    let weighted_scalar_opening_claim = scalar_opening_row_weight * inputs.claimed_scalar_opening;

    let d_d = inputs.level_params.role_dims().d_d();
    let physical_width = geometry.partial_base_field_width();
    let relation_coefficient_block_len = inputs
        .relation_plan
        .relation_address_geometry()
        .relation_coefficient_block_len();
    if d_d == 0
        || !physical_width.is_multiple_of(d_d)
        || !d_d.is_multiple_of(relation_coefficient_block_len)
    {
        return Err(AkitaError::InvalidSetup(
            "coefficient-packing E width does not factor the Stage 2 layout".into(),
        ));
    }
    let depth_open = group_params.num_digits_open();
    let opening_gadget = gadget_row_scalars::<F>(depth_open, group_params.log_basis_open())
        .into_iter()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    let basis = canonical_extension_basis::<F, E>(geometry.extension_degree())?;
    let mut opening_source = Vec::with_capacity(physical_width);
    for &basis_element in &basis {
        opening_source.extend(
            inputs
                .prepared_point
                .tail_weights()
                .iter()
                .map(|&tail_weight| basis_element * tail_weight),
        );
    }
    if opening_source.len() != physical_width {
        return Err(AkitaError::InvalidProof);
    }

    let mut segments = Vec::new();
    let mut terms = Vec::new();
    let opening_source: Arc<[E]> = opening_source.into();
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
                for (digit, &digit_weight) in opening_gadget.iter().enumerate() {
                    let factor =
                        scalar_opening_row_weight * claim_coefficient * block_weight * digit_weight;
                    let segment_start = segments.len();
                    for role_subcolumn in 0..physical_width / d_d {
                        segments.push(StructuredLinearSegment {
                            physical_coefficient_start: unit.e_coefficient_index(
                                d_d,
                                num_claims,
                                depth_open,
                                claim,
                                global_block,
                                role_subcolumn,
                                digit,
                                0,
                            )?,
                            source_coefficient_start: role_subcolumn * d_d,
                            coefficient_count: d_d,
                        });
                    }
                    terms.push(StructuredLinearTerm {
                        factor,
                        source_index: 0,
                        segment_range: segment_start..segments.len(),
                    });
                }
            }
        }
    }

    let stride = geometry.subring_embedding_stride();
    let s = geometry.challenge_subring_dimension();
    let d_a = geometry.a_ring_dimension();
    let alpha_powers = akita_algebra::ring::scalar_powers(inputs.alpha, s);
    let mut packing_z_source = vec![E::zero(); d_a];
    for (low_index, &packing_weight) in inputs.prepared_point.packing_weights().iter().enumerate() {
        for (subring_index, &alpha_power) in alpha_powers.iter().enumerate() {
            let physical = stride
                .checked_mul(subring_index)
                .and_then(|base| base.checked_add(low_index))
                .ok_or_else(|| AkitaError::InvalidSetup("packing-Z index overflow".into()))?;
            *packing_z_source
                .get_mut(physical)
                .ok_or(AkitaError::InvalidProof)? = packing_weight * alpha_power;
        }
    }
    let packing_z_source: Arc<[E]> = packing_z_source.into();
    let depth_witness = group_params.num_digits_inner();
    let depth_fold = group_params.num_digits_fold();
    let witness_gadget = gadget_row_scalars::<F>(depth_witness, group_params.log_basis_inner())
        .into_iter()
        .map(E::lift_base)
        .collect::<Vec<_>>();
    let fold_gadget = gadget_row_scalars::<F>(depth_fold, group_params.log_basis_open())
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
                    let segment_start = segments.len();
                    segments.push(StructuredLinearSegment {
                        physical_coefficient_start: unit.z_coefficient_index(
                            d_a,
                            group_params.num_positions_per_block(),
                            depth_witness,
                            depth_fold,
                            position,
                            witness_digit,
                            fold_digit,
                            0,
                        )?,
                        source_coefficient_start: 0,
                        coefficient_count: d_a,
                    });
                    terms.push(StructuredLinearTerm {
                        factor: -(consistency_row_weight
                            * position_weight
                            * witness_weight
                            * fold_weight),
                        source_index: 1,
                        segment_range: segment_start..segments.len(),
                    });
                }
            }
        }
    }
    let weights = StructuredLinearWeights {
        sources: vec![opening_source, packing_z_source],
        segments,
        terms,
        physical_field_len: inputs.relation_plan.digit_witness_domain().live_len(),
    };
    let linear_terms = PreparedProverLinearTerms::from_structured_weights(
        &weights,
        relation_coefficient_block_len,
    )?;
    Ok(PreparedCoefficientPackingLinearTerms {
        group_index: inputs.group_index,
        geometry,
        linear_terms,
        scalar_opening,
        weighted_scalar_opening_claim,
    })
}

#[cfg(test)]
#[path = "coefficient_packing_terms_tests.rs"]
mod tests;
