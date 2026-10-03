//! Ring-switch relation claim and the Stage 2 relation plan it fixes.
//!
//! Runs after the ring-switch challenges and before Stage 1, which consumes
//! the range-image plan.

use crate::coefficient_packing_relation::{
    prepare_coefficient_packing_verifier_batch_semantics, CoefficientPackingVerifierBatchSemantics,
};
use crate::stages::opening_claims::{FoldPrefix, PreparedFoldOpeningPoint};
use crate::stages::ring_switch::RingSwitchVerifyOutput;
use akita_error::AkitaError;
use akita_params::{CommittedGroupParams, DigitRangePlan, RelationWitnessGeometry};
use akita_types::{
    relation_claim_from_compressed_rhs_extension, FpExtEncoding, RelationRangeImagePlan,
    RingRelationInstance,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced};

/// Relation-side Stage 2 inputs fixed by the ring switch.
pub(crate) struct RelationClaim<E: Field> {
    /// Ring-switch relation claim `<tau1-row combination, rhs>` at alpha.
    pub(crate) claim: E,
    /// Range-image plan shared by Stage 1 and the Stage 2 opening semantics.
    pub(crate) range_image_plan: RelationRangeImagePlan,
    /// Coefficient-packing batch semantics when any group opens by packing.
    pub(crate) coefficient_packing_batch: Option<CoefficientPackingVerifierBatchSemantics<E>>,
}

/// Derive the relation claim, the range-image plan, and the coefficient-packing
/// batch semantics for one fold.
pub(crate) fn prepare_relation_claim<F, E>(
    lp: &CommittedGroupParams,
    relation_geometry: RelationWitnessGeometry,
    relation_instance: &RingRelationInstance<F>,
    prefix: &FoldPrefix<F, E>,
    rs: &RingSwitchVerifyOutput<E>,
) -> Result<RelationClaim<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F> + MulBaseUnreduced<F>,
{
    let relation_claim = relation_claim_from_compressed_rhs_extension::<F, E>(
        relation_geometry.rhs_layout(),
        &rs.tau1,
        rs.alpha,
        relation_instance.rhs(),
    )?;
    let opening_batch = relation_instance.opening_batch();
    let relation_range_image_plan = RelationRangeImagePlan::new(
        relation_geometry,
        rs.relation_address_geometry,
        DigitRangePlan::new(rs.b)?,
        rs.relation_matrix_evaluator.witness_layout()?.clone(),
        opening_batch,
    )?;
    let prepared_packing_points = prefix
        .prepared_points
        .iter()
        .enumerate()
        .filter_map(|(group_index, point)| match point {
            PreparedFoldOpeningPoint::SubringCoefficientPacking(point) => {
                Some((group_index, point))
            }
            PreparedFoldOpeningPoint::EvaluationTrace(_) => None,
        })
        .collect::<Vec<_>>();
    let coefficient_packing_batch = if prepared_packing_points.is_empty() {
        None
    } else {
        Some(prepare_coefficient_packing_verifier_batch_semantics(
            akita_types::CoefficientPackingBatchSemanticInputs {
                level_params: lp,
                opening_batch,
                relation_plan: &relation_range_image_plan,
                relation: relation_instance,
                prepared_points: &prepared_packing_points,
                alpha: rs.alpha,
                tau1: &rs.tau1,
                claim_coefficients: &prefix.trace_claim_coefficients,
            },
        )?)
    };
    Ok(RelationClaim {
        claim: relation_claim,
        range_image_plan: relation_range_image_plan,
        coefficient_packing_batch,
    })
}
