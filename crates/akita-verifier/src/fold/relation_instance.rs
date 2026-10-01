//! The fold's public ring-relation instance: payload shape checks before the
//! fold transcript, then instance assembly from the drawn fold challenges.

use super::PreparedFoldReplay;
use crate::stages::opening_claims::PreparedFoldOpeningPoint;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::{
    assemble_compressed_relation_rhs, assemble_relation_rhs, dispatch_for_field, FpExtEncoding,
    GroupFoldChallenges, OpeningFamily, RelationWitnessGeometry, RingRelationGroupOpening,
    RingRelationInstance, RingVec,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, Ring};

/// Check the opening and commitment payload shapes against the level geometry.
///
/// Runs before the fold response is read, so every rejection here precedes
/// any fold transcript operation.
pub(super) fn validate_fold_payloads<F, E>(
    prepared: &PreparedFoldReplay<'_, F, E>,
) -> Result<RelationWitnessGeometry, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let opening_shape = &prepared.opening_shape;
    if prepared.opening_payload.coeff_len() != prepared.level_layout.opening_payload_coeffs() {
        return Err(AkitaError::InvalidProof);
    }
    let num_groups = opening_shape.num_groups();
    let commitment_payloads = &prepared.commitment_payloads;
    let relation_geometry =
        RelationWitnessGeometry::for_level(prepared.lp, opening_shape, E::DEGREE).map_err(
            |error| {
                AkitaError::InvalidInput(format!("compressed relation layout failed: {error:?}"))
            },
        )?;
    let relation_rhs_layout = relation_geometry.rhs_layout();
    if commitment_payloads.len() != num_groups {
        return Err(AkitaError::InvalidInput(
            "commitment payload group count mismatch".into(),
        ));
    }
    for (relation_group_index, payload) in commitment_payloads.iter().enumerate() {
        if payload.coeff_len()
            != relation_rhs_layout
                .group_payload_geometry(relation_group_index)?
                .transmitted_coefficients()
        {
            return Err(AkitaError::InvalidInput(
                "commitment payload length mismatch".into(),
            ));
        }
    }
    prepared.lp.validate_opening_batch(opening_shape)?;
    if prepared.prefix.prepared_points.len() != num_groups {
        return Err(AkitaError::InvalidProof);
    }
    Ok(relation_geometry)
}

/// Assemble the fold's ring-relation instance from the evaluation-batch row
/// coefficients, the transmitted payloads, and the drawn fold challenges.
pub(super) fn assemble_relation_instance<F, E>(
    prepared: &PreparedFoldReplay<'_, F, E>,
    relation_geometry: &RelationWitnessGeometry,
    group_challenges: Vec<GroupFoldChallenges>,
) -> Result<RingRelationInstance<F>, AkitaError>
where
    F: Field + CanonicalEncoding + akita_serialization::AkitaSerialize + Ring,
    E: FpExtEncoding<F> + ExtField<F> + Ring + AkitaSerialize + MulBaseUnreduced<F>,
{
    let commitment_payloads = &prepared.commitment_payloads;
    let prefix = &prepared.prefix;
    let role_dims = prepared.lp.role_dims();
    let relation_rhs_layout = relation_geometry.rhs_layout();
    let (gamma, row_coefficient_rings) = dispatch_for_field!(
        ProtocolDispatchSlot::Role(RingRole::Inner),
        F,
        role_dims.d_a(),
        |D| {
            RingRelationInstance::<F>::gamma_and_row_rings_from_coefficients::<D, E>(
                &prefix.row_coefficients,
            )
        }
    )?;
    let commitment_rows = RingVec::from_coeffs(
        commitment_payloads
            .iter()
            .flat_map(|payload| payload.coeffs().iter().copied())
            .collect(),
    );
    let relation_rhs = if prepared.lp.payload_mode.is_compressed() {
        let group_payloads = commitment_payloads
            .iter()
            .map(|payload| payload.coeffs())
            .collect::<Vec<_>>();
        assemble_compressed_relation_rhs::<F>(
            relation_rhs_layout,
            &group_payloads,
            prepared.opening_payload.coeffs(),
        )?
    } else {
        assemble_relation_rhs::<F>(
            relation_rhs_layout,
            &prepared.opening_payload,
            &commitment_rows,
        )?
    };
    let group_openings = group_challenges
        .into_iter()
        .zip(&prefix.prepared_points)
        .map(|(challenges, point)| match (challenges, point) {
            (
                OpeningFamily::EvaluationTrace(challenges),
                PreparedFoldOpeningPoint::EvaluationTrace(point),
            ) => Ok(RingRelationGroupOpening::evaluation_trace(
                challenges,
                point.ring_multiplier_point.clone(),
            )),
            (
                OpeningFamily::SubringCoefficientPacking(challenges),
                PreparedFoldOpeningPoint::SubringCoefficientPacking(point),
            ) if point.geometry() == challenges.geometry() => {
                Ok(RingRelationGroupOpening::coefficient_packing(challenges))
            }
            _ => Err(AkitaError::InvalidProof),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let relation_instance = RingRelationInstance::new(
        group_openings,
        E::DEGREE,
        prepared.opening_shape.clone(),
        gamma,
        row_coefficient_rings,
        relation_rhs,
        role_dims,
    )?;
    if !prepared.lp.payload_mode.is_compressed() {
        RingRelationInstance::check_v_shape_for_level(&prepared.opening_payload, prepared.lp)?;
    }
    Ok(relation_instance)
}
