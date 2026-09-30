//! Opening-claim prefix replay, one module per scheduled opening method, and
//! the shared evaluation-batch draw that finalizes the fold prefix.

use akita_error::AkitaError;
use akita_types::{OpeningClaimsLayout, OpeningFamily, PreparedOpeningPoint};
use jolt_field::{CanonicalEncoding, ExtField, Field};

mod coefficient_packing;
mod extension_claim;
mod single_field;

pub(crate) use coefficient_packing::{
    verify_coefficient_packing_root_prefix, verify_coefficient_packing_suffix_prefix_native,
};
pub(crate) use extension_claim::{
    verify_extension_claim_suffix_prefix_native, verify_extension_claim_terminal_suffix_native,
};
pub(crate) use single_field::prepare_single_field_suffix_groups;

/// Common prepared fold prefix consumed by root and suffix finishing logic.
pub(crate) struct FoldPrefix<F: Field, E: Field> {
    pub(crate) prepared_points: Vec<PreparedFoldOpeningPoint<F, E>>,
    pub(crate) row_coefficients: Vec<E>,
    pub(crate) trace_eval_target: E,
    pub(crate) trace_claim_coefficients: Vec<E>,
    pub(crate) scalar_openings: Vec<E>,
}

pub(crate) type PreparedFoldOpeningPoint<F, E> = OpeningFamily<
    PreparedOpeningPoint<F, E>,
    akita_types::PreparedSubringCoefficientPackingPoint<E>,
>;

/// Fold material fixed before the shared opening payload is absorbed.
pub(crate) struct FoldClaimMaterial<F: Field, E: Field> {
    pub(crate) prepared_points: Vec<PreparedFoldOpeningPoint<F, E>>,
    pub(crate) openings: Vec<E>,
    pub(crate) reduction_final_claims: Option<Vec<E>>,
    pub(crate) reduction_factors: Option<Vec<E>>,
}

pub(crate) fn finalize_native_claims<F, E>(
    opening_shape: &OpeningClaimsLayout,
    material: FoldClaimMaterial<F, E>,
    grinding: &mut akita_types::NativeVerifierGrinding<'_, '_>,
    level: u32,
) -> Result<FoldPrefix<F, E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    if material.openings.len() != opening_shape.num_total_polynomials()
        || material.prepared_points.len() != opening_shape.num_groups()
    {
        return Err(AkitaError::InvalidProof);
    }
    let row_coefficients = akita_types::row_coefficients_native::<F, E, _>(
        opening_shape,
        akita_types::GrindingSite::EvaluationBatch { level },
        grinding,
    )?;
    let trace_claim_coefficients = material.reduction_factors.as_ref().map_or_else(
        || Ok(row_coefficients.clone()),
        |factors| opening_shape.scale_row_coefficients_by_group(&row_coefficients, factors),
    )?;
    let trace_eval_target = if let Some(final_claims) = &material.reduction_final_claims {
        if final_claims.len() != row_coefficients.len() || material.reduction_factors.is_none() {
            return Err(AkitaError::InvalidProof);
        }
        final_claims
            .iter()
            .zip(&row_coefficients)
            .fold(E::zero(), |acc, (&claim, &coefficient)| {
                acc + coefficient * claim
            })
    } else {
        if material.reduction_factors.is_some() {
            return Err(AkitaError::InvalidProof);
        }
        opening_shape.batched_eval_target(&trace_claim_coefficients, &material.openings)?
    };
    Ok(FoldPrefix {
        prepared_points: material.prepared_points,
        row_coefficients,
        trace_eval_target,
        trace_claim_coefficients,
        scalar_openings: material.openings,
    })
}
