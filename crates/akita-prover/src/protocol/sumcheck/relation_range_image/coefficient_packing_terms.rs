//! Prover preparation of shared coefficient-packing Stage 2 terms.

use super::PreparedProverLinearTerms;
use akita_error::AkitaError;
use akita_types::{CoefficientPackingGroupSemantics, SubringCoefficientPackingGeometry};
use jolt_field::Field;

/// Prepared method-specific weights and their authenticated input claim.
pub(in crate::protocol) struct PreparedCoefficientPackingLinearTerms<E: Field> {
    pub(in crate::protocol) group_index: usize,
    pub(in crate::protocol) geometry: SubringCoefficientPackingGeometry,
    pub(in crate::protocol) linear_terms: PreparedProverLinearTerms<E>,
    pub(in crate::protocol) weighted_scalar_opening_claim: E,
}

/// Move the shared packing semantics into the prover's sparse Stage 2 engine.
pub(in crate::protocol) fn prepare_coefficient_packing_linear_terms<E: Field>(
    semantics: CoefficientPackingGroupSemantics<E>,
    authenticated_scalar_opening: E,
) -> Result<PreparedCoefficientPackingLinearTerms<E>, AkitaError> {
    let (group_index, geometry, stage2_terms) = semantics.into_parts();
    let weighted_scalar_opening_claim =
        stage2_terms.scalar_claim_weight() * authenticated_scalar_opening;
    let linear_terms = PreparedProverLinearTerms::from_coefficient_packing(stage2_terms)?;
    Ok(PreparedCoefficientPackingLinearTerms {
        group_index,
        geometry,
        linear_terms,
        weighted_scalar_opening_claim,
    })
}

#[cfg(test)]
#[path = "coefficient_packing_terms_tests.rs"]
mod tests;
