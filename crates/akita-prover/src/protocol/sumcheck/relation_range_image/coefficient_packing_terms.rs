//! Prover preparation of shared coefficient-packing Stage 2 terms.

use std::sync::Arc;

use super::{
    PreparedProverLinearTerms, StructuredLinearSegment, StructuredLinearTerm,
    StructuredLinearWeights,
};
use akita_field::{AkitaError, FieldCore};
use akita_types::{
    CoefficientPackingGroupSemantics, CoefficientPackingStage2Source,
    SubringCoefficientPackingGeometry,
};

/// Prepared method-specific weights and their authenticated input claim.
pub(crate) struct PreparedCoefficientPackingLinearTerms<E: FieldCore> {
    pub(crate) group_index: usize,
    pub(crate) geometry: SubringCoefficientPackingGeometry,
    pub(crate) linear_terms: PreparedProverLinearTerms<E>,
    pub(crate) weighted_scalar_opening_claim: E,
}

/// Compile the shared packing semantics into the prover's sparse Stage 2
/// representation.
///
/// This is deliberately a representation-only adapter. Challenge evaluation,
/// relation signs, source weights, row weights, and physical addresses remain
/// owned by `CoefficientPackingGroupSemantics`.
pub(crate) fn prepare_coefficient_packing_linear_terms<E: FieldCore>(
    semantics: &CoefficientPackingGroupSemantics<E>,
    authenticated_scalar_opening: E,
) -> Result<PreparedCoefficientPackingLinearTerms<E>, AkitaError> {
    let shared = semantics.stage2_terms();
    let sources = vec![
        Arc::<[E]>::from(shared.direct_opening_source()),
        Arc::<[E]>::from(shared.packing_z_source()),
    ];
    let segments = shared
        .segments()
        .iter()
        .map(|segment| {
            let physical = segment.physical_coefficients();
            let source = segment.source_coefficients();
            if physical.len() != source.len() {
                return Err(AkitaError::InvalidProof);
            }
            Ok(StructuredLinearSegment {
                physical_coefficient_start: physical.start,
                source_coefficient_start: source.start,
                coefficient_count: physical.len(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let terms = shared
        .terms()
        .iter()
        .map(|term| StructuredLinearTerm {
            factor: term.factor(),
            source_index: match term.source() {
                CoefficientPackingStage2Source::DirectOpening => 0,
                CoefficientPackingStage2Source::PackingZ => 1,
            },
            segment_range: term.segments(),
        })
        .collect();
    let weights = StructuredLinearWeights {
        sources,
        segments,
        terms,
        physical_field_len: shared.physical_field_len(),
    };
    let linear_terms = PreparedProverLinearTerms::from_structured_weights(
        &weights,
        semantics.relation_events().relation_coefficient_block_len(),
    )?;
    Ok(PreparedCoefficientPackingLinearTerms {
        group_index: semantics.group_index(),
        geometry: semantics.geometry(),
        linear_terms,
        weighted_scalar_opening_claim: shared.scalar_claim_weight() * authenticated_scalar_opening,
    })
}

#[cfg(test)]
#[path = "coefficient_packing_terms_tests.rs"]
mod tests;
