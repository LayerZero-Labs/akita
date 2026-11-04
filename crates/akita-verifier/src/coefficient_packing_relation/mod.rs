//! Compact coefficient-packing relation semantics for the Stage 2 verifier.
//!
//! The prover prepares compact Stage 2 semantics; CPU consumers explicitly
//! expand validated packing groups into relation events.
//! The verifier builds compact tensor factors from the same validated groups
//! and evaluates them at the Stage 2 final point.

use akita_error::AkitaError;
use akita_types::{
    validate_coefficient_packing_batch_groups, CoefficientPackingBatchSemanticInputs,
    FpExtEncoding, ValidatedCoefficientPackingGroup,
};
use jolt_algebra::{CanonicalEncoding, ExtField, Field};

mod compact;

pub(crate) use compact::{
    CoefficientPackingVerifierBatchSemantics, CoefficientPackingVerifierGroupSemantics,
};

fn prepare_coefficient_packing_verifier_group<F, E>(
    validated: &ValidatedCoefficientPackingGroup<'_, '_, F, E>,
) -> Result<CoefficientPackingVerifierGroupSemantics<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    let compact_factors = compact::prepare_compact_factors(validated)?;
    Ok(CoefficientPackingVerifierGroupSemantics {
        group_index: validated.group_index(),
        group_claim_range: validated.group_claim_range(),
        scalar_claim_weight: validated.scalar_claim_weight(),
        compact_factors,
    })
}

/// Prepare the compact packing factors used by the Stage 2 verifier without
/// constructing the prover's expanded event or segment tables.
pub(crate) fn prepare_coefficient_packing_verifier_batch_semantics<F, E>(
    inputs: CoefficientPackingBatchSemanticInputs<'_, '_, '_, F, E>,
) -> Result<CoefficientPackingVerifierBatchSemantics<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let groups = validate_coefficient_packing_batch_groups(&inputs, |group| {
        prepare_coefficient_packing_verifier_group(&group)
    })?;
    Ok(CoefficientPackingVerifierBatchSemantics { groups })
}

#[cfg(test)]
pub(crate) mod tests;

mod packed_direct;
