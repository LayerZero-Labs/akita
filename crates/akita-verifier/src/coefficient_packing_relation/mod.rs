//! Compact coefficient-packing relation semantics for the Stage 2 verifier.
//!
//! The prover prepares compact Stage 2 semantics; CPU consumers explicitly
//! expand validated packing groups into relation events.
//! The verifier builds compact tensor factors from the same validated groups
//! and evaluates them at the Stage 2 final point.

use akita_error::AkitaError;
use akita_types::{
    validate_coefficient_packing_batch_groups, CoefficientPackingBatchSemanticInputs, FpExtEncoding,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

mod compact;

pub(crate) use compact::{
    CoefficientPackingVerifierBatchSemantics, CoefficientPackingVerifierGroupSemantics,
};

/// Prepare the compact packing factors used by the Stage 2 verifier without
/// constructing the prover's expanded event or segment tables.
pub(crate) fn prepare_coefficient_packing_verifier_batch_semantics<F, E>(
    inputs: CoefficientPackingBatchSemanticInputs<'_, '_, '_, F, E>,
    replayed: Option<&crate::relation::PreparedRelationGroups<E>>,
) -> Result<CoefficientPackingVerifierBatchSemantics<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + FpExtEncoding<F>,
{
    let groups = validate_coefficient_packing_batch_groups(&inputs, |group| {
        let computed;
        // Ambient A challenges represent C(X^(d_a/s)). Reuse their value at
        // this fold's alpha only when d_a == s, so both consumers need C(alpha).
        let shared = match replayed {
            Some(crate::relation::PreparedRelationGroups::QuotientLift(groups))
                if group.geometry().a_ring_dimension()
                    == group.geometry().challenge_subring_dimension() =>
            {
                Some(
                    groups
                        .iter()
                        .find(|g| g.group_id == group.group_index())
                        .ok_or(AkitaError::InvalidProof)?
                        .multipliers
                        .c_alphas
                        .as_slice(),
                )
            }
            _ => None,
        };
        let values = if let Some(values) = shared {
            values
        } else {
            computed = group.challenge_alpha_values()?;
            &computed
        };
        let expected = group
            .num_claims()
            .checked_mul(group.num_live_blocks())
            .ok_or(AkitaError::InvalidProof)?;
        if values.len() != expected {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: values.len(),
            });
        }
        Ok(CoefficientPackingVerifierGroupSemantics {
            group_index: group.group_index(),
            group_claim_range: group.group_claim_range(),
            scalar_claim_weight: group.scalar_claim_weight(),
            compact_factors: compact::prepare_compact_factors(&group, values)?,
        })
    })?;
    Ok(CoefficientPackingVerifierBatchSemantics { groups })
}

#[cfg(test)]
pub(crate) mod tests;

mod packed_direct;
