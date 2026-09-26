//! Challenge-free setup product envelope guard.
//!
//! Projection sizing lives in `layout/setup_projection.rs`.

use akita_error::AkitaError;

use jolt_field::Field;

use crate::proof::AkitaExpandedSetup;

/// Fail-closed envelope guard: `required` inner (`d_a`) rows must fit the shared
/// matrix prefix at `fold_ring_d`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidSetup`] when `required` exceeds the envelope.
pub fn ensure_setup_envelope<F: Field>(
    expanded: &AkitaExpandedSetup<F>,
    required: usize,
    fold_ring_d: usize,
) -> Result<(), AkitaError> {
    let required_fields = required
        .checked_mul(fold_ring_d)
        .ok_or_else(|| AkitaError::InvalidSetup("setup capacity requirement overflow".into()))?;
    if required_fields > expanded.shared_matrix().num_field_elements() {
        return Err(AkitaError::InvalidSetup(
            "shared matrix is too small for selected setup product".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    #[test]
    fn ensure_setup_envelope_rejects_undersized_matrix() {
        let seed = crate::AkitaSetupDescriptor {
            max_num_vars: 32,
            max_num_batched_polys: 1,
            num_field_elements: 32,
            setup_seed: [1u8; 32].into(),
        };
        let shared = crate::derive_public_matrix_prefix::<F>(32, &seed.setup_seed);
        let expanded =
            crate::AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared);
        let err = ensure_setup_envelope(&expanded, 2, 32).expect_err("undersized");
        assert!(matches!(err, AkitaError::InvalidSetup(_)));
    }
}
