//! Bounded access to a canonical independently seeded public-matrix page.
use super::{AkitaSetupSeed, SetupSeedPageXof};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Derive at most one canonical SHAKE page, without materializing its prefix.
/// Page indices and sampling are identical to `derive_public_matrix_prefix`.
pub fn derive_public_matrix_page<F: Field + CanonicalEncoding>(
    id: &AkitaSetupSeed,
    page_index: usize,
    count: usize,
) -> Result<Vec<F>, AkitaError> {
    akita_params::field_modulus_be_bytes::<F>()?;
    if count > id.derivation.page_field_elements() || u64::try_from(page_index).is_err() {
        return Err(AkitaError::InvalidSetup(
            "invalid public matrix page extent".into(),
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| AkitaError::InvalidSetup("public matrix page allocation failed".into()))?;
    let mut rng = SetupSeedPageXof::new::<F>(id, page_index);
    for _ in 0..count {
        values.push(F::random(&mut rng));
    }
    Ok(values)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use jolt_field::Prime128Offset275;
    #[test]
    fn bounded_pages_match_prefix_and_reject_oversized_page() {
        let seed = AkitaSetupSeed::shake256_paged_v1([31; 32]);
        let count = seed.derivation.page_field_elements();
        let dense =
            super::super::derive_public_matrix_prefix::<Prime128Offset275>(count + 3, &seed);
        let mut pages = derive_public_matrix_page::<Prime128Offset275>(&seed, 0, count).unwrap();
        pages.extend(derive_public_matrix_page::<Prime128Offset275>(&seed, 1, 3).unwrap());
        assert_eq!(pages, dense.as_field_slice());
        assert!(derive_public_matrix_page::<Prime128Offset275>(&seed, 0, count + 1).is_err());
    }
}
