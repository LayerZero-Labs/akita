//! Bounded sampling of the canonical paged public field stream.
use super::{AkitaSetupSeed, SetupSeedPageXof};
use akita_error::AkitaError;
use jolt_field::{CanonicalEncoding, Field};

/// Fill a coefficient range without materializing the preceding setup prefix.
/// Sampling within a page starts at its beginning, preserving rejection sampling.
pub fn derive_public_matrix_range<F: Field + CanonicalEncoding>(
    begin: usize,
    output: &mut [F],
    seed: &AkitaSetupSeed,
) -> Result<(), AkitaError> {
    let _ = begin
        .checked_add(output.len())
        .ok_or_else(|| AkitaError::InvalidSetup("public matrix range overflow".into()))?;
    let page_size = seed.derivation.page_field_elements();
    let mut position = begin;
    let mut remaining = output;
    while !remaining.is_empty() {
        let offset = position % page_size;
        let count = (page_size - offset).min(remaining.len());
        let (page, rest) = remaining.split_at_mut(count);
        fill_page(seed, position / page_size, offset, page);
        position += count;
        remaining = rest;
    }
    Ok(())
}

pub(super) fn fill_page<F: Field + CanonicalEncoding>(
    seed: &AkitaSetupSeed,
    page: usize,
    offset: usize,
    output: &mut [F],
) {
    let mut rng = SetupSeedPageXof::new::<F>(seed, page);
    for _ in 0..offset {
        let _ = F::random(&mut rng);
    }
    for value in output {
        *value = F::random(&mut rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fp32, Zero};

    #[test]
    fn ranges_preserve_sampling_and_page_boundaries() {
        type F = Fp32<251>;
        for seed in [super::super::sample_akita_setup_seed(), [37; 32].into()] {
            let expected = super::super::derive_public_matrix_prefix::<F>(12345, &seed);
            for range in [0..0, 0..513, 63..129, 4095..8193, 12288..12345] {
                let mut actual = vec![F::zero(); range.len()];
                derive_public_matrix_range(range.start, &mut actual, &seed).unwrap();
                assert_eq!(actual, expected.as_field_slice()[range]);
            }
            assert!(derive_public_matrix_range(usize::MAX, &mut [F::zero()], &seed).is_err());
        }
    }
}
