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

/// Check the coefficients present against the seed-derived public stream.
///
/// An empty or truncated prefix passes. Callers must separately validate the
/// length against their expected matrix shape.
///
/// Returns an error if the field modulus exceeds the descriptor bound or any
/// coefficient differs from the public stream.
pub fn check_public_matrix_prefix<F: Field + CanonicalEncoding>(
    matrix: &[F],
    seed: &AkitaSetupSeed,
) -> Result<(), AkitaError> {
    akita_params::field_modulus_be_bytes::<F>()?;
    let page_size = seed.derivation.page_field_elements();
    let mut expected = vec![F::zero(); page_size];
    for (page_index, coeffs) in matrix.chunks(page_size).enumerate() {
        let page = expected.get_mut(..coeffs.len()).ok_or_else(|| {
            AkitaError::InvalidSetup("public matrix page exceeds the page size".into())
        })?;
        fill_page(seed, page_index, 0, page);
        if coeffs != page {
            return Err(AkitaError::InvalidSetup(
                "public matrix does not match its seed".into(),
            ));
        }
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
    use super::super::{derive_public_matrix_prefix, AkitaExpandedSetup};
    use super::*;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid, Validate};
    use jolt_field::{CanonicalBytes, Fp32, One, Prime32Offset99, Zero};

    #[test]
    fn first_stream_bytes_are_frozen() {
        // SHAKE256-paged-v1, seed [31; 32], Prime32Offset99. These are the
        // canonical little-endian bytes of the first eight field elements.
        let seed = AkitaSetupSeed::shake256_paged_v1([31; 32]);
        let setup = AkitaExpandedSetup::<Prime32Offset99>::derive_stream(seed.clone(), 8).unwrap();
        let mut bytes = [0u8; 32];
        for (value, output) in setup
            .shared_matrix
            .as_field_slice()
            .iter()
            .zip(bytes.chunks_exact_mut(4))
        {
            value.to_bytes_le(output);
        }
        assert_eq!(
            bytes,
            [
                189, 156, 192, 58, 115, 132, 203, 167, 146, 68, 103, 10, 41, 131, 23, 136, 9, 207,
                205, 117, 130, 69, 10, 207, 86, 213, 173, 131, 242, 190, 107, 18,
            ]
        );
        check_public_matrix_prefix(setup.shared_matrix.as_field_slice(), &seed).unwrap();
    }

    #[test]
    fn canonical_seed_bytes_match_setup_wire_encoding() {
        let seed = AkitaSetupSeed::shake256_paged_v1(std::array::from_fn(|i| i as u8));
        let bytes = seed.to_canonical_bytes();
        assert_eq!(bytes[0], 1);
        assert_eq!(bytes[1..], seed.seed);
        assert_eq!(AkitaSetupSeed::from_canonical_bytes(&bytes).unwrap(), seed);
        for compress in [Compress::Yes, Compress::No] {
            let mut serialized = Vec::new();
            seed.serialize_with_mode(&mut serialized, compress).unwrap();
            assert_eq!(serialized, bytes);
            assert_eq!(
                AkitaSetupSeed::deserialize_with_mode(&bytes[..], compress, Validate::Yes, &())
                    .unwrap(),
                seed
            );
        }
        for tag in [0, 2, u8::MAX] {
            let mut unknown = bytes;
            unknown[0] = tag;
            assert!(AkitaSetupSeed::from_canonical_bytes(&unknown).is_err());
            assert!(AkitaSetupSeed::deserialize_compressed(&unknown[..], &()).is_err());
        }
    }

    #[test]
    fn stream_setup_matches_prefix_without_pcs_provisioning() {
        type F = Fp32<251>;
        let seed = AkitaSetupSeed::shake256_paged_v1([3; 32]);
        for len in [1, 4096, 4103] {
            let setup = AkitaExpandedSetup::<F>::derive_stream(seed.clone(), len).unwrap();
            assert_eq!(setup.descriptor.max_num_vars, 0);
            assert_eq!(setup.descriptor.max_num_batched_polys, 1);
            assert_eq!(setup.descriptor.num_field_elements, len);
            assert_eq!(setup.descriptor.setup_seed, seed);
            assert_eq!(
                setup.shared_matrix,
                derive_public_matrix_prefix::<F>(len, &seed)
            );
            setup.check().unwrap();
        }
        assert!(AkitaExpandedSetup::<F>::derive_stream(seed.clone(), 0).is_err());
        assert!(AkitaExpandedSetup::<F>::derive_stream(seed, usize::MAX).is_err());
    }

    #[test]
    fn prefix_check_rejects_changes_on_both_sides_of_page_boundary() {
        type F = Fp32<251>;
        let seed = AkitaSetupSeed::shake256_paged_v1([3; 32]);
        let matrix = derive_public_matrix_prefix::<F>(4103, &seed);
        for len in [0, 1, 4096, 4103] {
            check_public_matrix_prefix(&matrix.as_field_slice()[..len], &seed).unwrap();
        }
        for index in [0, 4095, 4096, 4102] {
            let mut changed = matrix.as_field_slice().to_vec();
            changed[index] += F::one();
            assert!(check_public_matrix_prefix(&changed, &seed).is_err());
        }
        assert!(check_public_matrix_prefix(matrix.as_field_slice(), &[4; 32].into()).is_err());
    }

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
