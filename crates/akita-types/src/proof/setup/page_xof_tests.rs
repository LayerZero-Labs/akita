//! Differential oracle: the unbuffered sampler from 89ceb0e1, test-only.
use super::*;
use jolt_field::{Fp32, Prime128Offset275, Prime64Offset59, Zero};

struct ReferencePageXof {
    reader: SetupSeedXofReader,
}

impl ReferencePageXof {
    fn new<F: Field + CanonicalEncoding>(id: &AkitaSetupSeed, page_index: usize) -> Self {
        let mut xof = Shake256::default();
        absorb_len_prefixed(&mut xof, b"domain", PUBLIC_MATRIX_DOMAIN);
        let derivation_tag = match id.derivation {
            PublicMatrixDerivation::Shake256PagedV1 => PUBLIC_MATRIX_DERIVATION_TAG,
        };
        absorb_len_prefixed(&mut xof, b"derivation", derivation_tag);
        absorb_len_prefixed(
            &mut xof,
            b"page_field_elements",
            &(id.derivation.page_field_elements() as u64).to_le_bytes(),
        );
        absorb_len_prefixed(&mut xof, b"seed", &id.seed);
        absorb_len_prefixed(&mut xof, b"field", &field_modulus_bytes::<F>());
        absorb_len_prefixed(&mut xof, b"page", &(page_index as u64).to_le_bytes());
        Self {
            reader: xof.finalize_xof(),
        }
    }
}

fn field_modulus_bytes<F: Field + CanonicalEncoding>() -> [u8; 32] {
    akita_params::field_modulus_be_bytes::<F>()
        .expect("setup fields must have a modulus of at most 256 bits")
}

impl RngCore for ReferencePageXof {
    fn next_u32(&mut self) -> u32 {
        let mut buf = [0u8; 4];
        self.fill_bytes(&mut buf);
        u32::from_le_bytes(buf)
    }

    fn next_u64(&mut self) -> u64 {
        let mut buf = [0u8; 8];
        self.fill_bytes(&mut buf);
        u64::from_le_bytes(buf)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        XofReader::read(&mut self.reader, dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for ReferencePageXof {}

fn compare_pages<F: Field + CanonicalEncoding>() {
    for seed in [
        sample_akita_setup_seed(),
        [0; 32].into(),
        [31; 32].into(),
        [255; 32].into(),
    ] {
        for page in [0, 1, 7, 8724] {
            let mut reference = ReferencePageXof::new::<F>(&seed, page);
            let expected = (0..4096)
                .map(|_| F::random(&mut reference))
                .collect::<Vec<_>>();
            for (offset, count) in [(0, 0), (0, 1), (0, 4096), (15, 35), (4087, 9)] {
                let mut actual = vec![F::zero(); count];
                range::fill_page(&seed, page, offset, &mut actual);
                assert_eq!(actual, expected[offset..offset + count]);
            }
        }
        let mut expected = Vec::new();
        for page in 0..3 {
            let mut reference = ReferencePageXof::new::<F>(&seed, page);
            expected.extend((0..4096).map(|_| F::random(&mut reference)));
        }
        for count in [0, 4095, 4096, 4097, 8193] {
            let actual = derive_public_matrix_prefix::<F>(count, &seed);
            assert_eq!(actual.as_field_slice(), &expected[..count]);
        }
        for span in [0..0, 16..35, 4095..8193] {
            let mut actual = vec![F::zero(); span.len()];
            derive_public_matrix_range(span.start, &mut actual, &seed).unwrap();
            assert_eq!(actual, expected[span]);
        }
    }
}

#[test]
fn buffered_pages_match_unbuffered_reference_p64_p128_and_rejections() {
    compare_pages::<Prime64Offset59>();
    compare_pages::<Prime128Offset275>();
    // Frequent rejections exercise byte consumption beyond the nominal page bytes.
    compare_pages::<Fp32<251>>();
}

#[test]
fn buffered_rng_preserves_mixed_reads_at_every_rate_offset() {
    let seed = sample_akita_setup_seed();
    for offset in 0..=SETUP_XOF_RATE {
        let mut actual = SetupSeedPageXof::new::<Prime64Offset59>(&seed, 7);
        let mut reference = ReferencePageXof::new::<Prime64Offset59>(&seed, 7);
        actual.fill_bytes(&mut vec![0; offset]);
        reference.fill_bytes(&mut vec![0; offset]);
        for count in [0, 1, 4, 8, 16, 135, 136, 137, 4096] {
            let mut got = vec![0; count];
            let mut expected = vec![0; count];
            actual.fill_bytes(&mut got);
            reference.fill_bytes(&mut expected);
            assert_eq!(got, expected);
            assert_eq!(actual.next_u32(), reference.next_u32());
            assert_eq!(actual.next_u64(), reference.next_u64());
            actual.try_fill_bytes(&mut got).unwrap();
            reference.try_fill_bytes(&mut expected).unwrap();
            assert_eq!(got, expected);
        }
    }
}

fn forced_rejections<F: Field + CanonicalEncoding>(modulus: u128, width: usize) {
    let mut rng = SetupSeedPageXof::new::<F>(&sample_akita_setup_seed(), 0);
    // Two rejected candidates followed by the largest accepted value and zero.
    for (chunk, value) in
        rng.buffer
            .chunks_exact_mut(width)
            .zip([u128::MAX, modulus, modulus - 1, 0])
    {
        chunk.copy_from_slice(&value.to_le_bytes()[..width]);
    }
    rng.position = 0;
    assert_eq!(F::random(&mut rng), F::from_u128(modulus - 1));
    assert_eq!(rng.position, 3 * width);
    assert_eq!(F::random(&mut rng), F::zero());
    assert_eq!(rng.position, 4 * width);

    // A rejection at the end of the buffer must refill without dropping bytes.
    let seed = sample_akita_setup_seed();
    let mut rng = SetupSeedPageXof::new::<F>(&seed, 0);
    let mut reference = ReferencePageXof::new::<F>(&seed, 0);
    rng.position = SETUP_XOF_RATE - width;
    rng.buffer[rng.position..].copy_from_slice(&modulus.to_le_bytes()[..width]);
    assert_eq!(F::random(&mut rng), F::random(&mut reference));
    assert_eq!(F::random(&mut rng), F::random(&mut reference));
}

#[test]
fn buffered_rng_keeps_p64_and_p128_rejection_boundaries() {
    forced_rejections::<Prime64Offset59>((1u128 << 64) - 59, 8);
    forced_rejections::<Prime128Offset275>(u128::MAX - 274, 16);
}

fn compare_four_page_groups<F: Field + CanonicalEncoding>() {
    for seed in [
        sample_akita_setup_seed(),
        [0; 32].into(),
        [31; 32].into(),
        [255; 32].into(),
    ] {
        let mut expected = Vec::new();
        for page in 0..9 {
            let mut reference = ReferencePageXof::new::<F>(&seed, page);
            expected.extend((0..4096).map(|_| F::random(&mut reference)));
        }
        assert!(derive_public_matrix_prefix::<F>(0, &seed)
            .as_field_slice()
            .is_empty());
        for pages in 1..=9 {
            for short_by in [0, 1, 2560, 4095] {
                let count = pages * 4096 - short_by;
                let actual = derive_public_matrix_prefix::<F>(count, &seed);
                assert_eq!(
                    actual.as_field_slice(),
                    &expected[..count],
                    "pages {pages}, short by {short_by}"
                );
            }
        }
    }
}

#[test]
fn four_page_groups_match_unbuffered_oracle_all_remainders() {
    compare_four_page_groups::<Prime64Offset59>();
    compare_four_page_groups::<Prime128Offset275>();
    // Frequent rejection also forces lanes to consume different stream lengths.
    compare_four_page_groups::<Fp32<251>>();
}

#[test]
fn scalar_fallback_matches_page_oracle() {
    let seed = sample_akita_setup_seed();
    let count = 9 * seed.derivation.page_field_elements() - 17;
    let actual = derive_public_matrix_prefix::<Prime64Offset59>(count, &seed);
    let mut expected = vec![Prime64Offset59::zero(); count];
    for (index, page) in expected.chunks_mut(4096).enumerate() {
        range::fill_page(&seed, index, 0, page);
    }
    assert_eq!(actual.as_field_slice(), expected);
    // The Linux runner enables this feature with a baseline CPU target.
    #[cfg(all(target_arch = "x86_64", feature = "setup-force-scalar"))]
    assert!(!super::four_way::available());
}
