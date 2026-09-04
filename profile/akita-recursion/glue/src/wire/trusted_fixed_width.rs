//! Trusted canonical fixed-width setup decoding for Jolt benchmarks.

use super::*;
use jolt_field::{Fp128, Fp32, Fp64};

trait TrustedFixedWidthField: Field + CanonicalEncoding {
    const NAME: &'static str;
    const WIRE_BYTES: usize;
    const WIRE_ALIGNMENT: usize;

    /// Read one canonical field value from its fixed-width wire bytes.
    ///
    /// # Safety
    ///
    /// `source` must point to `WIRE_BYTES` readable bytes. When `ALIGNED` is
    /// true, it must also satisfy `WIRE_ALIGNMENT`.
    unsafe fn read_canonical<const ALIGNED: bool>(source: *const u8) -> Option<Self>;
}

impl<const P: u32> TrustedFixedWidthField for Fp32<P> {
    const NAME: &'static str = "Fp32";
    const WIRE_BYTES: usize = core::mem::size_of::<u32>();
    const WIRE_ALIGNMENT: usize = core::mem::align_of::<u32>();

    #[inline(always)]
    unsafe fn read_canonical<const ALIGNED: bool>(source: *const u8) -> Option<Self> {
        let source = source.cast::<u32>();
        // SAFETY: the caller proves the complete word is readable and proves
        // alignment before selecting the aligned monomorphization.
        let word = unsafe {
            if ALIGNED {
                source.read()
            } else {
                source.read_unaligned()
            }
        };
        let canonical = u32::from_le(word);
        (canonical < P).then(|| Self::from_canonical_u32(canonical))
    }
}

impl<const P: u64> TrustedFixedWidthField for Fp64<P> {
    const NAME: &'static str = "Fp64";
    const WIRE_BYTES: usize = core::mem::size_of::<u64>();
    const WIRE_ALIGNMENT: usize = core::mem::align_of::<u64>();

    #[inline(always)]
    unsafe fn read_canonical<const ALIGNED: bool>(source: *const u8) -> Option<Self> {
        let source = source.cast::<u64>();
        // SAFETY: the caller proves the complete word is readable and proves
        // alignment before selecting the aligned monomorphization.
        let word = unsafe {
            if ALIGNED {
                source.read()
            } else {
                source.read_unaligned()
            }
        };
        let canonical = u64::from_le(word);
        (canonical < P).then(|| Self::from_canonical_u64(canonical))
    }
}

impl<const P: u128> TrustedFixedWidthField for Fp128<P> {
    const NAME: &'static str = "Fp128";
    const WIRE_BYTES: usize = 2 * core::mem::size_of::<u64>();
    const WIRE_ALIGNMENT: usize = core::mem::align_of::<u64>();

    #[inline(always)]
    unsafe fn read_canonical<const ALIGNED: bool>(source: *const u8) -> Option<Self> {
        let source = source.cast::<u64>();
        // SAFETY: the caller proves both words are readable and proves
        // alignment before selecting the aligned monomorphization.
        let (low, high) = unsafe {
            if ALIGNED {
                (source.read(), source.add(1).read())
            } else {
                (source.read_unaligned(), source.add(1).read_unaligned())
            }
        };
        let canonical = (u128::from(u64::from_le(high)) << 64) | u128::from(u64::from_le(low));
        Self::from_u128_checked(canonical)
    }
}

// The private bound deliberately seals this benchmark-only inherent API to the
// three canonical field implementations above without exporting the unsafe
// word-read contract to downstream crates.
#[allow(private_bounds)]
impl<F, const D: usize, E> AkitaJoltInputs<F, D, E>
where
    F: TrustedFixedWidthField + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
    E: ExtField<F> + AkitaSerialize + AkitaDeserialize<Context = ()> + Valid,
{
    #[inline(never)]
    fn decode_trusted_fixed_width_payload<const ALIGNED: bool>(
        payload: &[u8],
        expected_num_field_elements: usize,
    ) -> Result<FlatMatrix<F>, SerializationError> {
        let expected_bytes = checked::product([expected_num_field_elements, F::WIRE_BYTES])
            .ok_or_else(|| {
                SerializationError::InvalidData(format!(
                    "trusted {} setup payload length overflow",
                    F::NAME
                ))
            })?;
        if payload.len() != expected_bytes {
            return Err(SerializationError::InvalidData(format!(
                "trusted {} payload length disagrees with the setup shape",
                F::NAME
            )));
        }
        if ALIGNED && !payload.as_ptr().addr().is_multiple_of(F::WIRE_ALIGNMENT) {
            return Err(SerializationError::InvalidData(format!(
                "trusted {} aligned decoder received a misaligned payload",
                F::NAME
            )));
        }

        let mut data = Vec::new();
        data.try_reserve_exact(expected_num_field_elements)
            .map_err(|_| {
                SerializationError::InvalidData("flat matrix allocation failed".to_string())
            })?;
        let mut source = payload.as_ptr();
        for _ in 0..expected_num_field_elements {
            // SAFETY: the exact byte-count check proves one complete fixed
            // width value remains for every iteration. The aligned branch
            // checks the source address once. The fallback uses unaligned
            // word reads. No Rust field layout is read from the wire.
            let field = unsafe { F::read_canonical::<ALIGNED>(source) }.ok_or_else(|| {
                SerializationError::InvalidData(format!("{} out of range", F::NAME))
            })?;
            data.push(field);
            // SAFETY: the exact byte-count check and loop bound prove this
            // advances within the payload or exactly to its end.
            source = unsafe { source.add(F::WIRE_BYTES) };
        }
        Ok(FlatMatrix::from_flat_data(data))
    }

    fn deserialize_trusted_fixed_width_setup_matrix(
        rest: &mut &[u8],
        expected_num_field_elements: usize,
    ) -> Result<FlatMatrix<F>, SerializationError> {
        let encoded_num_field_elements =
            usize::deserialize_with_mode(&mut *rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        if encoded_num_field_elements != expected_num_field_elements {
            return Err(SerializationError::InvalidData(
                "flat matrix field count does not match expected setup shape".to_string(),
            ));
        }

        let payload_len = checked::product([expected_num_field_elements, F::WIRE_BYTES])
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "akita-jolt setup matrix payload length overflow".to_string(),
                )
            })?;
        if rest.len() < payload_len {
            return Err(SerializationError::InvalidData(format!(
                "akita-jolt setup matrix claims {payload_len} payload bytes but only {} remain",
                rest.len()
            )));
        }
        let (payload, tail) = rest.split_at(payload_len);
        let matrix = if payload.as_ptr().addr().is_multiple_of(F::WIRE_ALIGNMENT) {
            Self::decode_trusted_fixed_width_payload::<true>(payload, expected_num_field_elements)?
        } else {
            Self::decode_trusted_fixed_width_payload::<false>(payload, expected_num_field_elements)?
        };
        *rest = tail;
        Ok(matrix)
    }

    fn deserialize_trusted_fixed_width_host_setup(
        rest: &mut &[u8],
        total_blob_len: usize,
    ) -> Result<AkitaVerifierSetup<F>, SerializationError> {
        let (seed, shared_matrix) = Self::decode_seed_and_matrix_with(
            rest,
            total_blob_len,
            Self::deserialize_trusted_fixed_width_setup_matrix,
        )?;
        let prefix_slots = Self::decode_prefix_slots(rest)?;
        AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(seed, shared_matrix),
            ),
            prefix_slots,
        )
        .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }

    /// Decode a host-produced benchmark artifact with its fixed-width setup.
    ///
    /// The host must first strictly decode the artifact, derive the setup from
    /// its seed, and verify the proof. This explicitly trusted benchmark path
    /// then validates the wire format, setup shape, fixed-width canonical field
    /// values, and every allocation bound without deriving the matrix again.
    /// Aligned inputs use one word load for fp32 and fp64 or two word loads for
    /// fp128. Misaligned inputs use allocation-free unaligned word reads.
    pub fn read_trusted_benchmark_artifact_bytes<Cfg>(
        bytes: &[u8],
        schedules: &TrustedScheduleCatalog,
    ) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = F, ExtField = E>,
    {
        Self::decode_from_bytes_with_setup::<Cfg>(
            bytes,
            schedules,
            Self::deserialize_trusted_fixed_width_host_setup,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::{fp128, fp32, fp64};
    use jolt_field::{One, PseudoMersenne, Ring, Zero};

    const TEST_D: usize = 256;

    macro_rules! fixed_width_decoder_tests {
        ($module:ident, $field:ty, $wire_bytes:expr, $alignment:expr, $noncanonical:expr) => {
            mod $module {
                use super::*;

                type TestF = $field;
                type TestInputs = AkitaJoltInputs<TestF, TEST_D>;

                fn encoded_matrix() -> (FlatMatrix<TestF>, Vec<u8>) {
                    let expected = FlatMatrix::from_flat_data(vec![
                        TestF::zero(),
                        TestF::one(),
                        TestF::from_u64(7),
                        TestF::zero() - TestF::one(),
                    ]);
                    let mut encoded = Vec::new();
                    expected
                        .serialize_with_mode(&mut encoded, BLOB_COMPRESS)
                        .expect("serialize matrix");
                    (expected, encoded)
                }

                #[test]
                fn accepts_every_payload_alignment() {
                    let (expected, encoded) = encoded_matrix();
                    let mut saw_aligned = false;
                    let mut saw_unaligned = false;

                    for offset in 0..$alignment {
                        let mut framed = vec![0u8; offset];
                        framed.extend_from_slice(&encoded);
                        let encoded_at_offset = &framed[offset..];
                        let payload = &encoded_at_offset[core::mem::size_of::<u64>()..];
                        saw_aligned |= payload.as_ptr().addr().is_multiple_of($alignment);
                        saw_unaligned |= !payload.as_ptr().addr().is_multiple_of($alignment);

                        let mut rest = encoded_at_offset;
                        let decoded = TestInputs::deserialize_trusted_fixed_width_setup_matrix(
                            &mut rest,
                            expected.num_field_elements(),
                        )
                        .expect("trusted matrix decode");
                        assert!(rest.is_empty());
                        assert_eq!(decoded, expected);
                    }

                    assert!(saw_aligned, "alignment sweep must cover aligned loads");
                    assert!(saw_unaligned, "alignment sweep must cover unaligned loads");
                }

                #[test]
                fn rejects_noncanonical_fields_without_consuming_payload() {
                    let (expected, mut encoded) = encoded_matrix();
                    encoded[core::mem::size_of::<u64>()..][..$wire_bytes]
                        .copy_from_slice(&$noncanonical);
                    let original_len = encoded.len();
                    let mut rest = encoded.as_slice();
                    let error = TestInputs::deserialize_trusted_fixed_width_setup_matrix(
                        &mut rest,
                        expected.num_field_elements(),
                    )
                    .expect_err("noncanonical field value must fail");
                    assert!(error.to_string().contains("out of range"));
                    assert_eq!(rest.len(), original_len - core::mem::size_of::<u64>());
                }
            }
        };
    }

    fixed_width_decoder_tests!(
        fp32_tests,
        fp32::Field,
        4,
        core::mem::align_of::<u32>(),
        u32::MAX.to_le_bytes()
    );
    fixed_width_decoder_tests!(
        fp64_tests,
        fp64::Field,
        8,
        core::mem::align_of::<u64>(),
        u64::MAX.to_le_bytes()
    );

    mod fp128_tests {
        use super::*;

        type TestF = fp128::Field;
        type TestInputs = AkitaJoltInputs<TestF, TEST_D>;

        fn encoded_matrix() -> (FlatMatrix<TestF>, Vec<u8>) {
            let p_minus_one = u128::MAX - <TestF as PseudoMersenne>::OFFSET;
            let expected = FlatMatrix::from_flat_data(vec![
                TestF::zero(),
                TestF::one(),
                TestF::from_u64(7),
                TestF::from_u128_checked(p_minus_one).expect("P - 1 is canonical"),
            ]);
            let mut encoded = Vec::new();
            expected
                .serialize_with_mode(&mut encoded, BLOB_COMPRESS)
                .expect("serialize matrix");
            (expected, encoded)
        }

        #[test]
        fn accepts_every_payload_alignment() {
            let (expected, encoded) = encoded_matrix();
            let mut saw_aligned = false;
            let mut saw_unaligned = false;

            for offset in 0..core::mem::align_of::<u64>() {
                let mut framed = vec![0u8; offset];
                framed.extend_from_slice(&encoded);
                let encoded_at_offset = &framed[offset..];
                let payload = &encoded_at_offset[core::mem::size_of::<u64>()..];
                saw_aligned |= payload.as_ptr().cast::<u64>().is_aligned();
                saw_unaligned |= !payload.as_ptr().cast::<u64>().is_aligned();

                let mut rest = encoded_at_offset;
                let decoded = TestInputs::deserialize_trusted_fixed_width_setup_matrix(
                    &mut rest,
                    expected.num_field_elements(),
                )
                .expect("trusted matrix decode");
                assert!(rest.is_empty());
                assert_eq!(decoded, expected);
            }

            assert!(saw_aligned, "alignment sweep must cover aligned loads");
            assert!(saw_unaligned, "alignment sweep must cover unaligned loads");
        }

        #[test]
        fn rejects_noncanonical_fields_without_consuming_payload() {
            let (expected, mut encoded) = encoded_matrix();
            encoded[core::mem::size_of::<u64>()..][..16].fill(0xff);
            let original_len = encoded.len();
            let mut rest = encoded.as_slice();
            let error = TestInputs::deserialize_trusted_fixed_width_setup_matrix(
                &mut rest,
                expected.num_field_elements(),
            )
            .expect_err("noncanonical fp128 value must fail");
            assert!(error.to_string().contains("Fp128 out of range"));
            assert_eq!(rest.len(), original_len - core::mem::size_of::<u64>());
        }
    }
}
