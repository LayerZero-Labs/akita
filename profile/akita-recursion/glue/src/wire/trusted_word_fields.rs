//! Trusted one-word setup decoding for the fp32 and fp64 Jolt benchmarks.

use super::*;
use akita_field::{Fp32, Fp64};

macro_rules! impl_trusted_word_decoder {
    ($field:ident, $word:ty, $constructor:ident, $field_name:literal) => {
        impl<const P: $word, const D: usize, E> AkitaJoltInputs<$field<P>, D, E>
        where
            $field<P>: FieldCore
                + CanonicalField
                + FromPrimitiveInt
                + RandomSampling
                + AkitaSerialize
                + AkitaDeserialize<Context = ()>
                + Valid,
            E: FieldCore
                + ExtField<$field<P>>
                + AkitaSerialize
                + AkitaDeserialize<Context = ()>
                + Valid,
        {
            #[inline(never)]
            fn decode_trusted_word_payload<const ALIGNED: bool>(
                payload: &[u8],
                expected_num_field_elements: usize,
            ) -> Result<FlatMatrix<$field<P>>, SerializationError> {
                const WORD_BYTES: usize = std::mem::size_of::<$word>();
                let expected_bytes = expected_num_field_elements
                    .checked_mul(WORD_BYTES)
                    .ok_or_else(|| {
                        SerializationError::InvalidData(format!(
                            "trusted {} setup payload length overflow",
                            $field_name
                        ))
                    })?;
                if payload.len() != expected_bytes {
                    return Err(SerializationError::InvalidData(format!(
                        "trusted {} payload length disagrees with the setup shape",
                        $field_name
                    )));
                }
                if ALIGNED && !payload.as_ptr().cast::<$word>().is_aligned() {
                    return Err(SerializationError::InvalidData(format!(
                        "trusted {} aligned decoder received a misaligned payload",
                        $field_name
                    )));
                }

                let mut data = Vec::new();
                data.try_reserve_exact(expected_num_field_elements)
                    .map_err(|_| {
                        SerializationError::InvalidData("flat matrix allocation failed".to_string())
                    })?;
                let mut word_ptr = payload.as_ptr().cast::<$word>();
                for _ in 0..expected_num_field_elements {
                    // SAFETY: the exact byte-count check proves one complete
                    // word remains for every iteration. The `ALIGNED` branch
                    // checks the source address once. The fallback uses an
                    // unaligned read. Every integer bit pattern is valid.
                    let word = unsafe {
                        let word = if ALIGNED {
                            word_ptr.read()
                        } else {
                            word_ptr.read_unaligned()
                        };
                        word_ptr = word_ptr.add(1);
                        word
                    };
                    let canonical = <$word>::from_le(word);
                    if canonical >= P {
                        return Err(SerializationError::InvalidData(format!(
                            "{} out of range",
                            $field_name
                        )));
                    }
                    data.push($field::<P>::$constructor(canonical));
                }
                Ok(FlatMatrix::from_flat_data(data))
            }

            fn deserialize_trusted_word_setup_matrix(
                rest: &mut &[u8],
                expected_num_field_elements: usize,
            ) -> Result<FlatMatrix<$field<P>>, SerializationError> {
                let encoded_num_field_elements = usize::deserialize_with_mode(
                    &mut *rest,
                    BLOB_COMPRESS,
                    BLOB_VALIDATE,
                    &(),
                )?;
                if encoded_num_field_elements != expected_num_field_elements {
                    return Err(SerializationError::InvalidData(
                        "flat matrix field count does not match expected setup shape".to_string(),
                    ));
                }

                let payload_len = expected_num_field_elements
                    .checked_mul(std::mem::size_of::<$word>())
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
                let matrix = if payload.as_ptr().cast::<$word>().is_aligned() {
                    Self::decode_trusted_word_payload::<true>(
                        payload,
                        expected_num_field_elements,
                    )?
                } else {
                    Self::decode_trusted_word_payload::<false>(
                        payload,
                        expected_num_field_elements,
                    )?
                };
                *rest = tail;
                Ok(matrix)
            }

            fn deserialize_trusted_word_host_setup(
                rest: &mut &[u8],
                total_blob_len: usize,
            ) -> Result<AkitaVerifierSetup<$field<P>>, SerializationError> {
                let (seed, shared_matrix) = Self::decode_seed_and_matrix_with(
                    rest,
                    total_blob_len,
                    Self::deserialize_trusted_word_setup_matrix,
                )?;
                let prefix_slots = Self::decode_prefix_slots(rest)?;
                AkitaVerifierSetup::from_parts(
                    Arc::new(AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                        seed,
                        shared_matrix,
                    )),
                    prefix_slots,
                )
                .map_err(|err| SerializationError::InvalidData(err.to_string()))
            }

            /// Decode a host-produced recursion artifact while trusting the
            /// cached setup matrix.
            ///
            /// This benchmark path validates the wire format, fields, and
            /// setup shape, but deliberately skips rederiving matrix
            /// coefficients from the seed. Aligned matrix bytes use one word
            /// load per field. Misaligned bytes use direct unaligned reads
            /// without a payload-sized staging allocation.
            pub fn read_trusted_word_host_artifact_bytes<Cfg>(
                bytes: &[u8],
            ) -> Result<Self, SerializationError>
            where
                Cfg: CommitmentConfig<Field = $field<P>, ExtField = E>,
            {
                Self::decode_from_bytes_with_setup::<Cfg>(
                    bytes,
                    Self::deserialize_trusted_word_host_setup,
                )
            }
        }
    };
}

impl_trusted_word_decoder!(Fp32, u32, from_canonical_u32, "Fp32");
impl_trusted_word_decoder!(Fp64, u64, from_canonical_u64, "Fp64");

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::{fp32, fp64};

    const TEST_D: usize = 256;

    macro_rules! word_decoder_tests {
        ($module:ident, $field:ty, $word:ty, $field_name:literal) => {
            mod $module {
                use super::*;

                type TestF = $field;

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

                    for offset in 0..std::mem::align_of::<$word>() {
                        let mut framed = vec![0u8; offset];
                        framed.extend_from_slice(&encoded);
                        let encoded_at_offset = &framed[offset..];
                        let payload = &encoded_at_offset[std::mem::size_of::<u64>()..];
                        saw_aligned |= payload.as_ptr().cast::<$word>().is_aligned();
                        saw_unaligned |= !payload.as_ptr().cast::<$word>().is_aligned();

                        let mut rest = encoded_at_offset;
                        let decoded = AkitaJoltInputs::<TestF, TEST_D>::
                                                    deserialize_trusted_word_setup_matrix(
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
                    encoded[std::mem::size_of::<u64>()..][..std::mem::size_of::<$word>()]
                        .fill(0xff);
                    let original_len = encoded.len();
                    let mut rest = encoded.as_slice();
                    let error =
                        AkitaJoltInputs::<TestF, TEST_D>::deserialize_trusted_word_setup_matrix(
                            &mut rest,
                            expected.num_field_elements(),
                        )
                        .expect_err("noncanonical field value must fail");
                    assert!(error
                        .to_string()
                        .contains(concat!($field_name, " out of range")));
                    assert_eq!(rest.len(), original_len - std::mem::size_of::<u64>());
                }
            }
        };
    }

    word_decoder_tests!(fp32_tests, fp32::Field, u32, "Fp32");
    word_decoder_tests!(fp64_tests, fp64::Field, u64, "Fp64");
}
