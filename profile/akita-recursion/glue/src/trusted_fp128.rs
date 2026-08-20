//! Trusted cached-matrix decoding for the fp128 Jolt benchmark.

use super::*;
use akita_field::Fp128;

const FP128_BYTES: usize = 16;
const FP128_WORDS: usize = FP128_BYTES / std::mem::size_of::<u64>();

impl<const P: u128, const D: usize> AkitaJoltInputs<Fp128<P>, D>
where
    Fp128<P>: FieldCore
        + CanonicalField
        + FromPrimitiveInt
        + AkitaSerialize
        + AkitaDeserialize<Context = ()>
        + Valid,
{
    #[inline(never)]
    fn decode_trusted_fp128_payload<const ALIGNED: bool>(
        payload: &[u8],
        expected_num_field_elements: usize,
    ) -> Result<FlatMatrix<Fp128<P>>, SerializationError> {
        let expected_bytes = expected_num_field_elements
            .checked_mul(FP128_BYTES)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "trusted fp128 setup payload length overflow".to_string(),
                )
            })?;
        if payload.len() != expected_bytes {
            return Err(SerializationError::InvalidData(
                "trusted fp128 payload length disagrees with the setup shape".to_string(),
            ));
        }
        if ALIGNED && !payload.as_ptr().cast::<u64>().is_aligned() {
            return Err(SerializationError::InvalidData(
                "trusted fp128 aligned decoder received a misaligned payload".to_string(),
            ));
        }

        let mut data = Vec::new();
        data.try_reserve_exact(expected_num_field_elements)
            .map_err(|_| {
                SerializationError::InvalidData("flat matrix allocation failed".to_string())
            })?;
        let mut word_ptr = payload.as_ptr().cast::<u64>();
        for _ in 0..expected_num_field_elements {
            // SAFETY: the exact byte-count check above proves two complete u64
            // values remain for every iteration. The `ALIGNED` branch checks
            // the source address once and advances by two words, while the
            // fallback uses unaligned reads. Every u64 bit pattern is valid.
            let (low, high) = unsafe {
                let pair = if ALIGNED {
                    (word_ptr.read(), word_ptr.add(1).read())
                } else {
                    (word_ptr.read_unaligned(), word_ptr.add(1).read_unaligned())
                };
                word_ptr = word_ptr.add(FP128_WORDS);
                pair
            };
            let low = u64::from_le(low);
            let high = u64::from_le(high);
            let canonical = (u128::from(high) << 64) | u128::from(low);
            let field = Fp128::<P>::from_canonical_u128_checked(canonical)
                .ok_or_else(|| SerializationError::InvalidData("Fp128 out of range".to_string()))?;
            data.push(field);
        }
        Ok(FlatMatrix::from_flat_data(data))
    }

    fn deserialize_trusted_fp128_setup_matrix(
        rest: &mut &[u8],
        expected_num_field_elements: usize,
    ) -> Result<FlatMatrix<Fp128<P>>, SerializationError> {
        let encoded_num_field_elements =
            usize::deserialize_with_mode(&mut *rest, BLOB_COMPRESS, BLOB_VALIDATE, &())?;
        if encoded_num_field_elements != expected_num_field_elements {
            return Err(SerializationError::InvalidData(
                "flat matrix field count does not match expected setup shape".to_string(),
            ));
        }

        let payload_len = expected_num_field_elements
            .checked_mul(FP128_BYTES)
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
        let matrix = if payload.as_ptr().cast::<u64>().is_aligned() {
            Self::decode_trusted_fp128_payload::<true>(payload, expected_num_field_elements)?
        } else {
            Self::decode_trusted_fp128_payload::<false>(payload, expected_num_field_elements)?
        };
        *rest = tail;
        Ok(matrix)
    }

    fn deserialize_trusted_host_setup(
        rest: &mut &[u8],
        total_blob_len: usize,
    ) -> Result<AkitaVerifierSetup<Fp128<P>>, SerializationError> {
        let (seed, shared_matrix) = Self::decode_seed_and_matrix_with(
            rest,
            total_blob_len,
            Self::deserialize_trusted_fp128_setup_matrix,
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

    /// Decode a host-produced recursion artifact while trusting the cached
    /// setup matrix.
    ///
    /// This benchmark path validates the wire format, fields, and setup shape,
    /// but deliberately skips rederiving matrix coefficients from the seed.
    /// Aligned matrix bytes use two word loads per field. Misaligned bytes use
    /// direct unaligned reads without a payload-sized staging allocation.
    pub fn read_trusted_host_artifact_bytes<Cfg>(bytes: &[u8]) -> Result<Self, SerializationError>
    where
        Cfg: CommitmentConfig<Field = Fp128<P>, ExtField = Fp128<P>>,
    {
        Self::decode_from_bytes_with_setup::<Cfg>(bytes, Self::deserialize_trusted_host_setup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_config::proof_optimized::fp128;
    use akita_field::PseudoMersenneField;

    type TestF = fp128::Field;
    const TEST_D: usize = 256;

    fn encoded_matrix() -> (FlatMatrix<TestF>, Vec<u8>) {
        let p_minus_one = u128::MAX - <TestF as PseudoMersenneField>::MODULUS_OFFSET;
        let expected = FlatMatrix::from_flat_data(vec![
            TestF::zero(),
            TestF::one(),
            TestF::from_u64(7),
            TestF::from_canonical_u128_checked(p_minus_one).expect("P - 1 is canonical"),
        ]);
        let mut encoded = Vec::new();
        expected
            .serialize_with_mode(&mut encoded, BLOB_COMPRESS)
            .expect("serialize matrix");
        (expected, encoded)
    }

    #[test]
    fn trusted_decoder_accepts_every_payload_alignment() {
        let (expected, encoded) = encoded_matrix();
        let mut saw_aligned = false;
        let mut saw_unaligned = false;

        for offset in 0..std::mem::align_of::<u64>() {
            let mut framed = vec![0u8; offset];
            framed.extend_from_slice(&encoded);
            let encoded_at_offset = &framed[offset..];
            let payload = &encoded_at_offset[std::mem::size_of::<u64>()..];
            saw_aligned |= payload.as_ptr().cast::<u64>().is_aligned();
            saw_unaligned |= !payload.as_ptr().cast::<u64>().is_aligned();

            let mut rest = encoded_at_offset;
            let decoded = AkitaJoltInputs::<TestF, TEST_D>::deserialize_trusted_fp128_setup_matrix(
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
    fn trusted_decoder_rejects_noncanonical_fields_without_consuming_payload() {
        let (expected, mut encoded) = encoded_matrix();
        encoded[std::mem::size_of::<u64>()..][..FP128_BYTES].fill(0xff);
        let original_len = encoded.len();
        let mut rest = encoded.as_slice();
        let error = AkitaJoltInputs::<TestF, TEST_D>::deserialize_trusted_fp128_setup_matrix(
            &mut rest,
            expected.num_field_elements(),
        )
        .expect_err("noncanonical fp128 value must fail");
        assert!(error.to_string().contains("Fp128 out of range"));
        assert_eq!(rest.len(), original_len - std::mem::size_of::<u64>());
    }
}
