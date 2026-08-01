use super::*;
use crate::{
    CompressionChainPlan, CompressionChainWitness, PackedNegativeBinary, COMPRESSION_MAP_COUNT,
    COMPRESSION_TARGET_BYTES, MAX_COMPRESSION_INPUT_BYTES,
};

/// Prover-side semantic inner rows for one commitment bundle.
///
/// One entry belongs to each polynomial in claim order. Every entry stores
/// `[source block][A row][A coefficient]` in the shared A ring dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaCommitmentHint<F: FieldCore> {
    inner_rows: Vec<RingVec<F>>,
    ring_dim: usize,
    outer_compression_stages: Vec<Vec<u8>>,
}

impl<F: FieldCore> AkitaCommitmentHint<F> {
    /// Construct a hint from semantic A-ring rows in polynomial order.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero or inconsistent ring dimension, unequal
    /// per-polynomial coefficient lengths, or storage above repository limits.
    pub fn new(ring_dim: usize, inner_rows: Vec<RingVec<F>>) -> Result<Self, AkitaError> {
        let hint = Self {
            inner_rows,
            ring_dim,
            outer_compression_stages: Vec::new(),
        };
        hint.validate_shape()
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        Ok(hint)
    }

    /// Construct a one-polynomial hint.
    pub fn singleton(inner_rows: RingVec<F>) -> Result<Self, AkitaError> {
        Self::new(inner_rows.ring_dim(), vec![inner_rows])
    }

    /// Construct a hint carrying the two packed outer-compression stages.
    pub fn new_with_outer_compression(
        ring_dim: usize,
        inner_rows: Vec<RingVec<F>>,
        witness: &CompressionChainWitness,
    ) -> Result<Self, AkitaError> {
        let hint = Self {
            inner_rows,
            ring_dim,
            outer_compression_stages: witness
                .stages()
                .iter()
                .map(|stage| stage.bytes().to_vec())
                .collect(),
        };
        hint.validate_shape()
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        Ok(hint)
    }

    /// Construct a one-polynomial hint carrying outer-compression stages.
    pub fn singleton_with_outer_compression(
        inner_rows: RingVec<F>,
        witness: &CompressionChainWitness,
    ) -> Result<Self, AkitaError> {
        Self::new_with_outer_compression(inner_rows.ring_dim(), vec![inner_rows], witness)
    }

    /// Shared A ring dimension.
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Borrow semantic A rows in polynomial order.
    pub fn inner_rows(&self) -> &[RingVec<F>] {
        &self.inner_rows
    }

    /// Rebuild the checked packed witness under the plan derived from the
    /// frozen commitment profile.
    pub fn outer_compression_witness(
        &self,
        plan: &CompressionChainPlan,
    ) -> Result<CompressionChainWitness, AkitaError> {
        if self.outer_compression_stages.len() != plan.maps().len() {
            return Err(AkitaError::InvalidInput(
                "commitment hint compression stage count disagrees with the derived plan".into(),
            ));
        }
        let stages = self
            .outer_compression_stages
            .iter()
            .zip(plan.maps())
            .map(|(bytes, map)| PackedNegativeBinary::from_bytes(*map, bytes.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        CompressionChainWitness::new(plan.clone(), stages)
    }

    /// Consume the hint and return semantic A rows in polynomial order.
    pub fn into_rows(self) -> Vec<RingVec<F>> {
        self.inner_rows
    }

    fn validate_shape(&self) -> Result<(), SerializationError> {
        if self.ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "commitment hint A ring dimension must be nonzero".into(),
            ));
        }
        checked_shape_len(self.inner_rows.len())?;
        if !matches!(
            self.outer_compression_stages.len(),
            0 | COMPRESSION_MAP_COUNT
        ) {
            return Err(SerializationError::InvalidData(
                "commitment hint must contain zero or exactly two compression stages".into(),
            ));
        }
        let mut packed_bytes = 0usize;
        for stage in &self.outer_compression_stages {
            packed_bytes = packed_bytes.checked_add(stage.len()).ok_or_else(|| {
                SerializationError::InvalidData(
                    "commitment hint packed compression length overflow".into(),
                )
            })?;
        }
        if packed_bytes > MAX_COMPRESSION_INPUT_BYTES + COMPRESSION_TARGET_BYTES * 2 {
            return Err(SerializationError::InvalidData(
                "commitment hint packed compression data exceeds the protocol envelope".into(),
            ));
        }
        let mut expected_coefficients = None;
        let mut total_coefficients = 0usize;
        for rows in &self.inner_rows {
            if rows.ring_dim() != self.ring_dim || !rows.coeff_len().is_multiple_of(self.ring_dim) {
                return Err(SerializationError::InvalidData(
                    "commitment hint row storage disagrees with its A ring dimension".into(),
                ));
            }
            if expected_coefficients
                .replace(rows.coeff_len())
                .is_some_and(|expected| expected != rows.coeff_len())
            {
                return Err(SerializationError::InvalidData(
                    "commitment hint polynomials have inconsistent row lengths".into(),
                ));
            }
            total_coefficients = total_coefficients
                .checked_add(rows.coeff_len())
                .ok_or_else(|| {
                    SerializationError::InvalidData(
                        "commitment hint coefficient count overflow".into(),
                    )
                })?;
            checked_shape_len(total_coefficients)?;
        }
        Ok(())
    }
}

impl<F: FieldCore + Valid> Valid for AkitaCommitmentHint<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.validate_shape()?;
        self.inner_rows.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaCommitmentHint<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.validate_shape()?;
        self.inner_rows
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        self.ring_dim.serialize_with_mode(&mut writer, compress)?;
        for rows in &self.inner_rows {
            rows.coeff_len()
                .serialize_with_mode(&mut writer, compress)?;
            for coefficient in rows.coeffs() {
                coefficient.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.outer_compression_stages
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        for stage in &self.outer_compression_stages {
            stage.len().serialize_with_mode(&mut writer, compress)?;
            writer.write_all(stage)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.inner_rows.len().serialized_size(compress)
            + self.ring_dim.serialized_size(compress)
            + self
                .inner_rows
                .iter()
                .map(|rows| {
                    rows.coeff_len().serialized_size(compress)
                        + rows
                            .coeffs()
                            .iter()
                            .map(|coefficient| coefficient.serialized_size(compress))
                            .sum::<usize>()
                })
                .sum::<usize>()
            + self
                .outer_compression_stages
                .iter()
                .map(|stage| stage.len().serialized_size(compress) + stage.len())
                .sum::<usize>()
            + self
                .outer_compression_stages
                .len()
                .serialized_size(compress)
    }
}

impl<F> AkitaDeserialize for AkitaCommitmentHint<F>
where
    F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
{
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let polynomial_count = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        checked_shape_len(polynomial_count)?;
        let ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "commitment hint A ring dimension must be nonzero".into(),
            ));
        }

        let mut inner_rows = Vec::new();
        reserve_shape_len(&mut inner_rows, polynomial_count)?;
        let mut expected_coefficients = None;
        let mut total_coefficients = 0usize;
        for _ in 0..polynomial_count {
            let coefficient_count =
                usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            if !coefficient_count.is_multiple_of(ring_dim) {
                return Err(SerializationError::InvalidData(
                    "commitment hint coefficient count is not divisible by its A ring dimension"
                        .into(),
                ));
            }
            if expected_coefficients
                .replace(coefficient_count)
                .is_some_and(|expected| expected != coefficient_count)
            {
                return Err(SerializationError::InvalidData(
                    "commitment hint polynomials have inconsistent row lengths".into(),
                ));
            }
            total_coefficients = total_coefficients
                .checked_add(coefficient_count)
                .ok_or_else(|| {
                    SerializationError::InvalidData(
                        "commitment hint coefficient count overflow".into(),
                    )
                })?;
            checked_shape_len(total_coefficients)?;

            let mut coefficients = Vec::new();
            reserve_shape_len(&mut coefficients, coefficient_count)?;
            for _ in 0..coefficient_count {
                coefficients.push(F::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            inner_rows.push(
                RingVec::from_coeffs_with_ring_dim(coefficients, ring_dim).map_err(|_| {
                    SerializationError::InvalidData(
                        "commitment hint row storage is malformed".into(),
                    )
                })?,
            );
        }

        let compression_stage_count =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if !matches!(compression_stage_count, 0 | COMPRESSION_MAP_COUNT) {
            return Err(SerializationError::InvalidData(
                "commitment hint must contain zero or exactly two compression stages".into(),
            ));
        }
        let mut outer_compression_stages = Vec::new();
        reserve_shape_len(&mut outer_compression_stages, compression_stage_count)?;
        let mut packed_bytes = 0usize;
        for _ in 0..compression_stage_count {
            let byte_count = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            packed_bytes = packed_bytes.checked_add(byte_count).ok_or_else(|| {
                SerializationError::InvalidData(
                    "commitment hint packed compression length overflow".into(),
                )
            })?;
            if packed_bytes > MAX_COMPRESSION_INPUT_BYTES + COMPRESSION_TARGET_BYTES * 2 {
                return Err(SerializationError::InvalidData(
                    "commitment hint packed compression data exceeds the protocol envelope".into(),
                ));
            }
            let mut bytes = vec![0u8; byte_count];
            reader.read_exact(&mut bytes)?;
            outer_compression_stages.push(bytes);
        }

        let hint = Self {
            inner_rows,
            ring_dim,
            outer_compression_stages,
        };
        hint.validate_shape()?;
        if validate == Validate::Yes {
            hint.check()?;
        }
        Ok(hint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Fp32;

    type F = Fp32<251>;

    fn rows(start: u64, coefficient_count: usize, ring_dim: usize) -> RingVec<F> {
        RingVec::from_coeffs_with_ring_dim(
            (0..coefficient_count)
                .map(|offset| F::from_u64(start + offset as u64))
                .collect(),
            ring_dim,
        )
        .unwrap()
    }

    #[test]
    fn hint_encoding_is_polynomial_count_shared_dimension_then_field_rows() {
        let hint = AkitaCommitmentHint::new(4, vec![rows(10, 8, 4), rows(30, 8, 4)]).unwrap();
        let mut encoded = Vec::new();
        hint.serialize_uncompressed(&mut encoded).unwrap();

        let mut expected = Vec::new();
        2usize.serialize_uncompressed(&mut expected).unwrap();
        4usize.serialize_uncompressed(&mut expected).unwrap();
        for row in hint.inner_rows() {
            row.coeff_len()
                .serialize_uncompressed(&mut expected)
                .unwrap();
            for coefficient in row.coeffs() {
                coefficient.serialize_uncompressed(&mut expected).unwrap();
            }
        }
        0usize.serialize_uncompressed(&mut expected).unwrap();
        assert_eq!(encoded, expected);

        let decoded =
            AkitaCommitmentHint::<F>::deserialize_uncompressed(&encoded[..], &()).unwrap();
        assert_eq!(decoded, hint);
        assert_eq!(decoded.ring_dim(), 4);
        assert_eq!(decoded.inner_rows()[0].coeffs()[0], F::from_u64(10));
        assert_eq!(decoded.inner_rows()[1].coeffs()[0], F::from_u64(30));
    }

    #[test]
    fn hint_constructor_rejects_inconsistent_semantic_rows() {
        assert!(AkitaCommitmentHint::<F>::new(0, Vec::new()).is_err());
        assert!(AkitaCommitmentHint::new(4, vec![rows(1, 8, 4), rows(2, 12, 4)]).is_err());
        assert!(AkitaCommitmentHint::new(4, vec![rows(1, 8, 4), rows(2, 8, 2)]).is_err());
    }

    #[test]
    fn hint_decoder_rejects_nonintegral_and_oversized_shapes() {
        let mut nonintegral = Vec::new();
        1usize.serialize_uncompressed(&mut nonintegral).unwrap();
        4usize.serialize_uncompressed(&mut nonintegral).unwrap();
        3usize.serialize_uncompressed(&mut nonintegral).unwrap();
        assert!(AkitaCommitmentHint::<F>::deserialize_uncompressed(&nonintegral[..], &()).is_err());

        let mut oversized = Vec::new();
        (DEFAULT_MAX_SEQUENCE_LEN + 1)
            .serialize_uncompressed(&mut oversized)
            .unwrap();
        4usize.serialize_uncompressed(&mut oversized).unwrap();
        assert!(AkitaCommitmentHint::<F>::deserialize_uncompressed(&oversized[..], &()).is_err());
    }
}
