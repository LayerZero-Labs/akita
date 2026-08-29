use super::*;
use crate::{
    CompressionChainPlan, CompressionChainWitness, PackedNegativeBinary, COMPRESSION_MAP_COUNT,
    COMPRESSION_TARGET_BYTES, MAX_COMPRESSION_INPUT_BYTES,
};

fn validate_compression_stage_count(stage_count: usize) -> Result<(), SerializationError> {
    if matches!(stage_count, 0 | COMPRESSION_MAP_COUNT) {
        Ok(())
    } else {
        Err(SerializationError::InvalidData(
            "commitment hint must contain zero or exactly two compression stages".into(),
        ))
    }
}

fn validate_compression_component_counts(
    stage_count: usize,
    quotient_count: usize,
) -> Result<(), SerializationError> {
    validate_compression_stage_count(stage_count)?;
    if matches!(
        (stage_count, quotient_count),
        (0, 0) | (COMPRESSION_MAP_COUNT, 0) | (COMPRESSION_MAP_COUNT, COMPRESSION_MAP_COUNT)
    ) {
        Ok(())
    } else {
        Err(SerializationError::InvalidData(
            "commitment hint must contain zero or one quotient per compression stage".into(),
        ))
    }
}

/// Prover-side semantic inner rows for one commitment bundle.
///
/// One entry belongs to each polynomial in claim order. Every entry stores
/// `[source block][A row][A coefficient]` in the shared A ring dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaCommitmentHint<F: Field> {
    inner_rows: Vec<RingVec<F>>,
    ring_dim: usize,
    outer_compression_stages: Vec<Vec<u8>>,
    outer_compression_quotients: Vec<RingVec<F>>,
}

impl<F: Field> AkitaCommitmentHint<F> {
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
            outer_compression_quotients: Vec::new(),
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
        quotients: &[RingVec<F>],
    ) -> Result<Self, AkitaError> {
        let hint = Self {
            inner_rows,
            ring_dim,
            outer_compression_stages: witness
                .stages()
                .iter()
                .map(|stage| stage.bytes().to_vec())
                .collect(),
            outer_compression_quotients: quotients.to_vec(),
        };
        hint.validate_shape()
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        Ok(hint)
    }

    /// Construct a one-polynomial hint carrying outer-compression stages.
    pub fn singleton_with_outer_compression(
        inner_rows: RingVec<F>,
        witness: &CompressionChainWitness,
        quotients: &[RingVec<F>],
    ) -> Result<Self, AkitaError> {
        Self::new_with_outer_compression(
            inner_rows.ring_dim(),
            vec![inner_rows],
            witness,
            quotients,
        )
    }

    /// Construct a one-polynomial hint carrying reduced-mode compression
    /// stages and no polynomial-modulus quotient images.
    pub fn singleton_with_reduced_outer_compression(
        inner_rows: RingVec<F>,
        witness: &CompressionChainWitness,
    ) -> Result<Self, AkitaError> {
        let hint = Self {
            ring_dim: inner_rows.ring_dim(),
            inner_rows: vec![inner_rows],
            outer_compression_stages: witness
                .stages()
                .iter()
                .map(|stage| stage.bytes().to_vec())
                .collect(),
            outer_compression_quotients: Vec::new(),
        };
        hint.validate_shape()
            .map_err(|error| AkitaError::InvalidInput(error.to_string()))?;
        Ok(hint)
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

    /// Rebuild a reduced-mode compression witness and reject any retained
    /// polynomial-modulus quotient images.
    pub fn reduced_outer_compression_witness(
        &self,
        plan: &CompressionChainPlan,
    ) -> Result<CompressionChainWitness, AkitaError> {
        if !self.outer_compression_quotients.is_empty() {
            return Err(AkitaError::InvalidInput(
                "reduced commitment hint must not retain compression quotient rows".into(),
            ));
        }
        self.outer_compression_witness(plan)
    }

    /// Recover the retained quotient rows under the derived compression plan.
    pub fn outer_compression_quotients(
        &self,
        plan: &CompressionChainPlan,
    ) -> Result<Vec<RingVec<F>>, AkitaError> {
        if self.outer_compression_quotients.len() != plan.maps().len() {
            return Err(AkitaError::InvalidInput(
                "commitment hint compression quotient count disagrees with the derived plan".into(),
            ));
        }
        for (quotient, map) in self.outer_compression_quotients.iter().zip(plan.maps()) {
            if quotient.ring_dim() != map.ring_dimension()
                || quotient.coeff_len() != map.output_coefficients()
            {
                return Err(AkitaError::InvalidInput(
                    "commitment hint compression quotient shape disagrees with the derived plan"
                        .into(),
                ));
            }
        }
        Ok(self.outer_compression_quotients.clone())
    }

    /// Validate every retained outer-compression component against one plan.
    pub fn validate_outer_compression(
        &self,
        plan: &CompressionChainPlan,
    ) -> Result<(), AkitaError> {
        self.outer_compression_witness(plan)?;
        self.outer_compression_quotients(plan).map(|_| ())
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
        validate_compression_component_counts(
            self.outer_compression_stages.len(),
            self.outer_compression_quotients.len(),
        )?;
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
        let mut quotient_coefficients = 0usize;
        for quotient in &self.outer_compression_quotients {
            if quotient.ring_dim() == 0 || !quotient.coeff_len().is_multiple_of(quotient.ring_dim())
            {
                return Err(SerializationError::InvalidData(
                    "commitment hint compression quotient has malformed ring storage".into(),
                ));
            }
            quotient_coefficients = quotient_coefficients
                .checked_add(quotient.coeff_len())
                .ok_or_else(|| {
                    SerializationError::InvalidData(
                        "commitment hint compression quotient length overflow".into(),
                    )
                })?;
        }
        if quotient_coefficients > MAX_COMPRESSION_INPUT_BYTES {
            return Err(SerializationError::InvalidData(
                "commitment hint compression quotients exceed the protocol envelope".into(),
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

impl<F: Field + Valid> Valid for AkitaCommitmentHint<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.validate_shape()?;
        self.inner_rows.check()?;
        self.outer_compression_quotients.check()
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for AkitaCommitmentHint<F> {
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
        self.outer_compression_quotients
            .len()
            .serialize_with_mode(&mut writer, compress)?;
        for quotient in &self.outer_compression_quotients {
            quotient
                .ring_dim()
                .serialize_with_mode(&mut writer, compress)?;
            quotient
                .coeff_len()
                .serialize_with_mode(&mut writer, compress)?;
            for coefficient in quotient.coeffs() {
                coefficient.serialize_with_mode(&mut writer, compress)?;
            }
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
            + self
                .outer_compression_quotients
                .len()
                .serialized_size(compress)
            + self
                .outer_compression_quotients
                .iter()
                .map(|quotient| {
                    quotient.ring_dim().serialized_size(compress)
                        + quotient.coeff_len().serialized_size(compress)
                        + quotient
                            .coeffs()
                            .iter()
                            .map(|coefficient| coefficient.serialized_size(compress))
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
}

impl<F> AkitaDeserialize for AkitaCommitmentHint<F>
where
    F: Field + Valid + AkitaDeserialize<Context = ()>,
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
        validate_compression_stage_count(compression_stage_count)?;
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

        let compression_quotient_count =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        validate_compression_component_counts(compression_stage_count, compression_quotient_count)?;
        let mut outer_compression_quotients = Vec::new();
        reserve_shape_len(&mut outer_compression_quotients, compression_quotient_count)?;
        let mut quotient_coefficients = 0usize;
        for _ in 0..compression_quotient_count {
            let quotient_ring_dim =
                usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let quotient_coeff_len =
                usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            if quotient_ring_dim == 0 || !quotient_coeff_len.is_multiple_of(quotient_ring_dim) {
                return Err(SerializationError::InvalidData(
                    "commitment hint compression quotient has malformed ring storage".into(),
                ));
            }
            quotient_coefficients = quotient_coefficients
                .checked_add(quotient_coeff_len)
                .ok_or_else(|| {
                    SerializationError::InvalidData(
                        "commitment hint compression quotient length overflow".into(),
                    )
                })?;
            if quotient_coefficients > MAX_COMPRESSION_INPUT_BYTES {
                return Err(SerializationError::InvalidData(
                    "commitment hint compression quotients exceed the protocol envelope".into(),
                ));
            }
            let mut coefficients = Vec::new();
            reserve_shape_len(&mut coefficients, quotient_coeff_len)?;
            for _ in 0..quotient_coeff_len {
                coefficients.push(F::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            outer_compression_quotients.push(
                RingVec::from_coeffs_with_ring_dim(coefficients, quotient_ring_dim).map_err(
                    |_| {
                        SerializationError::InvalidData(
                            "commitment hint compression quotient is malformed".into(),
                        )
                    },
                )?,
            );
        }

        let hint = Self {
            inner_rows,
            ring_dim,
            outer_compression_stages,
            outer_compression_quotients,
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
    use crate::sis::SisModulusProfileId;
    use jolt_field::{Fp32, Ring, Zero};

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

    #[test]
    fn hint_round_trips_exactly_two_derived_compression_stages() {
        let plan =
            CompressionChainPlan::for_complete_source(SisModulusProfileId::Q32Offset99, 8).unwrap();
        let stages = plan
            .maps()
            .iter()
            .map(|map| PackedNegativeBinary::from_bytes(*map, vec![0; map.packed_digit_bytes()]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let witness = CompressionChainWitness::new(plan.clone(), stages).unwrap();
        let quotients = plan
            .maps()
            .iter()
            .map(|map| {
                RingVec::from_coeffs_with_ring_dim(
                    vec![F::zero(); map.output_coefficients()],
                    map.ring_dimension(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let hint = AkitaCommitmentHint::new_with_outer_compression(
            4,
            vec![rows(10, 8, 4)],
            &witness,
            &quotients,
        )
        .unwrap();
        assert!(hint.reduced_outer_compression_witness(&plan).is_err());

        let mut encoded = Vec::new();
        hint.serialize_uncompressed(&mut encoded).unwrap();
        let decoded =
            AkitaCommitmentHint::<F>::deserialize_uncompressed(&encoded[..], &()).unwrap();
        assert_eq!(decoded, hint);
        assert_eq!(decoded.outer_compression_witness(&plan).unwrap(), witness);
        assert_eq!(
            decoded.outer_compression_quotients(&plan).unwrap(),
            quotients
        );

        let mut wrong_count = hint.clone();
        wrong_count.outer_compression_stages.pop();
        assert!(wrong_count.serialize_uncompressed(Vec::new()).is_err());

        let mut wrong_quotient_dimension = hint.clone();
        let quotient = &wrong_quotient_dimension.outer_compression_quotients[0];
        wrong_quotient_dimension.outer_compression_quotients[0] =
            RingVec::from_coeffs_with_ring_dim(quotient.coeffs().to_vec(), quotient.ring_dim() / 2)
                .unwrap();
        assert!(wrong_quotient_dimension
            .outer_compression_quotients(&plan)
            .is_err());

        let mut wrong_length = hint;
        wrong_length.outer_compression_stages[0].pop();
        assert!(wrong_length.outer_compression_witness(&plan).is_err());
    }

    #[test]
    fn reduced_hint_round_trips_compression_stages_without_quotients() {
        let plan =
            CompressionChainPlan::for_complete_source(SisModulusProfileId::Q32Offset99, 8).unwrap();
        let stages = plan
            .maps()
            .iter()
            .map(|map| PackedNegativeBinary::from_bytes(*map, vec![0; map.packed_digit_bytes()]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let witness = CompressionChainWitness::new(plan.clone(), stages).unwrap();
        let hint =
            AkitaCommitmentHint::singleton_with_reduced_outer_compression(rows(10, 8, 4), &witness)
                .unwrap();

        let mut encoded = Vec::new();
        hint.serialize_uncompressed(&mut encoded).unwrap();
        let decoded =
            AkitaCommitmentHint::<F>::deserialize_uncompressed(&encoded[..], &()).unwrap();

        assert_eq!(decoded, hint);
        assert_eq!(decoded.outer_compression_witness(&plan).unwrap(), witness);
        assert_eq!(
            decoded.reduced_outer_compression_witness(&plan).unwrap(),
            witness
        );
        assert!(decoded.outer_compression_quotients(&plan).is_err());
    }
}
