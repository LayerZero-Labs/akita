use super::*;

/// Prover-side hint for one opening-point commitment bundle.
///
/// Stores flattened inner-row digit blocks and, when available, the corresponding
/// recomposed inner rows for all polynomials bundled into the single commitment
/// at one opening point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaCommitmentHint<F: FieldCore> {
    decomposed_digits: FlatDigitBlocks,
    recomposed_inner_row_coeffs: Vec<F>,
    recomposed_inner_row_block_sizes: Vec<usize>,
}

impl<F: FieldCore> AkitaCommitmentHint<F> {
    /// Flatten a batched commitment hint from per-polynomial kernel outputs.
    pub fn from_batched_commit<const D: usize>(
        decomposed: Vec<FlatDigitBlocks>,
        recomposed: Vec<Vec<Vec<CyclotomicRing<F, D>>>>,
    ) -> Self {
        let mut block_sizes = Vec::new();
        let total_planes: usize = decomposed.iter().map(|digits| digits.plane_count()).sum();
        let mut flat_digits = Vec::with_capacity(total_planes);
        for digits in &decomposed {
            block_sizes.extend_from_slice(digits.block_sizes());
            digits.extend_flat_digits::<D>(&mut flat_digits);
        }
        let decomposed_digits = FlatDigitBlocks::from_planes::<D>(flat_digits, block_sizes)
            .expect("batched hint flattening preserves block metadata");

        let (recomposed_inner_row_coeffs, recomposed_inner_row_block_sizes) =
            if recomposed.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                let recomposed_inner_row_block_sizes: Vec<usize> = recomposed
                    .iter()
                    .flat_map(|rows_by_poly| rows_by_poly.iter().map(Vec::len))
                    .collect();
                let total_recomposed_inner_rows: usize =
                    recomposed_inner_row_block_sizes.iter().sum();
                let mut recomposed_inner_row_coeffs =
                    Vec::with_capacity(total_recomposed_inner_rows * D);
                for rows_by_poly in &recomposed {
                    for block in rows_by_poly {
                        for ring in block {
                            recomposed_inner_row_coeffs.extend_from_slice(ring.coefficients());
                        }
                    }
                }
                (
                    recomposed_inner_row_coeffs,
                    recomposed_inner_row_block_sizes,
                )
            };

        Self {
            decomposed_digits,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
        }
    }

    /// Flatten a batched commitment hint that must carry recomposed inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the recomposed rows are absent.
    pub fn from_batched_commit_recursive<const D: usize>(
        decomposed: Vec<FlatDigitBlocks>,
        recomposed: Vec<Vec<Vec<CyclotomicRing<F, D>>>>,
    ) -> Result<Self, AkitaError> {
        let hint = Self::from_batched_commit::<D>(decomposed, recomposed);
        if hint.recomposed_inner_row_coeffs.is_empty() {
            return Err(AkitaError::InvalidInput(
                "missing recomposed inner rows in recursive commitment hint".to_string(),
            ));
        }
        Ok(hint)
    }

    /// Merge per-group commitment hints into one ring-relation witness hint.
    ///
    /// # Errors
    ///
    /// Returns an error if group counts mismatch or recomposed rows cannot be
    /// populated.
    pub fn merge_for_ring_relation<const D: usize>(
        hints: Vec<Self>,
        group_sizes: &[usize],
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<Self, AkitaError>
    where
        F: CanonicalField,
    {
        if hints.len() != group_sizes.len() {
            return Err(AkitaError::InvalidInput(
                "prover hint group count does not match commitment groups".to_string(),
            ));
        }

        let mut block_sizes = Vec::new();
        let mut flat_digits = Vec::new();
        let mut recomposed_inner_row_coeffs = Vec::new();
        let mut recomposed_inner_row_block_sizes = Vec::new();
        for (mut hint, &group_size) in hints.into_iter().zip(group_sizes.iter()) {
            if group_size == 0 {
                return Err(AkitaError::InvalidInput(
                    "prover hint group sizes must be nonzero".to_string(),
                ));
            }
            hint.ensure_recomposed_inner_rows::<D>(num_digits_open, log_basis)?;
            let digits = hint.decomposed_digits();
            block_sizes.extend_from_slice(digits.block_sizes());
            digits.extend_flat_digits::<D>(&mut flat_digits);
            recomposed_inner_row_block_sizes
                .extend_from_slice(hint.recomposed_inner_row_block_sizes());
            recomposed_inner_row_coeffs.extend_from_slice(hint.recomposed_inner_row_coeffs());
        }
        let decomposed_digits = FlatDigitBlocks::from_planes::<D>(flat_digits, block_sizes)
            .map_err(|_| {
                AkitaError::InvalidInput(
                    "ring-relation hint merge decomposed block metadata mismatch".to_string(),
                )
            })?;
        Ok(Self {
            decomposed_digits,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
        })
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        self.decomposed_digits.ensure_ring_dim::<D>()
    }

    /// Borrow flattened inner-row digit blocks for ring-switch (`t_hat` view).
    pub fn decomposed_digits(&self) -> &FlatDigitBlocks {
        &self.decomposed_digits
    }

    fn recomposed_inner_row_coeffs(&self) -> &[F] {
        &self.recomposed_inner_row_coeffs
    }

    fn recomposed_inner_row_block_sizes(&self) -> &[usize] {
        &self.recomposed_inner_row_block_sizes
    }

    /// Populate recomposed inner rows from flattened digit storage when absent.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_digits_open` is zero or block digit counts are inconsistent.
    pub fn ensure_recomposed_inner_rows<const D: usize>(
        &mut self,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField,
    {
        self.ensure_ring_dim::<D>()?;
        if !self.recomposed_inner_row_coeffs.is_empty() {
            if self.decomposed_digits.block_sizes().len()
                != self.recomposed_inner_row_block_sizes.len()
            {
                return Err(AkitaError::InvalidInput(
                    "commitment hint block metadata mismatch".to_string(),
                ));
            }
            let ring_count = self
                .recomposed_inner_row_block_sizes
                .iter()
                .try_fold(0usize, |acc, &block_size| {
                    acc.checked_add(block_size).ok_or_else(|| {
                        AkitaError::InvalidInput(
                            "commitment hint recomposed block size overflow".to_string(),
                        )
                    })
                })?;
            if ring_count.checked_mul(D) != Some(self.recomposed_inner_row_coeffs.len()) {
                return Err(AkitaError::InvalidInput(
                    "commitment hint recomposed block metadata does not cover coefficients"
                        .to_string(),
                ));
            }
            return Ok(());
        }
        if num_digits_open == 0 {
            return Err(AkitaError::InvalidSetup(
                "num_digits_open must be nonzero when recomposing inner rows".to_string(),
            ));
        }

        let digit_planes = self.decomposed_digits.flat_digits_trusted::<D>();

        let mut digit_offset = 0usize;
        let mut recomposed_inner_row_coeffs = Vec::new();
        let mut recomposed_inner_row_block_sizes =
            Vec::with_capacity(self.decomposed_digits.block_sizes().len());
        for &digit_block_size in self.decomposed_digits.block_sizes() {
            let digit_end = digit_offset + digit_block_size;
            if digit_end > digit_planes.len() {
                return Err(AkitaError::InvalidInput(
                    "commitment hint decomposed block data is truncated".to_string(),
                ));
            }
            let block_planes = &digit_planes[digit_offset..digit_end];
            digit_offset = digit_end;
            if !block_planes.len().is_multiple_of(num_digits_open) {
                return Err(AkitaError::InvalidSetup(format!(
                    "decomposed inner row block has {} planes, expected a multiple of num_digits_open={num_digits_open}",
                    block_planes.len()
                )));
            }
            let block_ring_count = block_planes.len() / num_digits_open;
            recomposed_inner_row_block_sizes.push(block_ring_count);
            for chunk in block_planes.chunks(num_digits_open) {
                let ring = CyclotomicRing::<F, D>::gadget_recompose_pow2_i8(chunk, log_basis);
                recomposed_inner_row_coeffs.extend_from_slice(ring.coefficients());
            }
        }
        if digit_offset != digit_planes.len() {
            return Err(AkitaError::InvalidInput(
                "commitment hint has trailing decomposed block data".to_string(),
            ));
        }
        self.recomposed_inner_row_coeffs = recomposed_inner_row_coeffs;
        self.recomposed_inner_row_block_sizes = recomposed_inner_row_block_sizes;
        Ok(())
    }

    /// Borrow recomposed inner rows as typed rings after [`Self::ensure_recomposed_inner_rows`].
    pub fn recomposed_inner_rows_trusted<const D: usize>(
        &self,
    ) -> Result<Vec<Vec<CyclotomicRing<F, D>>>, AkitaError> {
        self.ensure_ring_dim::<D>()?;
        if self.recomposed_inner_row_coeffs.is_empty() {
            return Err(AkitaError::InvalidInput(
                "missing recomposed inner rows in prover hint".to_string(),
            ));
        }
        let (ring_chunks, rem) = self.recomposed_inner_row_coeffs.as_chunks::<D>();
        if !rem.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.recomposed_inner_row_coeffs.len(),
            });
        }
        let mut rings = Vec::with_capacity(self.recomposed_inner_row_block_sizes.len());
        let mut offset = 0usize;
        for &block_size in &self.recomposed_inner_row_block_sizes {
            let end = offset + block_size;
            if end > ring_chunks.len() {
                return Err(AkitaError::InvalidInput(
                    "commitment hint recomposed block data is truncated".to_string(),
                ));
            }
            rings.push(
                ring_chunks[offset..end]
                    .iter()
                    .map(|coeffs| CyclotomicRing::from_coefficients(*coeffs))
                    .collect(),
            );
            offset = end;
        }
        if offset != ring_chunks.len() {
            return Err(AkitaError::InvalidInput(
                "commitment hint has trailing recomposed block data".to_string(),
            ));
        }
        Ok(rings)
    }

    #[must_use]
    pub fn ring_dim(&self) -> usize {
        self.decomposed_digits.ring_dim()
    }

    #[cfg(test)]
    pub(crate) fn from_storage_for_test(
        decomposed_digits: FlatDigitBlocks,
        recomposed_inner_row_coeffs: Vec<F>,
        recomposed_inner_row_block_sizes: Vec<usize>,
    ) -> Self {
        Self {
            decomposed_digits,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
        }
    }
}

impl<F: FieldCore + Valid> Valid for AkitaCommitmentHint<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.decomposed_digits.check()?;
        if self.decomposed_digits.block_sizes().len() != self.recomposed_inner_row_block_sizes.len()
        {
            return Err(SerializationError::InvalidData(
                "commitment hint block metadata mismatch".to_string(),
            ));
        }
        let ring_dim = self.decomposed_digits.ring_dim();
        if self.recomposed_inner_row_coeffs.is_empty() {
            if !self.recomposed_inner_row_block_sizes.is_empty() {
                return Err(SerializationError::InvalidData(
                    "commitment hint recomposed block metadata without coefficients".to_string(),
                ));
            }
            return Ok(());
        }
        if !self
            .recomposed_inner_row_coeffs
            .len()
            .is_multiple_of(ring_dim)
        {
            return Err(SerializationError::InvalidData(
                "commitment hint recomposed coefficient length is not a multiple of ring_dim"
                    .to_string(),
            ));
        }
        let ring_count = self
            .recomposed_inner_row_block_sizes
            .iter()
            .try_fold(0usize, |acc, &block_size| {
                acc.checked_add(block_size).ok_or_else(|| {
                    SerializationError::InvalidData(
                        "commitment hint recomposed block size overflow".to_string(),
                    )
                })
            })?;
        if ring_count.checked_mul(ring_dim) != Some(self.recomposed_inner_row_coeffs.len()) {
            return Err(SerializationError::InvalidData(
                "commitment hint recomposed block metadata does not cover coefficients".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaCommitmentHint<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.ring_dim()
            .serialize_with_mode(&mut writer, compress)?;
        self.decomposed_digits
            .serialize_with_mode(&mut writer, compress)?;
        self.recomposed_inner_row_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        self.recomposed_inner_row_block_sizes
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.ring_dim().serialized_size(compress)
            + self.decomposed_digits.serialized_size(compress)
            + self.recomposed_inner_row_coeffs.serialized_size(compress)
            + self.recomposed_inner_row_block_sizes.serialized_size(compress)
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
        let ring_dim = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let decomposed_digits = match ring_dim {
            32 => FlatDigitBlocks::deserialize_typed::<32, R>(&mut reader, compress, validate),
            64 => FlatDigitBlocks::deserialize_typed::<64, R>(&mut reader, compress, validate),
            128 => FlatDigitBlocks::deserialize_typed::<128, R>(&mut reader, compress, validate),
            256 => FlatDigitBlocks::deserialize_typed::<256, R>(&mut reader, compress, validate),
            _ => Err(SerializationError::InvalidData(format!(
                "unsupported ring dimension for commitment hint: {ring_dim}"
            ))),
        }?;
        if decomposed_digits.ring_dim() != ring_dim {
            return Err(SerializationError::InvalidData(format!(
                "commitment hint wire ring_dim {ring_dim} does not match digit storage {}",
                decomposed_digits.ring_dim()
            )));
        }
        let recomposed_inner_row_coeffs =
            Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let recomposed_inner_row_block_sizes =
            Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            decomposed_digits,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_algebra::CyclotomicRing;
    use akita_field::Fp32;
    use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Valid};

    type TestF = Fp32<251>;
    const D: usize = 32;

    fn sample_hint() -> AkitaCommitmentHint<TestF> {
        let decomposed = FlatDigitBlocks::from_blocks::<D>(vec![vec![[1i8; D], [2i8; D]]]);
        let ring = CyclotomicRing::<TestF, D>::from_coefficients([TestF::one(); D]);
        AkitaCommitmentHint::from_batched_commit::<D>(
            vec![decomposed],
            vec![vec![vec![ring]]],
        )
    }

    #[test]
    fn commitment_hint_serde_roundtrip() {
        let hint = sample_hint();
        let mut bytes = Vec::new();
        hint.serialize_compressed(&mut bytes).expect("serialize hint");
        assert_eq!(bytes.len(), hint.serialized_size(Compress::Yes));
        let decoded = AkitaCommitmentHint::<TestF>::deserialize_compressed(&bytes[..], &())
            .expect("deserialize hint");
        assert_eq!(decoded, hint);
    }

    #[test]
    fn commitment_hint_valid_rejects_recomposed_metadata_sum_mismatch() {
        let hint = sample_hint();
        let bad: AkitaCommitmentHint<TestF> = AkitaCommitmentHint::from_storage_for_test(
            hint.decomposed_digits().clone(),
            hint.recomposed_inner_row_coeffs().to_vec(),
            {
                let mut sizes = hint.recomposed_inner_row_block_sizes().to_vec();
                sizes.push(1);
                sizes
            },
        );
        let err = bad.check().expect_err("metadata sum mismatch must fail");
        assert!(matches!(err, SerializationError::InvalidData(_)));
    }

    #[test]
    fn commitment_hint_valid_rejects_metadata_without_coefficients() {
        let hint = sample_hint();
        let bad: AkitaCommitmentHint<TestF> = AkitaCommitmentHint::from_storage_for_test(
            hint.decomposed_digits().clone(),
            Vec::new(),
            hint.recomposed_inner_row_block_sizes().to_vec(),
        );
        let err = bad.check().expect_err("metadata without coeffs must fail");
        assert!(matches!(err, SerializationError::InvalidData(_)));
    }

    #[test]
    fn commitment_hint_deserialize_rejects_unsupported_ring_dim() {
        let mut bytes = Vec::new();
        sample_hint()
            .serialize_compressed(&mut bytes)
            .expect("serialize hint");
        bytes[0] = 48;
        let err = AkitaCommitmentHint::<TestF>::deserialize_compressed(&bytes[..], &());
        assert!(err.is_err());
    }
}
