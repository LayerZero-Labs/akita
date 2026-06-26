use super::*;

/// Prover-side hint for one opening-point commitment bundle.
///
/// Stores per-polynomial decomposed inner rows and, when available, the
/// corresponding recomposed inner rows for all polynomials bundled into the
/// single commitment at one opening point.
#[derive(Debug, Clone)]
pub struct AkitaCommitmentHint<F: FieldCore, const D: usize> {
    /// Per-polynomial digit decompositions of the inner `A * s_i` rows.
    pub decomposed_inner_rows: Vec<FlatDigitBlocks<D>>,
    /// Optional recomposed inner rows grouped by polynomial then block.
    recomposed_inner_rows: Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> AkitaCommitmentHint<F, D> {
    /// Construct a new batched hint from per-polynomial digit streams.
    pub fn new(decomposed_inner_rows: Vec<FlatDigitBlocks<D>>) -> Self {
        Self {
            decomposed_inner_rows,
            recomposed_inner_rows: None,
            _marker: PhantomData,
        }
    }

    /// Construct a singleton batched hint from one polynomial's digit stream.
    pub fn singleton(decomposed_inner_rows: FlatDigitBlocks<D>) -> Self {
        Self::new(vec![decomposed_inner_rows])
    }

    /// Construct a batched hint that also preserves recomposed inner rows.
    pub fn with_recomposed_inner_rows(
        decomposed_inner_rows: Vec<FlatDigitBlocks<D>>,
        recomposed_inner_rows: Vec<Vec<Vec<CyclotomicRing<F, D>>>>,
    ) -> Self {
        Self {
            decomposed_inner_rows,
            recomposed_inner_rows: Some(recomposed_inner_rows),
            _marker: PhantomData,
        }
    }

    /// Construct a singleton batched hint that also preserves recomposed rows.
    pub fn singleton_with_recomposed_inner_rows(
        decomposed_inner_rows: FlatDigitBlocks<D>,
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
    ) -> Self {
        Self::with_recomposed_inner_rows(vec![decomposed_inner_rows], vec![recomposed_inner_rows])
    }

    /// Get the optional recomposed inner rows grouped by polynomial.
    pub fn recomposed_inner_rows(&self) -> Option<&[Vec<Vec<CyclotomicRing<F, D>>>]> {
        self.recomposed_inner_rows.as_deref()
    }

    /// Consume the hint and return per-polynomial digit rows plus optional
    /// recomposed inner rows.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Vec<FlatDigitBlocks<D>>,
        Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
    ) {
        (self.decomposed_inner_rows, self.recomposed_inner_rows)
    }

    /// Populate recomposed inner rows from the decomposed rows when absent.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_digits_open` is zero or if any decomposed inner
    /// row block length is not a multiple of `num_digits_open`.
    pub fn ensure_recomposed_inner_rows(
        &mut self,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField,
    {
        if self.recomposed_inner_rows.is_some() {
            return Ok(());
        }
        if num_digits_open == 0 {
            return Err(AkitaError::InvalidSetup(
                "num_digits_open must be nonzero when recomposing inner rows".to_string(),
            ));
        }

        let recomposed_inner_rows = self
            .decomposed_inner_rows
            .iter()
            .map(|digits| {
                digits
                    .iter_blocks()
                    .map(|block| {
                        if block.len() % num_digits_open != 0 {
                            return Err(AkitaError::InvalidSetup(format!(
                                "decomposed inner row block has {} planes, expected a multiple of num_digits_open={num_digits_open}",
                                block.len()
                            )));
                        }
                        Ok(block
                            .chunks(num_digits_open)
                            .map(|digits| {
                                CyclotomicRing::gadget_recompose_pow2_i8(digits, log_basis)
                            })
                            .collect())
                    })
                    .collect()
            })
            .collect::<Result<Vec<Vec<Vec<CyclotomicRing<F, D>>>>, AkitaError>>()?;
        self.recomposed_inner_rows = Some(recomposed_inner_rows);
        Ok(())
    }

    /// Flatten the batched hint into the ring-switch view over all claims.
    ///
    ///
    /// # Panics
    ///
    /// Panics if the flattened digit planes do not match the concatenated
    /// block-size metadata. This would indicate an internal bug, since the
    /// flattened view is derived directly from well-formed component hints.
    #[allow(clippy::type_complexity)]
    pub fn into_flat_parts(self) -> (FlatDigitBlocks<D>, Option<Vec<Vec<CyclotomicRing<F, D>>>>) {
        let mut block_sizes = Vec::new();
        let total_planes: usize = self
            .decomposed_inner_rows
            .iter()
            .map(|digits| digits.flat_digits().len())
            .sum();
        let mut flat_digits = Vec::with_capacity(total_planes);
        for digits in &self.decomposed_inner_rows {
            block_sizes.extend_from_slice(digits.block_sizes());
            digits.extend_flat_digits(&mut flat_digits);
        }
        let decomposed_inner_rows = FlatDigitBlocks::new(flat_digits, block_sizes)
            .expect("batched hint flattening preserves block metadata");
        let recomposed_inner_rows = self
            .recomposed_inner_rows
            .map(|rows_by_poly| rows_by_poly.into_iter().flatten().collect());
        (decomposed_inner_rows, recomposed_inner_rows)
    }
}

impl<F: FieldCore, const D: usize> PartialEq for AkitaCommitmentHint<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.decomposed_inner_rows == other.decomposed_inner_rows
            && self.recomposed_inner_rows == other.recomposed_inner_rows
    }
}

impl<F: FieldCore, const D: usize> Eq for AkitaCommitmentHint<F, D> {}

impl<F: FieldCore + Valid, const D: usize> Valid for AkitaCommitmentHint<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        self.decomposed_inner_rows.check()?;
        if let Some(rows_by_poly) = &self.recomposed_inner_rows {
            if rows_by_poly.len() != self.decomposed_inner_rows.len() {
                return Err(SerializationError::InvalidData(
                    "recomposed hint rows must match decomposed polynomial count".to_string(),
                ));
            }
            rows_by_poly.check()?;
        }
        Ok(())
    }
}

impl<F: FieldCore + AkitaSerialize, const D: usize> AkitaSerialize for AkitaCommitmentHint<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.decomposed_inner_rows
            .serialize_with_mode(&mut writer, compress)?;
        self.recomposed_inner_rows
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(rows) = &self.recomposed_inner_rows {
            rows.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.decomposed_inner_rows.serialized_size(compress)
            + self
                .recomposed_inner_rows
                .is_some()
                .serialized_size(compress)
            + self
                .recomposed_inner_rows
                .as_ref()
                .map_or(0, |rows| rows.serialized_size(compress))
    }
}

impl<F, const D: usize> AkitaDeserialize for AkitaCommitmentHint<F, D>
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
        let decomposed_inner_rows =
            Vec::<FlatDigitBlocks<D>>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let has_recomposed = bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let recomposed_inner_rows = if has_recomposed {
            Some(
                Vec::<Vec<Vec<CyclotomicRing<F, D>>>>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?,
            )
        } else {
            None
        };
        let out = Self {
            decomposed_inner_rows,
            recomposed_inner_rows,
            _marker: PhantomData,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

/// Runtime ring-degree-erased commitment hint for setup-prefix slots and similar storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErasedCommitmentHint<F: FieldCore> {
    decomposed_inner_rows: Vec<i8>,
    decomposed_inner_row_block_sizes: Vec<usize>,
    recomposed_inner_row_coeffs: Vec<F>,
    recomposed_inner_row_block_sizes: Vec<usize>,
    ring_dim: usize,
}

impl<F: FieldCore> ErasedCommitmentHint<F> {
    /// Flatten a typed prover hint for D-free slot storage.
    pub fn from_typed<const D: usize>(hint: AkitaCommitmentHint<F, D>) -> Self {
        let (flat_hint_digits, recomposed_inner_rows) = hint.into_flat_parts();
        let decomposed_inner_row_block_sizes = flat_hint_digits.block_sizes().to_vec();
        let total_digit_planes: usize = flat_hint_digits.flat_digits().len();
        let mut decomposed_inner_rows = Vec::with_capacity(total_digit_planes * D);
        for plane in flat_hint_digits.flat_digits() {
            decomposed_inner_rows.extend_from_slice(plane);
        }

        let (recomposed_inner_row_coeffs, recomposed_inner_row_block_sizes) =
            if let Some(recomposed_inner_rows) = recomposed_inner_rows {
                let recomposed_inner_row_block_sizes: Vec<usize> =
                    recomposed_inner_rows.iter().map(Vec::len).collect();
                let total_recomposed_inner_rows: usize =
                    recomposed_inner_row_block_sizes.iter().sum();
                let mut recomposed_inner_row_coeffs =
                    Vec::with_capacity(total_recomposed_inner_rows * D);
                for block in &recomposed_inner_rows {
                    for ring in block {
                        recomposed_inner_row_coeffs.extend_from_slice(ring.coefficients());
                    }
                }
                (
                    recomposed_inner_row_coeffs,
                    recomposed_inner_row_block_sizes,
                )
            } else {
                (Vec::new(), Vec::new())
            };

        Self {
            decomposed_inner_rows,
            decomposed_inner_row_block_sizes,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
            ring_dim: D,
        }
    }

    /// Reconstruct the typed prover hint without recomputing inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match the stored hint,
    /// or if flattened block metadata is inconsistent.
    pub fn to_typed<const D: usize>(&self) -> Result<AkitaCommitmentHint<F, D>, AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "erased commitment hint ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if self.decomposed_inner_row_block_sizes.len()
            != self.recomposed_inner_row_block_sizes.len()
        {
            return Err(AkitaError::InvalidInput(
                "erased commitment hint block metadata mismatch".to_string(),
            ));
        }

        let (flat_digits, digit_remainder) = self.decomposed_inner_rows.as_chunks::<D>();
        if !digit_remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.decomposed_inner_rows.len(),
            });
        }
        let (flat_recomposed_rows, recomposed_remainder) =
            self.recomposed_inner_row_coeffs.as_chunks::<D>();
        if !recomposed_remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.recomposed_inner_row_coeffs.len(),
            });
        }

        let mut digit_offset = 0usize;
        let mut recomposed_offset = 0usize;
        let mut decomposed_inner_rows = Vec::with_capacity(flat_digits.len());
        let mut recomposed_inner_rows =
            Vec::with_capacity(self.recomposed_inner_row_block_sizes.len());

        for (&digit_block_size, &recomposed_block_size) in self
            .decomposed_inner_row_block_sizes
            .iter()
            .zip(self.recomposed_inner_row_block_sizes.iter())
        {
            let digit_end = digit_offset + digit_block_size;
            let recomposed_end = recomposed_offset + recomposed_block_size;
            if digit_end > flat_digits.len() || recomposed_end > flat_recomposed_rows.len() {
                return Err(AkitaError::InvalidInput(
                    "erased commitment hint block data is truncated".to_string(),
                ));
            }

            decomposed_inner_rows.extend_from_slice(&flat_digits[digit_offset..digit_end]);
            recomposed_inner_rows.push(
                flat_recomposed_rows[recomposed_offset..recomposed_end]
                    .iter()
                    .map(|coeffs| CyclotomicRing::from_coefficients(*coeffs))
                    .collect(),
            );
            digit_offset = digit_end;
            recomposed_offset = recomposed_end;
        }

        if digit_offset != flat_digits.len() || recomposed_offset != flat_recomposed_rows.len() {
            return Err(AkitaError::InvalidInput(
                "erased commitment hint has trailing block data".to_string(),
            ));
        }

        let decomposed_inner_rows = FlatDigitBlocks::new(
            decomposed_inner_rows,
            self.decomposed_inner_row_block_sizes.clone(),
        )?;
        Ok(AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
            decomposed_inner_rows,
            recomposed_inner_rows,
        ))
    }

    #[must_use]
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    pub(crate) fn serialize_with_mode_for_ring_dim<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
        ring_dim: usize,
    ) -> Result<(), SerializationError>
    where
        F: AkitaSerialize,
    {
        if self.ring_dim != ring_dim {
            return Err(SerializationError::InvalidData(format!(
                "erased commitment hint ring_d={} does not match slot id d_setup={ring_dim}",
                self.ring_dim
            )));
        }
        match ring_dim {
            32 => self
                .to_typed::<32>()
                .map_err(|err| {
                    SerializationError::InvalidData(format!(
                        "erased commitment hint retyped serialization failed: {err}"
                    ))
                })?
                .serialize_with_mode(&mut writer, compress),
            64 => self
                .to_typed::<64>()
                .map_err(|err| {
                    SerializationError::InvalidData(format!(
                        "erased commitment hint retyped serialization failed: {err}"
                    ))
                })?
                .serialize_with_mode(&mut writer, compress),
            128 => self
                .to_typed::<128>()
                .map_err(|err| {
                    SerializationError::InvalidData(format!(
                        "erased commitment hint retyped serialization failed: {err}"
                    ))
                })?
                .serialize_with_mode(&mut writer, compress),
            256 => self
                .to_typed::<256>()
                .map_err(|err| {
                    SerializationError::InvalidData(format!(
                        "erased commitment hint retyped serialization failed: {err}"
                    ))
                })?
                .serialize_with_mode(&mut writer, compress),
            _ => Err(SerializationError::InvalidData(format!(
                "unsupported ring dimension for erased commitment hint: {ring_dim}"
            ))),
        }
    }

    pub(crate) fn serialized_size_for_ring_dim(&self, compress: Compress, ring_dim: usize) -> usize
    where
        F: AkitaSerialize,
    {
        if self.ring_dim != ring_dim {
            return 0;
        }
        match ring_dim {
            32 => self
                .to_typed::<32>()
                .map(|typed| typed.serialized_size(compress))
                .unwrap_or(0),
            64 => self
                .to_typed::<64>()
                .map(|typed| typed.serialized_size(compress))
                .unwrap_or(0),
            128 => self
                .to_typed::<128>()
                .map(|typed| typed.serialized_size(compress))
                .unwrap_or(0),
            256 => self
                .to_typed::<256>()
                .map(|typed| typed.serialized_size(compress))
                .unwrap_or(0),
            _ => 0,
        }
    }

    pub(crate) fn deserialize_with_mode_for_ring_dim<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ring_dim: usize,
    ) -> Result<Self, SerializationError>
    where
        F: Valid + AkitaDeserialize<Context = ()>,
    {
        crate::dispatch_ring_dim_result!(ring_dim, |D| {
            let typed = AkitaCommitmentHint::<F, D>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?;
            Ok(Self::from_typed(typed))
        })
        .map_err(|err| SerializationError::InvalidData(err.to_string()))
    }
}

impl<F: FieldCore + Valid> Valid for ErasedCommitmentHint<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if !crate::SUPPORTED_RING_DIMS.contains(&self.ring_dim) {
            return Err(SerializationError::InvalidData(format!(
                "erased commitment hint has unsupported ring_dim={}",
                self.ring_dim
            )));
        }
        if self.decomposed_inner_row_block_sizes.len()
            != self.recomposed_inner_row_block_sizes.len()
        {
            return Err(SerializationError::InvalidData(
                "erased commitment hint block metadata mismatch".to_string(),
            ));
        }
        if !self.decomposed_inner_rows.is_empty()
            && !self.decomposed_inner_rows.len().is_multiple_of(self.ring_dim)
        {
            return Err(SerializationError::InvalidData(
                "erased commitment hint decomposed digit length is not a multiple of ring_dim"
                    .to_string(),
            ));
        }
        if !self.recomposed_inner_row_coeffs.is_empty()
            && !self
                .recomposed_inner_row_coeffs
                .len()
                .is_multiple_of(self.ring_dim)
        {
            return Err(SerializationError::InvalidData(
                "erased commitment hint recomposed coefficient length is not a multiple of ring_dim"
                    .to_string(),
            ));
        }
        Ok(())
    }
}
