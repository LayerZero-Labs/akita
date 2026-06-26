use super::*;

/// Prover-side hint for one opening-point commitment bundle.
///
/// Stores per-polynomial decomposed inner rows and, when available, the
/// corresponding recomposed inner rows for all polynomials bundled into the
/// single commitment at one opening point.
#[derive(Debug, Clone)]
pub struct AkitaCommitmentHint<F: FieldCore, const D: usize> {
    /// Per-polynomial digit decompositions of the inner `A * s_i` rows.
    pub decomposed_inner_rows: Vec<FlatDigitBlocks>,
    /// Optional recomposed inner rows grouped by polynomial then block.
    recomposed_inner_rows: Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> AkitaCommitmentHint<F, D> {
    /// Construct a new batched hint from per-polynomial digit streams.
    pub fn new(decomposed_inner_rows: Vec<FlatDigitBlocks>) -> Self {
        Self {
            decomposed_inner_rows,
            recomposed_inner_rows: None,
            _marker: PhantomData,
        }
    }

    /// Construct a singleton batched hint from one polynomial's digit stream.
    pub fn singleton(decomposed_inner_rows: FlatDigitBlocks) -> Self {
        Self::new(vec![decomposed_inner_rows])
    }

    /// Construct a batched hint that also preserves recomposed inner rows.
    pub fn with_recomposed_inner_rows(
        decomposed_inner_rows: Vec<FlatDigitBlocks>,
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
        decomposed_inner_rows: FlatDigitBlocks,
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
        Vec<FlatDigitBlocks>,
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
                    .iter_blocks::<D>()
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
    pub fn into_flat_parts(self) -> (FlatDigitBlocks, Option<Vec<Vec<CyclotomicRing<F, D>>>>) {
        let mut block_sizes = Vec::new();
        let total_planes: usize = self
            .decomposed_inner_rows
            .iter()
            .map(|digits| digits.plane_count())
            .sum();
        let mut flat_digits = Vec::with_capacity(total_planes);
        for digits in &self.decomposed_inner_rows {
            block_sizes.extend_from_slice(digits.block_sizes());
            digits.extend_flat_digits::<D>(&mut flat_digits);
        }
        let decomposed_inner_rows = FlatDigitBlocks::from_planes::<D>(flat_digits, block_sizes)
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
        for row in &self.decomposed_inner_rows {
            row.check()?;
        }
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
        let decomposed_len =
            u64::deserialize_with_mode(&mut reader, compress, validate, &())? as usize;
        let mut scratch = Vec::<FlatDigitBlocks>::new();
        super::reserve_shape_len(&mut scratch, decomposed_len)?;
        let mut decomposed_inner_rows = Vec::with_capacity(decomposed_len);
        for _ in 0..decomposed_len {
            decomposed_inner_rows.push(FlatDigitBlocks::deserialize_typed::<D, R>(
                &mut reader,
                compress,
                validate,
            )?);
        }
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
    decomposed_digits: FlatDigitBlocks,
    recomposed_inner_row_coeffs: Vec<F>,
    recomposed_inner_row_block_sizes: Vec<usize>,
}

impl<F: FieldCore> ErasedCommitmentHint<F> {
    /// Flatten a typed prover hint for D-free slot storage.
    pub fn from_typed<const D: usize>(hint: AkitaCommitmentHint<F, D>) -> Self {
        let (decomposed_digits, recomposed_inner_rows) = hint.into_flat_parts();
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
            decomposed_digits,
            recomposed_inner_row_coeffs,
            recomposed_inner_row_block_sizes,
        }
    }

    /// Flatten a typed recursive-commitment hint that must carry recomposed inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed hint does not carry recomposed inner rows.
    pub fn from_typed_recursive<const D: usize>(
        hint: AkitaCommitmentHint<F, D>,
    ) -> Result<Self, AkitaError> {
        let erased = Self::from_typed(hint);
        if erased.recomposed_inner_row_coeffs.is_empty() {
            return Err(AkitaError::InvalidInput(
                "missing recomposed inner rows in recursive commitment hint".to_string(),
            ));
        }
        Ok(erased)
    }

    /// Reconstruct the typed prover hint without recomputing inner rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match the stored hint,
    /// or if flattened block metadata is inconsistent.
    pub fn to_typed<const D: usize>(&self) -> Result<AkitaCommitmentHint<F, D>, AkitaError> {
        self.decomposed_digits.ensure_ring_dim::<D>()?;
        if self.decomposed_digits.block_sizes().len() != self.recomposed_inner_row_block_sizes.len()
        {
            return Err(AkitaError::InvalidInput(
                "erased commitment hint block metadata mismatch".to_string(),
            ));
        }

        let (flat_recomposed_rows, recomposed_remainder) =
            self.recomposed_inner_row_coeffs.as_chunks::<D>();
        if !recomposed_remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.recomposed_inner_row_coeffs.len(),
            });
        }

        let mut recomposed_offset = 0usize;
        let mut recomposed_inner_rows =
            Vec::with_capacity(self.recomposed_inner_row_block_sizes.len());

        for &recomposed_block_size in &self.recomposed_inner_row_block_sizes {
            let recomposed_end = recomposed_offset + recomposed_block_size;
            if recomposed_end > flat_recomposed_rows.len() {
                return Err(AkitaError::InvalidInput(
                    "erased commitment hint block data is truncated".to_string(),
                ));
            }
            recomposed_inner_rows.push(
                flat_recomposed_rows[recomposed_offset..recomposed_end]
                    .iter()
                    .map(|coeffs| CyclotomicRing::from_coefficients(*coeffs))
                    .collect(),
            );
            recomposed_offset = recomposed_end;
        }

        if recomposed_offset != flat_recomposed_rows.len() {
            return Err(AkitaError::InvalidInput(
                "erased commitment hint has trailing block data".to_string(),
            ));
        }

        Ok(AkitaCommitmentHint::singleton_with_recomposed_inner_rows(
            self.decomposed_digits.clone(),
            recomposed_inner_rows,
        ))
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
                    "erased commitment hint decomposed block data is truncated".to_string(),
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
                "erased commitment hint has trailing decomposed block data".to_string(),
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
                    "erased commitment hint recomposed block data is truncated".to_string(),
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
                "erased commitment hint has trailing recomposed block data".to_string(),
            ));
        }
        Ok(rings)
    }

    #[must_use]
    pub fn ring_dim(&self) -> usize {
        self.decomposed_digits.ring_dim()
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
        if self.decomposed_digits.ring_dim() != ring_dim {
            return Err(SerializationError::InvalidData(format!(
                "erased commitment hint ring_d={} does not match slot id d_setup={ring_dim}",
                self.decomposed_digits.ring_dim()
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
        if self.decomposed_digits.ring_dim() != ring_dim {
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
        self.decomposed_digits.check()?;
        if self.decomposed_digits.block_sizes().len() != self.recomposed_inner_row_block_sizes.len()
        {
            return Err(SerializationError::InvalidData(
                "erased commitment hint block metadata mismatch".to_string(),
            ));
        }
        let ring_dim = self.decomposed_digits.ring_dim();
        if !self.recomposed_inner_row_coeffs.is_empty()
            && !self
                .recomposed_inner_row_coeffs
                .len()
                .is_multiple_of(ring_dim)
        {
            return Err(SerializationError::InvalidData(
                "erased commitment hint recomposed coefficient length is not a multiple of ring_dim"
                    .to_string(),
            ));
        }
        Ok(())
    }
}
