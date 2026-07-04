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
