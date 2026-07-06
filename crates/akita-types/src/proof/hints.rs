use super::*;

/// Prover-side hint for one opening-point commitment bundle.
///
/// Stores per-polynomial decomposed inner rows for all polynomials bundled into
/// the single commitment at one opening point.
///
/// # S4: D-free split
///
/// This type is now D-free (`AkitaCommitmentHint<F>`). It holds only the
/// protocol-adjacent, serialized part — `decomposed_inner_rows: Vec<DigitBlocks>`.
/// The former prover-only `recomposed_inner_rows:
/// Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>` field was D-typed and cannot
/// live in a D-free struct, so it was removed here. S5 reintroduces a private
/// prover-side home for recomposed rows (recomposition can be redone from the
/// decomposed digit stream via `CyclotomicRing::gadget_recompose_pow2_i8`).
#[derive(Debug, Clone)]
pub struct AkitaCommitmentHint<F: FieldCore> {
    /// Per-polynomial digit decompositions of the inner `A * s_i` rows.
    pub decomposed_inner_rows: Vec<DigitBlocks>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore> AkitaCommitmentHint<F> {
    /// Construct a new batched hint from per-polynomial digit streams.
    pub fn new(decomposed_inner_rows: Vec<DigitBlocks>) -> Self {
        Self {
            decomposed_inner_rows,
            _marker: PhantomData,
        }
    }

    /// Construct a singleton batched hint from one decomposed digit stream.
    pub fn singleton(decomposed_inner_rows: DigitBlocks) -> Self {
        Self::new(vec![decomposed_inner_rows])
    }

    /// Borrow the per-polynomial decomposed inner rows.
    pub fn decomposed_inner_rows(&self) -> &[DigitBlocks] {
        &self.decomposed_inner_rows
    }

    /// Consume the hint and return the per-polynomial digit rows.
    pub fn into_parts(self) -> Vec<DigitBlocks> {
        self.decomposed_inner_rows
    }

    /// Flatten the batched hint into the ring-switch view over all claims.
    ///
    /// All component digit streams must share the same per-plane stride; the
    /// flattened result inherits it.
    ///
    /// # Errors
    ///
    /// Returns an error if the component digit streams have mismatched strides,
    /// the hint is empty, or the flattened digit planes do not match the
    /// concatenated block-size metadata.
    pub fn into_flat_parts(self) -> Result<DigitBlocks, AkitaError> {
        let digit_stride = self
            .decomposed_inner_rows
            .first()
            .map(DigitBlocks::digit_stride)
            .ok_or_else(|| {
                AkitaError::InvalidInput(
                    "cannot flatten an empty commitment hint into ring-switch view".to_string(),
                )
            })?;
        let mut block_sizes = Vec::new();
        let total_digits: usize = self
            .decomposed_inner_rows
            .iter()
            .map(|digits| digits.digits().len())
            .sum();
        let mut flat_digits = Vec::with_capacity(total_digits);
        for digits in &self.decomposed_inner_rows {
            if digits.digit_stride() != digit_stride {
                return Err(AkitaError::InvalidInput(
                    "commitment hint components have mismatched digit strides".to_string(),
                ));
            }
            block_sizes.extend_from_slice(digits.block_sizes());
            digits.extend_digits(&mut flat_digits);
        }
        DigitBlocks::new(flat_digits, block_sizes, digit_stride)
    }
}

impl<F: FieldCore> PartialEq for AkitaCommitmentHint<F> {
    fn eq(&self, other: &Self) -> bool {
        self.decomposed_inner_rows == other.decomposed_inner_rows
    }
}

impl<F: FieldCore> Eq for AkitaCommitmentHint<F> {}

impl<F: FieldCore + Valid> Valid for AkitaCommitmentHint<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.decomposed_inner_rows.check()
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for AkitaCommitmentHint<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.decomposed_inner_rows
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.decomposed_inner_rows.serialized_size(compress)
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
        let decomposed_inner_rows =
            Vec::<DigitBlocks>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            decomposed_inner_rows,
            _marker: PhantomData,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}
