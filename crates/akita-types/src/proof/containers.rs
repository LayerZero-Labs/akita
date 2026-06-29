use super::*;

/// D-erased storage for a sequence of ring elements as raw field-element
/// coefficients.
///
/// Each ring element of dimension `ring_dim` is stored as `ring_dim`
/// contiguous field elements in `coeffs`. The total number of ring elements
/// is `coeffs.len() / ring_dim`.
///
/// When `ring_dim` is 0 the vector is in "compact" mode: the ring dimension
/// is not known to this container and must be supplied externally (e.g. from
/// the public schedule). This is the mode used inside serialised proofs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatRingVec<F> {
    coeffs: Vec<F>,
    ring_dim: usize,
}

/// In-memory owned ring-element storage without a tagged ring dimension.
///
/// Wire encoding remains [`FlatRingVec`]; use [`FlatRingVec::as_ring_slice`] at
/// protocol boundaries where `D` is known from the schedule.
pub type RingBuf<F> = FlatRingVec<F>;

/// Serializer for a borrowed slice of ring elements without a length header.
pub struct RingSliceSerializer<'a, F: FieldCore, const D: usize>(pub &'a [CyclotomicRing<F, D>]);

impl<F: FieldCore + AkitaSerialize, const D: usize> AkitaSerialize
    for RingSliceSerializer<'_, F, D>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for ring in self.0 {
            ring.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0
            .iter()
            .map(|ring| ring.serialized_size(compress))
            .sum()
    }
}

impl<F: FieldCore> FlatRingVec<F> {
    /// Wrap a single ring element.
    pub fn from_single<const D: usize>(r: &CyclotomicRing<F, D>) -> Self {
        Self {
            coeffs: r.coefficients().to_vec(),
            ring_dim: D,
        }
    }

    /// Wrap a slice of ring elements.
    pub fn from_ring_elems<const D: usize>(elems: &[CyclotomicRing<F, D>]) -> Self {
        let mut coeffs = Vec::with_capacity(elems.len() * D);
        for e in elems {
            coeffs.extend_from_slice(e.coefficients());
        }
        Self {
            coeffs,
            ring_dim: D,
        }
    }

    /// Construct from raw field coefficients in compact mode (`ring_dim = 0`).
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self {
            coeffs,
            ring_dim: 0,
        }
    }

    /// Wrap a `RingCommitment`.
    pub fn from_commitment<const D: usize>(c: &RingCommitment<F, D>) -> Self {
        Self::from_ring_elems(&c.u)
    }

    /// Ring dimension (number of field-element coefficients per ring element),
    /// or 0 if the container is in compact mode.
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Number of ring elements stored.
    ///
    /// Returns 0 when `ring_dim` is unknown (compact mode).
    pub fn count(&self) -> usize {
        self.coeffs.len().checked_div(self.ring_dim).unwrap_or(0)
    }

    /// Raw coefficient slice.
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Number of stored field coefficients.
    pub fn coeff_len(&self) -> usize {
        self.coeffs.len()
    }

    /// Whether these coefficients can be decoded as a single ring element of
    /// dimension `d`.
    pub fn can_decode_single(&self, d: usize) -> bool {
        d != 0 && self.coeffs.len() == d
    }

    /// Whether these coefficients can be decoded as a vector of ring elements
    /// of dimension `d`.
    pub fn can_decode_vec(&self, d: usize) -> bool {
        if d == 0 {
            return false;
        }
        self.coeffs.len().is_multiple_of(d)
    }

    /// Return a copy with `ring_dim` cleared (compact mode).
    pub fn into_compact(self) -> Self {
        Self {
            coeffs: self.coeffs,
            ring_dim: 0,
        }
    }

    /// Reconstruct a single ring element.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim` (when ring_dim is known) or
    /// `coeffs.len() != D`.
    pub fn to_single<const D: usize>(&self) -> CyclotomicRing<F, D> {
        if self.ring_dim > 0 {
            assert_eq!(D, self.ring_dim, "D mismatch in to_single");
        }
        assert_eq!(
            self.coeffs.len(),
            D,
            "expected exactly one ring element of dimension {D}"
        );
        CyclotomicRing::from_slice(&self.coeffs)
    }

    /// Reconstruct a single ring element, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring dimension or
    /// element count does not match `D`.
    pub fn try_to_single<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, AkitaError> {
        if D == 0 || (self.ring_dim > 0 && self.ring_dim != D) || self.coeffs.len() != D {
            return Err(AkitaError::InvalidProof);
        }
        Ok(CyclotomicRing::from_slice(&self.coeffs))
    }

    /// Reconstruct a vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim` (when ring_dim is known) or
    /// `coeffs.len()` is not a multiple of `D`.
    pub fn to_vec<const D: usize>(&self) -> Vec<CyclotomicRing<F, D>> {
        if self.ring_dim > 0 {
            assert_eq!(D, self.ring_dim, "D mismatch in to_vec");
        }
        assert_eq!(
            self.coeffs.len() % D,
            0,
            "coeff count not a multiple of D={D}"
        );
        self.coeffs
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect()
    }

    /// Reconstruct a vector of ring elements, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring dimension does
    /// not match `D` or the coefficient buffer is not an exact multiple of `D`.
    pub fn try_to_vec<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        if D == 0
            || (self.ring_dim > 0 && self.ring_dim != D)
            || !self.coeffs.len().is_multiple_of(D)
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(self
            .coeffs
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect())
    }

    /// Hot-path borrow after construction or schedule dispatch has fixed `D`.
    ///
    /// Debug-asserts `ring_dim == D` (or compact mode with divisible length).
    /// Release builds perform no shape checks.
    #[inline]
    pub fn as_ring_slice_trusted<const D: usize>(&self) -> &[CyclotomicRing<F, D>] {
        debug_assert!(D > 0);
        debug_assert!(self.ring_dim == 0 || self.ring_dim == D);
        debug_assert!(self.coeffs.len().is_multiple_of(D));
        let ring_count = self.coeffs.len() / D;
        // SAFETY: `CyclotomicRing<F, D>` is `#[repr(transparent)]` over `[F; D]`.
        unsafe {
            std::slice::from_raw_parts(
                self.coeffs.as_ptr() as *const CyclotomicRing<F, D>,
                ring_count,
            )
        }
    }

    /// Hot-path borrow of a single ring element after construction fixed `D`.
    #[inline]
    pub fn as_single_ring_trusted<const D: usize>(&self) -> &CyclotomicRing<F, D> {
        debug_assert_eq!(self.coeffs.len(), D);
        debug_assert!(self.ring_dim == 0 || self.ring_dim == D);
        // SAFETY: one `D`-sized coefficient block is one ring element.
        unsafe { &*(self.coeffs.as_ptr() as *const CyclotomicRing<F, D>) }
    }

    /// Borrow the stored coefficients as a slice of ring elements.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    #[inline]
    pub fn as_ring_slice<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        if D == 0
            || (self.ring_dim > 0 && self.ring_dim != D)
            || !self.coeffs.len().is_multiple_of(D)
        {
            return Err(AkitaError::InvalidProof);
        }
        Ok(self.as_ring_slice_trusted::<D>())
    }

    /// Borrow the stored coefficients as a single typed ring element.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`, or if it contains more than one
    /// element.
    pub fn as_single_ring<const D: usize>(&self) -> Result<&CyclotomicRing<F, D>, AkitaError> {
        if self.ring_dim == D && self.coeffs.len() == D {
            return Ok(self.as_single_ring_trusted::<D>());
        }
        let rings = self.as_ring_slice::<D>()?;
        match rings {
            [ring] => Ok(ring),
            _ => Err(AkitaError::InvalidProof),
        }
    }

    /// Append the stored coefficients using the same transcript encoding as a
    /// typed [`RingCommitment`].
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub fn append_as_ring_commitment<T: Transcript<F>, const D: usize>(
        &self,
        label: &[u8],
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField,
    {
        let rings = self.as_ring_slice::<D>()?;
        transcript.append_serde(label, &RingSliceSerializer(rings));
        Ok(())
    }

    /// Reconstruct a `RingCommitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim` (when ring_dim is known).
    pub fn to_ring_commitment<const D: usize>(&self) -> RingCommitment<F, D> {
        RingCommitment { u: self.to_vec() }
    }

    /// Reconstruct a `RingCommitment`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub fn try_to_ring_commitment<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, AkitaError> {
        Ok(RingCommitment {
            u: self.try_to_vec()?,
        })
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for FlatRingVec<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for c in &self.coeffs {
            c.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.coeffs
            .iter()
            .map(|c| c.serialized_size(compress))
            .sum()
    }
}

impl<F: FieldCore + Valid> Valid for FlatRingVec<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for FlatRingVec<F> {
    /// Number of field-element coefficients to read.
    type Context = usize;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        num_coeffs: &usize,
    ) -> Result<Self, SerializationError> {
        let mut coeffs = Vec::new();
        reserve_shape_len(&mut coeffs, *num_coeffs)?;
        for _ in 0..*num_coeffs {
            coeffs.push(F::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self {
            coeffs,
            ring_dim: 0,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Flat digit-plane storage plus explicit block boundaries.
///
/// Ring dimension is stored at runtime; hot paths inside `dispatch_ring_dim`
/// closures borrow typed digit planes via [`Self::flat_digits_trusted`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatDigitBlocks {
    /// Digit planes stored row-major (`D` coefficients per plane).
    digits: Vec<i8>,
    block_sizes: Vec<usize>,
    ring_dim: usize,
}

/// Iterator over logical blocks inside [`FlatDigitBlocks`].
pub struct FlatDigitBlockIter<'a, const D: usize> {
    flat_digits: &'a [[i8; D]],
    block_sizes: &'a [usize],
    offset: usize,
}

impl FlatDigitBlocks {
    /// Construct an empty digit-block collection (ring dimension unset).
    pub fn empty() -> Self {
        Self {
            digits: Vec::new(),
            block_sizes: Vec::new(),
            ring_dim: 0,
        }
    }

    /// Stored ring dimension (coefficients per digit plane).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Number of digit planes in the flat stream.
    pub fn plane_count(&self) -> usize {
        self.digits
            .len()
            .checked_div(self.ring_dim.max(1))
            .unwrap_or(0)
    }

    /// # Errors
    ///
    /// Returns an error if the requested ring dimension does not match storage.
    pub fn ensure_ring_dim<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.ring_dim != D {
            return Err(AkitaError::InvalidInput(format!(
                "flat digit blocks ring_d={} does not match requested D={D}",
                self.ring_dim
            )));
        }
        if !self.digits.len().is_multiple_of(D) {
            return Err(AkitaError::InvalidSize {
                expected: D,
                actual: self.digits.len(),
            });
        }
        let expected_planes: usize = self.block_sizes.iter().sum();
        if self.digits.len() / D != expected_planes {
            return Err(AkitaError::InvalidInput(
                "flat digit block plane count mismatch".to_string(),
            ));
        }
        Ok(())
    }

    /// Construct zero-initialized flat digits for explicit block sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes overflow the total flat length.
    pub fn zeroed<const D: usize>(block_sizes: Vec<usize>) -> Result<Self, AkitaError> {
        let total_planes = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                AkitaError::InvalidInput("flat digit block size overflow".to_string())
            })
        })?;
        Ok(Self {
            digits: vec![0i8; total_planes.saturating_mul(D)],
            block_sizes,
            ring_dim: D,
        })
    }

    /// Construct from typed digit planes at a kernel boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes do not sum to the flat digit count.
    pub fn from_planes<const D: usize>(
        flat_digits: Vec<[i8; D]>,
        block_sizes: Vec<usize>,
    ) -> Result<Self, AkitaError> {
        let expected = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                AkitaError::InvalidInput("flat digit block size overflow".to_string())
            })
        })?;
        if expected != flat_digits.len() {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: flat_digits.len(),
            });
        }
        Ok(Self {
            digits: flat_digits
                .iter()
                .flat_map(|plane| plane.iter().copied())
                .collect(),
            block_sizes,
            ring_dim: D,
        })
    }

    /// Flatten a block-owned representation into canonical storage.
    pub fn from_blocks<const D: usize>(blocks: Vec<Vec<[i8; D]>>) -> Self {
        let block_sizes: Vec<usize> = blocks.iter().map(Vec::len).collect();
        let total_planes: usize = block_sizes.iter().sum();
        let mut digits = Vec::with_capacity(total_planes.saturating_mul(D));
        for block in blocks {
            for plane in block {
                digits.extend_from_slice(&plane);
            }
        }
        Self {
            digits,
            block_sizes,
            ring_dim: D,
        }
    }

    /// Number of logical blocks.
    pub fn block_count(&self) -> usize {
        self.block_sizes.len()
    }

    /// Number of logical blocks.
    pub fn len(&self) -> usize {
        self.block_count()
    }

    /// Whether there are no logical blocks.
    pub fn is_empty(&self) -> bool {
        self.block_sizes.is_empty()
    }

    /// Per-block digit-plane counts.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Borrow typed digit planes after [`Self::ensure_ring_dim`].
    pub fn flat_digits_trusted<const D: usize>(&self) -> &[[i8; D]] {
        debug_assert_eq!(self.ring_dim, D);
        let (chunks, rem) = self.digits.as_chunks::<D>();
        debug_assert!(rem.is_empty());
        chunks
    }

    /// Mutable typed digit planes after [`Self::ensure_ring_dim`].
    pub fn flat_digits_trusted_mut<const D: usize>(&mut self) -> &mut [[i8; D]] {
        debug_assert_eq!(self.ring_dim, D);
        let (chunks, rem) = self.digits.as_chunks_mut::<D>();
        debug_assert!(rem.is_empty());
        chunks
    }

    /// Split the flat digit stream into disjoint mutable block slices.
    pub fn split_blocks_mut<const D: usize>(&mut self) -> Vec<&mut [[i8; D]]> {
        debug_assert_eq!(self.ring_dim, D);
        let (planes, _) = self.digits.as_chunks_mut::<D>();
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut tail = planes;
        for &block_size in &self.block_sizes {
            let (head, rest) = tail.split_at_mut(block_size);
            blocks.push(head);
            tail = rest;
        }
        blocks
    }

    /// Iterate over blocks as slices into the flat digit stream.
    pub fn iter_blocks<const D: usize>(&self) -> FlatDigitBlockIter<'_, D> {
        FlatDigitBlockIter {
            flat_digits: self.flat_digits_trusted::<D>(),
            block_sizes: &self.block_sizes,
            offset: 0,
        }
    }

    /// Iterate over logical blocks.
    pub fn iter<const D: usize>(&self) -> FlatDigitBlockIter<'_, D> {
        self.iter_blocks::<D>()
    }

    /// Append the flat digit stream to `dst`.
    pub fn extend_flat_digits<const D: usize>(&self, dst: &mut Vec<[i8; D]>) {
        dst.extend_from_slice(self.flat_digits_trusted::<D>());
    }

    /// Truncate every block to at most `block_len` digit planes.
    pub fn truncate_each_block(&mut self, block_len: usize) {
        if self.ring_dim == 0 || self.block_sizes.iter().all(|&size| size <= block_len) {
            return;
        }

        let d = self.ring_dim;
        let total_planes: usize = self
            .block_sizes
            .iter()
            .map(|&size| size.min(block_len))
            .sum();
        let mut new_digits = Vec::with_capacity(total_planes.saturating_mul(d));
        let mut plane_idx = 0usize;
        for size in &mut self.block_sizes {
            let keep = (*size).min(block_len);
            for _ in 0..keep {
                let start = plane_idx * d;
                new_digits.extend_from_slice(&self.digits[start..start + d]);
                plane_idx += 1;
            }
            plane_idx += *size - keep;
            *size = keep;
        }
        self.digits = new_digits;
    }

    /// Consume the storage and rebuild owned blocks.
    pub fn into_blocks<const D: usize>(self) -> Vec<Vec<[i8; D]>> {
        debug_assert_eq!(self.ring_dim, D);
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let planes = self.flat_digits_trusted::<D>();
        let mut offset = 0usize;
        for &size in &self.block_sizes {
            blocks.push(planes[offset..offset + size].to_vec());
            offset += size;
        }
        blocks
    }

    /// Consume into typed digit planes and block sizes.
    pub fn into_planes<const D: usize>(self) -> (Vec<[i8; D]>, Vec<usize>) {
        debug_assert_eq!(self.ring_dim, D);
        let (chunks, rem) = self.digits.as_chunks::<D>();
        debug_assert!(rem.is_empty());
        (chunks.to_vec(), self.block_sizes)
    }
}

impl Valid for FlatDigitBlocks {
    fn check(&self) -> Result<(), SerializationError> {
        if self.ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "flat digit blocks require a non-zero ring dimension".to_string(),
            ));
        }
        let expected = self.block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                SerializationError::InvalidData("flat digit block size overflow".to_string())
            })
        })?;
        let plane_count = self
            .digits
            .len()
            .checked_div(self.ring_dim)
            .ok_or_else(|| {
                SerializationError::InvalidData(
                    "flat digit block digit length is not divisible by ring_dim".to_string(),
                )
            })?;
        if expected != plane_count {
            return Err(SerializationError::InvalidData(format!(
                "flat digit block sizes sum to {expected}, but digit stream has {plane_count} planes",
            )));
        }
        Ok(())
    }
}

impl AkitaSerialize for FlatDigitBlocks {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        if self.ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "cannot serialize flat digit blocks without ring_dim".to_string(),
            ));
        }
        self.block_sizes
            .serialize_with_mode(&mut writer, compress)?;
        for digit in &self.digits {
            digit.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.block_sizes.serialized_size(compress) + self.digits.len()
    }
}

impl FlatDigitBlocks {
    /// Deserialize digit blocks for a schedule-known ring dimension.
    pub fn deserialize_typed<const D: usize, R: Read>(
        reader: &mut R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let block_sizes =
            Vec::<usize>::deserialize_with_mode(&mut *reader, compress, validate, &())?;
        let total_planes = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                SerializationError::InvalidData("flat digit block size overflow".to_string())
            })
        })?;
        let mut digits = Vec::new();
        super::reserve_shape_len(&mut digits, total_planes.saturating_mul(D))?;
        for _ in 0..total_planes {
            for _ in 0..D {
                digits.push(i8::deserialize_with_mode(
                    &mut *reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
        }
        let out = Self {
            digits,
            block_sizes,
            ring_dim: D,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

impl<'a, const D: usize> Iterator for FlatDigitBlockIter<'a, D> {
    type Item = &'a [[i8; D]];

    fn next(&mut self) -> Option<Self::Item> {
        let size = *self.block_sizes.first()?;
        let start = self.offset;
        let end = start + size;
        self.offset = end;
        self.block_sizes = &self.block_sizes[1..];
        Some(&self.flat_digits[start..end])
    }
}
