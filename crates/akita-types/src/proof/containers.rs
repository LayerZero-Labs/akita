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

    /// Borrow the stored coefficients as a slice of ring elements.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub fn as_ring_slice<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], AkitaError> {
        if D == 0
            || (self.ring_dim > 0 && self.ring_dim != D)
            || !self.coeffs.len().is_multiple_of(D)
        {
            return Err(AkitaError::InvalidProof);
        }
        let ring_count = self.coeffs.len() / D;
        // SAFETY: `CyclotomicRing<F, D>` is `#[repr(transparent)]` over
        // `[F; D]`, so a contiguous coefficient buffer with length divisible by
        // `D` can be borrowed as contiguous ring elements.
        Ok(unsafe {
            std::slice::from_raw_parts(
                self.coeffs.as_ptr() as *const CyclotomicRing<F, D>,
                ring_count,
            )
        })
    }

    /// Borrow the stored coefficients as a single typed ring element.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`, or if it contains more than one
    /// element.
    pub fn as_single_ring<const D: usize>(&self) -> Result<&CyclotomicRing<F, D>, AkitaError> {
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

    /// Append the stored coefficients using the typed ring-vector transcript
    /// encoding (alias for [`Self::append_as_ring_commitment`]).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub fn append_as_ring_slice<T: Transcript<F>, const D: usize>(
        &self,
        label: &[u8],
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: CanonicalField,
    {
        self.append_as_ring_commitment::<T, D>(label, transcript)
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatDigitBlocks<const D: usize> {
    flat_digits: Vec<[i8; D]>,
    block_sizes: Vec<usize>,
}

/// Iterator over logical blocks inside [`FlatDigitBlocks`].
pub struct FlatDigitBlockIter<'a, const D: usize> {
    flat_digits: &'a [[i8; D]],
    block_sizes: &'a [usize],
    offset: usize,
}

impl<const D: usize> FlatDigitBlocks<D> {
    /// Construct an empty digit-block collection.
    pub fn empty() -> Self {
        Self {
            flat_digits: Vec::new(),
            block_sizes: Vec::new(),
        }
    }

    /// Construct zero-initialized flat digits for explicit block sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes overflow the total flat length.
    pub fn zeroed(block_sizes: Vec<usize>) -> Result<Self, AkitaError> {
        let total_planes = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                AkitaError::InvalidInput("flat digit block size overflow".to_string())
            })
        })?;
        Ok(Self {
            flat_digits: vec![[0i8; D]; total_planes],
            block_sizes,
        })
    }

    /// Construct from flat digits and explicit block sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes do not sum to the flat digit count.
    pub fn new(flat_digits: Vec<[i8; D]>, block_sizes: Vec<usize>) -> Result<Self, AkitaError> {
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
            flat_digits,
            block_sizes,
        })
    }

    /// Flatten a block-owned representation into canonical storage.
    pub fn from_blocks(blocks: Vec<Vec<[i8; D]>>) -> Self {
        let block_sizes: Vec<usize> = blocks.iter().map(Vec::len).collect();
        let total_planes: usize = block_sizes.iter().sum();
        let mut flat_digits = Vec::with_capacity(total_planes);
        for block in blocks {
            flat_digits.extend(block);
        }
        Self {
            flat_digits,
            block_sizes,
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

    /// Flat digit stream in column-major block order.
    pub fn flat_digits(&self) -> &[[i8; D]] {
        &self.flat_digits
    }

    /// Mutable flat digit stream in column-major block order.
    pub fn flat_digits_mut(&mut self) -> &mut [[i8; D]] {
        &mut self.flat_digits
    }

    /// Split the flat digit stream into disjoint mutable block slices.
    pub fn split_blocks_mut(&mut self) -> Vec<&mut [[i8; D]]> {
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut tail = self.flat_digits.as_mut_slice();
        for &block_size in &self.block_sizes {
            let (head, rest) = tail.split_at_mut(block_size);
            blocks.push(head);
            tail = rest;
        }
        blocks
    }

    /// Iterate over blocks as slices into the flat digit stream.
    pub fn iter_blocks(&self) -> FlatDigitBlockIter<'_, D> {
        FlatDigitBlockIter {
            flat_digits: &self.flat_digits,
            block_sizes: &self.block_sizes,
            offset: 0,
        }
    }

    /// Iterate over logical blocks.
    pub fn iter(&self) -> FlatDigitBlockIter<'_, D> {
        self.iter_blocks()
    }

    /// Append the flat digit stream to `dst`.
    pub fn extend_flat_digits(&self, dst: &mut Vec<[i8; D]>) {
        dst.extend_from_slice(&self.flat_digits);
    }

    /// Truncate every block to at most `block_len` digit planes.
    pub fn truncate_each_block(&mut self, block_len: usize) {
        if self.block_sizes.iter().all(|&size| size <= block_len) {
            return;
        }

        let total_planes: usize = self
            .block_sizes
            .iter()
            .map(|&size| size.min(block_len))
            .sum();
        let mut new_flat = Vec::with_capacity(total_planes);
        let mut offset = 0usize;
        for size in &mut self.block_sizes {
            let keep = (*size).min(block_len);
            new_flat.extend_from_slice(&self.flat_digits[offset..offset + keep]);
            offset += *size;
            *size = keep;
        }
        self.flat_digits = new_flat;
    }

    /// Consume the storage and rebuild owned blocks.
    pub fn into_blocks(self) -> Vec<Vec<[i8; D]>> {
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut offset = 0usize;
        for size in self.block_sizes {
            blocks.push(self.flat_digits[offset..offset + size].to_vec());
            offset += size;
        }
        blocks
    }

    /// Consume into the flat digits and block sizes.
    pub fn into_parts(self) -> (Vec<[i8; D]>, Vec<usize>) {
        (self.flat_digits, self.block_sizes)
    }
}

impl<const D: usize> Valid for FlatDigitBlocks<D> {
    fn check(&self) -> Result<(), SerializationError> {
        if D == 0 {
            return Err(SerializationError::InvalidData(
                "flat digit blocks require a non-zero ring dimension".to_string(),
            ));
        }
        let expected = self.block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                SerializationError::InvalidData("flat digit block size overflow".to_string())
            })
        })?;
        if expected != self.flat_digits.len() {
            return Err(SerializationError::InvalidData(format!(
                "flat digit block sizes sum to {expected}, but digit stream has {} planes",
                self.flat_digits.len()
            )));
        }
        Ok(())
    }
}

impl<const D: usize> AkitaSerialize for FlatDigitBlocks<D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.block_sizes
            .serialize_with_mode(&mut writer, compress)?;
        for plane in &self.flat_digits {
            for digit in plane {
                digit.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.block_sizes.serialized_size(compress) + self.flat_digits.len() * D
    }
}

impl<const D: usize> AkitaDeserialize for FlatDigitBlocks<D> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let block_sizes =
            Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let total_planes = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                SerializationError::InvalidData("flat digit block size overflow".to_string())
            })
        })?;
        let mut flat_digits = Vec::new();
        super::reserve_shape_len(&mut flat_digits, total_planes)?;
        for _ in 0..total_planes {
            let mut plane = [0i8; D];
            for digit in &mut plane {
                *digit = i8::deserialize_with_mode(&mut reader, compress, validate, &())?;
            }
            flat_digits.push(plane);
        }
        let out = Self {
            flat_digits,
            block_sizes,
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

/// ZK plain-opening hiding-factor commitment and payload.
///
/// `hiding_witness` contains all one-time pads used by the masked opening
/// protocol. Deferred relations are written directly over these slots and
/// public masked transcript values.
#[cfg(feature = "zk")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ZkHidingProof<F: FieldCore> {
    /// Wire-visible hiding-factor commitment `u_blind`.
    pub u_blind: Vec<F>,
    /// Plain opening of the committed hiding witness.
    pub hiding_witness: Vec<F>,
    /// Dedicated short Ajtai blinding digits used for `u_blind`.
    pub b_blinding_digits: Vec<i8>,
}

#[cfg(feature = "zk")]
impl<F: FieldCore> ZkHidingProof<F> {
    /// True when this proof carries no top-level hiding commitment or opening.
    pub fn is_empty(&self) -> bool {
        self.u_blind.is_empty()
            && self.hiding_witness.is_empty()
            && self.b_blinding_digits.is_empty()
    }
}

#[cfg(feature = "zk")]
impl<F: FieldCore + AkitaSerialize> AkitaSerialize for ZkHidingProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.u_blind.serialize_with_mode(&mut writer, compress)?;
        self.hiding_witness
            .serialize_with_mode(&mut writer, compress)?;
        self.b_blinding_digits
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.u_blind.serialized_size(compress)
            + self.hiding_witness.serialized_size(compress)
            + self.b_blinding_digits.serialized_size(compress)
    }
}

#[cfg(feature = "zk")]
impl<F: FieldCore + Valid> Valid for ZkHidingProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.u_blind.check()?;
        self.hiding_witness.check()?;
        self.b_blinding_digits.check()
    }
}

#[cfg(feature = "zk")]
impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for ZkHidingProof<F> {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        (): &(),
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            u_blind: Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?,
            hiding_witness: Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?,
            b_blinding_digits: Vec::<i8>::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?,
        })
    }
}
