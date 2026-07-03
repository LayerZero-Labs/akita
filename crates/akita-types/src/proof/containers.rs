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
pub struct RingVec<F> {
    coeffs: Vec<F>,
    ring_dim: usize,
}

/// D-free serializer for a flat coefficient buffer without a length header.
///
/// Serializes each field element in order using `serialize_with_mode`. This is
/// the canonical protocol encoding for ring-shaped data: the caller supplies
/// `ring_dim` at the call site rather than baking it into a const generic. The
/// S2 byte-identity test (`flat_absorption_byte_identical_to_typed`) proves this
/// matches the bytes the deleted typed `RingSliceSerializer` produced.
///
/// Used by [`append_flat_coefficients`] and [`RingVec::append_flat_to_transcript`].
pub struct FlatCoeffSerializer<'a, F: FieldCore>(pub &'a [F]);

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for FlatCoeffSerializer<'_, F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        for coeff in self.0 {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.iter().map(|c| c.serialized_size(compress)).sum()
    }
}

/// Absorb a flat coefficient buffer into `transcript` using the canonical
/// flat-coefficient encoding (the S4 protocol transcript encoding for
/// ring-shaped commitments).
///
/// The buffer must contain `n_rings * ring_dim` coefficients in coefficient-major
/// order (ring element 0 first, coefficients 0..ring_dim−1 within each element).
/// This function derives the total absorption length from `ring_dim` at runtime,
/// not from a const generic `D`.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] if `ring_dim == 0` or if
/// `coeffs.len()` is not a multiple of `ring_dim`.
pub fn append_flat_coefficients<F, T>(
    label: &[u8],
    coeffs: &[F],
    ring_dim: usize,
    transcript: &mut T,
) -> Result<(), AkitaError>
where
    F: FieldCore + AkitaSerialize + CanonicalField,
    T: Transcript<F>,
{
    if ring_dim == 0 || !coeffs.len().is_multiple_of(ring_dim) {
        return Err(AkitaError::InvalidProof);
    }
    transcript.append_serde(label, &FlatCoeffSerializer(coeffs));
    Ok(())
}

impl<F: FieldCore> RingVec<F> {
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

    /// Absorb the stored coefficients into `transcript` using the D-free flat
    /// encoding (the canonical S4 transcript encoding for ring-shaped data).
    ///
    /// `ring_dim` is the schedule-derived ring dimension.  When `self.ring_dim`
    /// is non-zero the supplied value must match it (mismatches are detected and
    /// returned as `InvalidProof` rather than panicked).
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if `ring_dim == 0`, if
    /// `coeffs.len()` is not a multiple of `ring_dim`, or (when stored
    /// `ring_dim > 0`) if the stored and supplied `ring_dim` disagree.
    pub fn append_flat_to_transcript<T: Transcript<F>>(
        &self,
        label: &[u8],
        ring_dim: usize,
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: AkitaSerialize + CanonicalField,
    {
        if self.ring_dim > 0 && self.ring_dim != ring_dim {
            return Err(AkitaError::InvalidProof);
        }
        append_flat_coefficients(label, &self.coeffs, ring_dim, transcript)
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for RingVec<F> {
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

impl<F: FieldCore + Valid> Valid for RingVec<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for RingVec<F> {
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

/// A borrowed, schedule-shaped view over a flat coefficient slice.
///
/// `RingView<'a, F>` pairs a coefficient slice from a [`RingVec`] (or any
/// contiguous field-element buffer) with an explicit `ring_dim` that comes
/// from the runtime schedule rather than a compile-time const. This is the
/// canonical borrowed accessor for ring-shaped protocol data: use it wherever
/// a callee needs to interpret a flat coefficient buffer under a known
/// schedule-derived ring dimension without taking ownership.
///
/// # Invariant
///
/// `ring_dim > 0` and `coeffs.len()` is a multiple of `ring_dim`.
/// The constructors enforce this; there is no way to build a `RingView` that
/// violates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingView<'a, F> {
    coeffs: &'a [F],
    ring_dim: usize,
}

impl<'a, F> RingView<'a, F> {
    /// Construct a `RingView` from a coefficient slice and a ring dimension.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if `ring_dim == 0` or if
    /// `coeffs.len()` is not a multiple of `ring_dim`.
    pub fn new(coeffs: &'a [F], ring_dim: usize) -> Result<Self, AkitaError> {
        if ring_dim == 0 || !coeffs.len().is_multiple_of(ring_dim) {
            return Err(AkitaError::InvalidProof);
        }
        Ok(Self { coeffs, ring_dim })
    }

    /// The ring dimension (number of field-element coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// The number of ring elements in this view.
    pub fn num_rings(&self) -> usize {
        self.coeffs.len() / self.ring_dim
    }

    /// Alias for [`num_rings`](Self::num_rings).
    pub fn count(&self) -> usize {
        self.num_rings()
    }

    /// The flat coefficient slice.
    pub fn coeffs(&self) -> &[F] {
        self.coeffs
    }

    /// Absorb this view's coefficients into `transcript` using the D-free flat
    /// encoding, byte-identical to the typed
    /// [`Commitment::append_to_transcript`] path.
    ///
    /// The `ring_dim` stored in this view is used directly; no external dimension
    /// is needed since `RingView` always carries a valid, non-zero `ring_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if flat absorption fails (should not
    /// occur when [`RingView::new`] invariants hold).
    pub fn append_flat_to_transcript<T>(
        &self,
        label: &[u8],
        transcript: &mut T,
    ) -> Result<(), AkitaError>
    where
        F: FieldCore + AkitaSerialize + CanonicalField,
        T: Transcript<F>,
    {
        append_flat_coefficients(label, self.coeffs, self.ring_dim, transcript)
    }
}

impl<F: FieldCore> RingVec<F> {
    /// Borrow this `RingVec` as a [`RingView`] using the stored `ring_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if `ring_dim == 0` (compact mode)
    /// or if `coeffs.len()` is not a multiple of `ring_dim`. Compact-mode
    /// vectors must use [`view_as`](Self::view_as) instead.
    pub fn view(&self) -> Result<RingView<'_, F>, AkitaError> {
        RingView::new(&self.coeffs, self.ring_dim)
    }

    /// Borrow this `RingVec` as a [`RingView`] under an externally supplied
    /// `ring_dim` (e.g. from the schedule). Use this for compact-mode vectors
    /// where `ring_dim` was not stored.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if `ring_dim == 0` or if
    /// `coeffs.len()` is not a multiple of `ring_dim`.
    pub fn view_as(&self, ring_dim: usize) -> Result<RingView<'_, F>, AkitaError> {
        RingView::new(&self.coeffs, ring_dim)
    }
}

/// Runtime digit-plane storage plus explicit block boundaries.
///
/// Replaces the former `FlatDigitBlocks<const D>`. Each digit plane is a row of
/// `digit_stride` signed digits (where `digit_stride` was the const generic `D`
/// ring dimension), stored flat in `digits` in plane-major order. `block_sizes`
/// gives the per-block plane count.
///
/// # Invariant
///
/// `(sum of block_sizes) * digit_stride == digits.len()`. This replaces the
/// compile-time guarantee that every plane was exactly `[i8; D]` wide. The
/// constructors and [`Valid::check`] enforce it at runtime; violations return an
/// [`AkitaError`] / [`SerializationError`] rather than panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DigitBlocks {
    /// Flat signed-digit stream, plane-major: `total_planes * digit_stride` digits.
    digits: Vec<i8>,
    /// Per-block plane counts.
    block_sizes: Vec<usize>,
    /// Number of signed digits per plane (the former ring dimension `D`).
    digit_stride: usize,
}

/// Iterator over logical blocks inside [`DigitBlocks`], yielding the flat digit
/// slice for each block (`block_size * digit_stride` digits).
pub struct DigitBlockIter<'a> {
    digits: &'a [i8],
    block_sizes: &'a [usize],
    digit_stride: usize,
    offset_planes: usize,
}

/// Sum block sizes with overflow checking.
fn checked_total_planes(block_sizes: &[usize]) -> Result<usize, AkitaError> {
    block_sizes.iter().try_fold(0usize, |acc, &size| {
        acc.checked_add(size)
            .ok_or_else(|| AkitaError::InvalidInput("digit block size overflow".to_string()))
    })
}

impl DigitBlocks {
    /// Construct an empty digit-block collection with the given per-plane stride.
    pub fn empty(digit_stride: usize) -> Self {
        Self {
            digits: Vec::new(),
            block_sizes: Vec::new(),
            digit_stride,
        }
    }

    /// Construct zero-initialized digits for explicit block sizes at the given
    /// per-plane stride.
    ///
    /// # Errors
    ///
    /// Returns an error if the block sizes overflow the total plane count or if
    /// the resulting digit length overflows.
    pub fn zeroed(block_sizes: Vec<usize>, digit_stride: usize) -> Result<Self, AkitaError> {
        let total_planes = checked_total_planes(&block_sizes)?;
        let total_digits = total_planes
            .checked_mul(digit_stride)
            .ok_or_else(|| AkitaError::InvalidInput("digit block length overflow".to_string()))?;
        Ok(Self {
            digits: vec![0i8; total_digits],
            block_sizes,
            digit_stride,
        })
    }

    /// Construct from a flat digit stream, explicit block sizes, and per-plane
    /// stride.
    ///
    /// # Errors
    ///
    /// Returns an error if `(sum of block_sizes) * digit_stride` does not equal
    /// `digits.len()`.
    pub fn new(
        digits: Vec<i8>,
        block_sizes: Vec<usize>,
        digit_stride: usize,
    ) -> Result<Self, AkitaError> {
        let total_planes = checked_total_planes(&block_sizes)?;
        let expected = total_planes
            .checked_mul(digit_stride)
            .ok_or_else(|| AkitaError::InvalidInput("digit block length overflow".to_string()))?;
        if expected != digits.len() {
            return Err(AkitaError::InvalidSize {
                expected,
                actual: digits.len(),
            });
        }
        Ok(Self {
            digits,
            block_sizes,
            digit_stride,
        })
    }

    /// Flatten a block-owned plane representation into canonical storage at the
    /// given per-plane stride.
    ///
    /// Every plane must have exactly `digit_stride` digits.
    ///
    /// # Errors
    ///
    /// Returns an error if any plane width differs from `digit_stride`.
    pub fn from_blocks(blocks: Vec<Vec<Vec<i8>>>, digit_stride: usize) -> Result<Self, AkitaError> {
        let block_sizes: Vec<usize> = blocks.iter().map(Vec::len).collect();
        let total_planes = checked_total_planes(&block_sizes)?;
        let total_digits = total_planes
            .checked_mul(digit_stride)
            .ok_or_else(|| AkitaError::InvalidInput("digit block length overflow".to_string()))?;
        let mut digits = Vec::with_capacity(total_digits);
        for block in blocks {
            for plane in block {
                if plane.len() != digit_stride {
                    return Err(AkitaError::InvalidSize {
                        expected: digit_stride,
                        actual: plane.len(),
                    });
                }
                digits.extend_from_slice(&plane);
            }
        }
        Ok(Self {
            digits,
            block_sizes,
            digit_stride,
        })
    }

    /// Number of signed digits per plane (the former ring dimension `D`).
    pub fn digit_stride(&self) -> usize {
        self.digit_stride
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

    /// Total number of digit planes across all blocks.
    pub fn total_planes(&self) -> usize {
        self.block_sizes.iter().sum()
    }

    /// Per-block digit-plane counts.
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Flat digit stream in plane-major block order.
    pub fn digits(&self) -> &[i8] {
        &self.digits
    }

    /// Mutable flat digit stream in plane-major block order.
    pub fn digits_mut(&mut self) -> &mut [i8] {
        &mut self.digits
    }

    /// Borrow the digit slice for plane index `plane`
    /// (`digit_stride` digits), or `None` if out of range.
    pub fn plane(&self, plane: usize) -> Option<&[i8]> {
        let start = plane.checked_mul(self.digit_stride)?;
        let end = start.checked_add(self.digit_stride)?;
        self.digits.get(start..end)
    }

    /// Split the flat digit stream into disjoint mutable per-block slices
    /// (each `block_size * digit_stride` digits long).
    pub fn split_blocks_mut(&mut self) -> Vec<&mut [i8]> {
        let stride = self.digit_stride;
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut tail = self.digits.as_mut_slice();
        for &block_size in &self.block_sizes {
            let (head, rest) = tail.split_at_mut(block_size * stride);
            blocks.push(head);
            tail = rest;
        }
        blocks
    }

    /// Iterate over blocks as flat digit slices into the digit stream.
    pub fn iter_blocks(&self) -> DigitBlockIter<'_> {
        DigitBlockIter {
            digits: &self.digits,
            block_sizes: &self.block_sizes,
            digit_stride: self.digit_stride,
            offset_planes: 0,
        }
    }

    /// Iterate over logical blocks.
    pub fn iter(&self) -> DigitBlockIter<'_> {
        self.iter_blocks()
    }

    /// Append the flat digit stream to `dst`.
    pub fn extend_digits(&self, dst: &mut Vec<i8>) {
        dst.extend_from_slice(&self.digits);
    }

    /// Truncate every block to at most `block_len` digit planes.
    pub fn truncate_each_block(&mut self, block_len: usize) {
        if self.block_sizes.iter().all(|&size| size <= block_len) {
            return;
        }
        let stride = self.digit_stride;
        let total_planes: usize = self
            .block_sizes
            .iter()
            .map(|&size| size.min(block_len))
            .sum();
        let mut new_digits = Vec::with_capacity(total_planes * stride);
        let mut offset_planes = 0usize;
        for size in &mut self.block_sizes {
            let keep = (*size).min(block_len);
            let start = offset_planes * stride;
            new_digits.extend_from_slice(&self.digits[start..start + keep * stride]);
            offset_planes += *size;
            *size = keep;
        }
        self.digits = new_digits;
    }

    /// Consume the storage and rebuild owned blocks of planes (each plane a
    /// `Vec<i8>` of length `digit_stride`).
    pub fn into_blocks(self) -> Vec<Vec<Vec<i8>>> {
        let stride = self.digit_stride;
        let mut blocks = Vec::with_capacity(self.block_sizes.len());
        let mut offset_planes = 0usize;
        for size in self.block_sizes {
            let mut block = Vec::with_capacity(size);
            for plane in 0..size {
                let start = (offset_planes + plane) * stride;
                block.push(self.digits[start..start + stride].to_vec());
            }
            blocks.push(block);
            offset_planes += size;
        }
        blocks
    }

    /// Consume into the flat digits, block sizes, and per-plane stride.
    pub fn into_parts(self) -> (Vec<i8>, Vec<usize>, usize) {
        (self.digits, self.block_sizes, self.digit_stride)
    }
}

impl Valid for DigitBlocks {
    fn check(&self) -> Result<(), SerializationError> {
        if self.digit_stride == 0 {
            return Err(SerializationError::InvalidData(
                "digit blocks require a non-zero digit stride".to_string(),
            ));
        }
        let total_planes = self.block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                SerializationError::InvalidData("digit block size overflow".to_string())
            })
        })?;
        let expected = total_planes.checked_mul(self.digit_stride).ok_or_else(|| {
            SerializationError::InvalidData("digit block length overflow".to_string())
        })?;
        if expected != self.digits.len() {
            return Err(SerializationError::InvalidData(format!(
                "digit blocks: {total_planes} planes * stride {} = {expected}, but digit \
                 stream has {} digits",
                self.digit_stride,
                self.digits.len()
            )));
        }
        Ok(())
    }
}

impl AkitaSerialize for DigitBlocks {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        // Explicit headers replace the const-`D` inference removed with the
        // generic: the per-plane stride is no longer known from the type, so it
        // is written first, followed by the block-size table and the flat
        // digit stream.
        self.digit_stride
            .serialize_with_mode(&mut writer, compress)?;
        self.block_sizes
            .serialize_with_mode(&mut writer, compress)?;
        for digit in &self.digits {
            digit.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.digit_stride.serialized_size(compress)
            + self.block_sizes.serialized_size(compress)
            + self.digits.len()
    }
}

impl AkitaDeserialize for DigitBlocks {
    type Context = ();

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let digit_stride = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let block_sizes =
            Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let total_planes = block_sizes.iter().try_fold(0usize, |acc, &size| {
            acc.checked_add(size).ok_or_else(|| {
                SerializationError::InvalidData("digit block size overflow".to_string())
            })
        })?;
        let total_digits = total_planes.checked_mul(digit_stride).ok_or_else(|| {
            SerializationError::InvalidData("digit block length overflow".to_string())
        })?;
        let mut digits = Vec::new();
        super::reserve_shape_len(&mut digits, total_digits)?;
        for _ in 0..total_digits {
            digits.push(i8::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let out = Self {
            digits,
            block_sizes,
            digit_stride,
        };
        if validate == Validate::Yes {
            out.check()?;
        }
        Ok(out)
    }
}

impl<'a> Iterator for DigitBlockIter<'a> {
    type Item = &'a [i8];

    fn next(&mut self) -> Option<Self::Item> {
        let size = *self.block_sizes.first()?;
        let start = self.offset_planes * self.digit_stride;
        let end = start + size * self.digit_stride;
        self.offset_planes += size;
        self.block_sizes = &self.block_sizes[1..];
        Some(&self.digits[start..end])
    }
}
