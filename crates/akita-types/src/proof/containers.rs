use super::*;
use akita_error::checked;

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

impl<F: Field> RingVec<F> {
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

    /// Construct owned ring storage with an explicit runtime dimension.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] when `ring_dim` is zero or the
    /// coefficient buffer does not contain an integral number of ring values.
    pub fn from_coeffs_with_ring_dim(coeffs: Vec<F>, ring_dim: usize) -> Result<Self, AkitaError> {
        if ring_dim == 0 || !coeffs.len().is_multiple_of(ring_dim) {
            return Err(AkitaError::InvalidInput(
                "ring coefficient storage does not match its declared dimension".into(),
            ));
        }
        Ok(Self { coeffs, ring_dim })
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

    /// Consume the container and return its coefficient buffer.
    pub fn into_coeffs(self) -> Vec<F> {
        self.coeffs
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
    fn as_ring_slice_trusted<const D: usize>(&self) -> &[CyclotomicRing<F, D>] {
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
    fn as_single_ring_trusted<const D: usize>(&self) -> &CyclotomicRing<F, D> {
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
        if D == 0 {
            return Err(AkitaError::InvalidProof);
        }
        if self.ring_dim == D && self.coeffs.len() == D {
            return Ok(self.as_single_ring_trusted::<D>());
        }
        let rings = self.as_ring_slice::<D>()?;
        match rings {
            [ring] => Ok(ring),
            _ => Err(AkitaError::InvalidProof),
        }
    }
}

impl<F: Field + AkitaSerialize> AkitaSerialize for RingVec<F> {
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

impl<F: Field + Valid> Valid for RingVec<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for RingVec<F> {
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
}

impl<F: Field> RingVec<F> {
    /// Borrow this `RingVec` as a [`RingView`] using the stored `ring_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if `ring_dim == 0` (compact mode)
    /// or if `coeffs.len()` is not a multiple of `ring_dim`. Compact-mode
    /// vectors must build a [`RingView`] with an explicit ring dimension.
    pub fn view(&self) -> Result<RingView<'_, F>, AkitaError> {
        RingView::new(&self.coeffs, self.ring_dim)
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
fn total_planes(block_sizes: &[usize]) -> Result<usize, AkitaError> {
    checked::sum(block_sizes.iter().copied())
        .ok_or_else(|| AkitaError::InvalidInput("digit block size overflow".to_string()))
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
        let total_planes = total_planes(&block_sizes)?;
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
        let total_planes = total_planes(&block_sizes)?;
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

    /// Borrow the digit slice for plane index `plane`
    /// (`digit_stride` digits), or `None` if out of range.
    pub fn plane(&self, plane: usize) -> Option<&[i8]> {
        let start = plane.checked_mul(self.digit_stride)?;
        let end = start.checked_add(self.digit_stride)?;
        self.digits.get(start..end)
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

    /// Consume into the flat digits, block sizes, and per-plane stride.
    pub fn into_parts(self) -> (Vec<i8>, Vec<usize>, usize) {
        (self.digits, self.block_sizes, self.digit_stride)
    }

    /// Validate that the per-plane stride matches the const-generic kernel `D`.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidInput`] on stride mismatch.
    pub fn ensure_stride<const D: usize>(&self) -> Result<(), AkitaError> {
        if self.digit_stride != D {
            return Err(AkitaError::InvalidInput(format!(
                "digit blocks stride {} does not match requested D={D}",
                self.digit_stride
            )));
        }
        Ok(())
    }

    /// Borrow digit planes as `&[[i8; D]]` after [`Self::ensure_stride`].
    ///
    /// # Errors
    ///
    /// Returns an error on stride mismatch or if the flat stream is not an
    /// exact multiple of `D`.
    pub fn typed_planes<const D: usize>(&self) -> Result<&[[i8; D]], AkitaError> {
        self.ensure_stride::<D>()?;
        let (planes, remainder) = self.digits.as_chunks::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: self.digits.len() - remainder.len() + D,
                actual: self.digits.len(),
            });
        }
        Ok(planes)
    }

    /// Mutably borrow digit planes as `&mut [[i8; D]]` after [`Self::ensure_stride`].
    ///
    /// # Errors
    ///
    /// Returns an error on stride mismatch or if the flat stream is not an
    /// exact multiple of `D`.
    pub fn typed_planes_mut<const D: usize>(&mut self) -> Result<&mut [[i8; D]], AkitaError> {
        self.ensure_stride::<D>()?;
        let digits_len = self.digits.len();
        let (planes, remainder) = self.digits.as_chunks_mut::<D>();
        if !remainder.is_empty() {
            return Err(AkitaError::InvalidSize {
                expected: digits_len - remainder.len() + D,
                actual: digits_len,
            });
        }
        Ok(planes)
    }

    /// Split mutable digit planes into per-block `&mut [[i8; D]]` slices.
    ///
    /// # Errors
    ///
    /// Returns an error on stride mismatch or malformed block boundaries.
    pub fn split_typed_blocks_mut<const D: usize>(
        &mut self,
    ) -> Result<Vec<&mut [[i8; D]]>, AkitaError> {
        self.ensure_stride::<D>()?;
        let block_sizes = self.block_sizes.clone();
        let mut blocks = Vec::with_capacity(block_sizes.len());
        let planes = self.typed_planes_mut::<D>()?;
        let mut tail = planes;
        for block_size in block_sizes {
            let (head, rest) = tail
                .split_at_mut_checked(block_size)
                .ok_or_else(|| AkitaError::InvalidInput("digit block boundary overflow".into()))?;
            blocks.push(head);
            tail = rest;
        }
        if !tail.is_empty() {
            return Err(AkitaError::InvalidInput(
                "digit block sizes do not cover all planes".into(),
            ));
        }
        Ok(blocks)
    }
}

impl Valid for DigitBlocks {
    fn check(&self) -> Result<(), SerializationError> {
        if self.digit_stride == 0 {
            return Err(SerializationError::InvalidData(
                "digit blocks require a non-zero digit stride".to_string(),
            ));
        }
        let total_planes = checked::sum(self.block_sizes.iter().copied()).ok_or_else(|| {
            SerializationError::InvalidData("digit block size overflow".to_string())
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
        let total_planes = checked::sum(block_sizes.iter().copied()).ok_or_else(|| {
            SerializationError::InvalidData("digit block size overflow".to_string())
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
