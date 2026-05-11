//! Proof structures for the Akita protocol.

//! Proof, commitment, setup, and claim data shapes.

pub mod batch;
pub mod commitment;
pub mod incidence;
pub mod relation;
pub mod scheme;
pub mod setup;
pub mod stage1;

pub use batch::{
    append_batched_commitments_to_transcript, append_claim_points_to_transcript,
    append_claim_values_to_transcript, append_prepared_root_opening_point, checked_total_claims,
    checked_total_groups, flatten_batched_commitment_rows, folded_root_supports_opening_shape,
    prepare_recursive_opening_point_ext, prepare_root_opening_point,
    prepare_root_opening_point_ext, ring_inner_product_with_extension_weights,
    validate_batched_inputs, PreparedRecursiveOpeningPoint, PreparedRootOpeningPoint,
    RingMultiplierOpeningPoint,
};
pub use commitment::{AkitaCommitment, DummyProof, RingCommitment};
pub use incidence::{
    append_claim_incidence_shape_to_transcript, verifier_claims_to_incidence, ClaimIncidence,
    ClaimIncidenceLimits, ClaimIncidenceSummary, CommitmentGroupOccurrence, IncidenceClaim,
};
pub use relation::{
    relation_claim_from_batched_root_rows_extension, relation_claim_from_rows,
    relation_claim_from_rows_extension,
};
pub use scheme::{CommitmentVerifier, CommittedOpenings, OpeningPoints, VerifierClaims};
pub use setup::{AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, PublicMatrixSeed};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};

use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt};
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
use akita_sumcheck::{
    EqFactoredSumcheckProof, EqFactoredSumcheckProofShape, SumcheckProof, SumcheckProofShape,
};
use akita_transcript::Transcript;
use std::io::{Read, Write};
use std::marker::PhantomData;

/// Bit-packed balanced digits for the final-level witness vector.
///
/// Each element is a signed value in `[-b/2, b/2)` where `b = 2^bits_per_elem`,
/// stored in two's-complement using exactly `bits_per_elem` bits per value.
/// This reduces proof size by ~32x compared to storing full field elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedDigits {
    /// Number of logical elements.
    pub num_elems: usize,
    /// Bits per element used for packing.
    pub bits_per_elem: u32,
    /// Bit-packed two's-complement data.
    pub data: Vec<u8>,
}

/// Terminal direct witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectWitnessProof<F: FieldCore> {
    /// Packed small signed digits, used by the current recursive terminal
    /// witness.
    PackedDigits(PackedDigits),
    /// Raw field elements, for direct witnesses that are not naturally digit
    /// bounded.
    FieldElements(FlatRingVec<F>),
}

impl<F: FieldCore> DirectWitnessProof<F> {
    /// Borrow the packed-digits payload, if present.
    pub fn as_packed_digits(&self) -> Option<&PackedDigits> {
        match self {
            Self::PackedDigits(packed) => Some(packed),
            Self::FieldElements(_) => None,
        }
    }

    /// Borrow the raw field-element payload, if present.
    pub fn as_field_elements(&self) -> Option<&FlatRingVec<F>> {
        match self {
            Self::PackedDigits(_) => None,
            Self::FieldElements(field_elems) => Some(field_elems),
        }
    }

    /// Shape descriptor for this direct witness payload.
    pub fn shape(&self) -> DirectWitnessShape {
        match self {
            Self::PackedDigits(packed) => {
                DirectWitnessShape::PackedDigits((packed.num_elems, packed.bits_per_elem))
            }
            Self::FieldElements(field_elems) => {
                DirectWitnessShape::FieldElements(field_elems.coeff_len())
            }
        }
    }

    /// Number of logical field elements carried by this witness payload.
    pub fn num_elems(&self) -> usize {
        match self {
            Self::PackedDigits(packed) => packed.num_elems,
            Self::FieldElements(field_elems) => field_elems.coeff_len(),
        }
    }
}

/// Shape descriptor for deserializing a direct witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectWitnessShape {
    /// Packed balanced digits.
    PackedDigits((usize, u32)),
    /// Raw field elements.
    FieldElements(usize),
}

impl PackedDigits {
    /// Smallest `bits_per_elem` that can encode every signed digit in `w`.
    pub fn required_bits_per_elem(w: &[i8]) -> u32 {
        let required_half_b = w.iter().fold(1i16, |acc, &signed| {
            let needed = if signed >= 0 {
                signed as i16 + 1
            } else {
                -(signed as i16)
            };
            acc.max(needed)
        });

        let mut bits = 1u32;
        let mut half_b = 1i16;
        while half_b < required_half_b {
            bits += 1;
            half_b <<= 1;
        }
        bits
    }

    /// Pack balanced i8 digits into bit-packed form.
    ///
    /// Each element must be in `[-b/2, b/2)` where `b = 2^log_basis`.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if any element does not fit in `log_basis` bits.
    pub fn from_i8_digits(w: &[i8], log_basis: u32) -> Self {
        assert!(log_basis > 0 && log_basis <= 6, "log_basis out of range");
        let half_b = 1i8 << (log_basis - 1);

        let bits = log_basis as usize;
        let total_bits = w.len() * bits;
        let num_bytes = total_bits.div_ceil(8);
        let mut data = vec![0u8; num_bytes];

        for (i, &signed) in w.iter().enumerate() {
            debug_assert!(
                signed >= -half_b && signed < half_b,
                "digit {signed} out of range for log_basis={log_basis}"
            );
            let unsigned = (signed as u8) & ((1u8 << bits) - 1);
            let bit_offset = i * bits;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            data[byte_idx] |= unsigned << bit_idx;
            if bit_idx + bits > 8 {
                data[byte_idx + 1] |= unsigned >> (8 - bit_idx);
            }
        }

        Self {
            num_elems: w.len(),
            bits_per_elem: log_basis,
            data,
        }
    }

    /// Pack digits using at least `min_bits_per_elem`, widening if needed so
    /// every element in `w` fits the chosen two's-complement range.
    pub fn from_i8_digits_with_min_bits(w: &[i8], min_bits_per_elem: u32) -> Self {
        let bits_per_elem = min_bits_per_elem.max(Self::required_bits_per_elem(w));
        Self::from_i8_digits(w, bits_per_elem)
    }

    /// Decode a single packed signed digit.
    ///
    /// # Panics
    ///
    /// Panics if the packed byte buffer is malformed relative to
    /// `num_elems`/`bits_per_elem`. Valid instances produced by
    /// [`PackedDigits::from_i8_digits`] or checked during deserialization are
    /// well-formed.
    pub fn digit_at(&self, idx: usize) -> Option<i8> {
        if idx >= self.num_elems {
            return None;
        }

        let bits = self.bits_per_elem as usize;
        let mask = (1u8 << bits) - 1;
        let sign_bit = 1u8 << (bits - 1);
        let bit_offset = idx * bits;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let mut raw = (self.data[byte_idx] >> bit_idx) & mask;
        if bit_idx + bits > 8 {
            raw |= (self.data[byte_idx + 1] << (8 - bit_idx)) & mask;
        }

        Some(if raw & sign_bit != 0 {
            raw as i8 | !(mask as i8)
        } else {
            raw as i8
        })
    }

    /// Unpack to field elements.
    ///
    /// # Panics
    ///
    /// Panics if the packed byte buffer is malformed relative to
    /// `num_elems`/`bits_per_elem`. Valid instances produced by
    /// [`PackedDigits::from_i8_digits`] or checked during deserialization are
    /// well-formed.
    pub fn to_field_elems<F: FieldCore + FromPrimitiveInt>(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(self.num_elems);
        for i in 0..self.num_elems {
            let signed = self
                .digit_at(i)
                .expect("PackedDigits::to_field_elems index in bounds");
            out.push(F::from_i64(signed as i64));
        }
        out
    }

    /// Number of packed data bytes.
    pub fn packed_byte_len(&self) -> usize {
        self.data.len()
    }
}

impl AkitaSerialize for PackedDigits {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer.write_all(&self.data)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        self.data.len()
    }
}

impl Valid for PackedDigits {
    fn check(&self) -> Result<(), SerializationError> {
        if self.bits_per_elem == 0 || self.bits_per_elem > 6 {
            return Err(SerializationError::InvalidData(
                "bits_per_elem out of range".to_string(),
            ));
        }
        let expected_bytes = (self.num_elems * self.bits_per_elem as usize).div_ceil(8);
        if self.data.len() != expected_bytes {
            return Err(SerializationError::InvalidData(
                "packed data length mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaDeserialize for PackedDigits {
    /// `(num_elems, bits_per_elem)` — shape of the packed digit vector.
    type Context = (usize, u32);
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
        ctx: &(usize, u32),
    ) -> Result<Self, SerializationError> {
        let (num_elems, bits_per_elem) = *ctx;
        let num_bytes = (num_elems * bits_per_elem as usize).div_ceil(8);
        let mut data = vec![0u8; num_bytes];
        reader.read_exact(&mut data)?;
        let out = Self {
            num_elems,
            bits_per_elem,
            data,
        };
        out.check()?;
        Ok(out)
    }
}

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
        if self.ring_dim == 0 {
            0
        } else {
            self.coeffs.len() / self.ring_dim
        }
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
        self.coeffs.len() == d
    }

    /// Whether these coefficients can be decoded as a vector of ring elements
    /// of dimension `d`.
    pub fn can_decode_vec(&self, d: usize) -> bool {
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
        if (self.ring_dim > 0 && self.ring_dim != D) || self.coeffs.len() != D {
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
        if (self.ring_dim > 0 && self.ring_dim != D) || !self.coeffs.len().is_multiple_of(D) {
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
        if (self.ring_dim > 0 && self.ring_dim != D) || !self.coeffs.len().is_multiple_of(D) {
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
        let mut coeffs = Vec::with_capacity(*num_coeffs);
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

/// Prover-side hint for one same-point commitment group.
///
/// Stores per-polynomial decomposed inner rows and, when available, the
/// corresponding recomposed inner rows for all claims that were aggregated into
/// the same commitment.
#[derive(Debug, Clone)]
pub struct AkitaCommitmentHint<F: FieldCore, const D: usize> {
    /// Per-polynomial digit decompositions of the inner `A * s_i` rows.
    pub decomposed_inner_rows: Vec<FlatDigitBlocks<D>>,
    /// Per-commitment fresh B-blinding digit streams.
    #[cfg(feature = "zk")]
    b_blinding_digits: Vec<FlatDigitBlocks<D>>,
    /// Optional recomposed inner rows grouped by polynomial then block.
    recomposed_inner_rows: Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> AkitaCommitmentHint<F, D> {
    /// Construct a new batched hint from per-polynomial digit streams.
    #[cfg(not(feature = "zk"))]
    pub fn new(decomposed_inner_rows: Vec<FlatDigitBlocks<D>>) -> Self {
        Self {
            decomposed_inner_rows,
            recomposed_inner_rows: None,
            _marker: PhantomData,
        }
    }

    /// Construct a singleton batched hint from one polynomial's digit stream.
    #[cfg(not(feature = "zk"))]
    pub fn singleton(decomposed_inner_rows: FlatDigitBlocks<D>) -> Self {
        Self::new(vec![decomposed_inner_rows])
    }

    /// Construct a batched hint that also preserves recomposed inner rows.
    pub fn with_recomposed_inner_rows(
        decomposed_inner_rows: Vec<FlatDigitBlocks<D>>,
        recomposed_inner_rows: Vec<Vec<Vec<CyclotomicRing<F, D>>>>,
        #[cfg(feature = "zk")] b_blinding_digits: Vec<FlatDigitBlocks<D>>,
    ) -> Self {
        Self {
            decomposed_inner_rows,
            #[cfg(feature = "zk")]
            b_blinding_digits,
            recomposed_inner_rows: Some(recomposed_inner_rows),
            _marker: PhantomData,
        }
    }

    /// Construct a singleton batched hint that also preserves recomposed rows.
    pub fn singleton_with_recomposed_inner_rows(
        decomposed_inner_rows: FlatDigitBlocks<D>,
        recomposed_inner_rows: Vec<Vec<CyclotomicRing<F, D>>>,
        #[cfg(feature = "zk")] b_blinding_digits: FlatDigitBlocks<D>,
    ) -> Self {
        Self::with_recomposed_inner_rows(
            vec![decomposed_inner_rows],
            vec![recomposed_inner_rows],
            #[cfg(feature = "zk")]
            vec![b_blinding_digits],
        )
    }

    /// Get the optional recomposed inner rows grouped by polynomial.
    pub fn recomposed_inner_rows(&self) -> Option<&[Vec<Vec<CyclotomicRing<F, D>>>]> {
        self.recomposed_inner_rows.as_deref()
    }

    /// Get the B-blinding digit streams, one per commitment group.
    #[cfg(feature = "zk")]
    pub fn b_blinding_digits(&self) -> &[FlatDigitBlocks<D>] {
        &self.b_blinding_digits
    }

    /// Consume the hint and return per-polynomial digit rows plus optional
    /// recomposed inner rows, plus B-blinding digits when `zk` is enabled.
    #[allow(clippy::type_complexity)]
    #[cfg(not(feature = "zk"))]
    pub fn into_parts(
        self,
    ) -> (
        Vec<FlatDigitBlocks<D>>,
        Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
    ) {
        (self.decomposed_inner_rows, self.recomposed_inner_rows)
    }

    /// Consume the hint and return per-polynomial digit rows plus optional
    /// recomposed inner rows, plus B-blinding digits when `zk` is enabled.
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "zk")]
    pub fn into_parts(
        self,
    ) -> (
        Vec<FlatDigitBlocks<D>>,
        Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
        Vec<FlatDigitBlocks<D>>,
    ) {
        (
            self.decomposed_inner_rows,
            self.recomposed_inner_rows,
            self.b_blinding_digits,
        )
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
    /// Returns B-blinding digits as an additional part when `zk` is enabled.
    ///
    /// # Panics
    ///
    /// Panics if the flattened digit planes do not match the concatenated
    /// block-size metadata. This would indicate an internal bug, since the
    /// flattened view is derived directly from well-formed component hints.
    #[allow(clippy::type_complexity)]
    #[cfg(not(feature = "zk"))]
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

    /// Flatten the batched hint into the ring-switch view over all claims.
    ///
    /// Returns B-blinding digits as an additional part when `zk` is enabled.
    ///
    /// # Panics
    ///
    /// Panics if the flattened digit planes do not match the concatenated
    /// block-size metadata. This would indicate an internal bug, since the
    /// flattened view is derived directly from well-formed component hints.
    #[allow(clippy::type_complexity)]
    #[cfg(feature = "zk")]
    pub fn into_flat_parts(
        self,
    ) -> (
        FlatDigitBlocks<D>,
        Option<Vec<Vec<CyclotomicRing<F, D>>>>,
        Vec<FlatDigitBlocks<D>>,
    ) {
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
        (
            decomposed_inner_rows,
            recomposed_inner_rows,
            self.b_blinding_digits,
        )
    }
}

impl<F: FieldCore, const D: usize> PartialEq for AkitaCommitmentHint<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.decomposed_inner_rows == other.decomposed_inner_rows && {
            #[cfg(feature = "zk")]
            {
                self.b_blinding_digits == other.b_blinding_digits
            }
            #[cfg(not(feature = "zk"))]
            {
                true
            }
        }
    }
}

impl<F: FieldCore, const D: usize> Eq for AkitaCommitmentHint<F, D> {}
/// One stage in the stage-1 range-check tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1StageProof<F: FieldCore> {
    /// Eq-factored sumcheck proof for this stage.
    pub sumcheck: EqFactoredSumcheckProof<F>,
    /// Claimed child-node evaluations at this stage's output point.
    ///
    /// Non-leaf stages populate these so the verifier can seed the next stage;
    /// the leaf stage leaves this empty and instead carries `s_claim` below.
    pub child_claims: Vec<F>,
}

/// Headerless shape context for one stage in the stage-1 range-check tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkitaStage1StageShape {
    /// Eq-factored sumcheck shape `(num_rounds, q_degree)`.
    pub sumcheck: EqFactoredSumcheckProofShape,
    /// Number of child claims serialized after the stage proof.
    pub child_claims: usize,
}
/// Proof payload for stage 1 of a single Akita level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage1Proof<F: FieldCore> {
    /// Root-to-leaf range-check stages.
    pub stages: Vec<AkitaStage1StageProof<F>>,
    /// Claimed evaluation of `S` at the final stage-1 output point.
    pub s_claim: F,
}

/// Proof payload for stage 2 of a single Akita level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaStage2Proof<F: FieldCore, L: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck: SumcheckProof<L>,
    /// Commitment to the next witness `w`
    /// (ring dim = next level's D, may differ from y_ring/v).
    pub next_w_commitment: FlatRingVec<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    pub next_w_eval: L,
}

/// Proof for a single fold level (quad_eq + ring_switch + sumcheck).
///
/// D-agnostic: proof-owned ring vectors are stored in compact mode
/// (`ring_dim = 0`), and callers recover the typed ring dimension from the
/// surrounding proof shape or runtime context.
///
/// One recursive Akita level proof with inline stage-1 and stage-2 sumchecks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaLevelProof<F: FieldCore, L: FieldCore> {
    /// `y_ring` from the §3.1 reduction (ring dim = current level's D).
    pub y_ring: FlatRingVec<F>,
    /// `v = D · ŵ` (ring dim = current level's D).
    pub v: FlatRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, L>,
}

impl<F: FieldCore, L: FieldCore> AkitaLevelProof<F, L> {
    /// Construct from typed ring elements for the current level and its
    /// inline two-stage norm-check payloads.
    pub fn new<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2: AkitaStage2Proof<F, L>,
    ) -> Self {
        Self {
            y_ring: FlatRingVec::from_single(&y_ring).into_compact(),
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            stage1,
            stage2,
        }
    }

    /// Construct a level proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2_sumcheck: SumcheckProof<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            y_ring,
            v,
            stage1,
            AkitaStage2Proof {
                sumcheck: stage2_sumcheck,
                next_w_commitment: next_w_commitment.into_compact(),
                next_w_eval,
            },
        )
    }

    /// Ring dimension of y_ring and v (current level).
    pub fn level_d(&self) -> usize {
        self.y_ring.coeff_len()
    }

    /// Reconstruct typed `y_ring`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn y_ring_typed<const D: usize>(&self) -> CyclotomicRing<F, D> {
        self.y_ring.to_single()
    }

    /// Reconstruct typed `y_ring`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored `y_ring` does not
    /// encode exactly one ring element at dimension `D`.
    pub fn try_y_ring_typed<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, AkitaError> {
        self.y_ring.try_to_single()
    }

    /// Reconstruct typed `v`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn v_typed<const D: usize>(&self) -> Vec<CyclotomicRing<F, D>> {
        self.v.to_vec()
    }

    /// Reconstruct typed `v`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored `v` payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_v_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.v.try_to_vec()
    }

    /// Commitment to the next witness `w`.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self.stage2.next_w_commitment
    }

    /// Number of stored field coefficients for the next witness commitment.
    pub fn next_w_commitment_coeff_len(&self) -> usize {
        self.stage2.next_w_commitment.coeff_len()
    }

    /// Reconstruct typed `w_commitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn w_commitment_typed<const D: usize>(&self) -> RingCommitment<F, D> {
        RingCommitment {
            u: self.next_w_commitment().to_vec(),
        }
    }

    /// Reconstruct typed `w_commitment`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored next-level commitment
    /// is not well-formed for ring dimension `D`.
    pub fn try_w_commitment_typed<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, AkitaError> {
        Ok(RingCommitment {
            u: self.next_w_commitment().try_to_vec()?,
        })
    }

    /// Claimed evaluation of the next witness `w` at the norm-check output point.
    pub fn next_w_eval(&self) -> L {
        self.stage2.next_w_eval
    }

    /// Derive the [`LevelProofShape`] for this level proof.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(self.y_ring.coeff_len(), &self.v, &self.stage1, &self.stage2)
    }
}

/// Fused batched-root payload for the two-stage folding protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedFoldRoot<F: FieldCore, L: FieldCore> {
    /// Per-point batched public outputs `(y_j)_j`, stored as a flat ring vector.
    pub y_rings: FlatRingVec<F>,
    /// Aggregated `v = Σ_ell D_ell · w_hat_ell`.
    pub v: FlatRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, L>,
}

/// Root proof payload for fused batched openings.
///
/// Mirrors the enum shape of [`AkitaProofStep`]: when the offline schedule
/// for the batch is small enough to skip folding entirely (zero-fold root)
/// the prover sends the per-claim polynomial coefficients directly instead
/// of running the two-stage norm-check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedRootProof<F: FieldCore, L: FieldCore> {
    /// Standard two-stage folded root proof.
    Fold(AkitaBatchedFoldRoot<F, L>),
    /// Root-direct batched fast path: one direct field-element witness per
    /// claim, in the normalized incidence claim order used by the prover.
    Direct {
        /// Per-claim direct witnesses.
        witnesses: Vec<DirectWitnessProof<F>>,
        /// Per-commitment B-blinding digit streams revealed for verifier
        /// recommitment in the root-direct zk fast path.
        #[cfg(feature = "zk")]
        b_blinding_digits: Vec<Vec<i8>>,
    },
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedRootProof<F, L> {
    /// Construct from typed ring elements for the batched root level.
    pub fn new<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2: AkitaStage2Proof<F, L>,
    ) -> Self {
        Self::Fold(AkitaBatchedFoldRoot {
            y_rings: FlatRingVec::from_ring_elems(&y_rings).into_compact(),
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            stage1,
            stage2,
        })
    }

    /// Construct a batched root proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        stage2_sumcheck: SumcheckProof<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            y_rings,
            v,
            stage1,
            AkitaStage2Proof {
                sumcheck: stage2_sumcheck,
                next_w_commitment: next_w_commitment.into_compact(),
                next_w_eval,
            },
        )
    }

    /// Construct the root-direct batched variant with one witness per claim.
    #[cfg(not(feature = "zk"))]
    pub fn new_direct(witnesses: Vec<DirectWitnessProof<F>>) -> Self {
        Self::Direct { witnesses }
    }

    /// Construct the root-direct batched variant with one witness per claim and
    /// one revealed B-blinding payload per commitment group.
    #[cfg(feature = "zk")]
    pub fn new_direct(
        witnesses: Vec<DirectWitnessProof<F>>,
        b_blinding_digits: Vec<Vec<i8>>,
    ) -> Self {
        Self::Direct {
            witnesses,
            b_blinding_digits,
        }
    }

    /// Borrow the fold payload when this is a fold root.
    pub fn as_fold(&self) -> Option<&AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Direct { .. } => None,
        }
    }

    /// Mutably borrow the fold payload when this is a fold root.
    pub fn as_fold_mut(&mut self) -> Option<&mut AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Direct { .. } => None,
        }
    }

    /// Borrow the per-claim direct witnesses when this is a root-direct
    /// batched proof.
    pub fn as_direct(&self) -> Option<&[DirectWitnessProof<F>]> {
        match self {
            Self::Fold(_) => None,
            Self::Direct { witnesses, .. } => Some(witnesses.as_slice()),
        }
    }

    /// Borrow the revealed root-direct B-blinding payloads.
    #[cfg(feature = "zk")]
    pub fn direct_b_blinding_digits(&self) -> Option<&[Vec<i8>]> {
        match self {
            Self::Fold(_) => None,
            Self::Direct {
                b_blinding_digits, ..
            } => Some(b_blinding_digits.as_slice()),
        }
    }

    /// True when this root proof is a root-direct batched fast path.
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    /// Borrow the stored root per-point `y_rings` payload (Fold only).
    ///
    /// # Panics
    ///
    /// Panics when called on a root-direct batched proof.
    pub fn y_rings(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("y_rings() called on a root-direct batched proof")
            .y_rings
    }

    /// Borrow the stored root `v` ring vector (Fold only).
    ///
    /// # Panics
    ///
    /// Panics when called on a root-direct batched proof.
    pub fn v(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("v() called on a root-direct batched proof")
            .v
    }

    /// Commitment to the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics when called on a root-direct batched proof.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("next_w_commitment() called on a root-direct batched proof")
            .stage2
            .next_w_commitment
    }

    /// Claimed evaluation of the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics when called on a root-direct batched proof.
    pub fn next_w_eval(&self) -> L {
        self.as_fold()
            .expect("next_w_eval() called on a root-direct batched proof")
            .stage2
            .next_w_eval
    }
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedFoldRoot<F, L> {
    /// Derive the [`LevelProofShape`] for this fold root.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(
            self.y_rings.coeff_len(),
            &self.v,
            &self.stage1,
            &self.stage2,
        )
    }
}

/// Akita PCS proof for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedProof<F: FieldCore, L: FieldCore> {
    /// Batched root proof over all original-polynomial claims.
    pub root: AkitaBatchedRootProof<F, L>,
    /// Recursive proof steps following the batched root proof.
    pub steps: Vec<AkitaProofStep<F, L>>,
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedProof<F, L> {
    /// Access the terminal direct witness of the recursive-suffix path.
    ///
    /// # Panics
    ///
    /// Panics on a root-direct batched proof (use
    /// [`AkitaBatchedRootProof::as_direct`] to access the per-claim witnesses
    /// in that case), and panics if a fold-rooted proof does not terminate
    /// with a direct witness step.
    pub fn final_witness(&self) -> &DirectWitnessProof<F> {
        self.steps
            .last()
            .and_then(AkitaProofStep::as_direct)
            .expect("batched Akita proof must terminate with a direct step")
    }

    /// Iterate over recursive fold levels.
    pub fn fold_levels(&self) -> impl Iterator<Item = &AkitaLevelProof<F, L>> {
        self.steps.iter().filter_map(AkitaProofStep::as_fold)
    }

    /// Number of recursive fold levels.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_levels().count()
    }

    /// True when this proof uses the root-direct batched fast path (no
    /// two-stage root proof and no recursive suffix).
    pub fn is_root_direct(&self) -> bool {
        self.root.is_direct()
    }

    /// Derive the [`AkitaBatchedProofShape`] for this proof.
    pub fn shape(&self) -> AkitaBatchedProofShape {
        match &self.root {
            AkitaBatchedRootProof::Fold(fold) => AkitaBatchedProofShape::Fold {
                root_shape: fold.shape(),
                step_shapes: self.steps.iter().map(AkitaProofStep::shape).collect(),
            },
            AkitaBatchedRootProof::Direct { witnesses, .. } => AkitaBatchedProofShape::Direct {
                witness_shapes: witnesses.iter().map(DirectWitnessProof::shape).collect(),
            },
        }
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaBatchedProof<F, L> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}

/// A recursive proof step, either a Akita fold or a direct packed-witness
/// handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaProofStep<F: FieldCore, L: FieldCore> {
    /// One recursive Akita fold.
    Fold(AkitaLevelProof<F, L>),
    /// Terminal direct witness handoff.
    Direct(DirectWitnessProof<F>),
}

impl<F: FieldCore, L: FieldCore> AkitaProofStep<F, L> {
    /// Borrow the fold proof when this is a fold step.
    pub fn as_fold(&self) -> Option<&AkitaLevelProof<F, L>> {
        match self {
            Self::Fold(level) => Some(level),
            Self::Direct(_) => None,
        }
    }

    /// Mutably borrow the fold proof when this is a fold step.
    pub fn as_fold_mut(&mut self) -> Option<&mut AkitaLevelProof<F, L>> {
        match self {
            Self::Fold(level) => Some(level),
            Self::Direct(_) => None,
        }
    }

    /// Borrow the packed witness when this is a direct step.
    pub fn as_direct(&self) -> Option<&DirectWitnessProof<F>> {
        match self {
            Self::Fold(_) => None,
            Self::Direct(direct) => Some(direct),
        }
    }

    /// Derive the shape for this proof step.
    pub fn shape(&self) -> AkitaProofStepShape {
        match self {
            Self::Fold(level) => AkitaProofStepShape::Fold(level.shape()),
            Self::Direct(direct) => AkitaProofStepShape::Direct(direct.shape()),
        }
    }
}

/// Shape descriptor for deserializing a [`AkitaLevelProof`] without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProofShape {
    /// Number of field coefficients in `y_ring`.
    pub y_ring_coeffs: usize,
    /// Number of field coefficients in `v`.
    pub v_coeffs: usize,
    /// Stage-1 tree stage shapes in root-to-leaf order.
    pub stage1_stages: Vec<AkitaStage1StageShape>,
    /// Stage-2 sumcheck shape: `(num_rounds, degree)`.
    pub stage2_sumcheck: SumcheckProofShape,
    /// Number of field coefficients in `next_w_commitment`.
    pub next_commit_coeffs: usize,
}

/// Shape descriptor for deserializing an [`AkitaBatchedProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedProofShape {
    /// Standard fold-rooted batched proof with a recursive suffix.
    Fold {
        /// Root-level shape (same field layout as a regular level).
        root_shape: LevelProofShape,
        /// Recursive proof step shapes following the batched root level.
        step_shapes: Vec<AkitaProofStepShape>,
    },
    /// Root-direct batched proof: one direct witness per claim.
    Direct {
        /// Per-claim direct witness shapes.
        witness_shapes: Vec<DirectWitnessShape>,
    },
}

/// Shape descriptor for deserializing a proof step without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaProofStepShape {
    /// Shape of a recursive fold level.
    Fold(LevelProofShape),
    /// Shape of a direct packed witness.
    Direct(DirectWitnessShape),
}

fn sumcheck_shape<F: FieldCore>(sc: &SumcheckProof<F>) -> SumcheckProofShape {
    let degree = sc
        .round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (sc.round_polys.len(), degree)
}

fn eq_factored_sumcheck_shape<F: FieldCore>(
    sc: &EqFactoredSumcheckProof<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = sc
        .round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (sc.round_polys.len(), degree)
}

fn level_proof_shape<F: FieldCore, L: FieldCore>(
    y_coeffs: usize,
    v: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<L>,
    stage2: &AkitaStage2Proof<F, L>,
) -> LevelProofShape {
    LevelProofShape {
        y_ring_coeffs: y_coeffs,
        v_coeffs: v.coeff_len(),
        stage1_stages: stage1
            .stages
            .iter()
            .map(|stage| AkitaStage1StageShape {
                sumcheck: eq_factored_sumcheck_shape(&stage.sumcheck),
                child_claims: stage.child_claims.len(),
            })
            .collect(),
        stage2_sumcheck: sumcheck_shape(&stage2.sumcheck),
        next_commit_coeffs: stage2.next_w_commitment.coeff_len(),
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaLevelProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring.serialize_with_mode(&mut writer, compress)?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        for stage in &self.stage1.stages {
            stage.sumcheck.serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_eval
            .serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        let base = self.y_ring.serialized_size(compress) + self.v.serialized_size(compress);
        base + self
            .stage1
            .stages
            .iter()
            .map(|stage| {
                stage.sumcheck.serialized_size(compress)
                    + stage
                        .child_claims
                        .iter()
                        .map(|claim| claim.serialized_size(compress))
                        .sum::<usize>()
            })
            .sum::<usize>()
            + self.stage1.s_claim.serialized_size(compress)
            + self.stage2.sumcheck.serialized_size(compress)
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaLevelProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_ring.check()?;
        if self.y_ring.coeff_len() == 0 {
            return Err(SerializationError::InvalidData(
                "Akita level y_ring must contain exactly one ring element".to_string(),
            ));
        }
        self.v.check()?;
        for stage in &self.stage1.stages {
            stage.sumcheck.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        self.stage2.sumcheck.check()?;
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval.check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaLevelProof<F, L>
{
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        let y_ring = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
        let mut stage1_stages = Vec::with_capacity(ctx.stage1_stages.len());
        for stage_shape in &ctx.stage1_stages {
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck,
            )?;
            let mut child_claims = Vec::with_capacity(stage_shape.child_claims);
            for _ in 0..stage_shape.child_claims {
                child_claims.push(L::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                sumcheck,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2 = AkitaStage2Proof {
            sumcheck: SumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck,
            )?,
            next_w_commitment: FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            next_w_eval: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            y_ring,
            v,
            stage1,
            stage2,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize> AkitaSerialize for DirectWitnessProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits(packed) => packed.serialize_with_mode(&mut writer, compress),
            Self::FieldElements(field_elems) => {
                field_elems.serialize_with_mode(&mut writer, compress)
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::PackedDigits(packed) => packed.serialized_size(compress),
            Self::FieldElements(field_elems) => field_elems.serialized_size(compress),
        }
    }
}

impl<F: FieldCore + Valid> Valid for DirectWitnessProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits(packed) => packed.check(),
            Self::FieldElements(field_elems) => field_elems.check(),
        }
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for DirectWitnessProof<F>
{
    type Context = DirectWitnessShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &DirectWitnessShape,
    ) -> Result<Self, SerializationError> {
        let out = match ctx {
            DirectWitnessShape::PackedDigits(shape) => Self::PackedDigits(
                PackedDigits::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
            DirectWitnessShape::FieldElements(num_coeffs) => Self::FieldElements(
                FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, num_coeffs)?,
            ),
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaProofStep<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Fold(level) => level.serialize_with_mode(&mut writer, compress),
            Self::Direct(direct) => direct.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::Fold(level) => level.serialized_size(compress),
            Self::Direct(direct) => direct.serialized_size(compress),
        }
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaProofStep<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Fold(level) => level.check(),
            Self::Direct(direct) => direct.check(),
        }
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaProofStep<F, L>
{
    type Context = AkitaProofStepShape;

    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &AkitaProofStepShape,
    ) -> Result<Self, SerializationError> {
        let out = match ctx {
            AkitaProofStepShape::Fold(shape) => Self::Fold(AkitaLevelProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                shape,
            )?),
            AkitaProofStepShape::Direct(shape) => Self::Direct(
                DirectWitnessProof::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedFoldRoot<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_rings.serialize_with_mode(&mut writer, compress)?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        for stage in &self.stage1.stages {
            stage.sumcheck.serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_eval
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_rings.serialized_size(compress)
            + self.v.serialized_size(compress)
            + self
                .stage1
                .stages
                .iter()
                .map(|stage| {
                    stage.sumcheck.serialized_size(compress)
                        + stage
                            .child_claims
                            .iter()
                            .map(|claim| claim.serialized_size(compress))
                            .sum::<usize>()
                })
                .sum::<usize>()
            + self.stage1.s_claim.serialized_size(compress)
            + self.stage2.sumcheck.serialized_size(compress)
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedFoldRoot<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_rings.check()?;
        self.v.check()?;
        for stage in &self.stage1.stages {
            stage.sumcheck.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        self.stage2.sumcheck.check()?;
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval.check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaBatchedFoldRoot<F, L>
{
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        let y_rings = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
        let mut stage1_stages = Vec::with_capacity(ctx.stage1_stages.len());
        for stage_shape in &ctx.stage1_stages {
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck,
            )?;
            let mut child_claims = Vec::with_capacity(stage_shape.child_claims);
            for _ in 0..stage_shape.child_claims {
                child_claims.push(L::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                sumcheck,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2 = AkitaStage2Proof {
            sumcheck: SumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck,
            )?,
            next_w_commitment: FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            next_w_eval: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            y_rings,
            v,
            stage1,
            stage2,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedRootProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Fold(fold) => fold.serialize_with_mode(&mut writer, compress),
            Self::Direct {
                witnesses,
                #[cfg(feature = "zk")]
                b_blinding_digits,
            } => {
                for witness in witnesses {
                    witness.serialize_with_mode(&mut writer, compress)?;
                }
                #[cfg(feature = "zk")]
                b_blinding_digits.serialize_with_mode(&mut writer, compress)?;
                Ok(())
            }
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::Fold(fold) => fold.serialized_size(compress),
            Self::Direct {
                witnesses,
                #[cfg(feature = "zk")]
                b_blinding_digits,
            } => {
                let witness_size = witnesses
                    .iter()
                    .map(|witness| witness.serialized_size(compress))
                    .sum::<usize>();
                #[cfg(feature = "zk")]
                {
                    witness_size + b_blinding_digits.serialized_size(compress)
                }
                #[cfg(not(feature = "zk"))]
                {
                    witness_size
                }
            }
        }
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedRootProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Fold(fold) => fold.check(),
            Self::Direct {
                witnesses,
                #[cfg(feature = "zk")]
                b_blinding_digits,
            } => {
                for witness in witnesses {
                    witness.check()?;
                }
                #[cfg(feature = "zk")]
                b_blinding_digits.check()?;
                Ok(())
            }
        }
    }
}

impl<F: FieldCore + AkitaSerialize, L: FieldCore + AkitaSerialize> AkitaSerialize
    for AkitaBatchedProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.root.serialize_with_mode(&mut writer, compress)?;
        for step in &self.steps {
            step.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.root.serialized_size(compress)
            + self
                .steps
                .iter()
                .map(|step| step.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.root.check()?;
        for step in &self.steps {
            step.check()?;
        }
        match &self.root {
            AkitaBatchedRootProof::Fold(_) => {
                let Some(AkitaProofStep::Direct(_)) = self.steps.last() else {
                    return Err(SerializationError::InvalidData(
                        "batched Akita proof must terminate with a direct step".to_string(),
                    ));
                };
                if self.steps[..self.steps.len().saturating_sub(1)]
                    .iter()
                    .any(|step| !matches!(step, AkitaProofStep::Fold(_)))
                {
                    return Err(SerializationError::InvalidData(
                        "batched Akita proof may only contain fold steps before the terminal direct step"
                            .to_string(),
                    ));
                }
                let mut levels = self.fold_levels();
                if let Some(first) = levels.next() {
                    if !self
                        .root
                        .next_w_commitment()
                        .can_decode_vec(first.level_d())
                    {
                        return Err(SerializationError::InvalidData(
                            "batched root proof has mismatched next-commitment dimension"
                                .to_string(),
                        ));
                    }
                }
                let fold_levels: Vec<_> = self.fold_levels().collect();
                for levels in fold_levels.windows(2) {
                    if !levels[0]
                        .next_w_commitment()
                        .can_decode_vec(levels[1].level_d())
                    {
                        return Err(SerializationError::InvalidData(
                            "adjacent Akita levels have mismatched commitment dimensions"
                                .to_string(),
                        ));
                    }
                }
            }
            AkitaBatchedRootProof::Direct { .. } => {
                if !self.steps.is_empty() {
                    return Err(SerializationError::InvalidData(
                        "root-direct batched proof must not carry recursive-suffix steps"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for AkitaBatchedProof<F, L>
{
    type Context = AkitaBatchedProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &AkitaBatchedProofShape,
    ) -> Result<Self, SerializationError> {
        let out = match ctx {
            AkitaBatchedProofShape::Fold {
                root_shape,
                step_shapes,
            } => {
                let fold = AkitaBatchedFoldRoot::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    root_shape,
                )?;
                let mut steps = Vec::with_capacity(step_shapes.len());
                for shape in step_shapes {
                    steps.push(AkitaProofStep::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        shape,
                    )?);
                }
                Self {
                    root: AkitaBatchedRootProof::Fold(fold),
                    steps,
                }
            }
            AkitaBatchedProofShape::Direct { witness_shapes } => {
                let mut witnesses = Vec::with_capacity(witness_shapes.len());
                for shape in witness_shapes {
                    witnesses.push(DirectWitnessProof::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        shape,
                    )?);
                }
                #[cfg(feature = "zk")]
                let b_blinding_digits =
                    Vec::<Vec<i8>>::deserialize_with_mode(&mut reader, compress, validate, &())?;
                Self {
                    root: AkitaBatchedRootProof::Direct {
                        witnesses,
                        #[cfg(feature = "zk")]
                        b_blinding_digits,
                    },
                    steps: Vec::new(),
                }
            }
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akita_field::Prime128Offset275;
    use akita_serialization::Valid;

    #[test]
    fn packed_digits_roundtrip_basis6() {
        let digits = vec![-32, -17, -1, 0, 1, 31];
        let packed = PackedDigits::from_i8_digits(&digits, 6);

        assert_eq!(packed.bits_per_elem, 6);
        let recovered: Vec<i8> = (0..digits.len())
            .map(|idx| packed.digit_at(idx).expect("packed index in bounds"))
            .collect();
        assert_eq!(recovered, digits);

        let expected_field: Vec<Prime128Offset275> = digits
            .iter()
            .map(|&digit| Prime128Offset275::from_i64(digit as i64))
            .collect();
        assert_eq!(packed.to_field_elems::<Prime128Offset275>(), expected_field);
    }

    #[test]
    fn packed_digits_reject_bits_above_six() {
        let packed = PackedDigits {
            num_elems: 1,
            bits_per_elem: 7,
            data: vec![0],
        };

        assert!(packed.check().is_err());
    }
}
