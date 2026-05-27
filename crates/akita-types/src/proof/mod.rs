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
    flatten_batched_commitment_rows, folded_root_supports_opening_shape,
    prepare_recursive_opening_point_ext, prepare_root_opening_point,
    prepare_root_opening_point_ext, ring_inner_product_with_extension_weights,
    ring_subfield_packed_extension_opening_point, root_tensor_projection_enabled,
    validate_batched_inputs, PreparedRecursiveOpeningPoint, PreparedRootOpeningPoint,
    RingMultiplierOpeningPoint,
};
pub use commitment::{AkitaCommitment, DummyProof, RingCommitment};
pub use incidence::{
    append_claim_incidence_shape_to_transcript, sample_public_row_coefficients,
    verifier_claims_to_incidence, ClaimIncidence, ClaimIncidenceLimits, ClaimIncidenceSummary,
    IncidenceClaim, PublicOpeningRow,
};
pub use relation::{relation_claim_from_rows, relation_claim_from_rows_extension};
pub use scheme::{CommitmentVerifier, CommittedOpenings, OpeningPoints, VerifierClaims};
pub use setup::{
    AkitaExpandedSetup, AkitaSetupSeed, AkitaVerifierSetup, PublicMatrixSeed, SetupMatrixEnvelope,
    ZkBlindingSeed,
};
pub use stage1::{
    absorb_interstage_claims, combine_polys, eval_poly, linear_combination,
    range_check_eval_from_s, reorder_stage1_coords, stage1_interstage_batch_weights,
    stage1_leaf_coeffs, stage1_stage_count, stage1_tree_product_stage_arities,
    stage1_tree_stage_shapes, validate_stage1_tree_basis,
};

use akita_algebra::CyclotomicRing;
use akita_field::AkitaError;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, DEFAULT_MAX_SEQUENCE_LEN};
use akita_serialization::{Compress, SerializationError};
use akita_serialization::{Valid, Validate};
use akita_sumcheck::{
    uniform_sumcheck_shape, EqFactoredSumcheckProofShape, SumcheckProofShape,
    EXTENSION_OPENING_REDUCTION_DEGREE,
};
#[cfg(not(feature = "zk"))]
use akita_sumcheck::{EqFactoredSumcheckProof, SumcheckProof};
#[cfg(feature = "zk")]
use akita_sumcheck::{EqFactoredSumcheckProofMasked, SumcheckProofMasked};
use akita_transcript::Transcript;
use std::io::{Read, Write};
use std::marker::PhantomData;

fn checked_shape_len(len: usize) -> Result<(), SerializationError> {
    if len > DEFAULT_MAX_SEQUENCE_LEN {
        return Err(SerializationError::LengthLimitExceeded {
            len: u64::try_from(len).unwrap_or(u64::MAX),
            max: DEFAULT_MAX_SEQUENCE_LEN,
        });
    }
    Ok(())
}

fn reserve_shape_len<T>(vec: &mut Vec<T>, len: usize) -> Result<(), SerializationError> {
    checked_shape_len(len)?;
    vec.try_reserve_exact(len)
        .map_err(|_| SerializationError::InvalidData("shape-backed allocation failed".to_string()))
}

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
    pub fn digit_at(&self, idx: usize) -> Option<i8> {
        if idx >= self.num_elems {
            return None;
        }

        let bits = self.bits_per_elem as usize;
        if bits == 0 || bits > 6 {
            return None;
        }
        let mask = (1u8 << bits) - 1;
        let sign_bit = 1u8 << (bits - 1);
        let bit_offset = idx.checked_mul(bits)?;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let mut raw = (self.data.get(byte_idx)? >> bit_idx) & mask;
        if bit_idx + bits > 8 {
            raw |= (self.data.get(byte_idx + 1)? << (8 - bit_idx)) & mask;
        }

        Some(if raw & sign_bit != 0 {
            raw as i8 | !(mask as i8)
        } else {
            raw as i8
        })
    }

    /// Unpack to field elements.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the packed byte buffer is
    /// malformed relative to `num_elems`/`bits_per_elem`.
    pub fn to_field_elems<F: FieldCore + FromPrimitiveInt>(&self) -> Result<Vec<F>, AkitaError> {
        let mut out = Vec::with_capacity(self.num_elems);
        for i in 0..self.num_elems {
            let signed = self.digit_at(i).ok_or(AkitaError::InvalidProof)?;
            out.push(F::from_i64(signed as i64));
        }
        Ok(out)
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
        let expected_bits = self
            .num_elems
            .checked_mul(self.bits_per_elem as usize)
            .ok_or(SerializationError::LengthLimitExceeded {
                len: u64::MAX,
                max: DEFAULT_MAX_SEQUENCE_LEN,
            })?;
        let expected_bytes = expected_bits.div_ceil(8);
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
        if matches!(_validate, Validate::Yes) {
            DirectWitnessShape::PackedDigits(*ctx).check()?;
        }
        let num_bits = num_elems.checked_mul(bits_per_elem as usize).ok_or(
            SerializationError::LengthLimitExceeded {
                len: u64::MAX,
                max: DEFAULT_MAX_SEQUENCE_LEN,
            },
        )?;
        let num_bytes = num_bits.div_ceil(8);
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

/// Prover-side hint for one opening-point commitment bundle.
///
/// Stores per-polynomial decomposed inner rows and, when available, the
/// corresponding recomposed inner rows for all polynomials bundled into the
/// single commitment at one opening point.
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

    /// Get the B-blinding digit streams, one per opening-point commitment.
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
    #[cfg(not(feature = "zk"))]
    pub sumcheck_proof: EqFactoredSumcheckProof<F>,
    /// ZK plain-opening masked round payload.
    #[cfg(feature = "zk")]
    pub sumcheck_proof_masked: EqFactoredSumcheckProofMasked<F>,
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
    pub sumcheck_proof: EqFactoredSumcheckProofShape,
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
    #[cfg(not(feature = "zk"))]
    pub sumcheck_proof: SumcheckProof<L>,
    /// ZK plain-opening masked compressed round payload.
    #[cfg(feature = "zk")]
    pub sumcheck_proof_masked: SumcheckProofMasked<L>,
    /// Commitment to the next witness `w`
    /// (ring dim = next level's D, may differ from y_ring/v).
    pub next_w_commitment: FlatRingVec<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    #[cfg(not(feature = "zk"))]
    pub next_w_eval: L,
    /// Masked claimed evaluation of the next witness `w` at the stage-2 challenge point.
    #[cfg(feature = "zk")]
    pub next_w_eval_masked: L,
}

impl<F: FieldCore, L: FieldCore> AkitaStage2Proof<F, L> {
    /// Wire value for the next-witness evaluation claim.
    ///
    /// In transparent builds this is the true evaluation; in ZK builds this is
    /// the masked evaluation carried on the proof transcript.
    pub fn next_w_eval(&self) -> L {
        #[cfg(not(feature = "zk"))]
        {
            self.next_w_eval
        }
        #[cfg(feature = "zk")]
        {
            self.next_w_eval_masked
        }
    }
}

/// Optional proof that reduces a logical extension-field opening into one
/// ordinary opening of the transformed committed witness.
///
/// This object is not serialized with a tag or length. Its presence and shape
/// are determined by the verifier's expected proof shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionProof<L: FieldCore> {
    /// Transcript-bound partial evaluations used by the basis-conversion
    /// check.
    pub partials: Vec<L>,
    /// Degree-two reduction sumcheck.
    #[cfg(not(feature = "zk"))]
    pub sumcheck: SumcheckProof<L>,
    /// ZK plain-opening masked compressed degree-two reduction sumcheck.
    #[cfg(feature = "zk")]
    pub sumcheck_proof_masked: SumcheckProofMasked<L>,
}

/// Headerless shape for [`ExtensionOpeningReductionProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionOpeningReductionShape {
    /// Number of partial evaluations serialized before the sumcheck.
    pub partials: usize,
    /// Reduction sumcheck shape: one compact coefficient count per round.
    pub sumcheck: SumcheckProofShape,
}

impl<L: FieldCore> ExtensionOpeningReductionProof<L> {
    /// Shape descriptor required for headerless deserialization.
    pub fn shape(&self) -> ExtensionOpeningReductionShape {
        ExtensionOpeningReductionShape {
            partials: self.partials.len(),
            #[cfg(not(feature = "zk"))]
            sumcheck: sumcheck_shape(&self.sumcheck),
            #[cfg(feature = "zk")]
            sumcheck: sumcheck_proof_masked_shape(&self.sumcheck_proof_masked),
        }
    }

    /// Number of sumcheck rounds in the reduction proof.
    pub fn num_rounds(&self) -> usize {
        #[cfg(not(feature = "zk"))]
        {
            self.sumcheck.round_polys.len()
        }
        #[cfg(feature = "zk")]
        {
            self.sumcheck_proof_masked.masked_round_polys.len()
        }
    }
}

impl ExtensionOpeningReductionShape {
    /// Construct the standard degree-two reduction shape.
    pub fn standard(partials: usize, num_rounds: usize) -> Self {
        Self {
            partials,
            sumcheck: uniform_sumcheck_shape(num_rounds, EXTENSION_OPENING_REDUCTION_DEGREE),
        }
    }
}

impl Valid for ExtensionOpeningReductionShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.partials)?;
        checked_shape_len(self.sumcheck.len())?;
        for &degree in &self.sumcheck {
            checked_shape_len(degree)?;
            if degree != EXTENSION_OPENING_REDUCTION_DEGREE {
                return Err(SerializationError::InvalidData(format!(
                    "extension opening reduction degree {} does not match expected degree {}",
                    degree, EXTENSION_OPENING_REDUCTION_DEGREE
                )));
            }
        }
        Ok(())
    }
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
    /// Optional extension-opening reduction payload. `None` for degree-one
    /// openings and proof paths that do not use extension-opening reduction.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
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
            extension_opening_reduction: None,
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
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            y_ring,
            v,
            stage1,
            AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof_masked,
                next_w_commitment: next_w_commitment.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: next_w_eval,
            },
        )
    }

    /// Construct a level proof for a multi-row public opening relation.
    ///
    /// The singleton recursive path is the `y_rings.len() == 1`
    /// specialization.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_many<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new_two_stage_many_with_extension_opening_reduction::<D>(
            y_rings,
            None,
            v,
            stage1,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            next_w_commitment,
            next_w_eval,
        )
    }

    /// Construct a level proof for a multi-row public opening relation with
    /// extension-opening reduction payloads already produced.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_many_with_extension_opening_reduction<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self {
            y_ring: FlatRingVec::from_ring_elems(&y_rings).into_compact(),
            extension_opening_reduction,
            v: FlatRingVec::from_ring_elems(&v).into_compact(),
            stage1,
            stage2: AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof_masked,
                next_w_commitment: next_w_commitment.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: next_w_eval,
            },
        }
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

    /// Reconstruct typed public opening rings.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_y_rings_typed<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.y_ring.try_to_vec()
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
        self.stage2.next_w_eval()
    }

    /// Derive the [`LevelProofShape`] for this level proof.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(
            self.y_ring.coeff_len(),
            self.extension_opening_reduction.as_ref(),
            &self.v,
            &self.stage1,
            &self.stage2,
        )
    }
}

/// Terminal fold-level proof.
///
/// Ships `final_witness` in cleartext, absorbed into the transcript at the
/// `ABSORB_SUMCHECK_W` position in place of the prior `next_w_commitment`.
/// Drops the redundant proof components at the terminal: `stage1`
/// (`PackedDigits` structurally enforces digit range), `next_w_commitment`
/// (replaced by `final_witness`), and `next_w_eval` (verifier computes
/// directly from `final_witness`). The terminal M-row layout also drops the
/// D-row block, so `v` is not serialized at the terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalLevelProof<F: FieldCore, L: FieldCore> {
    /// Public output ring(s). At a non-root terminal step this carries
    /// exactly one ring; at the root terminal (1-fold case) it carries one
    /// ring per opening point.
    pub y_rings: FlatRingVec<F>,
    /// Optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Stage-2 fused sumcheck proof.
    #[cfg(not(feature = "zk"))]
    pub stage2_sumcheck: SumcheckProof<L>,
    #[cfg(feature = "zk")]
    pub stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
    /// Terminal witness, absorbed via `ABSORB_SUMCHECK_W` in place of
    /// `next_w_commitment`.
    pub final_witness: DirectWitnessProof<F>,
}

impl<F: FieldCore, L: FieldCore> TerminalLevelProof<F, L> {
    /// Construct from typed ring elements and a terminal direct witness.
    ///
    /// Pass `extension_opening_reduction = None` for opening shapes that do
    /// not use extension-opening reduction.
    pub fn new_with_extension_opening_reduction<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        final_witness: DirectWitnessProof<F>,
    ) -> Self {
        Self {
            y_rings: FlatRingVec::from_ring_elems(&y_rings).into_compact(),
            extension_opening_reduction,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            final_witness,
        }
    }

    /// Reconstruct typed public opening rings.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidProof`] if the stored payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_y_rings_typed<const D: usize>(
        &self,
    ) -> Result<Vec<CyclotomicRing<F, D>>, AkitaError> {
        self.y_rings.try_to_vec()
    }

    /// Derive the [`TerminalLevelProofShape`] for this terminal-level proof.
    pub fn shape(&self) -> TerminalLevelProofShape {
        TerminalLevelProofShape {
            y_rings_coeffs: self.y_rings.coeff_len(),
            extension_opening_reduction: self
                .extension_opening_reduction
                .as_ref()
                .map(ExtensionOpeningReductionProof::shape),
            stage2_sumcheck: {
                #[cfg(not(feature = "zk"))]
                {
                    sumcheck_shape(&self.stage2_sumcheck)
                }
                #[cfg(feature = "zk")]
                {
                    sumcheck_proof_masked_shape(&self.stage2_sumcheck_proof_masked)
                }
            },
            final_witness: self.final_witness.shape(),
        }
    }
}

/// Shape descriptor for deserializing a [`TerminalLevelProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalLevelProofShape {
    /// Number of field coefficients in `y_rings`.
    pub y_rings_coeffs: usize,
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Stage-2 sumcheck shape: one compact coefficient count per round.
    pub stage2_sumcheck: SumcheckProofShape,
    /// Shape of the terminal direct witness.
    pub final_witness: DirectWitnessShape,
}

/// Fused batched-root payload for the two-stage folding protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedFoldRoot<F: FieldCore, L: FieldCore> {
    /// Per-point batched public outputs `(y_j)_j`, stored as a flat ring vector.
    pub y_rings: FlatRingVec<F>,
    /// Optional extension-opening reduction payload. `None` until the
    /// extension-opening reduction cutover is wired into the root path.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    /// Aggregated `v = Σ_ell D_ell · w_hat_ell`.
    pub v: FlatRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: AkitaStage1Proof<L>,
    /// Stage-2 fused payload.
    pub stage2: AkitaStage2Proof<F, L>,
}

/// Root proof payload for fused batched openings.
///
/// Three-way split:
///
/// * `Fold` — standard two-stage folded root proof followed by intermediate
///   steps and a terminal step.
/// * `Terminal` — 1-fold case where the root itself is the terminal level.
///   No recursive-suffix steps follow.
/// * `Direct` — 0-fold (root-direct) batched fast path: one direct
///   field-element witness per claim, in the normalized incidence claim order
///   used by the prover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedRootProof<F: FieldCore, L: FieldCore> {
    /// Standard two-stage folded root proof.
    Fold(AkitaBatchedFoldRoot<F, L>),
    /// 1-fold root: the root level is itself the terminal fold level.
    Terminal(TerminalLevelProof<F, L>),
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
            extension_opening_reduction: None,
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
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new_two_stage_with_extension_opening_reduction::<D>(
            y_rings,
            None,
            v,
            stage1,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck_proof,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            next_w_commitment,
            next_w_eval,
        )
    }

    /// Construct a batched root proof for the two-stage norm-check with
    /// extension-opening reduction payloads already produced.
    #[allow(clippy::too_many_arguments)]
    pub fn new_two_stage_with_extension_opening_reduction<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: AkitaStage1Proof<L>,
        #[cfg(not(feature = "zk"))] stage2_sumcheck_proof: SumcheckProof<L>,
        #[cfg(feature = "zk")] stage2_sumcheck_proof_masked: SumcheckProofMasked<L>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: L,
    ) -> Self {
        Self::new::<D>(
            y_rings,
            v,
            stage1,
            AkitaStage2Proof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: stage2_sumcheck_proof,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked: stage2_sumcheck_proof_masked,
                next_w_commitment: next_w_commitment.into_compact(),
                #[cfg(not(feature = "zk"))]
                next_w_eval,
                #[cfg(feature = "zk")]
                next_w_eval_masked: next_w_eval,
            },
        )
        .with_extension_opening_reduction(extension_opening_reduction)
    }

    /// Attach extension-opening reduction payloads to a folded root proof.
    pub fn with_extension_opening_reduction(
        mut self,
        extension_opening_reduction: Option<ExtensionOpeningReductionProof<L>>,
    ) -> Self {
        if let Self::Fold(fold) = &mut self {
            fold.extension_opening_reduction = extension_opening_reduction;
        }
        self
    }

    /// Construct the terminal-root variant (1-fold case): the root itself is
    /// the terminal fold level.
    pub fn new_terminal(terminal: TerminalLevelProof<F, L>) -> Self {
        Self::Terminal(terminal)
    }

    /// Construct the root-direct batched variant with one witness per claim.
    #[cfg(not(feature = "zk"))]
    pub fn new_direct(witnesses: Vec<DirectWitnessProof<F>>) -> Self {
        Self::Direct { witnesses }
    }

    /// Construct the root-direct batched variant with one witness per claim and
    /// one revealed B-blinding payload per opening-point commitment.
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
            Self::Terminal(_) | Self::Direct { .. } => None,
        }
    }

    /// Mutably borrow the fold payload when this is a fold root.
    pub fn as_fold_mut(&mut self) -> Option<&mut AkitaBatchedFoldRoot<F, L>> {
        match self {
            Self::Fold(fold) => Some(fold),
            Self::Terminal(_) | Self::Direct { .. } => None,
        }
    }

    /// Borrow the terminal-root payload when this is a terminal root.
    pub fn as_terminal_root(&self) -> Option<&TerminalLevelProof<F, L>> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            Self::Fold(_) | Self::Direct { .. } => None,
        }
    }

    /// Mutably borrow the terminal-root payload when this is a terminal root.
    pub fn as_terminal_root_mut(&mut self) -> Option<&mut TerminalLevelProof<F, L>> {
        match self {
            Self::Terminal(terminal) => Some(terminal),
            Self::Fold(_) | Self::Direct { .. } => None,
        }
    }

    /// Borrow the per-claim direct witnesses when this is a root-direct
    /// batched proof.
    pub fn as_direct(&self) -> Option<&[DirectWitnessProof<F>]> {
        match self {
            Self::Direct { witnesses, .. } => Some(witnesses.as_slice()),
            Self::Fold(_) | Self::Terminal(_) => None,
        }
    }

    /// Borrow the revealed root-direct B-blinding payloads.
    #[cfg(feature = "zk")]
    pub fn direct_b_blinding_digits(&self) -> Option<&[Vec<i8>]> {
        match self {
            Self::Direct {
                b_blinding_digits, ..
            } => Some(b_blinding_digits.as_slice()),
            Self::Fold(_) | Self::Terminal(_) => None,
        }
    }

    /// True when this root proof is a root-direct batched fast path.
    pub fn is_direct(&self) -> bool {
        matches!(self, Self::Direct { .. })
    }

    /// True when this root proof is itself the terminal fold level.
    pub fn is_terminal_root(&self) -> bool {
        matches!(self, Self::Terminal(_))
    }

    /// Borrow the stored root per-point `y_rings` payload (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
    pub fn y_rings(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("y_rings() called on a non-fold root proof")
            .y_rings
    }

    /// Borrow the stored root `v` ring vector (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
    pub fn v(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("v() called on a non-fold root proof")
            .v
    }

    /// Commitment to the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self
            .as_fold()
            .expect("next_w_commitment() called on a non-fold root proof")
            .stage2
            .next_w_commitment
    }

    /// Claimed evaluation of the next witness `w` (Fold only).
    ///
    /// # Panics
    ///
    /// Panics on terminal-root and root-direct batched proofs.
    pub fn next_w_eval(&self) -> L {
        self.as_fold()
            .expect("next_w_eval() called on a non-fold root proof")
            .stage2
            .next_w_eval()
    }
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedFoldRoot<F, L> {
    /// Derive the [`LevelProofShape`] for this fold root.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(
            self.y_rings.coeff_len(),
            self.extension_opening_reduction.as_ref(),
            &self.v,
            &self.stage1,
            &self.stage2,
        )
    }
}

/// Akita PCS proof for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkitaBatchedProof<F: FieldCore, L: FieldCore> {
    /// Plain-opening ZK hiding-factor commitment and opening payload.
    #[cfg(feature = "zk")]
    pub zk_hiding: ZkHidingProof<F>,
    /// Batched root proof over all original-polynomial claims.
    pub root: AkitaBatchedRootProof<F, L>,
    /// Recursive proof steps following the batched root proof.
    pub steps: Vec<AkitaProofStep<F, L>>,
}

impl<F: FieldCore, L: FieldCore> AkitaBatchedProof<F, L> {
    /// Access the terminal direct witness of the recursive-suffix path.
    ///
    /// Returns the `final_witness` from the terminal level: either the
    /// terminal step at the tail of a fold-rooted suffix, or directly from
    /// the [`AkitaBatchedRootProof::Terminal`] root (1-fold case).
    ///
    /// # Panics
    ///
    /// Panics on a root-direct batched proof (use
    /// [`AkitaBatchedRootProof::as_direct`] to access the per-claim witnesses
    /// in that case), and panics if a fold-rooted proof does not terminate
    /// with a terminal step.
    pub fn final_witness(&self) -> &DirectWitnessProof<F> {
        match &self.root {
            AkitaBatchedRootProof::Terminal(terminal) => &terminal.final_witness,
            AkitaBatchedRootProof::Fold(_) => {
                &self
                    .steps
                    .last()
                    .and_then(AkitaProofStep::as_terminal)
                    .expect("fold-rooted Akita proof must terminate with a terminal step")
                    .final_witness
            }
            AkitaBatchedRootProof::Direct { .. } => {
                panic!("final_witness() called on a root-direct batched proof")
            }
        }
    }

    /// Iterate over the intermediate (non-terminal) fold levels of the
    /// recursive suffix.
    pub fn fold_levels(&self) -> impl Iterator<Item = &AkitaLevelProof<F, L>> {
        self.steps
            .iter()
            .filter_map(AkitaProofStep::as_intermediate)
    }

    /// Number of intermediate recursive fold levels.
    pub fn num_fold_levels(&self) -> usize {
        self.fold_levels().count()
    }

    /// True when this proof uses the root-direct batched fast path (no
    /// two-stage root proof and no recursive suffix).
    pub fn is_root_direct(&self) -> bool {
        self.root.is_direct()
    }

    /// True when the batched root is itself the terminal fold level (1-fold
    /// case).
    pub fn is_root_terminal(&self) -> bool {
        self.root.is_terminal_root()
    }

    /// Derive the [`AkitaBatchedProofShape`] for this proof.
    pub fn shape(&self) -> AkitaBatchedProofShape {
        match &self.root {
            AkitaBatchedRootProof::Fold(fold) => AkitaBatchedProofShape::Fold {
                root_shape: fold.shape(),
                step_shapes: self.steps.iter().map(AkitaProofStep::shape).collect(),
            },
            AkitaBatchedRootProof::Terminal(terminal) => {
                AkitaBatchedProofShape::Terminal(terminal.shape())
            }
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

/// A recursive proof step.
///
/// Hard-split between intermediate fold levels (which still ship a recursive
/// `next_w_commitment`) and the terminal fold level (which ships the witness
/// in cleartext via `TerminalLevelProof::final_witness`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaProofStep<F: FieldCore, L: FieldCore> {
    /// Intermediate (non-terminal) fold level. Ships `next_w_commitment` and
    /// the stage-1 range-check tree.
    Intermediate(AkitaLevelProof<F, L>),
    /// Terminal fold level. Ships `final_witness` in cleartext (absorbed via
    /// `ABSORB_SUMCHECK_W`) and drops `stage1`, `next_w_commitment`,
    /// `next_w_eval`.
    Terminal(TerminalLevelProof<F, L>),
}

impl<F: FieldCore, L: FieldCore> AkitaProofStep<F, L> {
    /// Borrow the intermediate fold proof when this is an intermediate step.
    pub fn as_intermediate(&self) -> Option<&AkitaLevelProof<F, L>> {
        match self {
            Self::Intermediate(level) => Some(level),
            Self::Terminal(_) => None,
        }
    }

    /// Mutably borrow the intermediate fold proof when this is an
    /// intermediate step.
    pub fn as_intermediate_mut(&mut self) -> Option<&mut AkitaLevelProof<F, L>> {
        match self {
            Self::Intermediate(level) => Some(level),
            Self::Terminal(_) => None,
        }
    }

    /// Borrow the terminal level proof when this is a terminal step.
    pub fn as_terminal(&self) -> Option<&TerminalLevelProof<F, L>> {
        match self {
            Self::Intermediate(_) => None,
            Self::Terminal(terminal) => Some(terminal),
        }
    }

    /// Mutably borrow the terminal level proof when this is a terminal step.
    pub fn as_terminal_mut(&mut self) -> Option<&mut TerminalLevelProof<F, L>> {
        match self {
            Self::Intermediate(_) => None,
            Self::Terminal(terminal) => Some(terminal),
        }
    }

    /// Derive the shape for this proof step.
    pub fn shape(&self) -> AkitaProofStepShape {
        match self {
            Self::Intermediate(level) => AkitaProofStepShape::Intermediate(level.shape()),
            Self::Terminal(terminal) => AkitaProofStepShape::Terminal(terminal.shape()),
        }
    }
}

/// Shape descriptor for deserializing a [`AkitaLevelProof`] without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProofShape {
    /// Number of field coefficients in `y_ring`.
    pub y_ring_coeffs: usize,
    /// Shape of the optional extension-opening reduction payload.
    pub extension_opening_reduction: Option<ExtensionOpeningReductionShape>,
    /// Number of field coefficients in `v`.
    pub v_coeffs: usize,
    /// Stage-1 tree stage shapes in root-to-leaf order.
    pub stage1_stages: Vec<AkitaStage1StageShape>,
    /// Stage-2 sumcheck shape: `(num_rounds, degree)`.
    pub stage2_sumcheck_proof: SumcheckProofShape,
    /// Number of field coefficients in `next_w_commitment`.
    pub next_commit_coeffs: usize,
}

/// Shape descriptor for deserializing an [`AkitaBatchedProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaBatchedProofShape {
    /// Standard fold-rooted batched proof with a recursive suffix. The
    /// recursive suffix is a (possibly empty) sequence of
    /// [`AkitaProofStepShape::Intermediate`] step shapes followed by exactly
    /// one [`AkitaProofStepShape::Terminal`].
    Fold {
        /// Root-level shape (same field layout as a regular level).
        root_shape: LevelProofShape,
        /// Recursive proof step shapes following the batched root level.
        step_shapes: Vec<AkitaProofStepShape>,
    },
    /// Terminal-rooted batched proof (1-fold case): the root is itself the
    /// terminal fold level and no steps follow.
    Terminal(TerminalLevelProofShape),
    /// Root-direct batched proof: one direct witness per claim.
    Direct {
        /// Per-claim direct witness shapes.
        witness_shapes: Vec<DirectWitnessShape>,
    },
}

/// Shape descriptor for deserializing a proof step without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkitaProofStepShape {
    /// Shape of an intermediate fold level.
    Intermediate(LevelProofShape),
    /// Shape of the terminal fold level.
    Terminal(TerminalLevelProofShape),
}

#[cfg(not(feature = "zk"))]
fn sumcheck_shape<F: FieldCore>(sc: &SumcheckProof<F>) -> SumcheckProofShape {
    sc.round_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.len())
        .collect()
}

#[cfg(feature = "zk")]
fn sumcheck_proof_masked_shape<F: FieldCore>(masks: &SumcheckProofMasked<F>) -> SumcheckProofShape {
    masks
        .masked_round_polys
        .iter()
        .map(|p| p.coeffs_except_linear_term.len())
        .collect()
}

#[cfg(not(feature = "zk"))]
fn eq_factored_sumcheck_shape<F: FieldCore>(
    sc: &EqFactoredSumcheckProof<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = sc
        .round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (sc.round_polys.len(), degree)
}

#[cfg(feature = "zk")]
fn eq_factored_sumcheck_proof_masked_shape<F: FieldCore>(
    masks: &EqFactoredSumcheckProofMasked<F>,
) -> EqFactoredSumcheckProofShape {
    let degree = masks
        .masked_round_polys
        .first()
        .map_or(0, |p| p.coeffs_except_linear_term.len());
    (masks.masked_round_polys.len(), degree)
}

fn level_proof_shape<F: FieldCore, L: FieldCore>(
    y_coeffs: usize,
    extension_opening_reduction: Option<&ExtensionOpeningReductionProof<L>>,
    v: &FlatRingVec<F>,
    stage1: &AkitaStage1Proof<L>,
    stage2: &AkitaStage2Proof<F, L>,
) -> LevelProofShape {
    LevelProofShape {
        y_ring_coeffs: y_coeffs,
        extension_opening_reduction: extension_opening_reduction
            .map(ExtensionOpeningReductionProof::shape),
        v_coeffs: v.coeff_len(),
        stage1_stages: stage1
            .stages
            .iter()
            .map(|stage| AkitaStage1StageShape {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: eq_factored_sumcheck_shape(&stage.sumcheck_proof),
                #[cfg(feature = "zk")]
                sumcheck_proof: eq_factored_sumcheck_proof_masked_shape(
                    &stage.sumcheck_proof_masked,
                ),
                child_claims: stage.child_claims.len(),
            })
            .collect(),
        #[cfg(not(feature = "zk"))]
        stage2_sumcheck_proof: sumcheck_shape(&stage2.sumcheck_proof),
        #[cfg(feature = "zk")]
        stage2_sumcheck_proof: sumcheck_proof_masked_shape(&stage2.sumcheck_proof_masked),
        next_commit_coeffs: stage2.next_w_commitment.coeff_len(),
    }
}

fn serialize_extension_opening_reduction<L, W>(
    reduction: Option<&ExtensionOpeningReductionProof<L>>,
    mut writer: W,
    compress: Compress,
) -> Result<(), SerializationError>
where
    L: FieldCore + AkitaSerialize,
    W: Write,
{
    if let Some(reduction) = reduction {
        for partial in &reduction.partials {
            partial.serialize_with_mode(&mut writer, compress)?;
        }
        #[cfg(not(feature = "zk"))]
        reduction
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        reduction
            .sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
    }
    Ok(())
}

fn extension_opening_reduction_serialized_size<L>(
    reduction: Option<&ExtensionOpeningReductionProof<L>>,
    compress: Compress,
) -> usize
where
    L: FieldCore + AkitaSerialize,
{
    reduction.map_or(0, |reduction| {
        reduction
            .partials
            .iter()
            .map(|partial| partial.serialized_size(compress))
            .sum::<usize>()
            + {
                #[cfg(not(feature = "zk"))]
                {
                    reduction.sumcheck.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    reduction.sumcheck_proof_masked.serialized_size(compress)
                }
            }
    })
}

fn deserialize_extension_opening_reduction<L, R>(
    mut reader: R,
    compress: Compress,
    validate: Validate,
    shape: Option<&ExtensionOpeningReductionShape>,
) -> Result<Option<ExtensionOpeningReductionProof<L>>, SerializationError>
where
    L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    R: Read,
{
    let Some(shape) = shape else {
        return Ok(None);
    };
    shape.check()?;
    let mut partials = Vec::new();
    reserve_shape_len(&mut partials, shape.partials)?;
    for _ in 0..shape.partials {
        partials.push(L::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?);
    }
    #[cfg(not(feature = "zk"))]
    let sumcheck =
        SumcheckProof::deserialize_with_mode(&mut reader, compress, validate, &shape.sumcheck)?;
    #[cfg(feature = "zk")]
    let sumcheck_proof_masked = SumcheckProofMasked::deserialize_with_mode(
        &mut reader,
        compress,
        validate,
        &shape.sumcheck,
    )?;
    Ok(Some(ExtensionOpeningReductionProof {
        partials,
        #[cfg(not(feature = "zk"))]
        sumcheck,
        #[cfg(feature = "zk")]
        sumcheck_proof_masked,
    }))
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
        serialize_extension_opening_reduction(
            self.extension_opening_reduction.as_ref(),
            &mut writer,
            compress,
        )?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage
                .sumcheck_proof
                .serialize_with_mode(&mut writer, compress)?;
            #[cfg(feature = "zk")]
            stage
                .sumcheck_proof_masked
                .serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(not(feature = "zk"))]
        self.stage2
            .sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.stage2
            .sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_eval()
            .serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        let base = self.y_ring.serialized_size(compress)
            + extension_opening_reduction_serialized_size(
                self.extension_opening_reduction.as_ref(),
                compress,
            )
            + self.v.serialized_size(compress);
        base + self
            .stage1
            .stages
            .iter()
            .map(|stage| {
                ({
                    #[cfg(not(feature = "zk"))]
                    {
                        stage.sumcheck_proof.serialized_size(compress)
                    }
                    #[cfg(feature = "zk")]
                    {
                        stage.sumcheck_proof_masked.serialized_size(compress)
                    }
                }) + stage
                    .child_claims
                    .iter()
                    .map(|claim| claim.serialized_size(compress))
                    .sum::<usize>()
            })
            .sum::<usize>()
            + self.stage1.s_claim.serialized_size(compress)
            + ({
                #[cfg(not(feature = "zk"))]
                {
                    self.stage2.sumcheck_proof.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    self.stage2.sumcheck_proof_masked.serialized_size(compress)
                }
            })
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval().serialized_size(compress)
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
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            #[cfg(not(feature = "zk"))]
            reduction.sumcheck.check()?;
            #[cfg(feature = "zk")]
            reduction.sumcheck_proof_masked.check()?;
        }
        self.v.check()?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage.sumcheck_proof.check()?;
            #[cfg(feature = "zk")]
            stage.sumcheck_proof_masked.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        #[cfg(not(feature = "zk"))]
        self.stage2.sumcheck_proof.check()?;
        #[cfg(feature = "zk")]
        self.stage2.sumcheck_proof_masked.check()?;
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval().check()
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
        ctx.check()?;
        let y_ring = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let extension_opening_reduction = deserialize_extension_opening_reduction(
            &mut reader,
            compress,
            validate,
            ctx.extension_opening_reduction.as_ref(),
        )?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
        let mut stage1_stages = Vec::new();
        reserve_shape_len(&mut stage1_stages, ctx.stage1_stages.len())?;
        for stage_shape in &ctx.stage1_stages {
            #[cfg(not(feature = "zk"))]
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            #[cfg(feature = "zk")]
            let sumcheck_proof_masked = EqFactoredSumcheckProofMasked::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            let mut child_claims = Vec::new();
            reserve_shape_len(&mut child_claims, stage_shape.child_claims)?;
            for _ in 0..stage_shape.child_claims {
                child_claims.push(L::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: sumcheck,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2 = AkitaStage2Proof {
            #[cfg(not(feature = "zk"))]
            sumcheck_proof: SumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck_proof,
            )?,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: SumcheckProofMasked::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck_proof,
            )?,
            next_w_commitment: FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            #[cfg(not(feature = "zk"))]
            next_w_eval: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
            #[cfg(feature = "zk")]
            next_w_eval_masked: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            y_ring,
            extension_opening_reduction,
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
    for TerminalLevelProof<F, L>
{
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_rings.serialize_with_mode(&mut writer, compress)?;
        serialize_extension_opening_reduction(
            self.extension_opening_reduction.as_ref(),
            &mut writer,
            compress,
        )?;
        #[cfg(not(feature = "zk"))]
        self.stage2_sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.stage2_sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
        self.final_witness
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_rings.serialized_size(compress)
            + extension_opening_reduction_serialized_size(
                self.extension_opening_reduction.as_ref(),
                compress,
            )
            + {
                #[cfg(not(feature = "zk"))]
                {
                    self.stage2_sumcheck.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    self.stage2_sumcheck_proof_masked.serialized_size(compress)
                }
            }
            + self.final_witness.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for TerminalLevelProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_rings.check()?;
        if self.y_rings.coeff_len() == 0 {
            return Err(SerializationError::InvalidData(
                "terminal level y_rings must contain at least one ring element".to_string(),
            ));
        }
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            #[cfg(not(feature = "zk"))]
            reduction.sumcheck.check()?;
            #[cfg(feature = "zk")]
            reduction.sumcheck_proof_masked.check()?;
        }
        #[cfg(not(feature = "zk"))]
        self.stage2_sumcheck.check()?;
        #[cfg(feature = "zk")]
        self.stage2_sumcheck_proof_masked.check()?;
        self.final_witness.check()
    }
}

impl<
        F: FieldCore + Valid + AkitaDeserialize<Context = ()>,
        L: FieldCore + Valid + AkitaDeserialize<Context = ()>,
    > AkitaDeserialize for TerminalLevelProof<F, L>
{
    type Context = TerminalLevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &TerminalLevelProofShape,
    ) -> Result<Self, SerializationError> {
        let y_rings = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_rings_coeffs,
        )?;
        let extension_opening_reduction = deserialize_extension_opening_reduction(
            &mut reader,
            compress,
            validate,
            ctx.extension_opening_reduction.as_ref(),
        )?;
        #[cfg(not(feature = "zk"))]
        let stage2_sumcheck = SumcheckProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck,
        )?;
        #[cfg(feature = "zk")]
        let stage2_sumcheck_proof_masked = SumcheckProofMasked::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.stage2_sumcheck,
        )?;
        let final_witness = DirectWitnessProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.final_witness,
        )?;
        let out = Self {
            y_rings,
            extension_opening_reduction,
            #[cfg(not(feature = "zk"))]
            stage2_sumcheck,
            #[cfg(feature = "zk")]
            stage2_sumcheck_proof_masked,
            final_witness,
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
            Self::Intermediate(level) => level.serialize_with_mode(&mut writer, compress),
            Self::Terminal(terminal) => terminal.serialize_with_mode(&mut writer, compress),
        }
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        match self {
            Self::Intermediate(level) => level.serialized_size(compress),
            Self::Terminal(terminal) => terminal.serialized_size(compress),
        }
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaProofStep<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => level.check(),
            Self::Terminal(terminal) => terminal.check(),
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
            AkitaProofStepShape::Intermediate(shape) => Self::Intermediate(
                AkitaLevelProof::deserialize_with_mode(&mut reader, compress, validate, shape)?,
            ),
            AkitaProofStepShape::Terminal(shape) => Self::Terminal(
                TerminalLevelProof::deserialize_with_mode(&mut reader, compress, validate, shape)?,
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
        serialize_extension_opening_reduction(
            self.extension_opening_reduction.as_ref(),
            &mut writer,
            compress,
        )?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage
                .sumcheck_proof
                .serialize_with_mode(&mut writer, compress)?;
            #[cfg(feature = "zk")]
            stage
                .sumcheck_proof_masked
                .serialize_with_mode(&mut writer, compress)?;
            for claim in &stage.child_claims {
                claim.serialize_with_mode(&mut writer, compress)?;
            }
        }
        self.stage1
            .s_claim
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(not(feature = "zk"))]
        self.stage2
            .sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        #[cfg(feature = "zk")]
        self.stage2
            .sumcheck_proof_masked
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2
            .next_w_eval()
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_rings.serialized_size(compress)
            + extension_opening_reduction_serialized_size(
                self.extension_opening_reduction.as_ref(),
                compress,
            )
            + self.v.serialized_size(compress)
            + self
                .stage1
                .stages
                .iter()
                .map(|stage| {
                    ({
                        #[cfg(not(feature = "zk"))]
                        {
                            stage.sumcheck_proof.serialized_size(compress)
                        }
                        #[cfg(feature = "zk")]
                        {
                            stage.sumcheck_proof_masked.serialized_size(compress)
                        }
                    }) + stage
                        .child_claims
                        .iter()
                        .map(|claim| claim.serialized_size(compress))
                        .sum::<usize>()
                })
                .sum::<usize>()
            + self.stage1.s_claim.serialized_size(compress)
            + ({
                #[cfg(not(feature = "zk"))]
                {
                    self.stage2.sumcheck_proof.serialized_size(compress)
                }
                #[cfg(feature = "zk")]
                {
                    self.stage2.sumcheck_proof_masked.serialized_size(compress)
                }
            })
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval().serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedFoldRoot<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_rings.check()?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.partials.check()?;
            #[cfg(not(feature = "zk"))]
            reduction.sumcheck.check()?;
            #[cfg(feature = "zk")]
            reduction.sumcheck_proof_masked.check()?;
        }
        self.v.check()?;
        for stage in &self.stage1.stages {
            #[cfg(not(feature = "zk"))]
            stage.sumcheck_proof.check()?;
            #[cfg(feature = "zk")]
            stage.sumcheck_proof_masked.check()?;
            stage.child_claims.check()?;
        }
        self.stage1.s_claim.check()?;
        #[cfg(not(feature = "zk"))]
        self.stage2.sumcheck_proof.check()?;
        #[cfg(feature = "zk")]
        self.stage2.sumcheck_proof_masked.check()?;
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval().check()
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
        ctx.check()?;
        let y_rings = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let extension_opening_reduction = deserialize_extension_opening_reduction(
            &mut reader,
            compress,
            validate,
            ctx.extension_opening_reduction.as_ref(),
        )?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
        let mut stage1_stages = Vec::new();
        reserve_shape_len(&mut stage1_stages, ctx.stage1_stages.len())?;
        for stage_shape in &ctx.stage1_stages {
            #[cfg(not(feature = "zk"))]
            let sumcheck = EqFactoredSumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            #[cfg(feature = "zk")]
            let sumcheck_proof_masked = EqFactoredSumcheckProofMasked::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &stage_shape.sumcheck_proof,
            )?;
            let mut child_claims = Vec::new();
            reserve_shape_len(&mut child_claims, stage_shape.child_claims)?;
            for _ in 0..stage_shape.child_claims {
                child_claims.push(L::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(AkitaStage1StageProof {
                #[cfg(not(feature = "zk"))]
                sumcheck_proof: sumcheck,
                #[cfg(feature = "zk")]
                sumcheck_proof_masked,
                child_claims,
            });
        }
        let stage1 = AkitaStage1Proof {
            stages: stage1_stages,
            s_claim: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2 = AkitaStage2Proof {
            #[cfg(not(feature = "zk"))]
            sumcheck_proof: SumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck_proof,
            )?,
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: SumcheckProofMasked::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck_proof,
            )?,
            next_w_commitment: FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            #[cfg(not(feature = "zk"))]
            next_w_eval: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
            #[cfg(feature = "zk")]
            next_w_eval_masked: L::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let out = Self {
            y_rings,
            extension_opening_reduction,
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
            Self::Terminal(terminal) => terminal.serialize_with_mode(&mut writer, compress),
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
            Self::Terminal(terminal) => terminal.serialized_size(compress),
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
            Self::Terminal(terminal) => terminal.check(),
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
        #[cfg(feature = "zk")]
        self.zk_hiding.serialize_with_mode(&mut writer, compress)?;
        self.root.serialize_with_mode(&mut writer, compress)?;
        for step in &self.steps {
            step.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        #[cfg(feature = "zk")]
        let zk_size = self.zk_hiding.serialized_size(compress);
        #[cfg(not(feature = "zk"))]
        let zk_size = 0;
        zk_size
            + self.root.serialized_size(compress)
            + self
                .steps
                .iter()
                .map(|step| step.serialized_size(compress))
                .sum::<usize>()
    }
}

impl<F: FieldCore + Valid, L: FieldCore + Valid> Valid for AkitaBatchedProof<F, L> {
    fn check(&self) -> Result<(), SerializationError> {
        #[cfg(feature = "zk")]
        self.zk_hiding.check()?;
        self.root.check()?;
        for step in &self.steps {
            step.check()?;
        }
        match &self.root {
            AkitaBatchedRootProof::Fold(_) => {
                let Some(AkitaProofStep::Terminal(_)) = self.steps.last() else {
                    return Err(SerializationError::InvalidData(
                        "fold-rooted batched Akita proof must terminate with a terminal step"
                            .to_string(),
                    ));
                };
                if self.steps[..self.steps.len().saturating_sub(1)]
                    .iter()
                    .any(|step| !matches!(step, AkitaProofStep::Intermediate(_)))
                {
                    return Err(SerializationError::InvalidData(
                        "fold-rooted batched Akita proof may only contain intermediate steps before the terminal step"
                            .to_string(),
                    ));
                }
                // Headerless validity cannot infer the ring dimension from
                // `y_ring`: multipoint levels store one D-sized ring per
                // public row. Schedule-shaped deserialization and verifier
                // replay own the cross-level dimension checks.
            }
            AkitaBatchedRootProof::Terminal(_) => {
                if !self.steps.is_empty() {
                    return Err(SerializationError::InvalidData(
                        "terminal-rooted batched proof must not carry recursive-suffix steps"
                            .to_string(),
                    ));
                }
            }
            AkitaBatchedRootProof::Direct { .. } => {
                #[cfg(feature = "zk")]
                if !self.zk_hiding.is_empty() {
                    return Err(SerializationError::InvalidData(
                        "root-direct ZK hiding payload must be empty".to_string(),
                    ));
                }
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
        ctx.check()?;
        #[cfg(feature = "zk")]
        let zk_hiding =
            ZkHidingProof::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
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
                let mut steps = Vec::new();
                reserve_shape_len(&mut steps, step_shapes.len())?;
                for shape in step_shapes {
                    steps.push(AkitaProofStep::deserialize_with_mode(
                        &mut reader,
                        compress,
                        validate,
                        shape,
                    )?);
                }
                Self {
                    #[cfg(feature = "zk")]
                    zk_hiding,
                    root: AkitaBatchedRootProof::Fold(fold),
                    steps,
                }
            }
            AkitaBatchedProofShape::Terminal(terminal_shape) => {
                let terminal = TerminalLevelProof::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    terminal_shape,
                )?;
                Self {
                    #[cfg(feature = "zk")]
                    zk_hiding,
                    root: AkitaBatchedRootProof::Terminal(terminal),
                    steps: Vec::new(),
                }
            }
            AkitaBatchedProofShape::Direct { witness_shapes } => {
                let mut witnesses = Vec::new();
                reserve_shape_len(&mut witnesses, witness_shapes.len())?;
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
                    #[cfg(feature = "zk")]
                    zk_hiding,
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

// === Headerless shape (de)serialization ===
//
// These impls let callers bundle proof shapes alongside proofs (e.g. when
// shipping verifier inputs to a Jolt guest program), so that the proof can be
// deserialized in environments that don't reconstruct a `Schedule` first.

impl Valid for AkitaStage1StageShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.sumcheck_proof.0)?;
        checked_shape_len(self.sumcheck_proof.1)?;
        checked_shape_len(self.child_claims)?;
        Ok(())
    }
}

impl AkitaSerialize for AkitaStage1StageShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let (rounds, degree) = self.sumcheck_proof;
        rounds.serialize_with_mode(&mut writer, compress)?;
        degree.serialize_with_mode(&mut writer, compress)?;
        self.child_claims
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let (rounds, degree) = self.sumcheck_proof;
        rounds.serialized_size(compress)
            + degree.serialized_size(compress)
            + self.child_claims.serialized_size(compress)
    }
}

impl AkitaDeserialize for AkitaStage1StageShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let rounds = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let degree = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let child_claims = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            sumcheck_proof: (rounds, degree),
            child_claims,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for LevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.y_ring_coeffs)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        checked_shape_len(self.v_coeffs)?;
        checked_shape_len(self.stage1_stages.len())?;
        self.stage1_stages.check()?;
        checked_shape_len(self.stage2_sumcheck_proof.len())?;
        for &degree in &self.stage2_sumcheck_proof {
            checked_shape_len(degree)?;
        }
        checked_shape_len(self.next_commit_coeffs)?;
        Ok(())
    }
}

impl AkitaSerialize for LevelProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        self.extension_opening_reduction
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction
                .partials
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.v_coeffs.serialize_with_mode(&mut writer, compress)?;
        self.stage1_stages
            .serialize_with_mode(&mut writer, compress)?;
        self.stage2_sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.next_commit_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let reduction_size = true.serialized_size(compress)
            + self
                .extension_opening_reduction
                .as_ref()
                .map_or(0, |reduction| {
                    reduction.partials.serialized_size(compress)
                        + reduction.sumcheck.serialized_size(compress)
                });
        self.y_ring_coeffs.serialized_size(compress)
            + reduction_size
            + self.v_coeffs.serialized_size(compress)
            + self.stage1_stages.serialized_size(compress)
            + self.stage2_sumcheck_proof.serialized_size(compress)
            + self.next_commit_coeffs.serialized_size(compress)
    }
}

impl AkitaDeserialize for LevelProofShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let y_ring_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let has_extension_opening_reduction =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let extension_opening_reduction = if has_extension_opening_reduction {
            let partials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let sumcheck =
                SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
            Some(ExtensionOpeningReductionShape { partials, sumcheck })
        } else {
            None
        };
        let v_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let stage1_stages = Vec::<AkitaStage1StageShape>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let stage2_sumcheck =
            SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let next_commit_coeffs =
            usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            y_ring_coeffs,
            extension_opening_reduction,
            v_coeffs,
            stage1_stages,
            stage2_sumcheck_proof: stage2_sumcheck,
            next_commit_coeffs,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for DirectWitnessShape {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits((num_elems, bits_per_elem)) => {
                if *bits_per_elem == 0 || *bits_per_elem > 6 {
                    return Err(SerializationError::InvalidData(
                        "bits_per_elem out of range".to_string(),
                    ));
                }
                checked_shape_len(*num_elems)?;
            }
            Self::FieldElements(coeff_len) => checked_shape_len(*coeff_len)?,
        }
        Ok(())
    }
}

impl AkitaSerialize for DirectWitnessShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::PackedDigits((num_elems, bits_per_elem)) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                num_elems.serialize_with_mode(&mut writer, compress)?;
                bits_per_elem.serialize_with_mode(&mut writer, compress)?;
            }
            Self::FieldElements(coeff_len) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                coeff_len.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let tag = 1usize;
        match self {
            Self::PackedDigits((num_elems, bits_per_elem)) => {
                tag + num_elems.serialized_size(compress) + bits_per_elem.serialized_size(compress)
            }
            Self::FieldElements(coeff_len) => tag + coeff_len.serialized_size(compress),
        }
    }
}

impl AkitaDeserialize for DirectWitnessShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = match tag {
            0 => {
                let num_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
                let bits_per_elem =
                    u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
                Self::PackedDigits((num_elems, bits_per_elem))
            }
            1 => {
                let coeff_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
                Self::FieldElements(coeff_len)
            }
            other => {
                return Err(SerializationError::InvalidData(format!(
                    "unknown DirectWitnessShape tag {other}"
                )))
            }
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for TerminalLevelProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        checked_shape_len(self.y_rings_coeffs)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction.check()?;
        }
        checked_shape_len(self.stage2_sumcheck.len())?;
        for &degree in &self.stage2_sumcheck {
            checked_shape_len(degree)?;
        }
        self.final_witness.check()?;
        Ok(())
    }
}

impl AkitaSerialize for TerminalLevelProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_rings_coeffs
            .serialize_with_mode(&mut writer, compress)?;
        self.extension_opening_reduction
            .is_some()
            .serialize_with_mode(&mut writer, compress)?;
        if let Some(reduction) = &self.extension_opening_reduction {
            reduction
                .partials
                .serialize_with_mode(&mut writer, compress)?;
            reduction
                .sumcheck
                .serialize_with_mode(&mut writer, compress)?;
        }
        self.stage2_sumcheck
            .serialize_with_mode(&mut writer, compress)?;
        self.final_witness
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        let reduction_size = true.serialized_size(compress)
            + self
                .extension_opening_reduction
                .as_ref()
                .map_or(0, |reduction| {
                    reduction.partials.serialized_size(compress)
                        + reduction.sumcheck.serialized_size(compress)
                });
        self.y_rings_coeffs.serialized_size(compress)
            + reduction_size
            + self.stage2_sumcheck.serialized_size(compress)
            + self.final_witness.serialized_size(compress)
    }
}

impl AkitaDeserialize for TerminalLevelProofShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let y_rings_coeffs = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let has_extension_opening_reduction =
            bool::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let extension_opening_reduction = if has_extension_opening_reduction {
            let partials = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            let sumcheck =
                SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
            Some(ExtensionOpeningReductionShape { partials, sumcheck })
        } else {
            None
        };
        let stage2_sumcheck =
            SumcheckProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let final_witness =
            DirectWitnessShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
        Ok(Self {
            y_rings_coeffs,
            extension_opening_reduction,
            stage2_sumcheck,
            final_witness,
        })
    }
}

impl Valid for AkitaProofStepShape {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => level.check()?,
            Self::Terminal(terminal) => terminal.check()?,
        }
        Ok(())
    }
}

impl AkitaSerialize for AkitaProofStepShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Intermediate(level) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                level.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Terminal(terminal) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                terminal.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Intermediate(level) => level.serialized_size(compress),
            Self::Terminal(terminal) => terminal.serialized_size(compress),
        }
    }
}

impl AkitaDeserialize for AkitaProofStepShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = match tag {
            0 => Self::Intermediate(LevelProofShape::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?),
            1 => Self::Terminal(TerminalLevelProofShape::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?),
            other => {
                return Err(SerializationError::InvalidData(format!(
                    "unknown AkitaProofStepShape tag {other}"
                )))
            }
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl Valid for AkitaBatchedProofShape {
    fn check(&self) -> Result<(), SerializationError> {
        match self {
            Self::Fold {
                root_shape,
                step_shapes,
            } => {
                root_shape.check()?;
                checked_shape_len(step_shapes.len())?;
                step_shapes.check()?;
            }
            Self::Terminal(terminal) => {
                terminal.check()?;
            }
            Self::Direct { witness_shapes } => {
                checked_shape_len(witness_shapes.len())?;
                witness_shapes.check()?;
            }
        }
        Ok(())
    }
}

impl AkitaSerialize for AkitaBatchedProofShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        match self {
            Self::Fold {
                root_shape,
                step_shapes,
            } => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                root_shape.serialize_with_mode(&mut writer, compress)?;
                step_shapes.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Terminal(terminal_shape) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                terminal_shape.serialize_with_mode(&mut writer, compress)?;
            }
            Self::Direct { witness_shapes } => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                witness_shapes.serialize_with_mode(&mut writer, compress)?;
            }
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + match self {
            Self::Fold {
                root_shape,
                step_shapes,
            } => root_shape.serialized_size(compress) + step_shapes.serialized_size(compress),
            Self::Terminal(terminal_shape) => terminal_shape.serialized_size(compress),
            Self::Direct { witness_shapes } => witness_shapes.serialized_size(compress),
        }
    }
}

impl AkitaDeserialize for AkitaBatchedProofShape {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate, &())?;
        match tag {
            0 => {
                let root_shape =
                    LevelProofShape::deserialize_with_mode(&mut reader, compress, validate, &())?;
                let step_shapes = Vec::<AkitaProofStepShape>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?;
                let out = Self::Fold {
                    root_shape,
                    step_shapes,
                };
                if matches!(validate, Validate::Yes) {
                    out.check()?;
                }
                Ok(out)
            }
            1 => {
                let terminal_shape = TerminalLevelProofShape::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?;
                let out = Self::Terminal(terminal_shape);
                if matches!(validate, Validate::Yes) {
                    out.check()?;
                }
                Ok(out)
            }
            2 => {
                let witness_shapes = Vec::<DirectWitnessShape>::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?;
                let out = Self::Direct { witness_shapes };
                if matches!(validate, Validate::Yes) {
                    out.check()?;
                }
                Ok(out)
            }
            other => Err(SerializationError::InvalidData(format!(
                "unknown AkitaBatchedProofShape tag {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "zk"))]
    use akita_algebra::CompressedUniPoly;
    use akita_field::Prime128Offset275;
    use akita_serialization::Valid;
    #[cfg(not(feature = "zk"))]
    use akita_sumcheck::SumcheckProof;

    type F = Prime128Offset275;

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
        assert_eq!(
            packed.to_field_elems::<Prime128Offset275>().unwrap(),
            expected_field
        );
    }

    #[test]
    fn packed_digits_reject_bits_above_six() {
        let packed = PackedDigits {
            num_elems: 1,
            bits_per_elem: 7,
            data: vec![0],
        };

        assert!(packed.check().is_err());
        assert_eq!(packed.digit_at(0), None);
        assert!(packed.to_field_elems::<Prime128Offset275>().is_err());
    }

    #[test]
    fn packed_digits_malformed_buffer_returns_error() {
        let packed = PackedDigits {
            num_elems: 4,
            bits_per_elem: 6,
            data: vec![0],
        };

        assert!(packed.check().is_err());
        assert_eq!(packed.digit_at(3), None);
        assert!(packed.to_field_elems::<Prime128Offset275>().is_err());
    }

    #[test]
    fn direct_witness_shape_rejects_oversized_allocations() {
        let err = DirectWitnessShape::FieldElements(DEFAULT_MAX_SEQUENCE_LEN + 1)
            .check()
            .unwrap_err();
        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { .. }
        ));
    }

    #[test]
    fn packed_digits_deserialization_rejects_shape_before_allocation() {
        let ctx = (DEFAULT_MAX_SEQUENCE_LEN + 1, 6);

        let err =
            PackedDigits::deserialize_compressed(&[][..], &ctx).expect_err("shape exceeds cap");
        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { .. }
        ));
    }

    #[test]
    fn flat_ring_vec_deserialization_rejects_shape_before_allocation() {
        let coeffs = DEFAULT_MAX_SEQUENCE_LEN + 1;

        let err = FlatRingVec::<Prime128Offset275>::deserialize_compressed(&[][..], &coeffs)
            .expect_err("shape exceeds cap");
        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { .. }
        ));
    }

    #[test]
    fn flat_ring_vec_checked_decoders_reject_zero_dimension() {
        let flat = FlatRingVec::<Prime128Offset275>::from_coeffs(vec![]);

        assert!(!flat.can_decode_single(0));
        assert!(!flat.can_decode_vec(0));
        assert!(flat.try_to_single::<0>().is_err());
        assert!(flat.try_to_vec::<0>().is_err());
        assert!(flat.as_ring_slice::<0>().is_err());
        assert!(flat.try_to_ring_commitment::<0>().is_err());
    }

    #[test]
    fn batched_proof_shape_validation_recurses_into_witness_shapes() {
        let shape = AkitaBatchedProofShape::Direct {
            witness_shapes: vec![DirectWitnessShape::FieldElements(
                DEFAULT_MAX_SEQUENCE_LEN + 1,
            )],
        };

        let err = shape.check().unwrap_err();
        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { .. }
        ));
    }

    #[test]
    fn level_shape_validation_checks_extension_opening_reduction() {
        let oversized = LevelProofShape {
            y_ring_coeffs: 1,
            extension_opening_reduction: Some(ExtensionOpeningReductionShape::standard(
                DEFAULT_MAX_SEQUENCE_LEN + 1,
                1,
            )),
            v_coeffs: 1,
            stage1_stages: Vec::new(),
            stage2_sumcheck_proof: Vec::new(),
            next_commit_coeffs: 1,
        };

        let err = oversized.check().unwrap_err();
        assert!(matches!(
            err,
            SerializationError::LengthLimitExceeded { .. }
        ));

        let wrong_degree = LevelProofShape {
            extension_opening_reduction: Some(ExtensionOpeningReductionShape {
                partials: 1,
                sumcheck: vec![EXTENSION_OPENING_REDUCTION_DEGREE + 1],
            }),
            ..oversized
        };

        let err = wrong_degree.check().unwrap_err();
        assert!(matches!(err, SerializationError::InvalidData(_)));
    }

    fn tiny_stage1() -> AkitaStage1Proof<F> {
        AkitaStage1Proof {
            stages: Vec::new(),
            s_claim: F::zero(),
        }
    }

    fn tiny_stage2<const D: usize>() -> AkitaStage2Proof<F, F> {
        AkitaStage2Proof {
            #[cfg(not(feature = "zk"))]
            sumcheck_proof: SumcheckProof {
                round_polys: Vec::new(),
            },
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: SumcheckProofMasked {
                masked_round_polys: Vec::new(),
            },
            next_w_commitment: FlatRingVec::from_ring_elems(&[CyclotomicRing::<F, D>::zero()])
                .into_compact(),
            #[cfg(not(feature = "zk"))]
            next_w_eval: F::zero(),
            #[cfg(feature = "zk")]
            next_w_eval_masked: F::zero(),
        }
    }

    fn tiny_reduction() -> ExtensionOpeningReductionProof<F> {
        ExtensionOpeningReductionProof {
            partials: vec![F::zero(), F::one()],
            #[cfg(not(feature = "zk"))]
            sumcheck: SumcheckProof {
                round_polys: vec![CompressedUniPoly {
                    coeffs_except_linear_term: vec![F::zero(), F::one()],
                }],
            },
            #[cfg(feature = "zk")]
            sumcheck_proof_masked: SumcheckProofMasked {
                masked_round_polys: Vec::new(),
            },
        }
    }

    #[test]
    fn extension_opening_reduction_none_is_zero_proof_wire_bytes() {
        const D: usize = 8;
        let without_reduction = AkitaLevelProof::new::<D>(
            CyclotomicRing::<F, D>::zero(),
            vec![CyclotomicRing::<F, D>::zero()],
            tiny_stage1(),
            tiny_stage2::<D>(),
        );
        assert!(without_reduction.extension_opening_reduction.is_none());
        assert!(without_reduction
            .shape()
            .extension_opening_reduction
            .is_none());

        let mut bytes = Vec::new();
        without_reduction
            .serialize_uncompressed(&mut bytes)
            .expect("serialize proof without extension-opening reduction");
        assert_eq!(bytes.len(), without_reduction.serialized_size(Compress::No));

        let decoded =
            AkitaLevelProof::<F, F>::deserialize_uncompressed(&*bytes, &without_reduction.shape())
                .expect("deserialize proof without extension-opening reduction");
        assert!(decoded.extension_opening_reduction.is_none());
        assert_eq!(decoded, without_reduction);

        let with_reduction =
            AkitaLevelProof::new_two_stage_many_with_extension_opening_reduction::<D>(
                vec![CyclotomicRing::<F, D>::zero()],
                Some(tiny_reduction()),
                vec![CyclotomicRing::<F, D>::zero()],
                tiny_stage1(),
                #[cfg(not(feature = "zk"))]
                SumcheckProof {
                    round_polys: Vec::new(),
                },
                #[cfg(feature = "zk")]
                SumcheckProofMasked {
                    masked_round_polys: Vec::new(),
                },
                FlatRingVec::from_ring_elems(&[CyclotomicRing::<F, D>::zero()]).into_compact(),
                F::zero(),
            );
        let reduction_bytes = extension_opening_reduction_serialized_size(
            with_reduction.extension_opening_reduction.as_ref(),
            Compress::No,
        );
        assert!(reduction_bytes > 0);
        assert_eq!(
            with_reduction.serialized_size(Compress::No)
                - without_reduction.serialized_size(Compress::No),
            reduction_bytes
        );

        let mut bytes_with_reduction = Vec::new();
        with_reduction
            .serialize_uncompressed(&mut bytes_with_reduction)
            .expect("serialize proof with extension-opening reduction");
        let decoded_with_reduction = AkitaLevelProof::<F, F>::deserialize_uncompressed(
            &*bytes_with_reduction,
            &with_reduction.shape(),
        )
        .expect("deserialize proof with extension-opening reduction");
        assert_eq!(decoded_with_reduction, with_reduction);
    }

    #[cfg(not(feature = "zk"))]
    fn tiny_terminal_stage2() -> SumcheckProof<F> {
        SumcheckProof {
            round_polys: Vec::new(),
        }
    }

    #[cfg(feature = "zk")]
    fn tiny_terminal_stage2_masked() -> SumcheckProofMasked<F> {
        SumcheckProofMasked {
            masked_round_polys: Vec::new(),
        }
    }

    #[test]
    fn terminal_level_proof_serde_round_trip() {
        const D: usize = 8;
        let final_witness = DirectWitnessProof::PackedDigits(
            PackedDigits::from_i8_digits_with_min_bits(&[1i8, -1, 0, 2], 3),
        );

        let without_reduction = TerminalLevelProof::new_with_extension_opening_reduction::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            None,
            #[cfg(not(feature = "zk"))]
            tiny_terminal_stage2(),
            #[cfg(feature = "zk")]
            tiny_terminal_stage2_masked(),
            final_witness.clone(),
        );
        assert!(without_reduction.extension_opening_reduction.is_none());
        assert!(without_reduction
            .shape()
            .extension_opening_reduction
            .is_none());

        let mut bytes = Vec::new();
        without_reduction
            .serialize_uncompressed(&mut bytes)
            .expect("serialize terminal proof without extension-opening reduction");
        assert_eq!(bytes.len(), without_reduction.serialized_size(Compress::No));

        let decoded = TerminalLevelProof::<F, F>::deserialize_uncompressed(
            &*bytes,
            &without_reduction.shape(),
        )
        .expect("deserialize terminal proof without extension-opening reduction");
        assert_eq!(decoded, without_reduction);

        let with_reduction = TerminalLevelProof::new_with_extension_opening_reduction::<D>(
            vec![CyclotomicRing::<F, D>::zero()],
            Some(tiny_reduction()),
            #[cfg(not(feature = "zk"))]
            tiny_terminal_stage2(),
            #[cfg(feature = "zk")]
            tiny_terminal_stage2_masked(),
            final_witness,
        );
        let mut bytes_with_reduction = Vec::new();
        with_reduction
            .serialize_uncompressed(&mut bytes_with_reduction)
            .expect("serialize terminal proof with extension-opening reduction");
        let decoded_with_reduction = TerminalLevelProof::<F, F>::deserialize_uncompressed(
            &*bytes_with_reduction,
            &with_reduction.shape(),
        )
        .expect("deserialize terminal proof with extension-opening reduction");
        assert_eq!(decoded_with_reduction, with_reduction);

        with_reduction
            .shape()
            .check()
            .expect("terminal shape with reduction passes Valid::check()");
    }
}
