//! Proof structures for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::{Compress, SerializationError};
use crate::primitives::serialization::{Valid, Validate};
use crate::protocol::commitment::RingCommitment;
use crate::protocol::sumcheck::SumcheckProof;
use crate::{CanonicalField, FieldCore, FromSmallInt, HachiDeserialize, HachiSerialize};
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

/// Signed-digit lookup table used by proof tails and recursive commitments.
pub(crate) struct DigitLut<F> {
    table: Vec<F>,
    half_b: i8,
}

impl<F: FieldCore + FromSmallInt> DigitLut<F> {
    #[inline]
    pub(crate) fn new(log_basis: u32) -> Self {
        assert!(log_basis > 0 && log_basis <= 5, "log_basis out of range");
        let half_b = 1i8 << (log_basis - 1);
        let table = (-(half_b as i16)..(half_b as i16))
            .map(|digit| F::from_i64(digit as i64))
            .collect();
        Self { table, half_b }
    }

    #[inline(always)]
    pub(crate) fn get(&self, d: i8) -> F {
        self.table[(d + self.half_b) as usize]
    }
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
        assert!(log_basis > 0 && log_basis <= 5, "log_basis out of range");
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

    /// Unpack to field elements.
    pub fn to_field_elems<F: FieldCore + FromSmallInt>(&self) -> Vec<F> {
        let bits = self.bits_per_elem as usize;
        let mask = (1u8 << bits) - 1;
        let sign_bit = 1u8 << (bits - 1);

        let mut out = Vec::with_capacity(self.num_elems);
        for i in 0..self.num_elems {
            let bit_offset = i * bits;
            let byte_idx = bit_offset / 8;
            let bit_idx = bit_offset % 8;
            let mut raw = (self.data[byte_idx] >> bit_idx) & mask;
            if bit_idx + bits > 8 {
                raw |= (self.data[byte_idx + 1] << (8 - bit_idx)) & mask;
            }
            let signed = if raw & sign_bit != 0 {
                raw as i8 | !(mask as i8)
            } else {
                raw as i8
            };
            out.push(F::from_i64(signed as i64));
        }
        out
    }

    /// Number of packed data bytes.
    pub fn packed_byte_len(&self) -> usize {
        self.data.len()
    }
}

impl HachiSerialize for PackedDigits {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.num_elems as u64).serialize_with_mode(&mut writer, compress)?;
        (self.bits_per_elem as u8).serialize_with_mode(&mut writer, compress)?;
        writer.write_all(&self.data)?;
        Ok(())
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        8 + 1 + self.data.len()
    }
}

impl Valid for PackedDigits {
    fn check(&self) -> Result<(), SerializationError> {
        if self.bits_per_elem == 0 || self.bits_per_elem > 7 {
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

impl HachiDeserialize for PackedDigits {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let num_elems = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let bits_per_elem = u8::deserialize_with_mode(&mut reader, compress, validate)? as u32;
        let num_bytes = (num_elems * bits_per_elem as usize).div_ceil(8);
        let mut data = vec![0u8; num_bytes];
        reader.read_exact(&mut data)?;
        let out = Self {
            num_elems,
            bits_per_elem,
            data,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// D-erased storage for a sequence of ring elements as raw field-element
/// coefficients.
///
/// Each ring element of dimension `ring_dim` is stored as `ring_dim`
/// contiguous field elements in `coeffs`. The total number of ring elements
/// is `coeffs.len() / ring_dim`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatRingVec<F> {
    coeffs: Vec<F>,
    ring_dim: usize,
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

    /// Wrap a `RingCommitment`.
    pub fn from_commitment<const D: usize>(c: &RingCommitment<F, D>) -> Self {
        Self::from_ring_elems(&c.u)
    }

    /// Ring dimension (number of field-element coefficients per ring element).
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Number of ring elements stored.
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

    /// Reconstruct a single ring element.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim` or `count() != 1`.
    pub fn to_single<const D: usize>(&self) -> CyclotomicRing<F, D> {
        assert_eq!(D, self.ring_dim, "D mismatch in to_single");
        assert_eq!(self.count(), 1, "expected exactly one ring element");
        CyclotomicRing::from_slice(&self.coeffs)
    }

    /// Reconstruct a single ring element, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored ring dimension or
    /// element count does not match `D`.
    pub fn try_to_single<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, HachiError> {
        if self.ring_dim != D || self.coeffs.len() != D {
            return Err(HachiError::InvalidProof);
        }
        Ok(CyclotomicRing::from_slice(&self.coeffs))
    }

    /// Reconstruct a vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim`.
    pub fn to_vec<const D: usize>(&self) -> Vec<CyclotomicRing<F, D>> {
        assert_eq!(D, self.ring_dim, "D mismatch in to_vec");
        self.coeffs
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect()
    }

    /// Reconstruct a vector of ring elements, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored ring dimension does
    /// not match `D` or the coefficient buffer is not an exact multiple of `D`.
    pub fn try_to_vec<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
        if self.ring_dim != D || self.coeffs.len() % D != 0 {
            return Err(HachiError::InvalidProof);
        }
        Ok(self
            .coeffs
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect())
    }

    /// Reconstruct a `RingCommitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim`.
    pub fn to_ring_commitment<const D: usize>(&self) -> RingCommitment<F, D> {
        RingCommitment { u: self.to_vec() }
    }

    /// Reconstruct a `RingCommitment`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub fn try_to_ring_commitment<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, HachiError> {
        Ok(RingCommitment {
            u: self.try_to_vec()?,
        })
    }
}

impl<F: FieldCore> HachiSerialize for FlatRingVec<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.ring_dim as u32).serialize_with_mode(&mut writer, compress)?;
        self.coeffs.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        4 + self.coeffs.serialized_size(compress)
    }
}

impl<F: FieldCore> Valid for FlatRingVec<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.ring_dim == 0 {
            return Err(SerializationError::InvalidData(
                "ring_dim must be > 0".to_string(),
            ));
        }
        if self.coeffs.len() % self.ring_dim != 0 {
            return Err(SerializationError::InvalidData(
                "coeffs length not a multiple of ring_dim".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for FlatRingVec<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let ring_dim = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let coeffs = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self { coeffs, ring_dim };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// D-erased commitment hint for cross-level storage.
///
/// Stores the decomposed inner-opening digit blocks (formerly `t_hat`) as a
/// flat `Vec<i8>` with metadata about block sizes and ring dimension. Convert
/// to/from the typed
/// [`HachiCommitmentHint`] via [`from_typed`](Self::from_typed) and
/// [`to_typed`](Self::to_typed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatCommitmentHint {
    data: Vec<i8>,
    block_sizes: Vec<usize>,
    ring_dim: usize,
}

impl FlatCommitmentHint {
    /// Convert from a typed hint, consuming it.
    pub fn from_typed<F: FieldCore, const D: usize>(hint: HachiCommitmentHint<F, D>) -> Self {
        let block_sizes: Vec<usize> = hint.inner_opening_digits.iter().map(|b| b.len()).collect();
        let total_planes: usize = block_sizes.iter().sum();
        let mut data = Vec::with_capacity(total_planes * D);
        for block in &hint.inner_opening_digits {
            for plane in block {
                data.extend_from_slice(plane);
            }
        }
        Self {
            data,
            block_sizes,
            ring_dim: D,
        }
    }

    /// Reconstruct a typed hint.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim`.
    pub fn to_typed<F: FieldCore, const D: usize>(&self) -> HachiCommitmentHint<F, D> {
        assert_eq!(D, self.ring_dim, "D mismatch in to_typed");
        let mut inner_opening_digits = Vec::with_capacity(self.block_sizes.len());
        let mut offset = 0;
        for &block_size in &self.block_sizes {
            let mut block = Vec::with_capacity(block_size);
            for _ in 0..block_size {
                let mut plane = [0i8; D];
                plane.copy_from_slice(&self.data[offset..offset + D]);
                offset += D;
                block.push(plane);
            }
            inner_opening_digits.push(block);
        }
        HachiCommitmentHint::new(inner_opening_digits)
    }

    /// Reconstruct a typed hint and eagerly recompose the cached `t` rows in
    /// the same pass, avoiding a second traversal through `inner_opening_digits`.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_digits_open` is zero or if any digit block
    /// length is not divisible by `num_digits_open`.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim`.
    pub fn to_typed_with_t<F: CanonicalField, const D: usize>(
        &self,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<HachiCommitmentHint<F, D>, HachiError> {
        assert_eq!(D, self.ring_dim, "D mismatch in to_typed_with_t");
        if num_digits_open == 0 {
            return Err(HachiError::InvalidSetup(
                "num_digits_open must be nonzero when reconstructing commitment hint".to_string(),
            ));
        }
        let mut inner_opening_digits = Vec::with_capacity(self.block_sizes.len());
        let mut t = Vec::with_capacity(self.block_sizes.len());
        let mut offset = 0;
        for &block_size in &self.block_sizes {
            if block_size % num_digits_open != 0 {
                return Err(HachiError::InvalidSetup(format!(
                    "inner-opening digit block has {block_size} planes, expected a multiple of num_digits_open={num_digits_open}",
                )));
            }
            let mut block = Vec::with_capacity(block_size);
            for _ in 0..block_size {
                let mut plane = [0i8; D];
                plane.copy_from_slice(&self.data[offset..offset + D]);
                offset += D;
                block.push(plane);
            }
            let t_block = block
                .chunks(num_digits_open)
                .map(|digits| CyclotomicRing::gadget_recompose_pow2_i8(digits, log_basis))
                .collect();
            inner_opening_digits.push(block);
            t.push(t_block);
        }
        Ok(HachiCommitmentHint::with_t(inner_opening_digits, t))
    }

    /// Ring dimension stored in this hint.
    pub fn ring_dim(&self) -> usize {
        self.ring_dim
    }

    /// Empty hint (verifier side, where hint data is not available).
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            block_sizes: Vec::new(),
            ring_dim: 0,
        }
    }
}

/// Prover-side hint produced at commitment time.
///
/// Contains the decomposed inner-opening digits (formerly `t_hat`) needed by
/// the ring-switch step of the prover. The polynomial itself (ring
/// coefficients) is passed separately to `prove` via `HachiPolyOps`.
#[derive(Debug, Clone)]
pub struct HachiCommitmentHint<F: FieldCore, const D: usize> {
    /// Decomposed inner-opening digit blocks from the commitment phase as i8
    /// digit planes (formerly `t_hat`).
    pub inner_opening_digits: Vec<Vec<[i8; D]>>,
    /// Optional recomposed `t_i` rows cached for prover-side A-row work.
    t: Option<Vec<Vec<CyclotomicRing<F, D>>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> HachiCommitmentHint<F, D> {
    /// Construct a new hint from i8 digit plane blocks.
    pub fn new(inner_opening_digits: Vec<Vec<[i8; D]>>) -> Self {
        Self {
            inner_opening_digits,
            t: None,
            _marker: PhantomData,
        }
    }

    /// Construct a hint that also preserves the undecomposed `t_i` rows.
    pub fn with_t(
        inner_opening_digits: Vec<Vec<[i8; D]>>,
        t: Vec<Vec<CyclotomicRing<F, D>>>,
    ) -> Self {
        Self {
            inner_opening_digits,
            t: Some(t),
            _marker: PhantomData,
        }
    }

    /// Get the optional recomposed `t_i` rows.
    pub fn t(&self) -> Option<&[Vec<CyclotomicRing<F, D>>]> {
        self.t.as_deref()
    }

    /// Populate the recomposed `t_i` rows from the inner-opening digits when
    /// they are absent.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_digits_open` is zero or if any inner-opening
    /// digit block length is not a multiple of `num_digits_open`.
    pub fn ensure_t_recomposed(
        &mut self,
        num_digits_open: usize,
        log_basis: u32,
    ) -> Result<(), HachiError>
    where
        F: CanonicalField,
    {
        if self.t.is_some() {
            return Ok(());
        }
        if num_digits_open == 0 {
            return Err(HachiError::InvalidSetup(
                "num_digits_open must be nonzero when recomposing inner-opening digits".to_string(),
            ));
        }

        let t = self
            .inner_opening_digits
            .iter()
            .map(|block| {
                if block.len() % num_digits_open != 0 {
                    return Err(HachiError::InvalidSetup(format!(
                        "inner-opening digit block has {} planes, expected a multiple of num_digits_open={num_digits_open}",
                        block.len()
                    )));
                }
                Ok(block
                    .chunks(num_digits_open)
                    .map(|digits| CyclotomicRing::gadget_recompose_pow2_i8(digits, log_basis))
                    .collect())
            })
            .collect::<Result<Vec<Vec<CyclotomicRing<F, D>>>, HachiError>>()?;
        self.t = Some(t);
        Ok(())
    }
}

impl<F: FieldCore, const D: usize> PartialEq for HachiCommitmentHint<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.inner_opening_digits == other.inner_opening_digits
    }
}

impl<F: FieldCore, const D: usize> Eq for HachiCommitmentHint<F, D> {}

/// Proof payload for stage 1 of a single Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiStage1Proof<F: FieldCore> {
    /// Stage-1 sumcheck proof over the virtual `S = w(w+1)` table.
    pub sumcheck: SumcheckProof<F>,
    /// Claimed evaluation of `S` at the stage-1 output point.
    pub s_claim: F,
}

/// Proof payload for stage 2 of a single Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiStage2Proof<F: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck: SumcheckProof<F>,
    /// Commitment to the next witness `w`
    /// (ring dim = next level's D, may differ from y_ring/v).
    pub next_w_commitment: FlatRingVec<F>,
    /// Claimed evaluation of the next witness `w` at the stage-2 challenge point.
    pub next_w_eval: F,
}

/// Proof for a single fold level (quad_eq + ring_switch + sumcheck).
///
/// D-agnostic: ring elements are stored as [`FlatRingVec`] with their
/// ring dimension recorded. Use [`Self::y_ring_typed`], [`Self::v_typed`], and
/// [`Self::w_commitment_typed`] to reconstruct typed ring elements.
///
/// One recursive Hachi level proof with inline stage-1 and stage-2 sumchecks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiLevelProof<F: FieldCore> {
    /// `y_ring` from the §3.1 reduction (ring dim = current level's D).
    pub y_ring: FlatRingVec<F>,
    /// `v = D · ŵ` (ring dim = current level's D).
    pub v: FlatRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: HachiStage1Proof<F>,
    /// Stage-2 fused payload.
    pub stage2: HachiStage2Proof<F>,
}

impl<F: FieldCore> HachiLevelProof<F> {
    /// Construct from typed ring elements for the current level and its
    /// inline two-stage norm-check payloads.
    pub(crate) fn new<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: HachiStage1Proof<F>,
        stage2: HachiStage2Proof<F>,
    ) -> Self {
        Self {
            y_ring: FlatRingVec::from_single(&y_ring),
            v: FlatRingVec::from_ring_elems(&v),
            stage1,
            stage2,
        }
    }

    /// Construct a level proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_two_stage<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1_sumcheck: SumcheckProof<F>,
        stage1_s_claim: F,
        stage2_sumcheck: SumcheckProof<F>,
        next_w_commitment: FlatRingVec<F>,
        next_w_eval: F,
    ) -> Self {
        Self::new::<D>(
            y_ring,
            v,
            HachiStage1Proof {
                sumcheck: stage1_sumcheck,
                s_claim: stage1_s_claim,
            },
            HachiStage2Proof {
                sumcheck: stage2_sumcheck,
                next_w_commitment,
                next_w_eval,
            },
        )
    }

    /// Ring dimension of y_ring and v (current level).
    pub fn level_d(&self) -> usize {
        self.y_ring.ring_dim()
    }

    /// Ring dimension of the w_commitment (next level).
    pub fn w_commit_d(&self) -> usize {
        self.stage2.next_w_commitment.ring_dim()
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
    /// Returns [`HachiError::InvalidProof`] if the stored `y_ring` does not
    /// encode exactly one ring element at dimension `D`.
    pub fn try_y_ring_typed<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, HachiError> {
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
    /// Returns [`HachiError::InvalidProof`] if the stored `v` payload is not
    /// well-formed for ring dimension `D`.
    pub fn try_v_typed<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
        self.v.try_to_vec()
    }

    /// Commitment to the next witness `w`.
    pub fn next_w_commitment(&self) -> &FlatRingVec<F> {
        &self.stage2.next_w_commitment
    }

    /// Reconstruct typed `w_commitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn w_commitment_typed<const D: usize>(&self) -> RingCommitment<F, D> {
        self.next_w_commitment().to_ring_commitment()
    }

    /// Reconstruct typed `w_commitment`, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored next-level commitment
    /// is not well-formed for ring dimension `D`.
    pub fn try_w_commitment_typed<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, HachiError> {
        self.next_w_commitment().try_to_ring_commitment()
    }

    /// Claimed evaluation of the next witness `w` at the norm-check output point.
    pub fn next_w_eval(&self) -> F {
        self.stage2.next_w_eval
    }
}

/// Proof tail: the final witness sent in clear as packed balanced digits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProofTail<F: FieldCore> {
    /// Final witness sent in clear as packed balanced digits.
    pub direct: PackedDigits,
    _marker: PhantomData<F>,
}

impl<F: FieldCore> HachiProofTail<F> {
    /// Construct a direct proof tail from packed digits.
    pub fn new(packed: PackedDigits) -> Self {
        Self {
            direct: packed,
            _marker: PhantomData,
        }
    }
}

/// Hachi PCS proof with multi-level folding.
///
/// Each level runs the full protocol (quadratic equation, ring switch,
/// sumcheck) on the previous level's witness `w`. The tail contains
/// the final witness as packed digits.
///
/// D-agnostic: per-level ring dimensions are recorded in each
/// [`HachiLevelProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore> {
    /// Per-level proofs, from the original polynomial (level 0) through
    /// recursive w-openings.
    pub levels: Vec<HachiLevelProof<F>>,
    /// Proof tail: direct final witness as packed digits.
    pub tail: HachiProofTail<F>,
}

impl<F: FieldCore> HachiProof<F> {
    /// Access the final witness.
    pub fn final_w(&self) -> &PackedDigits {
        &self.tail.direct
    }
}

impl<F: FieldCore + HachiSerialize> HachiProof<F> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiCommitmentHint<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.inner_opening_digits.len() as u64).serialize_with_mode(&mut writer, compress)?;
        for block in &self.inner_opening_digits {
            (block.len() as u64).serialize_with_mode(&mut writer, compress)?;
            for plane in block {
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(plane.as_ptr().cast::<u8>(), D) };
                writer.write_all(bytes)?;
            }
        }
        Ok(())
    }
    fn serialized_size(&self, _compress: Compress) -> usize {
        8 + self
            .inner_opening_digits
            .iter()
            .map(|block| 8 + block.len() * D)
            .sum::<usize>()
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiCommitmentHint<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiCommitmentHint<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let num_blocks = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let mut inner_opening_digits = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block_len = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
            let mut block = Vec::with_capacity(block_len);
            for _ in 0..block_len {
                let mut plane = [0i8; D];
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(plane.as_mut_ptr().cast::<u8>(), D) };
                reader.read_exact(bytes)?;
                block.push(plane);
            }
            inner_opening_digits.push(block);
        }
        Ok(Self::new(inner_opening_digits))
    }
}

impl<F: FieldCore> HachiSerialize for HachiLevelProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring.serialize_with_mode(&mut writer, compress)?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        self.stage1
            .sumcheck
            .serialize_with_mode(&mut writer, compress)?;
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
        base + self.stage1.sumcheck.serialized_size(compress)
            + self.stage1.s_claim.serialized_size(compress)
            + self.stage2.sumcheck.serialized_size(compress)
            + self.stage2.next_w_commitment.serialized_size(compress)
            + self.stage2.next_w_eval.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> Valid for HachiLevelProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_ring.check()?;
        if self.y_ring.count() != 1 {
            return Err(SerializationError::InvalidData(
                "hachi level y_ring must contain exactly one ring element".to_string(),
            ));
        }
        self.v.check()?;
        if self.v.ring_dim() != self.y_ring.ring_dim() {
            return Err(SerializationError::InvalidData(
                "hachi level v ring dimension must match y_ring".to_string(),
            ));
        }
        self.stage1.sumcheck.check()?;
        self.stage1.s_claim.check()?;
        self.stage2.sumcheck.check()?;
        self.stage2.next_w_commitment.check()?;
        self.stage2.next_w_eval.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiLevelProof<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let y_ring = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?;
        let v = FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?;
        let stage1 = HachiStage1Proof {
            sumcheck: SumcheckProof::deserialize_with_mode(&mut reader, compress, validate)?,
            s_claim: F::deserialize_with_mode(&mut reader, compress, validate)?,
        };
        let stage2 = HachiStage2Proof {
            sumcheck: SumcheckProof::deserialize_with_mode(&mut reader, compress, validate)?,
            next_w_commitment: FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?,
            next_w_eval: F::deserialize_with_mode(&mut reader, compress, validate)?,
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

impl<F: FieldCore> HachiSerialize for HachiProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.levels.len() as u32).serialize_with_mode(&mut writer, compress)?;
        for level in &self.levels {
            level.serialize_with_mode(&mut writer, compress)?;
        }
        self.tail.direct.serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        4 + self
            .levels
            .iter()
            .map(|l| l.serialized_size(compress))
            .sum::<usize>()
            + self.tail.direct.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> Valid for HachiProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        for lp in &self.levels {
            lp.check()?;
        }
        for levels in self.levels.windows(2) {
            if levels[0].w_commit_d() != levels[1].level_d() {
                return Err(SerializationError::InvalidData(
                    "adjacent hachi levels have mismatched commitment dimensions".to_string(),
                ));
            }
        }
        self.tail.direct.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiProof<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let num_levels = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let mut levels = Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            levels.push(HachiLevelProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?);
        }
        let pw = PackedDigits::deserialize_with_mode(&mut reader, compress, validate)?;
        let tail = HachiProofTail::new(pw);
        let out = Self { levels, tail };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
