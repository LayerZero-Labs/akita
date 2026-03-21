//! Proof structures for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::{Compress, SerializationError};
use crate::primitives::serialization::{Valid, Validate};
use crate::protocol::commitment::RingCommitment;
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorProof, LabradorReductionConfig, LabradorWitness,
};
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
    #[tracing::instrument(skip_all, name = "FlatCommitmentHint::to_typed_with_t")]
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

/// D-erased Labrador level proof.
///
/// Mirrors [`LabradorLevelProof`] with ring elements stored as [`FlatRingVec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatLabradorLevelProof<F: FieldCore> {
    /// Whether this level uses tail semantics.
    pub tail: bool,
    /// Input row lengths per witness row.
    pub input_row_lengths: Vec<usize>,
    /// Configuration selected for this level.
    pub config: LabradorReductionConfig,
    /// Virtual row length after reshaping (formerly `nn`).
    pub virtual_row_len: usize,
    /// Per-original-row split counts from the fold plan (formerly `nu`).
    pub row_split_counts: Vec<usize>,
    /// Opening-side payload (formerly `u1`).
    pub inner_opening_payload: FlatRingVec<F>,
    /// Linear-garbage-side payload (formerly `u2`).
    pub linear_garbage_payload: FlatRingVec<F>,
    /// JL projection vector.
    pub jl_projection: [i64; 256],
    /// JL nonce used to regenerate projection matrix.
    pub jl_nonce: u64,
    /// JL lift residuals (formerly `bb`).
    pub jl_lift_residuals: FlatRingVec<F>,
    /// Output witness norm bound after reduction (formerly `norm_sq`).
    pub next_witness_norm_sq: u128,
}

impl<F: FieldCore> FlatLabradorLevelProof<F> {
    /// Convert from the typed `LabradorLevelProof<F, D>`.
    pub fn from_typed<const D: usize>(p: &LabradorLevelProof<F, D>) -> Self {
        Self {
            tail: p.tail,
            input_row_lengths: p.input_row_lengths.clone(),
            config: p.config,
            virtual_row_len: p.virtual_row_len,
            row_split_counts: p.row_split_counts.clone(),
            inner_opening_payload: FlatRingVec::from_ring_elems(&p.inner_opening_payload),
            linear_garbage_payload: FlatRingVec::from_ring_elems(&p.linear_garbage_payload),
            jl_projection: p.jl_projection,
            jl_nonce: p.jl_nonce,
            jl_lift_residuals: FlatRingVec::from_ring_elems(&p.jl_lift_residuals),
            next_witness_norm_sq: p.next_witness_norm_sq,
        }
    }

    /// Reconstruct the typed `LabradorLevelProof<F, D>`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn to_typed<const D: usize>(&self) -> LabradorLevelProof<F, D> {
        LabradorLevelProof {
            tail: self.tail,
            input_row_lengths: self.input_row_lengths.clone(),
            config: self.config,
            virtual_row_len: self.virtual_row_len,
            row_split_counts: self.row_split_counts.clone(),
            inner_opening_payload: self.inner_opening_payload.to_vec(),
            linear_garbage_payload: self.linear_garbage_payload.to_vec(),
            jl_projection: self.jl_projection,
            jl_nonce: self.jl_nonce,
            jl_lift_residuals: self.jl_lift_residuals.to_vec(),
            next_witness_norm_sq: self.next_witness_norm_sq,
        }
    }
}

/// D-erased Labrador witness (rows of ring elements).
///
/// Mirrors [`LabradorWitness`] with rows stored as [`FlatRingVec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatLabradorWitness<F: FieldCore> {
    /// Per-row ring element vectors.
    pub rows: Vec<FlatRingVec<F>>,
}

impl<F: FieldCore> FlatLabradorWitness<F> {
    /// Convert from the typed `LabradorWitness<F, D>`.
    pub fn from_typed<const D: usize>(w: &LabradorWitness<F, D>) -> Self {
        Self {
            rows: w
                .rows()
                .iter()
                .map(|r| FlatRingVec::from_ring_elems(r))
                .collect(),
        }
    }

    /// Reconstruct the typed `LabradorWitness<F, D>`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn to_typed<const D: usize>(&self) -> LabradorWitness<F, D> {
        let rows: Vec<Vec<CyclotomicRing<F, D>>> = self.rows.iter().map(|r| r.to_vec()).collect();
        LabradorWitness::new_unchecked(rows)
    }
}

/// D-erased Labrador proof (levels + final witness).
///
/// Mirrors [`LabradorProof`] with all ring data stored as [`FlatRingVec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatLabradorProof<F: FieldCore> {
    /// Recursive level payloads.
    pub levels: Vec<FlatLabradorLevelProof<F>>,
    /// Final clear witness opened at recursion termination.
    pub final_opening_witness: FlatLabradorWitness<F>,
}

impl<F: FieldCore> FlatLabradorProof<F> {
    /// Convert from the typed `LabradorProof<F, D>`.
    pub fn from_typed<const D: usize>(p: &LabradorProof<F, D>) -> Self {
        Self {
            levels: p
                .levels
                .iter()
                .map(FlatLabradorLevelProof::from_typed)
                .collect(),
            final_opening_witness: FlatLabradorWitness::from_typed(&p.final_opening_witness),
        }
    }

    /// Reconstruct the typed `LabradorProof<F, D>`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn to_typed<const D: usize>(&self) -> LabradorProof<F, D> {
        LabradorProof {
            levels: self.levels.iter().map(|l| l.to_typed()).collect(),
            final_opening_witness: self.final_opening_witness.to_typed(),
        }
    }
}

/// Labrador tail proof data.
///
/// Produced when Hachi's folding loop stops and the ring-level `Mz = y`
/// relation from the quadratic equation is handed directly to Labrador
/// without computing quotient `r`, evaluating at alpha, or running sumcheck.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabradorTail<F: FieldCore> {
    /// D-erased full Labrador recursive proof.
    pub labrador_proof: FlatLabradorProof<F>,
    /// Ring-valued prover message `v = D * w_hat` (public, used to rebuild constraints).
    pub v: FlatRingVec<F>,
    /// Ring-valued evaluation `y_ring` (public, used to rebuild constraints).
    pub y_ring: FlatRingVec<F>,
    /// Squared L2 norm bound of the Labrador witness (formerly `beta_sq`).
    pub witness_norm_bound_sq: u128,
}

/// Proof tail: either a direct witness or a Labrador handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HachiProofTail<F: FieldCore> {
    /// Final witness sent in clear as packed balanced digits.
    Direct(PackedDigits),
    /// Direct Labrador handoff from the quadratic equation.
    Labrador(Box<LabradorTail<F>>),
}

/// Hachi PCS proof with multi-level folding.
///
/// Each level runs the full protocol (quadratic equation, ring switch,
/// sumcheck) on the previous level's witness `w`. The tail is either
/// a direct witness (packed digits) or a Labrador handoff.
///
/// D-agnostic: per-level ring dimensions are recorded in each
/// [`HachiLevelProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore> {
    /// Per-level proofs, from the original polynomial (level 0) through
    /// recursive w-openings.
    pub levels: Vec<HachiLevelProof<F>>,
    /// Proof tail: direct witness or Labrador handoff.
    pub tail: HachiProofTail<F>,
}

impl<F: FieldCore> HachiProof<F> {
    /// Access the direct final witness when the proof ends with a clear tail.
    pub fn final_w(&self) -> Option<&PackedDigits> {
        match &self.tail {
            HachiProofTail::Direct(pw) => Some(pw),
            HachiProofTail::Labrador(_) => None,
        }
    }

    /// Whether this proof uses a Labrador tail (not a direct witness).
    pub fn has_handoff_tail(&self) -> bool {
        matches!(&self.tail, HachiProofTail::Labrador(_))
    }

    /// Whether this proof uses the direct Labrador tail.
    pub fn has_labrador_tail(&self) -> bool {
        matches!(&self.tail, HachiProofTail::Labrador(_))
    }
}

impl<F: FieldCore + HachiSerialize> HachiProof<F> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
    }
}

impl HachiSerialize for LabradorReductionConfig {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.witness_digit_parts
            .serialize_with_mode(&mut writer, compress)?;
        self.witness_digit_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.aux_digit_parts
            .serialize_with_mode(&mut writer, compress)?;
        self.aux_digit_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.inner_commit_rank
            .serialize_with_mode(&mut writer, compress)?;
        self.outer_commit_rank
            .serialize_with_mode(&mut writer, compress)?;
        (self.tail as u8).serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.witness_digit_parts.serialized_size(compress)
            + self.witness_digit_bits.serialized_size(compress)
            + self.aux_digit_parts.serialized_size(compress)
            + self.aux_digit_bits.serialized_size(compress)
            + self.inner_commit_rank.serialized_size(compress)
            + self.outer_commit_rank.serialized_size(compress)
            + 1
    }
}

impl Valid for LabradorReductionConfig {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl HachiDeserialize for LabradorReductionConfig {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let witness_digit_parts = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let witness_digit_bits = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let aux_digit_parts = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let aux_digit_bits = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let inner_commit_rank = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let outer_commit_rank = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let tail = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        if tail > 1 {
            return Err(SerializationError::InvalidData(
                "invalid LabradorReductionConfig tail flag".to_string(),
            ));
        }
        Ok(Self {
            witness_digit_parts,
            witness_digit_bits,
            aux_digit_parts,
            aux_digit_bits,
            inner_commit_rank,
            outer_commit_rank,
            tail: tail != 0,
        })
    }
}

impl<F: FieldCore> HachiSerialize for FlatLabradorLevelProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.tail as u8).serialize_with_mode(&mut writer, compress)?;
        self.input_row_lengths
            .serialize_with_mode(&mut writer, compress)?;
        self.config.serialize_with_mode(&mut writer, compress)?;
        self.virtual_row_len
            .serialize_with_mode(&mut writer, compress)?;
        self.row_split_counts
            .serialize_with_mode(&mut writer, compress)?;
        self.inner_opening_payload
            .serialize_with_mode(&mut writer, compress)?;
        self.linear_garbage_payload
            .serialize_with_mode(&mut writer, compress)?;
        for coeff in &self.jl_projection {
            coeff.serialize_with_mode(&mut writer, compress)?;
        }
        self.jl_nonce.serialize_with_mode(&mut writer, compress)?;
        self.jl_lift_residuals
            .serialize_with_mode(&mut writer, compress)?;
        self.next_witness_norm_sq
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        1 + self.input_row_lengths.serialized_size(compress)
            + self.config.serialized_size(compress)
            + self.virtual_row_len.serialized_size(compress)
            + self.row_split_counts.serialized_size(compress)
            + self.inner_opening_payload.serialized_size(compress)
            + self.linear_garbage_payload.serialized_size(compress)
            + self.jl_projection.len() * std::mem::size_of::<i64>()
            + self.jl_nonce.serialized_size(compress)
            + self.jl_lift_residuals.serialized_size(compress)
            + self.next_witness_norm_sq.serialized_size(compress)
    }
}

impl<F: FieldCore> Valid for FlatLabradorLevelProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        if self.tail != self.config.tail {
            return Err(SerializationError::InvalidData(
                "FlatLabradorLevelProof tail/config mismatch".to_string(),
            ));
        }
        if self.tail && self.config.outer_commit_rank != 0 {
            return Err(SerializationError::InvalidData(
                "FlatLabradorLevelProof tail level must have outer_commit_rank = 0".to_string(),
            ));
        }
        if !self.tail && self.config.outer_commit_rank == 0 {
            return Err(SerializationError::InvalidData(
                "FlatLabradorLevelProof non-tail level must have outer_commit_rank > 0".to_string(),
            ));
        }
        self.config.check()?;
        self.inner_opening_payload.check()?;
        self.linear_garbage_payload.check()?;
        if self.inner_opening_payload.ring_dim() != self.linear_garbage_payload.ring_dim()
            || self.inner_opening_payload.ring_dim() != self.jl_lift_residuals.ring_dim()
        {
            return Err(SerializationError::InvalidData(
                "FlatLabradorLevelProof ring-dimension mismatch".to_string(),
            ));
        }
        self.jl_lift_residuals.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for FlatLabradorLevelProof<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let tail = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        if tail > 1 {
            return Err(SerializationError::InvalidData(
                "invalid FlatLabradorLevelProof tail flag".to_string(),
            ));
        }

        let mut jl_projection = [0i64; 256];
        let input_row_lengths =
            Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate)?;
        let config =
            LabradorReductionConfig::deserialize_with_mode(&mut reader, compress, validate)?;
        let virtual_row_len = usize::deserialize_with_mode(&mut reader, compress, validate)?;
        let row_split_counts =
            Vec::<usize>::deserialize_with_mode(&mut reader, compress, validate)?;
        let inner_opening_payload =
            FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?;
        let linear_garbage_payload =
            FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?;
        for coeff in &mut jl_projection {
            *coeff = i64::deserialize_with_mode(&mut reader, compress, validate)?;
        }
        let jl_nonce = u64::deserialize_with_mode(&mut reader, compress, validate)?;
        let jl_lift_residuals =
            FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?;
        let next_witness_norm_sq = u128::deserialize_with_mode(&mut reader, compress, validate)?;

        let out = Self {
            tail: tail != 0,
            input_row_lengths,
            config,
            virtual_row_len,
            row_split_counts,
            inner_opening_payload,
            linear_garbage_payload,
            jl_projection,
            jl_nonce,
            jl_lift_residuals,
            next_witness_norm_sq,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore> HachiSerialize for FlatLabradorWitness<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.rows.len() as u32).serialize_with_mode(&mut writer, compress)?;
        for row in &self.rows {
            row.serialize_with_mode(&mut writer, compress)?;
        }
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        4 + self
            .rows
            .iter()
            .map(|row| row.serialized_size(compress))
            .sum::<usize>()
    }
}

impl<F: FieldCore> Valid for FlatLabradorWitness<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let expected_ring_dim = self.rows.first().map(FlatRingVec::ring_dim);
        for row in &self.rows {
            row.check()?;
            if expected_ring_dim.is_some_and(|d| row.ring_dim() != d) {
                return Err(SerializationError::InvalidData(
                    "FlatLabradorWitness ring-dimension mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for FlatLabradorWitness<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let num_rows = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let mut rows = Vec::with_capacity(num_rows);
        for _ in 0..num_rows {
            rows.push(FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?);
        }
        let out = Self { rows };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore> HachiSerialize for FlatLabradorProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.levels.len() as u32).serialize_with_mode(&mut writer, compress)?;
        for level in &self.levels {
            level.serialize_with_mode(&mut writer, compress)?;
        }
        self.final_opening_witness
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        4 + self
            .levels
            .iter()
            .map(|level| level.serialized_size(compress))
            .sum::<usize>()
            + self.final_opening_witness.serialized_size(compress)
    }
}

impl<F: FieldCore> Valid for FlatLabradorProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        let mut expected_ring_dim = self
            .final_opening_witness
            .rows
            .first()
            .map(FlatRingVec::ring_dim);
        for level in &self.levels {
            level.check()?;
            if let Some(d) = expected_ring_dim {
                if level.inner_opening_payload.ring_dim() != d {
                    return Err(SerializationError::InvalidData(
                        "FlatLabradorProof ring-dimension mismatch".to_string(),
                    ));
                }
            } else {
                expected_ring_dim = Some(level.inner_opening_payload.ring_dim());
            }
        }
        self.final_opening_witness.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for FlatLabradorProof<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let num_levels = u32::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let mut levels = Vec::with_capacity(num_levels);
        for _ in 0..num_levels {
            levels.push(FlatLabradorLevelProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?);
        }
        let final_opening_witness =
            FlatLabradorWitness::deserialize_with_mode(&mut reader, compress, validate)?;
        let out = Self {
            levels,
            final_opening_witness,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore> HachiSerialize for LabradorTail<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.labrador_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        self.y_ring.serialize_with_mode(&mut writer, compress)?;
        self.witness_norm_bound_sq
            .serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.labrador_proof.serialized_size(compress)
            + self.v.serialized_size(compress)
            + self.y_ring.serialized_size(compress)
            + self.witness_norm_bound_sq.serialized_size(compress)
    }
}

impl<F: FieldCore> Valid for LabradorTail<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.labrador_proof.check()?;
        self.v.check()?;
        self.y_ring.check()?;
        if self.v.ring_dim() != self.y_ring.ring_dim() {
            return Err(SerializationError::InvalidData(
                "LabradorTail ring-dimension mismatch".to_string(),
            ));
        }
        if self
            .labrador_proof
            .final_opening_witness
            .rows
            .first()
            .is_some_and(|row| row.ring_dim() != self.v.ring_dim())
        {
            return Err(SerializationError::InvalidData(
                "LabradorTail witness ring-dimension mismatch".to_string(),
            ));
        }
        for level in &self.labrador_proof.levels {
            if level.inner_opening_payload.ring_dim() != self.v.ring_dim() {
                return Err(SerializationError::InvalidData(
                    "LabradorTail level ring-dimension mismatch".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for LabradorTail<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let out = Self {
            labrador_proof: FlatLabradorProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?,
            v: FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?,
            y_ring: FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?,
            witness_norm_bound_sq: u128::deserialize_with_mode(&mut reader, compress, validate)?,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
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
        match &self.tail {
            HachiProofTail::Direct(pw) => {
                0u8.serialize_with_mode(&mut writer, compress)?;
                pw.serialize_with_mode(&mut writer, compress)
            }
            HachiProofTail::Labrador(tail) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                tail.serialize_with_mode(&mut writer, compress)
            }
        }
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        let base = 4
            + self
                .levels
                .iter()
                .map(|l| l.serialized_size(compress))
                .sum::<usize>()
            + 1; // tag byte
        match &self.tail {
            HachiProofTail::Direct(pw) => base + pw.serialized_size(compress),
            HachiProofTail::Labrador(tail) => base + tail.serialized_size(compress),
        }
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
        match &self.tail {
            HachiProofTail::Direct(pw) => pw.check(),
            HachiProofTail::Labrador(tail) => tail.check(),
        }
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
        let tag = u8::deserialize_with_mode(&mut reader, compress, validate)?;
        let tail = match tag {
            0 => {
                let pw = PackedDigits::deserialize_with_mode(&mut reader, compress, validate)?;
                HachiProofTail::Direct(pw)
            }
            1 => HachiProofTail::Labrador(Box::new(LabradorTail::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
            )?)),
            _ => {
                return Err(SerializationError::InvalidData(format!(
                    "unknown proof tail tag: {tag}"
                )));
            }
        };
        let out = Self { levels, tail };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}
