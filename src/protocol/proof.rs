//! Proof structures for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::error::HachiError;
use crate::primitives::serialization::{Compress, SerializationError};
use crate::primitives::serialization::{Valid, Validate};
use crate::protocol::commitment::RingCommitment;
use crate::protocol::sumcheck::types::{EqFactoredSumcheckProofShape, SumcheckProofShape};
use crate::protocol::sumcheck::{EqFactoredSumcheckProof, SumcheckProof};
use crate::protocol::transcript::Transcript;
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
    pub fn to_field_elems<F: FieldCore + FromSmallInt>(&self) -> Vec<F> {
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

impl HachiSerialize for PackedDigits {
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

impl HachiDeserialize for PackedDigits {
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatRingVec<F> {
    coeffs: Vec<F>,
    ring_dim: usize,
}

pub(crate) struct RingSliceSerializer<'a, F: FieldCore, const D: usize>(
    pub(crate) &'a [CyclotomicRing<F, D>],
);

impl<F: FieldCore, const D: usize> HachiSerialize for RingSliceSerializer<'_, F, D> {
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
        if self.ring_dim != D || !self.coeffs.len().is_multiple_of(D) {
            return Err(HachiError::InvalidProof);
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
    /// Returns [`HachiError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub(crate) fn as_ring_slice<const D: usize>(
        &self,
    ) -> Result<&[CyclotomicRing<F, D>], HachiError> {
        if self.ring_dim != D || !self.coeffs.len().is_multiple_of(D) {
            return Err(HachiError::InvalidProof);
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

    /// Append the stored coefficients using the same transcript encoding as a
    /// typed [`RingCommitment`].
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub(crate) fn append_as_ring_commitment<T: Transcript<F>, const D: usize>(
        &self,
        label: &[u8],
        transcript: &mut T,
    ) -> Result<(), HachiError>
    where
        F: CanonicalField,
    {
        let rings = self.as_ring_slice::<D>()?;
        transcript.append_serde(label, &RingSliceSerializer(rings));
        Ok(())
    }

    /// Convert to the verifier-facing proof-wire payload.
    pub fn to_proof_ring_vec(&self) -> ProofRingVec<F> {
        ProofRingVec {
            coeffs: self.coeffs.clone(),
        }
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
        if !self.coeffs.len().is_multiple_of(self.ring_dim) {
            return Err(SerializationError::InvalidData(
                "coeffs length not a multiple of ring_dim".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for FlatRingVec<F> {
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let ring_dim = u32::deserialize_with_mode(&mut reader, compress, validate, &())? as usize;
        let coeffs = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self { coeffs, ring_dim };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

/// Proof-wire storage for a sequence of ring-element coefficients.
///
/// Unlike [`FlatRingVec`], this schema does not encode the ring dimension in
/// the proof itself. The verifier recovers the correct `D` from the public
/// schedule and interprets the coefficient buffer accordingly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofRingVec<F> {
    coeffs: Vec<F>,
}

impl<F: FieldCore> ProofRingVec<F> {
    /// Construct directly from raw field coefficients.
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    /// Convert from a vector of ring elements.
    pub fn from_ring_elems<const D: usize>(elems: &[CyclotomicRing<F, D>]) -> Self {
        let mut coeffs = Vec::with_capacity(elems.len() * D);
        for elem in elems {
            coeffs.extend_from_slice(elem.coefficients());
        }
        Self { coeffs }
    }

    /// Convert from a single ring element.
    pub fn from_single<const D: usize>(elem: &CyclotomicRing<F, D>) -> Self {
        Self {
            coeffs: elem.coefficients().to_vec(),
        }
    }

    /// Raw field coefficients stored in the proof.
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

    /// Borrow the stored coefficients as a slice of typed ring elements.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored coefficient count is
    /// not divisible by `D`.
    pub fn as_ring_slice<const D: usize>(&self) -> Result<&[CyclotomicRing<F, D>], HachiError> {
        if !self.coeffs.len().is_multiple_of(D) {
            return Err(HachiError::InvalidProof);
        }
        let ring_count = self.coeffs.len() / D;
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
    /// Returns [`HachiError::InvalidProof`] if the stored coefficient count is
    /// not exactly `D`.
    pub fn as_single_ring<const D: usize>(&self) -> Result<&CyclotomicRing<F, D>, HachiError> {
        let rings = self.as_ring_slice::<D>()?;
        match rings {
            [ring] => Ok(ring),
            _ => Err(HachiError::InvalidProof),
        }
    }

    /// Append the stored coefficients using the typed ring-vector transcript
    /// encoding.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored ring data is not
    /// well-formed for ring dimension `D`.
    pub fn append_as_ring_slice<T: Transcript<F>, const D: usize>(
        &self,
        label: &[u8],
        transcript: &mut T,
    ) -> Result<(), HachiError>
    where
        F: CanonicalField,
    {
        let rings = self.as_ring_slice::<D>()?;
        transcript.append_serde(label, &RingSliceSerializer(rings));
        Ok(())
    }

    /// Reconstruct a single ring element.
    ///
    /// # Panics
    ///
    /// Panics if the stored coefficient count is not exactly `D`.
    pub fn to_single<const D: usize>(&self) -> CyclotomicRing<F, D> {
        assert_eq!(
            self.coeffs.len(),
            D,
            "D mismatch in ProofRingVec::to_single"
        );
        CyclotomicRing::from_slice(&self.coeffs)
    }

    /// Reconstruct a single ring element, returning `InvalidProof` on shape mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`HachiError::InvalidProof`] if the stored coefficient count is
    /// not exactly `D`.
    pub fn try_to_single<const D: usize>(&self) -> Result<CyclotomicRing<F, D>, HachiError> {
        if self.coeffs.len() != D {
            return Err(HachiError::InvalidProof);
        }
        Ok(CyclotomicRing::from_slice(&self.coeffs))
    }

    /// Reconstruct a vector of ring elements.
    ///
    /// # Panics
    ///
    /// Panics if the stored coefficient count is not divisible by `D`.
    pub fn to_vec<const D: usize>(&self) -> Vec<CyclotomicRing<F, D>> {
        assert_eq!(
            self.coeffs.len() % D,
            0,
            "D mismatch in ProofRingVec::to_vec"
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
    /// Returns [`HachiError::InvalidProof`] if the stored coefficient count is
    /// not divisible by `D`.
    pub fn try_to_vec<const D: usize>(&self) -> Result<Vec<CyclotomicRing<F, D>>, HachiError> {
        if !self.coeffs.len().is_multiple_of(D) {
            return Err(HachiError::InvalidProof);
        }
        Ok(self
            .coeffs
            .chunks_exact(D)
            .map(CyclotomicRing::from_slice)
            .collect())
    }
}

impl<F: FieldCore> HachiSerialize for ProofRingVec<F> {
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

impl<F: FieldCore + Valid> Valid for ProofRingVec<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.coeffs.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for ProofRingVec<F> {
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
        let out = Self { coeffs };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
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

    /// Consume the hint and return its decomposed digits plus the optional
    /// recomposed `t_i` rows.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (Vec<Vec<[i8; D]>>, Option<Vec<Vec<CyclotomicRing<F, D>>>>) {
        (self.inner_opening_digits, self.t)
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

/// Prover-side hint for one same-point commitment group.
///
/// Stores per-polynomial `t_hat` blocks and, when available, the corresponding
/// undecomposed `t_i` rows for all claims that were aggregated into the same
/// commitment.
#[derive(Debug, Clone)]
pub struct HachiBatchedCommitmentHint<F: FieldCore, const D: usize> {
    /// Per-polynomial decomposed inner-opening digit blocks.
    pub inner_opening_digits: Vec<Vec<Vec<[i8; D]>>>,
    /// Optional recomposed `t_i` rows grouped by polynomial then block.
    t: Option<Vec<Vec<Vec<CyclotomicRing<F, D>>>>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> HachiBatchedCommitmentHint<F, D> {
    /// Construct a new batched hint from per-polynomial i8 digit plane blocks.
    pub fn new(inner_opening_digits: Vec<Vec<Vec<[i8; D]>>>) -> Self {
        Self {
            inner_opening_digits,
            t: None,
            _marker: PhantomData,
        }
    }

    /// Construct a batched hint that also preserves the undecomposed `t_i` rows.
    pub fn with_t(
        inner_opening_digits: Vec<Vec<Vec<[i8; D]>>>,
        t: Vec<Vec<Vec<CyclotomicRing<F, D>>>>,
    ) -> Self {
        Self {
            inner_opening_digits,
            t: Some(t),
            _marker: PhantomData,
        }
    }

    /// Get the optional recomposed `t_i` rows grouped by polynomial.
    pub fn t(&self) -> Option<&[Vec<Vec<CyclotomicRing<F, D>>>]> {
        self.t.as_deref()
    }

    /// Flatten the batched hint into one root-hint view over all claims.
    pub fn into_flattened(self) -> HachiCommitmentHint<F, D> {
        let inner_opening_digits = self.inner_opening_digits.into_iter().flatten().collect();
        let t = self
            .t
            .map(|rows_by_poly| rows_by_poly.into_iter().flatten().collect());
        match t {
            Some(t) => HachiCommitmentHint::with_t(inner_opening_digits, t),
            None => HachiCommitmentHint::new(inner_opening_digits),
        }
    }

    /// Construct a batched hint by grouping standard per-polynomial hints.
    pub fn from_commit_hints(hints: Vec<HachiCommitmentHint<F, D>>) -> Self {
        let mut inner_opening_digits = Vec::with_capacity(hints.len());
        let mut t = Vec::with_capacity(hints.len());
        let mut has_t = true;
        for hint in hints {
            let (digits, rows) = hint.into_parts();
            inner_opening_digits.push(digits);
            match rows {
                Some(rows) if has_t => t.push(rows),
                Some(_) => {}
                None => has_t = false,
            }
        }
        if has_t {
            Self::with_t(inner_opening_digits, t)
        } else {
            Self::new(inner_opening_digits)
        }
    }
}

impl<F: FieldCore, const D: usize> PartialEq for HachiBatchedCommitmentHint<F, D> {
    fn eq(&self, other: &Self) -> bool {
        self.inner_opening_digits == other.inner_opening_digits
    }
}

impl<F: FieldCore, const D: usize> Eq for HachiBatchedCommitmentHint<F, D> {}
/// One stage in the stage-1 range-check tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiStage1StageProof<F: FieldCore> {
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
pub struct HachiStage1StageShape {
    /// Eq-factored sumcheck shape `(num_rounds, q_degree)`.
    pub sumcheck: EqFactoredSumcheckProofShape,
    /// Number of child claims serialized after the stage proof.
    pub child_claims: usize,
}
/// Proof payload for stage 1 of a single Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiStage1Proof<F: FieldCore> {
    /// Root-to-leaf range-check stages.
    pub stages: Vec<HachiStage1StageProof<F>>,
    /// Claimed evaluation of `S` at the final stage-1 output point.
    pub s_claim: F,
}

/// Proof payload for stage 2 of a single Hachi level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiStage2Proof<F: FieldCore> {
    /// Stage-2 fused sumcheck proof.
    pub sumcheck: SumcheckProof<F>,
    /// Commitment to the next witness `w`
    /// (ring dim = next level's D, may differ from y_ring/v).
    pub next_w_commitment: ProofRingVec<F>,
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
    pub y_ring: ProofRingVec<F>,
    /// `v = D · ŵ` (ring dim = current level's D).
    pub v: ProofRingVec<F>,
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
            y_ring: ProofRingVec::from_single(&y_ring),
            v: ProofRingVec::from_ring_elems(&v),
            stage1,
            stage2,
        }
    }

    /// Construct a level proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_two_stage<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: HachiStage1Proof<F>,
        stage2_sumcheck: SumcheckProof<F>,
        next_w_commitment: ProofRingVec<F>,
        next_w_eval: F,
    ) -> Self {
        Self::new::<D>(
            y_ring,
            v,
            stage1,
            HachiStage2Proof {
                sumcheck: stage2_sumcheck,
                next_w_commitment,
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
    pub fn next_w_commitment(&self) -> &ProofRingVec<F> {
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
    /// Returns [`HachiError::InvalidProof`] if the stored next-level commitment
    /// is not well-formed for ring dimension `D`.
    pub fn try_w_commitment_typed<const D: usize>(
        &self,
    ) -> Result<RingCommitment<F, D>, HachiError> {
        Ok(RingCommitment {
            u: self.next_w_commitment().try_to_vec()?,
        })
    }

    /// Claimed evaluation of the next witness `w` at the norm-check output point.
    pub fn next_w_eval(&self) -> F {
        self.stage2.next_w_eval
    }

    /// Derive the [`LevelProofShape`] for this level proof.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(self.y_ring.coeff_len(), &self.v, &self.stage1, &self.stage2)
    }
}

/// Root proof payload for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiBatchedRootProof<F: FieldCore> {
    /// Root public outputs `y_ell` stored as a ring vector.
    pub y_rings: ProofRingVec<F>,
    /// Aggregated `v = Σ_ell D_ell · w_hat_ell`.
    pub v: ProofRingVec<F>,
    /// Stage-1 norm-check payload.
    pub stage1: HachiStage1Proof<F>,
    /// Stage-2 fused payload.
    pub stage2: HachiStage2Proof<F>,
}

impl<F: FieldCore> HachiBatchedRootProof<F> {
    /// Construct from typed ring elements for the batched root level.
    pub(crate) fn new<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: HachiStage1Proof<F>,
        stage2: HachiStage2Proof<F>,
    ) -> Self {
        Self {
            y_rings: ProofRingVec::from_ring_elems(&y_rings),
            v: ProofRingVec::from_ring_elems(&v),
            stage1,
            stage2,
        }
    }

    /// Construct a batched root proof for the two-stage norm-check.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_two_stage<const D: usize>(
        y_rings: Vec<CyclotomicRing<F, D>>,
        v: Vec<CyclotomicRing<F, D>>,
        stage1: HachiStage1Proof<F>,
        stage2_sumcheck: SumcheckProof<F>,
        next_w_commitment: ProofRingVec<F>,
        next_w_eval: F,
    ) -> Self {
        Self::new::<D>(
            y_rings,
            v,
            stage1,
            HachiStage2Proof {
                sumcheck: stage2_sumcheck,
                next_w_commitment,
                next_w_eval,
            },
        )
    }

    /// Borrow the stored root `y` ring vector.
    pub fn y_rings(&self) -> &ProofRingVec<F> {
        &self.y_rings
    }

    /// Borrow the stored root `v` ring vector.
    pub fn v(&self) -> &ProofRingVec<F> {
        &self.v
    }

    /// Commitment to the next witness `w`.
    pub fn next_w_commitment(&self) -> &ProofRingVec<F> {
        &self.stage2.next_w_commitment
    }

    /// Claimed evaluation of the next witness `w` at the norm-check output point.
    pub fn next_w_eval(&self) -> F {
        self.stage2.next_w_eval
    }

    /// Derive the [`LevelProofShape`] for this root proof.
    pub fn shape(&self) -> LevelProofShape {
        level_proof_shape(
            self.y_rings.coeff_len(),
            &self.v,
            &self.stage1,
            &self.stage2,
        )
    }
}

/// Hachi PCS proof for fused batched openings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiBatchedProof<F: FieldCore> {
    /// Batched root proof over all original-polynomial claims.
    pub root: HachiBatchedRootProof<F>,
    /// Recursive proofs for subsequent `w` openings.
    pub levels: Vec<HachiLevelProof<F>>,
    /// Proof tail: direct final witness as packed digits.
    pub tail: HachiProofTail<F>,
}

impl<F: FieldCore> HachiBatchedProof<F> {
    /// Access the final witness.
    pub fn final_w(&self) -> &PackedDigits {
        &self.tail.direct
    }

    /// Derive the [`HachiBatchedProofShape`] for this proof.
    pub fn shape(&self) -> HachiBatchedProofShape {
        HachiBatchedProofShape {
            root_shape: self.root.shape(),
            level_shapes: self.levels.iter().map(|l| l.shape()).collect(),
            tail_shape: (self.tail.direct.num_elems, self.tail.direct.bits_per_elem),
        }
    }
}

impl<F: FieldCore + HachiSerialize> HachiBatchedProof<F> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.serialized_size(Compress::No)
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

/// Shape descriptor for deserializing a [`HachiLevelProof`] without headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelProofShape {
    /// Number of field coefficients in `y_ring`.
    pub y_ring_coeffs: usize,
    /// Number of field coefficients in `v`.
    pub v_coeffs: usize,
    /// Stage-1 tree stage shapes in root-to-leaf order.
    pub stage1_stages: Vec<HachiStage1StageShape>,
    /// Stage-2 sumcheck shape: `(num_rounds, degree)`.
    pub stage2_sumcheck: SumcheckProofShape,
    /// Number of field coefficients in `next_w_commitment`.
    pub next_commit_coeffs: usize,
}

/// Shape descriptor for deserializing an entire [`HachiProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProofShape {
    /// Per-level shapes in execution order.
    pub level_shapes: Vec<LevelProofShape>,
    /// Tail packed-digit shape: `(num_elems, bits_per_elem)`.
    pub tail_shape: (usize, u32),
}

/// Shape descriptor for deserializing an [`HachiBatchedProof`] without
/// headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiBatchedProofShape {
    /// Root-level shape (same field layout as a regular level).
    pub root_shape: LevelProofShape,
    /// Per-level shapes for the recursive folding levels.
    pub level_shapes: Vec<LevelProofShape>,
    /// Tail packed-digit shape: `(num_elems, bits_per_elem)`.
    pub tail_shape: (usize, u32),
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

fn level_proof_shape<F: FieldCore>(
    y_coeffs: usize,
    v: &ProofRingVec<F>,
    stage1: &HachiStage1Proof<F>,
    stage2: &HachiStage2Proof<F>,
) -> LevelProofShape {
    LevelProofShape {
        y_ring_coeffs: y_coeffs,
        v_coeffs: v.coeff_len(),
        stage1_stages: stage1
            .stages
            .iter()
            .map(|stage| HachiStage1StageShape {
                sumcheck: eq_factored_sumcheck_shape(&stage.sumcheck),
                child_claims: stage.child_claims.len(),
            })
            .collect(),
        stage2_sumcheck: sumcheck_shape(&stage2.sumcheck),
        next_commit_coeffs: stage2.next_w_commitment.coeff_len(),
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

    /// Derive the [`HachiProofShape`] for this proof.
    pub fn shape(&self) -> HachiProofShape {
        HachiProofShape {
            level_shapes: self.levels.iter().map(|l| l.shape()).collect(),
            tail_shape: (self.tail.direct.num_elems, self.tail.direct.bits_per_elem),
        }
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
    type Context = ();
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let num_blocks = u64::deserialize_with_mode(&mut reader, compress, validate, &())? as usize;
        let mut inner_opening_digits = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block_len =
                u64::deserialize_with_mode(&mut reader, compress, validate, &())? as usize;
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

impl<F: FieldCore + Valid> Valid for HachiLevelProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_ring.check()?;
        if self.y_ring.coeff_len() == 0 {
            return Err(SerializationError::InvalidData(
                "hachi level y_ring must contain exactly one ring element".to_string(),
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

impl<F: FieldCore + Valid> HachiDeserialize for HachiLevelProof<F> {
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        let y_ring = ProofRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let v =
            ProofRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
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
                child_claims.push(F::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(HachiStage1StageProof {
                sumcheck,
                child_claims,
            });
        }
        let stage1 = HachiStage1Proof {
            stages: stage1_stages,
            s_claim: F::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2 = HachiStage2Proof {
            sumcheck: SumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck,
            )?,
            next_w_commitment: ProofRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            next_w_eval: F::deserialize_with_mode(&mut reader, compress, validate, &())?,
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

impl<F: FieldCore> HachiSerialize for HachiBatchedRootProof<F> {
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

impl<F: FieldCore + Valid> Valid for HachiBatchedRootProof<F> {
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

impl<F: FieldCore + Valid> HachiDeserialize for HachiBatchedRootProof<F> {
    type Context = LevelProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &LevelProofShape,
    ) -> Result<Self, SerializationError> {
        let y_rings = ProofRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.y_ring_coeffs,
        )?;
        let v =
            ProofRingVec::deserialize_with_mode(&mut reader, compress, validate, &ctx.v_coeffs)?;
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
                child_claims.push(F::deserialize_with_mode(
                    &mut reader,
                    compress,
                    validate,
                    &(),
                )?);
            }
            stage1_stages.push(HachiStage1StageProof {
                sumcheck,
                child_claims,
            });
        }
        let stage1 = HachiStage1Proof {
            stages: stage1_stages,
            s_claim: F::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        let stage2 = HachiStage2Proof {
            sumcheck: SumcheckProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.stage2_sumcheck,
            )?,
            next_w_commitment: ProofRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.next_commit_coeffs,
            )?,
            next_w_eval: F::deserialize_with_mode(&mut reader, compress, validate, &())?,
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

impl<F: FieldCore> HachiSerialize for HachiBatchedProof<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.root.serialize_with_mode(&mut writer, compress)?;
        for level in &self.levels {
            level.serialize_with_mode(&mut writer, compress)?;
        }
        self.tail.direct.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.root.serialized_size(compress)
            + self
                .levels
                .iter()
                .map(|l| l.serialized_size(compress))
                .sum::<usize>()
            + self.tail.direct.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid> Valid for HachiBatchedProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.root.check()?;
        for level in &self.levels {
            level.check()?;
        }
        if let Some(first) = self.levels.first() {
            if !self
                .root
                .next_w_commitment()
                .can_decode_vec(first.level_d())
            {
                return Err(SerializationError::InvalidData(
                    "batched root proof has mismatched next-commitment dimension".to_string(),
                ));
            }
        }
        for levels in self.levels.windows(2) {
            if !levels[0]
                .next_w_commitment()
                .can_decode_vec(levels[1].level_d())
            {
                return Err(SerializationError::InvalidData(
                    "adjacent hachi levels have mismatched commitment dimensions".to_string(),
                ));
            }
        }
        self.tail.direct.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiBatchedProof<F> {
    type Context = HachiBatchedProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &HachiBatchedProofShape,
    ) -> Result<Self, SerializationError> {
        let root = HachiBatchedRootProof::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.root_shape,
        )?;
        let mut levels = Vec::with_capacity(ctx.level_shapes.len());
        for shape in &ctx.level_shapes {
            levels.push(HachiLevelProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                shape,
            )?);
        }
        let pw =
            PackedDigits::deserialize_with_mode(&mut reader, compress, validate, &ctx.tail_shape)?;
        let tail = HachiProofTail::new(pw);
        let out = Self { root, levels, tail };
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
        for level in &self.levels {
            level.serialize_with_mode(&mut writer, compress)?;
        }
        self.tail.direct.serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.levels
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
            if !levels[0]
                .next_w_commitment()
                .can_decode_vec(levels[1].level_d())
            {
                return Err(SerializationError::InvalidData(
                    "adjacent hachi levels have mismatched commitment dimensions".to_string(),
                ));
            }
        }
        self.tail.direct.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiProof<F> {
    type Context = HachiProofShape;
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &HachiProofShape,
    ) -> Result<Self, SerializationError> {
        let mut levels = Vec::with_capacity(ctx.level_shapes.len());
        for shape in &ctx.level_shapes {
            levels.push(HachiLevelProof::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                shape,
            )?);
        }
        let pw =
            PackedDigits::deserialize_with_mode(&mut reader, compress, validate, &ctx.tail_shape)?;
        let tail = HachiProofTail::new(pw);
        let out = Self { levels, tail };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Prime128Offset275;
    use crate::primitives::serialization::Valid;
    use crate::FromSmallInt;

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
