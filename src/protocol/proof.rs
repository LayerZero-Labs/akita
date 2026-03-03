//! Proof structures for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::primitives::serialization::{Compress, SerializationError};
use crate::primitives::serialization::{Valid, Validate};
use crate::protocol::commitment::RingCommitment;
use crate::protocol::sumcheck::SumcheckProof;
use crate::{FieldCore, FromSmallInt, HachiDeserialize, HachiSerialize};
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
    /// Bits per element (= `log_basis` from the commitment config).
    pub bits_per_elem: u32,
    /// Bit-packed two's-complement data.
    pub data: Vec<u8>,
}

impl PackedDigits {
    /// Pack balanced i8 digits into bit-packed form.
    ///
    /// Each element must be in `[-b/2, b/2)` where `b = 2^log_basis`.
    ///
    /// # Panics
    ///
    /// Panics (in debug) if any element does not fit in `log_basis` bits.
    pub fn from_i8_digits(w: &[i8], log_basis: u32) -> Self {
        assert!(log_basis > 0 && log_basis <= 7, "log_basis out of range");
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

    /// Unpack to field elements via `F::from_i64`.
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

/// Prover-side hint produced at commitment time.
///
/// Contains the decomposed inner-Ajtai outputs `t̂_i` needed by the
/// ring-switch step of the prover. The polynomial itself (ring coefficients)
/// is passed separately to `prove` via `HachiPolyOps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiCommitmentHint<F: FieldCore, const D: usize> {
    /// Decomposed `t̂_i` blocks from the commitment phase as i8 digit planes.
    pub t_hat: Vec<Vec<[i8; D]>>,
    _marker: PhantomData<F>,
}

impl<F: FieldCore, const D: usize> HachiCommitmentHint<F, D> {
    /// Construct a new hint from i8 digit plane blocks.
    pub fn new(t_hat: Vec<Vec<[i8; D]>>) -> Self {
        Self {
            t_hat,
            _marker: PhantomData,
        }
    }
}

/// Proof for a single fold level (quad_eq + ring_switch + sumcheck).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiLevelProof<F: FieldCore, const D: usize> {
    /// `y_ring` from the §3.1 reduction.
    pub y_ring: CyclotomicRing<F, D>,
    /// `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Batched sumcheck proof (F_0 norm + F_α relation, §4.3).
    pub sumcheck_proof: SumcheckProof<F>,
    /// Commitment to the sumcheck witness `w`.
    pub w_commitment: RingCommitment<F, D>,
    /// Claimed evaluation of w at the sumcheck challenge point.
    /// Used by the verifier to check the expected output claim without
    /// needing the full w vector.
    pub w_eval: F,
}

/// Hachi PCS proof with multi-level folding.
///
/// Each level runs the full protocol (quadratic equation, ring switch,
/// sumcheck) on the previous level's witness `w`. The final level sends
/// `w` directly for the verifier to check, packed as balanced digits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// Per-level proofs, from the original polynomial (level 0) through
    /// recursive w-openings.
    pub levels: Vec<HachiLevelProof<F, D>>,
    /// The witness vector at the deepest fold level, bit-packed as balanced
    /// digits in `[-b/2, b/2)`. Use [`PackedDigits::to_field_elems`] to
    /// reconstruct `Vec<F>`.
    pub final_w: PackedDigits,
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        let levels_size: usize = self
            .levels
            .iter()
            .map(|lp| {
                lp.y_ring.serialized_size(Compress::No)
                    + lp.v.serialized_size(Compress::No)
                    + lp.sumcheck_proof.serialized_size(Compress::No)
                    + lp.w_commitment.serialized_size(Compress::No)
                    + lp.w_eval.serialized_size(Compress::No)
            })
            .sum();
        levels_size + self.final_w.serialized_size(Compress::No)
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiCommitmentHint<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.t_hat.len() as u64).serialize_with_mode(&mut writer, compress)?;
        for block in &self.t_hat {
            (block.len() as u64).serialize_with_mode(&mut writer, compress)?;
            for plane in block {
                // Safety: i8 and u8 have identical layout.
                let bytes: &[u8] =
                    unsafe { std::slice::from_raw_parts(plane.as_ptr().cast::<u8>(), D) };
                writer.write_all(bytes)?;
            }
        }
        Ok(())
    }
    fn serialized_size(&self, _compress: Compress) -> usize {
        8 + self
            .t_hat
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
        let mut t_hat = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let block_len = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
            let mut block = Vec::with_capacity(block_len);
            for _ in 0..block_len {
                let mut plane = [0i8; D];
                // Safety: i8 and u8 have identical layout.
                let bytes: &mut [u8] =
                    unsafe { std::slice::from_raw_parts_mut(plane.as_mut_ptr().cast::<u8>(), D) };
                reader.read_exact(bytes)?;
                block.push(plane);
            }
            t_hat.push(block);
        }
        Ok(Self::new(t_hat))
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiLevelProof<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring.serialize_with_mode(&mut writer, compress)?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        self.sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.w_commitment
            .serialize_with_mode(&mut writer, compress)?;
        self.w_eval.serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_ring.serialized_size(compress)
            + self.v.serialized_size(compress)
            + self.sumcheck_proof.serialized_size(compress)
            + self.w_commitment.serialized_size(compress)
            + self.w_eval.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiLevelProof<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiLevelProof<F, D> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            y_ring: CyclotomicRing::deserialize_with_mode(&mut reader, compress, validate)?,
            v: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            sumcheck_proof: SumcheckProof::deserialize_with_mode(&mut reader, compress, validate)?,
            w_commitment: RingCommitment::deserialize_with_mode(&mut reader, compress, validate)?,
            w_eval: F::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiProof<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        (self.levels.len() as u32).serialize_with_mode(&mut writer, compress)?;
        for level in &self.levels {
            level.serialize_with_mode(&mut writer, compress)?;
        }
        self.final_w.serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        4 + self
            .levels
            .iter()
            .map(|l| l.serialized_size(compress))
            .sum::<usize>()
            + self.final_w.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiProof<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiProof<F, D> {
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
        let final_w = PackedDigits::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self { levels, final_w })
    }
}
