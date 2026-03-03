//! Proof structures for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::primitives::serialization::{Compress, SerializationError};
use crate::primitives::serialization::{Valid, Validate};
use crate::protocol::commitment::RingCommitment;
use crate::protocol::sumcheck::SumcheckProof;
use crate::{FieldCore, HachiDeserialize, HachiSerialize};
use std::io::{Read, Write};

/// Prover-side hint produced at commitment time.
///
/// Contains the decomposed inner-Ajtai outputs `t̂_i` needed by the
/// ring-switch step of the prover. The polynomial itself (ring coefficients)
/// is passed separately to `prove` via `HachiPolyOps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiCommitmentHint<F: FieldCore, const D: usize> {
    /// Decomposed `t̂_i` blocks from the commitment phase as i8 digit planes.
    pub t_hat: Vec<Vec<[i8; D]>>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: FieldCore, const D: usize> HachiCommitmentHint<F, D> {
    /// Construct a new hint from i8 digit plane blocks.
    pub fn new(t_hat: Vec<Vec<[i8; D]>>) -> Self {
        Self {
            t_hat,
            _marker: std::marker::PhantomData,
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
/// `w` directly for the verifier to check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// Per-level proofs, from the original polynomial (level 0) through
    /// recursive w-openings.
    pub levels: Vec<HachiLevelProof<F, D>>,
    /// The witness vector at the deepest fold level, sent directly to the
    /// verifier for the final oracle check.
    pub final_w: Vec<F>,
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
                    unsafe { std::slice::from_raw_parts(plane.as_ptr() as *const u8, D) };
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
                    unsafe { std::slice::from_raw_parts_mut(plane.as_mut_ptr() as *mut u8, D) };
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
        let final_w = Vec::<F>::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self { levels, final_w })
    }
}
