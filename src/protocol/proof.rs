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
    /// Decomposed `t̂_i` blocks from the commitment phase.
    pub t_hat: Vec<Vec<CyclotomicRing<F, D>>>,
}

/// Temporary auxiliary data the verifier needs for sumcheck output verification.
///
/// Will be removed once recursive PCS evaluation proofs replace the direct
/// oracle check at the end of each sumcheck instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SumcheckAux<F: FieldCore> {
    /// `w` coefficients (z and r coefficients, concatenated). The verifier
    /// reshapes this into sumcheck evaluation form to compute the expected
    /// output claims for F_0 and F_alpha.
    pub w: Vec<F>,
}

/// Hachi Proof for One Iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore, const D: usize> {
    /// `y_ring` from the §3.1 reduction.
    pub y_ring: CyclotomicRing<F, D>,
    /// `v = D · ŵ`.
    pub v: Vec<CyclotomicRing<F, D>>,
    /// Batched sumcheck proof (F_0 norm + F_α relation, §4.3).
    pub sumcheck_proof: SumcheckProof<F>,
    /// Temporary verifier auxiliary (will be removed with recursive PCS).
    pub sumcheck_aux: SumcheckAux<F>,
    /// Commitment to the sumcheck witness `w`.
    pub w_commitment: RingCommitment<F, D>,
}

impl<F: FieldCore + HachiSerialize, const D: usize> HachiProof<F, D> {
    /// Returns the proof size in bytes (uncompressed).
    pub fn size(&self) -> usize {
        self.v.serialized_size(Compress::No)
            + self.y_ring.serialized_size(Compress::No)
            + self.sumcheck_aux.w.serialized_size(Compress::No)
            + self.sumcheck_proof.serialized_size(Compress::No)
            + self.w_commitment.serialized_size(Compress::No)
    }
}

impl<F: FieldCore> HachiSerialize for SumcheckAux<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.w.serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.w.serialized_size(compress)
    }
}

impl<F: FieldCore> Valid for SumcheckAux<F> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for SumcheckAux<F> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            w: Vec::<F>::deserialize_with_mode(reader, compress, validate)?,
        })
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiCommitmentHint<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.t_hat.serialize_with_mode(writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.t_hat.serialized_size(compress)
    }
}

impl<F: FieldCore + Valid, const D: usize> Valid for HachiCommitmentHint<F, D> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<F: FieldCore + Valid, const D: usize> HachiDeserialize for HachiCommitmentHint<F, D> {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            t_hat: Vec::deserialize_with_mode(reader, compress, validate)?,
        })
    }
}

impl<F: FieldCore, const D: usize> HachiSerialize for HachiProof<F, D> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.y_ring.serialize_with_mode(&mut writer, compress)?;
        self.v.serialize_with_mode(&mut writer, compress)?;
        self.sumcheck_proof
            .serialize_with_mode(&mut writer, compress)?;
        self.sumcheck_aux
            .serialize_with_mode(&mut writer, compress)?;
        self.w_commitment.serialize_with_mode(&mut writer, compress)
    }
    fn serialized_size(&self, compress: Compress) -> usize {
        self.y_ring.serialized_size(compress)
            + self.v.serialized_size(compress)
            + self.sumcheck_proof.serialized_size(compress)
            + self.sumcheck_aux.serialized_size(compress)
            + self.w_commitment.serialized_size(compress)
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
        Ok(Self {
            y_ring: CyclotomicRing::deserialize_with_mode(&mut reader, compress, validate)?,
            v: Vec::deserialize_with_mode(&mut reader, compress, validate)?,
            sumcheck_proof: SumcheckProof::deserialize_with_mode(&mut reader, compress, validate)?,
            sumcheck_aux: SumcheckAux::deserialize_with_mode(&mut reader, compress, validate)?,
            w_commitment: RingCommitment::deserialize_with_mode(&mut reader, compress, validate)?,
        })
    }
}
