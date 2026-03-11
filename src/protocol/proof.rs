//! Proof structures for the Hachi protocol.

use crate::algebra::CyclotomicRing;
use crate::primitives::serialization::{Compress, SerializationError};
use crate::primitives::serialization::{Valid, Validate};
use crate::protocol::commitment::RingCommitment;
use crate::protocol::labrador::types::{
    LabradorLevelProof, LabradorProof, LabradorReductionConfig, LabradorWitness,
};
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

/// Precomputed lookup table mapping balanced digit index → field element.
///
/// Wraps `FromSmallInt::digit_lut` with convenient signed-digit indexing.
/// Index a digit `d ∈ [-b/2, b/2)` via [`get`](DigitLut::get).
pub(crate) struct DigitLut<F> {
    table: [F; 16],
    half_b: i8,
}

impl<F: FieldCore + FromSmallInt> DigitLut<F> {
    #[inline]
    pub(crate) fn new(log_basis: u32) -> Self {
        let half_b = 1i8 << (log_basis - 1);
        Self {
            table: F::digit_lut(log_basis),
            half_b,
        }
    }

    #[inline(always)]
    pub(crate) fn get(&self, d: i8) -> F {
        self.table[(d + self.half_b) as usize]
    }
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

    /// Unpack to field elements using a precomputed lookup table.
    pub fn to_field_elems<F: FieldCore + FromSmallInt>(&self) -> Vec<F> {
        let bits = self.bits_per_elem as usize;
        let mask = (1u8 << bits) - 1;
        let sign_bit = 1u8 << (bits - 1);
        let lut = DigitLut::<F>::new(self.bits_per_elem);

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
            out.push(lut.get(signed));
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

    /// Reconstruct a `RingCommitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D != ring_dim`.
    pub fn to_ring_commitment<const D: usize>(&self) -> RingCommitment<F, D> {
        RingCommitment { u: self.to_vec() }
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
/// Stores the decomposed `t̂_i` blocks as a flat `Vec<i8>` with metadata
/// about block sizes and ring dimension. Convert to/from the typed
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
        let block_sizes: Vec<usize> = hint.t_hat.iter().map(|b| b.len()).collect();
        let total_planes: usize = block_sizes.iter().sum();
        let mut data = Vec::with_capacity(total_planes * D);
        for block in &hint.t_hat {
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
        let mut t_hat = Vec::with_capacity(self.block_sizes.len());
        let mut offset = 0;
        for &block_size in &self.block_sizes {
            let mut block = Vec::with_capacity(block_size);
            for _ in 0..block_size {
                let mut plane = [0i8; D];
                plane.copy_from_slice(&self.data[offset..offset + D]);
                offset += D;
                block.push(plane);
            }
            t_hat.push(block);
        }
        HachiCommitmentHint::new(t_hat)
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
///
/// D-agnostic: ring elements are stored as [`FlatRingVec`] with their
/// ring dimension recorded. Use [`y_ring_typed`](Self::y_ring_typed),
/// [`v_typed`](Self::v_typed), and
/// [`w_commitment_typed`](Self::w_commitment_typed) to reconstruct
/// typed ring elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiLevelProof<F: FieldCore> {
    /// `y_ring` from the §3.1 reduction (ring dim = current level's D).
    pub y_ring: FlatRingVec<F>,
    /// `v = D · ŵ` (ring dim = current level's D).
    pub v: FlatRingVec<F>,
    /// Batched sumcheck proof (F_0 norm + F_α relation, §4.3).
    pub sumcheck_proof: SumcheckProof<F>,
    /// Commitment to the sumcheck witness `w`
    /// (ring dim = next level's D, may differ from y_ring/v).
    pub w_commitment: FlatRingVec<F>,
    /// Claimed evaluation of w at the sumcheck challenge point.
    pub w_eval: F,
}

impl<F: FieldCore> HachiLevelProof<F> {
    /// Construct from typed ring elements for the current level and a
    /// pre-erased `FlatRingVec` for the w-commitment (which may be at a
    /// different D).
    pub fn new<const D: usize>(
        y_ring: CyclotomicRing<F, D>,
        v: Vec<CyclotomicRing<F, D>>,
        sumcheck_proof: SumcheckProof<F>,
        w_commitment: FlatRingVec<F>,
        w_eval: F,
    ) -> Self {
        Self {
            y_ring: FlatRingVec::from_single(&y_ring),
            v: FlatRingVec::from_ring_elems(&v),
            sumcheck_proof,
            w_commitment,
            w_eval,
        }
    }

    /// Ring dimension of y_ring and v (current level).
    pub fn level_d(&self) -> usize {
        self.y_ring.ring_dim()
    }

    /// Ring dimension of the w_commitment (next level).
    pub fn w_commit_d(&self) -> usize {
        self.w_commitment.ring_dim()
    }

    /// Reconstruct typed `y_ring`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn y_ring_typed<const D: usize>(&self) -> CyclotomicRing<F, D> {
        self.y_ring.to_single()
    }

    /// Reconstruct typed `v`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn v_typed<const D: usize>(&self) -> Vec<CyclotomicRing<F, D>> {
        self.v.to_vec()
    }

    /// Reconstruct typed `w_commitment`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn w_commitment_typed<const D: usize>(&self) -> RingCommitment<F, D> {
        self.w_commitment.to_ring_commitment()
    }
}

// ---------------------------------------------------------------------------
// D-erased Greyhound / Labrador proof types for HachiProofTail
// ---------------------------------------------------------------------------

/// D-erased Greyhound evaluation proof.
///
/// Mirrors [`GreyhoundEvalProof`](crate::protocol::greyhound::GreyhoundEvalProof)
/// with ring elements stored as [`FlatRingVec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlatGreyhoundEvalProof<F: FieldCore> {
    /// Outer commitment to decomposed partial evaluations.
    pub u2: FlatRingVec<F>,
    /// Matrix row count from the Greyhound reshape.
    pub m_rows: usize,
    /// Matrix column count from the Greyhound reshape.
    pub n_cols: usize,
    /// Number of inner variables in the evaluation split.
    pub inner_vars: usize,
    /// Labrador reduction config selected by Greyhound.
    pub config: LabradorReductionConfig,
}

impl<F: FieldCore> FlatGreyhoundEvalProof<F> {
    /// Convert from the typed `GreyhoundEvalProof<F, D>`.
    pub fn from_typed<const D: usize>(
        p: &crate::protocol::greyhound::GreyhoundEvalProof<F, D>,
    ) -> Self {
        Self {
            u2: FlatRingVec::from_ring_elems(&p.u2),
            m_rows: p.m_rows,
            n_cols: p.n_cols,
            inner_vars: p.inner_vars,
            config: p.config,
        }
    }

    /// Reconstruct the typed `GreyhoundEvalProof<F, D>`.
    ///
    /// # Panics
    ///
    /// Panics if `D` does not match the stored ring dimension.
    pub fn to_typed<const D: usize>(&self) -> crate::protocol::greyhound::GreyhoundEvalProof<F, D> {
        crate::protocol::greyhound::GreyhoundEvalProof {
            u2: self.u2.to_vec(),
            m_rows: self.m_rows,
            n_cols: self.n_cols,
            inner_vars: self.inner_vars,
            config: self.config,
        }
    }

    /// Ring dimension of the stored proof elements.
    pub fn ring_dim(&self) -> usize {
        self.u2.ring_dim()
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
    /// Virtual row length after nu-reshaping.
    pub nn: usize,
    /// Per-original-row split counts from the fold plan.
    pub nu: Vec<usize>,
    /// First outer commitment.
    pub u1: FlatRingVec<F>,
    /// Second outer commitment.
    pub u2: FlatRingVec<F>,
    /// JL projection vector.
    pub jl_projection: [i64; 256],
    /// JL nonce used to regenerate projection matrix.
    pub jl_nonce: u64,
    /// Lift polynomials (constant term zeroed in proof).
    pub bb: FlatRingVec<F>,
    /// Output witness norm bound after reduction.
    pub norm_sq: u128,
}

impl<F: FieldCore> FlatLabradorLevelProof<F> {
    /// Convert from the typed `LabradorLevelProof<F, D>`.
    pub fn from_typed<const D: usize>(p: &LabradorLevelProof<F, D>) -> Self {
        Self {
            tail: p.tail,
            input_row_lengths: p.input_row_lengths.clone(),
            config: p.config,
            nn: p.nn,
            nu: p.nu.clone(),
            u1: FlatRingVec::from_ring_elems(&p.u1),
            u2: FlatRingVec::from_ring_elems(&p.u2),
            jl_projection: p.jl_projection,
            jl_nonce: p.jl_nonce,
            bb: FlatRingVec::from_ring_elems(&p.bb),
            norm_sq: p.norm_sq,
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
            nn: self.nn,
            nu: self.nu.clone(),
            u1: self.u1.to_vec(),
            u2: self.u2.to_vec(),
            jl_projection: self.jl_projection,
            jl_nonce: self.jl_nonce,
            bb: self.bb.to_vec(),
            norm_sq: self.norm_sq,
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

/// Greyhound/Labrador tail proof data for the ring dimension switch.
///
/// Produced when Hachi's folding loop stops and the remaining witness
/// is handed off to Greyhound (D'=64) + Labrador recursion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GreyhoundTail<F: FieldCore> {
    /// D-erased Greyhound evaluation proof.
    pub greyhound_proof: FlatGreyhoundEvalProof<F>,
    /// D-erased full Labrador recursive proof.
    pub labrador_proof: FlatLabradorProof<F>,
    /// Outer commitment `u1 = B * t_hat` for Labrador statement reconstruction.
    pub u1: FlatRingVec<F>,
    /// Ring-valued evaluation target from the Section 4.5 ring dimension switch.
    ///
    /// Contains D' coefficients of `ring_mle(ring_point)` where `ring_point`
    /// is the subset of the opening point indexing ring elements.
    pub eval_ring: FlatRingVec<F>,
    /// Squared L2 norm of the Greyhound witness (public bound for Labrador).
    pub beta_sq: u128,
}

/// Direct Labrador tail proof data.
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
    /// Ring-valued commitment `u = B * t_hat` at D_HANDOFF (fresh, used to rebuild constraints).
    pub u: FlatRingVec<F>,
    /// Ring-valued evaluation `y_ring` (public, used to rebuild constraints).
    pub y_ring: FlatRingVec<F>,
    /// Ring-valued evaluation target from the Section 4.5 ring dimension switch.
    pub eval_ring: FlatRingVec<F>,
    /// Labrador reduction config selected for the handoff witness.
    pub config: LabradorReductionConfig,
    /// Squared L2 norm bound of the Labrador witness.
    pub beta_sq: u128,
}

/// Proof tail: either a direct witness or a Greyhound/Labrador handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HachiProofTail<F: FieldCore> {
    /// Final witness sent in clear as packed balanced digits.
    Direct(PackedDigits),
    /// Greyhound evaluation proof + Labrador recursive proof.
    Greyhound(GreyhoundTail<F>),
    /// Direct Labrador handoff from the quadratic equation.
    Labrador(LabradorTail<F>),
}

/// Hachi PCS proof with multi-level folding.
///
/// Each level runs the full protocol (quadratic equation, ring switch,
/// sumcheck) on the previous level's witness `w`. The tail is either
/// a direct witness (packed digits) or a Greyhound/Labrador handoff.
///
/// D-agnostic: per-level ring dimensions are recorded in each
/// [`HachiLevelProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HachiProof<F: FieldCore> {
    /// Per-level proofs, from the original polynomial (level 0) through
    /// recursive w-openings.
    pub levels: Vec<HachiLevelProof<F>>,
    /// Proof tail: direct witness or Greyhound/Labrador.
    pub tail: HachiProofTail<F>,
}

impl<F: FieldCore> HachiProof<F> {
    /// Access the direct final witness.
    ///
    /// # Panics
    ///
    /// Panics if the proof uses a Greyhound tail instead of a direct witness.
    pub fn final_w(&self) -> &PackedDigits {
        match &self.tail {
            HachiProofTail::Direct(pw) => pw,
            HachiProofTail::Greyhound(_) | HachiProofTail::Labrador(_) => {
                panic!("final_w called on proof with non-direct tail")
            }
        }
    }

    /// Whether this proof uses a Greyhound or Labrador tail (not a direct witness).
    pub fn has_handoff_tail(&self) -> bool {
        matches!(
            &self.tail,
            HachiProofTail::Greyhound(_) | HachiProofTail::Labrador(_)
        )
    }

    /// Whether this proof uses the Greyhound/Labrador tail.
    pub fn has_greyhound_tail(&self) -> bool {
        matches!(&self.tail, HachiProofTail::Greyhound(_))
    }

    /// Whether this proof uses the direct Labrador tail.
    pub fn has_labrador_tail(&self) -> bool {
        matches!(&self.tail, HachiProofTail::Labrador(_))
    }
}

impl<F: FieldCore + HachiSerialize> HachiProof<F> {
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
        match &self.tail {
            HachiProofTail::Direct(pw) => levels_size + pw.serialized_size(Compress::No),
            HachiProofTail::Greyhound(_tail) => {
                // TODO: implement proper size calculation for Greyhound tail
                levels_size
            }
            HachiProofTail::Labrador(tail) => {
                let nc = Compress::No;
                let labrador_size = tail
                    .labrador_proof
                    .levels
                    .iter()
                    .map(|lv| {
                        lv.u1.serialized_size(nc)
                            + lv.u2.serialized_size(nc)
                            + lv.bb.serialized_size(nc)
                            + 8 // jl_nonce
                            + 256 * 8 // jl_projection
                            + 16 // norm_sq
                    })
                    .sum::<usize>()
                    + tail
                        .labrador_proof
                        .final_opening_witness
                        .rows
                        .iter()
                        .map(|r| r.serialized_size(nc))
                        .sum::<usize>();
                levels_size
                    + tail.v.serialized_size(nc)
                    + tail.u.serialized_size(nc)
                    + tail.y_ring.serialized_size(nc)
                    + tail.eval_ring.serialized_size(nc)
                    + 16 // beta_sq
                    + labrador_size
            }
        }
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

impl<F: FieldCore> HachiSerialize for HachiLevelProof<F> {
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

impl<F: FieldCore> Valid for HachiLevelProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.y_ring.check()?;
        self.v.check()?;
        self.w_commitment.check()
    }
}

impl<F: FieldCore + Valid> HachiDeserialize for HachiLevelProof<F> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(Self {
            y_ring: FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?,
            v: FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?,
            sumcheck_proof: SumcheckProof::deserialize_with_mode(&mut reader, compress, validate)?,
            w_commitment: FlatRingVec::deserialize_with_mode(&mut reader, compress, validate)?,
            w_eval: F::deserialize_with_mode(&mut reader, compress, validate)?,
        })
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
            HachiProofTail::Greyhound(_tail) => {
                1u8.serialize_with_mode(&mut writer, compress)?;
                // TODO: serialize Greyhound tail
                Ok(())
            }
            HachiProofTail::Labrador(_tail) => {
                2u8.serialize_with_mode(&mut writer, compress)?;
                // TODO: serialize Labrador tail
                Ok(())
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
            HachiProofTail::Greyhound(_tail) => base, // TODO
            HachiProofTail::Labrador(_tail) => base,  // TODO
        }
    }
}

impl<F: FieldCore> Valid for HachiProof<F> {
    fn check(&self) -> Result<(), SerializationError> {
        for lp in &self.levels {
            lp.check()?;
        }
        match &self.tail {
            HachiProofTail::Direct(pw) => pw.check(),
            HachiProofTail::Greyhound(_) => Ok(()),
            HachiProofTail::Labrador(_) => Ok(()),
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
            1 => {
                // TODO: deserialize Greyhound tail
                return Err(SerializationError::InvalidData(
                    "Greyhound tail deserialization not yet implemented".to_string(),
                ));
            }
            2 => {
                // TODO: deserialize Labrador tail
                return Err(SerializationError::InvalidData(
                    "Labrador tail deserialization not yet implemented".to_string(),
                ));
            }
            _ => {
                return Err(SerializationError::InvalidData(format!(
                    "unknown proof tail tag: {tag}"
                )));
            }
        };
        Ok(Self { levels, tail })
    }
}
