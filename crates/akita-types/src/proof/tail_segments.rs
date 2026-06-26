//! Segment-typed terminal witness layout, sizing, expansion, and construction.

use std::io::Write;

use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};

use crate::descriptor_bytes::{push_u32, push_usize};
use crate::golomb_rice::{
    analyze_z_fold_golomb_encoding, golomb_rice_decode_vec, golomb_rice_encode_vec,
    golomb_rice_max_quotient_for_cap, golomb_rice_rows_admit_terminal_wire,
    golomb_rice_total_wire_bits, golomb_rice_values_within_cap, golomb_rice_zigzag_width,
    tail_z_planner_bits_per_coord, ZFoldEncodingStats,
};
use crate::instance_descriptor::FoldLinfProtocolBinding;
use crate::layout::field_bytes;
use crate::proof::CleartextWitnessShape;
use crate::proof::{FlatRingVec, TerminalWitnessTranscriptParts};
use crate::sis::compute_num_digits_full_field;
use crate::tail_golomb_rice_low_bits::{cap_rice_low_bits, wire_rice_low_bits_from_rule};
use crate::{LevelParams, MRowLayout};

/// Public segment geometry for a transparent terminal witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    pub log_basis: u32,
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    pub r_field_elems: usize,
    /// Hypercube length after expansion to digit planes (legacy packed layout used the same count).
    pub logical_num_elems: usize,
}

/// Shape for a segment-typed terminal witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTypedWitnessShape {
    pub layout: TailSegmentLayout,
    pub z_payload_bytes: usize,
}

/// Segment-typed terminal witness carried on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTypedWitness<F: FieldCore> {
    pub layout: TailSegmentLayout,
    pub z_payload: Vec<u8>,
    pub e_fields: FlatRingVec<F>,
    pub t_fields: FlatRingVec<F>,
    pub r_fields: FlatRingVec<F>,
}

impl TailSegmentLayout {
    /// Append canonical Fiat-Shamir descriptor bytes (fixed little-endian).
    ///
    /// Single source of truth for the layout field order shared by the
    /// schedule digest and [`AkitaSerialize`].
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.ring_dimension);
        push_u32(bytes, self.log_basis);
        push_usize(bytes, self.z_coords);
        push_usize(bytes, self.e_field_elems);
        push_usize(bytes, self.t_field_elems);
        push_usize(bytes, self.r_field_elems);
        push_usize(bytes, self.logical_num_elems);
    }
}

impl Valid for TailSegmentLayout {
    fn check(&self) -> Result<(), SerializationError> {
        if self.ring_dimension == 0 {
            return Err(SerializationError::InvalidData(
                "tail segment layout has zero ring dimension".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for TailSegmentLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.ring_dimension
            .serialize_with_mode(&mut writer, compress)?;
        self.log_basis.serialize_with_mode(&mut writer, compress)?;
        self.z_coords.serialize_with_mode(&mut writer, compress)?;
        self.e_field_elems
            .serialize_with_mode(&mut writer, compress)?;
        self.t_field_elems
            .serialize_with_mode(&mut writer, compress)?;
        self.r_field_elems
            .serialize_with_mode(&mut writer, compress)?;
        self.logical_num_elems
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.ring_dimension.serialized_size(compress)
            + self.log_basis.serialized_size(compress)
            + self.z_coords.serialized_size(compress)
            + self.e_field_elems.serialized_size(compress)
            + self.t_field_elems.serialized_size(compress)
            + self.r_field_elems.serialized_size(compress)
            + self.logical_num_elems.serialized_size(compress)
    }
}

impl AkitaDeserialize for TailSegmentLayout {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let ring_dimension = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let log_basis = u32::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let z_coords = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let e_field_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let t_field_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let r_field_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let logical_num_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            ring_dimension,
            log_basis,
            z_coords,
            e_field_elems,
            t_field_elems,
            r_field_elems,
            logical_num_elems,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl SegmentTypedWitnessShape {
    /// Append canonical Fiat-Shamir descriptor bytes (fixed little-endian).
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.layout.append_descriptor_bytes(bytes);
        push_usize(bytes, self.z_payload_bytes);
    }
}

impl Valid for SegmentTypedWitnessShape {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()?;
        if self.layout.z_coords == 0 {
            return Err(SerializationError::InvalidData(
                "tail segment z_coords is zero".to_string(),
            ));
        }
        Ok(())
    }
}

impl AkitaSerialize for SegmentTypedWitnessShape {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.layout.serialize_with_mode(&mut writer, compress)?;
        self.z_payload_bytes
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.layout.serialized_size(compress) + self.z_payload_bytes.serialized_size(compress)
    }
}

impl AkitaDeserialize for SegmentTypedWitnessShape {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let layout =
            TailSegmentLayout::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let z_payload_bytes = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            layout,
            z_payload_bytes,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid> Valid for SegmentTypedWitness<F> {
    fn check(&self) -> Result<(), SerializationError> {
        SegmentTypedWitnessShape {
            layout: self.layout,
            z_payload_bytes: self.z_payload.len(),
        }
        .check()?;
        if self.e_fields.coeff_len() != self.layout.e_field_elems {
            return Err(SerializationError::InvalidData(
                "e segment field length mismatch".to_string(),
            ));
        }
        if self.t_fields.coeff_len() != self.layout.t_field_elems {
            return Err(SerializationError::InvalidData(
                "t segment field length mismatch".to_string(),
            ));
        }
        if self.r_fields.coeff_len() != self.layout.r_field_elems {
            return Err(SerializationError::InvalidData(
                "r segment field length mismatch".to_string(),
            ));
        }
        Ok(())
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize> AkitaSerialize for SegmentTypedWitness<F> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.append_wire_segments(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.z_payload
            .len()
            .serialized_size(compress)
            .saturating_add(self.z_payload.len())
            .saturating_add(
                (self.layout.e_field_elems + self.layout.t_field_elems + self.layout.r_field_elems)
                    .saturating_mul(field_bytes(F::modulus_bits())),
            )
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for SegmentTypedWitness<F>
{
    type Context = SegmentTypedWitnessShape;

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &SegmentTypedWitnessShape,
    ) -> Result<Self, SerializationError> {
        if matches!(validate, Validate::Yes) {
            ctx.check()?;
        }
        let z_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        if z_len > ctx.z_payload_bytes {
            return Err(SerializationError::InvalidData(format!(
                "segment-typed z payload length {z_len} exceeds scheduled budget {}",
                ctx.z_payload_bytes
            )));
        }
        let mut z_payload = vec![0u8; z_len];
        reader.read_exact(&mut z_payload)?;
        let e_fields = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.layout.e_field_elems,
        )?;
        let t_fields = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.layout.t_field_elems,
        )?;
        let r_fields = FlatRingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.layout.r_field_elems,
        )?;
        let out = Self {
            layout: ctx.layout,
            z_payload,
            e_fields,
            t_fields,
            r_fields,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize> SegmentTypedWitness<F> {
    /// Canonical segment bytes in wire order (`z ‖ e ‖ t ‖ r`).
    pub fn wire_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.append_wire_segments(&mut out, Compress::No)
            .expect("in-memory segment serialization cannot fail");
        out
    }

    pub(crate) fn append_wire_segments<W: Write>(
        &self,
        writer: &mut W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.z_payload
            .len()
            .serialize_with_mode(&mut *writer, compress)?;
        writer.write_all(&self.z_payload)?;
        append_field_coeffs(writer, self.e_fields.coeffs(), compress)?;
        append_field_coeffs(writer, self.t_fields.coeffs(), compress)?;
        append_field_coeffs(writer, self.r_fields.coeffs(), compress)?;
        Ok(())
    }

    /// Split wire bytes into transcript-bound `e` and remainder segments.
    pub fn terminal_transcript_parts(&self) -> Result<TerminalWitnessTranscriptParts, AkitaError> {
        let e_hat = field_segment_bytes(&self.e_fields);
        if e_hat.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        let mut remainder = Vec::new();
        remainder.extend_from_slice(&self.z_payload);
        append_field_coeffs_vec(&mut remainder, self.t_fields.coeffs())?;
        append_field_coeffs_vec(&mut remainder, self.r_fields.coeffs())?;
        if remainder.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(TerminalWitnessTranscriptParts { e_hat, remainder })
    }
}

fn append_field_coeffs<F: FieldCore + AkitaSerialize, W: Write>(
    writer: &mut W,
    coeffs: &[F],
    compress: Compress,
) -> Result<(), SerializationError> {
    for coeff in coeffs {
        coeff
            .serialize_with_mode(&mut *writer, compress)
            .map_err(|_| {
                SerializationError::InvalidData("field coeff serialize failed".to_string())
            })?;
    }
    Ok(())
}

fn append_field_coeffs_vec<F: FieldCore + AkitaSerialize>(
    out: &mut Vec<u8>,
    coeffs: &[F],
) -> Result<(), AkitaError> {
    for coeff in coeffs {
        coeff
            .serialize_with_mode(&mut *out, Compress::No)
            .map_err(|_| AkitaError::InvalidProof)?;
    }
    Ok(())
}

fn field_segment_bytes<F: FieldCore + AkitaSerialize>(fields: &FlatRingVec<F>) -> Vec<u8> {
    let mut out = Vec::new();
    append_field_coeffs_vec(&mut out, fields.coeffs()).expect("in-memory field serialization");
    out
}

/// Canonical transcript bytes for the terminal `e_folded` (`e`) segment.
///
/// Both the prover terminal absorb and the verifier's decoded-witness replay
/// route through this single routine, so the bound `e_hat` bytes are identical
/// by construction (it mirrors the `e_fields` the segment witness carries).
///
/// # Errors
///
/// Propagates field serialization failures as [`AkitaError::InvalidProof`].
pub fn e_folded_segment_bytes<F, const D: usize>(
    e_folded: &[CyclotomicRing<F, D>],
) -> Result<Vec<u8>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
{
    let fields = FlatRingVec::from_ring_elems(e_folded).into_compact();
    let mut out = Vec::new();
    append_field_coeffs_vec(&mut out, fields.coeffs())?;
    Ok(out)
}

/// `num_t_vectors` for terminal fold grind when Golomb encodability must match witness build.
///
/// Returns `None` for non-terminal layouts, non-segment-typed tails, or callers without a
/// scheduled shape (ZK packed-digit tails, unit tests).
///
/// # Errors
///
/// Propagates layout multiplicity errors.
pub fn terminal_golomb_grind_tail_t_vectors(
    lp: &LevelParams,
    m_row_layout: MRowLayout,
    witness_shape: Option<&CleartextWitnessShape>,
) -> Result<Option<usize>, AkitaError> {
    if !matches!(m_row_layout, MRowLayout::WithoutDBlock) {
        return Ok(None);
    }
    let Some(shape) = witness_shape else {
        return Ok(None);
    };
    let CleartextWitnessShape::SegmentTyped(scheduled) = shape else {
        return Ok(None);
    };
    let (_, num_t_vectors, _) = tail_segment_multiplicities_from_layout(lp, &scheduled.layout)?;
    Ok(Some(num_t_vectors))
}

/// Runtime Golomb-Rice **wire** parameters for terminal `z` encode/decode.
///
/// Uses wire low bits ([`crate::wire_rice_low_bits`]); planner byte budgets use
/// [`crate::cap_rice_low_bits`] via [`segment_typed_z_payload_bytes`].
/// Rice `k` and zigzag width `W` are derived from the per-coefficient fold-response
/// cap [`crate::LevelParams::fold_witness_linf_cap_for_claims`] (`min(β_inf, t*)` or `β_inf`
/// alone), matching [`crate::sis::num_digits_fold`] and grind acceptance.
///
/// # Errors
///
/// Propagates fold cap setup errors.
pub fn tail_golomb_rice_z_params(
    lp: &LevelParams,
    num_t_vectors: usize,
) -> Result<(u32, u32), AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    let binding = FoldLinfProtocolBinding::CURRENT;
    let rice_low_bits = wire_rice_low_bits_from_rule(
        cap,
        binding.wire_rice_low_bits_rule_id,
        binding.wire_rice_low_bits_delta,
    )?;
    let w = golomb_rice_zigzag_width(cap);
    Ok((rice_low_bits, w))
}

/// Decode terminal `z` Golomb payload and enforce public admissibility.
///
/// Rejects coefficients outside the fold `‖z‖_∞` cap, unary quotients above the cap-derived
/// maximum, non-minimal byte padding, and payloads exceeding the optional schedule byte budget.
///
/// # Errors
///
/// Returns [`AkitaError::InvalidProof`] when the payload is inadmissible.
pub fn decode_terminal_z_golomb_payload(
    payload: &[u8],
    z_coords: usize,
    lp: &LevelParams,
    num_t_vectors: usize,
    budget_bytes: Option<usize>,
) -> Result<Vec<i64>, AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    let binding = FoldLinfProtocolBinding::CURRENT;
    let rice_low_bits = wire_rice_low_bits_from_rule(
        cap,
        binding.wire_rice_low_bits_rule_id,
        binding.wire_rice_low_bits_delta,
    )?;
    let zigzag_w = golomb_rice_zigzag_width(cap);
    let max_quotient = golomb_rice_max_quotient_for_cap(cap, rice_low_bits, zigzag_w)?;
    let values = golomb_rice_decode_vec(payload, z_coords, rice_low_bits, zigzag_w, max_quotient)?;
    golomb_rice_values_within_cap(&values, cap)?;
    if let Some(budget_bytes) = budget_bytes {
        if payload.len() > budget_bytes {
            return Err(AkitaError::InvalidProof);
        }
        let budget_bits = tail_z_planner_bits_per_coord(cap_rice_low_bits(cap))
            .checked_mul(z_coords)
            .ok_or(AkitaError::InvalidProof)?;
        let total_bits = golomb_rice_total_wire_bits(&values, rice_low_bits, zigzag_w)?;
        if total_bits > budget_bits {
            return Err(AkitaError::InvalidProof);
        }
    }
    Ok(values)
}

/// Decode centered fold-response `z` coefficients from a segment-typed witness.
///
/// # Errors
///
/// Propagates decode and public-parameter setup errors.
pub fn z_fold_decoded_from_segment<F: FieldCore>(
    witness: &SegmentTypedWitness<F>,
    lp: &LevelParams,
    num_t_vectors: usize,
) -> Result<Vec<i64>, AkitaError> {
    decode_terminal_z_golomb_payload(
        &witness.z_payload,
        witness.layout.z_coords,
        lp,
        num_t_vectors,
        None,
    )
}

/// Distribution / Golomb model audit for a realized segment-typed `z` payload.
///
/// # Errors
///
/// Propagates decode and public-parameter setup errors.
pub fn z_fold_encoding_stats_from_segment<F: FieldCore>(
    witness: &SegmentTypedWitness<F>,
    lp: &LevelParams,
    num_t_vectors: usize,
    field_bits: u32,
) -> Result<ZFoldEncodingStats, AkitaError> {
    let z_values = z_fold_decoded_from_segment(witness, lp, num_t_vectors)?;
    let (_, zigzag_w) = tail_golomb_rice_z_params(lp, num_t_vectors)?;
    let witness_linf_cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    let depth_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    analyze_z_fold_golomb_encoding(
        &z_values,
        witness_linf_cap,
        zigzag_w,
        depth_fold,
        lp.log_basis,
        witness.z_payload.len(),
    )
}

/// Derive the terminal tail segment layout from public schedule data.
///
/// # Errors
///
/// Returns an error when counts overflow or digit depths are zero.
pub fn tail_segment_layout(
    lp: &LevelParams,
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_public_rows: usize,
    num_commitment_groups: usize,
    field_bits: u32,
) -> Result<TailSegmentLayout, AkitaError> {
    let d = lp.ring_dimension;
    if d == 0 {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout has zero ring dimension".to_string(),
        ));
    }
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let depth_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    if depth_open == 0 || depth_commit == 0 || depth_fold == 0 {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout has zero digit depth".to_string(),
        ));
    }
    let total_w_blocks = lp
        .num_blocks
        .checked_mul(num_w_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("tail e block count overflow".to_string()))?;
    let total_t_blocks = lp
        .num_blocks
        .checked_mul(num_t_vectors)
        .ok_or_else(|| AkitaError::InvalidSetup("tail t block count overflow".to_string()))?;
    let e_field_elems = total_w_blocks
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail e field count overflow".to_string()))?;
    let t_field_elems = total_t_blocks
        .checked_mul(lp.a_key.row_len())
        .and_then(|n| n.checked_mul(d))
        .ok_or_else(|| AkitaError::InvalidSetup("tail t field count overflow".to_string()))?;
    let z_coords = num_public_rows
        .checked_mul(lp.block_len)
        .and_then(|n| n.checked_mul(depth_commit))
        .and_then(|n| n.checked_mul(d))
        .ok_or_else(|| AkitaError::InvalidSetup("tail z coord count overflow".to_string()))?;
    let z_plane_rings = num_public_rows
        .checked_mul(lp.block_len)
        .and_then(|n| n.checked_mul(depth_commit))
        .and_then(|n| n.checked_mul(depth_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("tail z plane count overflow".to_string()))?;
    let e_plane_rings = total_w_blocks
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("tail e plane count overflow".to_string()))?;
    let t_plane_rings = total_t_blocks
        .checked_mul(lp.a_key.row_len())
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("tail t plane count overflow".to_string()))?;
    let r_plane_rings = lp
        .m_row_count_for(num_commitment_groups, 0, MRowLayout::WithoutDBlock)?
        .checked_mul(compute_num_digits_full_field(field_bits, lp.log_basis))
        .ok_or_else(|| AkitaError::InvalidSetup("tail r plane count overflow".to_string()))?;
    let total_plane_rings = z_plane_rings
        .checked_add(e_plane_rings)
        .and_then(|n| n.checked_add(t_plane_rings))
        .and_then(|n| n.checked_add(r_plane_rings))
        .ok_or_else(|| AkitaError::InvalidSetup("tail logical plane overflow".to_string()))?;
    let logical_num_elems = total_plane_rings
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail logical elem overflow".to_string()))?;
    let r_field_elems = lp
        .m_row_count_for(num_commitment_groups, 0, MRowLayout::WithoutDBlock)?
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail r field count overflow".to_string()))?;
    Ok(TailSegmentLayout {
        ring_dimension: d,
        log_basis: lp.log_basis,
        z_coords,
        e_field_elems,
        t_field_elems,
        r_field_elems,
        logical_num_elems,
    })
}

/// Recover tail multiplicities from a committed [`TailSegmentLayout`].
///
/// # Errors
///
/// Returns an error when the layout is inconsistent with `lp`.
pub fn tail_segment_multiplicities_from_layout(
    lp: &LevelParams,
    layout: &TailSegmentLayout,
) -> Result<(usize, usize, usize), AkitaError> {
    let d = layout.ring_dimension;
    if d == 0 || lp.num_blocks == 0 {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout has zero ring dimension or block count".to_string(),
        ));
    }
    let e_unit = d
        .checked_mul(lp.num_blocks)
        .ok_or_else(|| AkitaError::InvalidSetup("tail e unit overflow".to_string()))?;
    if !layout.e_field_elems.is_multiple_of(e_unit) {
        return Err(AkitaError::InvalidProof);
    }
    let num_w_vectors = layout.e_field_elems / e_unit;

    let t_unit = e_unit
        .checked_mul(lp.a_key.row_len())
        .ok_or_else(|| AkitaError::InvalidSetup("tail t unit overflow".to_string()))?;
    if !layout.t_field_elems.is_multiple_of(t_unit) {
        return Err(AkitaError::InvalidProof);
    }
    let num_t_vectors = layout.t_field_elems / t_unit;

    let z_unit = lp
        .block_len
        .checked_mul(lp.num_digits_commit)
        .and_then(|n| n.checked_mul(d))
        .ok_or_else(|| AkitaError::InvalidSetup("tail z unit overflow".to_string()))?;
    if !layout.z_coords.is_multiple_of(z_unit) {
        return Err(AkitaError::InvalidProof);
    }
    let num_public_rows = layout.z_coords / z_unit;

    Ok((num_w_vectors, num_t_vectors, num_public_rows))
}

/// Planner byte budget for the Golomb-coded terminal `z` segment.
///
/// Uses cap-derived low bits plus the average-case `cap_rice_low_bits + 2` bits/coord model so schedules
/// stay conservative across field families; on-wire encode/decode uses [`crate::wire_rice_low_bits`].
///
/// # Errors
///
/// Propagates fold cap setup errors.
pub fn segment_typed_z_payload_bytes(
    lp: &LevelParams,
    layout: &TailSegmentLayout,
    num_t_vectors: usize,
) -> Result<usize, AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    let low_bits_cap = cap_rice_low_bits(cap);
    let bits_per_coord = tail_z_planner_bits_per_coord(low_bits_cap);
    Ok(layout.z_coords.saturating_mul(bits_per_coord).div_ceil(8))
}

/// Serialized byte size for a segment-typed tail witness at a fixed `z` budget.
#[must_use]
pub fn segment_typed_witness_upper_bound_bytes(
    field_bits: u32,
    layout: &TailSegmentLayout,
    z_payload_bytes: usize,
) -> usize {
    let raw_elems = layout
        .e_field_elems
        .saturating_add(layout.t_field_elems)
        .saturating_add(layout.r_field_elems);
    raw_elems
        .saturating_mul(field_bytes(field_bits))
        .saturating_add(z_payload_bytes)
        .saturating_add(8)
}

/// Recompose balanced digit planes into a signed integer.
#[cfg(test)]
#[must_use]
pub(crate) fn recompose_balanced_i8_digits(digits: &[i8], log_basis: u32) -> i64 {
    let b = 1i128 << log_basis;
    let half_b = 1i128 << (log_basis - 1);
    let mut acc = 0i128;
    let mut pow = 1i128;
    for &digit in digits {
        let mut balanced = i128::from(digit);
        if balanced >= half_b {
            balanced -= b;
        }
        acc += balanced * pow;
        pow *= b;
    }
    acc as i64
}

/// Split a recomposed integer into balanced base-`2^log_basis` digits.
fn balanced_digits_from_i64(value: i64, num_digits: usize, log_basis: u32) -> Vec<i8> {
    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;
    let mut digits = Vec::with_capacity(num_digits);
    let mut c = i128::from(value);
    for _ in 0..num_digits {
        let d = c & mask;
        let balanced = if d >= half_b { d - b } else { d };
        c = (c - balanced) >> log_basis;
        digits.push(balanced as i8);
    }
    digits
}

/// Build Golomb-Rice `z` payload from centered fold-response ring coefficients.
pub(crate) fn encode_z_segment_from_centered<const D: usize>(
    centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    rice_low_bits: u32,
    zigzag_w_z: u32,
) -> Result<Vec<u8>, AkitaError> {
    let inner_width = block_len * depth_commit;
    if !centered.len().is_multiple_of(inner_width) {
        return Err(AkitaError::InvalidInput(
            "z_folded length does not match layout".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(centered.len() * D);
    for z_j in centered {
        for &coeff in z_j.iter() {
            values.push(i64::from(coeff));
        }
    }
    golomb_rice_encode_vec(&values, rice_low_bits, zigzag_w_z)
}

/// Construct a segment-typed terminal witness from ring-switch outputs.
///
/// # Errors
///
/// Returns an error when layout counts do not match the supplied witness parts.
#[allow(clippy::too_many_arguments)]
pub fn build_segment_typed_witness<const D: usize, F>(
    e_folded: &[CyclotomicRing<F, D>],
    recomposed_inner_rows: &[Vec<CyclotomicRing<F, D>>],
    z_folded_centered: &[[i32; D]],
    r: &[CyclotomicRing<F, D>],
    lp: &LevelParams,
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_public_rows: usize,
    num_commitment_groups: usize,
) -> Result<SegmentTypedWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
{
    let field_bits = F::modulus_bits();
    let layout = tail_segment_layout(
        lp,
        num_w_vectors,
        num_t_vectors,
        num_public_rows,
        num_commitment_groups,
        field_bits,
    )?;
    let (rice_low_bits, zigzag_w_z) = tail_golomb_rice_z_params(lp, num_t_vectors)?;
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    golomb_rice_rows_admit_terminal_wire(z_folded_centered, cap)?;
    let depth_commit = lp.num_digits_commit;
    let z_payload = encode_z_segment_from_centered(
        z_folded_centered,
        lp.block_len,
        depth_commit,
        rice_low_bits,
        zigzag_w_z,
    )?;
    let e_fields = FlatRingVec::from_ring_elems(e_folded).into_compact();
    if e_fields.coeff_len() != layout.e_field_elems {
        return Err(AkitaError::InvalidInput(
            "segment-typed e segment length mismatch".to_string(),
        ));
    }
    let mut t_rings = Vec::new();
    for block in recomposed_inner_rows {
        t_rings.extend_from_slice(block);
    }
    let t_fields = FlatRingVec::from_ring_elems(&t_rings).into_compact();
    if t_fields.coeff_len() != layout.t_field_elems {
        return Err(AkitaError::InvalidInput(
            "segment-typed t segment length mismatch".to_string(),
        ));
    }
    let r_fields = FlatRingVec::from_ring_elems(r).into_compact();
    if r_fields.coeff_len() != layout.r_field_elems {
        return Err(AkitaError::InvalidInput(
            "segment-typed r segment length mismatch".to_string(),
        ));
    }
    let witness = SegmentTypedWitness {
        layout,
        z_payload,
        e_fields,
        t_fields,
        r_fields,
    };
    let budget_bytes = segment_typed_z_payload_bytes(lp, &layout, num_t_vectors)?;
    validate_segment_typed_z_payload(&witness, lp, num_t_vectors, budget_bytes)?;
    Ok(witness)
}

/// Check a segment witness `z` payload against the schedule-bound byte budget and public
/// Golomb admissibility.
///
/// # Errors
///
/// Returns an error when the encoded `z` payload is inadmissible or exceeds the budget.
pub fn validate_segment_typed_z_payload<F: FieldCore>(
    witness: &SegmentTypedWitness<F>,
    lp: &LevelParams,
    num_t_vectors: usize,
    budget_bytes: usize,
) -> Result<(), AkitaError> {
    decode_terminal_z_golomb_payload(
        &witness.z_payload,
        witness.layout.z_coords,
        lp,
        num_t_vectors,
        Some(budget_bytes),
    )
    .map(|_| ())
    .map_err(|err| match err {
        AkitaError::InvalidProof => AkitaError::InvalidInput(format!(
            "segment-typed z payload {} bytes inadmissible or exceeds schedule budget {budget_bytes}",
            witness.z_payload.len()
        )),
        other => other,
    })
}

/// Expand a segment-typed witness into the legacy digit stream consumed by
/// stage-2 evaluation and packed-digit transcript helpers.
///
/// # Errors
///
/// Returns an error when decoding or decomposition fails.
pub fn expand_segment_typed_to_i8_digits<const D: usize, F>(
    witness: &SegmentTypedWitness<F>,
    lp: &LevelParams,
    num_commitment_groups: usize,
) -> Result<Vec<i8>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
{
    if D != witness.layout.ring_dimension {
        return Err(AkitaError::InvalidProof);
    }
    let field_bits = F::modulus_bits();
    let (num_w_vectors, num_t_vectors, num_public_rows) =
        tail_segment_multiplicities_from_layout(lp, &witness.layout)?;
    let expected_layout = tail_segment_layout(
        lp,
        num_w_vectors,
        num_t_vectors,
        num_public_rows,
        num_commitment_groups,
        field_bits,
    )?;
    if expected_layout != witness.layout {
        return Err(AkitaError::InvalidProof);
    }
    let log_basis = lp.log_basis;
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let num_digits_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    let levels = compute_num_digits_full_field(field_bits, log_basis);
    let budget_bytes = segment_typed_z_payload_bytes(lp, &witness.layout, num_t_vectors)?;
    let z_values = decode_terminal_z_golomb_payload(
        &witness.z_payload,
        witness.layout.z_coords,
        lp,
        num_t_vectors,
        Some(budget_bytes),
    )?;
    let inner_width = lp.block_len * depth_commit;
    let total_z_elems = z_values.len() / D;
    if total_z_elems * D != z_values.len() || !total_z_elems.is_multiple_of(inner_width) {
        return Err(AkitaError::InvalidProof);
    }
    let mut all_z_planes = vec![[0i8; D]; total_z_elems * num_digits_fold];
    for (elem_idx, chunk) in z_values.chunks_exact(D).enumerate() {
        for (coeff_idx, &value) in chunk.iter().enumerate() {
            let digits = balanced_digits_from_i64(value, num_digits_fold, log_basis);
            for (plane_idx, digit) in digits.into_iter().enumerate() {
                all_z_planes[elem_idx * num_digits_fold + plane_idx][coeff_idx] = digit;
            }
        }
    }

    let w_block_count = num_w_vectors * lp.num_blocks;
    let e_planes = decompose_field_segment_to_planes::<F, D>(
        witness.e_fields.coeffs(),
        w_block_count,
        depth_open,
        log_basis,
    )?;
    let t_block_count = num_t_vectors * lp.num_blocks;
    let t_planes_per_block = lp.a_key.row_len() * depth_open;
    let t_planes = decompose_field_segment_to_planes::<F, D>(
        witness.t_fields.coeffs(),
        t_block_count * lp.a_key.row_len(),
        depth_open,
        log_basis,
    )?;
    if t_planes.len() != t_block_count * t_planes_per_block {
        return Err(AkitaError::InvalidProof);
    }

    let r_rings = witness
        .r_fields
        .coeffs()
        .chunks_exact(D)
        .map(|chunk| {
            let coeffs: [F; D] = chunk.try_into().map_err(|_| AkitaError::InvalidProof)?;
            Ok(CyclotomicRing::<F, D>::from_coefficients(coeffs))
        })
        .collect::<Result<Vec<_>, AkitaError>>()?;
    let mut r_planes_flat = Vec::with_capacity(r_rings.len() * levels);
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    let mut scratch = vec![[0i8; D]; levels];
    for ring in &r_rings {
        scratch.fill([0i8; D]);
        ring.balanced_decompose_pow2_i8_into_with_params(&mut scratch, &decompose_params);
        r_planes_flat.extend(scratch.iter().copied());
    }

    let mut out = Vec::with_capacity(witness.layout.logical_num_elems);
    emit_witness_z_folded_planes_inner::<D>(
        &mut out,
        &all_z_planes,
        lp.block_len,
        depth_commit,
        num_digits_fold,
        total_z_elems,
    );
    emit_witness_planes_block_inner::<D>(&mut out, &e_planes, w_block_count, depth_open);
    emit_witness_planes_block_inner::<D>(&mut out, &t_planes, t_block_count, t_planes_per_block);
    for plane in &r_planes_flat {
        out.extend_from_slice(plane);
    }
    if out.len() != witness.layout.logical_num_elems {
        return Err(AkitaError::InvalidProof);
    }
    Ok(out)
}

fn decompose_field_segment_to_planes<F, const D: usize>(
    coeffs: &[F],
    ring_count: usize,
    depth_open: usize,
    log_basis: u32,
) -> Result<Vec<[i8; D]>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
{
    if coeffs.len() != ring_count.checked_mul(D).ok_or(AkitaError::InvalidProof)? {
        return Err(AkitaError::InvalidProof);
    }
    let levels = depth_open;
    let q = (-F::one()).to_canonical_u128() + 1;
    let decompose_params = BalancedDecomposePow2I8Params::new(levels, log_basis, q);
    let mut out = Vec::with_capacity(ring_count * levels);
    let mut scratch = vec![[0i8; D]; levels];
    for chunk in coeffs.chunks_exact(D) {
        let coeffs_array: [F; D] = chunk.try_into().map_err(|_| AkitaError::InvalidProof)?;
        let ring = CyclotomicRing::<F, D>::from_coefficients(coeffs_array);
        scratch.fill([0i8; D]);
        ring.balanced_decompose_pow2_i8_into_with_params(&mut scratch, &decompose_params);
        out.extend(scratch.iter().copied());
    }
    Ok(out)
}

/// Emit digit-major block planes (block index innermost).
pub fn emit_witness_planes_block_inner<const D: usize>(
    out: &mut Vec<i8>,
    flat: &[[i8; D]],
    total_blocks: usize,
    planes_per_block: usize,
) {
    for compound_dig in 0..planes_per_block {
        for blk in 0..total_blocks {
            out.extend_from_slice(&flat[blk * planes_per_block + compound_dig]);
        }
    }
}

/// Emit folded `z` digit planes in `(dc, df, point, block)` order.
pub fn emit_witness_z_folded_planes_inner<const D: usize>(
    out: &mut Vec<i8>,
    all_planes: &[[i8; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    total_elems: usize,
) {
    let inner_width = block_len * depth_commit;
    let num_points = total_elems / inner_width;
    for dc in 0..depth_commit {
        for df in 0..num_digits_fold {
            for pt in 0..num_points {
                for blk in 0..block_len {
                    let k = pt * inner_width + blk * depth_commit + dc;
                    out.extend_from_slice(&all_planes[k * num_digits_fold + df]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusFamily;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::CanonicalField;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;

    fn test_lp() -> LevelParams {
        LevelParams::params_only(
            SisModulusFamily::Q128,
            8,
            3,
            2,
            3,
            2,
            SparseChallengeConfig::Uniform {
                weight: 3,
                nonzero_coeffs: vec![-1, 1],
            },
        )
        .with_decomp(3, 2, 2, 3, 0)
        .expect("tail segment test params")
    }

    #[test]
    fn recompose_and_split_digits_round_trip() {
        let digits = vec![-2i8, 1, 0];
        let value = recompose_balanced_i8_digits(&digits, 3);
        let back = balanced_digits_from_i64(value, digits.len(), 3);
        assert_eq!(back, digits);
    }

    #[test]
    fn segment_typed_z_budget_uses_golomb_rate_not_packed_digit_width() {
        let lp = test_lp();
        let field_bits = F::modulus_bits();
        let layout = tail_segment_layout(&lp, 1, 1, 1, 1, field_bits).unwrap();
        let z_bytes = segment_typed_z_payload_bytes(&lp, &layout, 1).unwrap();
        let depth_fold = lp.num_digits_fold(1, field_bits).unwrap();
        let packed_z = crate::layout::proof_size::packed_digits_bytes(
            layout.z_coords.saturating_mul(depth_fold),
            8,
        );
        assert_ne!(z_bytes, packed_z);
    }

    #[test]
    fn segment_typed_wire_round_trip_with_scheduled_z_budget() {
        use akita_field::CanonicalField;
        use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};

        let lp = test_lp();
        let field_bits = F::modulus_bits();
        let layout = tail_segment_layout(&lp, 1, 1, 1, 1, field_bits).unwrap();
        let scheduled_z_bytes = segment_typed_z_payload_bytes(&lp, &layout, 1).unwrap();
        assert!(
            scheduled_z_bytes > 16,
            "test expects scheduled z budget to exceed a tight payload"
        );
        let (rice_low_bits, zigzag_w_z) = tail_golomb_rice_z_params(&lp, 1).unwrap();
        let centered = [[-3i32, 0, 1, 2, -1, 4, 0, 0]; 2];
        let z_payload = encode_z_segment_from_centered(
            &centered,
            1,
            lp.num_digits_commit,
            rice_low_bits,
            zigzag_w_z,
        )
        .unwrap();
        assert!(z_payload.len() < scheduled_z_bytes);
        let witness = SegmentTypedWitness {
            layout,
            z_payload,
            e_fields: FlatRingVec::from_coeffs(vec![F::zero(); layout.e_field_elems]),
            t_fields: FlatRingVec::from_coeffs(vec![F::zero(); layout.t_field_elems]),
            r_fields: FlatRingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
        };
        let scheduled_shape = SegmentTypedWitnessShape {
            layout,
            z_payload_bytes: scheduled_z_bytes,
        };
        let mut bytes = Vec::new();
        witness
            .serialize_with_mode(&mut bytes, Compress::No)
            .expect("serialize segment witness");
        let decoded = SegmentTypedWitness::<F>::deserialize_with_mode(
            &bytes[..],
            Compress::No,
            Validate::Yes,
            &scheduled_shape,
        )
        .expect("deserialize with scheduled z budget");
        assert_eq!(decoded, witness);
    }

    #[test]
    fn decode_terminal_z_rejects_coefficient_above_fold_cap() {
        use crate::golomb_rice::golomb_rice_encode_vec;

        let lp = test_lp();
        let cap = lp.fold_witness_linf_cap_for_claims(1).unwrap();
        let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
        let over_cap = cap as i64 + 1;
        let payload = golomb_rice_encode_vec(&[over_cap], rice_low_bits, zigzag_w)
            .expect("zigzag covers cap+1");
        assert!(decode_terminal_z_golomb_payload(&payload, 1, &lp, 1, None).is_err());
    }

    #[test]
    fn decode_terminal_z_rejects_trailing_zero_byte_padding() {
        use crate::golomb_rice::golomb_rice_encode_vec;

        let lp = test_lp();
        let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
        let mut payload = golomb_rice_encode_vec(&[-2i64, 1, 0], rice_low_bits, zigzag_w).unwrap();
        payload.push(0x00);
        assert!(decode_terminal_z_golomb_payload(&payload, 3, &lp, 1, None).is_err());
    }

    #[test]
    fn expand_segment_typed_rejects_inadmissible_z_payload() {
        use crate::golomb_rice::golomb_rice_encode_vec;

        let lp = test_lp();
        let field_bits = F::modulus_bits();
        let layout = tail_segment_layout(&lp, 1, 1, 1, 1, field_bits).unwrap();
        let (rice_low_bits, zigzag_w) = tail_golomb_rice_z_params(&lp, 1).unwrap();
        let cap = lp.fold_witness_linf_cap_for_claims(1).unwrap();
        let z_payload = golomb_rice_encode_vec(&[cap as i64 + 1], rice_low_bits, zigzag_w).unwrap();
        let witness = SegmentTypedWitness {
            layout,
            z_payload,
            e_fields: FlatRingVec::from_coeffs(vec![F::zero(); layout.e_field_elems]),
            t_fields: FlatRingVec::from_coeffs(vec![F::zero(); layout.t_field_elems]),
            r_fields: FlatRingVec::from_coeffs(vec![F::zero(); layout.r_field_elems]),
        };
        assert!(expand_segment_typed_to_i8_digits::<8, F>(&witness, &lp, 1).is_err());
    }
}
