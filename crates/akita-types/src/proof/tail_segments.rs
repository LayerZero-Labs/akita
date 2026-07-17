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
    golomb_rice_flat_admit_terminal_wire, golomb_rice_max_quotient_for_cap,
    golomb_rice_total_wire_bits, golomb_rice_values_within_cap, golomb_rice_zigzag_width,
    tail_z_planner_bits_per_coord, ZFoldEncodingStats,
};
use crate::instance_descriptor::FoldLinfProtocolBinding;
use crate::layout::field_bytes;
use crate::proof::CleartextWitnessShape;
use crate::proof::{RingVec, TerminalWitnessTranscriptParts};
use crate::sis::compute_num_digits_full_field;
use crate::tail_golomb_rice_low_bits::{cap_rice_low_bits, wire_rice_low_bits_from_rule};
use crate::{
    LevelParams, LevelParamsLike, RelationMatrixRowLayout, WitnessLayout, WitnessUnitLayout,
};

/// Public segment geometry for a transparent terminal witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    pub log_basis: u32,
    /// Per-group terminal segments in witness order. Scalar/single-group tails
    /// are represented as exactly one group.
    pub groups: Vec<TailSegmentGroupLayout>,
    /// Shared relation quotient tail, after all group-local z/e/t segments.
    pub r_field_elems: usize,
    /// Hypercube length after expansion to digit planes (legacy packed layout used the same count).
    pub logical_num_elems: usize,
}

/// Whether a transparent terminal witness carries quotient rows for stage 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalQuotientMode {
    /// Quotient-backed terminal stage 2 (`z | e | t | r`).
    Include,
    /// Direct reduced ring checks (`z | e | t`).
    Omit,
}

/// Per-group terminal segment geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailSegmentGroupLayout {
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    /// Scheduled byte budget for this group's Golomb-coded z payload.
    pub z_payload_bytes: usize,
}

/// Shape for a segment-typed terminal witness payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTypedWitnessShape {
    pub layout: TailSegmentLayout,
}

/// Segment-typed terminal witness carried on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTypedWitness<F: FieldCore> {
    pub layout: TailSegmentLayout,
    pub z_payloads: Vec<Vec<u8>>,
    pub e_fields: RingVec<F>,
    pub t_fields: RingVec<F>,
    pub r_fields: RingVec<F>,
}

pub struct SegmentTypedWitnessGroupParts<'a, F: FieldCore> {
    pub params: &'a dyn LevelParamsLike,
    pub num_w_vectors: usize,
    pub num_t_vectors: usize,
    pub num_z_segments: usize,
    pub e_folded: &'a RingVec<F>,
    pub recomposed_inner_rows: &'a [RingVec<F>],
    pub z_folded_centered_flat: &'a [i32],
}

impl TailSegmentLayout {
    /// Append canonical Fiat-Shamir descriptor bytes (fixed little-endian).
    ///
    /// Single source of truth for the layout field order shared by the
    /// schedule digest and [`AkitaSerialize`].
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.ring_dimension);
        push_u32(bytes, self.log_basis);
        push_usize(bytes, self.groups.len());
        for group in &self.groups {
            push_usize(bytes, group.z_coords);
            push_usize(bytes, group.e_field_elems);
            push_usize(bytes, group.t_field_elems);
            push_usize(bytes, group.z_payload_bytes);
        }
        push_usize(bytes, self.r_field_elems);
        push_usize(bytes, self.logical_num_elems);
    }

    #[must_use]
    pub fn z_coords(&self) -> usize {
        self.groups.iter().map(|group| group.z_coords).sum()
    }

    #[must_use]
    pub fn e_field_elems(&self) -> usize {
        self.groups.iter().map(|group| group.e_field_elems).sum()
    }

    #[must_use]
    pub fn t_field_elems(&self) -> usize {
        self.groups.iter().map(|group| group.t_field_elems).sum()
    }

    #[must_use]
    pub fn z_payload_bytes(&self) -> usize {
        self.groups.iter().map(|group| group.z_payload_bytes).sum()
    }

    #[must_use]
    pub fn admits_realized(&self, realized: &Self) -> bool {
        self.ring_dimension == realized.ring_dimension
            && self.log_basis == realized.log_basis
            && self.r_field_elems == realized.r_field_elems
            && self.logical_num_elems == realized.logical_num_elems
            && self.groups.len() == realized.groups.len()
            && self
                .groups
                .iter()
                .zip(&realized.groups)
                .all(|(scheduled, realized)| {
                    scheduled.z_coords == realized.z_coords
                        && scheduled.e_field_elems == realized.e_field_elems
                        && scheduled.t_field_elems == realized.t_field_elems
                        && realized.z_payload_bytes <= scheduled.z_payload_bytes
                })
    }
}

impl Valid for TailSegmentLayout {
    fn check(&self) -> Result<(), SerializationError> {
        if self.ring_dimension == 0 {
            return Err(SerializationError::InvalidData(
                "tail segment layout has zero ring dimension".to_string(),
            ));
        }
        if self.groups.is_empty() {
            return Err(SerializationError::InvalidData(
                "tail segment layout has no groups".to_string(),
            ));
        }
        for group in &self.groups {
            if group.z_coords == 0 {
                return Err(SerializationError::InvalidData(
                    "tail segment group has zero z_coords".to_string(),
                ));
            }
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
        self.groups.serialize_with_mode(&mut writer, compress)?;
        self.r_field_elems
            .serialize_with_mode(&mut writer, compress)?;
        self.logical_num_elems
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.ring_dimension.serialized_size(compress)
            + self.log_basis.serialized_size(compress)
            + self.groups.serialized_size(compress)
            + self.r_field_elems.serialized_size(compress)
            + self.logical_num_elems.serialized_size(compress)
    }
}

impl AkitaSerialize for TailSegmentGroupLayout {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.z_coords.serialize_with_mode(&mut writer, compress)?;
        self.e_field_elems
            .serialize_with_mode(&mut writer, compress)?;
        self.t_field_elems
            .serialize_with_mode(&mut writer, compress)?;
        self.z_payload_bytes
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.z_coords.serialized_size(compress)
            + self.e_field_elems.serialized_size(compress)
            + self.t_field_elems.serialized_size(compress)
            + self.z_payload_bytes.serialized_size(compress)
    }
}

impl AkitaDeserialize for TailSegmentGroupLayout {
    type Context = ();

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        _ctx: &(),
    ) -> Result<Self, SerializationError> {
        let out = Self {
            z_coords: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            e_field_elems: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            t_field_elems: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
            z_payload_bytes: usize::deserialize_with_mode(&mut reader, compress, validate, &())?,
        };
        Ok(out)
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
        let groups = Vec::<TailSegmentGroupLayout>::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &(),
        )?;
        let r_field_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let logical_num_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            ring_dimension,
            log_basis,
            groups,
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
    }
}

impl Valid for SegmentTypedWitnessShape {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()?;
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
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.layout.serialized_size(compress)
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
        let out = Self { layout };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + Valid> Valid for SegmentTypedWitness<F> {
    fn check(&self) -> Result<(), SerializationError> {
        SegmentTypedWitnessShape {
            layout: self.layout.clone(),
        }
        .check()?;
        if self.z_payloads.len() != self.layout.groups.len() {
            return Err(SerializationError::InvalidData(
                "z payload group count mismatch".to_string(),
            ));
        }
        for (payload, group) in self.z_payloads.iter().zip(&self.layout.groups) {
            if payload.len() > group.z_payload_bytes {
                return Err(SerializationError::InvalidData(
                    "z payload length exceeds scheduled budget".to_string(),
                ));
            }
        }
        if self.e_fields.coeff_len() != self.layout.e_field_elems() {
            return Err(SerializationError::InvalidData(
                "e segment field length mismatch".to_string(),
            ));
        }
        if self.t_fields.coeff_len() != self.layout.t_field_elems() {
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
        let z_bytes = self
            .z_payloads
            .iter()
            .map(|payload| {
                payload
                    .len()
                    .serialized_size(compress)
                    .saturating_add(payload.len())
            })
            .sum::<usize>();
        z_bytes.saturating_add(
            (self.layout.e_field_elems() + self.layout.t_field_elems() + self.layout.r_field_elems)
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
        let mut z_payloads = Vec::with_capacity(ctx.layout.groups.len());
        for group in &ctx.layout.groups {
            let z_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            if z_len > group.z_payload_bytes {
                return Err(SerializationError::InvalidData(format!(
                    "segment-typed z payload length {z_len} exceeds scheduled budget {}",
                    group.z_payload_bytes
                )));
            }
            let mut z_payload = vec![0u8; z_len];
            reader.read_exact(&mut z_payload)?;
            z_payloads.push(z_payload);
        }
        let e_fields = RingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.layout.e_field_elems(),
        )?;
        let t_fields = RingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.layout.t_field_elems(),
        )?;
        let r_fields = RingVec::deserialize_with_mode(
            &mut reader,
            compress,
            validate,
            &ctx.layout.r_field_elems,
        )?;
        let out = Self {
            layout: ctx.layout.clone(),
            z_payloads,
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
        for payload in &self.z_payloads {
            payload.len().serialize_with_mode(&mut *writer, compress)?;
            writer.write_all(payload)?;
        }
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
        for payload in &self.z_payloads {
            remainder.extend_from_slice(payload);
        }
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

fn field_segment_bytes<F: FieldCore + AkitaSerialize>(fields: &RingVec<F>) -> Vec<u8> {
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
pub fn e_folded_segment_bytes<F>(e_folded: &RingVec<F>) -> Result<Vec<u8>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
{
    let fields = e_folded.clone().into_compact();
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
    relation_matrix_row_layout: RelationMatrixRowLayout,
    witness_shape: Option<&CleartextWitnessShape>,
) -> Result<Option<usize>, AkitaError> {
    if !matches!(
        relation_matrix_row_layout,
        RelationMatrixRowLayout::WithoutDBlock
    ) {
        return Ok(None);
    }
    let Some(shape) = witness_shape else {
        return Ok(None);
    };
    let CleartextWitnessShape::SegmentTyped(scheduled) = shape else {
        return Ok(None);
    };
    let (_, num_t_vectors, _) = tail_segment_multiplicities_from_layout(lp, &scheduled.layout, 0)?;
    Ok(Some(num_t_vectors))
}

/// Runtime Golomb-Rice **wire** parameters for terminal `z` encode/decode.
///
/// Uses wire low bits ([`crate::wire_rice_low_bits`]); planner byte budgets use
/// [`crate::cap_rice_low_bits`] via [`segment_typed_z_payload_bytes`].
/// Rice `k` and zigzag width `W` are derived from the per-coefficient fold-response
/// cap [`crate::LevelParams::fold_witness_linf_cap_for_claims`] (`min(β_inf, t*)` or `β_inf`
/// alone), matching [`crate::sis::fold_witness_digit_plan`] and grind acceptance.
///
/// # Errors
///
/// Propagates fold cap setup errors.
pub fn tail_golomb_rice_z_params(
    lp: &LevelParams,
    num_t_vectors: usize,
) -> Result<(u32, u32), AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    tail_golomb_rice_z_params_from_cap(cap)
}

fn tail_golomb_rice_z_params_from_cap(cap: u128) -> Result<(u32, u32), AkitaError> {
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
    decode_terminal_z_golomb_payload_with_cap(payload, z_coords, cap, budget_bytes)
}

fn decode_terminal_z_golomb_payload_with_cap(
    payload: &[u8],
    z_coords: usize,
    cap: u128,
    budget_bytes: Option<usize>,
) -> Result<Vec<i64>, AkitaError> {
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
    let payload = witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?;
    let group = witness
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    decode_terminal_z_golomb_payload(payload, group.z_coords, lp, num_t_vectors, None)
}

fn z_payload_budget_from_cap(z_coords: usize, cap: u128) -> usize {
    let low_bits_cap = cap_rice_low_bits(cap);
    let bits_per_coord = tail_z_planner_bits_per_coord(low_bits_cap);
    z_coords.saturating_mul(bits_per_coord).div_ceil(8)
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
        witness.z_payloads.first().map_or(0, Vec::len),
    )
}

pub fn tail_segment_layout_from_groups<'a>(
    lp: &LevelParams,
    groups: impl IntoIterator<Item = (&'a dyn LevelParamsLike, usize, usize, usize)>,
    num_commitment_groups: usize,
    field_bits: u32,
    quotient_mode: TerminalQuotientMode,
) -> Result<TailSegmentLayout, AkitaError> {
    let d = lp.ring_dimension;
    if d == 0 {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout has zero ring dimension".to_string(),
        ));
    }
    let groups = groups.into_iter().collect::<Vec<_>>();
    if groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout requires at least one group".to_string(),
        ));
    }
    let mut group_layouts = Vec::with_capacity(groups.len());
    let mut total_plane_rings = 0usize;
    for (params, num_w_vectors, num_t_vectors, num_z_segments) in groups {
        let depth_open = params.num_digits_open();
        let depth_commit = params.num_digits_commit();
        let depth_fold = lp.num_digits_fold_for_params(params, num_t_vectors, field_bits)?;
        if depth_open == 0 || depth_commit == 0 || depth_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "tail segment layout has zero digit depth".to_string(),
            ));
        }
        let total_w_blocks = params
            .num_live_blocks()
            .checked_mul(num_w_vectors)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e block count overflow".to_string()))?;
        let total_t_blocks = params
            .num_live_blocks()
            .checked_mul(num_t_vectors)
            .ok_or_else(|| AkitaError::InvalidSetup("tail t block count overflow".to_string()))?;
        let e_field_elems = total_w_blocks
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e field count overflow".to_string()))?;
        let t_field_elems = total_t_blocks
            .checked_mul(params.a_rows_len())
            .and_then(|n| n.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("tail t field count overflow".to_string()))?;
        let z_coords = num_z_segments
            .checked_mul(params.num_positions_per_block())
            .and_then(|n| n.checked_mul(depth_commit))
            .and_then(|n| n.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("tail z coord count overflow".to_string()))?;
        let z_plane_rings = num_z_segments
            .checked_mul(params.num_positions_per_block())
            .and_then(|n| n.checked_mul(depth_commit))
            .and_then(|n| n.checked_mul(depth_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("tail z plane count overflow".to_string()))?;
        let e_plane_rings = total_w_blocks
            .checked_mul(depth_open)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e plane count overflow".to_string()))?;
        let t_plane_rings = total_t_blocks
            .checked_mul(params.a_rows_len())
            .and_then(|n| n.checked_mul(depth_open))
            .ok_or_else(|| AkitaError::InvalidSetup("tail t plane count overflow".to_string()))?;
        let z_cap = lp.fold_witness_linf_cap_for_params(params, num_t_vectors, field_bits)?;
        let z_payload_bytes = z_payload_budget_from_cap(z_coords, z_cap);
        group_layouts.push(TailSegmentGroupLayout {
            z_coords,
            e_field_elems,
            t_field_elems,
            z_payload_bytes,
        });
        total_plane_rings = total_plane_rings
            .checked_add(z_plane_rings)
            .and_then(|n| n.checked_add(e_plane_rings))
            .and_then(|n| n.checked_add(t_plane_rings))
            .ok_or_else(|| AkitaError::InvalidSetup("tail logical plane overflow".to_string()))?;
    }
    let quotient_rows = match quotient_mode {
        TerminalQuotientMode::Include => lp.relation_matrix_row_count_for(
            num_commitment_groups,
            RelationMatrixRowLayout::WithoutDBlock,
        )?,
        TerminalQuotientMode::Omit => 0,
    };
    let r_plane_rings = quotient_rows
        .checked_mul(compute_num_digits_full_field(field_bits, lp.log_basis))
        .ok_or_else(|| AkitaError::InvalidSetup("tail r plane count overflow".to_string()))?;
    let total_plane_rings = total_plane_rings
        .checked_add(r_plane_rings)
        .ok_or_else(|| AkitaError::InvalidSetup("tail logical plane overflow".to_string()))?;
    let logical_num_elems = total_plane_rings
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail logical elem overflow".to_string()))?;
    let r_field_elems = quotient_rows
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail r field count overflow".to_string()))?;
    Ok(TailSegmentLayout {
        ring_dimension: d,
        log_basis: lp.log_basis,
        groups: group_layouts,
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
    group_index: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    tail_segment_multiplicities_from_layout_for_params(lp, lp.ring_dimension, layout, group_index)
}

pub fn tail_segment_multiplicities_from_layout_for_params(
    params: &dyn LevelParamsLike,
    ring_dimension: usize,
    layout: &TailSegmentLayout,
    group_index: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    let d = layout.ring_dimension;
    if d == 0 || d != ring_dimension || params.num_live_blocks() == 0 {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout has zero ring dimension or block count".to_string(),
        ));
    }
    let group = layout
        .groups
        .get(group_index)
        .ok_or(AkitaError::InvalidProof)?;
    let e_unit = d
        .checked_mul(params.num_live_blocks())
        .ok_or_else(|| AkitaError::InvalidSetup("tail e unit overflow".to_string()))?;
    if !group.e_field_elems.is_multiple_of(e_unit) {
        return Err(AkitaError::InvalidProof);
    }
    let num_w_vectors = group.e_field_elems / e_unit;

    let t_unit = e_unit
        .checked_mul(params.a_rows_len())
        .ok_or_else(|| AkitaError::InvalidSetup("tail t unit overflow".to_string()))?;
    if !group.t_field_elems.is_multiple_of(t_unit) {
        return Err(AkitaError::InvalidProof);
    }
    let num_t_vectors = group.t_field_elems / t_unit;

    let z_unit = params
        .num_positions_per_block()
        .checked_mul(params.num_digits_commit())
        .and_then(|n| n.checked_mul(d))
        .ok_or_else(|| AkitaError::InvalidSetup("tail z unit overflow".to_string()))?;
    if !group.z_coords.is_multiple_of(z_unit) {
        return Err(AkitaError::InvalidProof);
    }
    let num_z_segments = group.z_coords / z_unit;

    Ok((num_w_vectors, num_t_vectors, num_z_segments))
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
    Ok(z_payload_budget_from_cap(layout.z_coords(), cap))
}

/// Serialized byte size for a segment-typed tail witness at a fixed `z` budget.
#[must_use]
pub fn segment_typed_witness_upper_bound_bytes(
    field_bits: u32,
    layout: &TailSegmentLayout,
    z_payload_bytes: usize,
) -> usize {
    let raw_elems = layout
        .e_field_elems()
        .saturating_add(layout.t_field_elems())
        .saturating_add(layout.r_field_elems);
    raw_elems
        .saturating_mul(field_bytes(field_bits))
        .saturating_add(z_payload_bytes)
        .saturating_add(8usize.saturating_mul(layout.groups.len()))
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

/// Build Golomb-Rice `z` payload from a flat centered coefficient stream.
fn encode_z_segment_from_centered_flat(
    centered_flat: &[i64],
    rice_low_bits: u32,
    zigzag_w_z: u32,
) -> Result<Vec<u8>, AkitaError> {
    golomb_rice_encode_vec(centered_flat, rice_low_bits, zigzag_w_z)
}

/// Construct a segment-typed terminal witness from ring-switch outputs.
///
/// # Errors
///
/// Returns an error when layout counts do not match the supplied witness parts.
#[allow(clippy::too_many_arguments)]
pub fn build_segment_typed_witness<F>(
    ring_d: usize,
    e_folded: &RingVec<F>,
    recomposed_inner_rows: &[RingVec<F>],
    z_folded_centered_flat: &[i32],
    r: &RingVec<F>,
    lp: &LevelParams,
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_z_segments: usize,
    num_commitment_groups: usize,
) -> Result<SegmentTypedWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
{
    build_segment_typed_witness_from_groups(
        ring_d,
        &[SegmentTypedWitnessGroupParts {
            params: lp,
            num_w_vectors,
            num_t_vectors,
            num_z_segments,
            e_folded,
            recomposed_inner_rows,
            z_folded_centered_flat,
        }],
        r,
        lp,
        num_commitment_groups,
    )
}

pub fn build_segment_typed_witness_from_groups<F>(
    ring_d: usize,
    groups: &[SegmentTypedWitnessGroupParts<'_, F>],
    r: &RingVec<F>,
    lp: &LevelParams,
    num_commitment_groups: usize,
) -> Result<SegmentTypedWitness<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
{
    if ring_d == 0 || lp.ring_dimension != ring_d {
        return Err(AkitaError::InvalidInput(
            "segment-typed witness ring dimension mismatch".to_string(),
        ));
    }
    if !r.can_decode_vec(ring_d) {
        return Err(AkitaError::InvalidInput(
            "segment-typed r segment ring layout mismatch".to_string(),
        ));
    }
    let group_shapes = groups
        .iter()
        .map(|group| {
            (
                group.params,
                group.num_w_vectors,
                group.num_t_vectors,
                group.num_z_segments,
            )
        })
        .collect::<Vec<_>>();
    let field_bits = F::modulus_bits();
    let layout = tail_segment_layout_from_groups(
        lp,
        group_shapes,
        num_commitment_groups,
        field_bits,
        TerminalQuotientMode::Include,
    )?;
    let mut z_payloads = Vec::with_capacity(groups.len());
    let mut e_coeffs = Vec::new();
    let mut t_coeffs = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        if !group.e_folded.can_decode_vec(ring_d) {
            return Err(AkitaError::InvalidInput(
                "segment-typed e segment ring layout mismatch".to_string(),
            ));
        }
        if !group.z_folded_centered_flat.len().is_multiple_of(ring_d) {
            return Err(AkitaError::InvalidInput(
                "segment-typed z segment ring layout mismatch".to_string(),
            ));
        }
        let z_centered_i64: Vec<i64> = group
            .z_folded_centered_flat
            .iter()
            .map(|&coeff| i64::from(coeff))
            .collect();
        let cap =
            lp.fold_witness_linf_cap_for_params(group.params, group.num_t_vectors, field_bits)?;
        golomb_rice_flat_admit_terminal_wire(&z_centered_i64, cap)?;
        let depth_commit = group.params.num_digits_commit();
        let inner_width = group.params.num_positions_per_block() * depth_commit;
        let row_count = group.z_folded_centered_flat.len() / ring_d;
        if inner_width == 0 || !row_count.is_multiple_of(inner_width) {
            return Err(AkitaError::InvalidInput(
                "z_folded length does not match layout".to_string(),
            ));
        }
        let (rice_low_bits, zigzag_w_z) = tail_golomb_rice_z_params_from_cap(cap)?;
        let z_payload =
            encode_z_segment_from_centered_flat(&z_centered_i64, rice_low_bits, zigzag_w_z)?;
        let group_layout = layout
            .groups
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if z_payload.len() > group_layout.z_payload_bytes {
            return Err(AkitaError::InvalidInput(
                "segment-typed z segment length mismatch".to_string(),
            ));
        }
        z_payloads.push(z_payload);
        let e_fields = group.e_folded.clone().into_compact();
        if e_fields.coeff_len() != group_layout.e_field_elems {
            return Err(AkitaError::InvalidInput(
                "segment-typed e segment length mismatch".to_string(),
            ));
        }
        e_coeffs.extend_from_slice(e_fields.coeffs());
        let before_t = t_coeffs.len();
        for block in group.recomposed_inner_rows {
            if !block.can_decode_vec(ring_d) {
                return Err(AkitaError::InvalidInput(
                    "segment-typed t segment ring layout mismatch".to_string(),
                ));
            }
            t_coeffs.extend_from_slice(block.coeffs());
        }
        if t_coeffs.len() - before_t != group_layout.t_field_elems {
            return Err(AkitaError::InvalidInput(
                "segment-typed t segment length mismatch".to_string(),
            ));
        }
    }
    let e_fields = RingVec::from_coeffs(e_coeffs);
    let t_fields = RingVec::from_coeffs(t_coeffs);
    let r_fields = r.clone().into_compact();
    if r_fields.coeff_len() != layout.r_field_elems {
        return Err(AkitaError::InvalidInput(
            "segment-typed r segment length mismatch".to_string(),
        ));
    }
    let witness = SegmentTypedWitness {
        layout: layout.clone(),
        z_payloads,
        e_fields,
        t_fields,
        r_fields,
    };
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
        witness
            .z_payloads
            .first()
            .ok_or(AkitaError::InvalidProof)?,
        witness
            .layout
            .groups
            .first()
            .ok_or(AkitaError::InvalidProof)?
            .z_coords,
        lp,
        num_t_vectors,
        Some(budget_bytes),
    )
    .map(|_| ())
    .map_err(|err| match err {
        AkitaError::InvalidProof => AkitaError::InvalidInput(format!(
            "segment-typed z payload {} bytes inadmissible or exceeds schedule budget {budget_bytes}",
            witness.z_payloads.first().map_or(0, Vec::len)
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
    let (num_w_vectors, num_t_vectors, num_z_segments) =
        tail_segment_multiplicities_from_layout(lp, &witness.layout, 0)?;
    let expected_layout = tail_segment_layout_from_groups(
        lp,
        [(
            lp as &dyn LevelParamsLike,
            num_w_vectors,
            num_t_vectors,
            num_z_segments,
        )],
        num_commitment_groups,
        field_bits,
        TerminalQuotientMode::Include,
    )?;
    if !expected_layout.admits_realized(&witness.layout) {
        return Err(AkitaError::InvalidProof);
    }
    if num_commitment_groups != 1 || num_w_vectors != num_t_vectors {
        return Err(AkitaError::InvalidProof);
    }

    let log_basis = lp.log_basis;
    let depth_open = lp.num_digits_open;
    let depth_commit = lp.num_digits_commit;
    let num_digits_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    let levels = compute_num_digits_full_field(field_bits, log_basis);
    let group_layout = witness
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    let z_values = decode_terminal_z_golomb_payload_with_cap(
        witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?,
        group_layout.z_coords,
        lp.fold_witness_linf_cap_for_params(lp, num_t_vectors, field_bits)?,
        Some(group_layout.z_payload_bytes),
    )?;
    let inner_width = lp.num_positions_per_block * depth_commit;
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

    let w_block_count = num_w_vectors * lp.num_live_blocks;
    let e_planes = decompose_field_segment_to_planes::<F, D>(
        witness.e_fields.coeffs(),
        w_block_count,
        depth_open,
        log_basis,
    )?;
    let t_block_count = num_t_vectors * lp.num_live_blocks;
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

    let opening_batch = crate::OpeningClaimsLayout::new(0, num_w_vectors)?;
    let physical_layout =
        WitnessLayout::new(lp, &opening_batch, num_z_segments, r_rings.len(), levels)?;
    let physical_len = physical_layout
        .total_len()
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("terminal witness length overflow".into()))?;
    if physical_len != witness.layout.logical_num_elems {
        return Err(AkitaError::InvalidProof);
    }
    let mut out = vec![0i8; physical_len];
    let z_planes_per_unit = lp
        .num_positions_per_block
        .checked_mul(depth_commit)
        .and_then(|n| n.checked_mul(num_digits_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("terminal Z plane count overflow".into()))?;
    for (unit_index, unit) in physical_layout.units().iter().enumerate() {
        let start = unit_index
            .checked_mul(z_planes_per_unit)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal Z offset overflow".into()))?;
        let end = start
            .checked_add(z_planes_per_unit)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal Z end overflow".into()))?;
        emit_witness_z_planes::<D>(
            &mut out,
            unit,
            lp.num_positions_per_block,
            depth_commit,
            num_digits_fold,
            all_z_planes
                .get(start..end)
                .ok_or(AkitaError::InvalidProof)?,
        )?;
    }
    emit_witness_e_planes::<D>(
        &mut out,
        &physical_layout,
        0,
        num_w_vectors,
        depth_open,
        &e_planes,
        lp.num_live_blocks,
    )?;
    emit_witness_t_planes::<D>(
        &mut out,
        &physical_layout,
        0,
        num_t_vectors,
        lp.a_key.row_len(),
        depth_open,
        &t_planes,
        lp.num_live_blocks,
    )?;
    emit_witness_r_planes::<D>(&mut out, &physical_layout, levels, &r_planes_flat)?;
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

/// Emit one group's E planes at canonical witness addresses.
pub fn emit_witness_e_planes<const D: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    group_id: usize,
    num_claims: usize,
    depth_open: usize,
    flat: &[[i8; D]],
    source_num_live_blocks: usize,
) -> Result<(), AkitaError> {
    let expected = num_claims
        .checked_mul(source_num_live_blocks)
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness E source length overflow".into()))?;
    if flat.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: flat.len(),
        });
    }
    for unit in layout.units_for_group(group_id)? {
        for claim in 0..num_claims {
            for global_block in unit.global_block_range() {
                for digit in 0..depth_open {
                    let source =
                        (claim * source_num_live_blocks + global_block) * depth_open + digit;
                    write_witness_plane(
                        out,
                        unit.e_index(num_claims, depth_open, claim, global_block, digit)?,
                        &flat[source],
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Emit one group's T planes at canonical witness addresses.
#[allow(clippy::too_many_arguments)]
pub fn emit_witness_t_planes<const D: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    group_id: usize,
    num_claims: usize,
    n_a: usize,
    depth_open: usize,
    flat: &[[i8; D]],
    source_num_live_blocks: usize,
) -> Result<(), AkitaError> {
    let expected = num_claims
        .checked_mul(source_num_live_blocks)
        .and_then(|n| n.checked_mul(n_a))
        .and_then(|n| n.checked_mul(depth_open))
        .ok_or_else(|| AkitaError::InvalidSetup("witness T source length overflow".into()))?;
    if flat.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: flat.len(),
        });
    }
    let planes_per_block = n_a
        .checked_mul(depth_open)
        .ok_or_else(|| AkitaError::InvalidSetup("witness T source stride overflow".into()))?;
    for unit in layout.units_for_group(group_id)? {
        for claim in 0..num_claims {
            for global_block in unit.global_block_range() {
                for a_row in 0..n_a {
                    for digit in 0..depth_open {
                        let source = (claim * source_num_live_blocks + global_block)
                            * planes_per_block
                            + a_row * depth_open
                            + digit;
                        write_witness_plane(
                            out,
                            unit.t_index(
                                num_claims,
                                n_a,
                                depth_open,
                                claim,
                                global_block,
                                a_row,
                                digit,
                            )?,
                            &flat[source],
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Emit one ownership unit's replicated Z planes at canonical addresses.
pub fn emit_witness_z_planes<const D: usize>(
    out: &mut [i8],
    unit: &WitnessUnitLayout,
    num_positions_per_block: usize,
    depth_commit: usize,
    depth_fold: usize,
    all_planes: &[[i8; D]],
) -> Result<(), AkitaError> {
    let expected = num_positions_per_block
        .checked_mul(depth_commit)
        .and_then(|n| n.checked_mul(depth_fold))
        .ok_or_else(|| AkitaError::InvalidSetup("witness Z source length overflow".into()))?;
    if all_planes.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: all_planes.len(),
        });
    }
    for position in 0..num_positions_per_block {
        for commit_digit in 0..depth_commit {
            for fold_digit in 0..depth_fold {
                let source = (position * depth_commit + commit_digit) * depth_fold + fold_digit;
                write_witness_plane(
                    out,
                    unit.z_index(
                        num_positions_per_block,
                        depth_commit,
                        depth_fold,
                        position,
                        commit_digit,
                        fold_digit,
                    )?,
                    &all_planes[source],
                )?;
            }
        }
    }
    Ok(())
}

/// Emit the shared R planes at canonical witness addresses.
pub fn emit_witness_r_planes<const D: usize>(
    out: &mut [i8],
    layout: &WitnessLayout,
    quotient_depth: usize,
    planes: &[[i8; D]],
) -> Result<(), AkitaError> {
    let expected = layout.r_range().len();
    if planes.len() != expected {
        return Err(AkitaError::InvalidSize {
            expected,
            actual: planes.len(),
        });
    }
    if quotient_depth == 0 || !expected.is_multiple_of(quotient_depth) {
        return Err(AkitaError::InvalidSetup(
            "witness R source shape is malformed".into(),
        ));
    }
    for row in 0..expected / quotient_depth {
        for digit in 0..quotient_depth {
            write_witness_plane(
                out,
                layout.r_index(quotient_depth, row, digit)?,
                &planes[row * quotient_depth + digit],
            )?;
        }
    }
    Ok(())
}

fn write_witness_plane<const D: usize>(
    out: &mut [i8],
    ring_index: usize,
    plane: &[i8; D],
) -> Result<(), AkitaError> {
    let start = ring_index
        .checked_mul(D)
        .ok_or_else(|| AkitaError::InvalidSetup("witness plane offset overflow".into()))?;
    let end = start
        .checked_add(D)
        .ok_or_else(|| AkitaError::InvalidSetup("witness plane end overflow".into()))?;
    let dst = out.get_mut(start..end).ok_or(AkitaError::InvalidProof)?;
    dst.copy_from_slice(plane);
    Ok(())
}

#[cfg(test)]
mod tests;
