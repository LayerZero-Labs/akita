//! Segment-typed terminal witness layout, sizing, and construction.

use std::io::Write;

use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};

use super::{checked_shape_len, checked_shape_sequence_len, reserve_shape_len};
use crate::descriptor_bytes::{push_u32, push_usize};
use crate::golomb_rice::{
    analyze_z_fold_golomb_encoding, golomb_rice_decode_vec, golomb_rice_encode_vec,
    golomb_rice_flat_admit_terminal_wire, golomb_rice_max_quotient_for_cap,
    golomb_rice_total_wire_bits, golomb_rice_values_within_cap, golomb_rice_zigzag_width,
    tail_z_planner_bits_per_coord, ZFoldEncodingStats,
};
use crate::instance_descriptor::FoldLinfProtocolBinding;
use crate::layout::field_bytes;
use crate::proof::{RingVec, TerminalWitnessTranscriptParts};
use crate::tail_golomb_rice_low_bits::{cap_rice_low_bits, wire_rice_low_bits_from_rule};
use crate::{
    CommittedGroupParams, LevelParamsLike, TerminalCommittedGroupParams, WitnessLayout,
    WitnessUnitLayout,
};

/// Public segment geometry for a transparent terminal witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    /// Per-group terminal segments in witness order. Scalar/single-group tails
    /// are represented as exactly one group.
    pub groups: Vec<TailSegmentGroupLayout>,
    /// Logical digit-plane length used for schedule sizing.
    pub logical_num_elems: usize,
}

/// Per-group terminal segment geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailSegmentGroupLayout {
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    /// Golomb-Rice remainder width selected from the honest response cap.
    pub z_rice_low_bits: u32,
    /// Scheduled byte budget for this group's Golomb-coded z payload.
    pub z_payload_bytes: usize,
}

/// Shape of the clear terminal response payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalResponseShape {
    pub layout: TailSegmentLayout,
}

/// Clear terminal response carried on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalResponse<F: FieldCore> {
    pub layout: TailSegmentLayout,
    pub z_payloads: Vec<Vec<u8>>,
    pub e_fields: RingVec<F>,
    pub t_fields: RingVec<F>,
}

pub struct TerminalResponseGroupParts<'a, F: FieldCore> {
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
        push_usize(bytes, self.groups.len());
        for group in &self.groups {
            push_usize(bytes, group.z_coords);
            push_usize(bytes, group.e_field_elems);
            push_usize(bytes, group.t_field_elems);
            push_u32(bytes, group.z_rice_low_bits);
            push_usize(bytes, group.z_payload_bytes);
        }
        push_usize(bytes, self.logical_num_elems);
    }

    #[must_use]
    pub fn z_coords(&self) -> usize {
        self.groups
            .iter()
            .fold(0usize, |total, group| total.saturating_add(group.z_coords))
    }

    #[must_use]
    pub fn e_field_elems(&self) -> usize {
        self.groups.iter().fold(0usize, |total, group| {
            total.saturating_add(group.e_field_elems)
        })
    }

    #[must_use]
    pub fn t_field_elems(&self) -> usize {
        self.groups.iter().fold(0usize, |total, group| {
            total.saturating_add(group.t_field_elems)
        })
    }

    #[must_use]
    pub fn z_payload_bytes(&self) -> usize {
        self.groups.iter().fold(0usize, |total, group| {
            total.saturating_add(group.z_payload_bytes)
        })
    }

    #[must_use]
    pub fn admits_realized(&self, realized: &Self) -> bool {
        self.ring_dimension == realized.ring_dimension
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
                        && scheduled.z_rice_low_bits == realized.z_rice_low_bits
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
        checked_shape_sequence_len(self.groups.len())?;
        checked_shape_len(self.logical_num_elems)?;
        let mut z_coords = 0usize;
        let mut e_field_elems = 0usize;
        let mut t_field_elems = 0usize;
        let mut z_payload_bytes = 0usize;
        for group in &self.groups {
            if group.z_coords == 0 {
                return Err(SerializationError::InvalidData(
                    "tail segment group has zero z_coords".to_string(),
                ));
            }
            z_coords = z_coords.checked_add(group.z_coords).ok_or_else(|| {
                SerializationError::InvalidData("tail z coordinate count overflow".to_string())
            })?;
            e_field_elems = e_field_elems
                .checked_add(group.e_field_elems)
                .ok_or_else(|| {
                    SerializationError::InvalidData("tail e field count overflow".to_string())
                })?;
            t_field_elems = t_field_elems
                .checked_add(group.t_field_elems)
                .ok_or_else(|| {
                    SerializationError::InvalidData("tail t field count overflow".to_string())
                })?;
            z_payload_bytes = z_payload_bytes
                .checked_add(group.z_payload_bytes)
                .ok_or_else(|| {
                    SerializationError::InvalidData("tail z payload budget overflow".to_string())
                })?;
        }
        checked_shape_len(z_coords)?;
        checked_shape_len(e_field_elems)?;
        checked_shape_len(t_field_elems)?;
        checked_shape_len(z_payload_bytes)?;
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
        self.groups.serialize_with_mode(&mut writer, compress)?;
        self.logical_num_elems
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.ring_dimension.serialized_size(compress)
            + self.groups.serialized_size(compress)
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
        self.z_rice_low_bits
            .serialize_with_mode(&mut writer, compress)?;
        self.z_payload_bytes
            .serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.z_coords.serialized_size(compress)
            + self.e_field_elems.serialized_size(compress)
            + self.t_field_elems.serialized_size(compress)
            + self.z_rice_low_bits.serialized_size(compress)
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
            z_rice_low_bits: u32::deserialize_with_mode(&mut reader, compress, validate, &())?,
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
        let encoded_group_len = u64::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let group_len = usize::try_from(encoded_group_len).map_err(|_| {
            SerializationError::LengthLimitExceeded {
                len: encoded_group_len,
                max: super::MAX_PROOF_SHAPE_SEQUENCE_LEN,
            }
        })?;
        checked_shape_sequence_len(group_len)?;
        let mut groups = Vec::new();
        reserve_shape_len(&mut groups, group_len)?;
        for _ in 0..group_len {
            groups.push(TailSegmentGroupLayout::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &(),
            )?);
        }
        let logical_num_elems = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
        let out = Self {
            ring_dimension,
            groups,
            logical_num_elems,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl TerminalResponseShape {
    /// Derive the scalar terminal response directly from raw response
    /// coordinates. No `t`/`e` gadget-plane equivalent is introduced.
    pub fn derive(
        params: &TerminalCommittedGroupParams,
        admission_cap: u128,
    ) -> Result<Self, AkitaError> {
        if admission_cap == 0 {
            return Err(AkitaError::InvalidSetup(
                "terminal response admission cap must be nonzero".to_string(),
            ));
        }
        let d = params.d_a();
        let z_coords = params
            .inner_width()
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal z coordinates overflow".into()))?;
        let e_field_elems = params
            .num_live_blocks
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal e coordinates overflow".into()))?;
        let t_field_elems = params
            .num_live_blocks
            .checked_mul(params.inner_commit_matrix.output_rank())
            .and_then(|value| value.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("terminal t coordinates overflow".into()))?;
        let z_rice_low_bits = cap_rice_low_bits(admission_cap);
        let z_payload_bytes = z_payload_budget_from_cap(z_coords, admission_cap);
        let logical_num_elems = z_coords
            .checked_add(e_field_elems)
            .and_then(|value| value.checked_add(t_field_elems))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("terminal response coordinates overflow".into())
            })?;
        Ok(Self {
            layout: TailSegmentLayout {
                ring_dimension: d,
                groups: vec![TailSegmentGroupLayout {
                    z_coords,
                    e_field_elems,
                    t_field_elems,
                    z_rice_low_bits,
                    z_payload_bytes,
                }],
                logical_num_elems,
            },
        })
    }

    /// Append canonical Fiat-Shamir descriptor bytes (fixed little-endian).
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.layout.append_descriptor_bytes(bytes);
    }
}

impl Valid for TerminalResponseShape {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()?;
        Ok(())
    }
}

impl AkitaSerialize for TerminalResponseShape {
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

impl AkitaDeserialize for TerminalResponseShape {
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

impl<F: FieldCore + Valid> Valid for TerminalResponse<F> {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()?;
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
        Ok(())
    }
}

impl<F: FieldCore> TerminalResponse<F> {
    /// Shape descriptor for this terminal witness.
    pub fn shape(&self) -> TerminalResponseShape {
        TerminalResponseShape {
            layout: self.layout.clone(),
        }
    }

    /// Number of logical field elements carried by this witness.
    pub fn num_elems(&self) -> usize {
        self.layout.logical_num_elems
    }
}

impl TerminalResponseShape {
    /// Number of logical field elements represented by this shape.
    #[must_use]
    pub fn logical_num_elems(&self) -> usize {
        self.layout.logical_num_elems
    }

    /// Whether a realized terminal layout fits this scheduled upper bound.
    #[must_use]
    pub fn admits_realized(&self, realized: &Self) -> bool {
        self.layout.admits_realized(&realized.layout)
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize> AkitaSerialize for TerminalResponse<F> {
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
            self.layout
                .e_field_elems()
                .saturating_add(self.layout.t_field_elems())
                .saturating_mul(field_bytes(F::modulus_bits())),
        )
    }
}

impl<F: FieldCore + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize
    for TerminalResponse<F>
{
    type Context = TerminalResponseShape;

    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
        ctx: &TerminalResponseShape,
    ) -> Result<Self, SerializationError> {
        if matches!(validate, Validate::Yes) {
            ctx.check()?;
        }
        let mut z_payloads = Vec::with_capacity(ctx.layout.groups.len());
        for group in &ctx.layout.groups {
            let z_len = usize::deserialize_with_mode(&mut reader, compress, validate, &())?;
            if z_len > group.z_payload_bytes {
                return Err(SerializationError::InvalidData(format!(
                    "terminal z payload length {z_len} exceeds scheduled budget {}",
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
        let out = Self {
            layout: ctx.layout.clone(),
            z_payloads,
            e_fields,
            t_fields,
        };
        if matches!(validate, Validate::Yes) {
            out.check()?;
        }
        Ok(out)
    }
}

impl<F: FieldCore + CanonicalField + AkitaSerialize> TerminalResponse<F> {
    /// Canonical segment bytes in wire order (`z ‖ e ‖ t`).
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
        Ok(())
    }

    /// Materialize pre-challenge `e` bytes and the post-challenge `z` response.
    /// This helper omits `t`: the predecessor binds it as outgoing state and
    /// terminal current-state replay owns its second transcript binding.
    pub fn terminal_transcript_parts(&self) -> Result<TerminalWitnessTranscriptParts, AkitaError> {
        let e_folded = raw_field_segment_bytes(&self.e_fields)?;
        if e_folded.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        if self.t_fields.coeffs().is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        let mut response = Vec::new();
        for payload in &self.z_payloads {
            response.extend_from_slice(payload);
        }
        if response.is_empty() {
            return Err(AkitaError::InvalidProof);
        }
        Ok(TerminalWitnessTranscriptParts { e_folded, response })
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

/// Canonical transcript bytes for a raw-field terminal segment.
///
/// Both the prover terminal absorb and the verifier's decoded-witness replay
/// route through this single routine, so the bound `e_hat` bytes are identical
/// by construction (it mirrors the `e_fields` the segment witness carries).
///
/// # Errors
///
/// Propagates field serialization failures as [`AkitaError::InvalidProof`].
pub fn raw_field_segment_bytes<F>(fields: &RingVec<F>) -> Result<Vec<u8>, AkitaError>
where
    F: FieldCore + CanonicalField + AkitaSerialize,
{
    let mut out = Vec::new();
    append_field_coeffs_vec(&mut out, fields.coeffs())?;
    Ok(out)
}

/// Runtime Golomb-Rice **wire** parameters for terminal `z` encode/decode.
///
/// Uses wire low bits ([`crate::wire_rice_low_bits`]); planner byte budgets use
/// [`crate::cap_rice_low_bits`] via [`terminal_response_z_payload_bytes`].
/// Rice `k` and zigzag width `W` are derived from the per-coefficient fold-response
/// cap [`crate::CommittedGroupParams::fold_witness_linf_cap_for_claims`] (`min(β_inf, t*)` or `β_inf`
/// alone), matching [`crate::sis::fold_witness_digit_plan`] and grind acceptance.
///
/// # Errors
///
/// Propagates fold cap setup errors.
pub fn tail_golomb_rice_z_params(
    lp: &CommittedGroupParams,
    num_t_vectors: usize,
) -> Result<(u32, u32), AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    tail_golomb_rice_z_params_from_cap(cap)
}

fn tail_golomb_rice_z_params_from_cap(cap: u128) -> Result<(u32, u32), AkitaError> {
    tail_golomb_rice_z_params_from_caps(cap, cap)
}

fn tail_golomb_rice_z_params_from_caps(
    coding_scale: u128,
    admissible_cap: u128,
) -> Result<(u32, u32), AkitaError> {
    let binding = FoldLinfProtocolBinding::CURRENT;
    let rice_low_bits = wire_rice_low_bits_from_rule(
        coding_scale,
        binding.wire_rice_low_bits_rule_id,
        binding.wire_rice_low_bits_delta,
    )?;
    let w = golomb_rice_zigzag_width(admissible_cap);
    Ok((rice_low_bits, w))
}

/// Decode terminal `z` using its single capacity-based admission and coding cap.
pub fn decode_terminal_z_golomb_payload(
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

/// Decode centered fold-response `z` coefficients from a terminal response.
///
/// # Errors
///
/// Propagates decode and public-parameter setup errors.
pub fn z_fold_decoded_from_terminal_response<F: FieldCore>(
    witness: &TerminalResponse<F>,
    lp: &CommittedGroupParams,
    num_t_vectors: usize,
) -> Result<Vec<i64>, AkitaError> {
    let payload = witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?;
    let group = witness
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    decode_terminal_z_golomb_payload(payload, group.z_coords, cap, None)
}

fn z_payload_budget_from_cap(z_coords: usize, cap: u128) -> usize {
    let low_bits_cap = cap_rice_low_bits(cap);
    let bits_per_coord = tail_z_planner_bits_per_coord(low_bits_cap);
    z_coords.saturating_mul(bits_per_coord).div_ceil(8)
}

/// Distribution / Golomb model audit for a realized terminal `z` payload.
///
/// # Errors
///
/// Propagates decode and public-parameter setup errors.
pub fn z_fold_encoding_stats_from_terminal_response<F: FieldCore>(
    witness: &TerminalResponse<F>,
    lp: &CommittedGroupParams,
    num_t_vectors: usize,
    field_bits: u32,
) -> Result<ZFoldEncodingStats, AkitaError> {
    let z_values = z_fold_decoded_from_terminal_response(witness, lp, num_t_vectors)?;
    let (_, zigzag_w) = tail_golomb_rice_z_params(lp, num_t_vectors)?;
    let witness_linf_cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    let depth_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    analyze_z_fold_golomb_encoding(
        &z_values,
        witness_linf_cap,
        zigzag_w,
        depth_fold,
        lp.log_basis_open,
        witness.z_payloads.first().map_or(0, Vec::len),
    )
}

fn tail_segment_layout_from_groups<'a>(
    lp: &CommittedGroupParams,
    groups: impl IntoIterator<Item = (&'a dyn LevelParamsLike, usize, usize, usize)>,
    _num_commitment_groups: usize,
    field_bits: u32,
) -> Result<TailSegmentLayout, AkitaError> {
    let d = lp.d_a();
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
        let depth_witness = params.num_digits_inner();
        let depth_commit = params.num_digits_outer();
        let depth_open = params.num_digits_open();
        let depth_fold = lp.num_digits_fold_for_params(params, num_t_vectors, field_bits)?;
        if depth_witness == 0 || depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
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
            .and_then(|n| n.checked_mul(depth_witness))
            .and_then(|n| n.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("tail z coord count overflow".to_string()))?;
        let z_plane_rings = num_z_segments
            .checked_mul(params.num_positions_per_block())
            .and_then(|n| n.checked_mul(depth_witness))
            .and_then(|n| n.checked_mul(depth_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("tail z plane count overflow".to_string()))?;
        let e_plane_rings = total_w_blocks
            .checked_mul(depth_open)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e plane count overflow".to_string()))?;
        let t_plane_rings = total_t_blocks
            .checked_mul(params.a_rows_len())
            .and_then(|n| n.checked_mul(depth_commit))
            .ok_or_else(|| AkitaError::InvalidSetup("tail t plane count overflow".to_string()))?;
        let z_cap = lp.fold_witness_linf_cap_for_params(params, num_t_vectors, field_bits)?;
        let security_cap = lp.terminal_response_linf_limit_for_params(params)?;
        if z_cap > security_cap {
            return Err(AkitaError::InvalidSetup(format!(
                "terminal honest response cap {z_cap} exceeds inner-matrix SIS capacity {security_cap}"
            )));
        }
        let z_payload_bytes = z_payload_budget_from_cap(z_coords, z_cap);
        group_layouts.push(TailSegmentGroupLayout {
            z_coords,
            e_field_elems,
            t_field_elems,
            z_rice_low_bits: cap_rice_low_bits(z_cap),
            z_payload_bytes,
        });
        total_plane_rings = total_plane_rings
            .checked_add(z_plane_rings)
            .and_then(|n| n.checked_add(e_plane_rings))
            .and_then(|n| n.checked_add(t_plane_rings))
            .ok_or_else(|| AkitaError::InvalidSetup("tail logical plane overflow".to_string()))?;
    }
    let logical_num_elems = total_plane_rings
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail logical elem overflow".to_string()))?;
    Ok(TailSegmentLayout {
        ring_dimension: d,
        groups: group_layouts,
        logical_num_elems,
    })
}

impl TerminalResponseShape {
    /// Derive the checked terminal witness shape for the scheduled groups.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when dimensions are empty or any
    /// derived segment size overflows.
    pub fn from_groups<'a>(
        lp: &CommittedGroupParams,
        field_bits: u32,
        groups: impl IntoIterator<Item = (&'a dyn LevelParamsLike, usize, usize, usize)>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            layout: tail_segment_layout_from_groups(lp, groups, 0, field_bits)?,
        })
    }
}

/// Recover tail multiplicities from a committed [`TailSegmentLayout`].
///
/// # Errors
///
/// Returns an error when the layout is inconsistent with `lp`.
pub fn tail_segment_multiplicities_from_layout(
    lp: &CommittedGroupParams,
    layout: &TailSegmentLayout,
    group_index: usize,
) -> Result<(usize, usize, usize), AkitaError> {
    tail_segment_multiplicities_from_layout_for_params(lp, lp.d_a(), layout, group_index)
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
        .checked_mul(params.num_digits_inner())
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
pub fn terminal_response_z_payload_bytes(
    lp: &CommittedGroupParams,
    layout: &TailSegmentLayout,
    num_t_vectors: usize,
) -> Result<usize, AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    Ok(z_payload_budget_from_cap(layout.z_coords(), cap))
}

/// Serialized byte size for a terminal response at a fixed `z` budget.
#[must_use]
pub fn terminal_response_upper_bound_bytes(
    field_bits: u32,
    layout: &TailSegmentLayout,
    z_payload_bytes: usize,
) -> usize {
    let raw_elems = layout
        .e_field_elems()
        .saturating_add(layout.t_field_elems());
    raw_elems
        .saturating_mul(field_bytes(field_bits))
        .saturating_add(z_payload_bytes)
        .saturating_add(8usize.saturating_mul(layout.groups.len()))
}

pub fn build_terminal_response_from_groups<F>(
    ring_d: usize,
    groups: &[TerminalResponseGroupParts<'_, F>],
    lp: &CommittedGroupParams,
) -> Result<TerminalResponse<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
{
    if ring_d == 0 || lp.d_a() != ring_d {
        return Err(AkitaError::InvalidInput(
            "terminal response ring dimension mismatch".to_string(),
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
    let layout = TerminalResponseShape::from_groups(lp, field_bits, group_shapes)?.layout;
    let mut z_payloads = Vec::with_capacity(groups.len());
    let mut e_coeffs = Vec::new();
    let mut t_coeffs = Vec::new();
    for (group_index, group) in groups.iter().enumerate() {
        if !group.e_folded.can_decode_vec(ring_d) {
            return Err(AkitaError::InvalidInput(
                "terminal e segment ring layout mismatch".to_string(),
            ));
        }
        if !group.z_folded_centered_flat.len().is_multiple_of(ring_d) {
            return Err(AkitaError::InvalidInput(
                "terminal z segment ring layout mismatch".to_string(),
            ));
        }
        let z_centered_i64: Vec<i64> = group
            .z_folded_centered_flat
            .iter()
            .map(|&coeff| i64::from(coeff))
            .collect();
        let honest_cap =
            lp.fold_witness_linf_cap_for_params(group.params, group.num_t_vectors, field_bits)?;
        let security_cap = lp.terminal_response_linf_limit_for_params(group.params)?;
        golomb_rice_flat_admit_terminal_wire(&z_centered_i64, honest_cap)?;
        let depth_witness = group.params.num_digits_inner();
        let inner_width = group.params.num_positions_per_block() * depth_witness;
        let row_count = group.z_folded_centered_flat.len() / ring_d;
        if inner_width == 0 || !row_count.is_multiple_of(inner_width) {
            return Err(AkitaError::InvalidInput(
                "z_folded length does not match layout".to_string(),
            ));
        }
        let (rice_low_bits, zigzag_w_z) =
            tail_golomb_rice_z_params_from_caps(honest_cap, security_cap)?;
        let z_payload = golomb_rice_encode_vec(&z_centered_i64, rice_low_bits, zigzag_w_z)?;
        let group_layout = layout
            .groups
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        if z_payload.len() > group_layout.z_payload_bytes {
            return Err(AkitaError::InvalidInput(
                "terminal z segment length mismatch".to_string(),
            ));
        }
        z_payloads.push(z_payload);
        let e_fields = group.e_folded.clone().into_compact();
        if e_fields.coeff_len() != group_layout.e_field_elems {
            return Err(AkitaError::InvalidInput(
                "terminal e segment length mismatch".to_string(),
            ));
        }
        e_coeffs.extend_from_slice(e_fields.coeffs());
        let before_t = t_coeffs.len();
        for block in group.recomposed_inner_rows {
            if !block.can_decode_vec(ring_d) {
                return Err(AkitaError::InvalidInput(
                    "terminal t segment ring layout mismatch".to_string(),
                ));
            }
            t_coeffs.extend_from_slice(block.coeffs());
        }
        if t_coeffs.len() - before_t != group_layout.t_field_elems {
            return Err(AkitaError::InvalidInput(
                "terminal t segment length mismatch".to_string(),
            ));
        }
    }
    let e_fields = RingVec::from_coeffs(e_coeffs);
    let t_fields = RingVec::from_coeffs(t_coeffs);
    let witness = TerminalResponse {
        layout: layout.clone(),
        z_payloads,
        e_fields,
        t_fields,
    };
    Ok(witness)
}

/// Build the scalar raw terminal response selected by the typed terminal
/// schedule. Neither `e` nor `t` is gadget decomposed.
pub fn build_terminal_response<F>(
    params: &TerminalCommittedGroupParams,
    sparse: &akita_challenges::SparseChallengeConfig,
    scheduled_shape: &TerminalResponseShape,
    e_folded: &RingVec<F>,
    t_fields: RingVec<F>,
    z_folded_centered_flat: &[i32],
) -> Result<TerminalResponse<F>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField + AkitaSerialize,
{
    let admission_cap = params.response_linf_policy(sparse)?.admission_cap;
    let expected = TerminalResponseShape::derive(params, admission_cap)?;
    if !scheduled_shape.admits_realized(&expected) || !expected.admits_realized(scheduled_shape) {
        return Err(AkitaError::InvalidSetup(
            "scheduled terminal response shape does not match terminal parameters".into(),
        ));
    }
    let group = scheduled_shape
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    if scheduled_shape.layout.groups.len() != 1
        || e_folded.coeff_len() != group.e_field_elems
        || z_folded_centered_flat.len() != group.z_coords
    {
        return Err(AkitaError::InvalidInput(
            "terminal response segment length mismatch".into(),
        ));
    }
    let z_values = z_folded_centered_flat
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    golomb_rice_flat_admit_terminal_wire(&z_values, admission_cap)?;
    let (rice_low_bits, zigzag_width) = tail_golomb_rice_z_params_from_cap(admission_cap)?;
    let z_payload = golomb_rice_encode_vec(&z_values, rice_low_bits, zigzag_width)?;
    if z_payload.len() > group.z_payload_bytes {
        return Err(AkitaError::InvalidInput(
            "terminal response exceeds its scheduled payload budget".into(),
        ));
    }
    if !t_fields.can_decode_vec(params.d_a()) {
        return Err(AkitaError::InvalidInput(
            "terminal t state is not inner-ring aligned".into(),
        ));
    }
    if t_fields.coeff_len() != group.t_field_elems {
        return Err(AkitaError::InvalidInput(
            "terminal t segment length mismatch".into(),
        ));
    }
    Ok(TerminalResponse {
        layout: scheduled_shape.layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: e_folded.clone().into_compact(),
        t_fields: t_fields.into_compact(),
    })
}

/// Check a segment witness `z` payload against the schedule-bound byte budget and public
/// Golomb admissibility.
///
/// # Errors
///
/// Returns an error when the encoded `z` payload is inadmissible or exceeds the budget.
pub fn validate_terminal_response_z_payload<F: FieldCore>(
    witness: &TerminalResponse<F>,
    lp: &CommittedGroupParams,
    num_t_vectors: usize,
    budget_bytes: usize,
) -> Result<(), AkitaError> {
    let cap = lp.fold_witness_linf_cap_for_claims(num_t_vectors)?;
    decode_terminal_z_golomb_payload(
        witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?,
        witness
            .layout
            .groups
            .first()
            .ok_or(AkitaError::InvalidProof)?
            .z_coords,
        cap,
        Some(budget_bytes),
    )
    .map(|_| ())
    .map_err(|err| match err {
        AkitaError::InvalidProof => AkitaError::InvalidInput(format!(
            "terminal z payload {} bytes inadmissible or exceeds schedule budget {budget_bytes}",
            witness.z_payloads.first().map_or(0, Vec::len)
        )),
        other => other,
    })
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
