//! Segment-typed terminal witness layout, sizing, and construction.

use std::io::Write;

use akita_error::AkitaError;

use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{CanonicalEncoding, Field};

use super::{checked_shape_len, checked_shape_sequence_len, reserve_shape_len};
use crate::descriptor_bytes::{push_u128, push_u32, push_usize};
use crate::golomb_rice::{
    golomb_rice_decode_vec, golomb_rice_encode_vec, golomb_rice_max_quotient_for_cap,
    golomb_rice_values_within_cap, golomb_rice_zigzag_width, tail_z_planner_bits_per_coord,
};
use crate::layout::field_bytes;
use crate::proof::{RingVec, TerminalWitnessTranscriptParts};
use crate::tail_golomb_rice_low_bits::{cap_rice_low_bits, wire_rice_low_bits};
use crate::{CommittedGroupParams, TerminalFoldParams};

/// Public segment geometry for a transparent terminal witness.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    /// Per-group terminal segments in witness order. Scalar/single-group tails
    /// are represented as exactly one group.
    pub groups: Vec<TailSegmentGroupLayout>,
    /// Logical digit-plane length used for schedule sizing.
    pub logical_num_elems: usize,
}

/// Per-group terminal segment geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailSegmentGroupLayout {
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    /// Verifier-enforced coefficient cap for a terminal Linf route.
    ///
    /// `None` means the terminal uses the complete L2 check instead. The wire
    /// still requires canonical Golomb-Rice encoding within `z_payload_bytes`
    /// and the signed-i16 coefficient representation.
    pub z_linf_cap: Option<u128>,
    /// Exact Golomb-Rice remainder width used on the wire.
    pub z_rice_low_bits: u32,
    /// Scheduled byte budget for this group's Golomb-coded z payload.
    pub z_payload_bytes: usize,
}

/// Shape of the clear terminal response payload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TerminalResponseShape {
    pub layout: TailSegmentLayout,
}

/// Clear terminal response carried on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalResponse<F: Field> {
    pub layout: TailSegmentLayout,
    pub z_payloads: Vec<Vec<u8>>,
    pub e_fields: RingVec<F>,
    pub t_fields: RingVec<F>,
}

pub struct TerminalResponseGroupParts<'a, F: Field> {
    pub params: crate::GroupOpenPhaseParams,
    pub num_w_vectors: usize,
    pub num_t_vectors: usize,
    pub num_z_segments: usize,
    pub e_folded: &'a RingVec<F>,
    /// Block-major native-A rows: `[block][A row][coefficient]`.
    pub recomposed_inner_rows: &'a RingVec<F>,
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
            push_u128(bytes, group.z_linf_cap.unwrap_or(0));
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
                        && scheduled.z_linf_cap == realized.z_linf_cap
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
            if group.z_linf_cap == Some(0) || group.z_rice_low_bits >= 64 {
                return Err(SerializationError::InvalidData(
                    "tail segment group has invalid z wire parameters".to_string(),
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
        self.z_linf_cap
            .unwrap_or(0)
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
            + 0u128.serialized_size(compress)
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
            z_linf_cap: match u128::deserialize_with_mode(&mut reader, compress, validate, &())? {
                0 => None,
                cap => Some(cap),
            },
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
    ///
    /// `encoding_scale` selects the frozen Golomb parameters and payload byte
    /// budget. It is also the verifier cap for a Linf route. An L2 route emits
    /// no Linf cap and enforces only its complete response energy.
    pub fn derive(params: &TerminalFoldParams, encoding_scale: u128) -> Result<Self, AkitaError> {
        if encoding_scale == 0 {
            return Err(AkitaError::InvalidSetup(
                "terminal response encoding scale must be nonzero".to_string(),
            ));
        }
        let d = params.d_a();
        let z_coords = params
            .inner_width()
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal z coordinates overflow".into()))?;
        let e_field_elems =
            params.blocks.live_blocks.checked_mul(d).ok_or_else(|| {
                AkitaError::InvalidSetup("terminal e coordinates overflow".into())
            })?;
        let t_field_elems = params
            .blocks
            .live_blocks
            .checked_mul(params.inner.matrix.output_rank())
            .and_then(|value| value.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("terminal t coordinates overflow".into()))?;
        let z_rice_low_bits = wire_rice_low_bits(encoding_scale);
        let z_payload_bytes = z_payload_budget_from_cap(z_coords, encoding_scale);
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
                    z_linf_cap: match params.inner.matrix.security_route() {
                        crate::sis::InnerCommitSecurityRoute::Linf(_) => Some(encoding_scale),
                        crate::sis::InnerCommitSecurityRoute::L2 { .. } => None,
                    },
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

impl<F: Field + Valid> Valid for TerminalResponse<F> {
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

impl<F: Field> TerminalResponse<F> {
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

impl<F: Field + CanonicalEncoding + AkitaSerialize> AkitaSerialize for TerminalResponse<F> {
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
                .saturating_mul(field_bytes(F::MODULUS_BITS)),
        )
    }
}

impl<F: Field + Valid + AkitaDeserialize<Context = ()>> AkitaDeserialize for TerminalResponse<F> {
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

impl<F: Field + CanonicalEncoding + AkitaSerialize> TerminalResponse<F> {
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

fn append_field_coeffs<F: Field + AkitaSerialize, W: Write>(
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

fn append_field_coeffs_vec<F: Field + AkitaSerialize>(
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
    F: Field + CanonicalEncoding + AkitaSerialize,
{
    let mut out = Vec::new();
    append_field_coeffs_vec(&mut out, fields.coeffs())?;
    Ok(out)
}

/// Decode terminal `z` with the exact schedule-owned norm route and wire shape.
pub fn decode_terminal_z_golomb_payload(
    payload: &[u8],
    group: &TailSegmentGroupLayout,
) -> Result<Vec<i16>, AkitaError> {
    if payload.len() > group.z_payload_bytes {
        return Err(AkitaError::InvalidProof);
    }
    let wire_abs_bound = group.z_linf_cap.unwrap_or(i16::MAX as u128);
    let rice_low_bits = group.z_rice_low_bits;
    let zigzag_w = golomb_rice_zigzag_width(wire_abs_bound);
    let max_quotient = golomb_rice_max_quotient_for_cap(wire_abs_bound, rice_low_bits, zigzag_w)?;
    let values = golomb_rice_decode_vec(
        payload,
        group.z_coords,
        rice_low_bits,
        zigzag_w,
        max_quotient,
        |value| {
            if group
                .z_linf_cap
                .is_some_and(|cap| i128::from(value).unsigned_abs() > cap)
            {
                return Err(AkitaError::InvalidProof);
            }
            i16::try_from(value).map_err(|_| AkitaError::InvalidProof)
        },
    )?;
    // Canonical decoding rejects nonzero padding and trailing bytes. Therefore
    // the exact wire bit length is at most `payload.len() * 8`, and the byte
    // bound above enforces the same rounded schedule budget without another
    // pass that re-encodes every decoded value.
    Ok(values)
}

fn z_payload_budget_from_cap(z_coords: usize, cap: u128) -> usize {
    let low_bits_cap = cap_rice_low_bits(cap);
    let bits_per_coord = tail_z_planner_bits_per_coord(low_bits_cap);
    z_coords.saturating_mul(bits_per_coord).div_ceil(8)
}

fn tail_segment_layout_from_groups(
    lp: &CommittedGroupParams,
    groups: impl IntoIterator<Item = (crate::GroupOpenPhaseParams, usize, usize, usize, u128)>,
    _num_commitment_groups: usize,
    _field_bits: u32,
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
    for (params, num_w_vectors, num_t_vectors, num_z_segments, z_cap) in groups {
        let depth_witness = params.num_digits_inner();
        let depth_commit = params.num_digits_outer();
        let depth_open = params.num_digits_open();
        let depth_fold = params.num_digits_fold();
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
        // Price this group's response against *this group's* challenge family.
        let security_cap = crate::sis::certified_terminal_response_linf_cap(
            params.inner_commit_matrix_params(),
            &params.fold_challenge_config(),
        )?;
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
            z_linf_cap: Some(z_cap),
            z_rice_low_bits: wire_rice_low_bits(z_cap),
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
    pub fn from_groups(
        lp: &CommittedGroupParams,
        field_bits: u32,
        groups: impl IntoIterator<Item = (crate::GroupOpenPhaseParams, usize, usize, usize, u128)>,
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
    tail_segment_multiplicities_from_layout_for_params(
        &lp.final_group_scalar()?,
        lp.d_a(),
        layout,
        group_index,
    )
}

pub fn tail_segment_multiplicities_from_layout_for_params(
    params: &crate::GroupOpenPhaseParams,
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
pub fn terminal_response_z_payload_bytes(layout: &TailSegmentLayout) -> usize {
    layout.z_payload_bytes()
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
    scheduled_shape: &TerminalResponseShape,
) -> Result<TerminalResponse<F>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
{
    if ring_d == 0 || lp.d_a() != ring_d {
        return Err(AkitaError::InvalidInput(
            "terminal response ring dimension mismatch".to_string(),
        ));
    }
    let layout = scheduled_shape.layout.clone();
    if layout.groups.len() != groups.len() {
        return Err(AkitaError::InvalidSetup(
            "terminal response group count does not match its schedule".into(),
        ));
    }
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
        // Price this group's response against *this group's* challenge family.
        let security_cap = crate::sis::certified_terminal_response_linf_cap(
            group.params.inner_commit_matrix_params(),
            &group.params.fold_challenge_config(),
        )?;
        let depth_witness = group.params.num_digits_inner();
        let inner_width = group.params.num_positions_per_block() * depth_witness;
        let row_count = group.z_folded_centered_flat.len() / ring_d;
        if inner_width == 0 || !row_count.is_multiple_of(inner_width) {
            return Err(AkitaError::InvalidInput(
                "z_folded length does not match layout".to_string(),
            ));
        }
        let group_layout = layout
            .groups
            .get(group_index)
            .ok_or(AkitaError::InvalidProof)?;
        let z_cap = group_layout.z_linf_cap.ok_or_else(|| {
            AkitaError::InvalidSetup("legacy terminal group requires a Linf cap".into())
        })?;
        if z_cap > security_cap {
            return Err(AkitaError::InvalidSetup(
                "terminal response cap exceeds its matrix-certified capacity".into(),
            ));
        }
        golomb_rice_values_within_cap(&z_centered_i64, z_cap)
            .map_err(|_| AkitaError::InvalidInput("terminal response exceeds its cap".into()))?;
        let zigzag_w_z = golomb_rice_zigzag_width(z_cap);
        let z_payload =
            golomb_rice_encode_vec(&z_centered_i64, group_layout.z_rice_low_bits, zigzag_w_z)?;
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
        if !group.recomposed_inner_rows.can_decode_vec(ring_d) {
            return Err(AkitaError::InvalidInput(
                "terminal t segment ring layout mismatch".to_string(),
            ));
        }
        if group.recomposed_inner_rows.coeff_len() != group_layout.t_field_elems {
            return Err(AkitaError::InvalidInput(
                "terminal t segment length mismatch".to_string(),
            ));
        }
        t_coeffs.extend_from_slice(group.recomposed_inner_rows.coeffs());
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
    params: &TerminalFoldParams,
    scheduled_shape: &TerminalResponseShape,
    e_folded: &RingVec<F>,
    t_fields: RingVec<F>,
    z_folded_centered_flat: &[i32],
) -> Result<TerminalResponse<F>, AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize,
{
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
    params.validate_terminal_linf_cap(group.z_linf_cap)?;
    let z_values = z_folded_centered_flat
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    if let Some(cap) = group.z_linf_cap {
        golomb_rice_values_within_cap(&z_values, cap).map_err(|_| {
            AkitaError::InvalidInput("terminal response exceeds its scheduled Linf cap".into())
        })?;
    }
    let zigzag_width = golomb_rice_zigzag_width(group.z_linf_cap.unwrap_or(i16::MAX as u128));
    let z_payload = golomb_rice_encode_vec(&z_values, group.z_rice_low_bits, zigzag_width)?;
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
pub fn validate_terminal_response_z_payload<F: Field>(
    witness: &TerminalResponse<F>,
) -> Result<(), AkitaError> {
    let group = witness
        .layout
        .groups
        .first()
        .ok_or(AkitaError::InvalidProof)?;
    decode_terminal_z_golomb_payload(
        witness.z_payloads.first().ok_or(AkitaError::InvalidProof)?,
        group,
    )
    .map(|_| ())
    .map_err(|err| match err {
        AkitaError::InvalidProof => AkitaError::InvalidInput(format!(
            "terminal z payload {} bytes is inadmissible or exceeds its schedule budget",
            witness.z_payloads.first().map_or(0, Vec::len)
        )),
        other => other,
    })
}

#[cfg(test)]
mod tests;
