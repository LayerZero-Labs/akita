//! Segment-typed terminal witness layout, sizing, and construction.

use std::io::Write;

use akita_error::AkitaError;

use akita_serialization::{
    AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid, Validate,
};
use jolt_field::{CanonicalEncoding, Field};

use crate::golomb_rice::{
    golomb_rice_decode_vec, golomb_rice_max_quotient_for_cap, golomb_rice_zigzag_width,
};
use crate::layout::field_bytes;
use crate::layout::tail_segments::{
    TailSegmentGroupLayout, TailSegmentLayout, TerminalResponseShape,
};
use crate::proof::RingVec;
use crate::TerminalFoldParams;

/// Clear terminal response carried on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalResponse<F: Field> {
    pub layout: TailSegmentLayout,
    pub z_payloads: Vec<Vec<u8>>,
    pub e_fields: RingVec<F>,
    pub t_fields: RingVec<F>,
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

/// Build a terminal response from an opaque backend-produced canonical Z payload.
pub fn build_terminal_response_from_payload<F>(
    params: &TerminalFoldParams,
    scheduled_shape: &TerminalResponseShape,
    e_folded: &RingVec<F>,
    t_fields: RingVec<F>,
    z_payload: Vec<u8>,
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
        || t_fields.coeff_len() != group.t_field_elems
        || !t_fields.can_decode_vec(params.d_a())
    {
        return Err(AkitaError::InvalidInput(
            "terminal response segment length mismatch".into(),
        ));
    }
    params.validate_terminal_linf_cap(group.z_linf_cap)?;
    decode_terminal_z_golomb_payload(&z_payload, group)?;
    Ok(TerminalResponse {
        layout: scheduled_shape.layout.clone(),
        z_payloads: vec![z_payload],
        e_fields: e_folded.clone().into_compact(),
        t_fields: t_fields.into_compact(),
    })
}

#[cfg(test)]
mod tests;
