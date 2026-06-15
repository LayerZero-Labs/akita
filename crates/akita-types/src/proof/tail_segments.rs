//! Segment-typed terminal witness layout, sizing, expansion, and construction.

use std::io::Write;

use akita_algebra::ring::cyclotomic::BalancedDecomposePow2I8Params;
use akita_algebra::CyclotomicRing;
use akita_field::{AkitaError, CanonicalField, FieldCore, HalvingField};
use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, SerializationError, Valid};

use crate::golomb_rice::{
    golomb_rice_decode_vec, golomb_rice_encode_vec, golomb_rice_max_bits_per_coord,
    golomb_rice_zigzag_width_z, optimal_rice_k,
};
use crate::layout::field_bytes;
use crate::proof::{ring_column_z_first, FlatRingVec, TerminalWitnessTranscriptParts};
use crate::sis::compute_num_digits_full_field;
use crate::{LevelParams, MRowLayout};

/// Public segment geometry for a transparent terminal witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    pub log_basis: u32,
    pub z_first: bool,
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    pub r_field_elems: usize,
    /// Hypercube length after expansion to digit planes (matches legacy `PackedDigits::num_elems`).
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

impl Valid for SegmentTypedWitnessShape {
    fn check(&self) -> Result<(), SerializationError> {
        if self.layout.ring_dimension == 0 {
            return Err(SerializationError::InvalidData(
                "tail segment layout has zero ring dimension".to_string(),
            ));
        }
        if self.layout.z_coords == 0 {
            return Err(SerializationError::InvalidData(
                "tail segment z_coords is zero".to_string(),
            ));
        }
        Ok(())
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
        self.z_payload.len().saturating_add(
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
        let mut z_payload = vec![0u8; ctx.z_payload_bytes];
        let e_fields;
        let t_fields;
        let r_fields;
        if ctx.layout.z_first {
            reader.read_exact(&mut z_payload)?;
            e_fields = FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.layout.e_field_elems,
            )?;
            t_fields = FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.layout.t_field_elems,
            )?;
            r_fields = FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.layout.r_field_elems,
            )?;
        } else {
            e_fields = FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.layout.e_field_elems,
            )?;
            reader.read_exact(&mut z_payload)?;
            t_fields = FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.layout.t_field_elems,
            )?;
            r_fields = FlatRingVec::deserialize_with_mode(
                &mut reader,
                compress,
                validate,
                &ctx.layout.r_field_elems,
            )?;
        }
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
    /// Canonical segment bytes in wire order (`z`/`e` permuted by `z_first`).
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
        if self.layout.z_first {
            writer.write_all(&self.z_payload)?;
            append_field_coeffs(writer, self.e_fields.coeffs(), compress)?;
            append_field_coeffs(writer, self.t_fields.coeffs(), compress)?;
            append_field_coeffs(writer, self.r_fields.coeffs(), compress)?;
        } else {
            append_field_coeffs(writer, self.e_fields.coeffs(), compress)?;
            writer.write_all(&self.z_payload)?;
            append_field_coeffs(writer, self.t_fields.coeffs(), compress)?;
            append_field_coeffs(writer, self.r_fields.coeffs(), compress)?;
        }
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
            .map_err(|_| SerializationError::InvalidData("field coeff serialize failed".to_string()))?;
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

/// Runtime Golomb-Rice parameters for terminal `z` from public schedule data.
///
/// # Errors
///
/// Propagates [`LevelParams::fold_response_sigma`] and [`LevelParams::num_digits_fold`] errors.
pub fn tail_golomb_rice_z_params(
    lp: &LevelParams,
    num_t_vectors: usize,
    num_public_rows: usize,
    field_bits: u32,
) -> Result<(u32, u32), AkitaError> {
    let sigma = lp.fold_response_sigma(num_t_vectors, num_public_rows)?;
    let k = optimal_rice_k(sigma);
    let depth_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    let w = golomb_rice_zigzag_width_z(depth_fold, lp.log_basis);
    Ok((k, w))
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
        z_first: ring_column_z_first(lp),
        z_coords,
        e_field_elems,
        t_field_elems,
        r_field_elems,
        logical_num_elems,
    })
}

/// Conservative planner upper bound for a segment-typed tail witness.
#[must_use]
pub fn segment_typed_witness_upper_bound_bytes(
    field_bits: u32,
    layout: &TailSegmentLayout,
    rice_k: u32,
    zigzag_w_z: u32,
) -> usize {
    let raw_elems = layout
        .e_field_elems
        .saturating_add(layout.t_field_elems)
        .saturating_add(layout.r_field_elems);
    let raw_bytes = raw_elems.saturating_mul(field_bytes(field_bits));
    let bits_per_z = golomb_rice_max_bits_per_coord(rice_k, zigzag_w_z);
    let z_bits = layout.z_coords.saturating_mul(bits_per_z);
    raw_bytes.saturating_add(z_bits.div_ceil(8))
}

/// Exact serialized byte length for a constructed segment-typed witness.
#[must_use]
pub fn segment_typed_witness_exact_bytes<F: FieldCore + CanonicalField + AkitaSerialize>(
    witness: &SegmentTypedWitness<F>,
) -> usize {
    witness.serialized_size(Compress::No)
}

/// Recompose balanced digit planes into a signed integer.
#[must_use]
pub fn recompose_balanced_i8_digits(digits: &[i8], log_basis: u32) -> i64 {
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
pub fn encode_z_segment_from_centered<const D: usize>(
    centered: &[[i32; D]],
    block_len: usize,
    depth_commit: usize,
    num_digits_fold: usize,
    log_basis: u32,
    rice_k: u32,
    zigzag_w_z: u32,
) -> Result<Vec<u8>, AkitaError> {
    let inner_width = block_len * depth_commit;
    if !centered.len().is_multiple_of(inner_width) {
        return Err(AkitaError::InvalidInput(
            "z_folded length does not match layout".to_string(),
        ));
    }
    let mut values = Vec::with_capacity(centered.len() * D);
    let mut planes = vec![[0i8; D]; num_digits_fold];
    for z_j in centered {
        balanced_decompose_centered_i32(z_j, &mut planes, log_basis);
        for coeff in 0..D {
            let digits: Vec<i8> = (0..num_digits_fold).map(|p| planes[p][coeff]).collect();
            values.push(recompose_balanced_i8_digits(&digits, log_basis));
        }
    }
    golomb_rice_encode_vec(&values, rice_k, zigzag_w_z)
}

fn balanced_decompose_centered_i32<const D: usize>(
    centered: &[i32; D],
    out: &mut [[i8; D]],
    log_basis: u32,
) {
    let half_b = 1i128 << (log_basis - 1);
    let b = half_b << 1;
    let mask = b - 1;
    for coeff_idx in 0..D {
        let mut c = i128::from(centered[coeff_idx]);
        for plane in out.iter_mut() {
            let d = c & mask;
            let balanced = if d >= half_b { d - b } else { d };
            c = (c - balanced) >> log_basis;
            plane[coeff_idx] = balanced as i8;
        }
    }
}

/// Construct a segment-typed terminal witness from ring-switch outputs.
///
/// # Errors
///
/// Returns an error when layout counts do not match the supplied witness parts.
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
    let (rice_k, zigzag_w_z) =
        tail_golomb_rice_z_params(lp, num_t_vectors, num_public_rows, field_bits)?;
    let depth_commit = lp.num_digits_commit;
    let num_digits_fold = lp.num_digits_fold(num_t_vectors, field_bits)?;
    let z_payload = encode_z_segment_from_centered(
        z_folded_centered,
        lp.block_len,
        depth_commit,
        num_digits_fold,
        lp.log_basis,
        rice_k,
        zigzag_w_z,
    )?;
    let e_fields = FlatRingVec::from_ring_elems(e_folded);
    if e_fields.coeff_len() != layout.e_field_elems {
        return Err(AkitaError::InvalidInput(
            "segment-typed e segment length mismatch".to_string(),
        ));
    }
    let mut t_rings = Vec::new();
    for block in recomposed_inner_rows {
        t_rings.extend_from_slice(block);
    }
    let t_fields = FlatRingVec::from_ring_elems(&t_rings);
    if t_fields.coeff_len() != layout.t_field_elems {
        return Err(AkitaError::InvalidInput(
            "segment-typed t segment length mismatch".to_string(),
        ));
    }
    let r_fields = FlatRingVec::from_ring_elems(r);
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
    Ok(witness)
}

/// Pad a segment witness `z` bitstream to the schedule-bound byte length.
///
/// # Errors
///
/// Returns an error when the encoded payload already exceeds `budget_bytes`.
pub fn pad_segment_typed_z_payload<F: FieldCore>(
    witness: &mut SegmentTypedWitness<F>,
    budget_bytes: usize,
) -> Result<(), AkitaError> {
    if witness.z_payload.len() > budget_bytes {
        return Err(AkitaError::InvalidInput(format!(
            "segment-typed z payload {} bytes exceeds schedule budget {budget_bytes}",
            witness.z_payload.len()
        )));
    }
    witness.z_payload.resize(budget_bytes, 0);
    Ok(())
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
    num_w_vectors: usize,
    num_t_vectors: usize,
    num_public_rows: usize,
    num_commitment_groups: usize,
) -> Result<Vec<i8>, AkitaError>
where
    F: FieldCore + CanonicalField + HalvingField,
{
    if D != witness.layout.ring_dimension {
        return Err(AkitaError::InvalidProof);
    }
    let field_bits = F::modulus_bits();
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
    let (rice_k, zigzag_w_z) =
        tail_golomb_rice_z_params(lp, num_t_vectors, num_public_rows, field_bits)?;

    let z_values = golomb_rice_decode_vec(
        &witness.z_payload,
        witness.layout.z_coords,
        rice_k,
        zigzag_w_z,
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
    if witness.layout.z_first {
        emit_z_folded_block_inner::<D>(
            &mut out,
            &all_z_planes,
            lp.block_len,
            depth_commit,
            num_digits_fold,
            total_z_elems,
        );
        emit_planes_block_inner::<D>(&mut out, &e_planes, w_block_count, depth_open);
        emit_planes_block_inner::<D>(&mut out, &t_planes, t_block_count, t_planes_per_block);
    } else {
        emit_planes_block_inner::<D>(&mut out, &e_planes, w_block_count, depth_open);
        emit_planes_block_inner::<D>(&mut out, &t_planes, t_block_count, t_planes_per_block);
        emit_z_folded_block_inner::<D>(
            &mut out,
            &all_z_planes,
            lp.block_len,
            depth_commit,
            num_digits_fold,
            total_z_elems,
        );
    }
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

fn emit_planes_block_inner<const D: usize>(
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

fn emit_z_folded_block_inner<const D: usize>(
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

use akita_serialization::Validate;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SisModulusFamily;
    use akita_challenges::SparseChallengeConfig;
    use akita_field::Prime128OffsetA7F7;

    type F = Prime128OffsetA7F7;
    const D: usize = 8;

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
    fn tail_segment_layout_is_non_empty() {
        let lp = test_lp();
        let layout = tail_segment_layout(&lp, 1, 1, 1, 1, F::modulus_bits()).unwrap();
        assert!(layout.logical_num_elems > 0);
        assert!(layout.z_coords > 0);
    }
}
