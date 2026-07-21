//! Header-stripped proof-size and planned-witness sizing formulas.

use akita_field::{AkitaError, CanonicalField};

use crate::sis::compute_num_digits_full_field;
use crate::PolynomialGroupLayout;
use crate::{CommittedGroupParams, TerminalResponseShape, EXTENSION_OPENING_REDUCTION_DEGREE};

/// Field element size in bytes for a field with `field_bits` bits.
pub fn field_bytes(field_bits: u32) -> usize {
    (field_bits as usize).div_ceil(8)
}

/// Ring vector bytes without a length prefix.
pub fn proof_ring_vec_bytes(ring_len: usize, ring_dim: usize, elem_bytes: usize) -> usize {
    ring_len.saturating_mul(ring_dim).saturating_mul(elem_bytes)
}

/// Packed digit bytes without a length/tag prefix.
pub fn packed_digits_bytes(num_elems: usize, bits_per_elem: u32) -> usize {
    num_elems.saturating_mul(bits_per_elem as usize).div_ceil(8)
}

/// Serialized byte size for a terminal direct witness shape.
pub fn terminal_response_bytes(field_bits: u32, shape: &TerminalResponseShape) -> usize {
    crate::proof::terminal_response_upper_bound_bytes(
        field_bits,
        &shape.layout,
        shape.layout.z_payload_bytes(),
    )
}

fn compressed_unipoly_bytes(degree: usize, elem_bytes: usize) -> usize {
    degree * elem_bytes
}

fn sumcheck_bytes(rounds: usize, degree: usize, elem_bytes: usize) -> usize {
    rounds * compressed_unipoly_bytes(degree, elem_bytes)
}

/// Header-stripped byte size of an extension-opening reduction proof.
///
/// The reduction proof serializes `partials` challenge-field elements followed
/// by a fixed degree-two sumcheck over `opening_vars - log2(extension_width)`
/// rounds. `extension_width = 1` means the claim field is already the base
/// field and contributes zero bytes.
///
/// # Errors
///
/// Returns an error when `extension_width` is not a power of two or when the
/// tensor split is wider than the opened Boolean cube.
pub fn extension_opening_reduction_proof_bytes(
    challenge_field_bits: u32,
    partials: usize,
    opening_vars: usize,
    extension_width: usize,
) -> Result<usize, AkitaError> {
    if extension_width <= 1 {
        return Ok(0);
    }
    if !extension_width.is_power_of_two() {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening width must be a power of two, got {extension_width}"
        )));
    }
    let split_bits = extension_width.trailing_zeros() as usize;
    if split_bits > opening_vars {
        return Err(AkitaError::InvalidSetup(format!(
            "extension opening split ({split_bits}) exceeds opening variables ({opening_vars})"
        )));
    }
    let elem_bytes = field_bytes(challenge_field_bits);
    let rounds = opening_vars - split_bits;
    Ok(partials
        .saturating_mul(elem_bytes)
        .saturating_add(sumcheck_bytes(
            rounds,
            EXTENSION_OPENING_REDUCTION_DEGREE,
            elem_bytes,
        )))
}

/// Log2 of the next power-of-two Boolean cube width for recursive opening.
pub fn padded_boolean_opening_vars(len: usize) -> Result<usize, AkitaError> {
    let padded = len
        .checked_next_power_of_two()
        .ok_or_else(|| AkitaError::InvalidSetup("opening witness length overflow".to_string()))?;
    Ok(padded.trailing_zeros() as usize)
}

/// Extension-opening reduction proof bytes for one fold level in a schedule.
pub fn extension_opening_reduction_level_bytes(
    challenge_field_bits: u32,
    extension_opening_width: usize,
    fold_level: usize,
    key: PolynomialGroupLayout,
    input_witness_len: usize,
) -> Result<usize, AkitaError> {
    if extension_opening_width <= 1 {
        return Ok(0);
    }
    let (partials, opening_vars) = if fold_level == 0 {
        (
            extension_opening_width.saturating_mul(key.num_polynomials()),
            key.num_vars(),
        )
    } else {
        (
            extension_opening_width,
            padded_boolean_opening_vars(input_witness_len)?,
        )
    };
    extension_opening_reduction_proof_bytes(
        challenge_field_bits,
        partials,
        opening_vars,
        extension_opening_width,
    )
}

/// Planned recursive witness size in ring elements for a singleton fold.
pub fn planned_w_ring_element_count<F: CanonicalField>(
    field_bits: u32,
    lp: &CommittedGroupParams,
) -> Result<usize, AkitaError> {
    let _field_marker = core::marker::PhantomData::<F>;
    let e_hat_count = lp
        .num_live_blocks
        .checked_mul(lp.num_digits_open)
        .ok_or_else(|| AkitaError::InvalidSetup("planned W width overflow".to_string()))?;
    let t_hat_count = lp
        .num_live_blocks
        .checked_mul(lp.inner_commit_matrix.output_rank())
        .and_then(|n| n.checked_mul(lp.num_digits_outer))
        .ok_or_else(|| AkitaError::InvalidSetup("planned T width overflow".to_string()))?;
    let z_pre_count = lp
        .inner_width()
        .checked_mul(lp.num_digits_fold(1, field_bits)?)
        .ok_or_else(|| AkitaError::InvalidSetup("planned Z width overflow".to_string()))?;
    let r_count = lp
        .relation_matrix_row_count(1)?
        .checked_mul(compute_num_digits_full_field(field_bits, lp.log_basis_open))
        .ok_or_else(|| AkitaError::InvalidSetup("planned r-tail width overflow".to_string()))?;

    e_hat_count
        .checked_add(t_hat_count)
        .and_then(|n| n.checked_add(z_pre_count))
        .and_then(|n| n.checked_add(r_count))
        .ok_or_else(|| AkitaError::InvalidSetup("planned witness width overflow".to_string()))
}

/// Planned recursive witness size in field elements for a singleton fold.
pub fn planned_output_witness_len<F: CanonicalField>(
    field_bits: u32,
    lp: &CommittedGroupParams,
) -> Result<usize, AkitaError> {
    planned_w_ring_element_count::<F>(field_bits, lp)?
        .checked_mul(lp.d_a())
        .ok_or_else(|| AkitaError::InvalidSetup("planned next witness length overflow".to_string()))
}

/// Total sumcheck rounds (`col_bits + ring_bits`) for one fold level.
pub fn sumcheck_rounds(level_d: usize, output_witness_len: usize) -> usize {
    let ring_bits = level_d.trailing_zeros() as usize;
    let num_ring_elems = output_witness_len / level_d;
    let col_bits = num_ring_elems.next_power_of_two().trailing_zeros() as usize;
    col_bits + ring_bits
}
